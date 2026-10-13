"""
40_export_site_data.py — Export everything the project website (docs/) plots into one JSON file.

Reads the outputs of scripts 30–32 and the data, refits the few small models the site needs
(pairwise fits, leave-one-pair-out fits, the λ sweep) and writes docs/data/site.json. Nothing
on the site is hand-entered: rerun this after rerunning the experiments.

Output: docs/data/site.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from gcmsalign.config import BIN_MIN, EVAL_START_BIN, N_BINS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import multibatch as mb
from provenance import gcmsalign_revision

RESULTS = ROOT / 'results'
OUT = ROOT / 'docs' / 'data' / 'site.json'
GRID = np.arange(N_BINS, dtype=float)
MAX_ANCHORS_PER_PAIR = 1200
TIC_SAMPLES_PER_BATCH = 4


def _s30():
    spec = importlib.util.spec_from_file_location('s30', ROOT / 'scripts' / '30_multibatch_align.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def r(x, nd: int = 3):
    """Round arrays/floats for compact JSON (NaN → None)."""
    a = np.asarray(x, dtype=float)
    out = np.round(a, nd).tolist()
    if a.ndim == 0:
        return None if np.isnan(a) else out
    return json.loads(json.dumps(out).replace('NaN', 'null'))


def short(b: str) -> str:
    return b.replace('batch_', '')


def shift(m: np.ndarray) -> np.ndarray:
    """Shift m(t) − t of a map evaluated on GRID, NaN where the map is clamped to the run ends
    (JointWarp.map saturates at bin 0 / N_BINS − 1 when the target falls outside batch B's run;
    drawing that as a shift would show a spurious cliff)."""
    out = m - GRID
    out[(m <= 1e-6) | (m >= N_BINS - 1 - 1e-6)] = np.nan
    return out


def main() -> None:
    s30 = _s30()
    z = np.load(RESULTS / 'multibatch_align.npz')
    names = [str(n) for n in z['names']]
    K = len(names)
    lam, ref = float(z['lam']), int(z['ref'])
    batches = [s30.load_batch(n, s30.DATA) for n in names]
    sizes = [len(b) for b in batches]
    tics = [b.tic for b in batches]
    za = np.load(RESULTS / 'anchors_all_pairs.npz')
    anc = mb.Anchors(*(za[k] for k in ('ba', 'bb', 'sa', 'sb', 'ta', 'tb')))
    joint = mb.fit_joint(anc, K, lam=lam, gauge=ref)
    if not np.allclose(joint.drift, z['drift'], atol=1e-6):
        raise SystemExit('results/multibatch_align.npz does not match the anchor cache; rerun script 30')
    site: dict = {
        'meta': {
            'generated': dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'gcmsalign': str(z['gcmsalign']) if 'gcmsalign' in z.files else gcmsalign_revision(),
            'lam': lam, 'ref': ref, 'bin_min': BIN_MIN, 'n_bins': N_BINS,
            'eval_start_bin': EVAL_START_BIN, 'n_anchors': len(anc),
            'n_sample_pairs': int(sum(sizes[a] * sizes[b] for a, b in combinations(range(K), 2))),
        },
        'batches': [{'name': short(n), 'n': s, 'start': int(b.acquisition_start.min())}
                    for n, s, b in zip(names, sizes, batches)],
    }

    # ── Chromatograms: a few TICs per batch, raw and in the reference batch's frame ──
    tic_out = []
    for k in range(K):
        src = joint.map(ref, k, GRID)                       # ref time → batch-k time
        sel = np.linspace(0, sizes[k] - 1, min(TIC_SAMPLES_PER_BATCH, sizes[k])).round().astype(int)
        raw = tics[k][sel]
        al = s30.warp_tics(raw, src)
        scale = np.percentile(tics[k][:, EVAL_START_BIN:], 99.5)
        tic_out.append({'raw': r(raw / scale), 'aligned': r(al / scale), 'samples': sel.tolist()})
    site['tic'] = tic_out

    # ── Warps: joint and pairwise for every ordered pair; latent drift per batch ──
    pairwise = {(a, b): mb.fit_pairwise(anc, a, b, K, lam=lam, gauge=0) for a, b in combinations(range(K), 2)}
    pw = lambda a, b: pairwise[(min(a, b), max(a, b))]
    site['warps'] = {
        'joint': {f'{a}-{b}': r(shift(joint.map(a, b, GRID))) for a, b in permutations(range(K), 2)},
        'pairwise': {f'{a}-{b}': r(shift(pw(a, b).map(a, b, GRID))) for a, b in permutations(range(K), 2)},
        'latent_drift': r(np.stack([joint.latent(k, GRID) - GRID for k in range(K)])),
    }
    tt = GRID[(GRID * BIN_MIN >= 5) & (GRID * BIN_MIN <= 40)]
    cyc = []
    for a, b, c in permutations(range(K), 3):
        g = lambda i, j, t: pw(i, j).map(i, j, t)
        cyc.append({'t': [a, b, c],
                    'pairwise': r(np.mean(np.abs(g(b, c, g(a, b, tt)) - g(a, c, tt))) * BIN_MIN),
                    'joint': r(np.mean(np.abs(joint.map(b, c, joint.map(a, b, tt)) - joint.map(a, c, tt)))
                               * BIN_MIN, 4)})
    site['cycle'] = cyc

    # ── Anchors per batch pair (subsampled), with the joint fit's robust weight ──
    rng = np.random.default_rng(0)
    anc_out = {}
    for a, b in combinations(range(K), 2):
        m = np.flatnonzero((anc.ba == a) & (anc.bb == b))
        keep = np.sort(rng.choice(m, min(len(m), MAX_ANCHORS_PER_PAIR), replace=False))
        anc_out[f'{a}-{b}'] = {'n': len(m), 'ta': r(anc.ta[keep], 2), 'tb': r(anc.tb[keep], 2),
                               'w': r(joint.weights[keep], 2),
                               'rejected': r(float(np.mean(joint.weights[m] == 0)), 4)}
    site['anchors'] = anc_out
    site['n_pair'] = z['n_pair'].tolist()

    # ── Uncertainty: bootstrap 90 % band of every pairwise map ──
    boots = [mb.JointWarp(joint.knots, d, joint.scale, joint.weights) for d in z['boot_drift']]
    site['band'] = {}
    for a, b in permutations(range(K), 2):
        q = np.percentile(np.stack([m.map(a, b, GRID) for m in boots]), [5, 95], axis=0)
        clamped = np.isnan(shift(joint.map(a, b, GRID)))
        q = q - GRID
        q[:, clamped] = np.nan
        site['band'][f'{a}-{b}'] = r(q)

    # ── Missing correspondences: joint fit without the pair's anchors (LOPO) ──
    idx = {n: i for i, n in enumerate(names)}
    lopo = {}
    for A_, B_ in s30.PAIRS:
        a, b = idx[A_], idx[B_]
        keep = ~(((anc.ba == a) & (anc.bb == b)) | ((anc.ba == b) & (anc.bb == a)))
        m = mb.fit_joint(anc.subset(keep), K, lam=lam, gauge=ref)
        own = anc.pair(a, b)
        lopo[f'{a}-{b}'] = {
            'map': r(shift(m.map(a, b, GRID))),
            'ticr': r(s30.tic_r(s30.warp_tics(tics[a], m.map(b, a, GRID)), tics[b]).mean()),
            'err_min': r(np.median(np.abs(m.map(a, b, own.ta) - own.tb)) * BIN_MIN),
            'err_joint_min': r(np.median(np.abs(joint.map(a, b, own.ta) - own.tb)) * BIN_MIN),
        }
    site['lopo'] = lopo

    # ── λ sweep: drift curves, RT-blocked CV error and post-hoc TIC r ──
    blk = lambda t: (np.floor(t / s30.RT_BLOCK).astype(int)) % s30.N_FOLDS
    sweep = []
    drifted = [p for p in s30.PAIRS if z[f'ticr_{short(p[0])}_{short(p[1])}_Unaligned'].mean() < 0.5]
    for lm in s30.LAMBDAS:
        errs = []
        for f in range(s30.N_FOLDS):
            te = (blk(anc.ta) == f) | (blk(anc.tb) == f)
            tr = (blk(anc.ta) != f) & (blk(anc.tb) != f)
            errs.append(s30.anchor_errors(mb.fit_joint(anc.subset(tr), K, lam=lm, gauge=ref), anc.subset(te)))
        m = mb.fit_joint(anc, K, lam=lm, gauge=ref)
        v = {p: s30.tic_r(s30.warp_tics(tics[idx[p[0]]], m.map(idx[p[1]], idx[p[0]], GRID)),
                          tics[idx[p[1]]]).mean() for p in s30.PAIRS}
        sweep.append({'lam': lm, 'cv_min': r(np.median(np.abs(np.concatenate(errs))) * BIN_MIN, 4),
                      'drift': r(np.stack([m.latent(k, GRID) - GRID for k in range(K)]), 2),
                      'ticr_drifted': r(np.mean([v[p] for p in drifted])),
                      'ticr_near': r(np.mean([v[p] for p in s30.PAIRS if p not in drifted]))})
    site['lambda_sweep'] = sweep

    # ── TIC r table from script 30 ──
    methods30 = ['Unaligned', 'Pooled PCHIP (cow-dcnn)', 'Dense PCHIP', 'Pairwise smooth', 'Joint', 'Joint CF']
    site['ticr30'] = {'methods': methods30, 'pairs': []}
    for A_, B_ in s30.PAIRS:
        row = {'a': idx[A_], 'b': idx[B_], 'drifted': (A_, B_) in drifted, 'mean': {}, 'per_a': {}}
        for mth in methods30:
            v = z[f'ticr_{short(A_)}_{short(B_)}_{mth.replace(" ", "_")}']
            row['mean'][mth] = r(v.mean())
            row['per_a'][mth] = r(v.mean(1))
        site['ticr30']['pairs'].append(row)

    txt = (RESULTS / 'multibatch_align.txt').read_text()
    cov = re.search(r'predictive \(band ⊕ anchor scatter\)\s*:\s*([\d.]+)', txt)
    epi = re.search(r'warp-only band \(epistemic\)\s*:\s*([\d.]+)', txt)
    site['coverage'] = {'predictive': float(cov.group(1)) if cov else None,
                        'epistemic': float(epi.group(1)) if epi else None, 'target': 0.9}

    # ── Fair benchmark (script 32), if present ──
    fb = RESULTS / 'fair_benchmark.npz'
    if fb.exists():
        zb = np.load(fb)
        methods = ['Unaligned', 'Pooled PCHIP', 'PCHIP + COW refine', 'COW-TIC', 'WarpTransformer',
                   'Joint', 'Neural joint']
        bench = {'methods': methods, 'protocols': ['B-frame', 'Midpoint'], 'pairs': [],
                 'gcmsalign': str(zb['gcmsalign']) if 'gcmsalign' in zb.files else None}
        for A_, B_ in s30.PAIRS:
            row = {'a': idx[A_], 'b': idx[B_], 'mean': {}, 'per_a': {}}
            for proto in bench['protocols']:
                for mth in methods:
                    key = f'{short(A_)}_{short(B_)}|{mth}|{proto}'
                    if key in zb.files:
                        row['mean'][f'{mth}|{proto}'] = r(zb[key].mean())
                        row['per_a'][f'{mth}|{proto}'] = r(zb[key].mean(1))
            bench['pairs'].append(row)
        site['benchmark'] = bench

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(site, separators=(',', ':')))
    print(f'{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.2f} MB '
          f'(benchmark {"included" if "benchmark" in site else "not yet available"})')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
