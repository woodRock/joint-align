"""
40_export_site_data.py — Export everything the project website (docs/) plots into one JSON file.

Reads the outputs of scripts 30–32 and the data, refits the few small models the site needs
(pairwise fits, leave-one-pair-out fits, the λ sweep) and writes docs/data/site.json. Nothing
on the site is hand-entered: rerun this after rerunning the experiments.

Output: docs/data/site.json and docs/data/site.js (the same data, loadable from file://)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from warpalign.config import BIN_MIN, EVAL_START_BIN, N_BINS

import analysis as an
from jointalign.provenance import warpalign_revision

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / 'results'
OUT = ROOT / 'docs' / 'data' / 'site.json'
GRID = an.GRID
MAX_ANCHORS_PER_PAIR = 1200
TIC_SAMPLES_PER_BATCH = 4
short, shift = an.short, an.shift


def saved_rev(z) -> str | None:
    """The warpalign revision a results .npz recorded; results written before the package was renamed
    (gcmsalign → warpalign) store it under the old key."""
    for key in ('warpalign', 'gcmsalign'):
        if key in z:
            return str(z[key])
    return None


def public_rev(rev: str) -> str:
    """Provenance line without the local checkout path (the site is public)."""
    return re.sub(r' from \S.*$', '', str(rev))


def r(x, nd: int = 3):
    """Round arrays/floats for compact JSON (NaN → None)."""
    a = np.asarray(x, dtype=float)
    out = np.round(a, nd).tolist()
    if a.ndim == 0:
        return None if np.isnan(a) else out
    return json.loads(json.dumps(out).replace('NaN', 'null'))


def main(data: Path | None = None) -> None:
    run = an.load_run(RESULTS, data)
    s30, z, names, K, anc, joint, ref = run.s30, run.z, run.names, run.K, run.anc, run.joint, run.ref
    sizes = [len(b) for b in run.batches]
    site: dict = {
        'meta': {
            'generated': dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'warpalign': public_rev(saved_rev(z) or warpalign_revision()),
            'lam': run.lam, 'ref': ref, 'bin_min': BIN_MIN, 'n_bins': N_BINS,
            'eval_start_bin': EVAL_START_BIN, 'n_anchors': len(anc),
            'n_sample_pairs': int(sum(sizes[a] * sizes[b] for a, b in combinations(range(K), 2))),
        },
        'batches': [{'name': short(n), 'n': s, 'start': int(b.acquisition_start.min())}
                    for n, s, b in zip(names, sizes, run.batches)],
    }

    # ── Chromatograms: a few TICs per batch, raw and in the reference batch's frame ──
    tic_out = []
    for k, tics in enumerate(run.tics):
        src = joint.map(ref, k, GRID)                       # ref time → batch-k time
        sel = np.linspace(0, sizes[k] - 1, min(TIC_SAMPLES_PER_BATCH, sizes[k])).round().astype(int)
        raw = tics[sel]
        al = s30.warp_tics(raw, src)
        scale = np.percentile(tics[:, EVAL_START_BIN:], 99.5)
        tic_out.append({'raw': r(raw / scale), 'aligned': r(al / scale), 'samples': sel.tolist()})
    site['tic'] = tic_out

    # ── Warps: joint and pairwise for every ordered pair; latent drift per batch ──
    site['warps'] = {
        'joint': {f'{a}-{b}': r(shift(joint.map(a, b, GRID))) for a, b in permutations(range(K), 2)},
        'pairwise': {f'{a}-{b}': r(shift(an.pw_map(run, a, b, GRID))) for a, b in permutations(range(K), 2)},
        'latent_drift': r(np.stack([joint.latent(k, GRID) - GRID for k in range(K)])),
    }
    site['cycle'] = [{'t': list(c['t']), 'pairwise': r(c['pairwise']), 'joint': r(c['joint'], 4)}
                     for c in an.cycle_table(run)]

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
    boots = an.bootstrap_models(run)
    site['band'] = {f'{a}-{b}': r(an.band(run, a, b, boots)) for a, b in permutations(range(K), 2)}

    # ── Missing correspondences: joint fit without the pair's anchors (LOPO) ──
    site['lopo'] = {f'{a}-{b}': {'map': r(shift(v['model'].map(a, b, GRID))), 'ticr': r(v['ticr']),
                                 'err_min': r(v['err_min']), 'err_joint_min': r(v['err_joint_min'])}
                    for (a, b), v in an.lopo(run).items()}

    # ── λ sweep: drift curves, RT-blocked CV error and post-hoc TIC r ──
    site['lambda_sweep'] = [{'lam': s['lam'], 'cv_min': r(s['cv_min'], 4), 'drift': r(s['drift'], 2),
                             'ticr_drifted': r(s['ticr_drifted']), 'ticr_near': r(s['ticr_near'])}
                            for s in an.lambda_sweep(run)]

    # ── TIC r table from script 30 ──
    methods30 = ['Unaligned', 'Pooled PCHIP (cow-dcnn)', 'Dense PCHIP', 'Pairwise smooth', 'Joint', 'Joint CF']
    site['ticr30'] = {'methods': methods30, 'pairs': []}
    for pair in run.pairs:
        row = {'a': run.idx[pair[0]], 'b': run.idx[pair[1]], 'drifted': pair in run.drifted,
               'mean': {}, 'per_a': {}}
        for mth in methods30:
            v = run.ticr(pair, mth)
            row['mean'][mth] = r(v.mean())
            row['per_a'][mth] = r(v.mean(1))
        site['ticr30']['pairs'].append(row)
    site['coverage'] = an.coverage(run)

    # ── Fair benchmark (script 32), if present ──
    fb = RESULTS / 'fair_benchmark.npz'
    if fb.exists():
        zb = np.load(fb)
        methods = ['Unaligned', 'Pooled PCHIP', 'PCHIP + COW refine', 'COW-TIC', 'WarpTransformer',
                   'Joint', 'Neural joint']
        bench = {'methods': methods, 'protocols': ['B-frame', 'Midpoint'], 'pairs': [],
                 'warpalign': public_rev(saved_rev(zb)) if saved_rev(zb) else None}
        for A_, B_ in run.pairs:
            row = {'a': run.idx[A_], 'b': run.idx[B_], 'mean': {}, 'per_a': {}}
            for proto in bench['protocols']:
                for mth in methods:
                    key = f'{short(A_)}_{short(B_)}|{mth}|{proto}'
                    if key in zb.files:
                        row['mean'][f'{mth}|{proto}'] = r(zb[key].mean())
                        row['per_a'][f'{mth}|{proto}'] = r(zb[key].mean(1))
            bench['pairs'].append(row)
        site['benchmark'] = bench

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(site, separators=(',', ':'))
    OUT.write_text(payload)
    # The same data as a script, so docs/index.html also works opened straight from disk (file://),
    # where browsers block fetch() of local JSON.
    OUT.with_suffix('.js').write_text(f'window.SITE_DATA = {payload};\n')
    print(f'{OUT}: {OUT.stat().st_size / 1e6:.2f} MB '
          f'(benchmark {"included" if "benchmark" in site else "not yet available"})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=None, help="batch_* folders (default: script 30's default)")
    ap.add_argument('--results', type=Path, default=RESULTS, help='outputs of scripts 30–32 (default: results/)')
    ap.add_argument('--out', type=Path, default=OUT, help='JSON to write (default: docs/data/site.json)')
    a = ap.parse_args()
    RESULTS, OUT = a.results, a.out
    main(a.data)
