"""
32_fair_benchmark.py — Every method scored under the same protocols on all 10 batch pairs.

The cow-dcnn numbers mix protocols: PCHIP and COW-TIC warp A into B's raw frame, the
WarpTransformer and "PCHIP + COW refine" resample both batches, and COW-TIC fits a separate
TIC-maximising warp for every (A_i, B_j). Here every method is reduced to what it actually
estimates, a correspondence map m_ij: RT in B_j → RT in A_i, and every map is scored the same
ways:

  B-frame    r( A_i(m(t)), B_j(t) )                      A resampled, B raw (cow-dcnn table)
  Midpoint   both resampled into the frame halfway between them: h(t) = (t + m(t)) / 2,
             s = h⁻¹(τ), r( A_i(m(s)), B_j(s) )            symmetric, same smoothing for all
  Native     the WarpTransformer's own consensus frame and "PCHIP + COW refine" with both
             batches refined (as published), for reference

All TIC r is Pearson r of the normalised TIC on the common acquisition window (alignbench
FISH_OIL_AXIS.eval_bins, bins ≥ 22). These protocols are alignbench.metrics.protocols (tic_bframe,
tic_midpoint), which reproduce this script bit for bit.

Methods
  Pooled PCHIP          alignbench match_pooled anchors (A[0..2] vs B[0], pooled, filtered) + pchip_warp
  PCHIP + COW refine    pooled PCHIP, then per-A_i COW (8 segments, ±4 bins) onto mean B TIC
  COW-TIC               alignbench.methods.tic.cow_tic_align per (A_i, B_j) (4 segments, ±22 bins)
  WarpTransformer       script 31 cache (warpalign calibrate_pair, v2 checkpoint, seed 0, CPU)
  Joint                 src/jointalign/multibatch.py, λ and reference from script 30
  Neural joint          src/jointalign/neural_joint.py on top of Joint (anchors + dense, sample level)

Output: results/fair_benchmark.txt, results/fair_benchmark.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from alignbench.anchors import match_pooled_arrays as extract_anchors  # = warpalign extract_anchors
from alignbench.axis import FISH_OIL_AXIS
from alignbench.datasets.fish_oil import load_batch
from alignbench.methods.tic import cow_tic_align
from alignbench.metrics.stats import holm, paired_wilcoxon
from alignbench.metrics.tic import pearson
from alignbench.warps import invert_src, pchip_warp, src_from_fn, warp_signal

from jointalign import multibatch as mb
from jointalign import neural_joint as nj
from jointalign.provenance import display_path, header

ROOT = Path(__file__).resolve().parent.parent

N_BINS = FISH_OIL_AXIS.n_bins
DATA, RESULTS = ROOT / 'data' / 'fish_oil', ROOT / 'results'
GRID = np.arange(N_BINS, dtype=float)
PAIRS = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
         ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016'),
         ('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016'),
         ('batch_aug2015', 'batch_sep2015'), ('batch_aug2015', 'batch_jan2016'),
         ('batch_aug2015', 'batch_apr2016'), ('batch_aug2015', 'batch_jul2016')]
GROUPS = {'drifted (cow-dcnn 4)': PAIRS[:4], 'near-aligned': PAIRS[4:6], 'aug2015 (n=2)': PAIRS[6:]}
METHODS = ['Unaligned', 'Pooled PCHIP', 'PCHIP + COW refine', 'COW-TIC', 'WarpTransformer',
           'Joint', 'Neural joint']


def short(b: str) -> str:
    return b.replace('batch_', '')


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r on the common acquisition window (0 if either signal is constant there)."""
    r = pearson(x, y)
    return 0.0 if np.isnan(r) else r


def sample(tic: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """tic read at fractional bins pos (clamped to the run)."""
    return warp_signal(np.asarray(tic, dtype=float), pos)


def read_warp(w: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Warp w [T] (output bin → source bin) at fractional output positions t, continued
    linearly with the end segments' slope (v2 warps are not pinned to [0, T−1])."""
    out = np.interp(t, GRID, w)
    out = np.where(t < 0, w[0] + t * (w[1] - w[0]), out)
    return np.where(t > N_BINS - 1, w[-1] + (t - N_BINS + 1) * (w[-1] - w[-2]), out)


def score(ta: np.ndarray, tb: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    """(B-frame r, midpoint r) for the map m: B time (GRID) → A time."""
    m = np.maximum.accumulate(np.clip(m, -N_BINS, 2 * N_BINS))
    p1 = corr(sample(ta, m), tb)
    h = (GRID + m) / 2 + 1e-9 * GRID
    s = np.interp(GRID, h, GRID)
    p2 = corr(sample(ta, np.interp(s, GRID, m)), sample(tb, s))
    return p1, p2


def strict(w: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(w) + 1e-6 * np.arange(len(w))


def cow_map(q: np.ndarray, r: np.ndarray, n_segs: int, slack: int) -> np.ndarray:
    """COW (alignbench, from warpalign) as a map: reference (B) bin → query (A) bin."""
    return cow_tic_align(q, r, n_segs=n_segs, slack=slack).src


def main(neural_steps: int) -> None:
    t0 = time.time()
    names = sorted(p.name for p in DATA.glob('batch_*') if p.is_dir())
    idx = {b: i for i, b in enumerate(names)}
    K = len(names)
    batches = [load_batch(b, DATA) for b in names]
    chroma = [bt.chromas for bt in batches]
    tics = [bt.tic for bt in batches]
    sizes = [len(bt) for bt in batches]
    offs = np.cumsum([0] + sizes[:-1])
    L = [__doc__.strip().splitlines()[0], '', *header(), f'Data: {display_path(DATA)}', '']
    say = lambda s='': (L.append(s), print(s, flush=True))

    z = np.load(RESULTS / 'anchors_all_pairs.npz')
    anc = mb.Anchors(*(z[k] for k in ('ba', 'bb', 'sa', 'sb', 'ta', 'tb')))
    cfg30 = np.load(RESULTS / 'multibatch_align.npz')
    lam, ref = float(cfg30['lam']), int(cfg30['ref'])
    joint = mb.fit_joint(anc, K, lam=lam, gauge=ref)
    say(f'Joint: λ = {lam:g}, reference {short(names[ref])} (selected in script 30)')

    # Neural joint (cached)
    g_path = RESULTS / f'neural_joint_G_{neural_steps}.npz'
    if g_path.exists():
        zz = np.load(g_path); taus, G = zz['taus'], zz['G']
    else:
        taus, g0 = nj.latent_grid_from_joint(joint, K)
        batch_of = np.concatenate([[k] * n for k, n in enumerate(sizes)])
        sidx = np.full((K, max(sizes)), -1)
        for k in range(K):
            sidx[k, :sizes[k]] = offs[k] + np.arange(sizes[k])
        torch.manual_seed(0)
        model = nj.NeuralJointWarp(taus, g0, batch_of, ref, sample_level=True)
        X = torch.tensor(np.concatenate(chroma))
        nj.train(model, X, anc, joint.weights, sidx, nj.TrainConfig(steps=neural_steps))
        with torch.no_grad():
            G = model()[0].numpy()
        np.savez(g_path, taus=taus, G=G)
    G = np.stack([strict(g) for g in G])

    R = {}          # (pair, method, protocol) → (n_a, n_b) array
    native = {}
    for A_, B_ in PAIRS:
        a, b = idx[A_], idx[B_]
        TA, TB = tics[a], tics[b]
        na, nb = len(TA), len(TB)
        maps = {}
        maps['Unaligned'] = lambda i, j: GRID
        pooled = extract_anchors(chroma[a], chroma[b], batches[a].tic_raw, batches[b].tic_raw)
        m24 = src_from_fn(pchip_warp(pooled.anchors))
        maps['Pooled PCHIP'] = lambda i, j, m24=m24: m24
        ref_b = np.mean(TB, axis=0)
        cref = [read_warp(m24, cow_map(sample(TA[i], m24), ref_b, 8, 4)) for i in range(na)]
        maps['PCHIP + COW refine'] = lambda i, j, c=cref: c[i]
        cache = RESULTS / 'cow_tic_maps' / f'{A_}__{B_}.npy'
        if cache.exists():
            ct = np.load(cache)
        else:
            cache.parent.mkdir(exist_ok=True)
            ct = np.stack([[cow_map(TA[i], TB[j], 4, 22) for j in range(nb)] for i in range(na)])
            np.save(cache, ct)
        maps['COW-TIC'] = lambda i, j, ct=ct: ct[i, j]
        wt_file = RESULTS / 'wt_warps' / f'{A_}__{B_}.npz'
        if wt_file.exists():
            wz = np.load(wt_file)
            W = np.stack([strict(w) for w in wz['warps']])
            maps['WarpTransformer'] = lambda i, j, W=W, na=na: read_warp(W[i], invert_src(W[na + j], GRID))
            native[(A_, B_, 'WarpTransformer')] = wz['r_native'].reshape(na, nb)
        mj = joint.map(b, a, GRID)
        maps['Joint'] = lambda i, j, mj=mj: mj
        maps['Neural joint'] = lambda i, j, a=a, b=b: np.interp(np.interp(GRID, G[offs[b] + j], taus),
                                                                taus, G[offs[a] + i])

        for meth, f in maps.items():
            out = np.array([[score(TA[i], TB[j], f(i, j)) for j in range(nb)] for i in range(na)])
            R[(A_, B_, meth, 'B-frame')] = out[..., 0]
            R[(A_, B_, meth, 'Midpoint')] = out[..., 1]
        # Native PCHIP + COW refine: both batches refined onto mean B (as in cow-dcnn script 25)
        ra = [sample(TA[i], cref[i]) for i in range(na)]
        rb = [sample(TB[j], cow_map(TB[j], ref_b, 8, 4)) for j in range(nb)]
        native[(A_, B_, 'PCHIP + COW refine')] = np.array([[corr(x, y) for y in rb] for x in ra])
        print(f'  {short(A_)}×{short(B_)} done ({time.time() - t0:.0f} s)', flush=True)

    for proto in ('B-frame', 'Midpoint'):
        say('\n' + '=' * 110)
        say(f'{proto}: mean TIC r over all A_i × B_j')
        say(f'  {"A × B":<18}' + ''.join(f'{m[:15]:>16}' for m in METHODS))
        for A_, B_ in PAIRS:
            vals = [R.get((A_, B_, m, proto)) for m in METHODS]
            best = max(v.mean() for v in vals if v is not None)
            say(f'  {short(A_) + "×" + short(B_):<18}' + ''.join(
                f'{"—":>16}' if v is None else f'{v.mean():>15.3f}{"*" if v.mean() == best else " "}'
                for v in vals))
        for g, ps in GROUPS.items():
            vals = []
            for m in METHODS:
                xs = [R.get((A_, B_, m, proto)) for A_, B_ in ps]
                vals.append(None if any(x is None for x in xs) else np.mean([x.mean() for x in xs]))
            say(f'  {"mean " + g:<18}' + ''.join(f'{"—":>16}' if v is None else f'{v:>16.3f}' for v in vals))
        say('  * best on that pair')

    say('\n' + '=' * 110)
    say('Native protocols as published (both batches resampled by the method itself), mean TIC r:')
    for A_, B_ in PAIRS:
        row = '  '.join(f'{m} {native[(A_, B_, m)].mean():.3f}' for m in ('WarpTransformer', 'PCHIP + COW refine')
                        if (A_, B_, m) in native)
        say(f'  {short(A_) + "×" + short(B_):<18} {row}')

    say('\nPaired two-sided Wilcoxon per A sample (r averaged over B_j), Joint vs each competitor; '
        'Holm over the pairs with n_A ≥ 6 per comparison:')
    tested = [p for p in PAIRS if sizes[idx[p[0]]] >= 6]
    for proto in ('B-frame', 'Midpoint'):
        for other in ('WarpTransformer', 'PCHIP + COW refine', 'COW-TIC', 'Neural joint'):
            res = []
            for A_, B_ in tested:
                x, y = R.get((A_, B_, 'Joint', proto)), R.get((A_, B_, other, proto))
                if y is None:
                    continue
                dx, dy = x.mean(1), y.mean(1)
                res.append((A_, B_, paired_wilcoxon(dx, dy)[1], float(np.median(dx - dy))))
            if not res:
                continue
            adj = holm([p for *_, p, _ in res])
            say(f'  {proto}: Joint vs {other}: ' + '; '.join(
                f'{short(a_)}×{short(b_)} Δ{md:+.3f} p={q:.1e}' for (a_, b_, _, md), q in zip(res, adj)))

    say(f'\nWall time {time.time() - t0:.0f} s')
    (RESULTS / 'fair_benchmark.txt').write_text('\n'.join(L) + '\n')
    np.savez(RESULTS / 'fair_benchmark.npz', provenance=np.array(header()),
             **{f'{short(k[0])}_{short(k[1])}|{k[2]}|{k[3]}': v for k, v in R.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--neural-steps', type=int, default=1500)
    ap.add_argument('--data', type=Path, default=DATA, help='batch_* folders (default: the fish oil data)')
    ap.add_argument('--results', type=Path, default=RESULTS,
                    help="script 30's output folder, also written to (default: results/)")
    a = ap.parse_args()
    DATA, RESULTS = a.data, a.results
    main(a.neural_steps)
