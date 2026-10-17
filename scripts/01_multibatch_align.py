"""
01_multibatch_align.py — Joint K-batch alignment on the fish oil data, with uncertainty.

Fits one monotone warp per batch into a shared latent RT frame (src/jointalign/multibatch.py) from
spectral anchors pooled over every cross-batch sample pair, and compares it with pairwise
methods on all 10 batch pairs of the five fish oil batches.

Hyper-parameters are chosen without TIC r: λ by blocked cross-validation over RT (whole
RT windows held out, so the score measures interpolation between compounds), the gauge
reference as the batch with the most anchors. TIC r is evaluation only: Pearson r of the
normalised TIC on the common acquisition window (alignbench.metrics.tic.pearson_rows, bins ≥ 22).

Data, anchors for the pooled-anchor PCHIP baseline, warping and statistics come from alignbench
(github.com/woodRock/align-bench, pinned in pyproject.toml); the report records its revision.

Experiments
  1. λ selection (held-out anchor error, RT-blocked folds) and gauge ablation
  2. TIC r on every batch pair, in-sample and cross-fitted (A_i's warp fitted without A_i)
  3. Held-out anchor error, cross-fitted
  4. Missing correspondences: drop all anchors of one batch pair and predict it through
     the other batches (joint) or through the best hub (composed pairwise warps); and a
     star design that only uses anchors against the reference batch
  5. Cycle consistency of pairwise warps over all batch triplets
  6. Uncertainty: bootstrap over samples within batches → 90 % bands for every pairwise
     warp; band width against direct anchor support; predictive-interval coverage of
     held-out anchors

Output: results/multibatch_align.txt, results/multibatch_align.npz, figs/multibatch_warps.png
"""
from __future__ import annotations

import argparse
import time
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from alignbench.anchors import match_pooled_arrays as extract_anchors  # = warpalign extract_anchors
from alignbench.axis import FISH_OIL_AXIS
from alignbench.datasets.fish_oil import load_batch
from alignbench.metrics.stats import holm, paired_wilcoxon
from alignbench.metrics.tic import pearson_rows
from alignbench.warps import pchip_warp, src_from_fn, warp_rows
from scipy.stats import spearmanr

from jointalign import multibatch as mb
from jointalign.provenance import display_path, header

ROOT = Path(__file__).resolve().parent.parent

BIN_MIN, N_BINS = FISH_OIL_AXIS.bin_min, FISH_OIL_AXIS.n_bins
DATA    = ROOT / 'data' / 'fish_oil'   # a symlink to the private data (README)
RESULTS = ROOT / 'results'
FIGS    = ROOT / 'figs'
LAMBDAS = [0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
RT_BLOCK = 12     # bins per held-out RT block in the λ-selection CV
N_FOLDS = 5
GRID    = np.arange(N_BINS, dtype=float)
# A is warped into B's frame. The first six follow cow-dcnn; aug2015 (2 samples) is A.
PAIRS = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
         ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016'),
         ('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016'),
         ('batch_aug2015', 'batch_sep2015'), ('batch_aug2015', 'batch_jan2016'),
         ('batch_aug2015', 'batch_apr2016'), ('batch_aug2015', 'batch_jul2016')]


def short(b: str) -> str:
    return b.replace('batch_', '')


def tic_r(tics_a, tics_b) -> np.ndarray:
    """(n_a, n_b) Pearson r between TICs on the common acquisition window."""
    A, B = np.asarray(tics_a), np.asarray(tics_b)
    return pearson_rows(A, B).reshape(len(A), len(B))


def warp_tics(tics, src: np.ndarray) -> np.ndarray:
    """Every TIC read at source positions src (bins, clamped to the run)."""
    return warp_rows(np.asarray(tics), src)


def assign_folds(sizes: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for n in sizes:
        f = np.empty(n, int)
        f[rng.permutation(n)] = np.arange(n) % N_FOLDS
        out.append(f)
    return out


def anchor_fold_masks(anc: mb.Anchors, folds: list[np.ndarray], f: int):
    fa = np.array([folds[b][s] for b, s in zip(anc.ba, anc.sa)])
    fb = np.array([folds[b][s] for b, s in zip(anc.bb, anc.sb)])
    test = (fa == f) | (fb == f)
    return ~test, test


def anchor_errors(model: mb.JointWarp, anc: mb.Anchors) -> np.ndarray:
    """Signed prediction error in batch-bb bins for every anchor (ta in ba → tb in bb)."""
    err = np.empty(len(anc))
    for a, b in combinations(range(model.drift.shape[0]), 2):
        m = (anc.ba == a) & (anc.bb == b)
        if m.any():
            err[m] = model.map(a, b, anc.ta[m]) - anc.tb[m]
    return err


def in_run(model: mb.JointWarp, a: int, b: int) -> np.ndarray:
    """Bins where the map a → b is not clamped at a run end. There every bootstrap fit is pinned to
    the same end value, so the band width is ~0 for a trivial reason; band statistics exclude them
    (scripts/analysis.py uses the same rule for the site and the figures)."""
    m = model.map(a, b, GRID)
    return (m > 1e-6) & (m < N_BINS - 1 - 1e-6)


def boot_weights(anc: mb.Anchors, sizes: list[int], rng: np.random.Generator) -> np.ndarray:
    cnt = [rng.multinomial(n, np.full(n, 1.0 / n)) for n in sizes]
    return np.array([cnt[a][i] * cnt[b][j] for a, b, i, j in zip(anc.ba, anc.bb, anc.sa, anc.sb)],
                    dtype=float)


class Composed:
    """Pairwise warps composed through one hub batch: a → h → b."""
    def __init__(self, m_ah: mb.JointWarp, a: int, h: int, m_hb: mb.JointWarp, b: int):
        self.parts = (m_ah, a, h, m_hb, b)

    def map(self, a: int, b: int, t: np.ndarray) -> np.ndarray:
        m_ah, a0, h, m_hb, b0 = self.parts
        if (a, b) == (a0, b0):
            return m_hb.map(h, b0, m_ah.map(a0, h, t))
        if (a, b) == (b0, a0):
            return m_ah.map(h, a0, m_hb.map(b0, h, t))
        raise ValueError


def main(n_boot: int, n_boot_cv: int, seed: int) -> None:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    names  = sorted(p.name for p in DATA.glob('batch_*') if p.is_dir())
    idx    = {b: i for i, b in enumerate(names)}
    K      = len(names)
    batches = [load_batch(b, DATA) for b in names]
    chroma = [bt.chromas for bt in batches]
    tics   = [bt.tic for bt in batches]
    sizes  = [len(bt) for bt in batches]
    L = [__doc__.strip().splitlines()[0], '', *header(), f'Data: {display_path(DATA)}', '']
    say = lambda s='': (L.append(s), print(s))
    say('Batches: ' + ', '.join(f'{short(b)} (n={n})' for b, n in zip(names, sizes)))

    # ── Anchors ──────────────────────────────────────────────────────────────
    cache = RESULTS / 'anchors_all_pairs.npz'
    if cache.exists():
        z   = np.load(cache)
        anc = mb.Anchors(*(z[k] for k in ('ba', 'bb', 'sa', 'sb', 'ta', 'tb')))
    else:
        anc = mb.harvest_anchors(chroma, [bt.tic_raw for bt in batches])
        np.savez(cache, ba=anc.ba, bb=anc.bb, sa=anc.sa, sb=anc.sb, ta=anc.ta, tb=anc.tb)
    n_pair = np.zeros((K, K), int)
    for a, b in combinations(range(K), 2):
        n_pair[a, b] = n_pair[b, a] = len(anc.pair(a, b))
    ref = int(np.argmax(n_pair.sum(1)))
    say(f'Anchors: {len(anc)} from every cross-batch sample pair. Gauge reference '
        f'(most anchors): {short(names[ref])}')
    say('  ' + '  '.join(f'{short(names[a])}×{short(names[b])}={n_pair[a, b]}'
                         for a, b in combinations(range(K), 2)))

    folds = assign_folds(sizes, rng)
    fold_masks = [anchor_fold_masks(anc, folds, f) for f in range(N_FOLDS)]

    # ── 1. λ selection and gauge ablation (held-out anchor error only) ───────
    # Sample folds leave held-out anchors at the same compounds' RTs as the training anchors,
    # so they cannot see wiggles between compounds. RT-blocked folds hold out whole RT windows
    # (in both batches of a pair) and score interpolation; λ is chosen on those.
    blk = lambda t: (np.floor(t / RT_BLOCK).astype(int)) % N_FOLDS
    rt_masks = []
    for f in range(N_FOLDS):
        te = (blk(anc.ta) == f) | (blk(anc.tb) == f)
        tr = (blk(anc.ta) != f) & (blk(anc.tb) != f)
        rt_masks.append((tr, te))
    say('\n' + '=' * 96)
    say(f'1. Hyper-parameters by 5-fold held-out anchor error (no TIC r). RT-blocked: RT windows of '
        f'{RT_BLOCK} bins held out; samples: folds over samples within each batch.')
    say(f'   {"λ":>6} {"gauge":>8} {"RT-blocked median |err| min":>28} {"≤1 bin":>7} '
        f'{"samples median |err| min":>25} {"≤1 bin":>7}')
    cv = {}
    for gauge in (ref, 'mean'):
        for lam in LAMBDAS:
            out = []
            for masks in (rt_masks, fold_masks):
                errs = []
                for tr, te in masks:
                    m = mb.fit_joint(anc.subset(tr), K, lam=lam, gauge=gauge)
                    errs.append(anchor_errors(m, anc.subset(te)))
                out.append(np.abs(np.concatenate(errs)))
            cv[(gauge, lam)] = float(np.median(out[0]))
            g = short(names[gauge]) if gauge != 'mean' else 'mean'
            say(f'   {lam:>6} {g:>8} {np.median(out[0]) * BIN_MIN:>28.4f} {np.mean(out[0] <= 1):>7.3f} '
                f'{np.median(out[1]) * BIN_MIN:>25.4f} {np.mean(out[1] <= 1):>7.3f}')
    lam = min(LAMBDAS, key=lambda l: cv[(ref, l)])
    say(f'   selected λ = {lam} with gauge = {short(names[ref])}')
    fit = lambda a, **kw: mb.fit_joint(a, K, lam=lam, gauge=ref, **kw)
    fit_pw = lambda a, i, j: mb.fit_pairwise(a, i, j, K, lam=lam, gauge=0)

    # ── Fits on all data ─────────────────────────────────────────────────────
    joint = fit(anc)
    joint_mean = mb.fit_joint(anc, K, lam=lam, gauge='mean')
    pairwise = {(a, b): fit_pw(anc, a, b) for a, b in combinations(range(K), 2)}
    fold_joint = [fit(anc.subset(tr)) for tr, _ in fold_masks]
    fold_pw = [{(a, b): fit_pw(anc.subset(tr), a, b) for a, b in combinations(range(K), 2)}
               for tr, _ in fold_masks]
    pw = lambda d, a, b: d[(min(a, b), max(a, b))]

    # ── 2. TIC r ─────────────────────────────────────────────────────────────
    say('\n' + '=' * 96)
    say('2. Mean between-batch TIC r (A warped into B frame; mean over all A_i × B_j)')
    say('   In-sample: warps fitted on all anchors. Cross-fit: A_i warped by the fold model '
        'that never saw A_i.')
    cols = ['Unaligned', 'Pooled PCHIP (cow-dcnn)', 'Dense PCHIP', 'Pairwise smooth',
            'Joint (mean gauge)', 'Joint', 'Pairwise smooth CF', 'Joint CF']
    say(f'   {"A × B":<18} {"n":>6} ' + ' '.join(f'{c[:12]:>12}' for c in cols))
    R = {}
    for A_, B_ in PAIRS:
        a, b = idx[A_], idx[B_]
        r = {}
        r['Unaligned'] = tic_r(tics[a], tics[b])
        # pooled-anchor PCHIP (warpalign's, now alignbench's): A[0..2] vs B[0], pooled and filtered, PCHIP B → A
        pooled = extract_anchors(chroma[a], chroma[b], batches[a].tic_raw, batches[b].tic_raw)
        r['Pooled PCHIP (cow-dcnn)'] = tic_r(warp_tics(tics[a], src_from_fn(pchip_warp(pooled.anchors))),
                                             tics[b])
        r['Dense PCHIP'] = tic_r(warp_tics(tics[a], mb.pchip_pairwise(anc, b, a)(GRID)), tics[b])
        r['Pairwise smooth'] = tic_r(warp_tics(tics[a], pw(pairwise, a, b).map(b, a, GRID)), tics[b])
        r['Joint (mean gauge)'] = tic_r(warp_tics(tics[a], joint_mean.map(b, a, GRID)), tics[b])
        r['Joint'] = tic_r(warp_tics(tics[a], joint.map(b, a, GRID)), tics[b])
        for name, models in (('Joint CF', fold_joint), ('Pairwise smooth CF', None)):
            rows = []
            for i, t in enumerate(tics[a]):
                f = folds[a][i]
                m = models[f] if models else pw(fold_pw[f], a, b)
                rows.append(tic_r(warp_tics([t], m.map(b, a, GRID)), tics[b])[0])
            r[name] = np.stack(rows)
        R[(A_, B_)] = r
        say(f'   {short(A_) + "×" + short(B_):<18} {r["Unaligned"].size:>6} '
            + ' '.join(f'{r[c].mean():>12.3f}' for c in cols))
    drifted = [p for p in PAIRS if R[p]['Unaligned'].mean() < 0.5]
    for label, sel in (('mean, drifted pairs', drifted),
                       ('mean, near-aligned pairs', [p for p in PAIRS if p not in drifted])):
        say(f'   {label:<25}' + ' '.join(f'{np.mean([R[p][c].mean() for p in sel]):>12.3f}'
                                         for c in cols))
    say('   Drifted = unaligned TIC r < 0.5 (the cow-dcnn exclusion rule).')

    say('\n   Post-hoc sensitivity of the joint model to λ (NOT used for selection), mean TIC r:')
    say(f'   {"λ":>8} {"drifted":>8} {"near-aligned":>13}')
    for l in LAMBDAS:
        m = mb.fit_joint(anc, K, lam=l, gauge=ref)
        v = {p: tic_r(warp_tics(tics[idx[p[0]]], m.map(idx[p[1]], idx[p[0]], GRID)),
                      tics[idx[p[1]]]).mean() for p in PAIRS}
        say(f'   {l:>8g} {np.mean([v[p] for p in drifted]):>8.3f} '
            f'{np.mean([v[p] for p in PAIRS if p not in drifted]):>13.3f}'
            + ('   ← selected' if l == lam else ''))

    say('\n   Paired two-sided Wilcoxon per A sample (r averaged over B_j), Holm over pairs '
        'with n_A ≥ 6:')
    tested = [p for p in PAIRS if sizes[idx[p[0]]] >= 6]
    for x, y in (('Joint', 'Pooled PCHIP (cow-dcnn)'), ('Joint', 'Pairwise smooth'),
                 ('Joint CF', 'Pairwise smooth CF'), ('Joint', 'Unaligned')):
        res = [(p, paired_wilcoxon(R[p][x].mean(1), R[p][y].mean(1))[1],
                float(np.median(R[p][x].mean(1) - R[p][y].mean(1)))) for p in tested]
        adj = holm([q for _, q, _ in res])
        say(f'   {x} vs {y}:')
        for (p, _, md), qa in zip(res, adj):
            say(f'     {short(p[0]) + "×" + short(p[1]):<18} Holm p={qa:.2e}  median Δr {md:+.4f}')

    # ── 3. Held-out anchor error ─────────────────────────────────────────────
    say('\n' + '=' * 96)
    say('3. Cross-fitted held-out anchor error, median |error| in minutes (fraction ≤ 1 bin)')
    say(f'   {"pair":<18} {"n":>6} {"Unaligned":>16} {"Dense PCHIP":>16} {"Pairwise smooth":>16} '
        f'{"Joint":>16}')
    E = {k: [] for k in ('Unaligned', 'Dense PCHIP', 'Pairwise smooth', 'Joint')}
    Epair = []
    for f, (tr, te) in enumerate(fold_masks):
        sub, tr_anc = anc.subset(te), anc.subset(tr)
        E['Unaligned'].append(sub.ta - sub.tb)
        E['Joint'].append(anchor_errors(fold_joint[f], sub))
        e_pw, e_pc = np.empty(len(sub)), np.empty(len(sub))
        for a, b in combinations(range(K), 2):
            m = (sub.ba == a) & (sub.bb == b)
            e_pw[m] = pw(fold_pw[f], a, b).map(a, b, sub.ta[m]) - sub.tb[m]
            e_pc[m] = mb.pchip_pairwise(tr_anc, a, b)(sub.ta[m]) - sub.tb[m]
        E['Pairwise smooth'].append(e_pw)
        E['Dense PCHIP'].append(e_pc)
        Epair.append(np.stack([sub.ba, sub.bb]))
    E = {k: np.abs(np.concatenate(v)) for k, v in E.items()}
    Epair = np.concatenate(Epair, axis=1)
    fmt = lambda e: f'{np.median(e) * BIN_MIN:.3f} ({np.mean(e <= 1):.2f})'
    for a, b in combinations(range(K), 2):
        m = (Epair[0] == a) & (Epair[1] == b)
        say(f'   {short(names[a]) + "×" + short(names[b]):<18} {m.sum():>6} '
            + ' '.join(f'{fmt(E[k][m]):>16}' for k in E))
    say(f'   {"all":<18} {len(Epair[0]):>6} ' + ' '.join(f'{fmt(E[k]):>16}' for k in E))
    say('   Sample folds; λ was selected on separate RT-blocked folds of the same data.')

    # ── 4. Missing correspondences ───────────────────────────────────────────
    say('\n' + '=' * 96)
    say('4. Missing correspondences: all anchors of the evaluated pair are removed')
    say('   Joint LOPO = joint fit without that pair; Hub = pairwise smooth warps composed '
        'through the hub h maximising min(n_ah, n_hb).')
    say(f'   {"A × B":<18} {"hub":>8} {"TIC unal":>9} {"Hub":>7} {"LOPO":>7} {"Joint":>7}   '
        f'{"|err| Hub":>9} {"LOPO":>7} {"Joint":>7} (min, vs the pair\'s own anchors)')
    lopo = {}
    for A_, B_ in PAIRS:
        a, b = idx[A_], idx[B_]
        keep = ~(((anc.ba == a) & (anc.bb == b)) | ((anc.ba == b) & (anc.bb == a)))
        m_lo = fit(anc.subset(keep))
        lopo[(a, b)] = m_lo
        h = max((h for h in range(K) if h not in (a, b)),
                key=lambda h: min(n_pair[a, h], n_pair[h, b]))
        hub = Composed(pw(pairwise, a, h), a, h, pw(pairwise, h, b), b)
        own = anc.pair(a, b)
        row = []
        for m in (hub, m_lo, joint):
            row.append(tic_r(warp_tics(tics[a], m.map(b, a, GRID)), tics[b]).mean())
        errs = [np.median(np.abs(m.map(a, b, own.ta) - own.tb)) * BIN_MIN for m in (hub, m_lo, joint)]
        say(f'   {short(A_) + "×" + short(B_):<18} {short(names[h]):>8} '
            f'{R[(A_, B_)]["Unaligned"].mean():>9.3f} ' + ' '.join(f'{x:>7.3f}' for x in row)
            + '   ' + ' '.join(f'{x:>9.3f}' if i == 0 else f'{x:>7.3f}' for i, x in enumerate(errs)))

    star_keep = (anc.ba == ref) | (anc.bb == ref)
    star = fit(anc.subset(star_keep))
    say(f'\n   Star design: only anchors against {short(names[ref])} '
        f'({star_keep.sum()} of {len(anc)} anchors); evaluated on pairs that exclude it')
    say(f'   {"A × B":<18} {"Unal":>7} {"Star":>7} {"Joint":>7}')
    for A_, B_ in PAIRS:
        a, b = idx[A_], idx[B_]
        if ref in (a, b):
            continue
        rs = tic_r(warp_tics(tics[a], star.map(b, a, GRID)), tics[b]).mean()
        say(f'   {short(A_) + "×" + short(B_):<18} {R[(A_, B_)]["Unaligned"].mean():>7.3f} '
            f'{rs:>7.3f} {R[(A_, B_)]["Joint"].mean():>7.3f}')

    # ── 5. Cycle consistency ─────────────────────────────────────────────────
    say('\n' + '=' * 96)
    say('5. Cycle inconsistency |g_bc(g_ab(t)) − g_ac(t)| over ordered triplets, t ∈ [5, 40] min')
    tt = GRID[(GRID * BIN_MIN >= 5) & (GRID * BIN_MIN <= 40)]
    cyc = {'Pairwise smooth': [], 'Joint': []}
    for a, b, c in permutations(range(K), 3):
        g = lambda i, j, t: pw(pairwise, i, j).map(i, j, t)
        cyc['Pairwise smooth'].append(np.mean(np.abs(g(b, c, g(a, b, tt)) - g(a, c, tt))))
        cyc['Joint'].append(np.mean(np.abs(joint.map(b, c, joint.map(a, b, tt)) - joint.map(a, c, tt))))
    for k, v in cyc.items():
        v = np.array(v) * BIN_MIN
        say(f'   {k:<16} mean {v.mean():.3f} min   max {v.max():.3f} min   (60 triplets)')
    worst = max(permutations(range(K), 3),
                key=lambda p: cyc['Pairwise smooth'][list(permutations(range(K), 3)).index(p)])
    say(f'   Worst pairwise triplet: {" → ".join(short(names[i]) for i in worst)}')

    # ── 6. Uncertainty ───────────────────────────────────────────────────────
    say('\n' + '=' * 96)
    say(f'6. Uncertainty: {n_boot} bootstrap refits, samples resampled within each batch')
    boots = [fit(anc, base_weight=boot_weights(anc, sizes, rng)) for _ in range(n_boot)]
    band = {}
    for a, b in permutations(range(K), 2):
        maps = np.stack([m.map(a, b, GRID) for m in boots])
        band[(a, b)] = np.percentile(maps, [5, 95], axis=0)
    say('   90 % band width (min) vs direct anchor support (anchors of the pair within ±2.5 bins), '
        't ∈ [2, 43] min where the warp is not clamped at a run end')
    say(f'   {"pair":<18} {"median width":>12} {"supported":>10} {"unsupported":>12} '
        f'{"Spearman(width, support)":>25}')
    allw, alls = [], []
    for a, b in combinations(range(K), 2):
        p = anc.pair(a, b)
        sup = np.array([np.sum(np.abs(p.ta - t) <= 2.5) for t in GRID])
        w = (band[(a, b)][1] - band[(a, b)][0]) * BIN_MIN
        inner = (GRID * BIN_MIN >= 2) & (GRID * BIN_MIN <= 43) & in_run(joint, a, b)
        allw.append(w[inner]); alls.append(sup[inner])
        ws = np.median(w[inner & (sup > 0)]) if (inner & (sup > 0)).any() else np.nan
        wu = np.median(w[inner & (sup == 0)]) if (inner & (sup == 0)).any() else np.nan
        rho = spearmanr(w[inner], sup[inner]).statistic
        say(f'   {short(names[a]) + "×" + short(names[b]):<18} {np.median(w[inner]):>12.3f} '
            f'{ws:>10.3f} {wu:>12.3f} {rho:>25.2f}')
    say(f'   all pairs pooled: Spearman ρ(width, support) = '
        f'{spearmanr(np.concatenate(allw), np.concatenate(alls)).statistic:.2f}')

    # Bands when the pair has no direct anchors at all (bootstrapped LOPO)
    say(f'\n   Band width with and without the pair\'s own anchors '
        f'({max(n_boot // 4, 20)} bootstrap refits each, median over t ∈ [2, 43] min):')
    inner = (GRID * BIN_MIN >= 2) & (GRID * BIN_MIN <= 43)
    lopo_width = {}
    for A_, B_ in PAIRS:
        a, b = idx[A_], idx[B_]
        keep = ~(((anc.ba == a) & (anc.bb == b)) | ((anc.ba == b) & (anc.bb == a)))
        sub = anc.subset(keep)
        maps = np.stack([fit(sub, base_weight=boot_weights(sub, sizes, rng)).map(a, b, GRID)
                         for _ in range(max(n_boot // 4, 20))])
        q = np.percentile(maps, [5, 95], axis=0)
        lopo_width[(a, b)] = q
        ok = inner & in_run(joint, a, b)
        w_with = np.median((band[(a, b)][1] - band[(a, b)][0])[ok]) * BIN_MIN
        w_wo   = np.median((q[1] - q[0])[ok]) * BIN_MIN
        say(f'   {short(A_) + "×" + short(B_):<18} with {w_with:.3f} min   without {w_wo:.3f} min'
            f'   ×{w_wo / max(w_with, 1e-9):.1f}')

    # Predictive-interval coverage on held-out anchors
    say(f'\n   Held-out anchor coverage of 90 % intervals ({n_boot_cv} bootstrap refits per fold):')
    cover_pi, cover_ci, gross = [], [], []
    for f, (tr, te) in enumerate(fold_masks):
        tr_anc, sub = anc.subset(tr), anc.subset(te)
        bs = [fit(tr_anc, base_weight=boot_weights(tr_anc, sizes, rng)) for _ in range(n_boot_cv)]
        for a, b in combinations(range(K), 2):
            m = (sub.ba == a) & (sub.bb == b)
            if not m.any():
                continue
            pred = fold_joint[f].map(a, b, sub.ta[m])
            q = np.percentile(np.stack([x.map(a, b, sub.ta[m]) for x in bs]), [5, 95], axis=0)
            hw_ci = (q[1] - q[0]) / 2
            sig = fold_joint[f].scale[a, b]
            hw_pi = np.sqrt(hw_ci ** 2 + (1.645 * sig) ** 2)
            err = np.abs(sub.tb[m] - pred)
            cover_ci.append(err <= hw_ci); cover_pi.append(err <= hw_pi); gross.append(err > 10)
    cover_ci, cover_pi, gross = map(np.concatenate, (cover_ci, cover_pi, gross))
    say(f'   warp-only band (epistemic)          : {cover_ci.mean():.3f}')
    say(f'   predictive (band ⊕ anchor scatter)  : {cover_pi.mean():.3f}   '
        f'excluding gross mismatches (|err| > 10 bins, {gross.mean():.1%} of anchors): '
        f'{cover_pi[~gross].mean():.3f}')

    np.savez(RESULTS / 'multibatch_align.npz', names=np.array(names), lam=lam, ref=ref,
             provenance=np.array(header()),
             knots=joint.knots, drift=joint.drift, n_pair=n_pair,
             boot_drift=np.stack([m.drift for m in boots]),
             **{f'ticr_{short(A_)}_{short(B_)}_{c.replace(" ", "_")}': v
                for (A_, B_), r in R.items() for c, v in r.items()})
    say(f'\nWall time {time.time() - t0:.0f} s')
    (RESULTS / 'multibatch_align.txt').write_text('\n'.join(L) + '\n')
    plot(names, anc, joint, pairwise, band, lopo_width, idx)


def plot(names, anc, joint, pairwise, band, lopo_width, idx) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#9a9893', '#0b0b0b', '#52514e'
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, constrained_layout=True)
    t_min = GRID * BIN_MIN
    for ax, (A_, B_) in zip(axes.flat, PAIRS):
        a, b = idx[A_], idx[B_]
        p = anc.pair(a, b)
        ax.scatter(p.ta * BIN_MIN, (p.tb - p.ta) * BIN_MIN, s=3, color=GRAY, alpha=0.25,
                   linewidths=0, rasterized=True, label='Direct anchors')
        lo, hi = sorted((a, b))
        q = lopo_width[(a, b)]
        ax.fill_between(t_min, (q[0] - GRID) * BIN_MIN, (q[1] - GRID) * BIN_MIN, color=BLUE,
                        alpha=0.10, linewidth=0, label='90 % band without direct anchors')
        q = band[(a, b)]
        g_joint = joint.map(a, b, GRID)
        ax.fill_between(t_min, (q[0] - GRID) * BIN_MIN, (q[1] - GRID) * BIN_MIN, color=BLUE,
                        alpha=0.30, linewidth=0, label='Joint 90 % band')
        ax.plot(t_min, (pairwise[(lo, hi)].map(a, b, GRID) - GRID) * BIN_MIN, color=ORANGE,
                lw=1.5, ls='--', label='Pairwise smooth')
        ax.plot(t_min, (g_joint - GRID) * BIN_MIN, color=BLUE, lw=2, label='Joint')
        d = (g_joint - GRID) * BIN_MIN
        span = max(1.0, np.ptp(d))
        ax.set_ylim(d.min() - 0.6 * span, d.max() + 0.6 * span)
        ax.set_title(f'{short(A_)} → {short(B_)}  (n={len(p)})', color=INK, fontsize=9)
        ax.grid(axis='y', color='#e6e5e0', lw=0.6)
    for ax in axes[1]:
        ax.set_xlabel('RT in batch A (min)')
    for ax in axes[:, 0]:
        ax.set_ylabel('RT shift A → B (min)')
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.suptitle('Joint many-batch warps vs pairwise fits, fish oil GC-MS', color=INK,
                 fontsize=11, y=1.06)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / 'multibatch_warps.png', dpi=160, bbox_inches='tight')
    print(f'  figure → {FIGS / "multibatch_warps.png"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--n-boot-cv', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data', type=Path, default=DATA,
                    help='batch_* folders in warpalign layout (default: the fish oil data; '
                         'scripts/make_synthetic_data.py writes a public synthetic stand-in)')
    ap.add_argument('--results', type=Path, default=RESULTS, help='output folder (default: results/)')
    ap.add_argument('--figs', type=Path, default=FIGS, help='figure folder (default: figs/)')
    a = ap.parse_args()
    DATA, RESULTS, FIGS = a.data, a.results, a.figs
    RESULTS.mkdir(parents=True, exist_ok=True)
    main(a.n_boot, a.n_boot_cv, a.seed)
