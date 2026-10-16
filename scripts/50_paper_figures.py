"""
50_paper_figures.py — Publication figures (vector PDF + PNG preview) and captions.

Reads the same results as the project site (scripts/analysis.py), so the paper and the site cannot
disagree. Main figures:

  fig1_problem_method    TICs of every batch on its own clock and in the shared frame; the
                         shared frame drawn from the fitted warps
  fig2_recovery          anchors and fits for one drifted pair; recovery of known drifts on the
                         synthetic data (src/jointalign/synthetic.py)
  fig3_comparison        TIC r per batch-A sample for the anchor-based methods; held-out anchor
                         error per pair
  fig4_consistency       cycle inconsistency over all batch triplets; the worst route
  fig5_borrowing         TIC r with and without each pair's own anchors; the largest change
  fig6_benchmark         fair benchmark: every method, both protocols, three pair groups

Supplementary: figS1_lambda (λ selection and sensitivity, both gauges), figS2_uncertainty
(coverage, band width vs anchor support), figS3_rejection (robust weights along the run).

Output: figs/paper/*.pdf, figs/paper/*.png, figs/paper/captions.md
"""
from __future__ import annotations

import argparse
import tempfile
from itertools import combinations, pairwise
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alignbench.axis import FISH_OIL_AXIS
from alignbench.datasets.fish_oil import load_batch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import analysis as an
from jointalign import multibatch as mb
from jointalign import synthetic as syn

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / 'results'
OUT = ROOT / 'figs' / 'paper'
DATA: Path | None = None              # None: script 30's default (the fish oil data)
GRID = an.GRID
BIN_MIN, EVAL_START_BIN = FISH_OIL_AXIS.bin_min, FISH_OIL_AXIS.eval_start_bin
T_MIN = GRID * BIN_MIN
FULL, HALF = 7.2, 3.5                                    # inches: double / single column

# Validated categorical palette (dataviz reference, light mode), assigned in fixed order.
S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, MUTED, GRIDC, AXIS, SURF2 = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#f2f1ed'
METHOD_COLOR = {'Joint': S[0], 'Pairwise smooth': S[1], 'Dense PCHIP': S[2], 'Pooled PCHIP (cow-dcnn)': S[3]}
METHOD_LABEL = {'Pooled PCHIP (cow-dcnn)': 'Pooled-anchor PCHIP'}

plt.rcParams.update({
    'font.family': ['Helvetica', 'DejaVu Sans'],      # DejaVu fills glyphs Helvetica lacks (∘, →, ⊕)
    'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': 0.6, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRIDC, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 1.3, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.dpi': 300, 'figure.dpi': 150, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left',
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  {name}')


def panel(ax, letter: str, title: str = '') -> None:
    ax.set_title(f'{letter}  {title}' if title else letter, loc='left', pad=6)


def lab(n: str) -> str:
    return an.short(n)


def abbr(n: str) -> str:
    """'batch_sep2015' → 'sep15' (compact tick labels)."""
    s = an.short(n)
    return s[:3] + s[-2:]


def dodge(ys, gap: float):
    """Spread label positions so neighbours are at least gap apart, keeping their order."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        if out[order[i]] - out[order[i - 1]] < gap:
            out[order[i]] = out[order[i - 1]] + gap
    return out


def bcolor(k: int) -> str:
    return S[k % len(S)]


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def fig1(run: an.Run, cap: dict) -> None:
    fig = plt.figure(figsize=(FULL, 4.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0], hspace=0.42, wspace=0.08)
    ax_raw, ax_al = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, :])
    K, ref = run.K, run.ref
    x0, x1 = 10.0, 32.0                                   # zoom: a 4-min drift is visible at this scale
    marks = np.array([16.3, 21.6, 27.0])                  # compounds, in the reference batch's time (min)
    for ax, aligned in ((ax_raw, False), (ax_al, True)):
        for k, tics in enumerate(run.tics):
            sel = np.linspace(0, len(tics) - 1, min(3, len(tics))).round().astype(int)
            scale = np.percentile(tics[:, EVAL_START_BIN:], 99.5)
            src = run.joint.map(ref, k, GRID) if aligned else GRID
            off = (K - 1 - k) * 1.0
            for s in sel:
                y = run.s30.warp_tics(tics[s:s + 1], src)[0] / scale
                ax.plot(T_MIN, off + 0.85 * np.clip(y, 0, 1.15), color=bcolor(k), lw=0.6, alpha=0.85)
            ax.axhline(off, color=GRIDC, lw=0.5, zorder=0)
            if not aligned:
                ax.text(x0 - 0.4, off + 0.25, lab(run.names[k]), color=bcolor(k), ha='right', va='center',
                        fontsize=7, fontweight='bold')
        # where each batch sees the marked compounds: staggered on own clocks, aligned in the shared frame
        for tau in marks:
            xs = [tau if aligned else run.joint.inverse(k, np.array([tau / BIN_MIN]))[0] * BIN_MIN
                  for k in range(K)]
            yk = [(K - 1 - k) * 1.0 + 1.0 for k in range(K)]
            ax.plot(xs, yk, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=4)
            ax.plot(xs, yk, 'v', ms=3.2, color=INK, zorder=5)
        ax.set_xlim(x0, x1); ax.set_ylim(-0.15, K + 0.25)
        ax.set_yticks([]); ax.grid(False); ax.spines['left'].set_visible(False)
        ax.set_xlabel('retention time (min)')
    panel(ax_raw, 'a', 'Each batch on its own clock (10–32 min)')
    panel(ax_al, 'b', f'Joint warps: every batch in the shared frame ({lab(run.names[ref])})')

    # c: the shared frame, drawn from the fitted warps. Each batch axis is mapped into the latent
    # axis by f_k; the same compounds (latent positions) land at different times in each batch.
    j = run.joint
    taus = np.array([9.0, 17.0, 25.0, 33.0, 41.0]) / BIN_MIN
    ys = {k: 1.0 + (K - 1 - k) * 0.42 for k in range(K)}
    for k in range(K):
        ax_s.plot([0, T_MIN[-1]], [ys[k], ys[k]], color=bcolor(k), lw=1.0)
        ax_s.text(-0.6, ys[k], lab(run.names[k]), color=bcolor(k), ha='right', va='center', fontsize=6.5)
    ax_s.plot([0, T_MIN[-1]], [0, 0], color=INK, lw=1.2)
    ax_s.text(-0.6, 0, 'shared clock', color=INK, ha='right', va='center', fontsize=6.5, fontweight='bold')
    for i, tau in enumerate(taus):
        xs = [j.inverse(k, np.array([tau]))[0] * BIN_MIN for k in range(K)]
        for k in range(K):
            ax_s.plot([xs[k], tau * BIN_MIN], [ys[k], 0], color=MUTED, lw=0.45, alpha=0.7, zorder=1)
            ax_s.plot(xs[k], ys[k], 'o', ms=3.2, color=bcolor(k), mec='white', mew=0.5, zorder=3)
        ax_s.plot(tau * BIN_MIN, 0, 'o', ms=4, color=INK, mec='white', mew=0.6, zorder=3)
        if i == 0:
            ax_s.annotate(r'$f_k$: batch time $\rightarrow$ shared time', xy=(tau * BIN_MIN, 0),
                          xytext=(tau * BIN_MIN + 1.5, -0.42), fontsize=6.5, color=INK2,
                          arrowprops={'arrowstyle': '-', 'color': MUTED, 'lw': 0.5})
    ax_s.set_xlim(0, T_MIN[-1]); ax_s.set_ylim(-0.6, ys[0] + 0.25)
    ax_s.set_yticks([]); ax_s.grid(False); ax_s.spines['left'].set_visible(False)
    ax_s.set_xlabel('retention time (min)')
    panel(ax_s, 'c', r'One monotone warp per batch; any pairwise warp is $f_l^{-1} \circ f_k$')
    save(fig, 'fig1_problem_method')
    cap['fig1'] = (
        f'**The problem and the method.** (a) Normalised TICs of three samples from each of the {K} fish '
        f'oil batches ({", ".join(f"{lab(n)} n={len(b)}" for n, b in zip(run.names, run.batches))}), each on '
        f'its own retention-time axis. Shaded: before {EVAL_START_BIN * BIN_MIN:.0f} min, when not every run '
        f'had started acquiring; excluded from every score. Triangles joined by dashed lines: the same three '
        f'compounds in every batch (shown 10–32 min). (b) The same TICs mapped into the shared frame by '
        f'the joint warps (gauge: {lab(run.names[run.ref])}). (c) The shared frame drawn from the fitted '
        f'warps: five compounds (black, shared clock) and where each batch sees them. Every batch has one '
        f'monotone warp f_k into the shared clock, so the warp between any two batches is f_l⁻¹ ∘ f_k and all '
        f'pairwise warps are mutually consistent.')


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def synthetic_fit(seed: int = 0):
    """Synthetic data with known drifts → anchors → λ by RT-blocked CV → joint fit (as script 30)."""
    s30 = an.load_script('30_multibatch_align.py', 's30_syn')
    with tempfile.TemporaryDirectory() as tmp:
        truth = syn.make_dataset(Path(tmp), seed=seed)
        names = sorted(truth['batches'])
        batches = [load_batch(n, Path(tmp)) for n in names]
    anc = mb.harvest_anchors([b.chromas for b in batches], [b.tic_raw for b in batches])
    K = len(names)
    ref = int(np.argmax([sum(len(anc.pair(a, b)) for b in range(K) if b != a) for a in range(K)]))
    blk = lambda t: (np.floor(t / s30.RT_BLOCK).astype(int)) % s30.N_FOLDS
    cv = {}
    for lm in s30.LAMBDAS:
        errs = []
        for f in range(s30.N_FOLDS):
            te = (blk(anc.ta) == f) | (blk(anc.tb) == f)
            tr = (blk(anc.ta) != f) & (blk(anc.tb) != f)
            errs.append(s30.anchor_errors(mb.fit_joint(anc.subset(tr), K, lam=lm, gauge=ref), anc.subset(te)))
        cv[lm] = float(np.median(np.abs(np.concatenate(errs))))
    lam = min(cv, key=cv.get)
    return truth, names, anc, mb.fit_joint(anc, K, lam=lam, gauge=ref), ref, lam


def fig2(run: an.Run, cap: dict) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.45),
                            gridspec_kw={'width_ratios': [1.35, 1, 0.75], 'wspace': 0.62})
    # a: anchors and fits for the drifted pair with the most anchors
    drifted_idx = [(run.idx[a], run.idx[b]) for a, b in run.drifted]
    a, b = max(drifted_idx, key=lambda p: len(run.anc.pair(*p)))
    p = run.anc.pair(a, b)
    rows = np.flatnonzero(((run.anc.ba == a) & (run.anc.bb == b)) | ((run.anc.ba == b) & (run.anc.bb == a)))
    w = run.joint.weights[rows]
    order = np.argsort(np.where(run.anc.ba[rows] == a, run.anc.ta[rows], run.anc.tb[rows]))
    ta = np.where(run.anc.ba[rows] == a, run.anc.ta[rows], run.anc.tb[rows])[order]
    tb = np.where(run.anc.ba[rows] == a, run.anc.tb[rows], run.anc.ta[rows])[order]
    w = w[order]
    d = (tb - ta) * BIN_MIN
    ax = axs[0]
    rng = np.random.default_rng(0)
    show = rng.random(len(ta)) < min(1.0, 2500 / len(ta))
    ax.scatter(ta[show & (w > 0)] * BIN_MIN, d[show & (w > 0)], s=2, color=MUTED, alpha=0.25, lw=0,
               rasterized=True)
    ax.scatter(ta[show & (w == 0)] * BIN_MIN, d[show & (w == 0)], s=7, facecolor='none', edgecolor=S[7], lw=0.5,
               rasterized=True)
    q = an.band(run, a, b) * BIN_MIN
    ax.fill_between(T_MIN, q[0], q[1], color=S[0], alpha=0.25, lw=0)
    dense = mb.pchip_pairwise(run.anc, a, b)(GRID)
    ax.plot(T_MIN, an.shift(dense) * BIN_MIN, color=S[2], lw=0.9, label='Dense PCHIP')
    ax.plot(T_MIN, an.shift(an.pw_map(run, a, b, GRID)) * BIN_MIN, color=S[1], lw=1.0, ls='--',
            label='Pairwise smooth')
    ax.plot(T_MIN, an.shift(run.joint.map(a, b, GRID)) * BIN_MIN, color=S[0], lw=1.4, label='Joint (90 % band)')
    lo, hi = np.nanpercentile(d[w > 0], [0.5, 99.5])
    ax.set_ylim(lo - 0.7, hi + 0.7)
    ax.set_xlim(0, T_MIN[-1])
    ax.set_xlabel(f'retention time in {lab(run.names[a])} (min)')
    ax.set_ylabel(f'shift into {lab(run.names[b])} (min)')
    h = ax.get_legend_handles_labels()
    extra = [Line2D([], [], marker='o', ls='', ms=3, color=MUTED, alpha=0.6),
             Line2D([], [], marker='o', ls='', ms=3.5, mfc='none', mec=S[7], mew=0.6)]
    ax.legend(h[0] + extra, h[1] + ['anchor', 'anchor, rejected'], loc='lower left', ncol=1, fontsize=6,
              handlelength=1.6, labelspacing=0.3)
    panel(ax, 'a', f'{lab(run.names[a])} × {lab(run.names[b])}: {len(p):,} anchors')

    # b, c: synthetic data with known drifts
    truth, names, anc_s, js, ref_s, lam_s = synthetic_fit()
    ax = axs[1]
    tau = np.asarray(truth['compound_tau_min'])
    spec = {s.name: s for s in syn.DEFAULT_BATCHES}
    span = {n: tau + spec[n].drift(tau) for n in names}
    ends, lab_names, lab_cols = [], [], []
    for k, n in enumerate(names):
        if k == ref_s:
            continue
        tt = np.linspace(span[n].min(), span[n].max(), 200)       # where batch k has compounds
        true = syn.true_map(truth, n, names[ref_s], tt) - tt
        est = js.map(k, ref_s, tt / BIN_MIN) * BIN_MIN - tt
        ax.plot(tt, true, color=bcolor(k), lw=2.6, alpha=0.3)
        ax.plot(tt, est, color=bcolor(k), lw=0.9)
        ends.append(est[-1]); lab_names.append(lab(n)); lab_cols.append(bcolor(k))
    for y, n, c in zip(dodge(ends, 0.42), lab_names, lab_cols):
        ax.text(43.0, y, n, color=c, fontsize=6.3, va='center')
    ax.set_xlabel('retention time in batch (min)'); ax.set_ylabel(f'shift into {lab(names[ref_s])} (min)')
    ax.set_xlim(2, 50)
    ax.legend([Line2D([], [], color=INK2, lw=2.6, alpha=0.3), Line2D([], [], color=INK2, lw=0.9)],
              ['true drift', 'joint estimate'], loc='upper left', fontsize=6.3)
    panel(ax, 'b', 'Synthetic batches: known drift')
    # c: error over every pair, where compounds exist in both batches
    errs, labels = [], []
    for i, j in combinations(range(len(names)), 2):
        t = np.linspace(8, 40, 200)
        true = syn.true_map(truth, names[i], names[j], t)
        ok = ((t >= span[names[i]].min()) & (t <= span[names[i]].max())
              & (true >= span[names[j]].min()) & (true <= span[names[j]].max()))
        e = np.abs(js.map(i, j, t[ok] / BIN_MIN) * BIN_MIN - true[ok])
        errs.append(e); labels.append(f'{abbr(names[i])} × {abbr(names[j])}')
    ax = axs[2]
    o = np.argsort([np.median(e) for e in errs])
    for r_, ii in enumerate(o):
        ax.plot([np.percentile(errs[ii], 5), np.percentile(errs[ii], 95)], [r_, r_], color=AXIS, lw=1.6,
                solid_capstyle='round')
        ax.plot(np.median(errs[ii]), r_, 'o', ms=3.5, color=S[0])
    ax.set_yticks(range(len(o))); ax.set_yticklabels([labels[i] for i in o], fontsize=6)
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_xlabel('error vs truth (min)')
    allerr = np.concatenate(errs)
    panel(ax, 'c', 'Error, every pair')
    save(fig, 'fig2_recovery')
    cap['fig2'] = (
        f'**Anchors, fits and recovery of known drift.** (a) The {len(p):,} anchors between '
        f'{lab(run.names[a])} and {lab(run.names[b])} (grey; red rings: weight 0 after robust reweighting, '
        f'{np.mean(w == 0):.1%}) with the joint warp and its 90 % bootstrap band, the same fitter on this '
        f'pair\'s anchors alone (pairwise smooth) and PCHIP through the median anchors (dense PCHIP). '
        f'(b) Synthetic batches (src/synthetic.py) with known drift: true shift into {lab(names[ref_s])} '
        f'(thick, faint) and the joint estimate (thin), fitted exactly as on the real data '
        f'({len(anc_s):,} anchors, λ = {lam_s:g} by RT-blocked cross-validation). (c) Absolute error of the '
        f'joint warp for every batch pair where compounds exist in both batches: median (dot) and 5–95th '
        f'percentile. Median {np.median(allerr):.3f} min, 95th percentile {np.percentile(allerr, 95):.3f} min, '
        f'maximum {allerr.max():.3f} min (one RT bin is {BIN_MIN} min).')


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def fig3(run: an.Run, cap: dict) -> None:
    methods = ['Pooled PCHIP (cow-dcnn)', 'Dense PCHIP', 'Pairwise smooth', 'Joint']
    pairs = [p for p in run.pairs if p in run.drifted] + [p for p in run.pairs if p not in run.drifted]
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw={'width_ratios': [2.6, 1], 'wspace': 0.28})
    ax = axs[0]
    width = 0.8 / len(methods)
    rng = np.random.default_rng(1)
    for i, pair in enumerate(pairs):
        un = run.ticr(pair, 'Unaligned').mean()
        ax.plot([i - 0.42, i + 0.42], [un, un], color=MUTED, lw=0.8, ls=':', zorder=1)
        for m_i, m in enumerate(methods):
            v = run.ticr(pair, m).mean(1)
            x = i - 0.4 + width * (m_i + 0.5)
            ax.scatter(x + rng.uniform(-width * 0.28, width * 0.28, len(v)), v, s=3, color=METHOD_COLOR[m],
                       alpha=0.45, lw=0, zorder=2)
            ax.plot([x - width * 0.38, x + width * 0.38], [v.mean()] * 2, color=METHOD_COLOR[m], lw=1.8,
                    solid_capstyle='butt', zorder=3)
    n_dr = sum(p in run.drifted for p in pairs)
    ax.axvline(n_dr - 0.5, color=AXIS, lw=0.6)
    ax.text(n_dr / 2 - 0.5, 1.02, 'drifted pairs (unaligned r < 0.5)', ha='center', fontsize=6.5, color=INK2,
            transform=ax.get_xaxis_transform())
    ax.text(n_dr + (len(pairs) - n_dr) / 2 - 0.5, 1.02, 'near-aligned pairs', ha='center', fontsize=6.5,
            color=INK2, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f'{abbr(a)}\n× {abbr(b)}' for a, b in pairs], fontsize=6.2)
    ax.set_xlim(-0.55, len(pairs) - 0.45)
    ax.set_ylabel('TIC r per batch-A sample')
    handles = [Line2D([], [], color=METHOD_COLOR[m], lw=1.8) for m in methods]
    handles.append(Line2D([], [], color=MUTED, ls=':'))
    ax.legend(handles, [METHOD_LABEL.get(m, m) for m in methods] + ['unaligned (mean)'], loc='lower left',
              ncol=5, fontsize=6.2, bbox_to_anchor=(0, -0.27), handlelength=1.4, columnspacing=1.0)
    panel(ax, 'a', '')
    # b: held-out anchor error per pair
    he = an.heldout_errors(run)
    ax = axs[1]
    keys = [k for k in he if k != 'all']
    keys.sort(key=lambda k: he[k]['Joint'])
    for r_, k in enumerate(keys):
        for m, dy in (('Dense PCHIP', -0.22), ('Pairwise smooth', 0.0), ('Joint', 0.22)):
            ax.plot(he[k][m], r_ + dy, 'o', ms=3.2, color=METHOD_COLOR[m])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([' × '.join(x[:3] + x[-2:] for x in k.split('×')) for k in keys], fontsize=6.2)
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_xlabel('held-out anchor error,\nmedian (min)')
    ax.legend([Line2D([], [], marker='o', ls='', ms=3.2, color=METHOD_COLOR[m])
               for m in ('Dense PCHIP', 'Pairwise smooth', 'Joint')],
              ['Dense PCHIP', 'Pairwise smooth', 'Joint'], loc='lower right', fontsize=6, handletextpad=0.2)
    ax.set_xlim(0, max(he[k]['Joint'] for k in keys) * 1.15)
    panel(ax, 'b', 'Held-out anchors')
    save(fig, 'fig3_comparison')
    dr, na = run.drifted, [p for p in run.pairs if p not in run.drifted]
    mean = lambda ps, m: np.mean([run.ticr(p, m).mean() for p in ps])
    cap['fig3'] = (
        f'**Alignment quality.** (a) TIC r of every batch-A sample against batch B (mean over B samples) after '
        f'warping A into B\'s frame, on the common acquisition window; bars are pair means, dotted lines the '
        f'unaligned mean. Means over drifted pairs: joint {mean(dr, "Joint"):.3f}, pairwise smooth '
        f'{mean(dr, "Pairwise smooth"):.3f}, dense PCHIP {mean(dr, "Dense PCHIP"):.3f}, pooled-anchor PCHIP '
        f'{mean(dr, "Pooled PCHIP (cow-dcnn)"):.3f}, unaligned {mean(dr, "Unaligned"):.3f}; near-aligned pairs: '
        f'joint {mean(na, "Joint"):.3f}, unaligned {mean(na, "Unaligned"):.3f}. (b) Cross-fitted held-out anchor '
        f'error (folds over samples), median per batch pair; over all pairs dense PCHIP '
        f'{he["all"]["Dense PCHIP"]:.3f}, pairwise smooth {he["all"]["Pairwise smooth"]:.3f}, joint '
        f'{he["all"]["Joint"]:.3f} min.')


# ── Figure 4 ──────────────────────────────────────────────────────────────────

def fig4(run: an.Run, cap: dict) -> None:
    cyc = sorted(an.cycle_table(run), key=lambda c: -c['pairwise'])
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.3), gridspec_kw={'width_ratios': [1.2, 1], 'wspace': 0.28})
    ax = axs[0]
    x = np.arange(len(cyc))
    ax.bar(x, [c['pairwise'] for c in cyc], width=0.78, color=S[1], label='pairwise warps')
    ax.plot(x, [c['joint'] for c in cyc], color=S[0], lw=1.4, label='joint warps')
    ax.set_xlim(-0.8, len(cyc) - 0.2); ax.set_xticks([])
    ax.set_xlabel(f'{len(cyc)} ordered batch triplets (a → b → c), worst first')
    ax.set_ylabel(r'mean $|g_{bc} \circ g_{ab} - g_{ac}|$ (min)')
    ax.legend(loc='upper right')
    panel(ax, 'a', 'Cycle inconsistency, 5–40 min')
    a, b, c = cyc[0]['t']
    ax = axs[1]
    direct = an.shift(an.pw_map(run, a, c, GRID)) * BIN_MIN
    comp = an.shift(an.pw_map(run, b, c, an.pw_map(run, a, b, GRID))) * BIN_MIN
    jd = an.shift(run.joint.map(a, c, GRID)) * BIN_MIN
    ax.fill_between(T_MIN, direct, comp, color=S[1], alpha=0.2, lw=0)
    ax.plot(T_MIN, direct, color=S[1], lw=1.2, label=f'pairwise {lab(run.names[a])} → {lab(run.names[c])}')
    ax.plot(T_MIN, comp, color=S[1], lw=1.2, ls='--', label=f'pairwise via {lab(run.names[b])}')
    ax.plot(T_MIN, jd, color=S[0], lw=1.2, label='joint, direct = via any batch')
    ax.axvspan(0, 5, color=SURF2, lw=0, zorder=0); ax.axvspan(40, T_MIN[-1], color=SURF2, lw=0, zorder=0)
    ax.set_xlim(0, T_MIN[-1])
    ax.set_xlabel(f'retention time in {lab(run.names[a])} (min)')
    ax.set_ylabel(f'shift into {lab(run.names[c])} (min)')
    ax.legend(loc='best', fontsize=6.2)
    panel(ax, 'b', 'Worst route')
    save(fig, 'fig4_consistency')
    pwv = np.array([q['pairwise'] for q in cyc]); jv = np.array([q['joint'] for q in cyc])
    cap['fig4'] = (
        f'**Transitivity.** (a) Disagreement between composing two pairwise warps and the direct warp, '
        f'|g_bc(g_ab(t)) − g_ac(t)| averaged over 5–40 min, for all {len(cyc)} ordered batch triplets. Pairwise '
        f'warps: mean {pwv.mean():.3f} min, maximum {pwv.max():.3f} min; joint warps: maximum {jv.max():.4f} min '
        f'(numerical). (b) The worst route, {lab(run.names[a])} → {lab(run.names[b])} → {lab(run.names[c])}: '
        f'the direct pairwise warp (solid) and the composed one (dashed) differ (shaded); the joint warp is the '
        f'same along every route by construction.')


# ── Figure 5 ──────────────────────────────────────────────────────────────────

def fig5(run: an.Run, cap: dict) -> None:
    lp = an.lopo(run)
    st = an.star(run)
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.5), gridspec_kw={'width_ratios': [1, 1.15], 'wspace': 0.42})
    ax = axs[0]
    keys = list(lp)
    for r_, (a, b) in enumerate(keys):
        v = lp[(a, b)]
        ax.plot([v['ticr'], v['ticr_joint']], [r_, r_], color=AXIS, lw=1.4, zorder=1)
        ax.plot(v['ticr_joint'], r_, 'o', ms=4.2, color=S[0], zorder=3)
        ax.plot(v['ticr'], r_, 'o', ms=6.2, mfc='none', mec=S[1], mew=1.0, zorder=2)
        if run.ref not in (a, b):
            ax.plot(run.tic_r_of_map(a, b, st), r_, 'x', ms=4, color=INK2, mew=0.9, zorder=4)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f'{lab(run.names[a])} × {lab(run.names[b])}' for a, b in keys], fontsize=6.3)
    ax.invert_yaxis(); ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_xlabel('TIC r (A warped into B)')
    ax.legend([Line2D([], [], marker='o', ls='', ms=4.2, color=S[0]),
               Line2D([], [], marker='o', ls='', ms=6.2, mfc='none', mec=S[1]),
               Line2D([], [], marker='x', ls='', ms=4, color=INK2)],
              ['all anchors', "pair's own anchors removed", f'star: only anchors vs {lab(run.names[run.ref])}'],
              loc='lower left', fontsize=6, bbox_to_anchor=(-0.05, -0.42), ncol=2)
    panel(ax, 'a', 'Without direct anchors')
    (a, b), v = max(lp.items(), key=lambda kv: kv[1]['ticr_joint'] - kv[1]['ticr'])
    ax = axs[1]
    own = run.anc.pair(a, b)
    ax.scatter(own.ta * BIN_MIN, (own.tb - own.ta) * BIN_MIN, s=1.5, color=MUTED, alpha=0.18, lw=0,
               rasterized=True)
    ax.plot(T_MIN, an.shift(run.joint.map(a, b, GRID)) * BIN_MIN, color=S[0], lw=1.3, label='all anchors')
    ax.plot(T_MIN, an.shift(v['model'].map(a, b, GRID)) * BIN_MIN, color=S[1], lw=1.3, ls='--',
            label="pair's anchors removed")
    dd = (own.tb - own.ta) * BIN_MIN
    lo, hi = np.percentile(dd, [1, 99])
    ax.set_ylim(lo - 0.5, hi + 0.5); ax.set_xlim(0, T_MIN[-1])
    ax.set_xlabel(f'retention time in {lab(run.names[a])} (min)')
    ax.set_ylabel(f'shift into {lab(run.names[b])} (min)')
    ax.legend(loc='best', fontsize=6.2)
    panel(ax, 'b', f'Largest change: {lab(run.names[a])} × {lab(run.names[b])}')
    save(fig, 'fig5_borrowing')
    diffs = sorted(abs(x['ticr_joint'] - x['ticr']) for x in lp.values())
    cap['fig5'] = (
        f'**Borrowing strength across batches.** (a) TIC r of each evaluated pair with the joint model fitted '
        f'on all anchors (filled) and refitted without any anchor of that pair (open); crosses: fitted only on '
        f'anchors against the reference batch {lab(run.names[run.ref])} (star design, pairs not involving it). '
        f'Removing a pair\'s anchors changes its TIC r by at most {diffs[-2]:.3f} on {len(diffs) - 1} of '
        f'{len(diffs)} pairs and by {diffs[-1]:.3f} on {lab(run.names[a])} × {lab(run.names[b])}. (b) That pair: '
        f'its own anchors (grey), the joint warp with them and without them; median error against its own '
        f'anchors rises from {v["err_joint_min"]:.3f} to {v["err_min"]:.3f} min.')


# ── Figure 6 ──────────────────────────────────────────────────────────────────

def fig6(run: an.Run, cap: dict) -> None:
    fb = RESULTS / 'fair_benchmark.npz'
    if not fb.exists():
        print('  fig6_benchmark skipped: results/fair_benchmark.npz missing (run script 32)')
        return
    zb = np.load(fb)
    methods = ['Unaligned', 'Pooled PCHIP', 'PCHIP + COW refine', 'COW-TIC', 'WarpTransformer', 'Joint',
               'Neural joint']
    aug = [p for p in run.pairs if p[0].endswith('aug2015')]
    groups = {'Drifted pairs': [p for p in run.pairs if p in run.drifted and p not in aug],
              'Near-aligned pairs': [p for p in run.pairs if p not in run.drifted and p not in aug],
              'aug2015 pairs (n = 2)': aug}
    fig, axs = plt.subplots(2, 3, figsize=(FULL, 3.6), sharey=True, gridspec_kw={'hspace': 0.55, 'wspace': 0.12})
    color = {m: (S[0] if m == 'Joint' else '#86b6ef' if m == 'Neural joint' else AXIS) for m in methods}
    summary = {}
    for r_, proto in enumerate(('B-frame', 'Midpoint')):
        for c_, (g, ps) in enumerate(groups.items()):
            ax = axs[r_, c_]
            vals = []
            for m in methods:
                ks = [f'{an.short(a)}_{an.short(b)}|{m}|{proto}' for a, b in ps]
                vals.append(np.mean([zb[k].mean() for k in ks]) if all(k in zb.files for k in ks) else np.nan)
            summary[(proto, g)] = dict(zip(methods, vals))
            y = np.arange(len(methods))
            ax.barh(y, vals, color=[color[m] for m in methods], height=0.68)
            best = np.nanmax(vals)
            for yi, v in zip(y, vals):
                if np.isfinite(v):
                    ax.text(max(v, 0) + 0.015, yi, f'{v:.2f}', va='center', fontsize=6,
                            fontweight='bold' if v == best else 'normal', color=INK if v == best else INK2)
            ax.set_xlim(0, 1.08); ax.grid(axis='x'); ax.grid(axis='y', visible=False)
            ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=6.3); ax.invert_yaxis()
            ax.set_title(f'{g} · {proto}', fontsize=7, loc='left', pad=4)
            if r_ == 1:
                ax.set_xlabel('mean TIC r')
    axs[0, 0].text(-0.62, 1.12, 'a', transform=axs[0, 0].transAxes, fontweight='bold', fontsize=8)
    axs[1, 0].text(-0.62, 1.12, 'b', transform=axs[1, 0].transAxes, fontweight='bold', fontsize=8)
    save(fig, 'fig6_benchmark')
    d_b, d_m = summary[('B-frame', 'Drifted pairs')], summary[('Midpoint', 'Drifted pairs')]
    best_b = max(methods, key=lambda m: d_b[m] if np.isfinite(d_b[m]) else -1)
    best_m = max(methods, key=lambda m: d_m[m] if np.isfinite(d_m[m]) else -1)
    cap['fig6'] = (
        f'**Fair benchmark.** Every method reduced to the correspondence map it estimates (RT in B → RT in A) and '
        f'scored the same way: (a) A warped into B\'s frame; (b) both batches resampled into the frame halfway '
        f'between them. Mean TIC r per group of pairs. Drifted pairs: best is {best_b} '
        f'({d_b[best_b]:.3f}) in B\'s frame and {best_m} ({d_m[best_m]:.3f}) at the midpoint. The WarpTransformer '
        f'is warpalign\'s v2 model with its own anchor calibration (seed 0). COW-TIC fits a separate warp per '
        f'sample pair to maximise TIC r. Blue: this work.')


# ── Supplementary ────────────────────────────────────────────────────────────

def figS1(run: an.Run, cap: dict) -> None:
    sw = an.lambda_sweep(run, gauges=[run.ref, 'mean'])
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.3), gridspec_kw={'wspace': 0.3})
    gauges = ((run.ref, S[0], f'reference gauge ({lab(run.names[run.ref])})'), ('mean', S[1], 'mean gauge'))
    for g, col, label in gauges:
        rows = [s for s in sw if s['gauge'] == g]
        axs[0].plot([s['lam'] for s in rows], [s['cv_min'] for s in rows], 'o-', ms=3, color=col, label=label)
    axs[0].axvline(run.lam, color=MUTED, ls=':', lw=0.8)
    axs[0].set_xscale('log'); axs[0].set_xlabel('λ'); axs[0].set_ylabel('median held-out anchor error (min)')
    axs[0].legend(); panel(axs[0], 'a', 'RT-blocked cross-validation')
    rows = [s for s in sw if s['gauge'] == run.ref]
    lams = [s['lam'] for s in rows]
    axs[1].plot(lams, [s['ticr_drifted'] for s in rows], 'o-', ms=3, color=S[0], label='drifted pairs')
    axs[1].plot(lams, [s['ticr_near'] for s in rows], 'o-', ms=3, color=S[2], label='near-aligned pairs')
    axs[1].axvline(run.lam, color=MUTED, ls=':', lw=0.8)
    axs[1].set_xscale('log'); axs[1].set_xlabel('λ'); axs[1].set_ylabel('mean TIC r (not used for selection)')
    axs[1].legend(); panel(axs[1], 'b', 'Sensitivity (post hoc)')
    save(fig, 'figS1_lambda')
    ref_sel = next(s for s in sw if s['gauge'] == run.ref and s['lam'] == run.lam)
    mean_sel = next(s for s in sw if s['gauge'] == 'mean' and s['lam'] == run.lam)
    dr = [s['ticr_drifted'] for s in rows if s['lam'] >= 1]
    cap['figS1'] = (
        f'**Choosing λ.** (a) Median held-out anchor error under RT-blocked cross-validation (whole 12-bin RT '
        f'windows held out) for both gauges; dotted: selected λ = {run.lam:g} (reference gauge, '
        f'{ref_sel["cv_min"]:.3f} min; the mean gauge reaches {mean_sel["cv_min"]:.3f} min there). (b) Mean TIC r '
        f'against λ, computed after the fact; for λ ≥ 1 the drifted-pair mean stays within '
        f'{min(dr):.3f}–{max(dr):.3f}.')


def figS2(run: an.Run, cap: dict) -> None:
    cov = an.coverage(run)
    boots = an.bootstrap_models(run)
    widths, supports = [], []
    inner = (T_MIN >= 2) & (T_MIN <= 43)
    for a, b in combinations(range(run.K), 2):
        q = an.band(run, a, b, boots)
        w = (q[1] - q[0]) * BIN_MIN
        p = run.anc.pair(a, b)
        sup = np.array([np.sum(np.abs(p.ta - t) <= 2.5) for t in GRID])
        ok = inner & np.isfinite(w)
        widths.append(w[ok]); supports.append(sup[ok])
    w, sup = np.concatenate(widths), np.concatenate(supports)
    rho = spearmanr(w, sup).statistic
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.2), gridspec_kw={'width_ratios': [0.8, 1.2], 'wspace': 0.35})
    ax = axs[0]
    rows = [('warp band only', cov['epistemic'], AXIS), ('band + anchor scatter', cov['predictive'], S[0])]
    ax.barh([0, 1], [r_[1] for r_ in rows], color=[r_[2] for r_ in rows], height=0.55)
    for i, (_, v, _) in enumerate(rows):
        ax.text(v + 0.02, i, f'{v:.0%}', va='center', fontsize=6.5)
    ax.axvline(0.9, color=INK, ls='--', lw=0.8); ax.text(0.9, 1.55, 'target 90 %', ha='center', fontsize=6.3)
    ax.set_yticks([0, 1]); ax.set_yticklabels([r_[0] for r_ in rows]); ax.set_xlim(0, 1.05); ax.set_ylim(-0.6, 1.8)
    ax.grid(axis='x'); ax.grid(axis='y', visible=False); ax.set_xlabel('held-out anchors inside the 90 % interval')
    panel(ax, 'a', 'Coverage')
    ax = axs[1]
    ax.scatter(sup + np.random.default_rng(0).uniform(-0.4, 0.4, len(sup)), w, s=1.5, color=S[0], alpha=0.15, lw=0,
               rasterized=True)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlabel('direct anchors within ±2.5 bins')
    ax.set_ylabel('90 % band width (min)')
    panel(ax, 'b', f'Band width vs anchor support (Spearman ρ = {rho:.2f})')
    save(fig, 'figS2_uncertainty')
    cap['figS2'] = (
        f'**Uncertainty.** (a) Share of held-out anchors inside their 90 % interval (cross-fitted): the bootstrap '
        f'band of the warp alone covers {cov["epistemic"]:.0%}, the predictive interval (band combined with the '
        f'pair\'s robust anchor scatter) {cov["predictive"]:.0%}, against the 90 % target. (b) Width of the 90 % '
        f'bootstrap band (200 refits, samples resampled within batches) against the number of direct anchors '
        f'nearby, every batch pair and RT bin between 2 and 43 min where the warp is not clamped at a run end '
        f'(the same rule as script 30); Spearman ρ = {rho:.2f}.')


def figS3(run: an.Run, cap: dict) -> None:
    t = run.anc.ta * BIN_MIN
    edges = np.arange(0, 50, 5.0)
    cnt, rej = [], []
    for lo, hi in pairwise(edges):
        m = (t >= lo) & (t < hi)
        cnt.append(int(m.sum())); rej.append(float(np.mean(run.joint.weights[m] == 0)) if m.any() else np.nan)
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    ax.bar(edges[:-1] + 2.5, rej, width=4.4, color=S[0])
    for x, r_, n in zip(edges[:-1] + 2.5, rej, cnt):
        if np.isfinite(r_):
            ax.text(x, r_ + 0.003, f'{n:,}', ha='center', fontsize=5.5, color=INK2)
    ax.set_xlabel('retention time in batch A (min)'); ax.set_ylabel('anchors rejected (weight 0)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    panel(ax, '', 'Robust rejection along the run')
    save(fig, 'figS3_rejection')
    ok = [i for i, r_ in enumerate(rej) if np.isfinite(r_)]
    top = sorted(ok, key=lambda i: -rej[i])[:3]
    rest = [rej[i] for i in ok if i not in top]
    win = lambda i: f'{edges[i]:.0f}–{edges[i + 1]:.0f} min ({rej[i]:.1%})'
    cap['figS3'] = (
        f'**Where anchors are rejected.** Share of anchors with weight 0 after Huber and Tukey reweighting, by '
        f'5-min window of retention time (numbers: anchors per window); {np.mean(run.joint.weights == 0):.1%} '
        f'overall. Highest at {", ".join(win(i) for i in sorted(top))}; {min(rest):.1%}–{max(rest):.1%} in the '
        f'other windows.')


def main(only: list[str] | None) -> None:
    run = an.load_run(RESULTS, DATA)
    cap: dict[str, str] = {}
    figs = {'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5, 'fig6': fig6,
            'figS1': figS1, 'figS2': figS2, 'figS3': figS3}
    print(f'Figures → {OUT}')
    for k, f in figs.items():
        if not only or k in only:
            f(run, cap)
    if not only:
        z = run.z
        rev = ' · '.join(map(str, z['provenance'][:-1])) if 'provenance' in z else str(z.get('warpalign', ''))
        lines = ['# Figure captions', '', f'Generated by scripts/50_paper_figures.py from results/ '
                 f'({rev.split(" from ")[0]}). Numbers in the captions are computed, not typed.', '']
        for k, c in cap.items():
            lines += [f'## {k}', '', c, '']
        (OUT / 'captions.md').write_text('\n'.join(lines))
        print(f'  captions.md ({len(cap)} captions)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=RESULTS)
    ap.add_argument('--data', type=Path, default=None, help="batch_* folders (default: script 30's default)")
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--only', nargs='*', help='e.g. fig2 figS1 (captions.md is written only for a full run)')
    a = ap.parse_args()
    RESULTS, OUT, DATA = a.results, a.out, a.data
    main(a.only)
