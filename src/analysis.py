"""
Derived analyses of a script-30 run, shared by the project site (script 40) and the paper
figures (script 50), so the two can never disagree: loading and checking a run, pairwise and
leave-one-pair-out refits, cycle consistency, the λ sweep, bootstrap bands, and the tables parsed
from results/multibatch_align.txt.
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations, permutations
from pathlib import Path
from types import ModuleType

import numpy as np
from gcmsalign.config import BIN_MIN, N_BINS

import multibatch as mb

ROOT = Path(__file__).resolve().parent.parent
GRID = np.arange(N_BINS, dtype=float)


def load_script(filename: str, name: str | None = None) -> ModuleType:
    """Import a numbered script from scripts/ as a module."""
    spec = importlib.util.spec_from_file_location(name or Path(filename).stem, ROOT / 'scripts' / filename)
    assert spec is not None and spec.loader is not None, filename
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def short(b: str) -> str:
    return b.replace('batch_', '')


def shift(m: np.ndarray) -> np.ndarray:
    """Shift m(t) − t of a map evaluated on GRID, NaN where the map is clamped to the run ends
    (JointWarp.map saturates at bin 0 / N_BINS − 1 when the target falls outside batch B's run;
    drawing that as a shift would show a spurious cliff)."""
    out = m - GRID
    out[(m <= 1e-6) | (m >= N_BINS - 1 - 1e-6)] = np.nan
    return out


@dataclass
class Run:
    """One script-30 run: its results, the data it used and the refitted joint model."""
    results: Path
    s30: ModuleType                   # the script-30 module (helpers and constants)
    z: dict
    names: list[str]
    batches: list
    anc: mb.Anchors
    joint: mb.JointWarp
    lam: float
    ref: int
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def K(self) -> int:
        return len(self.names)

    @property
    def tics(self) -> list[np.ndarray]:
        return [b.tic for b in self.batches]

    @property
    def idx(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return self.s30.PAIRS

    @cached_property
    def drifted(self) -> list[tuple[str, str]]:
        return [p for p in self.pairs if self.ticr(p, 'Unaligned').mean() < 0.5]

    def ticr(self, pair: tuple[str, str], method: str) -> np.ndarray:
        """Script 30's (n_a, n_b) TIC r array for a pair and method."""
        return self.z[f'ticr_{short(pair[0])}_{short(pair[1])}_{method.replace(" ", "_")}']

    def tic_r_of_map(self, a: int, b: int, model) -> float:
        """Mean TIC r of batch a warped into b's frame by model.map(b, a, ·)."""
        s30 = self.s30
        return float(s30.tic_r(s30.warp_tics(self.tics[a], model.map(b, a, GRID)), self.tics[b]).mean())

    @cached_property
    def txt(self) -> str:
        return (self.results / 'multibatch_align.txt').read_text()


def load_run(results: Path, data: Path | None = None) -> Run:
    s30 = load_script('30_multibatch_align.py', 's30_analysis')
    if data is not None:
        setattr(s30, 'DATA', data)          # noqa: B010  (module attribute, for mypy)
    z = np.load(results / 'multibatch_align.npz')
    names = [str(n) for n in z['names']]
    za = np.load(results / 'anchors_all_pairs.npz')
    anc = mb.Anchors(*(za[k] for k in ('ba', 'bb', 'sa', 'sb', 'ta', 'tb')))
    lam, ref = float(z['lam']), int(z['ref'])
    joint = mb.fit_joint(anc, len(names), lam=lam, gauge=ref)
    if not np.allclose(joint.drift, z['drift'], atol=1e-6):
        raise SystemExit(f'{results}/multibatch_align.npz does not match the anchor cache; rerun script 30')
    batches = [s30.load_batch(n, s30.DATA) for n in names]
    return Run(results, s30, dict(z), names, batches, anc, joint, lam, ref)


def pairwise_fits(run: Run) -> dict[tuple[int, int], mb.JointWarp]:
    """The same fitter on one pair's anchors, for every unordered batch pair (a < b)."""
    if 'pairwise' not in run._cache:
        run._cache['pairwise'] = {(a, b): mb.fit_pairwise(run.anc, a, b, run.K, lam=run.lam, gauge=0)
                                  for a, b in combinations(range(run.K), 2)}
    return run._cache['pairwise']


def pw_map(run: Run, a: int, b: int, t: np.ndarray) -> np.ndarray:
    return pairwise_fits(run)[(min(a, b), max(a, b))].map(a, b, t)


def cycle_table(run: Run, lo_min: float = 5.0, hi_min: float = 40.0) -> list[dict]:
    """Mean |g_bc(g_ab(t)) − g_ac(t)| (minutes) over t in [lo, hi] for every ordered triplet."""
    tt = GRID[(GRID * BIN_MIN >= lo_min) & (GRID * BIN_MIN <= hi_min)]
    j = run.joint
    out = []
    for a, b, c in permutations(range(run.K), 3):
        via_pw = pw_map(run, b, c, pw_map(run, a, b, tt)) - pw_map(run, a, c, tt)
        via_joint = j.map(b, c, j.map(a, b, tt)) - j.map(a, c, tt)
        out.append({'t': (a, b, c), 'pairwise': float(np.mean(np.abs(via_pw))) * BIN_MIN,
                    'joint': float(np.mean(np.abs(via_joint))) * BIN_MIN})
    return out


def lopo(run: Run) -> dict[tuple[int, int], dict]:
    """Joint fit without every anchor of the pair, for each evaluated pair (A, B)."""
    if 'lopo' in run._cache:
        return run._cache['lopo']
    anc, out = run.anc, {}
    for A_, B_ in run.pairs:
        a, b = run.idx[A_], run.idx[B_]
        keep = ~(((anc.ba == a) & (anc.bb == b)) | ((anc.ba == b) & (anc.bb == a)))
        m = mb.fit_joint(anc.subset(keep), run.K, lam=run.lam, gauge=run.ref)
        own = anc.pair(a, b)
        out[(a, b)] = {'model': m, 'ticr': run.tic_r_of_map(a, b, m),
                       'ticr_joint': float(run.ticr((A_, B_), 'Joint').mean()),
                       'err_min': float(np.median(np.abs(m.map(a, b, own.ta) - own.tb))) * BIN_MIN,
                       'err_joint_min': float(np.median(np.abs(run.joint.map(a, b, own.ta) - own.tb))) * BIN_MIN}
    run._cache['lopo'] = out
    return out


def star(run: Run) -> mb.JointWarp:
    """Joint fit on the anchors against the reference batch only."""
    keep = (run.anc.ba == run.ref) | (run.anc.bb == run.ref)
    return mb.fit_joint(run.anc.subset(keep), run.K, lam=run.lam, gauge=run.ref)


def lambda_sweep(run: Run, gauges=None) -> list[dict]:
    """For every λ of script 30 (and gauge): RT-blocked CV error, drift curves, post-hoc TIC r."""
    s30, anc, K = run.s30, run.anc, run.K
    gauges = [run.ref] if gauges is None else gauges
    blk = lambda t: (np.floor(t / s30.RT_BLOCK).astype(int)) % s30.N_FOLDS
    folds = [((blk(anc.ta) != f) & (blk(anc.tb) != f), (blk(anc.ta) == f) | (blk(anc.tb) == f))
             for f in range(s30.N_FOLDS)]
    out = []
    for gauge in gauges:
        for lm in s30.LAMBDAS:
            errs = np.concatenate([s30.anchor_errors(mb.fit_joint(anc.subset(tr), K, lam=lm, gauge=gauge),
                                                     anc.subset(te)) for tr, te in folds])
            m = mb.fit_joint(anc, K, lam=lm, gauge=gauge)
            v = {p: run.tic_r_of_map(run.idx[p[0]], run.idx[p[1]], m) for p in run.pairs}
            out.append({'lam': lm, 'gauge': gauge, 'cv_min': float(np.median(np.abs(errs))) * BIN_MIN,
                        'model': m, 'drift': np.stack([m.latent(k, GRID) - GRID for k in range(K)]),
                        'ticr_drifted': float(np.mean([v[p] for p in run.drifted])),
                        'ticr_near': float(np.mean([v[p] for p in run.pairs if p not in run.drifted]))})
    return out


def bootstrap_models(run: Run) -> list[mb.JointWarp]:
    j = run.joint
    return [mb.JointWarp(j.knots, d, j.scale, j.weights) for d in run.z['boot_drift']]


def band(run: Run, a: int, b: int, boots=None) -> np.ndarray:
    """(2, T) 5th/95th percentile of the bootstrap maps a → b, as shifts (bins), NaN where clamped."""
    boots = bootstrap_models(run) if boots is None else boots
    q = np.percentile(np.stack([m.map(a, b, GRID) for m in boots]), [5, 95], axis=0) - GRID
    q[:, np.isnan(shift(run.joint.map(a, b, GRID)))] = np.nan
    return q


def coverage(run: Run) -> dict:
    cov = re.search(r'predictive \(band ⊕ anchor scatter\)\s*:\s*([\d.]+)', run.txt)
    epi = re.search(r'warp-only band \(epistemic\)\s*:\s*([\d.]+)', run.txt)
    return {'predictive': float(cov.group(1)) if cov else None,
            'epistemic': float(epi.group(1)) if epi else None, 'target': 0.9}


def heldout_errors(run: Run) -> dict[str, dict[str, float]]:
    """Section 3 of the report: cross-fitted median |held-out anchor error| (min) per pair and method."""
    sec = run.txt[run.txt.index('3. Cross-fitted'):run.txt.index('4. Missing')]
    methods = ['Unaligned', 'Dense PCHIP', 'Pairwise smooth', 'Joint']
    out = {}
    for line in sec.splitlines():
        m = re.match(r'\s+(\S+)\s+(\d+)((?:\s+[\d.]+ \([\d.]+\)){4})\s*$', line)
        if m:
            vals = [float(v) for v in re.findall(r'([\d.]+) \(', m.group(3))]
            out[m.group(1)] = dict(zip(methods, vals))
    return out
