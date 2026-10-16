"""
Joint retention-time alignment of K batches into one shared latent RT frame.

Every batch k gets one monotone warp f_k from its own RT axis to a shared latent axis,
f_k(t) = t + d_k(t), with the drift d_k piecewise linear on a knot grid. Spectral anchor
pairs (t_a, t_b) harvested from every cross-batch sample pair constrain
f_a(t_a) ≈ f_b(t_b). The fit is a bounded least-squares problem:

  - anchor rows      f_a(t_a) − f_b(t_b) = 0, robustly reweighted (Huber, then Tukey)
  - smoothness       λ · second differences of d_k
  - slope bounds     S_MIN ≤ f_k'(t) ≤ S_MAX, so every f_k is strictly monotone
  - gauge            Σ_k d_k(t) = 0 at every knot (identity on average), or d_ref = 0

The warp from batch k to batch l is f_l⁻¹ ∘ f_k. It is transitive by construction, so a
batch pair with few or no direct anchors borrows strength from paths through other batches.

Times are RT bins (0 … N_BINS−1) throughout; multiply by BIN_MIN for minutes. The data model,
anchors and time axis come from alignbench (github.com/woodRock/align-bench).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from alignbench.anchors import Anchors, _parabolic  # noqa: F401  (_parabolic: re-export for tests)
from alignbench.anchors import harvest_arrays as harvest_anchors
from alignbench.anchors import match_subbin as match_anchors
from alignbench.axis import FISH_OIL_AXIS
from alignbench.maps import pchip_pairwise as _pchip_pairwise
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import lsq_linear

N_BINS         = FISH_OIL_AXIS.n_bins
N_KNOTS        = 41      # knot spacing ≈ 5 bins ≈ 1.1 min
S_MIN, S_MAX   = 0.5, 2.0
GAUGE_WEIGHT   = 1e3
SOLVERS        = ('bvls', 'active_set')

# Anchors (columnar, every cross-batch sample pair), match_anchors (sub-bin anchors for one sample
# pair) and harvest_anchors moved to alignbench in joint-align 2.0 (alignbench.anchors: Anchors,
# match_subbin, harvest_arrays); the names here are re-exports, bit-identical to joint-align 1.1.0.
__all__ = [
    'GAUGE_WEIGHT',
    'N_BINS',
    'N_KNOTS',
    'SOLVERS',
    'S_MAX',
    'S_MIN',
    'Anchors',
    'JointWarp',
    'fit_joint',
    'fit_pairwise',
    'harvest_anchors',
    'match_anchors',
    'pchip_pairwise',
]


# ── Joint warp model ──────────────────────────────────────────────────────────

@dataclass
class JointWarp:
    knots: np.ndarray        # (M,) RT bins
    drift: np.ndarray        # (K, M) d_k at the knots, bins
    scale: np.ndarray        # (K, K) robust residual scale per batch pair, bins (nan if no anchors)
    weights: np.ndarray      # final robust weight per anchor

    _grid = np.linspace(0, N_BINS - 1, 4 * N_BINS)

    def latent(self, k: int, t: np.ndarray) -> np.ndarray:
        return t + np.interp(t, self.knots, self.drift[k])

    def inverse(self, k: int, tau: np.ndarray) -> np.ndarray:
        return np.interp(tau, self.latent(k, self._grid), self._grid)

    def map(self, k: int, l: int, t: np.ndarray) -> np.ndarray:
        """RT in batch k → RT in batch l."""
        return self.inverse(l, self.latent(k, t))


def _hat_basis(t: np.ndarray, knots: np.ndarray) -> np.ndarray:
    idx  = np.clip(np.searchsorted(knots, t, side='right') - 1, 0, len(knots) - 2)
    frac = np.clip((t - knots[idx]) / (knots[idx + 1] - knots[idx]), 0.0, 1.0)
    B = np.zeros((len(t), len(knots)))
    B[np.arange(len(t)), idx]     = 1.0 - frac
    B[np.arange(len(t)), idx + 1] = frac
    return B


def _robust_sigma(r: np.ndarray) -> float:
    return max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 0.25) if len(r) else np.nan


def _box_qp_active_set(G: np.ndarray, h: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                      max_iter: int = 100, tol: float = 1e-9) -> tuple[np.ndarray, bool]:
    """
    min ½ xᵀGx − hᵀx subject to lb ≤ x ≤ ub, for symmetric positive definite G, by a
    primal-dual active-set method (Hintermüller, Ito & Kunisch 2003) started from the
    unconstrained minimiser. Each step fixes the variables predicted to sit at a bound and solves
    for the rest; it stops when the predicted sets repeat. Returns (x, ok): ok only if x satisfies
    the KKT conditions to tol (feasible; zero gradient on free variables; gradient pointing out of
    the box on bound ones), so the caller can fall back to an exact solver otherwise.
    """
    c = max(float(np.abs(np.diag(G)).max()), 1.0)
    x = cho_solve(cho_factor(G), h)
    g = np.zeros_like(h)
    at_lb = at_ub = np.zeros(len(h), bool)
    for it in range(max_iter + 1):
        new_lb = np.isfinite(lb) & ((lb - x) * c + g > 0)
        new_ub = np.isfinite(ub) & ((x - ub) * c - g > 0) & ~new_lb
        if it > 0 and np.array_equal(new_lb, at_lb) and np.array_equal(new_ub, at_ub):
            break
        if it == max_iter:
            return x, False
        at_lb, at_ub = new_lb, new_ub
        free = ~(at_lb | at_ub)
        x = np.where(at_lb, lb, np.where(at_ub, ub, 0.0))
        if free.any():
            rhs = h[free] - G[np.ix_(free, ~free)] @ x[~free]
            x[free] = cho_solve(cho_factor(G[np.ix_(free, free)]), rhs)
        g = G @ x - h
    gtol = tol * (float(np.abs(h).max()) + 1.0)
    free = ~(at_lb | at_ub)
    ok = bool(np.all(x >= lb - tol) and np.all(x <= ub + tol) and np.all(np.abs(g[free]) <= gtol)
              and np.all(g[at_lb] >= -gtol) and np.all(g[at_ub] <= gtol))
    return x, ok


def fit_joint(anchors: Anchors, n_batches: int, lam: float = 1.0,
              base_weight: np.ndarray | None = None, gauge: str | int = 'mean',
              n_huber: int = 3, n_tukey: int = 4, n_knots: int = N_KNOTS,
              solver: str = 'bvls') -> JointWarp:
    """
    Fit one monotone warp per batch to all anchors at once.

    lam          smoothness weight, relative to the anchor count (scale-free)
    base_weight  per-anchor multiplicity (bootstrap counts); default 1
    gauge        'mean' (Σ_k d_k = 0) or an int batch index whose drift is fixed at 0
    solver       the bounded least-squares solver: 'bvls' (scipy's lsq_linear, the default and
                 paper one's) or 'active_set' (_box_qp_active_set, falling back to BVLS whenever
                 its KKT check fails). Both solve the same convex problem; active_set is much faster
                 for many batches (K ≥ 10), where BVLS can take seconds per solve.
    """
    if solver not in SOLVERS:
        raise ValueError(f'solver must be one of {SOLVERS}, not {solver!r}')
    K, N = n_batches, len(anchors)
    knots = np.linspace(0, N_BINS - 1, n_knots)
    M     = n_knots
    dk    = np.diff(knots)
    Lcum  = np.tril(np.ones((M, M)))          # d = Lcum @ θ, θ = (c, δ_1 … δ_{M−1})
    w0    = np.ones(N) if base_weight is None else np.asarray(base_weight, dtype=float)

    # Anchor design: (B(ta) L) in block ba, −(B(tb) L) in block bb; target tb − ta.
    Ba, Bb = _hat_basis(anchors.ta, knots) @ Lcum, _hat_basis(anchors.tb, knots) @ Lcum
    X = np.zeros((N, K * M))
    for k in range(K):
        X[anchors.ba == k, k * M:(k + 1) * M] += Ba[anchors.ba == k]
        X[anchors.bb == k, k * M:(k + 1) * M] -= Bb[anchors.bb == k]
    y = anchors.tb - anchors.ta

    n_eff = max(float(w0.sum()), 1.0)
    lam_eff = lam * n_eff / (K * M)
    D2  = np.diff(np.eye(M), n=2, axis=0) @ Lcum
    P   = np.zeros((K * M, K * M))
    for k in range(K):
        sl = slice(k * M, (k + 1) * M)
        P[sl, sl] += lam_eff * D2.T @ D2
    G_rows = np.zeros((M, K * M))
    if gauge == 'mean':
        for k in range(K):
            G_rows[:, k * M:(k + 1) * M] = Lcum
    else:
        G_rows[:, int(gauge) * M:(int(gauge) + 1) * M] = Lcum
    P += GAUGE_WEIGHT * n_eff / M * G_rows.T @ G_rows
    P += 1e-8 * n_eff * np.eye(K * M)

    lb = np.tile(np.concatenate([[-np.inf], -(1 - S_MIN) * dk]), K)
    ub = np.tile(np.concatenate([[np.inf], (S_MAX - 1) * dk]), K)

    def solve(w: np.ndarray) -> np.ndarray:
        G = X.T @ (w[:, None] * X) + P
        h = X.T @ (w * y)
        if solver == 'active_set':
            x, ok = _box_qp_active_set(G, h, lb, ub)
            if ok:
                return x
        C = np.linalg.cholesky(G)
        z = solve_triangular(C, h, lower=True)
        return lsq_linear(C.T, z, bounds=(lb, ub), method='bvls').x

    w_rob = np.ones(N)
    for it in range(n_huber + n_tukey):
        theta = solve(w0 * w_rob)
        r = X @ theta - y
        sig = _robust_sigma(r[w0 > 0])
        a = np.abs(r)
        if it < n_huber:
            c = 1.345 * sig
            w_rob = np.where(a <= c, 1.0, c / np.maximum(a, 1e-12))
        else:
            c = 4.685 * sig
            w_rob = np.where(a < c, (1 - (a / c) ** 2) ** 2, 0.0)
    theta = solve(w0 * w_rob)
    r = X @ theta - y

    scale = np.full((K, K), np.nan)
    for a_, b_ in combinations(range(K), 2):
        m = (anchors.ba == a_) & (anchors.bb == b_) & (w0 > 0) & (w_rob > 0)
        scale[a_, b_] = scale[b_, a_] = _robust_sigma(r[m]) if m.any() else np.nan
    drift = np.stack([Lcum @ theta[k * M:(k + 1) * M] for k in range(K)])
    return JointWarp(knots, drift, scale, w0 * w_rob)


def fit_pairwise(anchors: Anchors, a: int, b: int, n_batches: int, **kw) -> JointWarp:
    """The same fitter restricted to one batch pair's anchors (other batches stay at identity)."""
    lo, hi = min(a, b), max(a, b)
    pa  = anchors.pair(lo, hi)
    sub = Anchors(np.zeros(len(pa), int), np.ones(len(pa), int), pa.sa, pa.sb, pa.ta, pa.tb)
    m2  = fit_joint(sub, 2, **kw)
    drift = np.zeros((n_batches, len(m2.knots)))
    drift[lo], drift[hi] = m2.drift[0], m2.drift[1]
    scale = np.full((n_batches, n_batches), np.nan)
    scale[lo, hi] = scale[hi, lo] = m2.scale[0, 1]
    return JointWarp(m2.knots, drift, scale, m2.weights)


def pchip_pairwise(anchors: Anchors, a: int, b: int):
    """
    Pairwise baseline with the old interpolation: median-aggregate the pooled anchors per
    integer bin of batch a, keep a monotone subsequence, PCHIP with end points pinned.
    Returns t_a → t_b. (alignbench.maps.pchip_pairwise since joint-align 2.0.)
    """
    return _pchip_pairwise(anchors, a, b)
