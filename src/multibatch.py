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

Times are RT bins (0 … N_BINS−1) throughout; multiply by BIN_MIN for minutes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from gcmsalign.anchors import detect_peaks, fingerprint
from gcmsalign.config import (BIN_MIN, MAX_DRIFT_MIN, N_BINS, RANSAC_RESID, REFINE_RADIUS,
                              SIM_THRESHOLD)
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_triangular
from scipy.optimize import linear_sum_assignment, lsq_linear
from sklearn.linear_model import RANSACRegressor

MAX_DRIFT_BINS = MAX_DRIFT_MIN / BIN_MIN
N_KNOTS        = 41      # knot spacing ≈ 5 bins ≈ 1.1 min
S_MIN, S_MAX   = 0.5, 2.0
GAUGE_WEIGHT   = 1e3


# ── Anchors ───────────────────────────────────────────────────────────────────

def _parabolic(ym: float, y0: float, yp: float) -> float:
    """Sub-bin offset of the vertex of the parabola through three equally spaced points."""
    den = ym - 2.0 * y0 + yp
    if den >= 0:
        return 0.0
    return float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5))


def _fingerprints(chroma: np.ndarray, pks: np.ndarray) -> np.ndarray:
    return np.array([fingerprint(chroma, int(p)) for p in pks])


def match_anchors(a: np.ndarray, b: np.ndarray, tic_a: np.ndarray | None = None,
                  tic_b: np.ndarray | None = None) -> np.ndarray:
    """
    Spectral anchors between chromatograms a and b, as an (n, 2) array of sub-bin RT
    positions (t_a, t_b). Matching follows gcmsalign.anchors.match_anchors: peaks on the raw
    TIC (tic_a, tic_b; the chromatogram sum if not given), context fingerprints, Hungarian
    assignment under MAX_DRIFT_MIN, adaptive cosine threshold, ±REFINE_RADIUS search in b,
    RANSAC (RANSAC_RESID bins), monotone filter. Adds parabolic sub-bin refinement on both
    sides, so anchors are fractional bins rather than gcmsalign's integer bins.
    """
    empty = np.empty((0, 2))
    tic_a = a.sum(axis=1) if tic_a is None else np.asarray(tic_a, dtype=float)
    tic_b = b.sum(axis=1) if tic_b is None else np.asarray(tic_b, dtype=float)
    pa, pb = detect_peaks(tic_a), detect_peaks(tic_b)
    if len(pa) < 2 or len(pb) < 2:
        return empty
    fa, fb = _fingerprints(a, pa), _fingerprints(b, pb)
    sim  = fa @ fb.T
    far  = np.abs(pa[:, None] - pb[None, :]) > MAX_DRIFT_BINS
    ri, ci = linear_sum_assignment(-np.where(far, -1.0, sim))
    avail = sim[~far]
    thr   = max(float(np.percentile(avail, 70)), SIM_THRESHOLD) if avail.size >= 3 else SIM_THRESHOLD
    keep  = (sim[ri, ci] >= thr) & ~far[ri, ci]
    ri, ci = ri[keep], ci[keep]
    if len(ri) < 2:
        return empty

    out = []
    for i, j in zip(ri, ci):
        p = int(pa[i])
        ta = p + (_parabolic(tic_a[p - 1], tic_a[p], tic_a[p + 1]) if 0 < p < N_BINS - 1 else 0.0)
        cands = np.arange(max(0, pb[j] - REFINE_RADIUS), min(N_BINS, pb[j] + REFINE_RADIUS + 1))
        s = np.array([float(fa[i] @ fingerprint(b, int(c))) for c in cands])
        k = int(np.argmax(s))
        tb = cands[k] + (_parabolic(s[k - 1], s[k], s[k + 1]) if 0 < k < len(s) - 1 else 0.0)
        out.append((ta, tb))
    anc = np.array(out, dtype=float)

    if len(anc) >= 4:
        try:
            rs = RANSACRegressor(residual_threshold=RANSAC_RESID, min_samples=2, random_state=0)
            rs.fit(anc[:, :1], anc[:, 1])
            if rs.inlier_mask_.sum() >= 2:
                anc = anc[rs.inlier_mask_]
        except ValueError:
            pass

    anc  = anc[np.argsort(anc[:, 0])]
    mono = [0]
    for i in range(1, len(anc)):
        if anc[i, 0] > anc[mono[-1], 0] and anc[i, 1] > anc[mono[-1], 1]:
            mono.append(i)
    return anc[mono] if len(mono) >= 2 else empty


@dataclass
class Anchors:
    """Anchor pairs pooled over every cross-batch sample pair (one row per anchor)."""
    ba: np.ndarray   # batch index of the first chromatogram
    bb: np.ndarray   # batch index of the second chromatogram (ba < bb)
    sa: np.ndarray   # sample index within batch ba
    sb: np.ndarray   # sample index within batch bb
    ta: np.ndarray   # RT bin in batch ba
    tb: np.ndarray   # RT bin in batch bb

    def __len__(self) -> int:
        return len(self.ta)

    def subset(self, mask: np.ndarray) -> 'Anchors':
        return Anchors(*(getattr(self, f)[mask] for f in ('ba', 'bb', 'sa', 'sb', 'ta', 'tb')))

    def pair(self, a: int, b: int) -> 'Anchors':
        """Anchors between batches a and b, oriented so that ta is in a and tb is in b."""
        if a < b:
            return self.subset((self.ba == a) & (self.bb == b))
        s = self.subset((self.ba == b) & (self.bb == a))
        return Anchors(s.bb, s.ba, s.sb, s.sa, s.tb, s.ta)


def harvest_anchors(chromas, tics=None) -> Anchors:
    """Run match_anchors on every cross-batch sample pair. chromas[k][i] is sample i of batch
    k; tics[k][i] its raw TIC for peak detection (default: the chromatogram sum)."""
    cols = [[] for _ in range(6)]
    tic = (lambda k, i: None) if tics is None else (lambda k, i: tics[k][i])
    for a, b in combinations(range(len(chromas)), 2):
        for i, ca in enumerate(chromas[a]):
            for j, cb in enumerate(chromas[b]):
                anc = match_anchors(ca, cb, tic(a, i), tic(b, j))
                n = len(anc)
                for col, v in zip(cols, (a, b, i, j)):
                    col.append(np.full(n, v, dtype=int))
                cols[4].append(anc[:, 0])
                cols[5].append(anc[:, 1])
    return Anchors(*(np.concatenate(c) for c in cols))


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


def fit_joint(anchors: Anchors, n_batches: int, lam: float = 1.0,
              base_weight: np.ndarray | None = None, gauge: str | int = 'mean',
              n_huber: int = 3, n_tukey: int = 4, n_knots: int = N_KNOTS) -> JointWarp:
    """
    Fit one monotone warp per batch to all anchors at once.

    lam          smoothness weight, relative to the anchor count (scale-free)
    base_weight  per-anchor multiplicity (bootstrap counts); default 1
    gauge        'mean' (Σ_k d_k = 0) or an int batch index whose drift is fixed at 0
    """
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
    Returns t_a → t_b.
    """
    pa = anchors.pair(a, b)
    if len(pa) < 2:
        return lambda t: np.asarray(t, dtype=float)
    ra = np.round(pa.ta).astype(int)
    xs, ys = [], []
    for r in np.unique(ra):
        xs.append(float(np.median(pa.ta[ra == r])))
        ys.append(float(np.median(pa.tb[ra == r])))
    mono = [0]
    for i in range(1, len(xs)):
        if ys[i] > ys[mono[-1]]:
            mono.append(i)
    xs = np.array(xs)[mono]
    ys = np.array(ys)[mono]
    keep = (xs > 0) & (xs < N_BINS - 1) & (ys > 0) & (ys < N_BINS - 1)
    f = PchipInterpolator(np.concatenate([[0], xs[keep], [N_BINS - 1]]),
                          np.concatenate([[0], ys[keep], [N_BINS - 1]]))
    return lambda t: f(np.asarray(t, dtype=float))

