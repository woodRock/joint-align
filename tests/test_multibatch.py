"""Tests for jointalign.multibatch: anchors, the joint monotone warp fit and its baselines."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from conftest import T, library_positions, make_chroma, make_library
from jointalign import multibatch as mb

GRID = np.arange(T, dtype=float)


def synthetic_anchors(drifts, n_per_pair=60, noise=0.0, seed=0, pairs=None):
    """
    Anchors generated from known batch warps f_k(t) = t + drift_k(t): a latent position τ
    is observed at f_a⁻¹(τ) in batch a and f_b⁻¹(τ) in batch b.
    """
    rng = np.random.default_rng(seed)
    K = len(drifts)
    fine = np.linspace(0, T - 1, 4000)
    cols = [[] for _ in range(6)]
    for a, b in (pairs or combinations(range(K), 2)):
        tau = rng.uniform(15, T - 16, n_per_pair)
        ta = np.interp(tau, fine + drifts[a](fine), fine)
        tb = np.interp(tau, fine + drifts[b](fine), fine)
        tb = tb + noise * rng.standard_normal(n_per_pair)
        for col, v in zip(cols, (a, b, 0, 0)):
            col.append(np.full(n_per_pair, v, dtype=int))
        cols[4].append(ta)
        cols[5].append(tb)
    return mb.Anchors(*(np.concatenate(c) for c in cols))


SHIFT_DRIFTS = [lambda t: 0.0 * t, lambda t: -4.0 + 0.0 * t, lambda t: 3.0 + 0.0 * t]
SMOOTH_DRIFTS = [lambda t: 0.0 * t,
                 lambda t: -2.0 - 3.0 * np.sin(np.pi * t / (T - 1)),
                 lambda t: 1.5 + 0.02 * t]


# ── _parabolic ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('v', [-0.4, -0.1, 0.0, 0.25, 0.45])
def test_parabolic_exact_for_parabola(v):
    y = lambda x: -(x - v) ** 2 + 3
    assert mb._parabolic(y(-1), y(0), y(1)) == pytest.approx(v)


def test_parabolic_flat_or_minimum_gives_zero():
    assert mb._parabolic(1.0, 1.0, 1.0) == 0.0
    assert mb._parabolic(2.0, 1.0, 2.0) == 0.0       # a minimum, not a peak


def test_parabolic_clipped():
    assert mb._parabolic(0.0, 1.0, 1.0 - 1e-9) == pytest.approx(0.5, abs=1e-6)
    assert abs(mb._parabolic(10.0, 10.0, -50.0)) <= 0.5


# ── match_anchors ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('shift', [-6.3, 0.0, 2.6, 9.0])
def test_match_anchors_recovers_shift(library, shift):
    p = library_positions(len(library))
    a = make_chroma(library, p, seed=1)
    b = make_chroma(library, p + shift, seed=2)
    anc = mb.match_anchors(a, b)
    assert len(anc) >= 10
    # every anchor sits on a true compound position in a
    assert np.abs(anc[:, 0][:, None] - p[None, :]).min(1).max() < 0.3
    np.testing.assert_allclose(anc[:, 1] - anc[:, 0], shift, atol=0.5)


def test_match_anchors_detects_peaks_on_raw_tic(library):
    """With a raw TIC given, peaks come from it, not from the (normalised) chromatogram."""
    p = library_positions(len(library))
    a = make_chroma(library, p, seed=1)
    b = make_chroma(library, p + 3, seed=2)
    keep = slice(0, 6)                                   # raw TIC shows only six compounds
    tic_a = make_chroma(library[keep], p[keep], noise=0.0).sum(1)
    tic_b = make_chroma(library[keep], p[keep] + 3, noise=0.0).sum(1)
    anc = mb.match_anchors(a, b, tic_a, tic_b)
    assert 2 <= len(anc) <= 6
    assert anc[:, 0].max() < p[6]
    assert len(mb.match_anchors(a, b)) > 6


def test_harvest_anchors_passes_raw_tics(warped_batches):
    chromas, _ = warped_batches
    small = [c[:1] for c in chromas]
    tics = [[c.sum(1) for c in cs] for cs in small]
    a, b = mb.harvest_anchors(small), mb.harvest_anchors(small, tics)
    np.testing.assert_allclose(a.ta, b.ta)


def test_match_anchors_strictly_monotone(library):
    p = library_positions(len(library))
    a = make_chroma(library, p, seed=1)
    b = make_chroma(library, p + 3 + 0.03 * (p - 100), seed=2)
    anc = mb.match_anchors(a, b)
    assert np.all(np.diff(anc[:, 0]) > 0) and np.all(np.diff(anc[:, 1]) > 0)


def test_match_anchors_rejects_swapped_order(library):
    """Two compounds swapping elution order cannot both survive the monotone filter."""
    p = library_positions(len(library))
    q = p.copy()
    q[5], q[6] = p[6], p[5]
    anc = mb.match_anchors(make_chroma(library, p), make_chroma(library, q, seed=3))
    assert np.all(np.diff(anc[:, 1]) > 0)
    assert len(anc) <= len(p) - 1


def test_match_anchors_respects_drift_limit(library):
    """Matches never pair peaks further apart than MAX_DRIFT_BINS."""
    p = library_positions(len(library))
    anc = mb.match_anchors(make_chroma(library, p), make_chroma(library, p[::-1].copy(), seed=4))
    if len(anc):
        assert np.all(np.abs(anc[:, 1] - anc[:, 0]) <= mb.MAX_DRIFT_BINS + mb.REFINE_RADIUS + 1)


def test_match_anchors_empty_inputs():
    flat = np.ones((T, 20), dtype=np.float32)
    out = mb.match_anchors(flat, flat)
    assert out.shape == (0, 2)
    lib = make_library(1, 20)
    one = make_chroma(lib, np.array([100.0]), noise=0.0)
    assert mb.match_anchors(one, one).shape == (0, 2)


def test_match_anchors_peak_at_run_edges():
    """Peaks on bin 0 / the last bin must not index out of range."""
    lib = make_library(4, 40, seed=3)
    p = np.array([0.0, 60.0, 120.0, T - 1.0])
    a = make_chroma(lib, p, sigma=1.0)
    anc = mb.match_anchors(a, a)
    assert np.all((anc >= 0) & (anc <= T - 1))


# ── Anchors ───────────────────────────────────────────────────────────────────

def _toy_anchors():
    return mb.Anchors(ba=np.array([0, 0, 1, 0]), bb=np.array([1, 2, 2, 1]),
                      sa=np.array([0, 1, 2, 3]), sb=np.array([4, 5, 6, 7]),
                      ta=np.array([10., 20., 30., 40.]), tb=np.array([11., 22., 33., 44.]))


def test_anchors_len_and_subset():
    a = _toy_anchors()
    assert len(a) == 4
    s = a.subset(np.array([True, False, False, True]))
    assert len(s) == 2
    np.testing.assert_array_equal(s.ta, [10., 40.])
    np.testing.assert_array_equal(s.sb, [4, 7])


def test_anchors_pair_orientation():
    a = _toy_anchors()
    fwd = a.pair(0, 1)
    np.testing.assert_array_equal(fwd.ta, [10., 40.])
    np.testing.assert_array_equal(fwd.tb, [11., 44.])
    rev = a.pair(1, 0)
    np.testing.assert_array_equal(rev.ba, [1, 1])
    np.testing.assert_array_equal(rev.bb, [0, 0])
    np.testing.assert_array_equal(rev.ta, [11., 44.])
    np.testing.assert_array_equal(rev.tb, [10., 40.])
    np.testing.assert_array_equal(rev.sa, [4, 7])
    np.testing.assert_array_equal(rev.sb, [0, 3])
    assert len(a.pair(1, 2)) == 1 and len(a.pair(0, 0)) == 0


def test_harvest_anchors_indices(warped_batches):
    chromas, shifts = warped_batches
    small = [c[:2] for c in chromas]
    anc = mb.harvest_anchors(small)
    assert len(anc) > 0
    assert np.all(anc.ba < anc.bb)
    for a, b in combinations(range(3), 2):
        p = anc.pair(a, b)
        assert set(np.unique(p.sa)) <= {0, 1} and set(np.unique(p.sb)) <= {0, 1}
        np.testing.assert_allclose(np.median(p.tb - p.ta), shifts[b] - shifts[a], atol=0.5)
    # rows of one sample pair equal a direct match_anchors call
    m = (anc.ba == 0) & (anc.bb == 2) & (anc.sa == 1) & (anc.sb == 0)
    direct = mb.match_anchors(small[0][1], small[2][0])
    np.testing.assert_allclose(np.c_[anc.ta[m], anc.tb[m]], direct)


# ── Hat basis / robust scale ──────────────────────────────────────────────────

def test_hat_basis_partition_of_unity_and_linear_reproduction():
    knots = np.linspace(0, T - 1, 11)
    t = np.random.default_rng(0).uniform(0, T - 1, 300)
    B = mb._hat_basis(t, knots)
    np.testing.assert_allclose(B.sum(1), 1.0)
    assert (B >= 0).all() and ((B > 0).sum(1) <= 2).all()
    np.testing.assert_allclose(B @ (2 * knots + 5), 2 * t + 5, atol=1e-9)


def test_hat_basis_at_knots_and_out_of_range():
    knots = np.linspace(0, T - 1, 5)
    B = mb._hat_basis(knots, knots)
    np.testing.assert_allclose(B, np.eye(5), atol=1e-12)
    B = mb._hat_basis(np.array([-10.0, T + 10.0]), knots)
    np.testing.assert_allclose(B[0], np.eye(5)[0])
    np.testing.assert_allclose(B[1], np.eye(5)[-1])


def test_robust_sigma():
    assert np.isnan(mb._robust_sigma(np.array([])))
    assert mb._robust_sigma(np.zeros(10)) == 0.25                    # floor
    r = np.random.default_rng(0).normal(0, 3.0, 20000)
    assert mb._robust_sigma(r) == pytest.approx(3.0, rel=0.03)
    r[:2000] = 1e6                                                    # 10 % gross outliers
    assert mb._robust_sigma(r) < 4.0


# ── JointWarp ─────────────────────────────────────────────────────────────────

def _manual_warp(drifts):
    knots = np.linspace(0, T - 1, mb.N_KNOTS)
    return mb.JointWarp(knots, np.stack([d(knots) for d in drifts]), np.full((3, 3), np.nan),
                        np.ones(1))


def test_jointwarp_latent_inverse_roundtrip():
    w = _manual_warp(SMOOTH_DRIFTS)
    t = np.linspace(10, T - 11, 50)
    for k in range(3):
        np.testing.assert_allclose(w.inverse(k, w.latent(k, t)), t, atol=1e-6)


def test_jointwarp_map_identity_and_transitivity():
    w = _manual_warp(SMOOTH_DRIFTS)
    t = np.linspace(20, T - 21, 40)
    np.testing.assert_allclose(w.map(1, 1, t), t, atol=1e-6)
    # the inverse is interpolated on a 4×N_BINS grid, so composition is exact to ~1e-4 bins
    np.testing.assert_allclose(w.map(1, 2, w.map(0, 1, t)), w.map(0, 2, t), atol=1e-3)
    np.testing.assert_allclose(w.map(2, 0, w.map(0, 2, t)), t, atol=1e-3)


# ── fit_joint ─────────────────────────────────────────────────────────────────

def test_fit_joint_recovers_shifts_reference_gauge():
    anc = synthetic_anchors(SHIFT_DRIFTS, noise=0.2)
    m = mb.fit_joint(anc, 3, lam=10.0, gauge=0)
    inner = (m.knots > 20) & (m.knots < T - 21)
    np.testing.assert_allclose(m.drift[0], 0.0, atol=1e-3)            # gauge
    np.testing.assert_allclose(m.drift[1][inner], -4.0, atol=0.15)
    np.testing.assert_allclose(m.drift[2][inner], 3.0, atol=0.15)
    t = np.linspace(20, T - 21, 30)
    np.testing.assert_allclose(m.map(1, 2, t), t - 7.0, atol=0.2)     # τ = t − 4 = t' + 3


def test_fit_joint_mean_gauge_sums_to_zero():
    anc = synthetic_anchors(SHIFT_DRIFTS, noise=0.2)
    m = mb.fit_joint(anc, 3, lam=10.0, gauge='mean')
    np.testing.assert_allclose(m.drift.sum(0), 0.0, atol=1e-2)
    # relative warps do not depend on the gauge
    r = mb.fit_joint(anc, 3, lam=10.0, gauge=0)
    t = np.linspace(20, T - 21, 30)
    np.testing.assert_allclose(m.map(1, 2, t), r.map(1, 2, t), atol=0.05)


def test_fit_joint_recovers_smooth_drift():
    anc = synthetic_anchors(SMOOTH_DRIFTS, n_per_pair=150, noise=0.2, seed=1)
    m = mb.fit_joint(anc, 3, lam=1.0, gauge=0)
    t = np.linspace(20, T - 21, 50)
    for a, b in combinations(range(3), 2):
        fine = np.linspace(0, T - 1, 4000)
        truth = np.interp(t + SMOOTH_DRIFTS[a](t), fine + SMOOTH_DRIFTS[b](fine), fine)
        np.testing.assert_allclose(m.map(a, b, t), truth, atol=0.3)


def test_fit_joint_borrows_strength_for_missing_pair():
    """With no 1×2 anchors, the 1→2 warp is still recovered via batch 0."""
    anc = synthetic_anchors(SHIFT_DRIFTS, noise=0.1, pairs=[(0, 1), (0, 2)])
    m = mb.fit_joint(anc, 3, lam=10.0, gauge=0)
    t = np.linspace(20, T - 21, 30)
    np.testing.assert_allclose(m.map(1, 2, t), t - 7.0, atol=0.2)
    assert np.isnan(m.scale[1, 2]) and np.isnan(m.scale[2, 1])
    assert np.isfinite(m.scale[0, 1]) and m.scale[0, 1] == m.scale[1, 0]


def test_fit_joint_slopes_within_bounds():
    """A drift steeper than the bounds allow is clipped, keeping every warp monotone."""
    steep = [lambda t: 0.0 * t, lambda t: -0.8 * t + 80, lambda t: 0.0 * t]
    anc = synthetic_anchors(steep, n_per_pair=100, seed=2)
    m = mb.fit_joint(anc, 3, lam=0.1, gauge=0)
    for k in range(3):
        f = m.knots + m.drift[k]
        slope = np.diff(f) / np.diff(m.knots)
        assert slope.min() >= mb.S_MIN - 1e-6 and slope.max() <= mb.S_MAX + 1e-6


def test_fit_joint_rejects_gross_outliers():
    anc = synthetic_anchors(SHIFT_DRIFTS, noise=0.2, seed=3)
    rng = np.random.default_rng(3)
    bad = rng.choice(len(anc), len(anc) // 10, replace=False)
    tb = anc.tb.copy()
    tb[bad] += rng.choice([-1, 1], len(bad)) * rng.uniform(25, 40, len(bad))
    corrupted = mb.Anchors(anc.ba, anc.bb, anc.sa, anc.sb, anc.ta, tb)
    m = mb.fit_joint(corrupted, 3, lam=10.0, gauge=0)
    assert np.all(m.weights[bad] == 0)
    good = np.setdiff1d(np.arange(len(anc)), bad)
    assert np.mean(m.weights[good] > 0) > 0.95
    inner = (m.knots > 20) & (m.knots < T - 21)
    np.testing.assert_allclose(m.drift[1][inner], -4.0, atol=0.2)


def test_fit_joint_zero_base_weight_equals_dropping_anchors():
    anc = synthetic_anchors(SMOOTH_DRIFTS, noise=0.3, seed=4)
    keep = np.random.default_rng(4).random(len(anc)) < 0.6
    a = mb.fit_joint(anc, 3, lam=3.0, gauge=0, base_weight=keep.astype(float))
    b = mb.fit_joint(anc.subset(keep), 3, lam=3.0, gauge=0)
    np.testing.assert_allclose(a.drift, b.drift, atol=1e-5)
    assert np.all(a.weights[~keep] == 0)


def test_fit_joint_integer_weights_equal_duplication():
    """Bootstrap counts must act like repeating anchors."""
    anc = synthetic_anchors(SMOOTH_DRIFTS, n_per_pair=40, noise=0.3, seed=5)
    w = np.random.default_rng(5).integers(0, 3, len(anc)).astype(float)
    a = mb.fit_joint(anc, 3, lam=3.0, gauge=0, base_weight=w)
    rep = np.repeat(np.arange(len(anc)), w.astype(int))
    b = mb.fit_joint(anc.subset(rep), 3, lam=3.0, gauge=0)
    np.testing.assert_allclose(a.drift, b.drift, atol=1e-4)


def test_fit_joint_large_lambda_gives_linear_drift():
    anc = synthetic_anchors(SMOOTH_DRIFTS, noise=0.2, seed=6)
    m = mb.fit_joint(anc, 3, lam=1e7, gauge=0)
    for k in range(3):
        assert np.abs(np.diff(m.drift[k], 2)).max() < 1e-3


def test_fit_joint_lambda_is_scale_free():
    """Duplicating every anchor must not change the fit (λ is relative to the anchor count)."""
    anc = synthetic_anchors(SMOOTH_DRIFTS, n_per_pair=40, noise=0.3, seed=7)
    a = mb.fit_joint(anc, 3, lam=3.0, gauge=0)
    b = mb.fit_joint(anc.subset(np.r_[np.arange(len(anc)), np.arange(len(anc))]), 3, lam=3.0, gauge=0)
    np.testing.assert_allclose(a.drift, b.drift, atol=1e-5)


def test_fit_joint_exact_on_noise_free_identity():
    anc = synthetic_anchors([lambda t: 0.0 * t] * 3, seed=8)
    m = mb.fit_joint(anc, 3, lam=1.0, gauge='mean')
    np.testing.assert_allclose(m.drift, 0.0, atol=1e-6)
    assert np.all(m.scale[~np.eye(3, dtype=bool)] == 0.25)       # floor of the robust scale


def test_fit_joint_custom_knots():
    anc = synthetic_anchors(SHIFT_DRIFTS, seed=9)
    m = mb.fit_joint(anc, 3, lam=1.0, gauge=0, n_knots=11)
    assert m.knots.shape == (11,) and m.drift.shape == (3, 11)


# ── solvers ───────────────────────────────────────────────────────────────────

def random_box_qp(n, seed, frac_inf=0.2):
    """A random positive definite QP whose unconstrained minimiser violates many bounds."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((2 * n, n))
    b = 3.0 * rng.standard_normal(2 * n)
    lb, ub = -rng.uniform(0.1, 1.0, n), rng.uniform(0.1, 1.0, n)
    lb[rng.random(n) < frac_inf] = -np.inf
    ub[rng.random(n) < frac_inf] = np.inf
    return A, b, lb, ub


@pytest.mark.parametrize('seed', range(8))
def test_active_set_matches_bvls_on_random_box_qps(seed):
    from scipy.optimize import lsq_linear
    A, b, lb, ub = random_box_qp(60, seed)
    x, ok = mb._box_qp_active_set(A.T @ A, A.T @ b, lb, ub)
    ref = lsq_linear(A, b, bounds=(lb, ub), method='bvls').x
    assert ok
    np.testing.assert_allclose(x, ref, atol=1e-8)
    assert np.sum(np.isclose(ref, lb) | np.isclose(ref, ub)) > 5      # the bounds were exercised


def test_active_set_reports_failure_when_not_converged():
    A, b, lb, ub = random_box_qp(30, 0)
    _, ok = mb._box_qp_active_set(A.T @ A, A.T @ b, lb, ub, max_iter=0)
    assert not ok


def test_default_solver_is_bvls(monkeypatch):
    """The default path is paper one's: lsq_linear with BVLS, the same output as solver='bvls'."""
    anc = synthetic_anchors(SMOOTH_DRIFTS, noise=0.2, seed=11)
    methods = []
    real = mb.lsq_linear
    monkeypatch.setattr(mb, 'lsq_linear', lambda *a, **kw: methods.append(kw['method']) or real(*a, **kw))
    a = mb.fit_joint(anc, 3, lam=3.0, gauge=0)
    assert methods and set(methods) == {'bvls'}
    b = mb.fit_joint(anc, 3, lam=3.0, gauge=0, solver='bvls')
    np.testing.assert_array_equal(a.drift, b.drift)
    np.testing.assert_array_equal(a.weights, b.weights)


@pytest.mark.parametrize('case', ['outliers', 'steep', 'mean_gauge'])
def test_active_set_fit_matches_bvls_fit(case, monkeypatch):
    anc = synthetic_anchors(SHIFT_DRIFTS, noise=0.2, seed=3)
    kw = dict(lam=10.0, gauge=0)
    if case == 'outliers':
        rng = np.random.default_rng(3)
        bad = rng.choice(len(anc), len(anc) // 5, replace=False)
        tb = anc.tb.copy()
        tb[bad] += rng.choice([-1, 1], len(bad)) * rng.uniform(25, 40, len(bad))
        anc = mb.Anchors(anc.ba, anc.bb, anc.sa, anc.sb, anc.ta, tb)
    elif case == 'steep':                       # slope bounds active over most of the run
        steep = [lambda t: 0.0 * t, lambda t: -0.8 * t + 80, lambda t: 0.0 * t]
        anc, kw = synthetic_anchors(steep, n_per_pair=100, seed=2), dict(lam=0.1, gauge=0)
    else:
        kw = dict(lam=1.0, gauge='mean')
    ref = mb.fit_joint(anc, 3, **kw)
    calls = []
    monkeypatch.setattr(mb, 'lsq_linear', lambda *a, **k: calls.append(1))     # no fallback needed
    fast = mb.fit_joint(anc, 3, solver='active_set', **kw)
    assert not calls
    # 'steep' is ill-conditioned (cond(G) ≈ 5e8; small λ, knots past the last anchor): there BVLS
    # stops ~4e-3 bins short of the optimum, which test_active_set_beats_bvls_when_ill_conditioned checks.
    np.testing.assert_allclose(fast.drift, ref.drift, atol=1e-2 if case == 'steep' else 1e-6)
    np.testing.assert_array_equal(fast.weights == 0, ref.weights == 0)
    np.testing.assert_allclose(fast.scale, ref.scale, atol=1e-6)


def test_active_set_beats_bvls_when_ill_conditioned(monkeypatch):
    """Where the two solvers differ, the active-set solution has the lower objective."""
    steep = [lambda t: 0.0 * t, lambda t: -0.8 * t + 80, lambda t: 0.0 * t]
    anc = synthetic_anchors(steep, n_per_pair=100, seed=2)
    probs = []
    real = mb.lsq_linear

    def spy(A, b, bounds, method):
        r = real(A, b, bounds=bounds, method=method)
        probs.append((A, b, bounds, r.x))
        return r
    monkeypatch.setattr(mb, 'lsq_linear', spy)
    mb.fit_joint(anc, 3, lam=0.1, gauge=0)
    assert probs
    for A, b, (lb, ub), x_bvls in probs:
        G, h = A.T @ A, A.T @ b
        x, ok = mb._box_qp_active_set(G, h, lb, ub)
        f = lambda z, G=G, h=h: 0.5 * z @ G @ z - h @ z
        assert ok and f(x) <= f(x_bvls) + 1e-12 * abs(f(x_bvls))


def test_active_set_falls_back_to_bvls(monkeypatch):
    """Whenever the KKT check fails, the fit is BVLS's, exactly."""
    anc = synthetic_anchors(SMOOTH_DRIFTS, noise=0.2, seed=12)
    ref = mb.fit_joint(anc, 3, lam=3.0, gauge=0)
    monkeypatch.setattr(mb, '_box_qp_active_set', lambda G, h, lb, ub: (np.full(len(h), np.nan), False))
    fb = mb.fit_joint(anc, 3, lam=3.0, gauge=0, solver='active_set')
    np.testing.assert_array_equal(fb.drift, ref.drift)


def test_solver_reaches_fit_pairwise_and_unknown_solver_is_refused():
    anc = synthetic_anchors(SMOOTH_DRIFTS, noise=0.2, seed=13)
    a = mb.fit_pairwise(anc, 0, 2, 3, lam=3.0, gauge=0)
    b = mb.fit_pairwise(anc, 0, 2, 3, lam=3.0, gauge=0, solver='active_set')
    np.testing.assert_allclose(a.drift, b.drift, atol=1e-6)
    with pytest.raises(ValueError, match='solver'):
        mb.fit_joint(anc, 3, solver='cvxpy')


# ── fit_pairwise ──────────────────────────────────────────────────────────────

def test_fit_pairwise_matches_two_batch_joint_fit():
    anc = synthetic_anchors(SMOOTH_DRIFTS, noise=0.2, seed=10)
    pw = mb.fit_pairwise(anc, 2, 0, 3, lam=3.0, gauge=0)
    np.testing.assert_allclose(pw.drift[1], 0.0)                       # untouched batch
    pa = anc.pair(0, 2)
    sub = mb.Anchors(np.zeros(len(pa), int), np.ones(len(pa), int), pa.sa, pa.sb, pa.ta, pa.tb)
    two = mb.fit_joint(sub, 2, lam=3.0, gauge=0)
    np.testing.assert_allclose(pw.drift[0], two.drift[0])
    np.testing.assert_allclose(pw.drift[2], two.drift[1])
    assert pw.scale[0, 2] == pw.scale[2, 0] == two.scale[0, 1]
    assert np.isnan(pw.scale[0, 1])
    t = np.linspace(20, T - 21, 30)
    fine = np.linspace(0, T - 1, 4000)
    truth = np.interp(t + SMOOTH_DRIFTS[0](t), fine + SMOOTH_DRIFTS[2](fine), fine)
    np.testing.assert_allclose(pw.map(0, 2, t), truth, atol=0.3)


# ── pchip_pairwise ────────────────────────────────────────────────────────────

def test_pchip_pairwise_identity_without_anchors():
    anc = synthetic_anchors(SHIFT_DRIFTS, pairs=[(0, 1)])
    f = mb.pchip_pairwise(anc, 0, 2)
    np.testing.assert_array_equal(f(GRID), GRID)


def test_pchip_pairwise_recovers_shift_both_directions():
    anc = synthetic_anchors(SHIFT_DRIFTS, n_per_pair=200, seed=11)
    t = np.linspace(30, T - 31, 30)
    np.testing.assert_allclose(mb.pchip_pairwise(anc, 1, 2)(t), t - 7.0, atol=0.3)
    np.testing.assert_allclose(mb.pchip_pairwise(anc, 2, 1)(t), t + 7.0, atol=0.3)


def test_pchip_pairwise_pinned_ends_and_monotone():
    anc = synthetic_anchors(SHIFT_DRIFTS, n_per_pair=80, noise=1.0, seed=12)
    f = mb.pchip_pairwise(anc, 1, 2)
    y = f(GRID)
    assert y[0] == pytest.approx(0.0) and y[-1] == pytest.approx(T - 1)
    assert np.all(np.diff(y) > 0)
