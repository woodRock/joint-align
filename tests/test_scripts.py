"""Unit tests for the helper functions inside scripts 01 and 03."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import T, load_script
from jointalign import multibatch as mb

GRID = np.arange(T, dtype=float)


@pytest.fixture(scope='module')
def s30():
    return load_script('01_multibatch_align.py')


@pytest.fixture(scope='module')
def s32():
    return load_script('03_fair_benchmark.py')


def _tic(pos, h=None, sigma=2.0):
    h = np.ones(len(pos)) if h is None else h
    return (h[None] * np.exp(-0.5 * ((GRID[:, None] - np.asarray(pos)[None]) / sigma) ** 2)).sum(1)


# irregular spacing, so a shifted copy has no aliased (periodic) alternative alignment
PEAKS = np.array([22.0, 31.0, 47.0, 55.0, 71.0, 84.0, 90.0, 108.0, 121.0, 133.0, 150.0, 163.0, 177.0])


# ── script 01 ─────────────────────────────────────────────────────────────────

def test_s30_short(s30):
    assert s30.short('batch_jan2016') == 'jan2016'


def test_s30_tic_r_matches_corrcoef(s30):
    rng = np.random.default_rng(0)
    A = [rng.random(T) for _ in range(3)]
    B = [rng.random(T) for _ in range(2)]
    r = s30.tic_r(A, B)
    assert r.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            assert r[i, j] == pytest.approx(np.corrcoef(A[i][22:], B[j][22:])[0, 1])


def test_s30_warp_tics(s30):
    t = _tic(PEAKS)
    out = s30.warp_tics([t], GRID + 5)[0]
    np.testing.assert_allclose(out[:-5], t[5:])
    np.testing.assert_allclose(out[-5:], t[-1])                      # clipped at the run end


def test_s30_assign_folds_balanced(s30):
    folds = s30.assign_folds([10, 3, 7], np.random.default_rng(0))
    assert [len(f) for f in folds] == [10, 3, 7]
    assert np.bincount(folds[0], minlength=5).tolist() == [2, 2, 2, 2, 2]
    assert sorted(folds[1].tolist()) == [0, 1, 2]
    assert np.bincount(folds[2], minlength=5).max() == 2


def test_s30_anchor_fold_masks(s30):
    anc = mb.Anchors(ba=np.array([0, 0, 1]), bb=np.array([1, 2, 2]), sa=np.array([0, 1, 0]),
                     sb=np.array([1, 0, 0]), ta=np.zeros(3), tb=np.zeros(3))
    folds = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
    tr, te = s30.anchor_fold_masks(anc, folds, 3)
    np.testing.assert_array_equal(te, [True, False, False])          # sample (1, 1) is in fold 3
    np.testing.assert_array_equal(tr, ~te)
    tr, te = s30.anchor_fold_masks(anc, folds, 4)
    np.testing.assert_array_equal(te, [False, True, True])


def test_s30_anchor_errors(s30):
    knots = np.linspace(0, T - 1, mb.N_KNOTS)
    drift = np.stack([np.zeros_like(knots), np.full_like(knots, -4.0), np.full_like(knots, 3.0)])
    model = mb.JointWarp(knots, drift, np.full((3, 3), np.nan), np.ones(1))
    ta = np.array([50.0, 60.0, 70.0])
    anc = mb.Anchors(ba=np.array([0, 0, 1]), bb=np.array([1, 2, 2]), sa=np.zeros(3, int),
                     sb=np.zeros(3, int), ta=ta, tb=np.array([54.0, 57.0, 63.0 + 1.0]))
    np.testing.assert_allclose(s30.anchor_errors(model, anc), [0.0, 0.0, -1.0], atol=1e-3)


def test_s30_boot_weights(s30):
    anc = mb.Anchors(ba=np.array([0, 0, 1]), bb=np.array([1, 1, 2]), sa=np.array([0, 1, 2]),
                     sb=np.array([0, 2, 1]), ta=np.zeros(3), tb=np.zeros(3))
    sizes = [2, 3, 4]
    w = s30.boot_weights(anc, sizes, np.random.default_rng(0))
    rng = np.random.default_rng(0)
    c = [rng.multinomial(n, np.full(n, 1.0 / n)) for n in sizes]
    np.testing.assert_array_equal(w, [c[0][0] * c[1][0], c[0][1] * c[1][2], c[1][2] * c[2][1]])
    # the expected count of a sample pair is 1
    W = np.mean([s30.boot_weights(anc, sizes, np.random.default_rng(s)) for s in range(4000)], 0)
    np.testing.assert_allclose(W, 1.0, atol=0.08)


def test_s30_composed(s30):
    knots = np.linspace(0, T - 1, mb.N_KNOTS)
    mk = lambda d: mb.JointWarp(knots, np.stack([np.full_like(knots, x) for x in d]),
                                np.full((3, 3), np.nan), np.ones(1))
    m_ah, m_hb = mk([0.0, -2.0, 0.0]), mk([0.0, 0.0, 5.0])     # a=0 → h=1, h=1 → b=2
    c = s30.Composed(m_ah, 0, 1, m_hb, 2)
    t = np.linspace(40, 150, 10)
    np.testing.assert_allclose(c.map(0, 2, t), t + 2 - 5, atol=1e-3)
    np.testing.assert_allclose(c.map(2, 0, c.map(0, 2, t)), t, atol=1e-3)
    with pytest.raises(ValueError):
        c.map(0, 1, t)


# ── script 03 ─────────────────────────────────────────────────────────────────

def test_s32_corr_uses_common_window(s32):
    x = np.random.default_rng(0).random(T)
    assert s32.corr(x, 2 * x + 1) == pytest.approx(1.0)
    assert s32.corr(x, np.ones(T)) == 0.0
    y = x.copy()
    y[:22] = np.random.default_rng(1).random(22)          # bins before the window are ignored
    assert s32.corr(x, y) == pytest.approx(1.0)


def test_s30_tic_r_uses_common_window(s30):
    rng = np.random.default_rng(0)
    a = rng.random(T)
    b = a.copy()
    b[:22] = 0.0
    assert s30.tic_r([a], [b])[0, 0] == pytest.approx(1.0)


def test_s32_read_warp_extends_linearly(s32):
    w = 2.0 * GRID + 3.0
    np.testing.assert_allclose(s32.read_warp(w, np.array([-2.0, 10.5, T + 1.0])),
                               [-1.0, 24.0, 2.0 * (T + 1) + 3.0])


def test_s32_sample_and_strict(s32):
    t = _tic(PEAKS)
    np.testing.assert_allclose(s32.sample(t, GRID + 0.5)[:-1], 0.5 * (t[:-1] + t[1:]))
    w = np.array([0.0, 2.0, 1.0, 1.0, 5.0])
    s = s32.strict(w)
    assert np.all(np.diff(s) > 0) and s[1] == pytest.approx(2.0, abs=1e-5)


def test_s32_score_identity_and_shift(s32):
    tb = _tic(PEAKS)
    p1, p2 = s32.score(tb, tb, GRID.copy())
    assert p1 == pytest.approx(1.0) and p2 == pytest.approx(1.0)
    ta = _tic(PEAKS + 6)                                              # A elutes 6 bins later
    p1, p2 = s32.score(ta, tb, GRID + 6)                              # correct map B → A
    assert p1 > 0.999 and p2 > 0.999
    u1, u2 = s32.score(ta, tb, GRID.copy())
    assert u1 < 0.2 and u2 < 0.2


def test_s32_score_midpoint_is_symmetric(s32):
    """Midpoint r of (A, B, m) equals midpoint r of (B, A, m⁻¹)."""
    ta, tb = _tic(PEAKS * 1.02 + 2), _tic(PEAKS)
    m = GRID * 1.02 + 2
    minv = np.interp(GRID, m, GRID)
    _, ab = s32.score(ta, tb, m)
    _, ba = s32.score(tb, ta, minv)
    assert ab == pytest.approx(ba, abs=0.01)


def test_s32_cow_map(s32):
    tb = _tic(PEAKS)
    ta = _tic(PEAKS + 5)
    m = s32.cow_map(ta, tb, 4, 22)
    inner = (GRID > 40) & (GRID < 160)
    np.testing.assert_allclose(m[inner], GRID[inner] + 5, atol=1.5)
