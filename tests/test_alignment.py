"""Tests for src/alignment.py: peak detection, pairwise PCHIP alignment and helpers."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import alignment as al
from encoder import ChromaSpectrumEncoder, DilatedSpectrumEncoder, PeakSequenceTransformer
from conftest import T, library_positions, make_chroma, make_library

GRID = np.arange(T, dtype=float)


def test_constants_consistent():
    assert al.N_BINS == T
    assert al.BIN_MIN == pytest.approx(al.RUN_MIN / al.N_BINS)
    assert al.TIME_AXIS[0] == 0 and al.TIME_AXIS[-1] == pytest.approx(al.RUN_MIN - al.BIN_MIN)


def test_time_axis_matches_bin_width():
    np.testing.assert_allclose(np.diff(al.TIME_AXIS), al.BIN_MIN)
    b = np.round(al.TIME_AXIS / al.BIN_MIN).astype(int)                # minutes → bins round trip
    np.testing.assert_array_equal(b, np.arange(al.N_BINS))


def test_l2():
    v = np.array([3.0, 4.0])
    np.testing.assert_allclose(al._l2(v), [0.6, 0.8])
    assert al._l2(v).dtype == np.float32
    np.testing.assert_array_equal(al._l2(np.zeros(3)), np.zeros(3))


def test_ctx_fp_mean_of_three_rows_and_edges():
    x = np.random.default_rng(0).random((T, 10)).astype(np.float32)
    np.testing.assert_allclose(al._ctx_fp(x, 50), al._l2(x[49:52].mean(0)), rtol=1e-6)
    np.testing.assert_allclose(al._ctx_fp(x, 0), al._l2(x[0:2].mean(0)), rtol=1e-6)
    np.testing.assert_allclose(al._ctx_fp(x, T - 1), al._l2(x[T - 2:T].mean(0)), rtol=1e-6)


def test_detect_peaks_finds_compounds(library):
    p = library_positions(len(library))
    pk = al.detect_peaks(make_chroma(library, p, noise=0.0))
    np.testing.assert_array_equal(pk, np.round(p).astype(int))


def test_detect_peaks_caps_count_keeps_tallest():
    lib = make_library(40, 60, seed=1)
    p = np.linspace(5, 194, 40)
    h = np.linspace(0.1, 4.0, 40)
    pk = al.detect_peaks(make_chroma(lib, p, heights=h, sigma=1.0, noise=0.0),
                         height_pct=0, min_dist=2, n_peaks=10)
    assert len(pk) == 10 and np.all(np.diff(pk) > 0)
    np.testing.assert_array_equal(pk, np.round(p[-10:]).astype(int))


def test_fingerprints_shape_and_norm(library):
    x = make_chroma(library, library_positions(len(library)))
    fp = al.fingerprints(x, np.array([30, 60, 90]))
    assert fp.shape == (3, x.shape[1])
    np.testing.assert_allclose(np.linalg.norm(fp, axis=1), 1.0, rtol=1e-5)


def test_peak_windows():
    x = np.random.default_rng(0).random((T, 12)).astype(np.float32)
    x[3] = 0.0
    w = al.peak_windows(x, np.array([2, 100]), half_width=2)
    assert w.shape == (2, 5, 12)
    np.testing.assert_allclose(w[0, 0], al._l2(x[0]), rtol=1e-6)       # t = pk − 2 = 0 is valid
    np.testing.assert_array_equal(w[0, 3], 0.0)                      # zero row stays zero
    np.testing.assert_allclose(np.linalg.norm(w[1], axis=1), 1.0, rtol=1e-5)
    w = al.peak_windows(x, np.array([0, T - 1]), half_width=2)
    np.testing.assert_array_equal(w[0, :2], 0.0)                     # before the run
    np.testing.assert_array_equal(w[1, 3:], 0.0)                     # after the run


# ── align_pair ────────────────────────────────────────────────────────────────

def test_align_pair_identical_is_identity(library):
    x = make_chroma(library, library_positions(len(library)))
    warped, n, prec = al.align_pair(x, x)
    assert n >= 10 and prec == 1.0
    np.testing.assert_allclose(warped, x.sum(1), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('shift', [-5, 6])
def test_align_pair_corrects_shift(library, shift):
    p = library_positions(len(library))
    ref = make_chroma(library, p, seed=1)
    q = make_chroma(library, p + shift, seed=2)
    out = al.align_pair(q, ref, return_anchors=True)
    warped, n, prec, proposed, warp, ransac = out
    assert len(out) == 6 and n >= 10 and warp is not None
    r0 = np.corrcoef(q.sum(1), ref.sum(1))[0, 1]
    r1 = np.corrcoef(warped, ref.sum(1))[0, 1]
    assert r0 < 0.5 and r1 > 0.95
    inner = (al.TIME_AXIS > 8) & (al.TIME_AXIS < 37)
    shift_min = (warp(al.TIME_AXIS) - al.TIME_AXIS)[inner]
    np.testing.assert_allclose(shift_min, shift * al.BIN_MIN, atol=0.25)
    assert all(len(a) == 3 for a in proposed)
    assert all(len(a) == 2 for a in ransac)


def test_align_pair_ransac_anchor_bins_match_peaks(library):
    """RANSAC anchors are reported as (ref_bin, query_bin) and must equal the matched peaks."""
    p = np.round(library_positions(len(library)))
    ref = make_chroma(library, p, seed=1, noise=0.0)
    q = make_chroma(library, p + 4, seed=2, noise=0.0)
    *_, ransac = al.align_pair(q, ref, return_anchors=True)
    r, qq = np.array(ransac).T
    # flat cosine across a pure peak lets the ±3-bin refinement tie-break by one bin
    assert np.median(qq - r) == 4
    assert np.all(np.abs(qq - r - 4) <= 1)
    assert set(r) <= set(p.astype(int))


def test_align_pair_ransac_anchor_bins_late_peaks(library):
    """Regression: bins ≥ 100 used to come back as bin + 1."""
    p = np.round(library_positions(len(library)))
    ref = make_chroma(library, p, seed=1, noise=0.0)
    q = make_chroma(library, p + 4, seed=2, noise=0.0)
    *_, ransac = al.align_pair(q, ref, return_anchors=True)
    r = np.array(ransac)[:, 0]
    assert set(r) <= set(p.astype(int))


def test_align_pair_with_encode_fn(library):
    x = make_chroma(library, library_positions(len(library)))
    calls = []

    def enc(fps, rts):
        calls.append((fps.shape, rts.shape))
        return fps
    warped, n, _ = al.align_pair(x, x, encode_fn=enc)
    assert len(calls) == 2 and n >= 10

    def win(w):
        calls.append(w.shape)
        return w.reshape(len(w), -1)
    win.uses_windows = True
    al.align_pair(x, x, encode_fn=win)
    assert calls[-1][1] == 2 * al.WINDOW_HALF + 1


def test_align_pair_failure_paths():
    flat = np.ones((T, 10), dtype=np.float32)
    warped, n, prec = al.align_pair(flat, flat)
    assert n == 0 and prec == 0.0
    np.testing.assert_array_equal(warped, flat.sum(1))
    out = al.align_pair(flat, flat, return_anchors=True)
    assert out[4] is None and out[3] == [] and out[5] == []
    # peaks exist but no spectrum matches confidently → fewer than 2 anchors
    lib_a = make_library(10, 60, seed=1)
    lib_a[:, 59] = 0.0
    lib_a /= np.linalg.norm(lib_a, axis=1, keepdims=True)
    lib_b = np.zeros((10, 60), dtype=np.float32)
    lib_b[:, 59] = 1.0
    a = make_chroma(lib_a, library_positions(10), noise=0.0)
    b = make_chroma(lib_b, library_positions(10) + 3, noise=0.0)
    out = al.align_pair(b, a, return_anchors=True)
    assert out[1] == 0 and out[4] is None


# ── warp_chroma_2d ────────────────────────────────────────────────────────────

def test_warp_chroma_2d_none_returns_copy():
    x = np.random.default_rng(0).random((T, 5)).astype(np.float32)
    y = al.warp_chroma_2d(x, None)
    np.testing.assert_array_equal(y, x)
    assert y is not x


def test_warp_chroma_2d_clips_and_interpolates():
    x = np.random.default_rng(0).random((T, 5)).astype(np.float32)
    y = al.warp_chroma_2d(x, lambda t: np.full_like(t, 100.0))   # beyond the run → last bin
    np.testing.assert_allclose(y, np.broadcast_to(x[-1], y.shape))
    y = al.warp_chroma_2d(x, lambda t: np.full_like(t, 10.5 * al.BIN_MIN))
    np.testing.assert_allclose(y[0], 0.5 * (x[10] + x[11]), rtol=1e-5)


def test_warp_chroma_2d_identity():
    x = np.random.default_rng(0).random((T, 5)).astype(np.float32)
    np.testing.assert_allclose(al.warp_chroma_2d(x, lambda t: t), x, atol=1e-5)


def test_max_projection():
    x = np.random.default_rng(0).random((T, 5)).astype(np.float32)
    np.testing.assert_array_equal(al.max_projection(x), x.max(0))


def test_embed_chromatogram(library):
    x = make_chroma(library, library_positions(len(library)))
    emb = al.embed_chromatogram(x, lambda fps, rts: np.c_[fps, rts])
    assert emb.shape == (x.shape[1] + 1,)
    np.testing.assert_array_equal(al.embed_chromatogram(np.ones((T, 4), np.float32), None),
                                  np.zeros(128))


# ── I/O ───────────────────────────────────────────────────────────────────────

def test_load_chromas_sorted_float32(tmp_path):
    for name, v in (('b.npz', 2.0), ('a.npz', 1.0), ('c.npz', 3.0)):
        np.savez(tmp_path / name, chroma=np.full((T, 3), v, dtype=np.float64))
    out = al.load_chromas(tmp_path)
    assert [float(c[0, 0]) for c in out] == [1.0, 2.0, 3.0]
    assert all(c.dtype == np.float32 for c in out)


@pytest.mark.parametrize('cls,uses_windows', [(DilatedSpectrumEncoder, False),
                                              (ChromaSpectrumEncoder, True),
                                              (PeakSequenceTransformer, False)])
def test_load_encoder_detects_type(tmp_path, cls, uses_windows):
    torch.manual_seed(0)
    path = tmp_path / 'enc.pt'
    torch.save(cls().state_dict(), path)
    fn = al.load_encoder(path, device=torch.device('cpu'))
    assert fn.uses_windows is uses_windows
    rng = np.random.default_rng(0)
    if uses_windows:
        out = fn(rng.random((3, 2 * al.WINDOW_HALF + 1, 1000)).astype(np.float32))
    else:
        out = fn(rng.random((3, 1000)).astype(np.float32), rng.random(3).astype(np.float32))
    assert out.shape == (3, 128)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_load_encoder_default_device(tmp_path):
    path = tmp_path / 'enc.pt'
    torch.save(PeakSequenceTransformer().state_dict(), path)
    fn = al.load_encoder(path)
    out = fn(np.random.default_rng(0).random((4, 1000)).astype(np.float32),
             np.linspace(0, 1, 4).astype(np.float32))
    assert out.shape == (4, 128)
