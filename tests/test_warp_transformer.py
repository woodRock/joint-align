"""Tests for src/warp_transformer.py."""
from __future__ import annotations

import pytest
import torch

import warp_transformer as wt


def test_apply_warp_integer_shift_and_fraction():
    x = torch.rand(2, 20, 3)
    w = (torch.arange(20.0) + 2).clamp(max=19).expand(2, -1)
    out = wt.apply_warp(x, w)
    torch.testing.assert_close(out[:, :17], x[:, 2:19])
    w = torch.full((2, 20), 4.25)
    torch.testing.assert_close(wt.apply_warp(x, w)[:, 0], 0.75 * x[:, 4] + 0.25 * x[:, 5])


def test_apply_warp_interior_identity():
    x = torch.rand(3, 20, 4)
    w = torch.arange(20.0).expand(3, -1)
    torch.testing.assert_close(wt.apply_warp(x, w)[:, :-1], x[:, :-1])


def test_apply_warp_identity_last_bin():
    """Regression: warps == T−1 used to return bin T−2."""
    x = torch.rand(3, 20, 4)
    w = torch.arange(20.0).expand(3, -1)
    torch.testing.assert_close(wt.apply_warp(x, w), x)


def test_apply_warp_clamps_out_of_range():
    x = torch.rand(1, 10, 2)
    w = torch.tensor([[-3.0] * 5 + [12.0] * 5])
    out = wt.apply_warp(x, w)
    torch.testing.assert_close(out[0, :5], x[0, :1].expand(5, -1))
    torch.testing.assert_close(out[0, 5:], x[0, -1:].expand(5, -1))


def test_apply_warp_gradient_flows_to_warps():
    x = torch.arange(10.0).view(1, 10, 1).repeat(1, 1, 2)
    w = torch.full((1, 10), 3.3, requires_grad=True)
    wt.apply_warp(x, w).sum().backward()
    torch.testing.assert_close(w.grad, torch.full((1, 10), 2.0))     # d/dw of a ramp, 2 channels


def test_make_monotone_warp():
    raw = torch.randn(4, 50) * 3
    w = wt.make_monotone_warp(raw)
    assert (w.diff(dim=1) > 0).all()
    torch.testing.assert_close(w[:, -1], torch.full((4,), 49.0))
    assert (w[:, 0] > 0).all()                       # first increment is positive, not 0
    w0 = wt.make_monotone_warp(torch.zeros(1, 50))
    torch.testing.assert_close(w0[0], (torch.arange(50.0) + 1) * 49 / 50)


def test_warp_smoothness_loss():
    ident = torch.arange(30.0).expand(2, -1)
    assert float(wt.warp_smoothness_loss(ident)) == 0.0
    assert float(wt.warp_smoothness_loss(ident * 2)) == pytest.approx(1.0)


def test_enable_cuda_determinism_is_noop_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    before = torch.are_deterministic_algorithms_enabled()
    wt.enable_cuda_determinism()
    assert torch.are_deterministic_algorithms_enabled() == before


def test_enable_cuda_determinism_with_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    before = torch.are_deterministic_algorithms_enabled()
    try:
        wt.enable_cuda_determinism()
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        import os
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    finally:
        torch.use_deterministic_algorithms(before)
        torch.backends.cudnn.deterministic = False


def _small_model():
    torch.manual_seed(0)
    return wt.ChromaWarpTransformer(mz_bins=16, n_bins=24, d_model=16, n_heads=2, n_layers=1)


def test_model_shapes_and_init_warp():
    m = _small_model().eval()
    x = torch.rand(5, 24, 16)
    ids = torch.tensor([0, 0, 1, 1, 1])
    aligned, warps = m(x, ids)
    assert aligned.shape == x.shape and warps.shape == (5, 24)
    assert (warps.diff(dim=1) > 0).all()
    # zero-init head → raw = 0 → the same evenly spaced warp for every sample
    torch.testing.assert_close(warps, wt.make_monotone_warp(torch.zeros(5, 24)))
    assert m.encode(x, ids).shape == (5, 24, 16)


def test_model_permutation_equivariant_over_samples():
    m = _small_model().eval()
    with torch.no_grad():
        m.warp_head[-1].weight.normal_()                 # make warps depend on the input
    x = torch.rand(4, 24, 16)
    ids = torch.tensor([0, 1, 0, 1])
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        _, w = m(x, ids)
        _, wp = m(x[perm], ids[perm])
    torch.testing.assert_close(wp, w[perm], atol=1e-5, rtol=1e-5)


def test_sinusoidal_pe():
    pe = wt.ChromaWarpTransformer._sinusoidal_pe(10, 8)
    assert pe.shape == (10, 8)
    torch.testing.assert_close(pe[0, 0::2], torch.zeros(4))
    torch.testing.assert_close(pe[0, 1::2], torch.ones(4))
