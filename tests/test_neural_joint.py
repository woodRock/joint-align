"""Tests for src/neural_joint.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import multibatch as mb
import neural_joint as nj
from conftest import T, library_positions, make_chroma

GRID = np.arange(T, dtype=float)


# ── primitives ────────────────────────────────────────────────────────────────

def test_torch_interp_matches_numpy():
    rng = np.random.default_rng(0)
    xp = np.sort(rng.uniform(0, 50, (3, 20)), axis=1)
    fp = rng.normal(size=(3, 20))
    x = rng.uniform(xp[:, :1], xp[:, -1:], (3, 15))
    out = nj.torch_interp(torch.tensor(x), torch.tensor(xp), torch.tensor(fp)).numpy()
    for i in range(3):
        np.testing.assert_allclose(out[i], np.interp(x[i], xp[i], fp[i]), atol=1e-9)


def test_torch_interp_1d_fp_and_linear_extrapolation():
    xp = torch.arange(10.0).expand(2, -1)
    fp = 2 * torch.arange(10.0)
    x = torch.tensor([[-1.0, 4.5], [9.5, 3.0]])
    torch.testing.assert_close(nj.torch_interp(x, xp, fp), 2 * x)


def test_torch_interp_gradients():
    xp = torch.arange(10.0).expand(1, -1).clone().requires_grad_(True)
    fp = (3 * torch.arange(10.0)).expand(1, -1).clone().requires_grad_(True)
    x = torch.tensor([[2.25]], requires_grad=True)
    nj.torch_interp(x, xp, fp).sum().backward()
    assert x.grad.item() == pytest.approx(3.0)
    assert fp.grad[0, 2].item() == pytest.approx(0.75) and fp.grad[0, 3].item() == pytest.approx(0.25)


def test_warp_rows_matches_warp_bins_including_last_bin():
    x = np.random.default_rng(0).random((T, 6)).astype(np.float32)
    pos = np.r_[np.linspace(-3, T + 3, T)]
    out = nj.warp_rows(torch.tensor(x)[None], torch.tensor(pos, dtype=torch.float32)[None])[0]
    np.testing.assert_allclose(out.numpy(), mb.warp_bins(x, pos), atol=1e-5)
    ident = nj.warp_rows(torch.tensor(x)[None], torch.tensor(GRID, dtype=torch.float32)[None])[0]
    np.testing.assert_allclose(ident.numpy(), x, atol=1e-6)


def test_warp_rows_different_output_length():
    x = torch.rand(2, T, 4)
    pos = torch.linspace(0, T - 1, 37).expand(2, -1)
    assert nj.warp_rows(x, pos).shape == (2, 37, 4)


def test_blur_rt():
    x = torch.rand(2, 50, 3)
    assert nj.blur_rt(x, 0) is x
    const = torch.ones(1, 40, 2)
    torch.testing.assert_close(nj.blur_rt(const, 2.0), const)
    spike = torch.zeros(1, 41, 2)
    spike[0, 20, 1] = 1.0
    b = nj.blur_rt(spike, 1.5)
    assert b.shape == spike.shape
    assert float(b.sum()) == pytest.approx(1.0, abs=1e-5)
    assert int(b[0, :, 1].argmax()) == 20 and float(b[0, :, 0].abs().max()) == 0.0
    torch.testing.assert_close(b[0, 20 - 3, 1], b[0, 20 + 3, 1])


# ── model ─────────────────────────────────────────────────────────────────────

def _joint(shifts=(0.0, -4.0, 3.0)):
    knots = np.linspace(0, T - 1, mb.N_KNOTS)
    drift = np.stack([np.full_like(knots, s) for s in shifts])
    return mb.JointWarp(knots, drift, np.full((3, 3), np.nan), np.ones(1))


def test_latent_grid_identity():
    tau, g0 = nj.latent_grid_from_joint(_joint((0.0, 0.0, 0.0)), 3)
    np.testing.assert_array_equal(tau, GRID)
    np.testing.assert_allclose(g0, np.broadcast_to(GRID, (3, T)), atol=1e-6)


def test_latent_grid_covers_all_batches_and_inverts():
    j = _joint()
    tau, g0 = nj.latent_grid_from_joint(j, 3)
    assert tau[0] == -4.0 and tau[-1] == T - 1 + 3.0
    assert np.all(np.diff(tau) == 1.0)
    assert np.all(np.diff(g0, axis=1) > 0)
    for k in range(3):
        np.testing.assert_allclose(j.latent(k, g0[k]), tau, atol=1e-6)


def _model(sample_level=True, sizes=(2, 3, 2), ref=0):
    torch.manual_seed(0)
    tau, g0 = nj.latent_grid_from_joint(_joint(), 3)
    batch_of = np.concatenate([[k] * n for k, n in enumerate(sizes)])
    return nj.NeuralJointWarp(tau, g0, batch_of, ref, sample_level=sample_level, hidden=16), tau, g0


@pytest.mark.parametrize('sample_level', [True, False])
def test_model_starts_at_joint_backbone(sample_level):
    m, tau, g0 = _model(sample_level)
    with torch.no_grad():
        G, s_b, s_s = m()
        Gb = m.batch_warps()
    np.testing.assert_allclose(Gb.numpy(), g0, atol=1e-3)
    np.testing.assert_allclose(G.numpy(), g0[m.batch_of.numpy()], atol=1e-3)
    assert float(s_b.abs().max()) == 0.0 and float(s_s.abs().max()) == 0.0
    assert G.shape == (7, len(tau)) and s_b.shape == (3, len(tau) - 1)


def test_model_reference_batch_is_fixed_and_warps_monotone():
    m, tau, g0 = _model(ref=1)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 1.0)
        Gb = m.batch_warps()
        G, s_b, _ = m()
    np.testing.assert_allclose(Gb[1].numpy(), g0[1], atol=1e-3)       # gauge: s_ref = c_ref = 0
    assert float(s_b[1].abs().max()) == 0.0
    assert not np.allclose(Gb[0].numpy(), g0[0], atol=1e-2)
    assert (G.diff(dim=1) > 0).all()
    slope_ratio = G.diff(dim=1) / m.g0_slope[m.batch_of]
    assert float(slope_ratio.min()) >= 0.5 - 1e-4 and float(slope_ratio.max()) <= 2.0 + 1e-4


def test_model_without_sample_level_ignores_sample_parameters():
    m, *_ = _model(sample_level=False)
    with torch.no_grad():
        m.sample_off.fill_(5.0)
        G, _, s_s = m()
    assert float(s_s.abs().max()) == 0.0
    np.testing.assert_allclose(G[0].numpy(), G[1].numpy())          # same batch → same warp


# ── training ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def tiny_problem(library):
    """3 batches × 2 samples shifted by (0, −4, +3) bins, anchors from match_anchors."""
    shifts = np.array([0.0, 4.0, -3.0])          # positions in batch k: p + shift_k
    p = library_positions(len(library))
    chromas = [[make_chroma(library, p + s, seed=10 * k + i) for i in range(2)]
               for k, s in enumerate(shifts)]
    anc = mb.harvest_anchors(chromas)
    joint = mb.fit_joint(anc, 3, lam=10.0, gauge=0)
    sizes = [2, 2, 2]
    batch_of = np.repeat(np.arange(3), 2)
    sidx = np.arange(6).reshape(3, 2)
    X = torch.tensor(np.stack([c for cs in chromas for c in cs]))
    return chromas, anc, joint, batch_of, sidx, X, shifts


def _train(tiny_problem, steps=12, sample_level=True, **kw):
    _, anc, joint, batch_of, sidx, X, _ = tiny_problem
    tau, g0 = nj.latent_grid_from_joint(joint, 3)
    torch.manual_seed(0)
    model = nj.NeuralJointWarp(tau, g0, batch_of, 0, sample_level=sample_level, hidden=16)
    logs = []
    cfg = nj.TrainConfig(steps=steps, n_mz=60, blur=(1.0, 0.5))
    out = nj.train(model, X, anc, joint.weights, sidx, cfg, log=logs.append, **kw)
    return model, out, logs


def test_joint_fit_on_tiny_problem(tiny_problem):
    *_, joint, _, _, _, shifts = tiny_problem
    t = np.linspace(30, 170, 20)
    np.testing.assert_allclose(joint.map(1, 2, t), t - 7.0, atol=0.5)


def test_train_runs_and_logs(tiny_problem):
    model, out, logs = _train(tiny_problem)
    assert out['a0'] > 0 and out['d0'] > 0
    assert [h[0] for h in out['hist']] == [0, 11]
    assert all(np.isfinite(h[1]) and np.isfinite(h[2]) for h in out['hist'])
    assert len(logs) == 2 and 'anchor' in logs[0]
    # losses are normalised by their initial value, so step 0 reads 1.0
    assert out['hist'][0][1] == pytest.approx(1.0, abs=1e-5)
    # d0 is measured at the finest blur, but step 0 trains at the coarsest, where samples
    # agree more; so the dense term starts below 1 (its effective weight changes with σ)
    assert 0.0 < out['hist'][0][2] < 1.0
    with torch.no_grad():
        G, _, _ = model()
    assert torch.isfinite(G).all() and (G.diff(dim=1) > 0).all()


def test_train_keeps_good_backbone(tiny_problem):
    """Starting from an already correct joint fit, training must not wreck the anchor fit."""
    model, out, _ = _train(tiny_problem, steps=30)
    assert out['hist'][-1][1] < 1.5


@pytest.mark.parametrize('kw', [dict(use_dense=False), dict(use_anchors=False),
                                dict(sample_level=False)])
def test_train_ablations(tiny_problem, kw):
    model, out, _ = _train(tiny_problem, steps=4, **kw)
    if kw.get('use_dense') is False:
        assert out['d0'] == 1.0 and all(h[2] == 0.0 for h in out['hist'])
    if kw.get('use_anchors') is False:
        assert out['a0'] == 1.0 and all(h[1] == 0.0 for h in out['hist'])
    assert all(np.isfinite(h[1:]).all() for h in out['hist'])


def test_train_ignores_rejected_anchors(tiny_problem):
    """Anchors with joint weight 0 must not enter the anchor loss."""
    _, anc, joint, batch_of, sidx, X, _ = tiny_problem
    w = joint.weights.copy()
    w[: len(w) // 2] = 0
    tau, g0 = nj.latent_grid_from_joint(joint, 3)
    torch.manual_seed(0)
    model = nj.NeuralJointWarp(tau, g0, batch_of, 0, hidden=16)
    corrupted = mb.Anchors(anc.ba, anc.bb, anc.sa, anc.sb, anc.ta,
                           np.where(w == 0, anc.tb + 50, anc.tb))
    cfg = nj.TrainConfig(steps=1, n_mz=60)
    out_bad = nj.train(model, X, corrupted, w, sidx, cfg, use_dense=False, log=lambda s: None)
    torch.manual_seed(0)
    model = nj.NeuralJointWarp(tau, g0, batch_of, 0, hidden=16)
    out_ok = nj.train(model, X, anc, w, sidx, cfg, use_dense=False, log=lambda s: None)
    assert out_bad['a0'] == pytest.approx(out_ok['a0'])
