"""Tests for src/encoder.py (shapes, normalisation, masking, contrastive loss)."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import encoder as enc


def _unit(x):
    torch.testing.assert_close(x.norm(dim=-1), torch.ones(x.shape[:-1]), atol=1e-5, rtol=1e-5)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_spectrum_encoder():
    m = enc.SpectrumEncoder(in_dim=40, emb_dim=16).eval()
    x = torch.rand(5, 40)
    z = m.encode(x)
    assert z.shape == (5, 16)
    _unit(z)
    _unit(m(x))
    assert m(x).shape == (5, 64)


def test_dilated_encoder_rt_conditioning():
    m = enc.DilatedSpectrumEncoder(in_dim=64, emb_dim=16, channels=8).eval()
    x = torch.rand(3, 64)
    z0 = m.encode(x)
    z1 = m.encode(x, torch.tensor([0.1, 0.5, 0.9]))
    assert z0.shape == (3, 16)
    _unit(z0)
    assert not torch.allclose(z0, z1)
    assert m(x, torch.rand(3)).shape == (3, 64)


def test_dilated_res_block_preserves_shape():
    b = enc._DilatedResBlock(4, dilation=8)
    x = torch.rand(2, 4, 50)
    assert b(x).shape == x.shape


def test_sparse_transformer_masks_padding():
    m = enc.SparseSpectrumTransformer(mz_vocab=50, d_model=16, nhead=2, n_layers=1, dim_ff=32,
                                      max_peaks=8, emb_dim=16).eval()
    x = torch.zeros(2, 50)
    x[0, [3, 7, 11]] = torch.tensor([1.0, 0.5, 0.2])
    x[1, :20] = torch.rand(20) + 0.1
    tokens, mask = m._tokenise(x)
    assert tokens.shape == (2, 8, 16)
    assert mask[0].sum() == 5 and mask[1].sum() == 0
    assert (tokens[0][mask[0]] == 0).all()
    z = m.encode(x, torch.tensor([0.2, 0.8]))
    assert z.shape == (2, 16)
    _unit(z)
    _unit(m(x))
    # tiny sub-threshold intensities are padding and do not change the embedding
    x2 = x.clone()
    x2[0, 30:40] = 1e-8
    torch.testing.assert_close(m.encode(x2), m.encode(x))


def test_attention_pool_is_weighted_mean():
    p = enc._AttentionPool1d(3)
    x = torch.rand(2, 3, 5)
    out = p(x)
    assert out.shape == (2, 3)
    assert (out <= x.max(-1).values + 1e-6).all() and (out >= x.min(-1).values - 1e-6).all()


def test_chroma_spectrum_encoder():
    m = enc.ChromaSpectrumEncoder(in_dim=32, emb_dim=16, channels=8).eval()
    x = torch.rand(3, 2 * enc.WINDOW_HALF + 1, 32)
    z = m.encode(x)
    assert z.shape == (3, 16)
    _unit(z)
    assert m(x).shape == (3, 64)


def test_peak_sequence_transformer():
    m = enc.PeakSequenceTransformer(in_dim=32, emb_dim=16, d_model=16, nhead=2, n_layers=1).eval()
    fps = torch.rand(2, 7, 32)
    rts = torch.rand(2, 7)
    z = m.encode(fps, rts)
    assert z.shape == (2, 7, 16)
    _unit(z)
    p = m(fps, rts)
    assert p.shape == (2, 7, 64)
    _unit(p)
    pe = m._rt_pe(torch.zeros(1, 1))
    torch.testing.assert_close(pe[0, 0, :8], torch.zeros(8))
    torch.testing.assert_close(pe[0, 0, 8:], torch.ones(8))


def test_nt_xent_matches_manual():
    z1 = F.normalize(torch.randn(4, 8), dim=-1)
    z2 = F.normalize(torch.randn(4, 8), dim=-1)
    tau = 0.5
    z = torch.cat([z1, z2])
    s = z @ z.T / tau
    loss = 0.0
    for i in range(8):
        pos = (i + 4) % 8
        den = sum(math.exp(s[i, k]) for k in range(8) if k != i)
        loss += -math.log(math.exp(s[i, pos]) / den)
    assert float(enc.nt_xent_loss(z1, z2, tau)) == pytest.approx(loss / 8, rel=1e-5)


def test_nt_xent_lower_for_matching_views():
    z1 = F.normalize(torch.randn(16, 32), dim=-1)
    same = enc.nt_xent_loss(z1, z1.clone())
    rand = enc.nt_xent_loss(z1, F.normalize(torch.randn(16, 32), dim=-1))
    assert same < rand
