"""
Neural joint warp: a groupwise, transitive refinement of the joint batch model.

Every sample i in batch k gets a monotone map g_i from the shared latent RT axis to its own
RT axis:

    g_i(τ) = g0_k(0) + c_k + c_i + Σ_{m<τ} g0_k'(m) · exp(ρ · tanh(s_k(m) + s_i(m)))

  g0_k   the anchor-only joint model (src/multibatch.py), used as the backbone and the init
  s_k    batch-level log-slope residual: an MLP of Fourier features of τ and a batch embedding
  s_i    sample-level log-slope residual: an MLP of τ, the batch and a sample embedding,
         shrunk to zero and mean-zero within each batch
  c      offsets; the reference batch keeps s_k = c_k = 0 (gauge)

Both residuals start at zero, so training starts exactly at the joint model. They are trained
on two signals:

  anchors  every cross-batch spectral anchor, mapped through the two samples' own warps
           (Huber, with the joint model's robust weights so rejected anchors stay rejected)
  dense    groupwise m/z-resolved similarity in the latent frame (cosine over the whole
           RT × m/z matrix of intensity**power):
             between batches  each sample vs a template averaged over the OTHER batches,
                              each batch weighted equally
             within batch     each sample vs the leave-one-out mean of its own batch
                              (sample-level model only)
           Blur along RT anneals from coarse to fine.

TIC r is never used. Both losses are normalised by their value at initialisation and
weighted equally; this rule is fixed up front, not tuned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multibatch import Anchors, JointWarp, N_BINS

RHO = math.log(2.0)          # slope relative to the backbone stays in [0.5, 2]


def torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Row-wise linear interpolation; xp (B, T) increasing, fp (B, T) or (T,), x (B, n)."""
    if fp.dim() == 1:
        fp = fp.expand_as(xp)
    T = xp.shape[1]
    i = (torch.searchsorted(xp.detach().contiguous(), x.detach().contiguous()) - 1).clamp(0, T - 2)
    x0, x1 = xp.gather(1, i), xp.gather(1, i + 1)
    f0, f1 = fp.gather(1, i), fp.gather(1, i + 1)
    w = (x - x0) / (x1 - x0).clamp_min(1e-6)
    return f0 + w * (f1 - f0)


def warp_rows(X: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """X (N, T, M) sampled at fractional RT positions pos (N, T) → (N, T, M)."""
    T = X.shape[1]
    pos = pos.clamp(0, T - 1)
    i0 = pos.detach().floor().long().clamp(0, T - 2)
    f = (pos - i0).unsqueeze(-1)
    n = torch.arange(X.shape[0], device=X.device)[:, None]
    return (1 - f) * X[n, i0] + f * X[n, i0 + 1]


def blur_rt(X: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return X
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=X.dtype, device=X.device) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    N, T, M = X.shape
    y = F.conv1d(F.pad(X.permute(0, 2, 1).reshape(N * M, 1, T), (r, r), mode='replicate'), k)
    return y.view(N, M, T).permute(0, 2, 1)


class _Field(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralJointWarp(nn.Module):
    def __init__(self, tau: np.ndarray, g0: np.ndarray, batch_of: np.ndarray, ref: int,
                 sample_level: bool = True, d_emb: int = 8, hidden: int = 64, n_freq: int = 8):
        super().__init__()
        K, T = g0.shape
        self.K, self.T, self.ref, self.sample_level = K, T, ref, sample_level
        self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32))
        self.register_buffer('g0_start', torch.tensor(g0[:, 0], dtype=torch.float32))
        self.register_buffer('g0_slope', torch.tensor(np.diff(g0, axis=1), dtype=torch.float32)
                             .clamp_min(1e-3))
        self.register_buffer('batch_of', torch.tensor(batch_of, dtype=torch.long))
        tau = torch.arange(T - 1, dtype=torch.float32) + 0.5
        freqs = 2.0 ** torch.arange(n_freq) * math.pi / T
        ff = torch.cat([torch.sin(tau[:, None] * freqs), torch.cos(tau[:, None] * freqs),
                        (tau / T)[:, None]], dim=1)
        self.register_buffer('ff', ff)                                     # (T−1, 2F+1)
        self.batch_emb  = nn.Parameter(0.1 * torch.randn(K, d_emb))
        self.batch_off  = nn.Parameter(torch.zeros(K))
        self.batch_field = _Field(ff.shape[1] + d_emb, hidden)
        mask = torch.ones(K); mask[ref] = 0
        self.register_buffer('free', mask)
        Ns = len(batch_of)
        self.sample_emb = nn.Parameter(0.1 * torch.randn(Ns, d_emb))
        self.sample_off = nn.Parameter(torch.zeros(Ns))
        self.sample_field = _Field(ff.shape[1] + 2 * d_emb, hidden)

    def residuals(self) -> tuple[torch.Tensor, torch.Tensor]:
        T1, F_ = self.ff.shape
        e = self.batch_emb[:, None, :].expand(-1, T1, -1)
        s_b = self.batch_field(torch.cat([self.ff.expand(self.K, -1, -1), e], -1)) * self.free[:, None]
        if not self.sample_level:
            return s_b, torch.zeros(len(self.batch_of), T1, device=s_b.device)
        Ns = len(self.batch_of)
        x = torch.cat([self.ff.expand(Ns, -1, -1),
                       self.batch_emb[self.batch_of][:, None, :].expand(-1, T1, -1),
                       self.sample_emb[:, None, :].expand(-1, T1, -1)], -1)
        return s_b, self.sample_field(x)

    def _integrate(self, start: torch.Tensor, slope: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        inc = slope * torch.exp(RHO * torch.tanh(s))
        return torch.cat([start[:, None], start[:, None] + torch.cumsum(inc, 1)], 1)

    def batch_warps(self) -> torch.Tensor:
        s_b, _ = self.residuals()
        return self._integrate(self.g0_start + self.batch_off * self.free, self.g0_slope, s_b)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample warps g_i on the latent grid (Ns, T), plus the residuals for penalties."""
        s_b, s_s = self.residuals()
        k = self.batch_of
        off = self.sample_off if self.sample_level else torch.zeros_like(self.sample_off)
        start = self.g0_start[k] + (self.batch_off * self.free)[k] + off
        return self._integrate(start, self.g0_slope[k], s_b[k] + s_s), s_b, s_s


@dataclass
class TrainConfig:
    steps: int = 1500
    lr: float = 3e-3
    blur: tuple = (2.0, 1.0, 0.5)     # RT-bin σ per third of training
    n_mz: int = 400                   # m/z channels with the highest total intensity
    smooth: float = 1.0               # second differences of the log-slope residuals
    shrink: float = 1.0               # L2 on sample residuals and offsets
    huber: float = 1.0                # bins
    power: float = 0.5                # intensity transform x**power before the dense loss
    within: float = 1.0               # weight of the within-batch (leave-one-out) dense term
    seed: int = 0


def train(model: NeuralJointWarp, chromas: torch.Tensor, anchors: Anchors,
          anchor_w: np.ndarray, sample_index: np.ndarray, cfg: TrainConfig,
          use_dense: bool = True, use_anchors: bool = True, log=print) -> dict:
    """
    chromas       (Ns, T, M) raw intensities, rows in the model's sample order
    sample_index  (K, max_n) → global sample row, so anchors (ba, sa) map to rows
    anchor_w      robust weights from the joint model (0 = rejected)
    """
    torch.manual_seed(cfg.seed)
    dev = chromas.device
    X = chromas.clamp_min(0).pow(cfg.power)
    top = torch.argsort(X.sum((0, 1)), descending=True)[:cfg.n_mz]
    X = X[:, :, top]
    X = X / X.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]
    blurred = {s: blur_rt(X, s) for s in cfg.blur}

    keep = anchor_w > 0
    a = anchors.subset(keep)
    w = torch.tensor(anchor_w[keep], dtype=torch.float32, device=dev)
    pa = torch.tensor(sample_index[a.ba, a.sa], device=dev)
    pb = torch.tensor(sample_index[a.bb, a.sb], device=dev)
    ta = torch.tensor(a.ta, dtype=torch.float32, device=dev)[:, None]
    tb = torch.tensor(a.tb, dtype=torch.float32, device=dev)
    grid = model.tau
    bo = model.batch_of
    K = model.K
    onehot = F.one_hot(bo, K).float()                      # (Ns, K)
    n_k = onehot.sum(0)

    def anchor_loss(G):
        tau = torch_interp(ta, G[pa], grid)                # latent position of the a-side anchor
        pred = torch_interp(tau, grid.expand(len(pb), -1), G[pb]).squeeze(1)
        return (w * F.huber_loss(pred, tb, reduction='none', delta=cfg.huber)).sum() / w.sum()

    def dense_loss(G, Xs):
        inside = ((G >= 0) & (G <= N_BINS - 1)).float().unsqueeze(-1)
        W = warp_rows(Xs, G) * inside                      # (Ns, T, M) in the latent frame
        bmean = torch.einsum('nk,ntm->ktm', onehot, W) / n_k[:, None, None]
        other = (bmean.sum(0, keepdim=True) - bmean) / (K - 1)     # template without batch k
        Tk = other[bo]
        cos = (W * Tk).flatten(1).sum(1) / (W.flatten(1).norm(dim=1) * Tk.flatten(1).norm(dim=1))
        loss = ((onehot * (1 - cos)[:, None]).sum(0) / n_k).mean()
        if model.sample_level and cfg.within > 0:
            bsum = torch.einsum('nk,ntm->ktm', onehot, W)
            multi = (n_k[bo] > 1).float()
            loo = (bsum[bo] - W) / (n_k[bo] - 1).clamp_min(1)[:, None, None]
            c_in = (W * loo).flatten(1).sum(1) / (W.flatten(1).norm(dim=1)
                                                  * loo.flatten(1).norm(dim=1).clamp_min(1e-8))
            per = (onehot * ((1 - c_in) * multi)[:, None]).sum(0) / n_k
            loss = loss + cfg.within * per.sum() / (n_k > 1).sum()
        return loss

    def penalties(s_b, s_s):
        d2 = lambda s: (s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]).pow(2).mean()
        p = cfg.smooth * d2(s_b)
        if model.sample_level:
            mean_s = (onehot.T @ s_s) / n_k[:, None]
            mean_o = (onehot.T @ model.sample_off) / n_k
            p = p + cfg.smooth * d2(s_s) + cfg.shrink * (s_s.pow(2).mean()
                                                          + model.sample_off.pow(2).mean())
            p = p + 10 * cfg.shrink * (mean_s.pow(2).mean() + mean_o.pow(2).mean())
        return p

    with torch.no_grad():
        G, _, _ = model()
        a0 = float(anchor_loss(G)) if use_anchors else 1.0
        d0 = float(dense_loss(G, blurred[cfg.blur[-1]])) if use_dense else 1.0
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    hist = []
    for step in range(cfg.steps):
        sig = cfg.blur[min(len(cfg.blur) - 1, step * len(cfg.blur) // cfg.steps)]
        G, s_b, s_s = model()
        la = anchor_loss(G) / a0 if use_anchors else torch.zeros((), device=dev)
        ld = dense_loss(G, blurred[sig]) / d0 if use_dense else torch.zeros((), device=dev)
        loss = la + ld + penalties(s_b, s_s)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == cfg.steps - 1:
            hist.append((step, float(la), float(ld)))
            log(f'    step {step:>5}  σ={sig:.1f}  anchor {float(la):.3f}  dense {float(ld):.3f}')
    return {'a0': a0, 'd0': d0, 'hist': hist}


def latent_grid_from_joint(joint: JointWarp, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Latent grid (unit spacing) covering every batch's run, and g0_k(τ) = f_k⁻¹(τ) on it.
    f_k is extended with slope 1 beyond the run, so every g0_k is strictly increasing and
    may leave [0, N_BINS − 1] where batch k has no data.
    """
    ext = np.linspace(-N_BINS, 2 * N_BINS, 12 * N_BINS)
    lat = [joint.latent(k, ext) for k in range(K)]
    lo = np.floor(min(joint.latent(k, np.array([0.0]))[0] for k in range(K)))
    hi = np.ceil(max(joint.latent(k, np.array([N_BINS - 1.0]))[0] for k in range(K)))
    tau = np.arange(lo, hi + 1)
    return tau, np.stack([np.interp(tau, lat[k], ext) for k in range(K)])
