"""
Synthetic GC-MS batches with known retention-time drift, in the gcmsalign data layout.

The fish oil data is private, so this generates a public stand-in with the same design: five
batches with the same names (including the two-sample aug2015), runs that start acquiring at
different times, and drifts shaped like the real ones (apr2016 and jul2016 elute 2.5–5 min early
relative to sep2015, growing along the run; small nonlinear wiggles everywhere). Because the
true drifts are known, a run on this data can be checked against the truth (truth.json).

Generative model, per batch k and sample i:
  a compound at latent retention time τ elutes in batch k at  τ + d_k(τ) + ε_ki   (ε: jitter)
  its mass spectrum is shared by every batch; its abundance varies per sample; ~8 % of
  compounds are missing from any one sample; the run has column bleed late and no signal
  before the batch's acquisition start.
Stored like gcmsalign's preprocessing: chroma = sqrt-then-L2 per RT bin [T, M], tic_raw = the
summed raw intensity per bin [T].

make_dataset writes a world to disk in the gcmsalign layout; make_world returns the same world in
memory (no disk writes), for experiments that draw thousands of worlds. Same arguments and seed,
same arrays and truth.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

N_BINS = 200
RUN_MIN = 45.0
BIN_MIN = RUN_MIN / N_BINS
TIME_AXIS = np.arange(N_BINS) * BIN_MIN


@dataclass(frozen=True)
class BatchSpec:
    name: str
    n: int                  # samples
    shift: float            # drift at the start of the run (min)
    ramp: float             # extra drift reached by the end of the run (min), smooth S-curve
    wiggle: float           # amplitude of a slow sinusoidal drift component (min)
    phase: float            # phase of the wiggle (rad)
    start: float            # acquisition start (min): no signal before this

    def drift(self, tau: np.ndarray) -> np.ndarray:
        """d_k(τ) in minutes: where a compound at latent τ elutes in this batch, minus τ."""
        u = np.clip(np.asarray(tau, dtype=float) / RUN_MIN, 0.0, 1.0)
        s = u * u * (3 - 2 * u)                                  # smoothstep 0 → 1
        return self.shift + self.ramp * s + self.wiggle * np.sin(2 * np.pi * u + self.phase)


# Same names and the same pattern of drifts as the fish oil batches (sep2015 is the anchor-rich
# reference there); fewer samples so the demo runs in a couple of minutes.
DEFAULT_BATCHES = (
    BatchSpec('batch_apr2016', 5, -2.5, -2.3, 0.15, 0.4, 3.0),
    BatchSpec('batch_aug2015', 2, -0.2, 0.0, 0.20, 2.0, 5.0),
    BatchSpec('batch_jan2016', 8, 0.1, 0.1, 0.20, 1.1, 5.0),
    BatchSpec('batch_jul2016', 6, -2.6, -2.5, 0.15, 3.0, 3.0),
    BatchSpec('batch_sep2015', 10, 0.0, 0.0, 0.0, 0.0, 5.0),
)


def _library(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """n EI-like spectra over m channels: a few shared fragment ions plus compound-specific lines."""
    common = rng.choice(np.arange(40, 120), 6, replace=False)
    lib = np.zeros((n, m))
    for i in range(n):
        own = rng.choice(np.arange(30, m), rng.integers(8, 16), replace=False)
        lib[i, own] = rng.gamma(1.5, 1.0, len(own))
        shared = rng.choice(common, 3, replace=False)
        lib[i, shared] += rng.uniform(0.3, 1.5, 3)
    return lib / lib.sum(axis=1, keepdims=True)


def _sample(lib, tau, spec: BatchSpec, abundance, rng, sigma_bins: float, jitter_min: float):
    """One chromatogram: (chroma [T, M] sqrt-L2 normalised, tic_raw [T])."""
    m = lib.shape[1]
    present = rng.random(len(tau)) > 0.08
    rt = tau + spec.drift(tau) + rng.normal(0.0, jitter_min, len(tau))
    pos = rt / BIN_MIN
    t = np.arange(N_BINS, dtype=float)[:, None]
    h = abundance * rng.lognormal(0.0, 0.35, len(tau)) * present
    prof = h[None, :] * np.exp(-0.5 * ((t - pos[None, :]) / sigma_bins) ** 2)
    raw = prof @ lib
    bleed = np.clip((TIME_AXIS - 36.0) / 9.0, 0, None) ** 2 * 0.25        # column bleed late in the run
    bleed_spec = np.zeros(m)
    bleed_spec[[73, 147, 207, 281]] = [1.0, 0.6, 0.8, 0.5]                # siloxane-like ions
    raw = raw + bleed[:, None] * bleed_spec[None, :] + rng.exponential(2e-4, raw.shape)
    raw[TIME_AXIS < spec.start] = 0.0
    tic_raw = raw.sum(axis=1)
    sq = np.sqrt(raw)
    norm = np.linalg.norm(sq, axis=1, keepdims=True)
    chroma = np.where(norm > 1e-12, sq / np.maximum(norm, 1e-12), 0.0)
    return chroma.astype(np.float32), tic_raw.astype(np.float32)


def _generate(batches, n_compounds: int, mz_bins: int, sigma_bins: float, jitter_min: float, seed: int):
    """Draw one world: per batch, chromas [n, T, M] and tic_raws [n, T] (float32), plus the latent
    compound times τ. The order of the random draws defines the world; keep it fixed."""
    rng = np.random.default_rng(seed)
    lib = _library(n_compounds, mz_bins, rng)
    tau = np.sort(rng.uniform(5.5, 42.5, n_compounds))
    abundance = rng.lognormal(0.0, 0.8, n_compounds)
    chromas, tic_raws = [], []
    for spec in batches:
        pairs = [_sample(lib, tau, spec, abundance, rng, sigma_bins, jitter_min) for _ in range(spec.n)]
        chromas.append(np.stack([c for c, _ in pairs]))
        tic_raws.append(np.stack([t for _, t in pairs]))
    return chromas, tic_raws, tau


def _truth(batches, tau: np.ndarray, seed: int) -> dict:
    return {
        'description': 'Synthetic stand-in for the fish oil batches (src/synthetic.py). A compound '
                       'at latent time tau elutes in batch k at tau + drift_k(tau) (minutes).',
        'seed': seed, 'bin_min': BIN_MIN, 'n_bins': N_BINS,
        'compound_tau_min': np.round(tau, 4).tolist(),
        'latent_grid_min': TIME_AXIS.tolist(),
        'batches': {s.name: {**asdict(s), 'drift_min': np.round(s.drift(TIME_AXIS), 5).tolist()}
                    for s in batches},
    }


def make_world(batches=DEFAULT_BATCHES, n_compounds: int = 60, mz_bins: int = 1000, sigma_bins: float = 0.9,
               jitter_min: float = 0.04, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """One synthetic world in memory: (chromas, tic_raws, truth), where chromas[k] is batch k's
    [n_k, T, M] float32 array and tic_raws[k] its [n_k, T] raw TICs, in the order of `batches`.
    Identical to what make_dataset writes for the same arguments; nothing touches the disk."""
    chromas, tic_raws, tau = _generate(batches, n_compounds, mz_bins, sigma_bins, jitter_min, seed)
    return chromas, tic_raws, _truth(batches, tau, seed)


def make_dataset(out_dir: Path, batches=DEFAULT_BATCHES, n_compounds: int = 60, mz_bins: int = 1000,
                 sigma_bins: float = 0.9, jitter_min: float = 0.04, seed: int = 0) -> dict:
    """Write <out_dir>/<batch>/chroma/*.npz (chroma, tic_raw) and <out_dir>/truth.json.
    Returns the truth dictionary."""
    out_dir = Path(out_dir)
    chromas, tic_raws, tau = _generate(batches, n_compounds, mz_bins, sigma_bins, jitter_min, seed)
    for spec, chroma_k, tic_k in zip(batches, chromas, tic_raws):
        d = out_dir / spec.name / 'chroma'
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob('*.npz'):
            old.unlink()
        for i in range(spec.n):
            np.savez_compressed(d / f'{spec.name}_{i:02d}.npz', chroma=chroma_k[i], tic_raw=tic_k[i])
    truth = _truth(batches, tau, seed)
    (out_dir / 'truth.json').write_text(json.dumps(truth, indent=1))
    return truth


def true_map(truth: dict, a: str, b: str, t_min: np.ndarray) -> np.ndarray:
    """The true correspondence: retention time (min) in batch a → the same compound's time in b."""
    grid = np.asarray(truth['latent_grid_min'])
    ra = grid + np.asarray(truth['batches'][a]['drift_min'])          # latent → time in a
    rb = grid + np.asarray(truth['batches'][b]['drift_min'])
    tau = np.interp(t_min, ra, grid)
    return np.interp(tau, grid, rb)
