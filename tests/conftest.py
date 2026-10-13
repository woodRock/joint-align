"""Shared fixtures: synthetic GC-MS chromatograms with known retention-time warps."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

DATA = ROOT / 'data' / 'fish_oil'
RESULTS = ROOT / 'results'
T = 200          # RT bins, matches alignment.N_BINS
M = 120          # m/z channels for synthetic data (the code is agnostic to M)

_scripts: dict[str, object] = {}


def load_script(filename: str):
    """Import a numbered script as a module (cached, so each script is executed once)."""
    if filename not in _scripts:
        name = 'script_' + Path(filename).stem
        spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _scripts[filename] = mod
    return _scripts[filename]


def make_library(n: int = 18, m: int = M, seed: int = 0) -> np.ndarray:
    """n compound spectra, each a few sparse m/z lines, L2-normalised."""
    rng = np.random.default_rng(seed)
    lib = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        lines = rng.choice(m, 6, replace=False)
        lib[i, lines] = rng.uniform(0.2, 1.0, 6)
    return lib / np.linalg.norm(lib, axis=1, keepdims=True)


def library_positions(n: int = 18) -> np.ndarray:
    """Evenly spaced true elution positions (bins), away from the run ends."""
    return np.linspace(25, 175, n)


def make_chroma(lib: np.ndarray, pos: np.ndarray, heights: np.ndarray | None = None,
                sigma: float = 1.5, noise: float = 1e-3, seed: int = 0) -> np.ndarray:
    """(T, M) chromatogram with Gaussian peaks of each compound at fractional bins pos."""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)[:, None]
    h = np.ones(len(pos)) if heights is None else heights
    prof = h[None, :] * np.exp(-0.5 * ((t - pos[None, :]) / sigma) ** 2)
    x = prof @ lib + noise * rng.random((T, lib.shape[1]))
    return x.astype(np.float32)


@pytest.fixture(scope='session')
def library():
    return make_library()


@pytest.fixture(scope='session')
def warped_batches(library):
    """
    Three batches with known drifts. Batch k's compound positions are f_k⁻¹(p) for true
    latent positions p; here the drifts are simple shifts, so batch k sees p + shift[k].
    Returns (chromas[k][i], shifts).
    """
    shifts = np.array([0.0, 4.0, -3.0])
    p = library_positions(len(library))
    rng = np.random.default_rng(1)
    chromas = []
    for k, s in enumerate(shifts):
        chromas.append([make_chroma(library, p + s, heights=rng.uniform(0.6, 1.4, len(p)),
                                    seed=10 * k + i) for i in range(3)])
    return chromas, shifts


def data_available() -> bool:
    return DATA.exists() and any(DATA.glob('batch_*/chroma/*.npz'))


requires_data = pytest.mark.skipif(not data_available(), reason='fish oil data not available')
