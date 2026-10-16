"""Tests for jointalign.synthetic: the public stand-in dataset with known drifts."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from alignbench.datasets.fish_oil import load_batch

from jointalign import synthetic as syn


def test_dataset_layout_and_truth(tmp_path):
    truth = syn.make_dataset(tmp_path, seed=1)
    for spec in syn.DEFAULT_BATCHES:
        b = load_batch(spec.name, tmp_path)                       # readable by the fish-oil loader
        assert b.chromas.shape == (spec.n, syn.N_BINS, 1000) and b.tic_raw.shape == (spec.n, syn.N_BINS)
        start = int(np.ceil(spec.start / syn.BIN_MIN))
        assert np.all(b.tic_raw[:, :start] == 0) and np.all(b.chromas[:, :start] == 0)
        assert np.all(b.acquisition_start == start)
        on = b.chromas[:, start:]
        np.testing.assert_allclose(np.linalg.norm(on, axis=-1), 1.0, rtol=1e-4)   # sqrt + L2 per bin
    saved = json.loads((tmp_path / 'truth.json').read_text())
    assert saved['batches'].keys() == truth['batches'].keys()
    assert len(saved['compound_tau_min']) == 60


def test_deterministic_and_rewrites(tmp_path):
    syn.make_dataset(tmp_path / 'a', seed=3)
    syn.make_dataset(tmp_path / 'b', seed=3)
    syn.make_dataset(tmp_path / 'b', seed=3)                       # rerun replaces, not appends
    fa = sorted((tmp_path / 'a').rglob('*.npz'))
    fb = sorted((tmp_path / 'b').rglob('*.npz'))
    assert [f.name for f in fa] == [f.name for f in fb]
    np.testing.assert_array_equal(np.load(fa[0])['chroma'], np.load(fb[0])['chroma'])


def test_drift_shape():
    sep = next(b for b in syn.DEFAULT_BATCHES if b.name == 'batch_sep2015')
    jul = next(b for b in syn.DEFAULT_BATCHES if b.name == 'batch_jul2016')
    np.testing.assert_allclose(sep.drift(syn.TIME_AXIS), 0.0)        # reference: no drift
    d = jul.drift(np.array([0.0, 45.0]))
    assert d[0] < -2 and d[1] < d[0]                                 # early, more so late in the run


def test_true_map_round_trip(tmp_path):
    truth = syn.make_dataset(tmp_path, seed=0)
    t = np.linspace(8, 40, 20)
    there = syn.true_map(truth, 'batch_sep2015', 'batch_jul2016', t)
    assert np.all(there < t)                                         # jul elutes earlier
    np.testing.assert_allclose(syn.true_map(truth, 'batch_jul2016', 'batch_sep2015', there), t, atol=1e-3)


def test_compounds_peak_where_the_truth_says(tmp_path):
    """The biggest compound's peak in each batch sits at τ + d_k(τ) (within jitter + binning)."""
    syn.make_dataset(tmp_path, seed=0, jitter_min=0.0)
    truth = json.loads((tmp_path / 'truth.json').read_text())
    tau = np.asarray(truth['compound_tau_min'])
    for spec in syn.DEFAULT_BATCHES:
        tic = load_batch(spec.name, tmp_path).tic_raw.mean(0)
        expect_bins = (tau + spec.drift(tau)) / syn.BIN_MIN
        top = int(np.argmax(tic[: int(36 / syn.BIN_MIN)]))         # before the column bleed
        assert np.min(np.abs(expect_bins - top)) <= 1.0


# SHA-256 of every batch's chromas then tic_raws (float32 bytes, DEFAULT_BATCHES order) for the
# default world at seed 0, computed from data written by v1 (commit f2e6510, before make_world).
V1_DEFAULT_WORLD_SHA256 = 'a0d617da71458e02591472f7719f44d461bc6d66e347af70227ff7adb44abd3d'


def test_make_world_default_is_bit_identical_to_v1():
    chromas, tic_raws, _ = syn.make_world()
    h = hashlib.sha256()
    for c, t in zip(chromas, tic_raws):
        h.update(np.ascontiguousarray(c, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(t, dtype=np.float32).tobytes())
    assert h.hexdigest() == V1_DEFAULT_WORLD_SHA256


def test_make_world_matches_make_dataset(tmp_path):
    batches = (syn.BatchSpec('batch_x', 3, 1.0, -0.5, 0.1, 0.3, 3.0),
               syn.BatchSpec('batch_y', 2, -1.5, 0.8, 0.2, 1.0, 5.0))
    kw = dict(batches=batches, n_compounds=20, mz_bins=300, sigma_bins=1.1, jitter_min=0.02, seed=5)
    chromas, tic_raws, truth = syn.make_world(**kw)
    assert truth == syn.make_dataset(tmp_path, **kw)
    assert truth == json.loads((tmp_path / 'truth.json').read_text())
    for spec, c, t in zip(batches, chromas, tic_raws):
        b = load_batch(spec.name, tmp_path)
        assert c.dtype == np.float32 and c.shape == (spec.n, syn.N_BINS, 300) and t.shape == (spec.n, syn.N_BINS)
        np.testing.assert_array_equal(c, b.chromas)
        np.testing.assert_array_equal(t, b.tic_raw)


def test_make_world_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    syn.make_world(n_compounds=10, mz_bins=300, seed=2)
    assert os.listdir(tmp_path) == []
