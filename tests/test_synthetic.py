"""Tests for src/synthetic.py: the public stand-in dataset with known drifts."""
from __future__ import annotations

import json

import numpy as np
from gcmsalign.data import load_batch

import synthetic as syn


def test_dataset_layout_and_truth(tmp_path):
    truth = syn.make_dataset(tmp_path, seed=1)
    for spec in syn.DEFAULT_BATCHES:
        b = load_batch(spec.name, tmp_path)                       # readable by gcmsalign
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
