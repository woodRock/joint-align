"""joint-align's methods as alignbench plugins, and the 2.0 re-exports."""
from __future__ import annotations

import numpy as np
import pytest
from alignbench import anchors as ab_anchors
from alignbench import bench, datasets, methods
from alignbench.metrics import protocols as P
from alignbench.metrics.anchors import LAMBDAS, rt_folds

import jointalign
from jointalign import alignbench_plugin as plugin
from jointalign import multibatch as mb
from jointalign import synthetic as syn

GRID = np.arange(200, dtype=float)


@pytest.fixture(scope='module')
def k3():
    return datasets.load('synthetic:k3', seed=2, mz_bins=300)


@pytest.fixture(scope='module')
def anc(k3):
    return ab_anchors.harvest(k3)


def test_reexports_are_alignbench():
    assert jointalign.Anchors is ab_anchors.Anchors and mb.harvest_anchors is ab_anchors.harvest_arrays
    assert mb.match_anchors is ab_anchors.match_subbin
    from alignbench.datasets import synthetic as ab_syn
    assert syn.make_world is ab_syn.make_world and syn.N_BINS == 200 and syn.BIN_MIN == 0.225


def test_plugins_are_discovered():
    assert {'joint', 'star', 'composed-pairwise', 'neural-joint'} <= set(methods.available())
    assert isinstance(methods.get('joint'), plugin.Joint)


def test_joint_cv_matches_script30_selection(k3, anc):
    fit = methods.get('joint').fit(k3, anc)
    ref = plugin.most_anchors(anc, 3)
    # script 01: median |error| over RT-blocked folds per λ, smallest wins (ties: smaller λ)
    cv = {}
    for lam in LAMBDAS:
        errs = []
        for tr, te in rt_folds(anc):
            m = mb.fit_joint(anc.subset(tr), 3, lam=lam, gauge=ref)
            sub = anc.subset(te)
            e = np.empty(len(sub))
            for a in range(3):
                for b in range(a + 1, 3):
                    sel = (sub.ba == a) & (sub.bb == b)
                    e[sel] = m.map(a, b, sub.ta[sel]) - sub.tb[sel]
            errs.append(np.abs(e))
        cv[lam] = float(np.median(np.concatenate(errs)))
    lam = min(LAMBDAS, key=lambda l: cv[l])
    assert fit.params == {'lam': lam, 'gauge': ref, 'solver': 'bvls'}
    m = mb.fit_joint(anc, 3, lam=lam, gauge=ref)
    for k, l in ((0, 1), (2, 0), (1, 2)):
        assert np.array_equal(fit.map(k, l, GRID), m.map(k, l, GRID))
    assert fit.used_anchors.all() and fit.model.drift.shape == (3, mb.N_KNOTS)


def test_star_and_composed_pairwise(k3, anc):
    star = methods.get('star').fit(k3, anc, hub=1, lam=10.0)
    assert star.params['hub'] == 1 and star.params['gauge'] == 1
    assert np.array_equal(star.used_anchors, (anc.ba == 1) | (anc.bb == 1))
    cp = methods.get('composed-pairwise').fit(k3, anc, lam=10.0, edges={(0, 1), (1, 2)})
    assert cp.extras['paths'][(0, 2)] == [0, 1, 2] and cp.params['lam'] == 10.0
    m01 = mb.fit_pairwise(anc, 0, 1, 3, lam=10.0, gauge=0)
    m12 = mb.fit_pairwise(anc, 1, 2, 3, lam=10.0, gauge=0)
    assert np.array_equal(cp.map(0, 2, GRID), m12.map(1, 2, m01.map(0, 1, GRID)))
    assert methods.get('composed-pairwise').fit(k3, anc, lam='cv').params['lam'] in LAMBDAS
    with pytest.raises(ValueError, match='needs anchors'):
        methods.get('joint').fit(k3, None)


def test_neural_joint_is_per_sample(k3, anc):
    pytest.importorskip('torch')
    fit = methods.get('neural-joint').fit(k3, anc, steps=2, lam=10.0)
    assert fit.per_sample and fit.params['steps'] == 2
    m = fit.map(1, 0, GRID, i=0, j=1)
    assert m.shape == (200,) and np.all(np.diff(m) >= 0)
    with pytest.raises(ValueError, match='per sample'):
        fit.map(1, 0, GRID)


def test_benchmark_cell_with_joint(tmp_path):
    spec = bench.CellSpec('joint', 'synthetic:k3', 2, ('tic_bframe', 'map_error', 'heldout_anchor_error'),
                          {'lam': 10.0}, {'mz_bins': 300})
    cell = bench.compute_cell(spec)
    assert not cell['skipped'] and cell['fit']['params']['lam'] == 10.0
    assert any(line.startswith('joint-align ') for line in cell['provenance'])
    rows = [r for r in cell['rows'] if r['protocol'] == 'map_error' and r['batch_a'] == '*']
    summary = {r['metric']: r['value'] for r in rows}
    assert summary['median_all_min'] < 0.1
    n = [len(b) for b in datasets.load('synthetic:k3', seed=2, mz_bins=300).batches]
    assert P.as_array(cell['rows'], 'batch_00', 'batch_01', 'tic_r').shape == (n[0], n[1])
