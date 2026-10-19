"""scripts/export_ledger.py: script 01's npz → taura ledger results."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import load_script


@pytest.fixture(scope='module')
def ex():
    return load_script('export_ledger.py')


def _npz(path, ex, unaligned):
    """A minimal npz with script 01's key layout: two pairs, every method, known means."""
    names = np.array(['batch_apr2016', 'batch_jan2016', 'batch_sep2015'])
    n_pair = np.array([[0, 11, 22], [11, 0, 33], [22, 33, 0]])
    arrays = {}
    for i, (a, b) in enumerate((('sep2015', 'apr2016'), ('jan2016', 'apr2016'))):
        for j, method in enumerate(ex.METHODS):
            v = unaligned[i] if method == 'Unaligned' else 0.1 * j + 0.01 * i
            arrays[f'ticr_{a}_{b}_{method}'] = np.full((3, 2), v)
    np.savez(path, names=names, n_pair=n_pair, provenance=np.array(['joint-align test']), **arrays)
    return path


def test_results(ex, tmp_path):
    out = ex.results(_npz(tmp_path / 'm.npz', ex, unaligned=[0.05, 0.8]))
    assert set(out) == {'pairs'} | {name for name, _ in ex.METHODS.values()}
    pairs = out['pairs']
    assert pairs['groups'] == {'drifted': ['sep2015-apr2016'], 'near-aligned': ['jan2016-apr2016']}
    assert pairs['metrics'] == {'sep2015-apr2016': {'anchors': 22}, 'jan2016-apr2016': {'anchors': 11}}
    joint = out['ticr-joint']['metrics']        # Joint is the 6th method: 0.5 (+0.01 for the 2nd pair)
    assert joint['sep2015-apr2016'] == {'mean': pytest.approx(0.5), 'n': 6}
    assert joint['jan2016-apr2016']['mean'] == pytest.approx(0.51)
    assert joint['mean-drifted'] == {'mean': pytest.approx(0.5), 'n': 1}
    assert out['ticr-unaligned']['metrics']['mean-near-aligned']['mean'] == pytest.approx(0.8)
    assert out['ticr-joint']['source']['hash'].startswith('sha256:')


def test_main_is_deterministic(ex, tmp_path):
    npz = _npz(tmp_path / 'm.npz', ex, unaligned=[0.05, 0.8])
    first = {p.name: p.read_bytes() for p in ex.main(npz, tmp_path / 'out', reports=False)}
    second = {p.name: p.read_bytes() for p in ex.main(npz, tmp_path / 'out', reports=False)}
    assert first == second and 'pairs.json' in first
    assert json.loads(first['ticr-joint.json'])['name'] == 'ticr-joint'


def test_unknown_method(ex, tmp_path):
    np.savez(tmp_path / 'bad.npz', names=np.array(['batch_apr2016', 'batch_sep2015']),
             n_pair=np.zeros((2, 2), int), provenance=np.array(['x']),
             ticr_sep2015_apr2016_Mystery=np.zeros((1, 1)))
    with pytest.raises(ValueError, match='Mystery'):
        ex.results(tmp_path / 'bad.npz')


# The reports below are the committed outputs of scripts 01, 03 and 04.

def test_report01(ex):
    r = ex.report01(ex.TXT01.read_text())
    assert r['anchors'] == {'all': 25011, 'star-design': 17595}
    assert r['lambda-cv']['sep2015-lam10']['rt-blocked'] == 0.1106
    assert r['lambda-cv']['selected'] == {'lambda': 10}
    assert r['lambda-sweep']['lam0.1'] == {'drifted': 0.622, 'near-aligned': 0.766}
    assert r['lambda-sweep']['lam-ge1'] == {'drifted-min': 0.595, 'drifted-max': 0.6}
    assert set(r) >= {'wilcoxon-joint-vs-pooled-pchip', 'wilcoxon-joint-vs-pairwise',
                      'wilcoxon-joint-cf-vs-pairwise-cf', 'wilcoxon-joint-vs-unaligned'}
    w = r['wilcoxon-joint-vs-pooled-pchip']
    assert w['sep2015-jul2016'] == {'holm-p': 1.75e-10, 'delta': 0.0969}
    assert w['all'] == {'delta-min': 0.0203, 'delta-max': 0.1365, 'holm-p-max': 0.00195}
    assert r['heldout-joint']['all'] == {'median': 0.06, 'within-bin': 0.91}
    assert r['heldout-dense-pchip']['all']['median'] == 0.035
    assert r['lopo']['sep2015-jan2016']['ticr-lopo'] == 0.755 and len(r['lopo']) == 10
    assert r['star']['aug2015-jan2016'] == {'ticr-unaligned': 0.578, 'ticr-star': 0.656, 'ticr-joint': 0.672}
    assert r['band']['all'] == {'width-min': 0.044, 'width-max': 0.054}
    assert r['band']['pooled'] == {'spearman': -0.4}
    assert r['coverage'] == {'bootstrap': 0.154, 'predictive': 0.809, 'predictive-no-gross': 0.811}


def test_report03(ex):
    r = ex.report03(ex.TXT03.read_text(), ['sep2015-jul2016', 'sep2015-apr2016', 'jan2016-apr2016',
                                           'jan2016-jul2016'])
    assert r['bench-bframe-joint']['mean-drifted4'] == {'mean': 0.601}
    assert r['bench-bframe-cow-tic']['mean-drifted4'] == {'mean': 0.539}
    assert r['bench-midpoint-neural-joint']['mean-drifted4'] == {'mean': 0.673}
    assert r['bench-midpoint-pchip-cow-refine']['sep2015-apr2016'] == {'mean': 0.721}
    assert r['bench-native-warptransformer']['drifted4'] == {'min': 0.538, 'max': 0.641}
    wt = r['bench-wilcoxon-b-frame-joint-vs-warptransformer']
    assert wt['all'] == {'delta-min': 0.057, 'delta-max': 0.123, 'p-max': 0.002}
    with pytest.raises(ValueError, match='unexpected columns'):
        ex.report03(ex.TXT03.read_text().replace('Neural joint', 'Neural'), [])


def test_report_results(ex, tmp_path):
    # results/*.npz are not in git (CI has only the reports): a synthetic npz for the TIC r part
    npz = _npz(tmp_path / 'm.npz', ex, unaligned=[0.05, 0.8])
    out = ex.report_results(npz)
    assert out['ticr-gaps']['metrics']['joint-minus-pairwise']['mean-drifted'] == pytest.approx(0.2)
    assert out['recovery']['metrics']['synthetic'] == {'anchors': 3811, 'lambda': 100}
    assert out['cycle']['source']['path'] == 'figs/paper/numbers.json'
    assert out['wilcoxon-joint-vs-pairwise']['source']['path'] == 'results/multibatch_align.txt'
    # every result names the commits it was computed at
    prov01 = out['wilcoxon-joint-vs-pairwise']['source']['provenance']
    assert any(p.startswith('joint-align ') and ' @ ' in p for p in prov01)
    assert prov01[-1].startswith('environment:')
    assert out['bench-bframe-joint']['source']['provenance'] == ex.provenance(ex.TXT03.read_text())
    assert out['cycle']['source']['provenance']['script'] == 'scripts/04_paper_figures.py'
    assert out['ticr-gaps']['source']['provenance'] == out['pairs']['source']['provenance']
    assert all(r['source'].get('provenance') for r in out.values())
    names = {p.stem for p in ex.main(npz, out=tmp_path / 'out')}
    assert names == set(out) and len(names) > 50
    with pytest.raises(ValueError, match='not a batch pair'):
        ex._pair('sep2015-jul2016')
