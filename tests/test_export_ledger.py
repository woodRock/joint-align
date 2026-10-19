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
    first = {p.name: p.read_bytes() for p in ex.main(npz, tmp_path / 'out')}
    second = {p.name: p.read_bytes() for p in ex.main(npz, tmp_path / 'out')}
    assert first == second and 'pairs.json' in first
    assert json.loads(first['ticr-joint.json'])['name'] == 'ticr-joint'


def test_unknown_method(ex, tmp_path):
    np.savez(tmp_path / 'bad.npz', names=np.array(['batch_apr2016', 'batch_sep2015']),
             n_pair=np.zeros((2, 2), int), provenance=np.array(['x']),
             ticr_sep2015_apr2016_Mystery=np.zeros((1, 1)))
    with pytest.raises(ValueError, match='Mystery'):
        ex.results(tmp_path / 'bad.npz')
