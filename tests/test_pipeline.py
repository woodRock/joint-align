"""
End-to-end smoke tests: scripts 30, 31 and 32 on five small synthetic batches with known
shifts, written to a temporary directory. Marked slow (about a minute together).
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import load_script, make_chroma, make_library, T

pytestmark = pytest.mark.slow

# batch → (n samples, shift in bins); same names as the fish oil batches
BATCHES = {'batch_aug2015': (2, 5.0), 'batch_sep2015': (6, 0.0), 'batch_jan2016': (6, 1.0),
           'batch_apr2016': (4, -6.0), 'batch_jul2016': (3, -7.0)}
PEAKS = np.array([22.0, 31.0, 40.0, 47.0, 55.0, 63.0, 71.0, 84.0, 90.0, 99.0, 108.0, 114.0, 121.0,
                  133.0, 141.0, 150.0, 163.0, 170.0, 177.0])


@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp('ws')
    lib = make_library(len(PEAKS), 80, seed=7)
    rng = np.random.default_rng(7)
    for k, (name, (n, shift)) in enumerate(BATCHES.items()):
        d = root / 'data' / name / 'chroma'
        d.mkdir(parents=True)
        for i in range(n):
            jitter = rng.normal(0, 0.3, len(PEAKS))
            x = make_chroma(lib, PEAKS + shift + jitter, heights=rng.uniform(0.5, 1.5, len(PEAKS)),
                            seed=100 * k + i)
            np.savez(d / f'{i:02d}.npz', chroma=x, tic_raw=x.sum(1))   # gcmsalign layout
    (root / 'results').mkdir()
    (root / 'figs').mkdir()
    return root


@pytest.fixture(scope='module')
def run30(workspace):
    s30 = load_script('30_multibatch_align.py')
    mp = pytest.MonkeyPatch()
    mp.setattr(s30, 'DATA', workspace / 'data')
    mp.setattr(s30, 'RESULTS', workspace / 'results')
    mp.setattr(s30, 'FIGS', workspace / 'figs')
    mp.setattr(s30, 'LAMBDAS', [1.0, 100.0])
    try:
        s30.main(n_boot=4, n_boot_cv=3, seed=0)
    finally:
        mp.undo()
    return workspace


def test_script30_outputs(run30):
    res = run30 / 'results'
    txt = (res / 'multibatch_align.txt').read_text()
    assert 'gcmsalign 2' in txt and '@' in txt                     # provenance line
    for section in ('1. Hyper-parameters', '2. Mean between-batch TIC r', '3. Cross-fitted',
                    '4. Missing correspondences', '5. Cycle inconsistency', '6. Uncertainty',
                    'Wall time'):
        assert section in txt
    assert (run30 / 'figs' / 'multibatch_warps.png').stat().st_size > 0
    assert (res / 'anchors_all_pairs.npz').exists()
    z = np.load(res / 'multibatch_align.npz')
    assert float(z['lam']) in (1.0, 100.0)
    assert z['boot_drift'].shape[0] == 4
    names = list(z['names'])
    assert names == sorted(BATCHES)


def test_script30_joint_aligns_synthetic_batches(run30):
    z = np.load(run30 / 'results' / 'multibatch_align.npz')
    # sep2015 → jul2016 is shifted by 7 bins: joint warp must fix it, unaligned must not
    unal = z['ticr_sep2015_jul2016_Unaligned'].mean()
    joint = z['ticr_sep2015_jul2016_Joint'].mean()
    cf = z['ticr_sep2015_jul2016_Joint_CF'].mean()
    # random peak heights cap between-sample TIC r near 0.85 even when perfectly aligned
    assert unal < 0.5 and joint > 0.75 and cf > 0.75
    # recovered drifts match the simulated shifts (reference gauge)
    names = list(z['names'])
    ref = int(z['ref'])
    inner = (z['knots'] > 30) & (z['knots'] < 170)
    for k, n in enumerate(names):
        expected = BATCHES[names[ref]][1] - BATCHES[n][1]              # f_k(t) = t + d_k
        np.testing.assert_allclose(np.median(z['drift'][k][inner]), expected, atol=0.6)


def test_script32_runs_on_script30_outputs(run30):
    s32 = load_script('32_fair_benchmark.py')
    res = run30 / 'results'
    # a cached WarpTransformer result for one pair (identity warps) exercises that branch
    na, nb = BATCHES['batch_sep2015'][0], BATCHES['batch_jul2016'][0]
    (res / 'wt_warps').mkdir()
    np.savez(res / 'wt_warps' / 'batch_sep2015__batch_jul2016.npz',
             warps=np.tile(np.arange(T, dtype=float), (na + nb, 1)), r_native=np.zeros(na * nb))
    mp = pytest.MonkeyPatch()
    mp.setattr(s32, 'DATA', run30 / 'data')
    mp.setattr(s32, 'RESULTS', res)
    try:
        s32.main(neural_steps=3)
        s32.main(neural_steps=3)                  # second run hits the neural + COW caches
    finally:
        mp.undo()
    txt = (res / 'fair_benchmark.txt').read_text()
    assert 'B-frame' in txt and 'Midpoint' in txt and 'Native protocols' in txt
    assert txt.splitlines()[2].startswith('gcmsalign ')
    assert (res / 'neural_joint_G_3.npz').exists()
    assert len(list((res / 'cow_tic_maps').glob('*.npy'))) == 10
    z = np.load(res / 'fair_benchmark.npz')
    j = z['sep2015_jul2016|Joint|B-frame']
    u = z['sep2015_jul2016|Unaligned|B-frame']
    assert j.shape == (na, nb) and j.mean() > 0.75 and u.mean() < 0.5
    np.testing.assert_allclose(z['sep2015_jul2016|WarpTransformer|B-frame'], u, atol=1e-6)
    assert 'aug2015_apr2016|WarpTransformer|B-frame' not in z.files


def test_script31_skips_cached_pairs(tmp_path, monkeypatch, capsys):
    s31 = load_script('31_wt_warps.py')
    out = tmp_path / 'results' / 'wt_warps'
    out.mkdir(parents=True)
    for a, b in s31.PAIRS:
        (out / f'{a}__{b}.npz').write_bytes(b'')
    monkeypatch.setattr(s31, 'ROOT', tmp_path)
    s31.main(seed=0, device='cpu')
    assert capsys.readouterr().out.count('cached') == len(s31.PAIRS)
