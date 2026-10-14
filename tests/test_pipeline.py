"""
End-to-end runs of scripts 30, 31 and 32 on the public synthetic dataset (jointalign.synthetic),
whose true drifts are known. Marked slow (well under a minute together).
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT, T, load_script
from jointalign import synthetic as syn

pytestmark = pytest.mark.slow

BATCHES = {b.name: b for b in syn.DEFAULT_BATCHES}


@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp('ws')
    syn.make_dataset(root / 'data', seed=0)
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
    assert 'warpalign 2' in txt and '@' in txt                     # provenance line
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
    unal = z['ticr_sep2015_jul2016_Unaligned'].mean()
    joint = z['ticr_sep2015_jul2016_Joint'].mean()
    cf = z['ticr_sep2015_jul2016_Joint_CF'].mean()
    assert unal < 0.2 and joint > 0.7 and cf > 0.7


def test_script30_recovers_true_drifts(run30):
    """The joint warps between every pair of batches match the generator's true drifts."""
    from jointalign import multibatch as mb
    z = np.load(run30 / 'results' / 'multibatch_align.npz')
    names = [str(n) for n in z['names']]
    model = mb.JointWarp(z['knots'], z['drift'], np.full((5, 5), np.nan), np.ones(1))
    truth = json.loads((run30 / 'data' / 'truth.json').read_text())
    t = np.linspace(8.0, 40.0, 40)                                   # minutes, inside every run
    # judge recovery where compounds exist in both batches; beyond the last compound the warp is
    # an extrapolation (a documented limitation), and the map saturates at the run ends
    tau = np.asarray(truth['compound_tau_min'])
    span = {n: tau + BATCHES[n].drift(tau) for n in names}
    for a in range(5):
        for b in range(5):
            if a == b:
                continue
            est = model.map(a, b, t / syn.BIN_MIN) * syn.BIN_MIN
            true = syn.true_map(truth, names[a], names[b], t)
            inside = ((t >= span[names[a]].min()) & (t <= span[names[a]].max())
                      & (true >= span[names[b]].min()) & (true <= span[names[b]].max()))
            assert inside.sum() > 20
            np.testing.assert_allclose(est[inside], true[inside], atol=0.15,
                                       err_msg=f'{names[a]} → {names[b]}')


def test_script32_runs_on_script30_outputs(run30):
    s32 = load_script('32_fair_benchmark.py')
    res = run30 / 'results'
    # a cached WarpTransformer result for one pair (identity warps) exercises that branch
    na, nb = BATCHES['batch_sep2015'].n, BATCHES['batch_jul2016'].n
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
    assert txt.splitlines()[2].startswith('warpalign ')
    assert (res / 'neural_joint_G_3.npz').exists()
    assert len(list((res / 'cow_tic_maps').glob('*.npy'))) == 10
    z = np.load(res / 'fair_benchmark.npz')
    j = z['sep2015_jul2016|Joint|B-frame']
    u = z['sep2015_jul2016|Unaligned|B-frame']
    assert j.shape == (na, nb) and j.mean() > 0.7 and u.mean() < 0.2
    np.testing.assert_allclose(z['sep2015_jul2016|WarpTransformer|B-frame'], u, atol=1e-6)
    assert 'aug2015_apr2016|WarpTransformer|B-frame' not in z.files

    # the website export runs on these outputs too (script 40)
    s40 = load_script('40_export_site_data.py')
    out = run30 / 'site' / 'site.json'
    mp = pytest.MonkeyPatch()
    mp.setattr(s40, 'RESULTS', res)
    mp.setattr(s40, 'OUT', out)
    try:
        s40.main(run30 / 'data')
    finally:
        mp.undo()
    site = json.loads(out.read_text())
    assert out.with_suffix('.js').read_text().startswith('window.SITE_DATA = {')
    assert {'meta', 'batches', 'tic', 'warps', 'cycle', 'anchors', 'band', 'lopo', 'lambda_sweep',
            'ticr30', 'coverage', 'benchmark'} <= site.keys()
    assert len(site['batches']) == 5 and len(site['cycle']) == 60 and len(site['lopo']) == 10
    assert site['meta']['n_anchors'] == len(np.load(res / 'anchors_all_pairs.npz')['ta'])
    assert str(ROOT.parent) not in out.read_text() and '/Users/' not in site['meta']['warpalign']   # public
    shifts = [v for v in site['warps']['joint']['4-3'] if v is not None]
    assert len(shifts) > 150 and all(np.isfinite(shifts))                # clamped ends are null

    # the paper figures render from the same outputs (script 50)
    s50 = load_script('50_paper_figures.py')
    figs = run30 / 'paper'
    mp = pytest.MonkeyPatch()
    for k, v in (('RESULTS', res), ('OUT', figs), ('DATA', run30 / 'data')):
        mp.setattr(s50, k, v)
    try:
        s50.main(None)
    finally:
        mp.undo()
    names = ['fig1_problem_method', 'fig2_recovery', 'fig3_comparison', 'fig4_consistency', 'fig5_borrowing',
             'fig6_benchmark', 'figS1_lambda', 'figS2_uncertainty', 'figS3_rejection']
    for n in names:
        assert (figs / f'{n}.pdf').stat().st_size > 5_000 and (figs / f'{n}.png').exists(), n
    captions = (figs / 'captions.md').read_text()
    assert captions.count('## fig') == len(names) and 'nan' not in captions.lower()


def test_script31_skips_cached_pairs(tmp_path, monkeypatch, capsys):
    s31 = load_script('31_wt_warps.py')
    out = tmp_path / 'results' / 'wt_warps'
    out.mkdir(parents=True)
    for a, b in s31.PAIRS:
        (out / f'{a}__{b}.npz').write_bytes(b'')
    monkeypatch.setattr(s31, 'ROOT', tmp_path)
    s31.main(seed=0, device='cpu')
    assert capsys.readouterr().out.count('cached') == len(s31.PAIRS)


def test_demo_commands_from_the_docs(tmp_path):
    """The synthetic demo exactly as documented: script 00, then 30 with --data/--results/--figs."""
    run = lambda *args: subprocess.run([sys.executable, *args], cwd=ROOT, check=True,
                                       capture_output=True, text=True, timeout=600)
    data, res, figs = tmp_path / 'data', tmp_path / 'results', tmp_path / 'figs'
    run('scripts/00_make_synthetic_data.py', '--out', str(data))
    assert (data / 'truth.json').exists() and len(list(data.glob('batch_*/chroma/*.npz'))) == 31
    run('scripts/30_multibatch_align.py', '--data', str(data), '--results', str(res), '--figs', str(figs),
        '--n-boot', '2', '--n-boot-cv', '2')
    assert f'Data: {data}' in (res / 'multibatch_align.txt').read_text()
