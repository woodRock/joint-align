"""
Integrity checks on the real fish oil data and the cached results the README reports.
These catch stale caches (results computed from older code or data) before they are
written up. Skipped when the data or results are not present.
"""
from __future__ import annotations

import json
import re
from itertools import combinations

import numpy as np
import pytest

import multibatch as mb
from alignment import load_chromas
from conftest import DATA, RESULTS, ROOT, requires_data

pytestmark = requires_data


def _need(path):
    if not path.exists():
        pytest.skip(f'{path.relative_to(ROOT)} not present')
    return path


@pytest.fixture(scope='module')
def names():
    return sorted(p.name for p in DATA.glob('batch_*') if p.is_dir())


@pytest.fixture(scope='module')
def chroma(names):
    return [load_chromas(DATA / b / 'chroma') for b in names]


@pytest.fixture(scope='module')
def anchors():
    z = np.load(_need(RESULTS / 'anchors_all_pairs.npz'))
    return mb.Anchors(*(z[k] for k in ('ba', 'bb', 'sa', 'sb', 'ta', 'tb')))


def test_chromatograms_well_formed(names, chroma):
    manifest = json.loads((DATA / 'batch_manifest.json').read_text())
    assert set(names) == set(manifest)
    for b, cs in zip(names, chroma):
        assert len(cs) == manifest[b]['n_samples']
        for c in cs:
            assert c.shape == (200, 1000) and c.dtype == np.float32
            assert np.isfinite(c).all() and (c >= 0).all() and c.sum() > 0


def test_anchor_cache_indices_valid(anchors, chroma):
    sizes = [len(c) for c in chroma]
    assert np.all(anchors.ba < anchors.bb) and anchors.bb.max() < len(chroma)
    for k in range(len(chroma)):
        assert anchors.sa[anchors.ba == k].max(initial=-1) < sizes[k]
        assert anchors.sb[anchors.bb == k].max(initial=-1) < sizes[k]
    assert np.all((anchors.ta >= 0) & (anchors.ta <= 199) & (anchors.tb >= 0) & (anchors.tb <= 199))


def test_anchor_cache_matches_current_code(anchors, chroma):
    """Recompute a sample of sample pairs: the cache must equal what match_anchors gives now."""
    rng = np.random.default_rng(0)
    pairs = [(a, b) for a, b in combinations(range(len(chroma)), 2)]
    for a, b in pairs:
        i, j = rng.integers(len(chroma[a])), rng.integers(len(chroma[b]))
        m = (anchors.ba == a) & (anchors.bb == b) & (anchors.sa == i) & (anchors.sb == j)
        fresh = mb.match_anchors(chroma[a][i], chroma[b][j])
        np.testing.assert_allclose(np.c_[anchors.ta[m], anchors.tb[m]], fresh, atol=1e-9,
                                   err_msg=f'stale anchor cache for batches {a}×{b}, samples {i}×{j}')


def test_script30_config_consistent(anchors, names):
    z = np.load(_need(RESULTS / 'multibatch_align.npz'))
    assert list(z['names']) == names
    s30_lambdas = [0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    assert float(z['lam']) in s30_lambdas
    K = len(names)
    n = np.zeros((K, K), int)
    for a, b in combinations(range(K), 2):
        n[a, b] = n[b, a] = len(anchors.pair(a, b))
    np.testing.assert_array_equal(z['n_pair'], n)
    assert int(z['ref']) == int(np.argmax(n.sum(1)))
    # the saved drift is the joint fit at the saved λ and reference
    m = mb.fit_joint(anchors, K, lam=float(z['lam']), gauge=int(z['ref']))
    np.testing.assert_allclose(m.drift, z['drift'], atol=1e-6)


def test_readme_matches_results(anchors):
    """Headline numbers in the README must come from the current results file."""
    readme = (ROOT / 'README.md').read_text()
    txt = _need(RESULTS / 'multibatch_align.txt').read_text()
    assert f'{len(anchors):,}' in readme
    assert f'Anchors: {len(anchors)} ' in txt
    row = lambda label: [float(x) for x in re.search(rf'{label}\s+([-\d. ]+)', txt).group(1).split()]
    drifted, near = row('mean, drifted pairs'), row('mean, near-aligned pairs')
    # columns: Unaligned, Pooled PCHIP, Dense PCHIP, Pairwise smooth, Joint (mean), Joint, PW CF, Joint CF
    for label, i in (('Unaligned', 0), ('Pooled-anchor PCHIP', 1), ('Dense PCHIP', 2),
                     ('Pairwise smooth', 3), (r'\*\*Joint\*\*', 5), ('Joint, cross-fitted', 7)):
        line = re.search(rf'^\| {label}[^|]*\|([^|]+)\|([^|]+)\|', readme, re.M)
        assert line, f'README row {label} missing'
        vals = [float(v.strip().strip('*')) for v in line.groups()]
        assert vals == pytest.approx([drifted[i], near[i]], abs=5e-4), label


def test_wt_cache_shapes(names, chroma):
    d = RESULTS / 'wt_warps'
    files = sorted(d.glob('*.npz')) if d.exists() else []
    if not files:
        pytest.skip('no WarpTransformer cache')
    size = {b: len(c) for b, c in zip(names, chroma)}
    for f in files:
        a, b = f.stem.split('__')
        z = np.load(f)
        na, nb = size[a], size[b]
        assert (int(z['n_a']), int(z['n_b'])) == (na, nb)
        assert z['warps'].shape == (na + nb, 200)
        assert z['r_native'].shape == (na * nb,)
        assert np.all(np.diff(z['warps'], axis=1) >= 0), f'{f.name}: non-monotone warp'


def test_neural_and_cow_caches_shapes(names, chroma):
    sizes = [len(c) for c in chroma]
    g = RESULTS / 'neural_joint_G_1500.npz'
    if g.exists():
        z = np.load(g)
        assert z['G'].shape == (sum(sizes), len(z['taus']))
        assert np.all(np.diff(z['G'], axis=1) > 0)
    d = RESULTS / 'cow_tic_maps'
    for f in (sorted(d.glob('*.npy')) if d.exists() else []):
        a, b = f.stem.split('__')
        assert np.load(f).shape == (sizes[names.index(a)], sizes[names.index(b)], 200)


def test_cached_tic_r_matches_current_data(names, chroma):
    """Unaligned TIC r depends only on the data: a mismatch means results predate the data."""
    z = np.load(_need(RESULTS / 'multibatch_align.npz'))
    tics = {b.replace('batch_', ''): [c.sum(1) for c in cs] for b, cs in zip(names, chroma)}
    stale = []
    for key in z.files:
        m = re.fullmatch(r'ticr_(\w+?)_(\w+?)_Unaligned', key)
        if not m:
            continue
        a, b = m.groups()
        now = np.array([[np.corrcoef(x, y)[0, 1] for y in tics[b]] for x in tics[a]])
        if z[key].shape != now.shape or not np.allclose(z[key], now, atol=1e-4):
            stale.append(f'{a}×{b}: cached {z[key].mean():.3f}, data now {now.mean():.3f}')
    assert not stale, 'results/ were computed on different data:\n  ' + '\n  '.join(stale)
