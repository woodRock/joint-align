"""Unit tests for the helper functions inside the numbered scripts."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import multibatch as mb
from conftest import M, T, library_positions, load_script, make_chroma, make_library

GRID = np.arange(T, dtype=float)


@pytest.fixture(scope='module')
def s30():
    return load_script('30_multibatch_align.py')


@pytest.fixture(scope='module')
def s32():
    return load_script('32_fair_benchmark.py')


@pytest.fixture(scope='module')
def s17():
    return load_script('17_literature_baselines.py')


@pytest.fixture(scope='module')
def s16():
    return load_script('16_align_testtime_calib.py')


@pytest.fixture(scope='module')
def s24():
    return load_script('24_pchip_pooled_anchors.py')


def _tic(pos, h=None, sigma=2.0):
    h = np.ones(len(pos)) if h is None else h
    return (h[None] * np.exp(-0.5 * ((GRID[:, None] - np.asarray(pos)[None]) / sigma) ** 2)).sum(1)


# irregular spacing, so a shifted copy has no aliased (periodic) alternative alignment
PEAKS = np.array([22.0, 31.0, 47.0, 55.0, 71.0, 84.0, 90.0, 108.0, 121.0, 133.0, 150.0, 163.0, 177.0])


# ── script 30 ─────────────────────────────────────────────────────────────────

def test_s30_short(s30):
    assert s30.short('batch_jan2016') == 'jan2016'


def test_s30_tic_r_matches_corrcoef(s30):
    rng = np.random.default_rng(0)
    A = [rng.random(T) for _ in range(3)]
    B = [rng.random(T) for _ in range(2)]
    r = s30.tic_r(A, B)
    assert r.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            assert r[i, j] == pytest.approx(np.corrcoef(A[i], B[j])[0, 1])


def test_s30_warp_tics(s30):
    t = _tic(PEAKS)
    out = s30.warp_tics([t], GRID + 5)[0]
    np.testing.assert_allclose(out[:-5], t[5:])
    np.testing.assert_allclose(out[-5:], t[-1])                      # clipped at the run end


def test_s30_assign_folds_balanced(s30):
    folds = s30.assign_folds([10, 3, 7], np.random.default_rng(0))
    assert [len(f) for f in folds] == [10, 3, 7]
    assert np.bincount(folds[0], minlength=5).tolist() == [2, 2, 2, 2, 2]
    assert sorted(folds[1].tolist()) == [0, 1, 2]
    assert np.bincount(folds[2], minlength=5).max() == 2


def test_s30_anchor_fold_masks(s30):
    anc = mb.Anchors(ba=np.array([0, 0, 1]), bb=np.array([1, 2, 2]), sa=np.array([0, 1, 0]),
                     sb=np.array([1, 0, 0]), ta=np.zeros(3), tb=np.zeros(3))
    folds = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
    tr, te = s30.anchor_fold_masks(anc, folds, 3)
    np.testing.assert_array_equal(te, [True, False, False])          # sample (1, 1) is in fold 3
    np.testing.assert_array_equal(tr, ~te)
    tr, te = s30.anchor_fold_masks(anc, folds, 4)
    np.testing.assert_array_equal(te, [False, True, True])


def test_s30_anchor_errors(s30):
    knots = np.linspace(0, T - 1, mb.N_KNOTS)
    drift = np.stack([np.zeros_like(knots), np.full_like(knots, -4.0), np.full_like(knots, 3.0)])
    model = mb.JointWarp(knots, drift, np.full((3, 3), np.nan), np.ones(1))
    ta = np.array([50.0, 60.0, 70.0])
    anc = mb.Anchors(ba=np.array([0, 0, 1]), bb=np.array([1, 2, 2]), sa=np.zeros(3, int),
                     sb=np.zeros(3, int), ta=ta, tb=np.array([54.0, 57.0, 63.0 + 1.0]))
    np.testing.assert_allclose(s30.anchor_errors(model, anc), [0.0, 0.0, -1.0], atol=1e-3)


def test_s30_boot_weights(s30):
    anc = mb.Anchors(ba=np.array([0, 0, 1]), bb=np.array([1, 1, 2]), sa=np.array([0, 1, 2]),
                     sb=np.array([0, 2, 1]), ta=np.zeros(3), tb=np.zeros(3))
    sizes = [2, 3, 4]
    w = s30.boot_weights(anc, sizes, np.random.default_rng(0))
    cnt = [np.random.default_rng(0).multinomial(n, np.full(n, 1 / n)) for n in []]  # noqa: F841
    rng = np.random.default_rng(0)
    c = [rng.multinomial(n, np.full(n, 1.0 / n)) for n in sizes]
    np.testing.assert_array_equal(w, [c[0][0] * c[1][0], c[0][1] * c[1][2], c[1][2] * c[2][1]])
    # the expected count of a sample pair is 1
    W = np.mean([s30.boot_weights(anc, sizes, np.random.default_rng(s)) for s in range(4000)], 0)
    np.testing.assert_allclose(W, 1.0, atol=0.08)


@pytest.mark.parametrize('mod', ['30_multibatch_align.py', '24_pchip_pooled_anchors.py'])
def test_holm(mod):
    holm = load_script(mod).holm
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
    np.testing.assert_allclose(holm([0.5, 0.6]), [1.0, 1.0])
    assert holm([]) == []


def test_s30_composed(s30):
    knots = np.linspace(0, T - 1, mb.N_KNOTS)
    mk = lambda d: mb.JointWarp(knots, np.stack([np.full_like(knots, x) for x in d]),
                                np.full((3, 3), np.nan), np.ones(1))
    m_ah, m_hb = mk([0.0, -2.0, 0.0]), mk([0.0, 0.0, 5.0])     # a=0 → h=1, h=1 → b=2
    c = s30.Composed(m_ah, 0, 1, m_hb, 2)
    t = np.linspace(40, 150, 10)
    np.testing.assert_allclose(c.map(0, 2, t), t + 2 - 5, atol=1e-3)
    np.testing.assert_allclose(c.map(2, 0, c.map(0, 2, t)), t, atol=1e-3)
    with pytest.raises(ValueError):
        c.map(0, 1, t)


# ── script 32 ─────────────────────────────────────────────────────────────────

def test_s32_corr(s32):
    x = np.random.default_rng(0).random(50)
    assert s32.corr(x, 2 * x + 1) == pytest.approx(1.0)
    assert s32.corr(x, np.ones(50)) == 0.0


def test_s32_sample_and_strict(s32):
    t = _tic(PEAKS)
    np.testing.assert_allclose(s32.sample(t, GRID + 0.5)[:-1], 0.5 * (t[:-1] + t[1:]))
    w = np.array([0.0, 2.0, 1.0, 1.0, 5.0])
    s = s32.strict(w)
    assert np.all(np.diff(s) > 0) and s[1] == pytest.approx(2.0, abs=1e-5)


def test_s32_score_identity_and_shift(s32):
    tb = _tic(PEAKS)
    p1, p2 = s32.score(tb, tb, GRID.copy())
    assert p1 == pytest.approx(1.0) and p2 == pytest.approx(1.0)
    ta = _tic(PEAKS + 6)                                              # A elutes 6 bins later
    p1, p2 = s32.score(ta, tb, GRID + 6)                              # correct map B → A
    assert p1 > 0.999 and p2 > 0.999
    u1, u2 = s32.score(ta, tb, GRID.copy())
    assert u1 < 0.2 and u2 < 0.2


def test_s32_score_midpoint_is_symmetric(s32):
    """Midpoint r of (A, B, m) equals midpoint r of (B, A, m⁻¹)."""
    ta, tb = _tic(PEAKS * 1.02 + 2), _tic(PEAKS)
    m = GRID * 1.02 + 2
    minv = np.interp(GRID, m, GRID)
    _, ab = s32.score(ta, tb, m)
    _, ba = s32.score(tb, ta, minv)
    assert ab == pytest.approx(ba, abs=0.01)


def test_s32_cow_map(s32, s17):
    tb = _tic(PEAKS)
    ta = _tic(PEAKS + 5)
    m = s32.cow_map(s17, ta, tb, 4, 22)
    inner = (GRID > 40) & (GRID < 160)
    np.testing.assert_allclose(m[inner], GRID[inner] + 5, atol=1.5)

    class Failing:
        @staticmethod
        def cow_tic_align(q, r, n_segs, slack):
            return q, 0.0
    np.testing.assert_array_equal(s32.cow_map(Failing, ta, tb, 4, 22), GRID)


# ── script 17 ─────────────────────────────────────────────────────────────────

def test_s17_pearson_fast(s17):
    x = np.random.default_rng(0).random(30)
    assert s17._pearson_fast(x, 3 * x) == pytest.approx(1.0)
    assert s17._pearson_fast(x, np.ones(30)) == 0.0


def test_s17_cow_identity(s17):
    t = _tic(PEAKS)
    warped, mag, wf = s17.cow_tic_align(t, t)
    np.testing.assert_allclose(warped, t, rtol=1e-5)
    assert mag == 0.0
    np.testing.assert_allclose(wf(GRID * s17.BIN_MIN), GRID * s17.BIN_MIN)


def test_s17_cow_recovers_shift(s17):
    ref, q = _tic(PEAKS), _tic(PEAKS + 8)
    warped, mag, wf = s17.cow_tic_align(q, ref)
    # end nodes are pinned, so a constant shift stretches the outer segments: r is only partly fixed
    assert np.corrcoef(warped, ref)[0, 1] > np.corrcoef(q, ref)[0, 1] + 0.5
    assert mag > 0
    inner = (GRID > 40) & (GRID < 160)
    np.testing.assert_allclose(wf(GRID * s17.BIN_MIN)[inner] / s17.BIN_MIN, GRID[inner] + 8, atol=1.5)


def test_s17_icoshift(s17):
    ref, q = _tic(PEAKS), _tic(PEAKS + 3)
    warped, mag, wf = s17.icoshift_align(q, ref)
    assert np.corrcoef(warped, ref)[0, 1] > np.corrcoef(q, ref)[0, 1]
    # intervals that contain peaks shift by 3 bins; peak-free intervals may stay at 0
    lag = (np.array([wf(np.array([t]))[0] for t in (s17.TIME_AXIS[[30, 50, 90, 130, 150]])])
           - s17.TIME_AXIS[[30, 50, 90, 130, 150]]) / s17.BIN_MIN
    np.testing.assert_allclose(lag, 3.0, atol=0.01)               # ref time → query time
    assert 0 < mag <= 3 * s17.BIN_MIN + 1e-9
    w0, m0, _ = s17.icoshift_align(ref, ref)
    np.testing.assert_allclose(w0, ref, rtol=1e-6)
    assert m0 == 0.0


def test_s17_evaluate_pairwise_and_silhouette(s17, library):
    p = PEAKS
    lib = library[:len(p)]
    A = [make_chroma(lib, p + 5, seed=i) for i in range(3)]
    B = [make_chroma(lib, p, seed=10 + i) for i in range(3)]
    stats, cors, mags = s17.evaluate_pairwise(A, B, lambda q, r: s17.cow_tic_align(q, r))
    assert len(cors) == len(mags) == 9
    unal = np.mean([np.corrcoef(a.sum(1), b.sum(1))[0, 1] for a in A for b in B])
    assert stats['tic_r'] > unal + 0.3 and stats['warp_mag'] > 0
    assert -1 <= stats['study_sil'] <= 1


def test_s17_apply_warp_2d_shift(s17):
    x = np.random.default_rng(0).random((T, 4)).astype(np.float32)
    y = s17._apply_warp_2d(x, lambda t: t + 2 * s17.BIN_MIN)
    np.testing.assert_allclose(y[:-2], x[2:], rtol=1e-5)
    np.testing.assert_allclose(s17._apply_warp_2d(x, lambda t: t), x, rtol=1e-6)


def test_s17_tee(s17, tmp_path, capsys):
    with open(tmp_path / 'log.txt', 'w') as fh:
        tee = s17._Tee(fh)
        tee.write('hello\n')
        tee.flush()
    assert (tmp_path / 'log.txt').read_text() == 'hello\n'
    assert 'hello' in capsys.readouterr().out


def test_s17_main_runs(s17, tmp_path, monkeypatch, library):
    p = library_positions(len(library))
    for name, shift, n in (('a', 5, 3), ('b', 0, 3)):
        d = tmp_path / name / 'chroma'
        d.mkdir(parents=True)
        for i in range(n):
            np.savez(d / f'{i}.npz', chroma=make_chroma(library, p + shift, seed=20 * n + i + shift))
    monkeypatch.setattr(s17, 'RESULTS_DIR', tmp_path / 'results')
    s17.main(tmp_path / 'a', tmp_path / 'b', 'a', 'b')
    z = np.load(tmp_path / 'results' / 'per_pair' / 'tic_r_baselines_a_b.npz')
    assert z['cow_tic'].shape == (9,) and z['cow_tic'].mean() > z['unaligned'].mean()


# ── script 16 ─────────────────────────────────────────────────────────────────

def test_s16_extract_anchors_orientation(s16, library):
    """Anchors are (A bin, B bin): B elutes 6 bins later → q − r ≈ +6."""
    p = library_positions(len(library))
    A = [make_chroma(library, p, seed=i) for i in range(3)]
    B = [make_chroma(library, p + 6, seed=10)]
    anc = s16.extract_anchors(A, B)
    assert len(anc) >= 8
    d = np.array([q - r for r, q in anc])
    assert np.median(d) == 6
    assert np.all(np.abs(d - 6) <= 1)                    # ±1: refinement tie-break on flat cosine
    assert all(isinstance(r, int) and isinstance(q, int) for r, q in anc)
    assert [r for r, _ in anc] == sorted(r for r, _ in anc)


def test_s16_extract_anchors_empty(s16):
    flat = [np.ones((T, 10), np.float32)]
    assert s16.extract_anchors(flat, flat) == []


def test_s16_extract_anchors_filters_outlier(s16, monkeypatch):
    fake = [(0, 1, 0, 0, 0, [(50, 55), (80, 85), (110, 115), (140, 100)])]
    monkeypatch.setattr(s16, 'align_pair', lambda *a, **k: fake[0])
    out = s16.extract_anchors([None], [None], n_ref=1)
    assert (140, 100) not in out and len(out) == 3


def test_s16_anchor_loss(s16):
    ids = torch.tensor([0, 0, 1])
    w = torch.arange(T, dtype=torch.float32).repeat(3, 1)
    w[2] += 4
    dev = torch.device('cpu')
    assert float(s16.anchor_loss(w, ids, [(50, 54), (100, 104)], dev)) == pytest.approx(0.0)
    assert float(s16.anchor_loss(w, ids, [(50, 52)], dev)) == pytest.approx(4.0)
    assert float(s16.anchor_loss(w, ids, [], dev)) == 0.0


def test_s16_metrics(s16, library):
    p = library_positions(len(library))
    A = [make_chroma(library, p, seed=i) for i in range(2)]
    B = [make_chroma(library, p + 0.5, seed=5 + i) for i in range(3)]
    r = s16.tic_r_per_pair(A, B)
    assert r.shape == (6,)
    assert r[1] == pytest.approx(np.corrcoef(A[0].sum(1), B[1].sum(1))[0, 1], rel=1e-5)  # i-major
    assert s16.tic_correlations(A, B) == pytest.approx(r.mean())
    assert -1 <= s16.study_silhouette(A, B) <= 1
    assert s16.pchip_warp_magnitude(None) == 0.0
    assert s16.pchip_warp_magnitude(lambda t: t + 1.0) == pytest.approx(1.0)


def test_s16_pairwise_helpers(s16, library):
    p = library_positions(len(library))
    A = [make_chroma(library, p + 5, seed=i) for i in range(2)]
    B = [make_chroma(library, p, seed=5 + i) for i in range(2)]
    out, mags = s16.align_pairwise_cosine(A, B)
    assert len(out) == 2 and all(m > 0 for m in mags)
    rs, mags = s16.pchip_all_pairs(A, B)
    assert rs.shape == (4,) and rs.mean() > 0.9


def test_s16_set_seed_reproducible(s16):
    s16.set_seed(3)
    a = (np.random.rand(), torch.rand(1).item())
    s16.set_seed(3)
    assert a == (np.random.rand(), torch.rand(1).item())


def test_s16_pick_device(s16):
    assert s16.pick_device('cpu') == torch.device('cpu')
    assert s16.pick_device('auto').type in ('cpu', 'cuda', 'mps')


def test_s16_calibrate_reduces_anchor_loss(s16, monkeypatch, capsys):
    import warp_transformer as wt
    monkeypatch.setattr(s16, 'N_CALIB_STEPS', 50)
    torch.manual_seed(0)
    model = wt.ChromaWarpTransformer(mz_bins=M, n_bins=T, d_model=16, n_heads=2, n_layers=1,
                                     dropout=0.0)
    lib = make_library(12)
    p = library_positions(12)
    X = torch.tensor(np.stack([make_chroma(lib, p, seed=i) for i in range(2)]
                              + [make_chroma(lib, p + 5, seed=9 + i) for i in range(2)]))
    ids = torch.tensor([0, 0, 1, 1])
    anchors = [(int(r), int(r) + 5) for r in p[2:-2]]
    dev = torch.device('cpu')
    with torch.no_grad():
        before = float(s16.anchor_loss(model(X, ids)[1], ids, anchors, dev))
    secs = s16.calibrate(model, X, ids, anchors, dev, seed=0, log_labels=('A', 'B'))
    with torch.no_grad():
        after = float(s16.anchor_loss(model(X, ids)[1], ids, anchors, dev))
    assert secs > 0 and after < before
    assert not model.training
    assert 'step   50/50' in capsys.readouterr().out


# ── script 24 ─────────────────────────────────────────────────────────────────

def test_s24_pooled_pchip_warp(s24):
    assert s24.pooled_pchip_warp([]) == (None, [])
    assert s24.pooled_pchip_warp([(50, 55)])[0] is None
    warp, used = s24.pooled_pchip_warp([(100, 104), (50, 55), (150, 153), (120, 90)])
    assert used == [(50, 55), (100, 104), (150, 153)]                 # (120, 90) breaks monotone
    # maps B time → A time
    assert float(warp(55 * s24.BIN_MIN)) == pytest.approx(50 * s24.BIN_MIN)
    assert float(warp(0.0)) == 0.0 and float(warp(s24.RUN_MIN)) == pytest.approx(s24.RUN_MIN)


def test_s24_per_a(s24):
    np.testing.assert_allclose(s24.per_a(np.arange(6.0), 2, 3), [1.0, 4.0])


# ── script 13 ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def s13():
    pytest.importorskip('h5py')
    return load_script('13_train_warp_transformer.py')


def test_s13_random_positions(s13):
    pos = s13._random_positions(20, np.random.default_rng(0))
    assert len(pos) == 20 and np.all(np.diff(pos) >= 6)
    assert pos.min() >= 10 and pos.max() < T - 10


def test_s13_drifted_positions_monotone(s13):
    rng = np.random.default_rng(0)
    ref = s13._random_positions(25, rng)
    out = s13._drifted_positions(ref, -20, 4, rng)
    assert np.all(np.diff(out) >= 0) and out.min() >= 2 and out.max() <= T - 3


def test_s13_build_chromatogram(s13):
    rng = np.random.default_rng(0)
    comps = np.abs(rng.normal(size=(3, s13.MZ_BINS))).astype(np.float32)
    x = s13._build_chromatogram(comps, np.array([20, 100, 197]), np.ones(3, np.float32), rng)
    assert x.shape == (T, s13.MZ_BINS) and x.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, rtol=1e-4)


def test_s13_generate_mini_dataset(s13):
    rng = np.random.default_rng(0)
    mona = np.abs(rng.normal(size=(100, s13.MZ_BINS))).astype(np.float32)
    X, W, ids = s13._generate_mini_dataset(mona, 6, rng)
    assert X.shape == (6, T, s13.MZ_BINS) and W.shape == (6, T)
    np.testing.assert_array_equal(ids, [0, 0, 0, 1, 1, 1])
    assert np.all(np.diff(W, axis=1) >= 0) and W.min() >= 0 and W.max() <= T - 1


def test_s13_load_mona(s13, tmp_path):
    import h5py
    specs = np.abs(np.random.default_rng(0).normal(size=(4, 10))).astype(np.float32)
    specs[3] = 0
    with h5py.File(tmp_path / 'm.h5', 'w') as f:
        f['spectra'] = specs
        f['split'] = np.array([b'train', b'test', b'train', b'train'])
    out = s13.load_mona(tmp_path / 'm.h5')
    assert out.shape == (3, 10)
    np.testing.assert_allclose(np.linalg.norm(out[:2], axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(out[2], 0.0)


# ── preprocessing ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def pre():
    pytest.importorskip('tqdm')
    return load_script('preprocess_fish_oil_batches.py')


@pytest.mark.parametrize('stem,expected', [
    ('4 - BCO4 Frame A', ('BCO4', 'Frame')),
    ('5 - bco5 Gonad Female C', ('BCO5', 'Gonad')),
    ('12 - SNA1 Rest of Guts B', ('SNA1', 'Rest of Guts')),
    ('3 - GUR2 ROG A', ('GUR2', 'Rest of Guts')),
    ('7 - BCO6 Liver Male', ('BCO6', 'Liver')),
    ('random file', None),
    ('4 - BCO4 Fin A', None),
])
def test_parse_filename(pre, stem, expected):
    assert pre._parse_filename(stem) == expected


def _raw_file(path, date='160112', blank=False, scans=((1.0, 300.0, [(100.0, 50.2), (200.0, 73.0)]),
                                                         (1.1, 50.0, [(50.0, 50.0), (80.0, 49.6)]))):
    lines = [f'comment1 = {date}' + (' BLANK' if blank else ' fish'), 'other = 1']
    for k, (rt, tic, pk) in enumerate(scans):
        lines += [f'ScanHeader # {k}', f'start_time = {rt}, end_time = {rt + 0.05}',
                  f'num_readings = {len(pk)}, integ_intens = {tic}', 'DataPeaks', '']
        lines += [f'Packet # {i}, intensity = {a}, mass/position = {m}' for i, (a, m) in enumerate(pk)]
    path.write_text('\n'.join(lines) + '\n')
    return path


def test_parse_txt(pre, tmp_path):
    spectra, tic, date = pre.parse_txt(_raw_file(tmp_path / 'x.txt'), mz_max=100)
    assert date == '160112' and spectra.shape == (2, 100)
    np.testing.assert_allclose(tic, [300.0, 50.0])
    assert spectra[0, 50] == 100.0 and spectra[0, 73] == 200.0
    assert spectra[1, 50] == 80.0                        # 50.0 and 49.6 both round to 50 → max


def test_parse_txt_blank_and_empty(pre, tmp_path):
    assert pre.parse_txt(_raw_file(tmp_path / 'b.txt', blank=True)) is None
    assert pre.parse_txt(_raw_file(tmp_path / 'e.txt', scans=())) is None
    (tmp_path / 'n.txt').write_text('ScanHeader # 0\n')
    assert pre.parse_txt(tmp_path / 'n.txt') is None


def test_bin_chromatogram(pre):
    spectra = np.arange(10, dtype=np.float32)[:, None].repeat(3, 1)
    tic = np.array([1, 5, 2, 9, 0, 0, 3, 1, 8, 2], dtype=np.float32)
    out = pre.bin_chromatogram(spectra, tic, 5)
    np.testing.assert_array_equal(out[:, 0], [1, 3, 4, 6, 8])
    out = pre.bin_chromatogram(spectra[:3], tic[:3], 5)              # fewer scans than bins
    assert out.shape == (5, 3)


def test_sqrt_l2(pre):
    out = pre.sqrt_l2(np.array([[4.0, 0.0, -1.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_array_equal(out[1], 0.0)


def test_preprocess_main(pre, tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    scans = tuple((0.1 * k, float(k % 7), [(float(1 + k), 40.0 + k % 5)]) for k in range(30))
    _raw_file(raw / '4 - BCO5 Frame A.txt', scans=scans)
    _raw_file(raw / '4 - BCO5 Skin B.txt', scans=scans)
    _raw_file(raw / '5 - BCO4 Head A.txt', date='150807', scans=scans)
    _raw_file(raw / '6 - BCO4 Head B.txt', date='999999', scans=scans)        # unknown date
    _raw_file(raw / '7 - BCO4 Head C.txt', blank=True, scans=scans)
    _raw_file(raw / 'unparseable.txt', scans=scans)
    out = tmp_path / 'out'
    (out / 'chroma').mkdir(parents=True)                                      # stale flat layout
    pre.main(raw, out, n_bins=10, mz_max=60)
    jan = out / 'batch_jan2016'
    files = sorted((jan / 'chroma').glob('*.npz'))
    assert [f.name for f in files] == ['4___BCO5_Frame_A.npz', '4___BCO5_Skin_B.npz']
    c = np.load(files[0])['chroma']
    assert c.shape == (10, 60)
    np.testing.assert_array_equal(np.load(jan / 'y.npy'), [0, 5])
    assert (jan / 'groups.txt').read_text().splitlines() == ['BCO5', 'BCO5']
    import json
    man = json.loads((out / 'batch_manifest.json').read_text())
    assert set(man) == {'batch_aug2015', 'batch_jan2016'}
    assert man['batch_jan2016'] == {'n_samples': 2, 'tissues': {'Frame': 1, 'Skin': 1},
                                    'species': ['BCO5']}
    pre.clean_old_flat(out)
    assert not (out / 'chroma').exists()
