"""
export_ledger.py — The numbers behind the paper as taura ledger results (ledger/results/*.json).

  results/multibatch_align.npz   mean TIC r of every batch pair and method (A warped into B's frame;
                                 mean over all A_i × B_j, as in section 2 of multibatch_align.txt) and
                                 the means over drifted and near-aligned pairs; anchors per pair
  results/multibatch_align.txt   script 01's report: λ selection and sweep, Wilcoxon tests, held-out
                                 anchor error, leave-one-pair-out and star refits, band widths, coverage
  results/fair_benchmark.txt     script 03's report: both protocols, native protocols, Wilcoxon tests
  figs/paper/numbers.json        script 04's caption numbers, unrounded (drift, recovery, cycles, ...)

Values parsed from the reports carry the precision they were printed with. Ranges, maxima and gaps
the paper quotes are exported as their own keys ("all"/"delta-min", "lam-ge1"/"drifted-max", ...), so
every number in the text is one \\res{} key. Drifted pairs: unaligned TIC r < 0.5 (the cow-dcnn
exclusion rule); script 03 uses its own first four pairs ("drifted4").

Result keys, for \\res{} in the paper: ticr-joint/sep2015-jul2016/mean, ticr-joint/mean-drifted/mean,
pairs/sep2015-jul2016/anchors, wilcoxon-joint-vs-pairwise/all/delta-max, ... The output depends only
on these files, so re-running is a no-op.

Output: ledger/results/*.json
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / 'results' / 'multibatch_align.npz'
TXT01 = ROOT / 'results' / 'multibatch_align.txt'
TXT03 = ROOT / 'results' / 'fair_benchmark.txt'
NUMBERS = ROOT / 'figs' / 'paper' / 'numbers.json'
OUT = ROOT / 'ledger' / 'results'
DRIFTED_BELOW = 0.5   # unaligned mean TIC r; script 01's rule

# npz method suffix → (result name, label as in multibatch_align.txt)
METHODS = {
    'Unaligned':               ('ticr-unaligned', 'Unaligned'),
    'Pooled_PCHIP_(cow-dcnn)': ('ticr-pooled-pchip', 'Pooled PCHIP (cow-dcnn)'),
    'Dense_PCHIP':             ('ticr-dense-pchip', 'Dense PCHIP'),
    'Pairwise_smooth':         ('ticr-pairwise', 'Pairwise smooth'),
    'Joint_(mean_gauge)':      ('ticr-joint-mean-gauge', 'Joint (mean gauge)'),
    'Joint':                   ('ticr-joint', 'Joint'),
    'Joint_CF':                ('ticr-joint-cf', 'Joint CF'),
    'Pairwise_smooth_CF':      ('ticr-pairwise-cf', 'Pairwise smooth CF'),
}
KEY = re.compile(r'^ticr_([a-z]+\d{4})_([a-z]+\d{4})_(.+)$')


def load(npz: Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]], dict[str, int], list[str]]:
    """(pairs in script order, {pair: {method: TIC r matrix}}, {pair: anchors}, provenance)."""
    d = np.load(npz)
    names = [str(n).replace('batch_', '') for n in d['names']]
    pairs: list[str] = []
    R: dict[str, dict[str, np.ndarray]] = {}
    anchors: dict[str, int] = {}
    for k in d.files:   # savez keeps insertion order: script 01's PAIRS
        m = KEY.match(k)
        if not m:
            continue
        a, b, method = m.groups()
        if method not in METHODS:
            raise ValueError(f'{npz.name}: unknown method {method!r} in {k}')
        pair = f'{a}-{b}'
        if pair not in R:
            pairs.append(pair)
            R[pair] = {}
            anchors[pair] = int(d['n_pair'][names.index(a), names.index(b)])
        R[pair][method] = d[k]
    return pairs, R, anchors, [str(p) for p in d['provenance']]


def results(npz: Path) -> dict[str, dict]:
    pairs, R, anchors, provenance = load(npz)
    drifted = [p for p in pairs if R[p]['Unaligned'].mean() < DRIFTED_BELOW]
    near = [p for p in pairs if p not in drifted]
    source = {'path': str(npz.relative_to(ROOT)) if npz.is_relative_to(ROOT) else str(npz),
              'hash': 'sha256:' + hashlib.sha256(npz.read_bytes()).hexdigest(),
              'provenance': provenance}
    common = {'runs': [], 'produced_by': 'scripts/export_ledger.py', 'source': source}

    out = {'pairs': {'name': 'pairs', 'label': 'Batch pairs (A warped into B)',
                     'metrics': {p: {'anchors': anchors[p]} for p in pairs},
                     'groups': {'drifted': drifted, 'near-aligned': near},
                     'rule': f'drifted: unaligned mean TIC r < {DRIFTED_BELOW}', **common}}
    for method, (name, label) in METHODS.items():
        metrics: dict[str, dict] = {p: {'mean': float(R[p][method].mean()), 'n': int(R[p][method].size)}
                                    for p in pairs}
        for group, sel in (('mean-drifted', drifted), ('mean-near-aligned', near)):
            metrics[group] = {'mean': float(np.mean([R[p][method].mean() for p in sel])), 'n': len(sel)}
        out[name] = {'name': name, 'label': f'Mean TIC r, {label}', 'metrics': metrics, **common}
    return out


# ── the reports (scripts 01 and 03) and script 04's numbers ──────────────────

NUM = r'[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?'
PAIR = re.compile(r'([a-z]+\d{4})×([a-z]+\d{4})')


PROVENANCE_LINE = re.compile(r'^(?:\S+ \S+ @ \S+ \(.*\) from .*|environment: .*)$', re.M)


def provenance(txt: str) -> list[str]:
    """The version lines a report starts with (jointalign.provenance.header(): packages @ commit, environment)."""
    return PROVENANCE_LINE.findall(txt[:2000])


def _source(path: Path, prov: list[str] | dict | None = None) -> dict:
    out = {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
           'hash': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()}
    if prov:
        out['provenance'] = prov
    return out


def _pair(text: str) -> str:
    m = PAIR.fullmatch(text)
    if not m:
        raise ValueError(f'not a batch pair: {text!r}')
    return f'{m.group(1)}-{m.group(2)}'


def _section(txt: str, start: str, end: str | None) -> str:
    i = txt.index(start)
    return txt[i:txt.index(end, i)] if end else txt[i:]


def _slug(label: str) -> str:
    label = re.sub(r'\bCOW ref\b', 'COW refine', label.replace('(cow-dcnn)', '')).replace('+', ' ')
    return re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-').replace('pairwise-smooth', 'pairwise')


def _num(v: str) -> int | float:
    return int(v) if re.fullmatch(r'[-+]?\d+', v) else float(v)


def _span(metrics: dict, stat: str) -> dict:
    vals = [m[stat] for m in metrics.values()]
    return {f'{stat}-min': min(vals), f'{stat}-max': max(vals)}


def report01(txt: str) -> dict[str, dict]:
    """Script 01's report → {result: metrics}."""
    out: dict[str, dict] = {}
    m = re.search(r'Anchors: (\d+) from', txt)
    star = re.search(r'Star design: .*?\((\d+) of \d+ anchors\)', txt)
    out['anchors'] = {'all': int(m.group(1)) if m else None, 'star-design': int(star.group(1)) if star else None}

    sec = _section(txt, '1. Hyper-parameters', '2. Mean between-batch')
    cv: dict[str, dict] = {}
    for m in re.finditer(rf'^\s+({NUM})\s+(\S+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$', sec, re.M):
        lam, gauge, *v = m.groups()
        cv[f'{gauge}-lam{float(lam):g}'] = dict(zip(('rt-blocked', 'rt-blocked-within-bin', 'samples',
                                                     'samples-within-bin'), map(float, v)))
    sel = re.search(rf'selected λ = ({NUM}) with gauge = (\S+)', sec)
    cv['selected'] = {'lambda': _num(f'{float(sel.group(1)):g}')}
    out['lambda-cv'] = cv

    sec = _section(txt, 'Post-hoc sensitivity', 'Paired two-sided')
    sweep = {f'lam{float(m.group(1)):g}': {'drifted': float(m.group(2)), 'near-aligned': float(m.group(3))}
             for m in re.finditer(rf'^\s+({NUM})\s+({NUM})\s+({NUM})(?:\s+←.*)?$', sec, re.M)}
    ge1 = [v['drifted'] for k, v in sweep.items() if float(k[3:]) >= 1]
    sweep['lam-ge1'] = {'drifted-min': min(ge1), 'drifted-max': max(ge1)}
    out['lambda-sweep'] = sweep

    sec = _section(txt, 'Paired two-sided Wilcoxon', '3. Cross-fitted')
    for block in re.split(r'\n   (?=\S.*:\n)', sec)[1:]:
        head, _, body = block.partition(':\n')
        tests = {_pair(m.group(1)): {'holm-p': float(m.group(2)), 'delta': float(m.group(3))}
                 for m in re.finditer(rf'^\s+(\S+)\s+Holm p=({NUM})\s+median Δr ({NUM})', body, re.M)}
        tests['all'] = _span(tests, 'delta') | {'holm-p-max': max(t['holm-p'] for t in tests.values())}
        out[f'wilcoxon-{_slug(head)}'] = tests

    sec = _section(txt, '3. Cross-fitted', '4. Missing')
    methods = ['unaligned', 'dense-pchip', 'pairwise', 'joint']
    for m in re.finditer(r'^\s+(\S+)\s+\d+((?:\s+[\d.]+ \([\d.]+\)){4})\s*$', sec, re.M):
        key = 'all' if m.group(1) == 'all' else _pair(m.group(1))
        for meth, (med, within) in zip(methods, re.findall(r'([\d.]+) \(([\d.]+)\)', m.group(2))):
            out.setdefault(f'heldout-{meth}', {})[key] = {'median': float(med), 'within-bin': float(within)}

    sec = _section(txt, '4. Missing', '5. Cycle')
    lopo_cols = ('ticr-unaligned', 'ticr-hub', 'ticr-lopo', 'ticr-joint', 'err-hub', 'err-lopo', 'err-joint')
    for m in re.finditer(rf'^\s+(\S+×\S+)\s+([a-z]+\d{{4}})((?:\s+{NUM}){{7}})\s*$', sec, re.M):
        out.setdefault('lopo', {})[_pair(m.group(1))] = dict(zip(lopo_cols, map(float, m.group(3).split())))
    star_sec = sec[sec.index('Star design'):]
    for m in re.finditer(rf'^\s+(\S+×\S+)\s+({NUM})\s+({NUM})\s+({NUM})\s*$', star_sec, re.M):
        out.setdefault('star', {})[_pair(m.group(1))] = {'ticr-unaligned': float(m.group(2)),
                                                          'ticr-star': float(m.group(3)),
                                                          'ticr-joint': float(m.group(4))}

    sec = _section(txt, '6. Uncertainty', None)
    band = {_pair(m.group(1)): dict(zip(('width', 'width-supported', 'width-unsupported', 'spearman'),
                                        map(float, m.groups()[1:])))
            for m in re.finditer(rf'^\s+(\S+×\S+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$', sec, re.M)}
    band['all'] = _span(band, 'width')
    pooled = re.search(rf'all pairs pooled: Spearman ρ\(width, support\) = ({NUM})', sec)
    band['pooled'] = {'spearman': float(pooled.group(1))}
    out['band'] = band
    epi = re.search(rf'warp-only band \(epistemic\)\s*:\s*({NUM})', sec)
    pred = re.search(rf'predictive \(band ⊕ anchor scatter\)\s*:\s*({NUM}).*?:\s*({NUM})', sec)
    out['coverage'] = {'bootstrap': float(epi.group(1)), 'predictive': float(pred.group(1)),
                       'predictive-no-gross': float(pred.group(2))}
    return out


BENCH_METHODS = ['Unaligned', 'Pooled PCHIP', 'PCHIP + COW ref', 'COW-TIC', 'WarpTransformer', 'Joint',
                 'Neural joint']   # script 03's table columns


def report03(txt: str, drifted4: list[str]) -> dict[str, dict]:
    """Script 03's report → {result: metrics}. drifted4: script 03's drifted group (its first four pairs)."""
    out: dict[str, dict] = {}
    groups = {'mean drifted (cow-dcnn 4)': 'mean-drifted4', 'mean near-aligned': 'mean-near-aligned',
              'mean aug2015 (n=2)': 'mean-aug2015'}
    for title, proto in (('B-frame: mean TIC r', 'bframe'), ('Midpoint: mean TIC r', 'midpoint')):
        sec = _section(txt, title, '* best')
        header = ' '.join(sec.splitlines()[1].split())
        if not header.endswith(' '.join(BENCH_METHODS)):
            raise ValueError(f'{title}: unexpected columns {header!r}')
        cols = [_slug(c) for c in BENCH_METHODS]
        for line in sec.splitlines()[2:]:
            vals = re.findall(rf'{NUM}(?=\*?\s|\*?$)', line.replace('×', ' × '))
            label = line.strip()
            if label.startswith('mean'):
                row = next(v for k, v in groups.items() if label.startswith(k))
                nums = vals[-len(cols):]
            elif PAIR.match(label):
                row, nums = _pair(label.split()[0]), vals[-len(cols):]
            else:
                continue
            for col, v in zip(cols, nums):
                out.setdefault(f'bench-{proto}-{col}', {})[row] = {'mean': float(v)}

    sec = _section(txt, 'Native protocols', 'Paired two-sided')
    for m in re.finditer(rf'^\s+(\S+×\S+)\s+WarpTransformer ({NUM})\s+PCHIP \+ COW refine ({NUM})', sec, re.M):
        out.setdefault('bench-native-warptransformer', {})[_pair(m.group(1))] = {'mean': float(m.group(2))}
        out.setdefault('bench-native-pchip-cow-refine', {})[_pair(m.group(1))] = {'mean': float(m.group(3))}
    for name in ('bench-native-warptransformer', 'bench-native-pchip-cow-refine'):
        vals = [out[name][p]['mean'] for p in drifted4]
        out[name]['drifted4'] = {'min': min(vals), 'max': max(vals)}

    for m in re.finditer(r'^\s+(B-frame|Midpoint): Joint vs (.+?): (.*)$', txt, re.M):
        tests = {_pair(t.group(1)): {'delta': float(t.group(2)), 'p': float(t.group(3))}
                 for t in re.finditer(rf'(\S+×\S+) Δ({NUM}) p=({NUM})', m.group(3))}
        tests['all'] = _span(tests, 'delta') | {'p-max': max(t['p'] for t in tests.values())}
        out[f'bench-wilcoxon-{_slug(m.group(1))}-joint-vs-{_slug(m.group(2))}'] = tests
    return out


LABELS = {
    'anchors': 'Anchor counts', 'lambda-cv': 'λ selection: median held-out anchor error (min)',
    'lambda-sweep': 'λ sweep: mean TIC r after the fact', 'lopo': 'Leave-one-pair-out refits',
    'star': 'Star design (anchors against the reference only)', 'band': '90 % bootstrap band width (min)',
    'coverage': 'Held-out anchor coverage of 90 % intervals', 'drift': 'Fitted shift into the reference (min)',
    'recovery': 'Recovery of known drift (fig 2)', 'cycle': 'Cycle inconsistency over triplets (min)',
    'borrowing': 'Borrowing strength (fig 5)', 'rejection': 'Share of anchors rejected by reweighting',
    'ticr-gaps': 'Gaps between mean TIC r of methods',
}


def report_results(npz: Path = NPZ, txt01: Path = TXT01, txt03: Path = TXT03,
                   numbers: Path = NUMBERS) -> dict[str, dict]:
    base = results(npz)
    common = {'runs': [], 'produced_by': 'scripts/export_ledger.py'}
    drifted4 = base['pairs']['groups']['drifted'][:4]
    found: dict[str, tuple[dict, Path, list[str] | dict]] = {}
    t01, t03 = txt01.read_text(), txt03.read_text()
    for name, m in report01(t01).items():
        found[name] = (m, txt01, provenance(t01))
    for name, m in report03(t03, drifted4).items():
        found[name] = (m, txt03, provenance(t03))
    nums = json.loads(numbers.read_text())
    for name, m in nums['results'].items():
        found[name] = (m, numbers, nums['provenance'])

    def mean(method: str, group: str) -> float:
        return base[method]['metrics'][group]['mean']
    gaps = {'cf': {'max': max(abs(mean('ticr-joint', g) - mean('ticr-joint-cf', g))
                              for g in ('mean-drifted', 'mean-near-aligned'))},
            'joint-minus-pairwise': {'mean-drifted': mean('ticr-joint', 'mean-drifted')
                                     - mean('ticr-pairwise', 'mean-drifted')}}
    found['ticr-gaps'] = (gaps, npz, base['pairs']['source']['provenance'])

    out = dict(base)
    for name, (metrics, path, prov) in found.items():
        label = LABELS.get(name) or name.replace('-', ' ')
        out[name] = {'name': name, 'label': label, 'metrics': metrics, **common, 'source': _source(path, prov)}
    return out


def main(npz: Path = NPZ, out: Path = OUT, reports: bool = True) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, r in (report_results(npz) if reports else results(npz)).items():
        path = out / f'{name}.json'
        path.write_text(json.dumps(r, indent=2) + '\n')
        written.append(path)
    return written


if __name__ == '__main__':
    for p in main():
        print(p.relative_to(ROOT))
