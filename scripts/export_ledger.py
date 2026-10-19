"""
export_ledger.py — Script 01's TIC r results as taura ledger results (ledger/results/*.json).

Reads results/multibatch_align.npz and writes, for every method, the mean TIC r of each batch pair
(A warped into B's frame; mean over all A_i × B_j, as in section 2 of multibatch_align.txt) and
the means over drifted and near-aligned pairs. pairs.json holds the anchor count of every pair and
which pairs are drifted (unaligned TIC r < 0.5, the cow-dcnn exclusion rule).

Result keys, for \\res{} in the paper: ticr-joint/sep2015-jul2016/mean, ticr-joint/mean-drifted/mean,
pairs/sep2015-jul2016/anchors. The output depends only on the npz, so re-running is a no-op.

Output: ledger/results/ticr-*.json, ledger/results/pairs.json
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / 'results' / 'multibatch_align.npz'
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


def main(npz: Path = NPZ, out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, r in results(npz).items():
        path = out / f'{name}.json'
        path.write_text(json.dumps(r, indent=2) + '\n')
        written.append(path)
    return written


if __name__ == '__main__':
    for p in main():
        print(p.relative_to(ROOT))
