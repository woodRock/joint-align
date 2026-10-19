"""Table II: mean TIC r per batch pair (A warped into B), drifted then near-aligned, then group means.

Run by `taura build`; reads $TAURA_RESULTS and writes a taura table spec to $TAURA_OUTPUT.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# (column label, result); Joint-CF is cross-fitted, so it is never bolded as "best in-sample".
METHODS = [('Unal.', 'ticr-unaligned', True), ('Pooled', 'ticr-pooled-pchip', True),
           ('Dense', 'ticr-dense-pchip', True), ('Pairw.', 'ticr-pairwise', True),
           ('Joint', 'ticr-joint', True), ('Joint-CF', 'ticr-joint-cf', False)]
GROUPS = [('drifted', r'\textit{Drifted}', 'Mean, drifted'),
          ('near-aligned', r'\textit{Near-aligned}', 'Mean, near-al.')]


def short(batch: str) -> str:
    return batch[:3] + batch[-2:]          # sep2015 → sep15


def label(pair: str) -> str:
    a, b = pair.split('-')
    return rf'{short(a)} $\times$ {short(b)}'


def main() -> None:
    R = json.loads(Path(os.environ['TAURA_RESULTS']).read_text())
    pairs = R['pairs']
    cols = [{'label': r'$A\times B$'}, {'label': 'Anchors', 'metric': 'anchors', 'bold': 'none'}]
    cols += [{'label': lab, 'metric': 'ticr', **({} if best else {'bold': 'none'})} for lab, _, best in METHODS]

    def values(key: str) -> list:
        return [R[name]['metrics'][key]['mean'] for _, name, _ in METHODS]

    rows: list = []
    for group, heading, _ in GROUPS:
        rows.append({'group': heading})
        for p in pairs['groups'][group]:
            rows.append([label(p), pairs['metrics'][p]['anchors'], *values(p)])
    rows.append({'rule': True})
    for group, _, mean_label in GROUPS:
        rows.append([mean_label, '', *values(f'mean-{group}')])

    Path(os.environ['TAURA_OUTPUT']).write_text(json.dumps({'columns': cols, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
