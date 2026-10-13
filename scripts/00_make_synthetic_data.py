"""
00_make_synthetic_data.py — A public synthetic stand-in for the (private) fish oil data.

Five batches with the fish oil batch names and design, drifts shaped like the real ones, and
the true drifts saved alongside (truth.json), so the whole pipeline can be run and checked
against the truth:

    python scripts/00_make_synthetic_data.py
    python scripts/30_multibatch_align.py --data data/synthetic --results results/synthetic --figs figs/synthetic
    python scripts/32_fair_benchmark.py   --data data/synthetic --results results/synthetic --neural-steps 300

(Script 31, the WarpTransformer baseline, needs the gcmsalign checkpoint; script 32 skips that
method when its warps are absent.) See src/synthetic.py for the generative model.

Output: data/synthetic/<batch>/chroma/*.npz, data/synthetic/truth.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from synthetic import DEFAULT_BATCHES, make_dataset


def main(out: Path, seed: int) -> None:
    make_dataset(out, seed=seed)
    print(f'{out}: ' + ', '.join(f'{b.name.replace("batch_", "")} (n={b.n})' for b in DEFAULT_BATCHES)
          + f'; true drifts in {out / "truth.json"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'synthetic')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.seed)
