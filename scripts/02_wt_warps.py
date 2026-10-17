"""
02_wt_warps.py — Cache anchor-calibrated WarpTransformer warps for every batch pair.

Runs warpalign's own calibration step (warpalign.steps.calibrate_pair.compute: pooled anchors,
N_CALIB_STEPS of anchor calibration, seed fixed before the loop) with warpalign's default
checkpoint (WarpTransformer v2), and saves the per-sample warps, so script 03 can score the
WarpTransformer under the same protocols as every other method. The protocol is warpalign's,
not a copy of it.

Output: results/wt_warps/<A>__<B>.npz with warps (n_a + n_b, T), the anchors, the native TIC r
(both batches resampled by the model, common window) and the warpalign revision.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from alignbench.datasets.fish_oil import load_batch
from warpalign.calibration import pick_device
from warpalign.config import Paths
from warpalign.steps.calibrate_pair import compute

from jointalign.provenance import warpalign_revision

ROOT = Path(__file__).resolve().parent.parent

DATA = Paths().fish_oil
PAIRS = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
         ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016'),
         ('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016'),
         ('batch_aug2015', 'batch_sep2015'), ('batch_aug2015', 'batch_jan2016'),
         ('batch_aug2015', 'batch_apr2016'), ('batch_aug2015', 'batch_jul2016')]


def main(seed: int, device: str, ckpt: Path | None = None, calib_steps: int | None = None) -> None:
    out_dir = ROOT / 'results' / 'wt_warps'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt or Paths().default_ckpt
    dev = pick_device(device)
    rev = warpalign_revision()
    print(rev, flush=True)
    for a, b in PAIRS:
        out = out_dir / f'{a}__{b}.npz'
        if out.exists():
            print(f'{a} × {b}: cached')
            continue
        A, B = load_batch(a, DATA), load_batch(b, DATA)
        kw = {} if calib_steps is None else {'calib_steps': calib_steps}
        res = compute(A, B, ckpt, dev, seed, **kw)
        np.savez(out, warps=res.warps, n_a=len(A), n_b=len(B), anchors=np.array(res.anchors.anchors),
                 r_native=res.wt.r, seed=seed, ckpt=str(ckpt), warpalign=rev)
        print(f'{a} × {b}: {len(res.anchors.anchors)} anchors, native TIC r {res.wt.mean:.3f}, '
              f'{res.log.seconds:.0f} s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ckpt', type=Path, default=None, help='default: warpalign Paths().default_ckpt')
    a = ap.parse_args()
    main(a.seed, a.device, a.ckpt)
