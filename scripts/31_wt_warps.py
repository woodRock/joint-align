"""
31_wt_warps.py — Cache anchor-calibrated WarpTransformer warps for every batch pair.

Runs the cow-dcnn protocol (script 16: extract_anchors, 500 calibration steps, seed fixed
before the loop) and saves the per-sample warps, so script 32 can score the WarpTransformer
under the same protocols as every other method.

Output: results/wt_warps/<A>__<B>.npz with warps (n_a + n_b, T) and the native TIC r
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('calib16', ROOT / 'scripts' / '16_align_testtime_calib.py')
calib16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib16)

PAIRS = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
         ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016'),
         ('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016'),
         ('batch_aug2015', 'batch_sep2015'), ('batch_aug2015', 'batch_jan2016'),
         ('batch_aug2015', 'batch_apr2016'), ('batch_aug2015', 'batch_jul2016')]


def main(seed: int, device: str) -> None:
    out_dir = ROOT / 'results' / 'wt_warps'
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    torch.set_num_threads(8)
    for a, b in PAIRS:
        out = out_dir / f'{a}__{b}.npz'
        if out.exists():
            print(f'{a} × {b}: cached')
            continue
        ca = calib16.load_chromas(ROOT / 'data' / 'fish_oil' / a / 'chroma')
        cb = calib16.load_chromas(ROOT / 'data' / 'fish_oil' / b / 'chroma')
        anchors = calib16.extract_anchors(ca, cb)
        model = calib16.ChromaWarpTransformer()
        model.load_state_dict(torch.load(ROOT / 'checkpoints' / 'warp_transformer.pt', map_location=dev))
        model.to(dev)
        ids = torch.tensor([0] * len(ca) + [1] * len(cb), dtype=torch.long, device=dev)
        X = torch.from_numpy(np.stack(ca + cb)).to(dev)
        t0 = time.time()
        if anchors:
            calib16.calibrate(model, X, ids, anchors, dev, seed=seed)
        model.eval()
        with torch.no_grad():
            al, w = model(X, ids)
        al = al.cpu().numpy()
        r_native = calib16.tic_r_per_pair(list(al[:len(ca)]), list(al[len(ca):]))
        np.savez(out, warps=w.cpu().numpy(), n_a=len(ca), n_b=len(cb), anchors=np.array(anchors),
                 r_native=r_native, seed=seed)
        print(f'{a} × {b}: {len(anchors)} anchors, native TIC r {r_native.mean():.3f}, '
              f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    main(a.seed, a.device)
