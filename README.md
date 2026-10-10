# many-batch-align

Retention time alignment of GC-MS chromatograms across many (K > 2) batches,
with a shared latent RT frame and calibrated uncertainty.

Follow-up to the pairwise Test-Time Calibrated Warp Transformer
([woodRock/cow-dcnn](https://github.com/woodRock/cow-dcnn)). The core code was copied
from cow-dcnn at commit `c7baa82` and is expected to diverge; bug fixes are ported by hand.

## Layout

- `src/warp_transformer.py`: `ChromaWarpTransformer` and the warp smoothness loss
- `src/alignment.py`, `src/encoder.py`: chromatogram loading, spectral anchor extraction, warping
- `scripts/preprocess_fish_oil_batches.py`: raw fish oil CSVs → `data/fish_oil/`
- `scripts/13_train_warp_transformer.py`: synthetic pretraining
- `scripts/16_align_testtime_calib.py`: pairwise test-time calibration (anchor extraction lives here)
- `scripts/24_pchip_pooled_anchors.py`: pooled-anchor PCHIP baseline

## Data

Data and checkpoints are not tracked. `data/fish_oil` is a local symlink to
`../alignment/data/fish_oil`, and `checkpoints/warp_transformer.pt` is copied from cow-dcnn.

## Open design questions

- **K-way study embeddings.** Replace the 2-way batch A/B embedding, and draw anchors from all batch pairs.
- **Gauge fixing.** Use an identity-on-average penalty, or fix a reference batch.
- **Uncertainty.** Use ensembles over seeds and anchor subsets, MC dropout, or a probabilistic warp head.
- **Transitive consistency.** Check that the A↔B and B↔C warps compose to the A↔C warp.
- **Model choice.** In the cow-dcnn mechanism ablation, batch-shared warps matched per-sample warps,
  and pooled PCHIP (+ COW) beat the warp transformer by a wide margin.
- **Testbed.** Four fish oil batches is a thin testbed, and batch is confounded with species and tissue.
  A dataset with more batches and repeated QC samples is needed.
