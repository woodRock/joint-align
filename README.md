# many-batch-align

Retention time alignment of GC-MS chromatograms across many (K > 2) batches,
with a shared latent RT frame and calibrated uncertainty.

Follow-up to the pairwise Test-Time Calibrated Warp Transformer
([woodRock/cow-dcnn](https://github.com/woodRock/cow-dcnn), now the `gcmsalign` package in
`~/Desktop/alignment`).

## Dependency: gcmsalign

Everything shared with the pairwise project comes from `gcmsalign`, not from a copy: the
data loader and the fish oil data (`gcmsalign.config.Paths().fish_oil`), peak detection and
fingerprints, the pooled-anchor extraction, PCHIP/COW baselines, warping, the WarpTransformer
and its calibration, and the metrics (TIC r on the common acquisition window, bins ≥ 22). A fix
in gcmsalign applies here immediately.

```
pip install -e ~/Desktop/alignment        # editable: one source of truth
python -m pytest                          # ~40 s; -m "not slow" skips the end-to-end runs
```

Every result file records the gcmsalign version, commit and whether its `src/` was dirty
(`src/provenance.py`). Commit gcmsalign before a run whose results you intend to keep. For a
frozen environment, pin a commit instead:
`gcmsalign @ git+ssh://git@github.com/woodRock/cow-dcnn.git@<sha>`.

## Layout

- `src/multibatch.py`: joint K-batch alignment (sub-bin anchor harvesting on every sample pair,
  joint fit, pairwise baselines)
- `src/neural_joint.py`: neural refinement of the joint model
- `src/provenance.py`: the gcmsalign revision line written into results
- `scripts/30_multibatch_align.py`: the full fish oil evaluation of the joint model
- `scripts/31_wt_warps.py`: WarpTransformer warps for every pair via `gcmsalign.steps.calibrate_pair`
- `scripts/32_fair_benchmark.py`: every method under the same scoring protocols
- `tests/`: unit, synthetic end-to-end and real-data integrity tests

## Method: joint monotone batch warps (`src/multibatch.py`)

Each batch k gets one warp f_k(t) = t + d_k(t) into a shared latent RT frame. d_k is piecewise
linear on 41 knots. All warps are fitted in one bounded least-squares problem:

- **Anchors.** Spectral anchor pairs come from *every* cross-batch sample pair (17,647 anchors on
  the fish data). Each pair contributes the constraint f_a(t_a) = f_b(t_b). Anchors are
  robustly reweighted, first with Huber weights and then with Tukey weights.
- **Smoothness.** A penalty λ on the second differences of d_k. λ is chosen by **RT-blocked**
  cross-validation of held-out anchor error. TIC r is never used for selection.
- **Monotonicity.** The slope of each f_k is bounded to 0.5–2.
- **Gauge.** The reference batch is the one with the most anchors, and its drift is fixed at 0.
  An identity-on-average gauge is also implemented.

The warp from batch k to batch l is f_l⁻¹ ∘ f_k. It is transitive by construction.

Uncertainty comes from a bootstrap over samples within each batch. The 90 % predictive
intervals combine the bootstrap band with each batch pair's robust anchor scatter.

## Results on fish oil (`scripts/30_multibatch_align.py` → `results/multibatch_align.txt`)

> **Stale.** These numbers predate the 2026-10-11 re-preprocessing of the data and the move to gcmsalign (common-window TIC r, gcmsalign anchors, WarpTransformer v2). Re-run scripts 30–32 after clearing `results/` caches; `tests/test_real_data.py` checks the results match the data.

Mean TIC r between batches. A is warped into B's frame. "Drifted" means unaligned TIC r < 0.5;
there are 5 drifted pairs and 5 near-aligned pairs.

| Method | Drifted | Near-aligned |
|---|---|---|
| Unaligned | 0.089 | 0.761 |
| Pooled-anchor PCHIP (cow-dcnn script 24) | 0.410 | 0.671 |
| Dense PCHIP (all anchors, pairwise) | 0.554 | 0.567 |
| Pairwise smooth (same fitter, one pair's anchors) | 0.538 | 0.790 |
| **Joint** | **0.607** | **0.789** |
| Joint, cross-fitted (A_i's warp never saw A_i) | 0.606 | 0.785 |

- **Joint beats pooled PCHIP on every pair with n_A ≥ 6.** The median gain is +0.06 to +0.19 r
  (Holm p ≤ 2e-3).
- **Joint does no harm on near-aligned pairs.** Every other warping method lowered TIC r there.
- **Where direct anchors are good, joint ≈ pairwise.** The per-pair difference is ≤ 0.01 r.
- **The gain comes from borrowing across batches.** aug2015 × jul2016 has 102 mostly
  mismatched direct anchors. Pairwise smooth gets 0.167 there; joint gets 0.572.
- **Missing correspondences.** Removing *all* anchors of a pair changes TIC r by at most 0.03.
  Fitting only against the reference batch (star design) is equally close.
- **Cycle consistency.** Pairwise warps disagree with each other by 1.07 min on average over
  batch triplets, and by up to 3.6 min. Joint warps disagree by 0.001 min.
- **Compared with cow-dcnn pairwise numbers on its 4 drifted pairs.** Joint gets 0.58–0.64.
  COW-TIC got 0.56–0.63 and the WarpTransformer 0.63–0.67.

**Limitations found**

- **Bands are too narrow.** At the selected λ the drift is close to linear and the bootstrap
  bands are narrow (0.03–0.07 min). They widen only weakly where anchors are sparse
  (Spearman ρ = −0.23).
- **Predictive intervals under-cover.** They cover 83 % of held-out anchors against a 90 %
  target. A bootstrap of a heavily regularised fit misses bias, so the intervals need
  calibration, for example conformal scaling.
- **Low λ trades the two pair types off.** At λ = 0.1, drifted-pair TIC r reaches 0.64, but
  near-aligned pairs get worse.
- **The gauge matters under smoothing.** At low λ, the mean gauge put a spurious bend in
  aug2015's warp where it had no anchors (TIC r below unaligned). The reference gauge did not.
- **End-of-run anchors are rejected.** Above about 40 min, anchors are conflicting. They are
  likely bleed-like peaks with similar spectra, and the robust fit rejects them.

## Data

Data and checkpoints are not tracked here. The scripts read the fish oil data and the
WarpTransformer checkpoint (`warp_transformer_v2.pt`) from the gcmsalign repo through
`gcmsalign.config.Paths()`. The old `data/fish_oil` symlink and the v1
`checkpoints/warp_transformer.pt` are no longer used.

## Open design questions

- **K-way study embeddings.** Replace the 2-way batch A/B embedding, and draw anchors from all batch pairs.
- **Gauge fixing.** Use an identity-on-average penalty, or fix a reference batch.
- **Uncertainty.** Use ensembles over seeds and anchor subsets, MC dropout, or a probabilistic warp head.
- **Transitive consistency.** Check that the A↔B and B↔C warps compose to the A↔C warp.
- **Model choice.** In the cow-dcnn mechanism ablation, batch-shared warps matched per-sample warps,
  so a batch-level model is enough (TIC r 0.611–0.663 for both; Holm p ≥ 0.13). The warp
  transformer beat pooled-anchor PCHIP (0.611–0.663 vs 0.476–0.517). Against PCHIP + COW
  refinement the result was mixed: COW did better on two pairs and the WT on one.
- **Testbed.** Four fish oil batches is a thin testbed, and batch is confounded with species and tissue.
  A dataset with more batches and repeated QC samples is needed.
