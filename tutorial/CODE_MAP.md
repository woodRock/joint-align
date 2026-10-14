# Code map: paper concepts → implementation

Where each piece of Many-Batch Alignment (MBA) lives in the code, with the names the notebook
(`joint_from_scratch.ipynb`) uses. Paths are relative to the repository root. Everything was read from
the code (many-batch-align at the time of writing, gcmsalign `2ec54ea`).

## The joint model (`src/jointalign/multibatch.py`)

| Paper | Code | Key names |
|---|---|---|
| Retention-time grid, T = 200 bins, 0.225 min per bin | `gcmsalign.config` | `N_BINS = 200`, `BIN_MIN = 0.225` (hard-coded: the fitter's knots and inverse grid span `0 … N_BINS − 1`) |
| Warp f_k(t) = t + d_k(t) | `JointWarp.latent(k, t)` | `t + np.interp(t, knots, drift[k])` |
| Knots u_0 … u_{Q−1}, equally spaced | `fit_joint` | `knots = np.linspace(0, N_BINS - 1, n_knots)`, `M = n_knots` (Q in the paper), default `N_KNOTS = 41` |
| Hat basis B(t) | `_hat_basis(t, knots)` | `idx` (left knot), `frac` (position inside the interval, **clipped to [0, 1]**, so t outside the knot range is clamped) |
| θ_k = (c_k, δ_k1 … δ_k,Q−1), d_k(u) = Lθ_k | `fit_joint` | `Lcum = np.tril(np.ones((M, M)))` (L), `theta[k*M:(k+1)*M]` (θ_k) |
| Slope bounds 0.5 ≤ f_k′ ≤ 2 (Eq. 2) | `fit_joint` | `S_MIN, S_MAX = 0.5, 2.0`; `lb`, `ub` = `(S_MIN − 1)·Δu`, `(S_MAX − 1)·Δu` for each δ; c_k unbounded (`±inf`) |
| Anchors (a_j, b_j) in batches (k_j, l_j) | `Anchors` dataclass | `ba, bb` (batch indices, `ba < bb`), `sa, sb` (sample indices), `ta, tb` (RT bins) |
| Residual r_j (Eq. 3) | `fit_joint` | row of `X` = `+B(ta)L` in block `ba`, `−B(tb)L` in block `bb`; `y = tb − ta`; `r = X @ theta − y` |
| Smoothness λ′ Σ‖D₂Lθ_k‖² (Eq. 4) | `fit_joint` | `D2 = np.diff(np.eye(M), n=2, axis=0) @ Lcum`; `lam_eff = lam * n_eff / (K * M)`; block-diagonal `P` |
| Gauge γ‖Gθ‖² | `fit_joint` | `G_rows` = `Lcum` in every block (`gauge='mean'`) or in one block (`gauge=<int>`); weight `GAUGE_WEIGHT * n_eff / M`, `GAUGE_WEIGHT = 1e3`. **A penalty, not an exact constraint** |
| N_eff | `fit_joint` | `n_eff = max(w0.sum(), 1)`, `w0` = `base_weight` (bootstrap counts) or ones |
| (not in the paper) | `fit_joint` | **ridge `1e-8 * n_eff * I`** added to `P` so the Cholesky factorisation always exists |
| Bounded least squares | `fit_joint.solve(w)` | normal matrix `G = XᵀWX + P`, `C = cholesky(G)`, `z = C⁻¹ Xᵀ W y`, then `lsq_linear(C.T, z, bounds=(lb, ub), method='bvls')`. Equivalent to stacking √w·X and the penalty rows (the notebook's route), because both minimise the same quadratic |
| IRLS: 3 Huber rounds (c = 1.345σ̂), 4 Tukey rounds (c = 4.685σ̂) | `fit_joint` | `n_huber=3`, `n_tukey=4`; weights `w_rob`; one final solve after the 7 reweights (8 solves in all) |
| σ̂ = max(1.4826·MAD(r), 0.25) | `_robust_sigma(r)` | computed on `r[w0 > 0]` (anchors with zero bootstrap weight are excluded) |
| Per-pair robust scale σ_kl | `fit_joint` | `scale[a, b]` = `_robust_sigma` of the pair's residuals with `w0 > 0` and `w_rob > 0` |
| Inverse f_l⁻¹ and map g_k→l = f_l⁻¹ ∘ f_k | `JointWarp.inverse`, `JointWarp.map` | `_grid = np.linspace(0, N_BINS - 1, 4 * N_BINS)`; `np.interp(tau, latent(l, _grid), _grid)` (clamps at the run ends) |
| Pairwise baseline (same fitter, one pair) | `fit_pairwise(anchors, a, b, K, **kw)` | fits a 2-batch problem; other batches left at identity |
| Dense PCHIP baseline | `pchip_pairwise(anchors, a, b)` | per-integer-bin medians, monotone filter, PCHIP with pinned ends |

## Anchor harvesting (`src/jointalign/multibatch.py`, building blocks from gcmsalign)

| Step | Code | Key names / constants |
|---|---|---|
| Peak detection on the raw TIC | `gcmsalign.anchors.detect_peaks(tic)` | `scipy.signal.find_peaks`, height ≥ 80th percentile (`PEAK_HEIGHT_PCT`), `distance=5` (`PEAK_MIN_DIST`), keep the 30 tallest (`PEAK_MAX_N`) |
| Spectral fingerprint | `gcmsalign.anchors.fingerprint(chroma, pk)` | mean spectrum of bins pk−1 … pk+1, L2-normalised, `float32` |
| Cosine similarity | `match_anchors` | `sim = fa @ fb.T` (unit vectors, so a dot product is the cosine) |
| Drift limit | `match_anchors` | `MAX_DRIFT_BINS = MAX_DRIFT_MIN / BIN_MIN` (20 min ≈ 88.9 bins); `far` mask |
| Hungarian matching | `match_anchors` | `scipy.optimize.linear_sum_assignment(-np.where(far, -1.0, sim))` |
| q70 threshold | `match_anchors` | `thr = max(percentile(sim[~far], 70), SIM_THRESHOLD)` with `SIM_THRESHOLD = 0.30` (only if ≥ 3 candidates) |
| Refinement | `match_anchors`, `_parabolic` | t_a: parabola through the TIC around the peak; t_b: best fingerprint match within ±`REFINE_RADIUS` = 3 bins, then a parabola through those similarities; offsets clipped to ±0.5 bin |
| RANSAC | `match_anchors` | `sklearn.linear_model.RANSACRegressor(residual_threshold=RANSAC_RESID (3.0 bins), min_samples=2, random_state=0)`, only with ≥ 4 anchors |
| Monotone filter | `match_anchors` | sort by t_a, keep anchors increasing in both t_a and t_b |
| All cross-batch sample pairs | `harvest_anchors(chromas, tics)` | returns an `Anchors` with one row per anchor |

## Experiments and metrics (`scripts/30_multibatch_align.py`, `gcmsalign.metrics`)

| Paper | Code | Key names |
|---|---|---|
| λ grid | script 30 | `LAMBDAS = [0.1, 1, 10, 30, 100, 300, 1000, 3000, 10000]` |
| RT-blocked CV (12-bin blocks, block b → fold b mod 5) | script 30 | `RT_BLOCK = 12`, `N_FOLDS = 5`, `blk(t) = floor(t / RT_BLOCK) % N_FOLDS`; test = **either** end in the fold, train = **neither** end in the fold; score = median `|anchor_errors|` |
| Sample-fold CV (for comparison) | `assign_folds`, `anchor_fold_masks` | samples assigned to folds within each batch |
| Held-out anchor error | `anchor_errors(model, anc)` | `model.map(a, b, ta) − tb` for every anchor |
| Bootstrap over samples | `boot_weights(anc, sizes, rng)` | multinomial counts per batch; an anchor's weight = count(sample a) × count(sample b); passed as `base_weight` |
| 90 % band | script 30 | `np.percentile(maps, [5, 95], axis=0)` over 200 refits |
| Predictive interval | script 30 | `hw_pi = sqrt(hw_ci² + (1.645·scale[a,b])²)` with `hw_ci` = half the band width, centred on the point estimate |
| Gauge reference | script 30 | `ref = argmax(anchor count per batch)` (sep2015 on the fish data) |
| TIC r | `gcmsalign.metrics.pearson_rows` | Pearson r on `EVAL_BINS = slice(22, 200)` (from 5 min: `EVAL_START_BIN = int(5.0 // BIN_MIN) = 22`) |

## From gcmsalign rather than this repository

`gcmsalign.config` (all constants above), `gcmsalign.anchors.detect_peaks` and `fingerprint`,
`gcmsalign.data.load_batch` / `Batch` (real data only), `gcmsalign.metrics.pearson_rows`,
`gcmsalign.warps.warp_rows` (used by script 30's `warp_tics`), and the baselines and WarpTransformer
used in the benchmark (`gcmsalign.baselines`, `gcmsalign.calibration`, `gcmsalign.steps.calibrate_pair`).

## Synthetic data

`src/jointalign/synthetic.py` (`make_dataset`, `BatchSpec`, `true_map`) writes the public stand-in for the fish
oil data (T = 200, M = 1000, 60 compounds, 31 samples). The notebook builds its own, smaller toy
world from scratch so every array can be printed.

## Neural refinement (appendix only)

`src/jointalign/neural_joint.py`: `NeuralJointWarp` (per-sample monotone warps g_i on the latent grid, starting at
the joint model), `train` (anchor Huber loss + groupwise dense similarity), `latent_grid_from_joint`.
