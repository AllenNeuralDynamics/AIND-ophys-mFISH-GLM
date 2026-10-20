# AIND-ophys-mFISH-GLM

Reproducible encoding GLM capsule for AIND multiplane ophys (mFISH) sessions.
Fits a ridge-regularized linear model per cell across all imaging planes of a session,
using stimulus, behavioral, and task-event regressors defined in versioned kernel JSON files.

## Data layout expected under `/data/`

| Directory pattern | Contents |
|---|---|
| `multiplane-ophys_<id>_<date>_<time>/` | Raw session data |
| `multiplane-ophys_<id>_<date>_<time>_processed_<date>/` | Processed ophys (planes as `VISp_0/`, `VISp_1/`, …) |
| `multiplane-ophys_<id>_<date>_<time>_lp-eye*/` (or `_dlc-eye*/`) | Eye-tracking output |

All three directories are discovered automatically; no path arguments are required for standard runs.

## Outputs written to `/results/{session_key}_glm_v{version:02}/`

| File | Description |
|---|---|
| `design_matrix.nc` | Full design matrix (T × K xarray) |
| `{data_type}_activity_trace_matrix.nc` | Neural activity (T × N xarray) |
| `{data_type}_activity_trace_info.npy` | Timestamps, time bins, frame rate |
| `unstd_features.npy` | Un-standardized continuous regressors |
| `run_params.json` | Expanded kernel and dropout definitions |
| `glm_results_v{version:02}_{session_key}_{data_type}.npy` | CV weights, VE, lambdas |
| `qc_summary_{session_key}_{data_type}.png` | VE distribution, train/test scatter, kernel contributions |
| `heatmap_{session_key}_{data_type}.png` | Depth-sorted z-scored heatmap + behavioral traces |
| `top_cells/rank{N:02d}_{cell_id}_ve{ve:.3f}.png` | Single-cell figures for top 10 cells by test-set VE (see below) |

Core JSON files (`data_description.json`, `processing.json`, `session.json`, `subject.json`, `procedures.json`, `rig.json`) are also copied to `/results/`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--kernels_config` | `kernel_json_files/kernel_v01.json` | Path to kernel JSON file |
| `--data_type` | `events` | Neural signal: `events` or `dff` |
| `--cv_folds` | `5` | Outer cross-validation folds |
| `--cv_nested_folds` | `5` | Inner (lambda-selection) CV folds |
| `--n_lambdas` | `40` | Number of ridge λ values in log-spaced grid |
| `--lambda_min` | `1.0` | Minimum ridge λ |
| `--lambda_max` | `10000.0` | Maximum ridge λ |
| `--min_activity_support` | `0.05` | Minimum fraction of active frames to include a cell |
| `--test` | `0` | Test mode (`1`): load one plane only, cap at 30 cells |

## Kernel configuration

Regressors are defined in versioned JSON files under `code/kernel_json_files/`.
Pass `--kernels_config` to select a file; the version number in the filename (e.g. `v01`)
determines the output subdirectory name.

Each kernel entry has the following fields:

```json
"kernel_name": {
  "feature":     "hit",       // internal feature label
  "type":        "discrete",  // "discrete" or "continuous"
  "length":      2.25,        // kernel duration in seconds
  "offset":      0,           // onset offset in seconds (negative = anticipatory)
  "num_weights": null,        // null → inferred from length × frame rate
  "dropout":     true         // include in dropout (ablation) analysis
}
```

**Special feature labels:**
- `each-image` — expands to one kernel per unique image identity (mutually exclusive with `any-image`)
- `any-image` — single shared kernel for all non-omitted images
- `intercept` — constant offset (set `length: 0`)

**Available kernel sets:**

| File | Description |
|---|---|
| `kernel_v01.json` | Full set: intercept, hits, misses, omissions, each-image, running, pupil, licks (all with dropout) |
| `kernel_test.json` | Minimal set for fast iteration: intercept, each-image, any-image, hits, misses — no dropout analysis |

## Test mode

Set `--test 1` to run a fast smoke-test:
- `kernel_test.json` is used automatically (overrides `--kernels_config`)
- Only the first imaging plane is loaded
- Cells are capped at 30 (or the plane's cell count, whichever is smaller)
- All other parameters apply normally

## Improvements over the previous GLM (Garrett et al., 2026)

**Model performance**
- *Stratified cross-validation folds.* Frames are stratified by time (5 contiguous bins) and by
  binarized behavioral state (running speed, lick rate, rolling reward rate, pupil), then every
  combination of strata is split evenly across the outer and inner folds. Each fold therefore
  samples all session epochs and behavioral states, including rewarded and non-rewarded periods.
  Variables missing from a session, or with an empty stratum, are skipped. Settings are in
  `default_fit_params()` (`cv_stratify`).
- *Per-cell ridge regularization via nested cross-validation.* For each outer fold, the inner CV
  scores a log-spaced λ grid (`--n_lambdas`, `--lambda_min`, `--lambda_max`) and picks the λ with
  the highest mean inner-fold VE separately for every cell. The outer fit uses that cell's own λ.

**Fitting speed**
- *SVD-based vectorized ridge.* One economy SVD of the training design matrix per fold replaces
  per-λ and per-cell linear solves. All λ values and all cells are scored with elementwise scaling
  `S / (S² + λ)` and a single matrix multiply. The final fit applies each cell's own λ through an
  `(r × N)` scaling matrix in one matmul.
- *Flat (fold × dropout-model) parallelism.* One persistent fork-based `multiprocessing` pool runs
  all tasks (replacing Ray and the per-fold pools). Fold indices and design/activity arrays live in
  fork-shared memory, and workers slice them per task, so large arrays are neither serialized nor
  duplicated.

## Known limitations

**Possible data leakage from frame-level fold assignment.** Within each stratum, frames are
shuffled individually before being split into folds (`get_stratified_folds`). Folds are *not*
chunked by trial, image flash, or time window, so a held-out frame usually has temporally adjacent
frames in the training set.

*Why this is a problem.* Adjacent frames (about 10 Hz) have nearly identical design-matrix rows
(kernels span up to 2.25 s) and highly correlated activity (indicator kinetics, slow drift,
autocorrelated residuals). Held-out prediction then partly amounts to interpolation between
neighboring training frames rather than generalization to new events. The train/test VE ratio
(overfitting check) cannot detect this, because test VE is inflated together with train VE and the
ratio stays near 1. Consequences:
- Absolute VE may be optimistic.
- The inner CV is also frame-shuffled, so it favors weak regularization and the per-cell λ may be
  biased low.

*Why the effect may be modest.* The model is linear ridge with roughly 200 weights per cell fit to
thousands of frames, so it cannot memorize individual frames. Leakage mainly helps through slow
residual structure that the regressors can partly absorb. Dropout and kernel comparisons are
expected to be less affected, since all models are inflated similarly. This has not been measured.

*How to test.*
1. On a few sessions, compute lag-1 and lag-5 autocorrelation of test-set residuals. High values
   indicate leakage potential.
2. Refit with chunked folds (contiguous blocks, e.g. one image flash of 0.75 s or fixed 1–2 s
   windows, ideally longer than the longest kernel) and compare against the current frame-shuffled
   folds: per-cell test VE (scatter), distributions of the selected λ, and dropout VE differences.
3. If VE and λ change little, the current scheme is adequate; otherwise move to chunked folds and
   bump the version.

## Code structure

| Module | Role |
|---|---|
| `run_capsule.py` | Orchestration entry point |
| `load_data.py` | Session discovery, plane loading, COMB data extraction |
| `design_matrix_tools.py` | Design matrix construction (`build_design_matrix`) and kernel addition |
| `kernel_tools.py` | Kernel expansion (`each-image` → per-image), dropout definitions |
| `glm_fit_tools.py` | Ridge GLM fitting, CV, results I/O |
| `DesignMatrix.py` | Design matrix class |
| `qc_figures.py` | QC summary figure and depth-sorted heatmap |
| `glm_cell_analysis.py` | Per-cell QC figures — `GLMCellAnalysis` class (see below) |
| `aind_metadata_utils.py` | AIND-schema `processing.json` and `data_description.json` |
| `kernel_json_files/` | Versioned kernel configuration files |

## Single-cell QC figures (`glm_cell_analysis.py`)

After fitting, `run_capsule.py` automatically generates one combined figure per cell for the top 10 cells ranked by test-set variance explained, saving them to `top_cells/` inside the results directory.

Each figure has two panels:

**Panel 1 — full traces + zoom windows**
- Three stacked traces (actual / model / residual) spanning the whole session.
- Three zoomed 60 s windows selected from the first, middle, and last thirds of the session (highest-variance window per third); colored boxes in the full trace mark each window and dashed lines connect them to the zoom subplots.

**Panel 2 — kernels and PSTHs**
- *Images (top):* kernel weight time course (navy) and mean response PSTH (black = actual, blue = model) for each image identity, with matched y-limits across all image kernels and across all image PSTHs.
- *Omissions / Hits / Misses / Behavioral (bottom):* same layout for omission, hit, and miss kernels (each with a PSTH), plus running-speed, pupil, and lick kernels (kernel only — no PSTH). All kernel y-limits unified across the entire panel; all PSTH y-limits unified across the entire panel.

The class can also be used standalone:

```python
from glm_cell_analysis import GLMCellAnalysis

glm = GLMCellAnalysis(
    results_path='/results/800792_2025-08-14_glm_v01',
    session_key='800792_2025-08-14',
    data_type='events',   # or 'dff'
    version=1,
)
glm.save_top_cells(n=10)                        # → results_path/top_cells/
glm.save_cell_figure(cell_idx=0, out_path='cell0.png')  # single cell
```

Or from the command line:

```bash
python glm_cell_analysis.py \
    --results_path /results/800792_2025-08-14_glm_v01 \
    --session_key  800792_2025-08-14 \
    --data_type    events \
    --n            10
```

## Dependencies

- `aind-data-schema==1.2.0`
- `comb` (AIND behavior-ophys data loading)
- `lamf-analysis` (utilities)
- numpy, xarray, scipy, pandas, matplotlib, scikit-learn
