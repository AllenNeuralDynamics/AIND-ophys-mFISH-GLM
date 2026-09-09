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
| `processing.json` | AIND data schema `Processing` record |
| `data_description.json` | AIND `DerivedDataDescription` derived from processed folder |

Core JSON files (`session.json`, `subject.json`, `procedures.json`, `rig.json`) are also copied to `/results/`.

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
- Only the first imaging plane is loaded
- Cells are capped at 30 (or the plane's cell count, whichever is smaller)
- All other parameters apply normally

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
| `aind_metadata_utils.py` | AIND-schema `processing.json` and `data_description.json` |
| `kernel_json_files/` | Versioned kernel configuration files |

## Dependencies

- `aind-data-schema==1.2.0`
- `comb` (AIND behavior-ophys data loading)
- `lamf-analysis` (utilities)
- numpy, xarray, scipy, pandas, matplotlib, scikit-learn
