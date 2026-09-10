"""
run_capsule.py — Reproducible GLM fit capsule for AIND mFISH multiplane-ophys sessions.

Usage (CodeOcean reproducible run)
-----------------------------------
python run_capsule.py [options]

Data is discovered automatically from /data/:
  raw session dir   multiplane-ophys_<id>_<date>_<time>/
  processed dir     multiplane-ophys_<id>_<date>_<time>_processed_<date>/
  eye-tracking dir  multiplane-ophys_<id>_<date>_<time>_lp-eye*/  (or _dlc-eye*)

Results are written to /results/{session_key}_glm_v{kernel_version}/:
  design_matrix.nc
  {data_type}_activity_trace_matrix.nc
  {data_type}_activity_trace_info.npy
  unstd_features.npy
  run_params.json
  glm_results_v{version:02}_{session_key}_{data_type}.npy
  qc_summary_{session_key}_{data_type}.png
  heatmap_{session_key}_{data_type}.png
  processing.json
  data_description.json
"""

# Thread limits must be set BEFORE numpy is imported so child processes inherit them.
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import datetime
import json
import multiprocessing
import re
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
CODE_DIR    = Path(__file__).parent
KERNEL_DIR  = CODE_DIR / 'kernel_json_files'
DATA_DIR    = Path('/data')
RESULTS_DIR = Path('/results')

for _p in [str(CODE_DIR), '/comb', '/lamf-analysis']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import xarray as xr

import glm_fit_tools as gft
import design_matrix_tools as dmtools
import load_data
from qc_figures import save_qc_summary, save_heatmap_figure
from glm_cell_analysis import GLMCellAnalysis
from aind_metadata_utils import write_metadata_files


# ── fork-based parallel GLM fit (replaces ray) ───────────────────────────────
_WORKER_DATA: dict = {}


def _one_model_mp(model_label):
    d = _WORKER_DATA
    return gft.collect_model_results(
        d['run_params'], d['fit_params'],
        d['X_tr'], d['X_te'], d['y_tr'], d['y_te'],
        d['nested'], model_label)


def _collect_fold_results_mp(run_params, fit_params,
                              X_tr, X_te, y_tr, y_te,
                              nested, num_cores=None):
    """Drop-in for gft.collect_fold_results_parallel using fork + Pool."""
    global _WORKER_DATA
    X_tr.load(); X_te.load(); y_tr.load(); y_te.load()
    _WORKER_DATA.update(
        run_params=run_params, fit_params=fit_params,
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, nested=nested,
    )
    models    = list(run_params['dropouts'].keys())
    n_workers = num_cores or min(len(models), os.cpu_count() or 16)
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_one_model_mp, models)

    lam_f = W_f = vetr_f = vete_f = ver_f = None
    for mi, (lam, W, ve_tr, ve_te, ve_r) in enumerate(results):
        if mi == 0:
            lam_f, W_f, vetr_f, vete_f, ver_f = lam, W, ve_tr, ve_te, ve_r
        else:
            lam_f  = xr.concat([lam_f,  lam],   dim='model')
            W_f    = xr.concat([W_f,    W],      dim='model')
            vetr_f = xr.concat([vetr_f, ve_tr],  dim='model')
            vete_f = xr.concat([vete_f, ve_te],  dim='model')
            ver_f  = xr.concat([ver_f,  ve_r],   dim='model')
    return lam_f, W_f, vetr_f, vete_f, ver_f


# ── entry point ───────────────────────────────────────────────────────────────
def run():
    parser = argparse.ArgumentParser(
        description='Reproducible GLM fit for AIND mFISH multiplane-ophys sessions.')

    parser.add_argument('--kernels_config', default='kernel_v01.json',
        help='Path to kernel JSON file (default directory: KERNEL_DIR)')
    parser.add_argument('--data_type', default='events', choices=['dff', 'events'])
    parser.add_argument('--cv_folds',             type=int,   default=5)
    parser.add_argument('--cv_nested_folds',      type=int,   default=5)
    parser.add_argument('--n_lambdas',            type=int,   default=40)
    parser.add_argument('--lambda_min',           type=float, default=1.0)
    parser.add_argument('--lambda_max',           type=float, default=10000.0)
    parser.add_argument('--min_activity_support', type=float, default=0.05)

    parser.add_argument('--test', type=int, default=0, choices=[0, 1],
                        help='Test mode (1): load one plane only, cap at 30 cells')

    parser.add_argument('--data_dir',    default=str(DATA_DIR))
    parser.add_argument('--results_dir', default=str(RESULTS_DIR))

    args = parser.parse_args()
    test_mode = bool(args.test)

    start_time  = datetime.datetime.now()
    data_dir    = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover session paths
    session_name, session_key, raw_dir, proc_dir, eye_dir = \
        load_data.discover_session(data_dir)
    print(f'Session : {session_name}')
    print(f'Key     : {session_key}')
    print(f'Proc    : {proc_dir.name}')
    print(f'Eye     : {eye_dir.name}')

    # 2. Load kernel config
    kernel_config_path = KERNEL_DIR / 'kernel_test.json' if test_mode \
        else Path(args.kernels_config)
    if not kernel_config_path.is_absolute():
        kernel_config_path = KERNEL_DIR / kernel_config_path
    if not kernel_config_path.exists():
        raise FileNotFoundError(f'Kernel config not found: {kernel_config_path}')
    with open(kernel_config_path) as f:
        raw_kernel_dict = json.load(f)
    kernel_dict = {k: v for k, v in raw_kernel_dict.items() if not k.startswith('_')}
    for k in kernel_dict:
        kernel_dict[k].setdefault('num_weights', None)

    m = re.search(r'v(\d+)', kernel_config_path.stem)
    version = int(m.group(1)) if m else 1
    print(f'Kernels : {list(kernel_dict.keys())}  (config: {kernel_config_path.name}, v={version})')

    # 3. Output directory
    save_dir = results_dir / f'{session_key}_glm_v{version:02}'
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output  : {save_dir}')

    # 4. Build design matrix (skip if cached)
    dm_file = save_dir / 'design_matrix.nc'
    at_file = save_dir / f'{args.data_type}_activity_trace_matrix.nc'
    if dm_file.exists() and at_file.exists():
        print('Design matrix artifacts already exist — skipping build.')
    else:
        print('Loading planes...')
        bod_list = load_data.load_all_planes(proc_dir, raw_dir, eye_dir)
        if test_mode:
            bod_list = bod_list[:1]
            print(f'TEST MODE: using 1 plane ({bod_list[0].metadata["ophys_plane_id"]})')
        print('Building design matrix...')
        run_params, design, X, activity_trace = dmtools.build_design_matrix(
            bod_list, kernel_dict, args.data_type)
        print(f'Design matrix: {X.shape}  |  cells: {activity_trace["activity_trace_arr"].shape[1]}')

        rp_serial = {k: (list(v) if isinstance(v, set) else v) for k, v in run_params.items()}
        with open(save_dir / 'run_params.json', 'w') as f:
            json.dump(rp_serial, f, indent=2, default=str)
        X.to_netcdf(dm_file)
        activity_trace['activity_trace_arr'].to_netcdf(at_file)
        np.save(save_dir / f'{args.data_type}_activity_trace_info.npy', {
            'timestamps':       activity_trace['timestamps'],
            'time_bins':        activity_trace['time_bins'],
            'ophys_frame_rate': activity_trace['ophys_frame_rate'],
        })
        np.save(save_dir / 'unstd_features.npy', design.unstd_features)
        print('Design matrix artifacts saved.')

    # 5. Configure and run GLM
    gft.collect_fold_results_parallel = _collect_fold_results_mp

    fit_params = gft.default_fit_params()
    fit_params['cv_fold']        = args.cv_folds
    fit_params['cv_nested_fold'] = args.cv_nested_folds
    fit_params['L2_grid_num']    = args.n_lambdas
    fit_params['L2_grid_range']  = [args.lambda_min, args.lambda_max]

    X_load, at_arr, at_info, run_params_load, unstd_features, use_indices = \
        gft.load_data(session_key, args.data_type, version, load_path=results_dir)

    ophys_frame_rate = at_info['ophys_frame_rate']
    X_trim           = X_load[use_indices, :]
    at_trim          = at_arr[use_indices, :]
    at_trim_filtered = gft.filter_activity_trace_matrix(
        at_trim, prop_support_threshold=args.min_activity_support)
    print(f'Cells after filtering: {at_trim_filtered.shape[1]} / {at_trim.shape[1]}')
    if test_mode:
        n_test = min(30, at_trim_filtered.shape[1])
        at_trim_filtered = at_trim_filtered[:, :n_test]
        print(f'TEST MODE: capped to {n_test} cells')

    stratified_list = gft.set_stratified_list(
        fit_params, X_load, unstd_features, use_indices, ophys_frame_rate)
    stratified_frames, cv_inds_stratified = gft.get_stratified_folds(fit_params, stratified_list)

    print(f'Fitting GLM ({fit_params["cv_fold"]}-fold CV)...')
    lambdas_cv, W_cv, ve_train_cv, ve_test_cv, ve_ratio_cv = gft.collect_session_results(
        run_params_load, fit_params, X_trim, at_trim_filtered,
        stratified_frames, cv_inds_stratified, parallel=True)
    print('GLM fit complete.')

    var_explained_mean = gft.get_full_session_var_explained_from_mean_model(
        W_cv, X_trim, at_trim_filtered)

    gft.save_glm_results(
        version, session_key, args.data_type,
        fit_params, use_indices, stratified_frames, cv_inds_stratified,
        lambdas_cv, W_cv, ve_train_cv, ve_test_cv, ve_ratio_cv,
        var_explained_mean,
        save_dir=save_dir,
    )
    print(f'Results saved → {save_dir}')

    # 6. QC figures
    results_fn = save_dir / f'glm_results_v{version:02}_{session_key}_{args.data_type}.npy'
    results    = np.load(results_fn, allow_pickle=True).item()
    with open(save_dir / 'run_params.json') as f:
        run_params_fig = json.load(f)

    print('Generating QC figures...')
    save_qc_summary(session_key, args.data_type, results, run_params_fig, save_dir)
    save_heatmap_figure(session_key, args.data_type, results, save_dir,
                        proc_dir, raw_dir, eye_dir)

    print('Generating single-cell figures (top 10)...')
    glm_ca = GLMCellAnalysis(
        results_path=save_dir,
        session_key=session_key,
        data_type=args.data_type,
        version=version,
    )
    glm_ca.save_top_cells(n=10, out_dir=save_dir / 'top_cells')

    # 7. Metadata
    end_time = datetime.datetime.now()
    run_parameters = {
        'session_key':         session_key,
        'data_type':           args.data_type,
        'test':                args.test,
        'kernels_config':      str(kernel_config_path),
        'kernel_version':      version,
        'kernels':             kernel_dict,
        'cv_folds':            args.cv_folds,
        'cv_nested_folds':     args.cv_nested_folds,
        'n_lambdas':           args.n_lambdas,
        'lambda_min':          args.lambda_min,
        'lambda_max':          args.lambda_max,
        'min_activity_support': args.min_activity_support,
    }
    write_metadata_files(
        session_name=session_name,
        proc_dir=proc_dir,
        save_dir=save_dir,
        start_dt=start_time,
        end_dt=end_time,
        run_parameters=run_parameters,
        data_dir=data_dir,
        results_dir=results_dir,
    )


if __name__ == '__main__':
    run()
