"""
run_capsule.py — Reproducible GLM fit capsule for AIND mFISH multiplane-ophys sessions.

Usage (CodeOcean reproducible run)
-----------------------------------
python run_capsule.py [options]

Data is discovered automatically from /data/:
  - raw session dir         multiplane-ophys_<id>_<date>_<time>/
  - processed dir           multiplane-ophys_<id>_<date>_<time>_processed_<date>/
  - eye-tracking dir        multiplane-ophys_<id>_<date>_<time>_lp-eye*/
                            multiplane-ophys_<id>_<date>_<time>_dlc-eye*/

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
import platform
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr

# ── path setup ────────────────────────────────────────────────────────────────
CODE_DIR    = Path(__file__).parent
KERNEL_DIR  = CODE_DIR / 'kernel_json_files'
DATA_DIR    = Path('/data')
RESULTS_DIR = Path('/results')

for _p in [str(CODE_DIR), '/comb', '/lamf-analysis']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glm_fit_tools as gft
import kernel_tools as ktools
import load_data as ld
import design_matrix_tools as dmtools
from DesignMatrix import DesignMatrix


# ── session discovery ─────────────────────────────────────────────────────────
def discover_session(data_dir: Path):
    """Locate raw, processed, and eye-tracking directories under data_dir."""
    candidates = sorted(data_dir.iterdir())

    def _is_raw(p):
        return (p.is_dir()
                and re.match(r'^multiplane-ophys_\d+_\d{4}-\d{2}-\d{2}_', p.name)
                and '_processed' not in p.name
                and '_lp-eye'    not in p.name
                and '_dlc-eye'   not in p.name
                and '_stim'      not in p.name)

    raw_dirs = [p for p in candidates if _is_raw(p)]
    if len(raw_dirs) != 1:
        raise ValueError(f'Expected 1 raw session dir in {data_dir}, found {len(raw_dirs)}: {[p.name for p in raw_dirs]}')
    raw_dir      = raw_dirs[0]
    session_name = raw_dir.name
    session_key  = '_'.join(session_name.split('_')[1:3])   # e.g. '800792_2025-08-18'

    proc_dirs = sorted([p for p in candidates if p.is_dir()
                        and p.name.startswith(session_name + '_processed')])
    if not proc_dirs:
        raise ValueError(f'No processed dir for {session_name}')
    proc_dir = proc_dirs[-1]

    eye_dirs = sorted([p for p in candidates if p.is_dir()
                       and p.name.startswith(session_name)
                       and ('_lp-eye' in p.name or '_dlc-eye' in p.name)])
    if not eye_dirs:
        raise ValueError(f'No eye-tracking dir for {session_name}')
    # prefer lp-eye over dlc-eye
    lp = [p for p in eye_dirs if '_lp-eye' in p.name]
    eye_dir = lp[-1] if lp else eye_dirs[-1]

    return session_name, session_key, raw_dir, proc_dir, eye_dir


# ── plane loading ─────────────────────────────────────────────────────────────
def _is_plane_dir(p: Path) -> bool:
    return p.is_dir() and bool(re.match(r'^[A-Za-z]+_\d+$', p.name))


def _merge_trials(bod):
    stim   = bod.stimulus_presentations
    trials = bod.trials
    stim['is_change'] = stim.is_change.astype(bool)
    stim['hit']  = False
    stim['miss'] = False
    stim.loc[stim.start_time.isin(trials.query('hit').change_time.values),  'hit']  = True
    stim.loc[stim.start_time.isin(trials.query('miss').change_time.values), 'miss'] = True


def _load_plane(plane_dir, raw_dir, eye_dir):
    from comb.behavior_ophys_dataset import BehaviorOphysDataset
    bod = BehaviorOphysDataset(
        plane_folder_path=plane_dir,
        raw_folder_path=raw_dir,
        eye_tracking_path=eye_dir,
        pipeline_version='v6',
    )
    bod.metadata['ophys_plane_id'] = plane_dir.name
    _merge_trials(bod)
    return bod


def load_all_planes(proc_dir, raw_dir, eye_dir, n_workers=4):
    plane_dirs = sorted([p for p in proc_dir.iterdir() if _is_plane_dir(p)])
    print(f'Loading {len(plane_dirs)} planes ({n_workers} threads): {[p.name for p in plane_dirs]}')
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_load_plane, pd, raw_dir, eye_dir): pd for pd in plane_dirs}
        bod_map = {}
        for fut, pd in futures.items():
            bod_map[pd.name] = fut.result()
            print(f'  loaded {pd.name}')
    return [bod_map[pd.name] for pd in plane_dirs]


# ── design matrix ─────────────────────────────────────────────────────────────
def build_design_matrix(bod_list, kernel_dict, data_type):
    run_params = {'data_type': data_type}
    at_list, rp_list = [], []
    for bod in bod_list:
        run_params = ktools.process_kernels(kernel_dict.copy(), run_params, bod)
        at, run_params = ld.extract_and_annotate_ophys_plane(bod, run_params)
        at_list.append(at)
        rp_list.append(run_params)
    run_params = rp_list[0]
    run_params['input_kernel_dict'] = kernel_dict

    at_arr = xr.concat([r['activity_trace_arr'] for r in at_list], dim='cell_roi_id')
    activity_trace = {
        'activity_trace_arr': at_arr,
        'timestamps':         at_list[0]['timestamps'],
        'time_bins':          at_list[0]['time_bins'],
        'ophys_frame_rate':   at_list[0]['ophys_frame_rate'],
    }
    design = DesignMatrix(activity_trace['timestamps'], activity_trace['ophys_frame_rate'])
    dmtools.add_kernels(design, run_params, bod_list[-1], activity_trace)
    X = design.get_X()
    return run_params, design, X, activity_trace


# ── multiprocessing parallel fit (replaces ray) ───────────────────────────────
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
    global _WORKER_DATA
    X_tr.load(); X_te.load(); y_tr.load(); y_te.load()
    _WORKER_DATA.update(
        run_params=run_params, fit_params=fit_params,
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, nested=nested,
    )
    models   = list(run_params['dropouts'].keys())
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


# ── QC figure 1: 3-panel summary ─────────────────────────────────────────────
def save_qc_summary(session_key, data_type, results, run_params, save_dir):
    ve_test  = np.asarray(results['var_explained_test_cv'])
    ve_train = np.asarray(results['var_explained_train_cv'])
    dropout_labels = list(run_params['dropouts'].keys())
    full_idx = dropout_labels.index('Full')

    ve_full_test  = np.nanmean(ve_test[:,  full_idx, :], axis=0)
    ve_full_train = np.nanmean(ve_train[:, full_idx, :], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. VE distribution
    ax = axes[0]
    ax.hist(ve_full_test, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(np.nanmedian(ve_full_test), color='orange', lw=1.5, ls='--',
               label=f'median={np.nanmedian(ve_full_test):.3f}')
    ax.set_xlabel('Variance explained (test)')
    ax.set_ylabel('Number of cells')
    ax.set_title('Full model VE')
    ax.legend(fontsize=8)

    # 2. Train vs test scatter
    ax = axes[1]
    ax.scatter(ve_full_train, ve_full_test, s=4, alpha=0.4, color='steelblue')
    lo = min(ve_full_train.min(), ve_full_test.min()) - 0.01
    hi = max(ve_full_train.max(), ve_full_test.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_xlabel('VE train')
    ax.set_ylabel('VE test')
    ax.set_title('Train vs Test VE')

    # 3. Kernel dropout contributions
    cf_mean, k_labels = [], []
    for di, label in enumerate(dropout_labels):
        if label == 'Full':
            continue
        ve_without = np.nanmean(ve_test[:, di, :], axis=0)
        cf_mean.append(float(np.nanmean(np.clip(ve_full_test - ve_without, 0, None))))
        k_labels.append(label)
    ax = axes[2]
    y_pos = np.arange(len(k_labels))
    ax.barh(y_pos, cf_mean, color='steelblue', edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(k_labels, fontsize=8)
    ax.set_xlabel('Mean unique VE')
    ax.set_title('Kernel contributions')
    ax.invert_yaxis()

    plt.suptitle(f'{session_key} | {data_type}', fontsize=10)
    plt.tight_layout()
    out = save_dir / f'qc_summary_{session_key}_{data_type}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'QC summary figure saved → {out.name}')


# ── QC figure 2: depth-sorted heatmap + behavioral traces ────────────────────
def save_heatmap_figure(session_key, data_type, results, save_dir,
                        proc_dir, raw_dir, eye_dir):
    # Load saved matrices
    at   = xr.open_dataarray(save_dir / f'{data_type}_activity_trace_matrix.nc', mmap=False)
    X_da = xr.open_dataarray(save_dir / 'design_matrix.nc', mmap=False)
    Y_raw    = np.asarray(at)
    cell_ids   = list(at.cell_roi_id.values)
    timestamps = np.asarray(at.timestamps)

    # Prediction via xarray @ so weight coordinates align regardless of order
    W_full  = results['W_cv'].mean(dim='test_fold_ind').sel(model='Full')
    Y_pred  = np.asarray(X_da @ W_full)
    Y_resid = Y_raw - Y_pred

    # Imaging depth from metadata.nd.json
    plane_depth = {}
    nd_path = proc_dir / 'metadata.nd.json'
    if nd_path.exists():
        try:
            fovs = json.loads(nd_path.read_text())['session']['data_streams'][0]['ophys_fovs']
            plane_depth = {f'VISp_{fov["index"]}': fov['imaging_depth'] for fov in fovs}
        except Exception as e:
            print(f'  Warning: could not parse metadata.nd.json ({e})')

    # Sort by imaging depth (fallback: plane index)
    cell_planes = [cid.rsplit('_', 1)[0] for cid in cell_ids]
    if plane_depth:
        cell_depths = np.array([plane_depth.get(p, 0) for p in cell_planes])
    else:
        cell_depths = np.array([int(p.split('_')[-1]) for p in cell_planes])
    sort_idx     = np.argsort(cell_depths, kind='stable')
    plane_sorted = [cell_planes[i] for i in sort_idx]

    Y_raw_s   = Y_raw[:,  sort_idx]
    Y_pred_s  = Y_pred[:, sort_idx]
    Y_resid_s = Y_resid[:, sort_idx]

    # Behavioral traces from saved unstd_features
    uf          = np.load(save_dir / 'unstd_features.npy', allow_pickle=True).item()
    running_trace = uf['running']
    pupil_trace   = uf['pupil']
    licks_trace   = uf['licks'].astype(float)

    # Reward timestamps from BOD (optional)
    reward_t_min = np.array([])
    try:
        from comb.behavior_ophys_dataset import BehaviorOphysDataset
        plane_dirs = sorted([p for p in proc_dir.iterdir() if _is_plane_dir(p)])
        bod0 = BehaviorOphysDataset(
            plane_folder_path=plane_dirs[0],
            raw_folder_path=raw_dir,
            eye_tracking_path=eye_dir,
            pipeline_version='v6',
        )
        reward_t_min = bod0.behavior_dataset.rewards['timestamps'].values.astype(float) / 60
    except Exception as e:
        print(f'  Warning: could not load rewards ({e}); reward ticks omitted')

    # Downsample 10×
    DS   = 10
    T_ds = len(timestamps) // DS
    t_min = timestamps[:T_ds * DS:DS] / 60
    xlim  = (t_min[0], t_min[-1])

    def _ds_mean(v): return v[:T_ds * DS].reshape(T_ds, DS).mean(axis=1)
    def _ds_max(v):  return v[:T_ds * DS].reshape(T_ds, DS).max(axis=1)
    def _ds2(M):     return M[:T_ds * DS].reshape(T_ds, DS, -1).mean(axis=1)

    Zraw   = _zscore_cols(_ds2(Y_raw_s))
    Zpred  = _zscore_cols(_ds2(Y_pred_s))
    Zresid = _zscore_cols(_ds2(Y_resid_s))
    run_ds  = _ds_mean(running_trace)
    pup_ds  = _ds_mean(pupil_trace)
    lick_ds = _ds_max(licks_trace)

    # Plane boundary positions for y-axis ticks
    bounds, prev = [], None
    for ci, pl in enumerate(plane_sorted):
        if pl != prev:
            if prev is not None:
                bounds.append((ci, plane_depth.get(pl, pl)))
            prev = pl

    bnd_cells = [0] + [b[0] for b in bounds]
    bnd_labels = []
    for i, ci in enumerate(bnd_cells):
        pl = plane_sorted[ci]
        d  = plane_depth.get(pl)
        bnd_labels.append(f'{d}µm' if d else pl)

    # Layout
    n_cells = Zraw.shape[1]
    hr_heat = max(n_cells // 25, 6)
    hr_beh  = max(hr_heat // 4, 3)
    hr = [hr_heat, hr_heat, hr_heat, hr_beh * 2, hr_beh * 2, hr_beh]
    clim = 2.5

    fig, axes = plt.subplots(6, 1, figsize=(18, sum(hr) * 0.13 + 1),
                             gridspec_kw={'height_ratios': hr, 'hspace': 0.25},
                             sharex=True)

    def _heatmap(ax, Z, ylabel):
        im = ax.imshow(Z.T, aspect='auto', origin='upper',
                       extent=[xlim[0], xlim[1], Z.shape[1], 0],
                       vmin=-clim, vmax=clim, cmap='RdBu_r', interpolation='nearest')
        for ci, _ in bounds:
            ax.axhline(ci, color='white', lw=0.5, ls='--')
        ax.set_yticks(bnd_cells)
        ax.set_yticklabels(bnd_labels, fontsize=6)
        ax.set_ylabel(ylabel, fontsize=9, labelpad=4)
        # Colorbar outside axis boundary so it does not steal width from the sharex group
        cbax = ax.inset_axes([1.005, 0.0, 0.012, 1.0])
        cb = fig.colorbar(im, cax=cbax)
        cb.set_label('z-score', fontsize=6)
        cb.ax.tick_params(labelsize=6)

    _heatmap(axes[0], Zraw,   'Events')
    _heatmap(axes[1], Zpred,  'GLM')
    _heatmap(axes[2], Zresid, 'Residual')

    ax = axes[3]
    ax.fill_between(t_min, run_ds, alpha=0.75, color='black', linewidth=0)
    ax.set_ylabel('Running\n(cm/s)', fontsize=8, labelpad=4)
    run_max = round(float(run_ds.max()), 1)
    ax.set_yticks([0, run_max])
    ax.set_yticklabels(['0', str(run_max)], fontsize=7)

    ax = axes[4]
    ax.plot(t_min, pup_ds, color='darkorange', lw=0.8)
    ax.set_ylabel('Pupil\n(a.u.)', fontsize=8, labelpad=4)
    p_lo = round(float(np.nanmin(pup_ds)), 1)
    p_hi = round(float(np.nanmax(pup_ds)), 1)
    ax.set_yticks([p_lo, p_hi])
    ax.set_yticklabels([str(p_lo), str(p_hi)], fontsize=7)

    ax = axes[5]
    ax.fill_between(t_min, lick_ds, alpha=0.75, color='tomato', linewidth=0)
    lk_max = max(int(lick_ds.max()), 1)
    if len(reward_t_min):
        rew_in = reward_t_min[(reward_t_min >= xlim[0]) & (reward_t_min <= xlim[1])]
        ax.vlines(rew_in, ymin=lk_max * 0.85, ymax=lk_max * 1.15,
                  color='royalblue', lw=1.2, zorder=5, clip_on=False, label='reward')
        ax.legend(fontsize=7, loc='upper right', framealpha=0.6)
    ax.set_ylabel('Licks\n(count)', fontsize=8, labelpad=4)
    ax.set_yticks([0, lk_max])
    ax.set_yticklabels(['0', str(lk_max)], fontsize=7)
    ax.set_xlabel('Time (min)', fontsize=8)

    plt.suptitle(f'{session_key} | {data_type} | z-scored per cell | sorted by imaging depth',
                 fontsize=10)
    out = save_dir / f'heatmap_{session_key}_{data_type}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Heatmap figure saved → {out.name}')


def _zscore_cols(M):
    mu  = np.nanmean(M, axis=0, keepdims=True)
    sig = np.nanstd(M,  axis=0, keepdims=True)
    sig[sig == 0] = 1
    return (M - mu) / sig


# ── processing.json ───────────────────────────────────────────────────────────
def save_processing_json(session_key, session_name, data_type,
                         kernel_config_path, kernel_dict,
                         fit_params, results_dir, save_dir,
                         start_time, end_time):
    doc = {
        'schema_version': '1.0',
        'session_key':    session_key,
        'session_name':   session_name,
        'data_type':      data_type,
        'start_time':     start_time.isoformat(),
        'end_time':       end_time.isoformat(),
        'duration_s':     round((end_time - start_time).total_seconds(), 1),
        'output_dir':     str(save_dir),
        'kernel_config': {
            'file':    str(kernel_config_path),
            'kernels': kernel_dict,
        },
        'fit_params': {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in fit_params.items()
        },
        'environment': {
            'python':     platform.python_version(),
            'numpy':      np.__version__,
            'xarray':     xr.__version__,
            'matplotlib': matplotlib.__version__,
        },
    }
    out = results_dir / 'processing.json'
    with open(out, 'w') as f:
        json.dump(doc, f, indent=2, default=str)
    print(f'processing.json saved → {out}')


# ── entry point ───────────────────────────────────────────────────────────────
def run():
    parser = argparse.ArgumentParser(
        description='Reproducible GLM fit for AIND mFISH multiplane-ophys sessions.')
    parser.add_argument(
        '--kernels_config', default=str(KERNEL_DIR / 'kernel_v01.json'),
        help='Path to kernel JSON file (default: kernel_json_files/kernel_v01.json)')
    parser.add_argument('--data_type', default='events', choices=['dff', 'events'])
    parser.add_argument('--cv_folds',             type=int,   default=5,
                        help='Number of outer cross-validation folds')
    parser.add_argument('--cv_nested_folds',      type=int,   default=5,
                        help='Number of inner (lambda-selection) CV folds')
    parser.add_argument('--n_lambdas',            type=int,   default=40,
                        help='Number of ridge lambda values in the log-spaced grid')
    parser.add_argument('--lambda_min',           type=float, default=1.0,
                        help='Minimum ridge lambda value')
    parser.add_argument('--lambda_max',           type=float, default=10000.0,
                        help='Maximum ridge lambda value')
    parser.add_argument('--min_activity_support', type=float, default=0.05,
                        help='Minimum fraction of active frames required to include a cell')
    parser.add_argument('--data_dir',    default=str(DATA_DIR))
    parser.add_argument('--results_dir', default=str(RESULTS_DIR))
    args = parser.parse_args()

    start_time  = datetime.datetime.now()
    data_dir    = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover session paths
    session_name, session_key, raw_dir, proc_dir, eye_dir = discover_session(data_dir)
    print(f'Session : {session_name}')
    print(f'Key     : {session_key}')
    print(f'Proc    : {proc_dir.name}')
    print(f'Eye     : {eye_dir.name}')

    # 2. Load kernel config; extract version number for output naming
    kernel_config_path = Path(args.kernels_config)
    if not kernel_config_path.is_absolute():
        kernel_config_path = (CODE_DIR / kernel_config_path).resolve()
    if not kernel_config_path.exists():
        raise FileNotFoundError(f'Kernel config not found: {kernel_config_path}')
    with open(kernel_config_path) as f:
        raw_kernel_dict = json.load(f)
    # drop metadata keys (prefixed with _)
    kernel_dict = {k: v for k, v in raw_kernel_dict.items() if not k.startswith('_')}
    for k in kernel_dict:
        kernel_dict[k].setdefault('num_weights', None)

    version_match = re.search(r'v(\d+)', kernel_config_path.stem)
    version = int(version_match.group(1)) if version_match else 1
    print(f'Kernels : {list(kernel_dict.keys())}  (config: {kernel_config_path.name}, version={version})')

    # 3. Build design matrix and activity trace
    save_dir = results_dir / f'{session_key}_glm_v{version:02}'
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output  : {save_dir}')

    dm_file = save_dir / 'design_matrix.nc'
    at_file = save_dir / f'{args.data_type}_activity_trace_matrix.nc'
    if dm_file.exists() and at_file.exists():
        print('Design matrix artifacts already exist — skipping build.')
    else:
        print('Loading planes...')
        bod_list = load_all_planes(proc_dir, raw_dir, eye_dir)
        print('Building design matrix...')
        run_params, design, X, activity_trace = build_design_matrix(
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

    # 4. Configure and run GLM
    gft.collect_fold_results_parallel = _collect_fold_results_mp

    fit_params = gft.default_fit_params()
    fit_params['cv_fold']        = args.cv_folds
    fit_params['cv_nested_fold'] = args.cv_nested_folds
    fit_params['L2_grid_num']    = args.n_lambdas
    fit_params['L2_grid_range']  = [args.lambda_min, args.lambda_max]

    X_load, at_arr, at_info, run_params_load, unstd_features, use_indices = \
        gft.load_data(session_key, args.data_type, version, load_path=results_dir)

    ophys_frame_rate         = at_info['ophys_frame_rate']
    X_trim                   = X_load[use_indices, :]
    at_trim                  = at_arr[use_indices, :]
    at_trim_filtered         = gft.filter_activity_trace_matrix(
        at_trim, prop_support_threshold=args.min_activity_support)
    print(f'Cells after filtering: {at_trim_filtered.shape[1]} / {at_trim.shape[1]}')

    stratified_list = gft.set_stratified_list(
        fit_params, X_load, unstd_features, use_indices, ophys_frame_rate)
    stratified_frames, cv_inds_stratified = gft.get_stratified_folds(fit_params, stratified_list)

    print(f'Fitting GLM ({fit_params["cv_fold"]}-fold CV, parallel over models)...')
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

    # 5. Load results for figure generation
    results_fn = save_dir / f'glm_results_v{version:02}_{session_key}_{args.data_type}.npy'
    results    = np.load(results_fn, allow_pickle=True).item()
    with open(save_dir / 'run_params.json') as f:
        run_params_fig = json.load(f)

    # 6. QC figures
    print('Generating QC figures...')
    save_qc_summary(session_key, args.data_type, results, run_params_fig, save_dir)
    save_heatmap_figure(session_key, args.data_type, results, save_dir,
                        proc_dir, raw_dir, eye_dir)

    # 7. processing.json
    end_time = datetime.datetime.now()
    save_processing_json(
        session_key, session_name, args.data_type,
        kernel_config_path, kernel_dict,
        fit_params, results_dir, save_dir,
        start_time, end_time,
    )


if __name__ == '__main__':
    run()
