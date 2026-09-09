"""
qc_figures.py — QC figure generation for the mFISH GLM capsule.

Figures produced:
  qc_summary_{session_key}_{data_type}.png   — VE distribution, train/test scatter,
                                               kernel dropout contributions (3 panels)
  heatmap_{session_key}_{data_type}.png      — z-scored actual / GLM / residual heatmaps
                                               (sorted by imaging depth) + behavioral traces
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr

from load_data import is_plane_dir


def _zscore_cols(M):
    mu  = np.nanmean(M, axis=0, keepdims=True)
    sig = np.nanstd(M,  axis=0, keepdims=True)
    sig[sig == 0] = 1
    return (M - mu) / sig


def save_qc_summary(session_key, data_type, results, run_params, save_dir):
    """3-panel QC figure: VE distribution, train vs test, kernel contributions."""
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
    out = Path(save_dir) / f'qc_summary_{session_key}_{data_type}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'QC summary figure saved → {out.name}')


def save_heatmap_figure(session_key, data_type, results, save_dir,
                        proc_dir, raw_dir, eye_dir):
    """Depth-sorted heatmap (events / GLM / residual) + behavioral traces."""
    save_dir = Path(save_dir)
    proc_dir = Path(proc_dir)
    raw_dir  = Path(raw_dir)
    eye_dir  = Path(eye_dir)

    # Load saved matrices
    at   = xr.open_dataarray(save_dir / f'{data_type}_activity_trace_matrix.nc', mmap=False)
    X_da = xr.open_dataarray(save_dir / 'design_matrix.nc', mmap=False)
    # Subset to cells the GLM was actually fitted on (test mode may cap cells)
    W_full   = results['W_cv'].mean(dim='test_fold_ind').sel(model='Full')
    fitted_cells = W_full.cell_roi_id.values
    at_fitted = at.sel(cell_roi_id=fitted_cells)

    Y_raw    = np.asarray(at_fitted)
    cell_ids   = list(at_fitted.cell_roi_id.values)
    timestamps = np.asarray(at_fitted.timestamps)

    # Prediction via xarray @ so weight coordinates align regardless of build order
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

    # Sort cells by imaging depth
    cell_planes = [cid.rsplit('_', 1)[0] for cid in cell_ids]
    cell_depths = (np.array([plane_depth.get(p, 0) for p in cell_planes])
                   if plane_depth
                   else np.array([int(p.split('_')[-1]) for p in cell_planes]))
    sort_idx     = np.argsort(cell_depths, kind='stable')
    plane_sorted = [cell_planes[i] for i in sort_idx]

    Y_raw_s   = Y_raw[:,  sort_idx]
    Y_pred_s  = Y_pred[:, sort_idx]
    Y_resid_s = Y_resid[:, sort_idx]

    # Behavioral traces from saved unstd_features (may be absent for minimal kernel sets)
    T = len(timestamps)
    uf            = np.load(save_dir / 'unstd_features.npy', allow_pickle=True).item()
    running_trace = uf.get('running', np.zeros(T))
    pupil_trace   = uf.get('pupil',   np.zeros(T))
    licks_trace   = uf.get('licks',   np.zeros(T, dtype=float))

    # Reward timestamps (optional — loads one BOD)
    reward_t_min = np.array([])
    try:
        from comb.behavior_ophys_dataset import BehaviorOphysDataset
        plane_dirs = sorted([p for p in proc_dir.iterdir() if is_plane_dir(p)])
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

    def _dsm(v): return v[:T_ds * DS].reshape(T_ds, DS).mean(axis=1)
    def _dsx(v): return v[:T_ds * DS].reshape(T_ds, DS).max(axis=1)
    def _ds2(M): return M[:T_ds * DS].reshape(T_ds, DS, -1).mean(axis=1)

    Zraw   = _zscore_cols(_ds2(Y_raw_s))
    Zpred  = _zscore_cols(_ds2(Y_pred_s))
    Zresid = _zscore_cols(_ds2(Y_resid_s))
    run_ds  = _dsm(running_trace)
    pup_ds  = _dsm(pupil_trace)
    lick_ds = _dsx(licks_trace)

    # Plane boundary tick positions
    bounds, prev = [], None
    for ci, pl in enumerate(plane_sorted):
        if pl != prev:
            if prev is not None:
                bounds.append((ci, plane_depth.get(pl, pl)))
            prev = pl
    bnd_cells  = [0] + [b[0] for b in bounds]
    bnd_labels = [
        f'{plane_depth[plane_sorted[ci]]}µm' if plane_depth else plane_sorted[ci]
        for ci in bnd_cells
    ]

    # Layout
    n_cells = Zraw.shape[1]
    hr_heat = max(n_cells // 25, 6)
    hr_beh  = max(hr_heat // 4, 3)
    hr   = [hr_heat, hr_heat, hr_heat, hr_beh * 2, hr_beh * 2, hr_beh]
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
        # Inset colorbar outside axis boundary — preserves width of all sharex panels
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

    plt.suptitle(
        f'{session_key} | {data_type} | z-scored per cell | sorted by imaging depth',
        fontsize=10)
    out = save_dir / f'heatmap_{session_key}_{data_type}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Heatmap figure saved → {out.name}')
