"""
glm_cell_analysis.py — per-cell analysis and figure generation for GLM results.

Usage
-----
from glm_cell_analysis import GLMCellAnalysis

glm = GLMCellAnalysis(
    results_path='/results/800792_2025-08-14_glm_v01',
    session_key='800792_2025-08-14',
    data_type='events',
    version=1,
)
glm.save_top_cells(n=10)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import blended_transform_factory
import xarray as xr

warnings.filterwarnings('ignore')


class GLMCellAnalysis:
    """Load GLM results for one session and generate per-cell figures."""

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(self, results_path, session_key, data_type='events', version=1):
        self.results_path = Path(results_path)
        self.session_key  = session_key
        self.data_type    = data_type
        self.version      = version
        self._load()

    def _load(self):
        rp = self.results_path
        sk = self.session_key
        dt = self.data_type
        v  = self.version

        results_fn = rp / f'glm_results_v{v:02}_{sk}_{dt}.npy'
        self.results = np.load(results_fn, allow_pickle=True).item()

        with open(rp / 'run_params.json') as f:
            self.run_params = json.load(f)

        X_da = xr.open_dataarray(rp / 'design_matrix.nc', mmap=False)
        self.X_mat        = np.asarray(X_da)
        self.weight_labels = list(X_da.weights.values)

        at_da = xr.open_dataarray(rp / f'{dt}_activity_trace_matrix.nc', mmap=False)
        self.Y_raw    = np.asarray(at_da)
        self.cell_ids = list(at_da.cell_roi_id.values)
        self.timestamps = np.asarray(at_da.timestamps)

        at_info = np.load(rp / f'{dt}_activity_trace_info.npy', allow_pickle=True).item()
        self.fs = at_info.get('frame_rate') or at_info['ophys_frame_rate']

        dropout_labels   = list(self.run_params['dropouts'].keys())
        self._full_idx   = dropout_labels.index('Full')
        self._dropout_labels = dropout_labels

        W_full_da = self.results['W_cv'].mean(dim='test_fold_ind').sel(model='Full')
        self.W_mean  = W_full_da.values.T
        self.w_labels = list(W_full_da.weights.values)

        Y_pred_da    = X_da @ W_full_da
        self.Y_pred  = np.asarray(Y_pred_da)

        # W_full_da only covers fitted cells; subset Y_raw to match
        fitted_ids   = list(W_full_da.cell_roi_id.values)
        at_fitted    = at_da.sel(cell_roi_id=fitted_ids)
        self.Y_raw   = np.asarray(at_fitted)
        self.cell_ids = fitted_ids
        self.Y_resid  = self.Y_raw - self.Y_pred

        ve_test = np.asarray(self.results['var_explained_test_cv'])
        self.ve_test = np.nanmean(ve_test[:, self._full_idx, :], axis=0)

        self.kernel_info = self.run_params.get('input_kernel_dict', {})

        self._image_names = sorted(set(
            lbl.rsplit('_', 1)[0] for lbl in self.weight_labels
            if lbl.startswith('im')
        ))

        print(f'Loaded {len(self.cell_ids)} cells | '
              f'fs={self.fs:.2f} Hz | '
              f'median VE={np.nanmedian(self.ve_test):.3f}')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _kernel_weights(self, cell_idx, kernel_prefix):
        inds = [i for i, w in enumerate(self.w_labels)
                if w.startswith(kernel_prefix + '_')]
        if not inds:
            return None, None
        offset = self.kernel_info.get(kernel_prefix, {}).get('offset', 0)
        t = np.arange(len(inds)) / self.fs + offset
        return t, self.W_mean[cell_idx, inds]

    def _event_frames(self, feature_prefix):
        zero_lag = f'{feature_prefix}_0'
        if zero_lag not in self.weight_labels:
            return np.array([], dtype=int)
        col = self.weight_labels.index(zero_lag)
        return np.where(self.X_mat[:, col] > 0.5)[0]

    def _psth(self, cell_idx, event_frames, pre_s=0.5, post_s=1.5):
        pre  = int(np.round(pre_s  * self.fs))
        post = int(np.round(post_s * self.fs))
        T    = len(self.timestamps)
        snippets_raw, snippets_pred = [], []
        for f in event_frames:
            s, e = f - pre, f + post
            if s >= 0 and e < T:
                snippets_raw.append(self.Y_raw[s:e, cell_idx])
                snippets_pred.append(self.Y_pred[s:e, cell_idx])
        if not snippets_raw:
            return None, None, None, None, None
        A = np.array(snippets_raw)
        P = np.array(snippets_pred)
        n = A.shape[0]
        t = (np.arange(pre + post) - pre) / self.fs
        return (t,
                A.mean(axis=0), A.std(axis=0) / np.sqrt(n),
                P.mean(axis=0), P.std(axis=0) / np.sqrt(n))

    def cell_label(self, cell_idx):
        return f'{self.cell_ids[cell_idx]}  VE={self.ve_test[cell_idx]:.3f}'

    # ── combined single-cell figure ───────────────────────────────────────────

    def save_cell_figure(self, cell_idx, out_path):
        """
        Two-panel combined figure per cell:
          Panel 1: full traces with colored boxes marking zoomed windows;
                   dashed connecting lines to zoom subplots below
          Panel 2: all kernels + PSTHs in uniform grid
                   — images on top, omissions/hits/misses/behavioral below
                   — y-tick labels only on leftmost column
                   — no x-labels on kernel rows (PSTHs carry them)
                   — all kernel y-limits unified; all PSTH y-limits unified
        """
        stim_set   = set(self._image_names) | {'omissions', 'hits', 'misses'}
        beh_ks     = [k for k in self.kernel_info
                      if k != 'intercept' and not k.startswith('each')
                      and k not in stim_set]

        n_img      = len(self._image_names)
        other_stim = ['omissions', 'hits', 'misses']
        other_all  = other_stim + beh_ks
        n_other    = len(other_all)
        n_cols     = max(n_img, n_other, 1)

        fig_w = max(2.2 * n_cols, 18)
        fig = plt.figure(figsize=(fig_w, 20))
        fig.suptitle(
            f'{self.session_key} | {self.data_type} | {self.cell_label(cell_idx)}',
            fontsize=11, y=0.997)

        sfs = fig.subfigures(2, 1, height_ratios=[8, 7], hspace=0.04)

        # ── 1. Full trace + zoomed windows ───────────────────────────────
        inner = sfs[0].subfigures(2, 1, height_ratios=[3, 2], hspace=0.12)

        # left/right match the bottom gridspec so trace width aligns with kernel plots
        _L, _R = 0.05, 0.98
        inner[0].subplots_adjust(left=_L, right=_R, top=0.95, bottom=0.12)

        trace_axes = inner[0].subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.08})
        t_min    = self.timestamps / 60
        y_raw_c  = self.Y_raw[:, cell_idx]
        y_pred_c = self.Y_pred[:, cell_idx]
        y_res_c  = self.Y_resid[:, cell_idx]
        for ax, y, lbl, color in zip(
                trace_axes,
                [y_raw_c, y_pred_c, y_res_c],
                ['Actual', 'Model', 'Residual'],
                ['#333333', 'steelblue', 'tomato']):
            ax.plot(t_min, y, lw=0.4, color=color)
            ax.set_ylabel(lbl, fontsize=10)
            ax.tick_params(labelsize=9)
            ax.axhline(0, color='gray', lw=0.5, ls='--')
        trace_axes[-1].set_xlabel('Time (min)', fontsize=10)

        # Compute best 60 s windows
        T    = len(self.timestamps)
        win  = max(int(60 * self.fs), 1)
        step = max(1, win // 4)
        thirds = [(0, T // 3), (T // 3, 2 * T // 3), (2 * T // 3, T)]
        zoom_colors  = ['#e69f00', '#56b4e9', '#009e73']
        third_labels = ['First third', 'Middle third', 'Last third']

        windows = []
        for (s0, s1) in thirds:
            best_var, best_s = -1, s0
            for s in range(s0, max(s0 + 1, s1 - win + 1), step):
                v = np.var(y_raw_c[s:s + win])
                if v > best_var:
                    best_var, best_s = v, s
            windows.append((best_s, min(best_s + win, T)))

        # Colored boxes in full trace marking each zoom window
        for (ws, we), color in zip(windows, zoom_colors):
            t_s = self.timestamps[ws] / 60
            t_e = self.timestamps[min(we, T) - 1] / 60
            for trace_ax in trace_axes:
                trace_ax.axvspan(t_s, t_e, alpha=0.12, color=color, zorder=0)
                trans = blended_transform_factory(trace_ax.transData, trace_ax.transAxes)
                rect = mpatches.Rectangle(
                    (t_s, 0), t_e - t_s, 1,
                    transform=trans, fill=False,
                    edgecolor=color, lw=1.2, clip_on=False, zorder=3)
                trace_ax.add_patch(rect)

        # Zoom subplots with actual session timestamps
        inner[1].subplots_adjust(left=_L, right=_R, top=0.82, bottom=0.18)
        zoom_axes = inner[1].subplots(1, 3, gridspec_kw={'wspace': 0.12})
        for ax, (ws, we), lbl, color in zip(zoom_axes, windows, third_labels, zoom_colors):
            t_actual = self.timestamps[ws:we] / 60
            ax.plot(t_actual, y_raw_c[ws:we],  lw=0.7, color='#333333', label='actual')
            ax.plot(t_actual, y_pred_c[ws:we], lw=0.7, color='steelblue', label='model')
            ax.axhline(0, color='gray', lw=0.5, ls='--')
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel('Time (min)', fontsize=9)
            ax.tick_params(labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)
            if ax is zoom_axes[0]:
                ax.set_ylabel(self.data_type, fontsize=9)
                ax.legend(fontsize=8)
            else:
                ax.tick_params(labelleft=False)
        inner[1].suptitle('Highest-variance 60 s window per third', fontsize=11)

        # Dashed connecting lines from box corners to zoom subplot corners
        for ax_zoom, (ws, we), color in zip(zoom_axes, windows, zoom_colors):
            t_s    = self.timestamps[ws] / 60
            t_e    = self.timestamps[min(we, T) - 1] / 60
            ylim_t = trace_axes[-1].get_ylim()
            xlim_z = ax_zoom.get_xlim()
            ylim_z = ax_zoom.get_ylim()
            for xy_t, xy_z in [
                ((t_s, ylim_t[0]), (xlim_z[0], ylim_z[1])),
                ((t_e, ylim_t[0]), (xlim_z[1], ylim_z[1])),
            ]:
                con = ConnectionPatch(
                    xyA=xy_t, coordsA='data', axesA=trace_axes[-1],
                    xyB=xy_z, coordsB='data', axesB=ax_zoom,
                    color=color, lw=0.9, linestyle='--', alpha=0.75,
                )
                fig.add_artist(con)

        # Font size constants for the kernel/PSTH panel
        FS_TITLE  = 10
        FS_LABEL  =  9
        FS_TICK   =  8
        FS_LEGEND =  8

        # ── 2. All kernels + PSTHs in uniform grid ───────────────────────
        # hspace large enough so kernel x-labels don't overlap PSTH content
        gs = sfs[1].add_gridspec(5, n_cols,
                                  height_ratios=[1, 1, 0.6, 1, 1],
                                  hspace=0.55, wspace=0.08,
                                  top=0.95, bottom=0.02, left=0.05, right=0.98)

        all_k_axes = []
        all_p_axes = []

        def _draw_kernel(row, col, name, first_col=False):
            ax = sfs[1].add_subplot(gs[row, col])
            t_k, w = self._kernel_weights(cell_idx, name)
            if t_k is not None:
                ax.plot(t_k, w, color='navy', lw=1.5)
                ax.axhline(0, color='gray', lw=0.5, ls='--')
                ax.axvline(0, color='gray', lw=0.5, ls=':')
                all_k_axes.append(ax)
            ax.set_title(name, fontsize=FS_TITLE)
            ax.set_xlabel('Time (s)', fontsize=FS_LABEL)
            ax.tick_params(labelsize=FS_TICK)
            if first_col:
                ax.set_ylabel('Kernel\nWeight', fontsize=FS_LABEL)
            else:
                ax.tick_params(labelleft=False)

        def _draw_psth(row, col, name, post_s=1.5, first_col=False):
            ax = sfs[1].add_subplot(gs[row, col])
            ev  = self._event_frames(name)
            out = self._psth(cell_idx, ev, pre_s=0.5, post_s=post_s)
            if out[0] is not None:
                t_p, mu_r, sem_r, mu_p, sem_p = out
                ax.fill_between(t_p, mu_r - sem_r, mu_r + sem_r, alpha=0.2, color='#333333')
                ax.plot(t_p, mu_r, color='#333333', lw=1.2, label='actual')
                ax.fill_between(t_p, mu_p - sem_p, mu_p + sem_p, alpha=0.2, color='steelblue')
                ax.plot(t_p, mu_p, color='steelblue', lw=1.2, label='model')
                ax.axvline(0, color='gray', lw=0.5, ls=':')
                ax.axhline(0, color='gray', lw=0.5, ls='--')
                if first_col:
                    ax.legend(fontsize=FS_LEGEND)
                all_p_axes.append(ax)
            ax.set_xlabel('Time (s)', fontsize=FS_LABEL)
            ax.tick_params(labelsize=FS_TICK)
            if first_col:
                ax.set_ylabel(f'{self.data_type}\nMean resp.', fontsize=FS_LABEL)
            else:
                ax.tick_params(labelleft=False)

        # Rows 0–1: images
        for c, name in enumerate(self._image_names):
            _draw_kernel(0, c, name, first_col=(c == 0))
            _draw_psth(1, c, name, post_s=1.5, first_col=(c == 0))

        # Row 2: spacer | Rows 3–4: omissions/hits/misses/behavioral
        for c, name in enumerate(other_all):
            _draw_kernel(3, c, name, first_col=(c == 0))
            if c < len(other_stim):
                post_s = 2.5 if name in ('hits', 'misses') else 1.5
                _draw_psth(4, c, name, post_s=post_s, first_col=(c == 0))

        # Section labels
        sfs[1].text(0.01, 0.98, 'Images', fontsize=12, fontweight='bold',
                    va='top', transform=sfs[1].transSubfigure)
        sfs[1].text(0.01, 0.47, 'Omissions / Hits / Misses / Behavioral',
                    fontsize=12, fontweight='bold', va='top',
                    transform=sfs[1].transSubfigure)

        # Unified y-limits
        if all_k_axes:
            klims = [ax.get_ylim() for ax in all_k_axes]
            for ax in all_k_axes:
                ax.set_ylim(min(l[0] for l in klims), max(l[1] for l in klims))

        if all_p_axes:
            plims = [ax.get_ylim() for ax in all_p_axes]
            for ax in all_p_axes:
                ax.set_ylim(min(l[0] for l in plims), max(l[1] for l in plims))

        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── save top-N cells ──────────────────────────────────────────────────────

    def save_top_cells(self, n=10, out_dir=None):
        """
        Generate and save one combined figure per top-n cell (by test-set VE).
        Saves to out_dir (default: results_path/top_cells/).
        """
        if out_dir is None:
            out_dir = self.results_path / 'top_cells'
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        top_idx = np.argsort(self.ve_test)[::-1][:n]
        print(f'Saving figures for top {n} cells → {out_dir}')

        saved = []
        for rank, ci in enumerate(top_idx):
            cid = self.cell_ids[ci]
            ve  = self.ve_test[ci]
            stem = f'rank{rank+1:02d}_{cid}_ve{ve:.3f}'
            print(f'  [{rank+1}/{n}] {cid}  VE={ve:.3f}')
            p = out_dir / f'{stem}.png'
            self.save_cell_figure(ci, p)
            saved.append(p)

        print(f'Done. {len(saved)} figures saved to:\n  {out_dir}')
        return saved


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', required=True)
    parser.add_argument('--session_key',  required=True)
    parser.add_argument('--data_type',    default='events')
    parser.add_argument('--version',      type=int, default=1)
    parser.add_argument('--n',            type=int, default=10)
    parser.add_argument('--out_dir',      default=None)
    args = parser.parse_args()

    glm = GLMCellAnalysis(
        results_path=args.results_path,
        session_key=args.session_key,
        data_type=args.data_type,
        version=args.version,
    )
    glm.save_top_cells(n=args.n, out_dir=args.out_dir)
