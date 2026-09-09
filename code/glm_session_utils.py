"""
glm_session_utils.py — Session discovery, plane loading, and parallel GLM fitting.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import multiprocessing
import numpy as np
import xarray as xr


# ── session discovery ─────────────────────────────────────────────────────────

def discover_session(data_dir: Path):
    """Locate raw, processed, and eye-tracking directories under data_dir.

    Returns (session_name, session_key, raw_dir, proc_dir, eye_dir).
    """
    candidates = sorted(Path(data_dir).iterdir())

    def _is_raw(p):
        return (p.is_dir()
                and re.match(r'^multiplane-ophys_\d+_\d{4}-\d{2}-\d{2}_', p.name)
                and '_processed' not in p.name
                and '_lp-eye'    not in p.name
                and '_dlc-eye'   not in p.name
                and '_stim'      not in p.name)

    raw_dirs = [p for p in candidates if _is_raw(p)]
    if len(raw_dirs) != 1:
        raise ValueError(
            f'Expected 1 raw session dir in {data_dir}, found {len(raw_dirs)}: '
            f'{[p.name for p in raw_dirs]}'
        )
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
    lp = [p for p in eye_dirs if '_lp-eye' in p.name]
    eye_dir = lp[-1] if lp else eye_dirs[-1]

    return session_name, session_key, raw_dir, proc_dir, eye_dir


# ── plane loading ─────────────────────────────────────────────────────────────

def is_plane_dir(p: Path) -> bool:
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
    """Load all ophys planes from a processed session directory in parallel."""
    plane_dirs = sorted([p for p in Path(proc_dir).iterdir() if is_plane_dir(p)])
    print(f'Loading {len(plane_dirs)} planes ({n_workers} threads): {[p.name for p in plane_dirs]}')
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_load_plane, pd, raw_dir, eye_dir): pd for pd in plane_dirs}
        bod_map = {}
        for fut, pd in futures.items():
            bod_map[pd.name] = fut.result()
            print(f'  loaded {pd.name}')
    return [bod_map[pd.name] for pd in plane_dirs]


# ── parallel GLM fitting (fork-based replacement for ray) ────────────────────

_WORKER_DATA: dict = {}


def _one_model_mp(model_label):
    import glm_fit_tools as gft
    d = _WORKER_DATA
    return gft.collect_model_results(
        d['run_params'], d['fit_params'],
        d['X_tr'], d['X_te'], d['y_tr'], d['y_te'],
        d['nested'], model_label)


def collect_fold_results_mp(run_params, fit_params,
                             X_tr, X_te, y_tr, y_te,
                             nested, num_cores=None):
    """Drop-in replacement for gft.collect_fold_results_parallel using fork+Pool.

    Uses fork so large shared arrays (X, y) are inherited via copy-on-write
    rather than being pickled through the queue.
    """
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
