import xarray as xr
import numpy as np
from pathlib import Path

import glm_fit_tools as gft


def generate_session_model_adjusted_variance_explained(glm_results, run_params, X_trim,
                                                       activity_trace_trim_filtered):
    """ Generate adjusted variance explained for each model in a session.
    Adjusting for the variance explained by the full model, only at the time points where the model has support.
    (i.e. the input matrix weights are non-zero)
    Also calculate the adjusted variance explained for the full model with the mask of the partial model.

    for now, use mean W model. The difference between VE of mean model and mean of VE across test fold is small.
    #TODO: think about alternative (e.g., concatenated model from each fold)
    """    
    # building session model
    mean_W = glm_results['W_cv'].mean(dim='test_fold_ind')
    ve_session_model = glm_results['var_explained_mean_model']
    
    # calculating adjusted variance explained for each model
    adj_var_explained_session_model = xr.full_like(ve_session_model,
                                               fill_value=np.nan).drop_sel(model=['Full', 'intercept'])
    adj_var_explained_session_model_full_mask = xr.full_like(ve_session_model,
                                                fill_value=np.nan).drop_sel(model=['Full', 'intercept'])

    all_weights = mean_W.weights.values
    all_models = adj_var_explained_session_model.model.values
    for model in all_models:
        run_param = run_params['dropouts'][model]
        if run_param['is_single']:
            kernels = run_param['kernels']
        else:
            kernels = run_param['dropped_kernels']
        # remove intercept from kernels for the mask
        kernels = np.setdiff1d(kernels, ['intercept'])
        
        mask_weights = [w for w in all_weights if np.any([mk in w for mk in kernels])]
        mask_trim = (X_trim.sel(weights=mask_weights)!=0).any(dim='weights')
        if mask_trim.sum() == X_trim.sizes['timestamps']:
            adj_var_explained_session_model.loc[{'model': model}] = ve_session_model.sel(model=model)
            adj_var_explained_session_model_full_mask.loc[{'model': model}] = ve_session_model.sel(model='Full')
        else:
            if run_param['is_single']:
                run_weights = ['intercept_0', *mask_weights]
            else:
                run_weights = np.setdiff1d(all_weights, mask_weights)
            
            mask_cv = (X_trim.sel(weights=mask_weights)!=0).any(dim='weights')
            W_model = mean_W.sel(model=model, weights=run_weights)
            X_model = X_trim.sel(weights=run_weights)
            assert np.isnan(W_model).any() == False
            
            adj_var_explained_session_model.loc[{'model': model}] = \
                gft.compute_adjusted_variance_explained(activity_trace_trim_filtered, W_model, X_model, mask_cv)
            
            W_full = mean_W.sel(model='Full')
        
            adj_var_explained_session_model_full_mask.loc[{'model': model}] = \
                gft.compute_adjusted_variance_explained(activity_trace_trim_filtered, W_full, X_trim, mask_cv)
    return adj_var_explained_session_model, adj_var_explained_session_model_full_mask


def calculate_coding_score(adj_var_explained_session_model,
                           adj_var_explained_session_model_full_mask,
                           run_params,
                           filter_threshold=0.005):
    """ Calculate coding score for each model in a session.
    For single models, -VE/VE_full (at the support)
        How much variance explained using single kernel, 
        normalized to that of the full model.        
    For dropout models, -(1 - VE/VE_full) (at the support)
        How much variance explained is reduced compared to the full model, 
        normalized to that of the full model.        
    Filter based on the threshold, then clip to [-1, 0]
    Negative sign as convention
    """
    coding_score_session_model = xr.full_like(adj_var_explained_session_model, 
                                              fill_value=np.nan)
    all_models = adj_var_explained_session_model.model.values
    single_models = [m for m in all_models if run_params['dropouts'][m]['is_single']]
    dropout_models = [m for m in all_models if not run_params['dropouts'][m]['is_single']]

    coding_score_session_model.loc[{'model': single_models}] = \
                -adj_var_explained_session_model.sel(model=single_models) / \
                adj_var_explained_session_model_full_mask.sel(model=single_models)
    coding_score_session_model.loc[{'model': dropout_models}] = \
                -(1 - (adj_var_explained_session_model.sel(model=dropout_models) / \
                adj_var_explained_session_model_full_mask.sel(model=dropout_models)))

    # filtering
    coding_score_session_model_filtered_before_clipping = coding_score_session_model.copy()
    session_adjVE_mask = adj_var_explained_session_model < filter_threshold
    session_adjVE_full_mask_mask = adj_var_explained_session_model_full_mask < filter_threshold
    coding_score_session_model_filtered_before_clipping = \
            coding_score_session_model_filtered_before_clipping.where(
            ~(session_adjVE_mask + session_adjVE_full_mask_mask), other=0)

    # clipping
    coding_score_session_model_filtered = \
            coding_score_session_model_filtered_before_clipping.clip(min=-1, max=0)
    
    return coding_score_session_model_filtered


def cross_session_normalization(matched_roi_df, session_info_df,
                    dm_version=3, data_type='events', suffix='',
                    glm_dir=Path('/root/capsule/scratch/glm'),
                    include_models=['all-images', 'omissions', 'task', 'behavioral'],
                    filter_threshold=0.005
                    ):
    """ Normalize coding scores across sessions.
    Start from adjusted variance, not from coding score calculated in each session.
    """
    coding_score_table = matched_roi_df.reset_index()[['session_roi_name',
                                            'unique_roi_name', 'session_ind',
                                            'session_key', 'fov_name',
                                            'roi_session_index']].copy()
    session_keys = np.sort(coding_score_table.session_key.unique())

    coding_score_raw = None
    for session_key in session_keys:
        session_matched_roi_name = ['_'.join(rn.split('_')[-3:]) for rn in \
            matched_roi_df.query(f'session_key=="{session_key}"').index.values]
        
        # session_name = session_info_df.loc[session_key, 'session_name']
        # session_ind = session_info_df.loc[session_key, 'session_ind']
        coding_score_fn = Path(glm_dir) / f'{session_key}_glm_v{dm_version:02}/\
coding_score_v{dm_version:02}_{session_key}_{data_type}{suffix}.nc'
        with xr.open_dataarray(coding_score_fn) as ds:
            coding_score_session = ds.expand_dims(
                                    'session').assign_coords(
                                        session=[session_key])
            cell_roi_ids = coding_score_session.cell_roi_id.values
            
            # get filtered cell_roi_ids during glm fitting (<1% active frames)
            X, activity_trace, activity_trace_info, run_params, unstd_features, use_indices = \
                gft.load_data(session_key, data_type, dm_version, load_path=glm_dir)
            activity_trace_trim = activity_trace[use_indices, :]
            activity_trace_trim_filtered = gft.filter_activity_trace_matrix(activity_trace_trim)
            filtered_cell_roi_id = np.setdiff1d(activity_trace_trim.cell_roi_id.values,
                                    activity_trace_trim_filtered.cell_roi_id.values)
            
            # if there are missing cell_roi_ids among filtered cell_roi_ids,
            # add them to coding_score_session and fill with 0
            missing_cell_roi_ids = np.setdiff1d(filtered_cell_roi_id, cell_roi_ids)
            if len(missing_cell_roi_ids) > 0:
                new_cell_roi_ids = np.concatenate([cell_roi_ids, missing_cell_roi_ids])
                coding_score_session = coding_score_session.reindex(
                                        cell_roi_id=new_cell_roi_ids, fill_value=0)
            
            # Choose only the matched cell_roi_ids
            # Also only from models of interest
            coding_score_session = coding_score_session.sel(
                                    cell_roi_id=session_matched_roi_name,
                                    model=include_models).copy()
            
            # Swap cell_roi_id which is session-specific to unique_roi_name
            # for concatenation across sessions
            unique_roi_names = []
            for cell_roi_id in coding_score_session.cell_roi_id.values:
                roi_name = f'{session_key}_{cell_roi_id}'
                unique_roi_names.append(matched_roi_df.loc[
                                    roi_name, 'unique_roi_name'])

            coding_score_session = coding_score_session.assign_coords(
                                    unique_roi_name=(
                                        "cell_roi_id", unique_roi_names))
            coding_score_session = coding_score_session.swap_dims(
                                    {"cell_roi_id": "unique_roi_name"})
            coding_score_session = coding_score_session.reset_coords(
                                    'cell_roi_id', drop=True)
        if coding_score_raw is None:
            coding_score_raw = coding_score_session
        else:
            coding_score_raw = xr.concat([coding_score_raw, coding_score_session],
                                dim='session')
            
    max_full_score = coding_score_raw.max(dim='session')  # for filtering, adj_ve_full only
    assert len(np.where(coding_score_raw.isnull())[0]) == 0
    
    # Normalized coding score
    coding_score_normalized = \
        -((coding_score_raw.sel(metric='adj_ve_full') - \
            coding_score_raw.sel(metric='adj_ve_model')) / \
                coding_score_raw.sel(metric='adj_ve_full').max(dim='session'))
        
    # filtering
    mask_from_max = max_full_score.sel(metric='adj_ve_full') < filter_threshold
    sessions = coding_score_raw.session.values
    mask_from_max = xr.concat([mask_from_max] * len(sessions), dim="session")
    mask_from_max = mask_from_max.assign_coords(session=sessions)
    mask_from_raw = coding_score_raw.sel(metric='adj_ve_model') < filter_threshold

    coding_score_normalized_filtered = coding_score_normalized.where(~(mask_from_max + mask_from_raw), other=0)

    # clipping
    coding_score_normalized_filtered_clipped = coding_score_normalized_filtered.clip(min=-1, max=0)
    assert len(np.where(coding_score_normalized_filtered_clipped.isnull())[0]) == 0
    coding_score_normalized_filtered_clipped = \
        coding_score_normalized_filtered_clipped.reset_coords('metric', drop=True)

    return coding_score_normalized_filtered_clipped


####################################################################################################
# Specific to Novelty project (FNN for now)
def concatenate_normalized_coding_scores(mouse_ids,
                                dm_version=3, data_type='events', suffix='',
                                glm_dir=Path('/root/capsule/scratch/glm')):
    """ Concatenate normalized coding scores across mice.
    Assume that session keys are already visually validated (from a notebook).
    Unique_roi_name is unique across mice.
    Make sure there is no nan value.
    """
    coding_score_normalized_concat = None
    for mouse_id in mouse_ids:
        cs_norm_fn = f'coding_score_fnn_normalized_v{dm_version:02}_{mouse_id}_{data_type}{suffix}.nc'
        with xr.open_dataarray(glm_dir / cs_norm_fn) as ds:
            ds = swap_dims_for_fnn(ds)
        if coding_score_normalized_concat is None:
            coding_score_normalized_concat = ds
        else:
            coding_score_normalized_concat = xr.concat([coding_score_normalized_concat, ds],
                                            dim=('unique_roi_name'))
    assert len(np.where(np.isnan(coding_score_normalized_concat.values))[0]) == 0
    return coding_score_normalized_concat


def swap_dims_for_fnn(ds):
    """ Change dimension from session_key to session, for FNN analysis.
    Assume key-name relationship has been validated.    
    """
    session_keys = np.sort(ds.session.values)
    experience_levels = ['Familiar', 'Novel', 'Novel+1']
    experience_level_dict = {k: v for k, v in zip(session_keys, experience_levels)}
    ds = ds.assign_coords(experience_level=('session', [experience_level_dict[s] for s in session_keys]))
    ds = ds.swap_dims({'session': 'experience_level'}).reset_coords('session', drop=True)
    return ds
