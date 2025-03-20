import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import skimage
import glob
import os
import xarray as xr
    
from pathlib import Path
import json
from typing import Union, Dict
import h5py
from skimage import measure, io
from skimage.measure import regionprops
import ray

import aind_session
from aind_ophys_data_access import capsule
from lamf_analysis.code_ocean import capsule_data_utils as cdu

import glm_fit_tools as gft
import kernel_tools as ktools
import load_data
import design_matrix_tools as dmtools
from DesignMatrix import DesignMatrix
import coding_score as cs


###############################################################
# Wrapper for running both generating input matrices and fitting 
# for all the session from a mouse
# for natural images sessions only for now
def process_mouse_glm(mouse_id, version, data_type,
                    glm_path=Path('/root/capsule/scratch/glm/'),
                    natural_images_only=True,
                    parallel=True):
    if isinstance(glm_path, str):
        glm_path = Path(glm_path)
    processed_date_after = None
    processed_date_before = None
    if mouse_id == 747107:
        processed_date_after = '2025-02-01' # inclusive
    elif mouse_id == 755252:
        processed_date_after = '2025-02-07' # inclusive
    elif mouse_id == 736963:
        processed_date_before = '2024-12-31' # inclusive

    success, mouse_session_df = cdu.get_mouse_session_df(mouse_id,
                                                        processed_date_after=processed_date_after,
                                                        processed_date_before=processed_date_before)
    # assert success 
    # can't assert all the sessions attached because there are cortical z-stack sessions
    # they don't have pupil and ophys processing
    # Remove them
    if success == False:
        # either one of two reasons - duplicated raw data and missing pupil data
        # If duplicated raw data, then fail
        # If missing pupil data, then remove them
        if len(np.where(mouse_session_df.num_raw_data_asset_ids.values != 1)[0]) > 0:
            raise ValueError('Multiple raw data asset ids found for a single processed data asset id')
        else:
            missing_pupil_data = np.where(mouse_session_df.pupil_data_asset_id.values == 0)[0]
            if len(missing_pupil_data) > 0:
                print(f'Missing pupil data found for {mouse_session_df.iloc[missing_pupil_data].raw_data_date.values}')
                mouse_session_df = mouse_session_df.drop(missing_pupil_data)
    assert cdu.attach_mouse_data_assets(mouse_session_df)
    session_info_df = cdu.get_session_info(mouse_id)

    if natural_images_only:
        session_info_df = session_info_df[session_info_df.stimulus.str.startswith('images_')]
        # Another temporary filter - remove extinction sessions for now
        # Due to stratification issues (with hits) - need to solve this first
        session_info_df = session_info_df[~session_info_df.session_type.str.contains('OPHYS_6_')]

    for session_key, row in session_info_df.iterrows():
        fit_fn = glm_path / f'{session_key}_glm_v{version:02}' / \
            f'glm_results_v03_{session_key}_{data_type}.npy'
        if os.path.exists(fit_fn):
            print(f'{session_key} already processed')
        else:
            print(f'Processing {session_key}')
            raw_path = row.raw_path
            generate_and_save_matrices(session_key, raw_path, data_type, version, glm_path)
            fit_and_save_session(session_key, data_type, version, glm_path,
                                 parallel=parallel)
            print(f"Done: {session_key} ({row.session_type})")
            print('---------------------------------------\n')


def process_mouse_coding_score(mouse_id, version, data_type, glm_path):
    session_info_df = cdu.get_session_info(mouse_id)
    for session_key, row in session_info_df.iterrows():
        calculate_and_save_coding_score(session_key, version, data_type, glm_path)

####################################################################################################
# Specific utils for running GLM

def build_input_kernels():
    # Passive change removed for now. (03/07/2025)
    kernels = {
        'intercept':    {'feature':'intercept',   'type':'continuous',    'length':0,     'offset':0,     'num_weights':None, 'dropout':True, 'text': 'constant value'},
        'hits':         {'feature':'hit',         'type':'discrete',      'length':2.25,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'lick to image change'},
        'misses':       {'feature':'miss',        'type':'discrete',      'length':2.25,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'no lick to image change'},
        # 'passive_change':   {'feature':'passive_change','type':'discrete','length':2.25,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'passive session image change'},
        # 'hits':         {'feature':'hit',         'type':'discrete',      'length':1.5,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'lick to image change'},
        # 'misses':       {'feature':'miss',        'type':'discrete',      'length':1.5,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'no lick to image change'},
        # 'passive_change':   {'feature':'passive_change','type':'discrete','length':1.5,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'passive session image change'},
        #'hits':         {'feature':'hit',         'type':'discrete',      'length':.75,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'lick to image change'},
        #'misses':       {'feature':'miss',        'type':'discrete',      'length':.75,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'no lick to image change'},
        #'passive_change':   {'feature':'passive_change','type':'discrete','length':.75,   'offset':0,    'num_weights':None, 'dropout':True, 'text': 'passive session image change'},
        #'post-hits':    {'feature':'hit',         'type':'discrete',      'length':1.5,   'offset':0.75,    'num_weights':None, 'dropout':True, 'text': 'lick to image change'},
        #'post-misses':  {'feature':'miss',        'type':'discrete',      'length':1.5,   'offset':0.75,    'num_weights':None, 'dropout':True, 'text': 'no lick to image change'},
        #'post-passive_change': {'feature':'passive_change','type':'discrete','length':1.5,   'offset':0.75,    'num_weights':None, 'dropout':True, 'text': 'passive session image change'},
        'omissions':        {'feature':'omissions',   'type':'discrete',  'length':3,      'offset':0,     'num_weights':None, 'dropout':True, 'text': 'image was omitted'},
        # 'omissions':        {'feature':'omissions',   'type':'discrete',  'length':1.5,      'offset':0,     'num_weights':None, 'dropout':True, 'text': 'image was omitted'},
        #'omissions':        {'feature':'omissions',   'type':'discrete',  'length':0.75,      'offset':0,     'num_weights':None, 'dropout':True, 'text': 'image was omitted'},
        #'post-omissions':   {'feature':'omissions',   'type':'discrete',  'length':2.25,   'offset':0.75,  'num_weights':None, 'dropout':True, 'text': 'images after omission'},
        'each-image':   {'feature':'each-image',  'type':'discrete',      'length':0.75,  'offset':0,     'num_weights':None, 'dropout':True, 'text': 'image presentation'},
        # 'running':      {'feature':'running',     'type':'continuous',    'length':1,     'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'normalized running speed'},
        # 'pupil':        {'feature':'pupil',       'type':'continuous',    'length':1,     'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'Z-scored pupil diameter'},
        # 'licks':        {'feature':'licks',       'type':'discrete',      'length':1,     'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'mouse lick'},
        'running':      {'feature':'running',     'type':'continuous',    'length':2,     'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'normalized running speed'},
        'pupil':        {'feature':'pupil',       'type':'continuous',    'length':2,     'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'Z-scored pupil diameter'},
        'licks':        {'feature':'licks',       'type':'discrete',      'length':2,     'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'mouse lick'},
        #'false_alarms':     {'feature':'false_alarm',   'type':'discrete','length':5.5,   'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'lick on catch trials'},
        #'correct_rejects':  {'feature':'correct_reject','type':'discrete','length':5.5,   'offset':-1,    'num_weights':None, 'dropout':True, 'text': 'no lick on catch trials'},
        #'time':         {'feature':'time',        'type':'continuous',    'length':0,     'offset':0,    'num_weights':None,  'dropout':True, 'text': 'linear ramp from 0 to 1'},
        #'beh_model':    {'feature':'beh_model',   'type':'continuous',    'length':.5,    'offset':-.25, 'num_weights':None,  'dropout':True, 'text': 'behavioral model weights'},
        #'lick_bouts':   {'feature':'lick_bouts',  'type':'discrete',      'length':4,     'offset':-2,   'num_weights':None,  'dropout':True, 'text': 'lick bout'},
        #'lick_model':   {'feature':'lick_model',  'type':'continuous',    'length':2,     'offset':-1,   'num_weights':None,  'dropout':True, 'text': 'lick probability from video'},
        #'groom_model':  {'feature':'groom_model', 'type':'continuous',    'length':2,     'offset':-1,   'num_weights':None,  'dropout':True, 'text': 'groom probability from video'},
    }
    ## add face motion energy PCs
    # for PC in range(5):
    #     kernels['face_motion_PC_{}'.format(PC)] = {'feature':'face_motion_PC_{}'.format(PC), 'type':'continuous', 'length':2, 'offset':-1, 'dropout':True, 'text':'PCA from face motion videos'}
    return kernels

#######################################
# For generating input matrices
def generate_activity_trace_and_design_matrices(bod_list, data_type='events'):
    ###################################################
    # Generate run_params, design matrix, and activity traces
    run_params = {'data_type': data_type}
    input_kernel_dict = build_input_kernels()

    activity_trace_list = []
    run_params_list = []
    for bod in bod_list:
        run_params = ktools.process_kernels(input_kernel_dict, run_params, bod)
        activity_trace, run_params = \
            load_data.extract_and_annotate_ophys_plane(bod, run_params)
        activity_trace_list.append(activity_trace)
        run_params_list.append(run_params)
    
    # assert all run_params are the same
    assert all([run_params == run_params_list[0] for run_params in run_params_list])
    timestamps = [r['timestamps'] for r in activity_trace_list]
    assert all([(ts == timestamps[0]).all() for ts in timestamps])
    timebins = [r['time_bins'] for r in activity_trace_list]
    assert all([(tb == timebins[0]).all() for tb in timebins])
    ophys_frame_rates = [r['ophys_frame_rate'] for r in activity_trace_list]
    assert ~np.diff(ophys_frame_rates).any()
    
    # run params should be the same across planes
    run_params = run_params_list[0]
    run_params['input_kernel_dict'] = input_kernel_dict  # add for future use

    # activity_trace session array from concatenating all planes
    activity_trace_session_arr = xr.concat([r['activity_trace_arr'] for r in activity_trace_list], dim='cell_roi_id')

    # set up a activity_trace dictionary for adding kernels
    activity_trace = {}
    activity_trace['activity_trace_arr'] = activity_trace_session_arr
    activity_trace['timestamps'] = timestamps[0]
    activity_trace['time_bins'] = timebins[0]
    activity_trace['ophys_frame_rate'] = ophys_frame_rates[0]

    # activity_trace archive for saving
    activity_trace_archive = [r['stimulus_interpolation'] for r in activity_trace_list]

    # activity_trace matrix for saving
    activity_trace_matrix = activity_trace['activity_trace_arr']

    # activity_trace info dictionary for saving
    activity_trace_info = {}
    activity_trace_info['timestamps'] = activity_trace['timestamps']
    activity_trace_info['time_bins'] = activity_trace['time_bins']
    activity_trace_info['ophys_frame_rate'] = activity_trace['ophys_frame_rate']
    
    # add kernels to the designMatrix class
    design = DesignMatrix(activity_trace['timestamps'], activity_trace['ophys_frame_rate'])
    dmtools.add_kernels(design, run_params, bod, activity_trace) # need the trace array for kernels like population mean or PC1. 
    
    # assertion
    X = design.get_X()
    assert X.shape[0] == activity_trace['activity_trace_arr'].shape[0]

    return run_params, design, activity_trace_matrix, activity_trace_info, activity_trace_archive


def generate_and_save_matrices(session_key, raw_path, data_type, version, glm_path):
    ''' Generate and save design matrix and activity trace matrix for a session
    
    Parameters
    ----------
    session_key : str
        Key for the session
    raw_path : str
        Path to the raw data folder for the session
    data_type : str
        'events' or 'dff'
    version : int
        Suffix for the design matrix file name
        For now, manually curated.
        v00: for testing (without pupil)
        v01: with pupil (pupil stratified by the mean)
        v02: kernel lenghts updated
        v03: pupil patched, all inputs and results in the same folder
    glm_path : str or Path
        Path to save the files
    
    Returns
    -------
    None
    
    '''
    bod_list = cdu.get_bod_list(raw_path)
    run_params, design, activity_trace_matrix, activity_trace_info, activity_trace_archive = \
        generate_activity_trace_and_design_matrices(bod_list, data_type=data_type)
    
    ###############
    # Save them all
    if isinstance(glm_path, str):
        glm_path = Path(glm_path)
    session_save_path = glm_path / f'{session_key}_glm_v{version:02}'
    session_save_path.mkdir(parents=True, exist_ok=True)

    # serialize sets in run_params
    run_params_keys = list(run_params.keys())
    set_inds = np.where([type(run_params[key])==set for key in run_params_keys])[0]
    for i in set_inds:
        key = run_params_keys[i]
        run_params[key] = list(run_params[key])

    run_params_fn = session_save_path / 'run_params.json'
    with open(run_params_fn, 'w') as f:
        json.dump(run_params, f, indent=4)

    design_matrix_fn = session_save_path / 'design_matrix.nc'
    X = design.get_X()
    X.to_netcdf(design_matrix_fn)

    dm_unstd_features_fn = session_save_path / 'unstd_features.npy'
    np.save(dm_unstd_features_fn, design.unstd_features)


    activity_trace_matrix_fn = session_save_path / f'{data_type}_activity_trace_matrix.nc'
    if not os.path.exists(activity_trace_matrix_fn):
        activity_trace_matrix.to_netcdf(activity_trace_matrix_fn)

    activity_trace_info_fn = session_save_path / f'{data_type}_activity_trace_info.npy'
    if not os.path.exists(activity_trace_info_fn):
        np.save(activity_trace_info_fn, activity_trace_info)
        
    activity_trace_archive_fn = session_save_path / f'{data_type}_activity_trace_archive.npy'
    if not os.path.exists(activity_trace_archive_fn):
        np.save(activity_trace_archive_fn, activity_trace_archive)



####################################################################################################
# For fitting GLM
def fit_and_save_session(session_key, data_type, version, glm_path, parallel=True):
    X, activity_trace, activity_trace_info, run_params, unstd_features, use_indices = \
            gft.load_data(session_key, data_type, version, load_path=glm_path)

    print('-------------------------------')
    print('Input data loaded.')
    print('-------------------------------\n')
    
    ophys_frame_rate = activity_trace_info['ophys_frame_rate']
    # trim X and activity_trace based on the shift
    X_trim = X[use_indices, :]
    activity_trace_trim = activity_trace[use_indices, :]

    # Filter out cells based on prop event frames (<1%)
    activity_trace_trim_filtered = gft.filter_activity_trace_matrix(activity_trace_trim)
    
    fit_params = gft.default_fit_params()
    # TODO: stratification parameters (threshold) can be improved
    stratified_list = gft.set_stratified_list(fit_params, X, unstd_features, use_indices, ophys_frame_rate)
    stratified_frames, cv_inds_stratified = gft.get_stratified_folds(fit_params, stratified_list)
    
    lambdas_cv, W_cv, var_explained_train_cv, var_explained_test_cv, ve_test_train_ratio_cv = \
            gft.collect_session_results(run_params, fit_params, X_trim,
                                        activity_trace_trim_filtered, stratified_frames, cv_inds_stratified,
                                        parallel=parallel)
    print('-------------------------------')
    print('Model fit.')
    print('-------------------------------\n')
    if isinstance(glm_path, str):
        glm_path = Path(glm_path)
    save_dir = glm_path / f'{session_key}_glm_v{version:02}'

    var_explained_mean_model = \
        gft.get_full_session_var_explained_from_mean_model(W_cv, X_trim, activity_trace_trim_filtered)
    
    gft.save_glm_results(version, session_key, data_type, 
                    fit_params,  use_indices, stratified_frames, cv_inds_stratified,
                    lambdas_cv, W_cv,
                    var_explained_train_cv, var_explained_test_cv, ve_test_train_ratio_cv, 
                    var_explained_mean_model,
                    save_dir=save_dir)
    
    print('-------------------------------')
    print('Results saved.')
    print('-------------------------------\n')


####################################################################################################
# For coding scores
def calculate_and_save_coding_score(session_key, version, data_type, glm_path):
    if isinstance(glm_path, str):
        glm_path = Path(glm_path)
    save_dir = glm_path / f'{session_key}_glm_v{version:02}'
    glm_fn = save_dir / f'glm_results_v{version:02}_{session_key}_{data_type}.npy'
    if not os.path.exists(glm_fn):
        print(f'{session_key} not processed yet')
        return
    glm_results = np.load(glm_fn, allow_pickle=True).item()

    X, activity_trace, activity_trace_info, run_params, unstd_features, use_indices = \
        gft.load_data(session_key, data_type, version, glm_path)

    # trim X and activity_trace based on the shift
    X_trim = X[use_indices, :]
    activity_trace_trim = activity_trace[use_indices, :]
    activity_trace_trim_filtered = gft.filter_activity_trace_matrix(activity_trace_trim)

    adj_var_explained_session_model, adj_var_explained_session_model_full_mask = \
        cs.generate_session_model_adjusted_variance_explained(
            glm_results, run_params, X_trim, activity_trace_trim_filtered)
    coding_score = cs.calculate_coding_score(adj_var_explained_session_model,
                                    adj_var_explained_session_model_full_mask,
                                    run_params)

    adj_var_explained_session_model = adj_var_explained_session_model.expand_dims(
        "metric").assign_coords(metric=["adj_ve_model"])
    adj_var_explained_session_model_full_mask = \
        adj_var_explained_session_model_full_mask.expand_dims(
            "metric").assign_coords(metric=["adj_ve_full"])
    coding_score = coding_score.expand_dims("metric").assign_coords(
                                    metric=["coding_score"])

    cs_metrics = xr.concat([adj_var_explained_session_model,
                            adj_var_explained_session_model_full_mask,
                            coding_score],
                            dim="metric")
    cs_metrics.name = 'coding_score_metrics_from_session_model'

    save_fn = save_dir / f'coding_score_v{version:02}_{session_key}_{data_type}.nc'
    cs_metrics.to_netcdf(save_fn)


def calculate_and_save_cross_session_normalized_coding_score(mouse_id,
                                                             version=3,
                                                             data_type='events',
                                                             glm_path=Path('/root/capsule/scratch/glm/')):
    if isinstance(glm_path, str):
        glm_path = Path(glm_path)
    # get roicat_df
    roicat_df_fn = glm_path / f'roicat_df_{mouse_id}.pkl'
    if not os.path.exists(roicat_df_fn):
        raise ValueError(f'ROICaT results not found for {mouse_id}')
    roicat_df = pd.read_pickle(roicat_df_fn)

    # get session_info_df
    session_info_df = cdu.get_session_info(mouse_id)

    # focus on FNN sessions only
    fnn_session_keys = get_fnn_session_keys(session_info_df)
    fnn_all_matched_roi_df = get_fnn_all_matched_roi_df(fnn_session_keys, roicat_df)

    cs_norm = cs.cross_session_normalization(fnn_all_matched_roi_df, session_info_df)
    save_fn = glm_path / f'coding_score_fnn_normalized_v{version:02}_{mouse_id}_{data_type}.nc'
    try:
        cs_norm.to_netcdf(save_fn)
    except:
        print(f'Error saving {save_fn}')
        new_array = fix_dataarray_serialization_issue(cs_norm)
        new_array.to_netcdf(save_fn)
        print(f'Saved {save_fn} after fixing serialization issue')



def get_fnn_session_keys(session_info_df,
                         familiar_session_type='OPHYS_1_images_A',
                         novel_session_type='OPHYS_4_images_B'):
    """ Getting session keys of FNN sessions.
    F: the last Familiar session (OPHYS_1_images_A)
    N: the two Novel sessions (OPHYS_4_images_B)
    *Important* Keep the order of the sessions.
    Assumed that session keys are in the form of '{mouse_id}_{date}',
    so the order is determined by the date (sorted by session key).
    """
    fnn_session_inds = []
    potential_f_sessions = session_info_df.query(f'session_type == "{familiar_session_type}"')
    if len(potential_f_sessions) == 0:
        raise ValueError('No Familiar sessions found')
    fnn_session_inds.append(potential_f_sessions.session_ind.max())

    potential_n_sessions = session_info_df.query(f'session_type == "{novel_session_type}"')
    if len(potential_n_sessions) == 0:
        raise ValueError('No Novel sessions found')
    elif len(potential_n_sessions) == 1:
        raise ValueError('Only one Novel session found')
    elif len(potential_n_sessions) > 2:
        raise ValueError('More than two Novel sessions found')
    for si in potential_n_sessions.session_ind.values:
        fnn_session_inds.append(si)
    fnn_session_keys = session_info_df.query('session_ind in @fnn_session_inds').index.values
    fnn_session_keys = np.sort(fnn_session_keys)

    return fnn_session_keys


def get_fnn_all_matched_roi_df(fnn_session_keys, roicat_df):
    """ Get all matched ROIs for FNN sessions.
    """
    fnn_roi_df = roicat_df.query('session_key in @fnn_session_keys and valid_roi and matched')
    fnn_uc_series = fnn_roi_df.groupby('unique_cell_name').size()
    fnn_matched_uc = fnn_uc_series.index.values[np.where(fnn_uc_series==len(fnn_session_keys))[0]]    
    fnn_all_matched_roi_df = fnn_roi_df.query('unique_cell_name in @fnn_matched_uc')
    assert fnn_all_matched_roi_df.valid_roi.all()
    return fnn_all_matched_roi_df


def fix_dataarray_serialization_issue(da):
    """ Fix (potentially) serialization issue of DataArray.
    Sometimes saving to netCDF errors out, but if I copy the DataArray
    to a new array with copying values and coordinates, it works.
    """
    # First, extract your original data and coordinate values
    data = da.values

    # Create a new dataarray with correct dimensions but carefully transfer coordinates
    new_arr = xr.DataArray(
        data=data,
        dims=da.dims
    )

    # Add coordinates one by one, checking each for consistency
    for dim in da.dims:
        # Get original coordinate values for this dimension
        try:
            # This gets the 1D coordinate values along the dimension
            coord_values = da.coords[dim].values
            
            # Verify length matches the corresponding dimension in data
            expected_length = data.shape[da.dims.index(dim)]
            if len(coord_values) == expected_length:
                new_arr.coords[dim] = coord_values
            else:
                print(f"Warning: Coordinate '{dim}' has length {len(coord_values)} but dimension has length {expected_length}")
                # Use default indices instead
                new_arr.coords[dim] = range(expected_length)
        except KeyError:
            # If dimension doesn't have a coordinate with same name
            print(f"Note: Creating default coordinate for dimension '{dim}'")
            new_arr.coords[dim] = range(data.shape[da.dims.index(dim)])

    # Now add any non-dimension coordinates
    for coord_name in da.coords:
        if coord_name not in da.dims:
            try:
                # Check shape compatibility
                if all(dim in new_arr.dims for dim in da[coord_name].dims):
                    # Verify sizes match
                    sizes_match = True
                    for dim in da[coord_name].dims:
                        if len(da[coord_name][dim]) != len(new_arr[dim]):
                            sizes_match = False
                            print(f"Warning: Non-dimension coordinate '{coord_name}' has incompatible shape")
                            break
                    
                    if sizes_match:
                        new_arr.coords[coord_name] = da.coords[coord_name]
            except Exception as e:
                print(f"Skipping coordinate '{coord_name}': {str(e)}")

    # Add any attributes
    new_arr.attrs = da.attrs
    
    # Check if the new array is the same as the original
    assert new_arr.equals(da)
    return new_arr

####################################################################################################
# general codeocean functions
# Many are redundant in lamf_analysis code_ocean general_bod_utils and general_data_utils
# (in general_capsule_utilities branch)

ID_NAME_MAP = {719374: 'Tunsten',
717824: 'Chromium',
729417: 'Cardamon',
721291: 'Saffron', 
736963: 'Thyme',
739564: 'Paprika',
747107: 'Ginger',
747667: 'Pepper',
749315: 'Fennel',
755252: 'Anise',
767018: 'Oregano',
767002: 'Rosemary'}


def get_mouse_session_df(mouse_id, cutoff_processed_date=None,
                         default_mount=['fb4b5cef-4505-4145-b8bd-e41d6863d7a9', # Ophys_Extension_schema_10_14_2024_13_44
                                        '35d1284e-4dfa-4ac3-9ba8-5ea1ae2fdaeb'], # ROI classifier V1
                         include_pupil=True):
    success = True
    mouse_sessions = aind_session.get_sessions(subject_id=mouse_id)

    raw_data_date_list = []
    processed_data_date_list = []
    capsule_ids_list = []
    commit_ids_list = []
    processed_data_asset_ids_list = []
    raw_data_asset_ids_list = []
    
    num_provenence_data_assets_list = []
    if include_pupil:
        pupil_data_asset_ids_list = []
    for session in mouse_sessions:
        raw_date = session.raw_data_asset.name.split('_')[2] 
        processed_data = [da for da in session.data_assets if '_processed_' in da.name]
        processed_data = [da for da in processed_data if (da.provenance.commit is not None)]
        if include_pupil:
            pupil_data = [da for da in session.data_assets if 'dlc-eye' in da.name]
            pupil_raw_data = [np.setdiff1d(da.provenance.data_assets, default_mount) for da in pupil_data]

        processed_data_dates = [da.name.split('_processed_')[1].split('_')[0] for da in processed_data]
        capsule_ids = [da.provenance.capsule for da in processed_data]
        commit_ids = [da.provenance.commit for da in processed_data]
        data_asset_ids = [da.id for da in processed_data]
        raw_data_asset_ids = [np.setdiff1d(da.provenance.data_assets, default_mount) for da in processed_data]
        num_provenence_data_assets = [len(da.provenance.data_assets) for da in processed_data]
        for i in range(len(processed_data)):
            if cutoff_processed_date is not None:
                if processed_data_dates[i] < cutoff_processed_date:
                    continue
            raw_data_asset_id = raw_data_asset_ids[i]

            if include_pupil:
                matching_pupil_data_ind = np.where([raw_data_asset_id in pupil_raw_data[j] for j in range(len(pupil_raw_data))])[0]
                if len(matching_pupil_data_ind) == 1:
                    pupil_data_asset_ids_list.append(pupil_data[matching_pupil_data_ind[0]].id)
                elif len(matching_pupil_data_ind) == 0:
                    pupil_data_asset_ids_list.append(0)
                else:
                    raise ValueError(f'More than one matching pupil data asset found for {raw_data_asset_id} from {session}')

            raw_data_date_list.append(raw_date)
            capsule_ids_list.append(capsule_ids[i])
            commit_ids_list.append(commit_ids[i])
            processed_data_asset_ids_list.append(data_asset_ids[i])
            processed_data_date_list.append(processed_data_dates[i])
            raw_data_asset_ids_list.append(raw_data_asset_id)
            num_provenence_data_assets_list.append(num_provenence_data_assets[i])
    mouse_session_df = pd.DataFrame({'raw_data_date': raw_data_date_list,
                                        'processed_data_date': processed_data_date_list,
                                        'capsule_id': capsule_ids_list,
                                        'commit_id': commit_ids_list,
                                        'processed_data_asset_id': processed_data_asset_ids_list,
                                        'raw_data_asset_id': raw_data_asset_ids_list,
                                        'num_provenence_data_assets': num_provenence_data_assets_list})
    if include_pupil:
        mouse_session_df['pupil_data_asset_id'] = pupil_data_asset_ids_list

    mouse_session_df['num_raw_data_asset_ids'] = mouse_session_df['raw_data_asset_id'].apply(len)
    if np.all(mouse_session_df['num_raw_data_asset_ids'].values == 1):
        mouse_session_df['raw_data_asset_id'] = mouse_session_df['raw_data_asset_id'].apply(lambda x: x[0])
    else:
        success = False
        warnings.warn('Multiple raw data asset ids found for a single processed data asset id')
    mouse_session_df['num_raw_data_asset_ids'] = mouse_session_df['raw_data_asset_id'].apply(len)

    if include_pupil:
        if np.any(mouse_session_df['pupil_data_asset_id'].values == 0):
            success = False
            warnings.warn(f'No matching pupil data asset found for {mouse_session_df[mouse_session_df["pupil_data_asset_id"] == 0].raw_data_date.values}')
        
    return success, mouse_session_df
        

def attach_mouse_data_assets(mouse_session_df, include_pupil=True):
    assert np.all([isinstance(raw_id, str) for raw_id in mouse_session_df.raw_data_asset_id.values]), \
        f'raw data asset ids must be str'
    if include_pupil:
        assert np.all([isinstance(raw_id, str) for raw_id in mouse_session_df.raw_data_asset_id.values]), \
        f'"include_pupil" set to {include_pupil}, so must provide appropriate pupil data asset ids'
    success = True
    try:
        capsule.attach_assets(mouse_session_df.raw_data_asset_id.values)
        capsule.attach_assets(mouse_session_df.processed_data_asset_id.values)
        if include_pupil:
            capsule.attach_assets(mouse_session_df.pupil_data_asset_id.values)
    except:
        success = False
    return success


def get_session_info(mouse_id, data_dir='/root/capsule/data'):
    ''' Get all raw data paths in the data directory
    '''
    data_folders = [d for d in glob.glob(data_dir + '/*') if Path(d).is_dir()]
    raw_paths = np.sort([d for d in data_folders if ('processed' not in d.split('/')[-1]) and
                        ('dlc-eye' not in d.split('/')[-1]) and
                        ('multiplane-ophys' in d.split('/')[-1]) and 
                        ('stimuli' not in d.split('/')[-1]) and
                        ('stim-response' not in d.split('/')[-1]) and
                        ('ROICat' not in d.split('/')[-1]) and
                        (str(mouse_id) in d.split('/')[-1])])

    session_names = [d.split('/')[-1] for d in raw_paths]
    session_keys = ['_'.join(sn.split('_')[1:3]) for sn in session_names]
    session_inds = np.arange(len(session_names))
    session_types = []
    for raw_path in raw_paths:
        session_json_fn = Path(raw_path) / 'session.json'
        with open(session_json_fn) as f:
            session_json = json.load(f)
        session_types.append(session_json['session_type'])
    session_info_df = pd.DataFrame({'session_name': session_names,
                                'session_key': session_keys,
                                'session_ind': session_inds,
                                'session_type': session_types,
                                'raw_path': raw_paths}) 
    session_info_df.set_index('session_key', drop=True, inplace=True)

    def _map_session_type_to_stimulus(x):
        if any(substring in x for substring in ('gratings', 'STAGE_0', 'STAGE_1')):
            return 'gratings'
        elif 'images_A' in x:
            return 'images_A'
        elif 'images_B' in x:
            return 'images_B'
        else:
            return 'unknown'

    session_info_df['stimulus'] = session_info_df['session_type'].apply(lambda x: _map_session_type_to_stimulus(x))
    image_sets = [s for s in session_info_df.stimulus.unique() if 'images' in s]
    image_order = np.argsort([np.where(session_info_df.stimulus.values == s)[0][0] for s in image_sets])
    familiar_image = image_sets[image_order[0]]
    novel_image = image_sets[image_order[1]]
    # count the number of exposure to each stimuli and session_type
    stimulus_exposures = []
    session_type_exposures = []
    for i in range(session_info_df.shape[0]):
        row = session_info_df.iloc[i]
        stimulus_exposures.append(np.where(session_info_df.iloc[:i+1].stimulus.values == row.stimulus)[0].shape[0])
        session_type_exposures.append(np.where(session_info_df.iloc[:i+1].session_type.values == row.session_type)[0].shape[0])
    session_info_df['stimulus_exposures'] = stimulus_exposures
    session_info_df['session_type_exposures'] = session_type_exposures

    return session_info_df


####################################################################################################
# General wrapper for using COMB
def get_bod_list(raw_path):
    ''' Get all BehaviorOphysDataset objects from a raw path
    '''
    session_name = str(raw_path).split('/')[-1]
    data_dir = Path(raw_path).parent
    processed_path = list(data_dir.glob(f'{session_name}_processed*'))[0]
    
    opids = []
    for plane_folder in processed_path.glob("*"):
        if plane_folder.is_dir() and not plane_folder.name.startswith("nextflow") \
            and not ('nwb' in plane_folder.name):
            opid = plane_folder.name
            opids.append(opid)

    bod_list = []
    for opid in opids:
        bod = load_plane_data(session_name, opid=opid)
        stim_table = merge_trials_to_stim_table(bod)
        bod_list.append(bod)
    return bod_list


def get_any_bod(raw_path):
    ''' to get any bod object from a raw path.
    This is mostly for getting session information (e.g., stim_table)
    '''
    session_name = str(raw_path).split('/')[-1]
    data_dir = Path(raw_path).parent
    processed_path = list(data_dir.glob(f'{session_name}_processed*'))[0]

    opids = []
    for plane_folder in processed_path.glob("*"):
        if plane_folder.is_dir() and not plane_folder.name.startswith("nextflow") \
            and not ('nwb' in plane_folder.name):
            opid = plane_folder.name
            break
    bod = load_plane_data(session_name, opid=opid)
    stim_table = merge_trials_to_stim_table(bod)
    return bod


def load_plane_data(session_name, opid=None, opid_ind=None, data_dir='/root/capsule/data/',
                    verbose=False):
    ''' Load data using COMB.

    Parameters
    ----------
    session_name : str
        name of the session (e.g., 'multiplane-ophys_721291_2024-05-08_08-05-54')
    opid : str (optional)
        ophys plane ID (e.g., '1365108570', or 'VISp_0')
    opid_ind : int (optional)
        index of the ophys plane ID (e.g., 0)
        if opid is provided, this parameter is ignored
    data_dir : str (optional)
        path to the data directory
    
    Returns
    -------
    bod : BehaviorOphysDataset
        COMB object containing the ophys plane data
    '''

    if opid is None and opid_ind is None:
        raise ValueError('Must provide either opid or opid_ind')
    data_dir = Path(data_dir)
    processed_dirs = glob.glob(str(data_dir / f'{session_name}*processed*'))
    eye_dirs = glob.glob(str(data_dir / f'{session_name}*dlc-eye*'))
    if len(eye_dirs) == 0:
        raise ValueError(f'No eye tracking data found for session {session_name}')
    elif len(eye_dirs) > 1:
        raise ValueError(f'Multiple eye tracking data found for session {session_name}')
    else:
        eye_path = eye_dirs[0]
        
    if len(processed_dirs) == 0:
        raise ValueError(f'No processed data found for session {session_name}')
    elif len(processed_dirs) > 1:
        raise ValueError(f'Multiple processed data found for session {session_name}')
    else:
        plane_dirs = []
        for path in glob.glob(processed_dirs[0] + '/*'):
            if os.path.isdir(path) and ('nwb' not in path.split('/')[-1]):
                plane_dirs.append(path)

        if opid is not None:
            plane_path = [p for p in plane_dirs if opid in p]
            if len(plane_path) > 1:
                raise ValueError(f'Multiple {opid} found for session {session_name}')
            elif len(plane_path) == 0:
                raise ValueError(f'No {opid} found for session {session_name}')
            else:
                plane_path = plane_path[0]
        else:
            if len(plane_dirs) > opid_ind:
                plane_path = plane_dirs[opid_ind]
                opid = plane_path.split('/')[-1]
                if verbose:
                    print(f'Using plane {opid} for session {session_name}')
            else:
                raise ValueError(f'Processed data for session {session_name} has less than {opid_ind} planes')
    raw_path = Path(plane_path.split('_processed')[0])

    if not raw_path.exists():
        raise ValueError(f'No raw data found for session {session_name}')
    bod = BehaviorOphysDataset(plane_folder_path=plane_path,
                               raw_folder_path=raw_path,
                               eye_tracking_path=eye_path,
                               pipeline_version='v6',
                               verbose=verbose)
    bod.metadata['ophys_plane_id'] = opid
    
    return bod


def merge_trials_to_stim_table(bod):
    ''' To add hits and misses to stim_table.
    Non-change stimulus presentations will be False.
    This mutates bod.stimulus_presentations, so need to be run only once per bod loading
    '''
    # check if bod has trials
    if not hasattr(bod, 'trials'):
        bod = add_trials_to_bod(bod)
    trials = bod.trials
    stim_table = bod.stimulus_presentations
    stim_table['is_change'] = stim_table.is_change.astype(bool)
    assert np.array_equal(stim_table.query('is_change').start_time.values, trials.change_time.values)
    stim_table['hit'] = False
    stim_table['miss'] = False
    stim_table.loc[stim_table.start_time.isin(trials.query('hit').change_time.values), 'hit'] = True
    stim_table.loc[stim_table.start_time.isin(trials.query('miss').change_time.values), 'miss'] = True

    assert np.all(bod.stimulus_presentations.query('is_change').hit.values == trials.hit.values)
    return stim_table




####################################################################################################
# Previous capsule_utils (by MJD)
def session_type_from_session(capsule_files):

    
    dict_map =  {}
    
    for session_name,session_dict in capsule_files.items():
        # load session_json from session_path
        with open(session_dict['raw_path'] / "session.json", 'r') as f:
            session_json = json.load(f)
        dict_map[session_name]  = session_json['session_type']
        
    return dict_map

################################################################################################
# Functions for data access - FROM OTHER REPOS SHOULD DELETE
################################################################################################
MULTIPLANE_FILE_PARTS = {"processing_json": "processing.json",
                           "params_json": "_params.json",
                           "registered_metrics_json": "_registered_metrics.json",
                           "average_projection_png": "_average_projection.png",
                           "max_projection_png": "_maximum_projection.png",
                           "motion_transform_csv": "_motion_transform.csv",
                           "segmentation_output_json": "segmentation_output.json",
                           "roi_traces_h5": "roi_traces.h5",
                           "neuropil_correction_h5": "neuropil_correction.h5",
                           "neuropil_masks_json": "neuropil_masks.json",
                           "neuropil_trace_output_json": "neuropil_trace_output.json",
                           #"demixing_h5": "demixing_output.h5",
                           #"demixing_json": "demixing_output.json",
                           "dff_h5": "dff.h5",
                           "extract_traces_json": "extract_traces.json",
                           "events_oasis_h5": "events_oasis.h5",
                           "suite2p_ops": "ops.npy",}

def multiplane_session_data_files(input_path):
    """Find all data files in a multiplane session directory."""
    input_path = Path(input_path)
    data_files = {}
    for key, value in MULTIPLANE_FILE_PARTS.items():
        data_files[key] = find_data_file(input_path, value)
    return data_files


def find_data_file(input_path, file_part, verbose=False):
    """Find a file in a directory given a partial file name.

    Example
    -------
    input_path = /root/capsule/data/multiplane-ophys_724567_2024-05-20_12-00-21
    file_part = "_sync.h5"
    return: "/root/capsule/data/multiplane-ophys_724567_2024-05-20_12-00-21/ophys/1367710111_sync.h5"
    
    
    Parameters
    ----------
    input_path : str or Path
        The path to the directory to search.
    file_part : str
        The partial file name to search for.
    """
    input_path = Path(input_path)
    try:
        file = list(input_path.glob(f'**/*{file_part}*'))[0]
    except IndexError:
        if verbose:
            logger.warning(f"File with '{file_part}' not found in {input_path}")
        file = None
    return file


def get_file_paths_dict(file_parts_dict, input_path):
    file_paths = {}
    for key, value in file_parts_dict.items():
        file_paths[key] = find_data_file(input_path, value)
    return file_paths


def check_ophys_folder(path):
    """ophys folders can have multiple names, check for all of them"""
    ophys_names = ['ophys', 'pophys', 'mpophys']
    ophys_folder = None
    for ophys_name in ophys_names:
        ophys_folder = path / ophys_name
        if ophys_folder.exists():
            break
        else:
            ophys_folder = None

    return ophys_folder


def check_behavior_folder(path):
    behavior_names = ['behavior', 'behavior_videos']
    behavior_folder = None
    for behavior_name in behavior_names:
        behavior_folder = path / behavior_name
        if behavior_folder.exists():
            break
        else:
            behavior_folder = None
    return behavior_folder


def get_sync_file_path(input_path, verbose=False):
    """Find the Sync file"""
    file_parts = {}
    input_path = Path(input_path)
    try: 
        # method 1: find sync_file by name
        file_parts = {"sync_h5": "_sync.h5"}
        sync_file_path = find_data_file(input_path, file_parts["sync_h5"], verbose=False)
    except IndexError as e:
        if verbose:
            logger.info("file with '*_sync.h5' not found, trying platform json")

    if sync_file_path is None:
        # method 2: load platform json
        # Note: sometimes fails if platform json has incorrect sync_file path
        logging.info(f"Trying to find sync file using platform json for {input_path}")
        file_parts = {"platform_json": "_platform.json"}
        platform_path = find_data_file(input_path, file_parts["platform_json"])
        with open(platform_path, 'r') as f:
            platform_json = json.load(f)
        ophys_folder = check_ophys_folder(input_path)
        sync_file_path = ophys_folder / platform_json['sync_file']

        if not sync_file_path.exists():
            logger.error(f"Unsupported data asset structure, sync file not found in {sync_file_path}")
            sync_file_path = None
        else:
            logger.info(f"Sync file found in {sync_file_path}")

    return sync_file_path


def plane_paths_from_session(session_path: Union[Path, str],
                             data_level: str = "raw") -> list:
    """Get plane paths from a session directory

    Parameters
    ----------
    session_path : Union[Path, str]
        Path to the session directory
    data_level : str, optional
        Data level, by default "raw". Options: "raw", "processed"

    Returns
    -------
    list
        List of plane paths
    """
    session_path = Path(session_path)
    if data_level == "processed":
        planes = [x for x in session_path.iterdir() if x.is_dir()]
        planes = [x for x in planes if 'nextflow' not in x.name]
    elif data_level == "raw":
        planes = list((session_path / 'ophys').glob('ophys_experiment_*'))
    return planes


def all_planes_file_paths_dict(processed_path, raw_path = None):

    processed_plane_paths = plane_paths_from_session(processed_path, data_level = "processed")

    file_paths = {}
    file_paths['planes'] = {}
    # build file paths dict
    for plane_path in processed_plane_paths:
        plane_path = Path(plane_path)
        plane_name = plane_path.name
        file_paths['planes'][plane_name] = multiplane_session_data_files(plane_path)
        file_paths['planes'][plane_name]["processed_plane_path"] = plane_path
    file_paths["processed_path"] = processed_path
    file_paths["raw_path"] = raw_path

    return file_paths

### new functions ###
def all_session_files_dict_in_capsule(data_dir = Path("../data/")):


    session_paths = list(data_dir.glob("*multiplane-ophys*processed*"))

    # sort by date
    session_paths = sorted(session_paths, key=lambda x: x.name.split("_")[2])

    # sort into dict by mouse
    session_dict = {}
    for session_path in session_paths:
        mouse = session_path.name.split("_")[1]
        if mouse not in session_dict:
            session_dict[mouse] = []
        session_dict[mouse].append(session_path)

    session_files_dict = {}
    for mouse_id, session_list in session_dict.items():
        session_files_dict[mouse_id] = {}
        
        for session_path in session_list:
            session_files_dict[mouse_id][session_path.name] = all_planes_file_paths_dict(session_path)
            raw_path = Path(str(session_path).split("_processed")[0])
            # check if raw path exists
            if raw_path.exists():
                session_files_dict[mouse_id][session_path.name]["raw_path"] = raw_path
        
    return session_files_dict


def plot_projection_with_scale(img, ax = None, title = "", scale_bar = True):
    
    sns.set_context("talk")
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    vmax = np.percentile(img, 99.5)
    ax.imshow(img, vmax = vmax, cmap="gray")
    #plt.title(f"{gcamp} \n {td}")
    ax.set_title(title)
    ax.axis("off")

    # add scale bar bottom right (each pixel is 0.78 um, show 100 um)
    
    if scale_bar:
        scale_bar_length = 50
        scale_bar_length_pixels = scale_bar_length / 0.78
        scale_bar_height = 10
        scale_bar_height_pixels = scale_bar_height / 0.78
        scale_bar_y = img.shape[0] - 40
        scale_bar_x = img.shape[1] - 40
        ax.plot([scale_bar_x, scale_bar_x - scale_bar_length_pixels], [scale_bar_y, scale_bar_y], color="white", linewidth=5)
        # add text
        ax.text(scale_bar_x - scale_bar_length_pixels/2, scale_bar_y + 15, "50 um", color="white", fontsize=12, ha="center")
    return ax 

#### metadata ####
