from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import json
from glob import glob


def _ffill_bfill(da):
    """Forward-fill then back-fill a 1-D DataArray without requiring bottleneck."""
    return da.copy(data=pd.Series(da.values).ffill().bfill().values)
# from dask import delayed, compute
# from dask.distributed import Client

##############################################################################################################
## Loading and trimming data
def load_data(session_key, data_type, version, load_path):
    if type(load_path) == str:
        load_path = Path(load_path)
    
    data_path = load_path / f'{session_key}_glm_v{version:02}'
    
    x_fn = data_path / 'design_matrix.nc'

    X = xr.open_dataarray(x_fn, mmap=False)

    run_params_fn = data_path / 'run_params.json'
    with open(run_params_fn, 'r') as f:
        run_params = json.load(f)

    unstd_features_fn = data_path / 'unstd_features.npy'
    unstd_features = np.load(unstd_features_fn, allow_pickle=True).item()

    activity_trace_fn = data_path / f'{data_type}_activity_trace_matrix.nc'
    activity_trace = xr.open_dataarray(activity_trace_fn, mmap=False)

    assert X.shape[0] == activity_trace.shape[0]

    activity_trace_info_fn = data_path / f'{data_type}_activity_trace_info.npy'
    activity_trace_info = np.load(activity_trace_info_fn, allow_pickle=True).item()
    # ophys_frame_rate = activity_trace_info['ophys_frame_rate']
    
    # Trim frames based on kernel offsets
    offsets = [int(w.split('_')[-1]) for w in X.weights.values]
    min_offset = min(min(offsets), 0)
    max_offset = max(max(offsets), 0)
    use_indices = range(max_offset, X.shape[0] + min_offset)  # min_offset is negative or 0
    
    return X, activity_trace, activity_trace_info, run_params, unstd_features, use_indices 


def get_prop_support(activity_trace_trim):
    ''' Get proportion of support for each cell.
    
    Parameters
    ----------
    activity_trace_trim : xr.DataArray
        Trimmed activity_trace matrix (time x cell_roi_id)
        
    Returns
    -------
    prop_support : np.array
        Proportion of support for each cell
    '''
    prop_support = []
    for cri in activity_trace_trim.cell_roi_id.values:
        trace = activity_trace_trim.sel(cell_roi_id=cri)
        prop_support.append(len(np.where(trace)[0]) / len(trace))
    prop_support = np.array(prop_support)
    return prop_support


def filter_activity_trace_matrix(activity_trace, prop_support_threshold=0.01):
    '''Filter out cells based on proportion of event frames
    Particularly useful for events (may not needed for dff)
    
    Parameters
    ----------
    activity_trace : xr.DataArray
        activity_trace matrix (time x cell_roi_id)
    prop_support_threshold : float, optional
        Threshold for proportion of event frames, by default 0.01
        
    Returns
    -------
    activity_trace_filtered : xr.DataArray
        Filtered activity_trace matrix (time x cell_roi_id)
        
    '''
    prop_support = get_prop_support(activity_trace)
    filtered_inds = np.where(prop_support >= prop_support_threshold)[0]
    activity_trace_filtered = activity_trace.isel(cell_roi_id=filtered_inds)
    return activity_trace_filtered


## Fitting parameters
def default_fit_params():
    fit_params = {
        'ElasticNet': False,                        # Use ElasticNet (True) or Ridge (False) regression
        'ElasticNet_alpha':0.95,                    #
        'L2_grid_range':[.1, 10000],                # Min/Max L2 values
        'L2_grid_num': 40,                          # Number of L2 values
        'L2_grid_type':'log',                       # how to space L2 options, must be: 'log' or 'linear'
        'cv_fold': 5,                               # Number of cross-validation folds
        'cv_nested_fold': 5,                        # Number of nested cross-validation folds, for hyperparameter optimization (e.g., L2 lambda)
        'cv_stratify': {                            # Stratify cross-validation folds by these variables (['time', 'image_name', 'running_speed', 'lick_frequency', 'engagement', 'rolling_performance'])
            'variables': ['time', 'running_speed', 'lick', 'rolling_performance', 'pupil'], # 'time', 'running_speed', 'lick', 'rolling_performance' 
            'num_time_stratification': 5,           # Number of stratification for time. The rest are binary (at least for now)
            # 'stratification_threshold': 0.2,        # Threshold for stratification variable. If smaller portion is lower than this threshold, stratification is not performed (too rare incidents)
            'running_smoothing_s': 10,              # Smoothing window for running speed (seconds)
            'running_threshold': 5,                 # Threshold for running variable (cm/s)
            'rolling_performance_threshold': 1.5,   # Threshold for rolling performance variable (x rewards / minute). Previously 'engagement', but to generalize across learning phases
            'lick_threshold': 20,                   # Threshold for lick variable (licks / min)
            }
    }
    return fit_params


##############################################################################################################
## Stratification
# For multiple stratification, stratify per variable and then make combinations
def get_feature_traces_from_X(X, use_indices, keyword):
    ''' Get the feature trace from the design matrix X. 
    TODO: If the feature is not in the weights, get the feature from BOD and interpolate to X_trim timepoints.
    
    Parameters
    ----------
    X : xr.DataArray
        Design matrix
    use_indices : list
        Indices to use in the design matrix. Defined by the kernel offsets
    keyword : str
        Keyword to search in the weights of the design matrix
        
    Returns
    -------
    feature_traces : xr.DataArray
        Feature trace from the design matrix
    '''
    feature_inds = np.where([keyword in w for w in X.weights.values])[0]
    if len(feature_inds) > 0:
        #TODO: consider when running is not in the weights. Need to load BOD and interpolate running to X_trim timepoints
        if len(np.where([f'{keyword}_0' in w for w in X.weights.values])[0]) > 0:
            feature_traces = X.sel(weights=f'{keyword}_0')[use_indices]
        else:
            first_delay_trace = X.isel(weights=feature_inds[0])
            first_delay = int(str(first_delay_trace.weights.values).split('_')[-1])
            feature_traces = first_delay_trace.shift(timestamps=-first_delay)[use_indices]            
    else:  #TODO: implement getting the feature trace from... bod?
        feature_traces = None        
    return feature_traces


def set_stratified_list(fit_params, X, unstd_features, use_indices, ophys_frame_rate):
    ''' Set the stratification.
    #IMPORTANT: Returned list contains indices within use_indices.
    
    Parameters
    ----------
    fit_params : dict
        Fitting parameters
    X : xr.DataArray
        Design matrix
    unstd_features : dict
        Unstandardized features. To apply binarized stratification.
    use_indices : list
        Indices to use in the design matrix. Defined by the kernel offsets
    ophys_frame_rate : float
        Frame rate of the ophys data
        
    Returns
    -------
    stratified_list : list
        List of stratified indices. 
        #IMPORTANT: Indices within use_indices.
    '''
    X_trim = X[use_indices, :]
    stratification_features = ['time', 'running_speed', 'rolling_performance', 'lick', 'pupil']
    num_stratify_vars = len(fit_params['cv_stratify']['variables'])
    stratified_list = []
    for var in fit_params['cv_stratify']['variables']:    
        if var == 'time':  # divide all frames into cv_fold
            stratify_borders = np.floor(np.linspace(0, X_trim.shape[0], fit_params['cv_stratify']['num_time_stratification'] + 1)).astype(int)        
            stratified = []
            for i in range(fit_params['cv_stratify']['num_time_stratification']):
                stratified.append(np.arange(stratify_borders[i], stratify_borders[i+1]))
            stratified_list.append(stratified)
        elif var == 'running_speed':
            keyword = 'running'
            if keyword not in unstd_features:
                continue
            running_speed = unstd_features['running'][use_indices]
            assert len(running_speed) == X_trim.shape[0]
            # smooth running_speed
            running_speed = xr.DataArray(running_speed, dims='timestamps')
            running_speed['timestamps'] = X_trim['timestamps']
            smoothing_window = np.round(ophys_frame_rate * fit_params['cv_stratify']['running_smoothing_s']).astype(int)
            sm_running_speed = _ffill_bfill(running_speed.rolling(timestamps=smoothing_window, center=True).mean())

            # binarization
            running_frames = np.where(sm_running_speed > fit_params['cv_stratify']['running_threshold'])[0]
            stationary_frames = np.setdiff1d(np.arange(X_trim.shape[0]), running_frames)
            stratified = [running_frames, stationary_frames]
            if any(len(s) == 0 for s in stratified):
                continue
            stratified_list.append(stratified)
        elif var == 'rolling_performance':
            keyword = 'hits'
            correct = get_feature_traces_from_X(X, use_indices, keyword)
            if correct is None:
                continue
            num_frames_one_min = np.round(ophys_frame_rate * 60).astype(int)
            rolling_performance = _ffill_bfill(correct.rolling(timestamps=num_frames_one_min, center=True).sum())
            performing_frames = np.where(rolling_performance > fit_params['cv_stratify']['rolling_performance_threshold'])[0]
            nonperforming_frames = np.setdiff1d(np.arange(X_trim.shape[0]), performing_frames)
            stratified = [performing_frames, nonperforming_frames]
            if any(len(s) == 0 for s in stratified):
                continue
            stratified_list.append(stratified)
        elif var == 'lick':
            if 'licks' not in unstd_features:
                continue
            licks_trace = unstd_features['licks'][use_indices]
            assert len(licks_trace) == X_trim.shape[0]
            licks = xr.DataArray(licks_trace, dims='timestamps')
            licks['timestamps'] = X_trim['timestamps']
            num_frames_one_min = np.round(ophys_frame_rate * 60).astype(int)
            lick_freq = _ffill_bfill(licks.rolling(timestamps=num_frames_one_min, center=True).sum())
            licking_frames = np.where(lick_freq > fit_params['cv_stratify']['lick_threshold'])[0]
            nonlicking_frames = np.setdiff1d(np.arange(X_trim.shape[0]), licking_frames)
            stratified = [licking_frames, nonlicking_frames]
            if any(len(s) == 0 for s in stratified):
                continue
            stratified_list.append(stratified)
        elif var == 'pupil':
            keyword = 'pupil'
            pupil_trace = get_feature_traces_from_X(X, use_indices, keyword)
            if pupil_trace is None:
                continue
            stratified = [np.where(pupil_trace > 0)[0], np.where(pupil_trace <= 0)[0]]
            if any(len(s) == 0 for s in stratified):
                continue
            stratified_list.append(stratified)
        else:
            print(f'{var} not implemented for stratification.\nImplemented feature keywords: {stratification_features}\nContinue...')
    return stratified_list
    
    
def get_stratified_folds(fit_params, stratified_list):
    ''' Get stratified folds for nested cross-validation.
    
    Parameters
    ----------
    fit_params : dict
        Fitting parameters
    stratified_list : list
        List of stratified indices
        
    Returns
    -------
    stratified_frames : list
        List of stratified frames
    cv_inds_stratified : list
        List of stratified indices for cross-validation
    '''
    cv_fold = fit_params['cv_fold']
    cv_nested_fold = fit_params['cv_nested_fold']

    # collect unique sets across all stratification variables
    unique_sets = []
    for i in range(len(stratified_list)):
        if i == 0:
            unique_sets = stratified_list[i]
        else:
            unique_sets = [np.intersect1d(a, b) for a in unique_sets for b in stratified_list[i]]
    assert len(unique_sets) == np.multiply.reduce([len(s) for s in stratified_list])

    # Randomly divide each set into cv_fold * cv_nested_fold
    total_folds = cv_fold * cv_nested_fold
    stratify_folds_each_set = []
    for us in unique_sets:
        np.random.shuffle(us)
        stratify_folds_each_set.append(np.array_split(us, total_folds))

    # Combine each folds in each set to generate stratified folds
    stratified_frames = [np.concatenate([s[i] for s in stratify_folds_each_set]) for i in range(total_folds)]
    assert len(stratified_frames) == total_folds

    # shuffle and split stratified folds into cv_fold
    # To randomly distribute total number of frames
    stratified_frames = [stratified_frames[i] for i in np.random.permutation(np.arange(total_folds))]
    cv_inds_stratified = np.array_split(np.arange(total_folds), cv_fold)
    assert np.all([len(cv_inds_stratified[i]) == cv_nested_fold for i in range(cv_fold)])    
    
    return stratified_frames, cv_inds_stratified


def get_train_test_inds(test_fold_ind, fit_params, stratified_frames, cv_inds_stratified):
    ''' Get train and test indices for cross-validation.
    
    Parameters
    ----------
    test_fold_ind : int
        Index of the test fold
    fit_params : dict
        Fitting parameters
    stratified_frames : list
        List of stratified frames
    cv_inds_stratified : list
        List of stratified indices for cross-validation
        
    Returns
    -------
    train_frames : np.array
        Training frames
    test_frames : np.array
        Testing frames
    nested_fold_inds : list
        List of nested fold indices
    '''
    train_fold_inds = np.setdiff1d(np.arange(fit_params['cv_fold']), test_fold_ind)

    test_stratified_fold_inds = cv_inds_stratified[test_fold_ind]
    train_stratified_fold_inds = np.concatenate([cv_inds_stratified[i] for i in train_fold_inds])

    test_frames = np.concatenate([stratified_frames[i] for i in test_stratified_fold_inds])
    train_frames = np.concatenate([stratified_frames[i] for i in train_stratified_fold_inds])

    # get cross validation folds for hyperparameter optimization within the training set
    nested_stratified_fold_inds_arr = np.array([cv_inds_stratified[i] for i in train_fold_inds])
    nested_stratified_fold_inds = [nested_stratified_fold_inds_arr[:, i] for i in range(fit_params['cv_nested_fold'])]
    nested_fold_frames = [np.sort(np.concatenate([stratified_frames[i] for i in nested_stratified_fold_inds[j]])) for j in range(fit_params['cv_nested_fold'])]
    nested_fold_inds = [np.where(np.isin(train_frames, nff))[0] for nff in nested_fold_frames]

    # assert no overlap between nested folds
    for i in range(len(nested_fold_inds)):
        for j in range(len(nested_fold_inds)):
            if i != j:
                assert len(np.intersect1d(nested_fold_inds[i], nested_fold_inds[j])) == 0
    # assert all train frames are included in nested folds
    assert (np.sort(np.concatenate([nested_fold_frames[i] for i in range(len(nested_fold_frames))])) == np.sort(train_frames)).all()
    assert (np.sort(np.concatenate([nested_fold_inds[i] for i in range(len(nested_fold_inds))])) == np.arange(len(train_frames))).all()
    
    return train_frames, test_frames, nested_fold_inds


##############################################################################################################
## Basic fitting functions
def fit(y, X):
    '''
    Analytical OLS solution to linear regression. 

    y: shape (n_timestamps * n_cells)
    X: shape (n_timestamps * n_kernel_params)
    '''
    W = np.dot(np.linalg.inv(np.dot(X.T.values, X.values)), np.dot(X.T.values, y.values))
    return W


def fit_regularized(y, X, lam):
    '''
    Analytical OLS solution with added L2 regularization penalty. 

    y: xr DataArray shape (n_timestamps * n_cells)
    X: xr DataArray shape (n_timestamps * n_kernel_params)
    lam (float): Strength of L2 regularization per cell (hyperparameter to tune)

    Returns: XArray
    '''
    assert len(y.shape) == 2 # 2 dimensional, even if there is only one cell
    
    # Compute the weights
    if lam == 0:
        W = fit(y, X)
    else:
        W = np.dot(np.linalg.inv(np.dot(X.T.values, X.values) + lam * np.eye(X.shape[-1])),
               np.dot(X.T.values, y.values))
    # if len(W.shape) == 1: # in case of single neuron
    #     W = W[:, None]

    # Make xarray
    cellids = y['cell_roi_id'].values
    W_xarray= xr.DataArray(
            W, 
            dims =('weights','cell_roi_id'), 
            coords = {  'weights':X.weights.values, 
                        'cell_roi_id':cellids}
            )
    return W_xarray


def compute_variance_explained(y, W, X): 
    '''
    Computes the fraction of variance in y explained by the linear model Y = X*W
    
    y: (n_timepoints, n_cells)
    W: Xarray (n_kernel_params, n_cells)
    X: Xarray (n_timepoints, n_kernel_params)
    '''
    y_hat = X @ W
    # y_hat = X.values @ W.values # if we want to use values directly, we need to validate if kernel names are in the same order
    
    var_total = y.var(dim='timestamps')   # Total variance in the ophys trace for each cell
    var_resid = (y - y_hat).var(dim='timestamps') # Residual variance in the difference between the model and data
    var_explained = (var_total - var_resid) / var_total  # Fraction of variance explained by linear model

    return var_explained


def compute_adjusted_variance_explained(y, W, X, mask):
    '''
    Computes the fraction of variance in y explained by the linear model Y = X*W
    but only looks at the timepoints in mask
    
    y: (n_timepoints, n_cells)
    W: Xarray (n_kernel_params, n_cells)
    X: Xarray (n_timepoints, n_kernel_params)
    mask: bool vector (n_timepoints,)
    '''

    y_hat = X @ W
    # y_hat = X.values @ W.values # if we want to use values directly, we need to validate if kernel names are in the same order

    # Define variance function that lets us isolate the mask timepoints
    def my_var(trace, support_mask):
        mu = trace.mean(dim='timestamps')
        return ((trace[support_mask, :] - mu)**2).mean(dim='timestamps')

    var_total = my_var(y, mask) # Total variance in the ophys trace for each cell
    var_resid = my_var(y - y_hat, mask) # Residual variance in the difference between the model and data
    return (var_total - var_resid) / var_total  # Fraction of variance explained by linear model


##############################################################################################################
## Calculating lambdas for ridge regression
def collect_var_explained_across_lambdas_and_nested_folds(X, y, nested_fold_inds, fit_params):
    ''' Collect variance ratio across lambdas and nested folds.
    
    Parameters
    ----------
    X : xr.DataArray
        Design matrix
    y : xr.DataArray
        activity_trace matrix
    nested_fold_inds : list
        List of indices for nested cross-validation
    fit_params : dict
        Fitting parameters
        
    Returns
    -------
    var_explained_xr_collected : xr.DataArray
        Variance ratio across lambdas and nested folds
    '''
    assert fit_params['cv_nested_fold'] == len(nested_fold_inds)
    
    test_lams = np.geomspace(fit_params['L2_grid_range'][0], fit_params['L2_grid_range'][1], fit_params['L2_grid_num'])

    var_explained_xr_collected = None  # Start with no data
    for nfi in range(fit_params['cv_nested_fold']):
        nested_train_inds = np.concatenate([nested_fold_inds[i] for i in range(len(nested_fold_inds)) if i != nfi])
        nested_test_inds = nested_fold_inds[nfi]
        X_train = X[nested_train_inds, :]
        X_test = X[nested_test_inds, :]
        y_train = y[nested_train_inds, :]
        y_test = y[nested_test_inds, :]
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0]

        var_explained_xr, _ = get_var_explained_xr_across_lambdas(X_train, X_test, y_train, y_test, test_lams)
        var_explained_xr = var_explained_xr.expand_dims(nested_fold_ind=[nfi])

        if var_explained_xr_collected is None:
            var_explained_xr_collected = var_explained_xr
        else:
            var_explained_xr_collected = xr.concat([var_explained_xr_collected, var_explained_xr], dim="nested_fold_ind")
        
    return var_explained_xr_collected


def get_var_explained_xr_across_lambdas(X_train, X_test, y_train, y_test, test_lams):
    ''' Get variance explained xarray across lambdas.
    
    Parameters
    ----------
    X_train : xr.DataArray
        Design matrix for training
    X_test : xr.DataArray
        Design matrix for testing
    y_train : xr.DataArray
        activity_trace matrix for training
    y_test : xr.DataArray
        activity_trace matrix for testing
    test_lams : list
        List of lambda values to run
        
    Returns
    -------
    var_explained_xr : xr.DataArray
        Variance ratio across lambdas
    W_all : xr.DataArray
        Weights across lambdas
    '''
    var_explained_xr = None  # Start with no data
    W_all = None

    for lam in test_lams:
        # Fit the regularized model
        W = fit_regularized(y_train, X_train, lam)
        
        # Compute the variance ratio
        var_explained = compute_variance_explained(y_test, W, X_test)

        # Expand along the 'lam' dimension
        var_explained = var_explained.expand_dims(lam=[lam])
        
        # Concatenate with the existing var_explained_xr
        if var_explained_xr is None:
            var_explained_xr = var_explained  # Initialize with the first DataArray
        else:
            var_explained_xr = xr.concat([var_explained_xr, var_explained], dim="lam")
        
        # Concatenate with the existing W_all
        if W_all is None:
            W_all = W
        else:
            W_all = xr.concat([W_all, W], dim="lam")
        
    return var_explained_xr, W_all


##########################
## Running the whole session data
def collect_session_results(run_params, fit_params, X_trim, activity_trace_trim, stratified_frames, cv_inds_stratified,
                            parallel=True, num_cores=None):
    ''' Collect session results, from collect_fold_results, using nested cross-validation for lambda selection and model fitting.
    
    Parameters
    ----------
    run_params : dict
        Run parameters, to be loaded from the design matrix results
    fit_params : dict
        Fitting parameters
    X_trim : xr.DataArray
        Trimmed design matrix
    activity_trace_trim : xr.DataArray
        Trimmed activity_trace matrix
    stratified_frames : list
        List of stratified frames
    cv_inds_stratified : list
        List of stratified indices for cross-validation
        
    Returns
    -------
    lambdas_cv : xr.DataArray
        Lambda values
    W_cv : xr.DataArray
        Weights
    var_explained_train_cv : xr.DataArray
        Variance ratio for training
    var_explained_test_cv : xr.DataArray
        Variance ratio for testing
    vr_test_train_ratio_cv : xr.DataArray
        Ratio of variance ratio for testing and training
    '''
    for test_fold_ind in range(fit_params['cv_fold']):
        # get test and train fold inds for one cross-validation set
        train_frames, test_frames, nested_fold_inds = \
            get_train_test_inds(test_fold_ind, fit_params, stratified_frames, cv_inds_stratified)
        
        X_train_outer = X_trim[train_frames, :]
        X_test_outer = X_trim[test_frames, :]
        y_train_outer = activity_trace_trim[train_frames, :]
        y_test_outer = activity_trace_trim[test_frames, :]
        
        lambdas_fold, W_fold, var_explained_train_fold, var_explained_test_fold, \
            vr_test_train_ratio_fold = \
                collect_fold_results(run_params, fit_params, X_train_outer, X_test_outer,
                                     y_train_outer, y_test_outer, nested_fold_inds,
                                    parallel=parallel, num_cores=num_cores)
        
        lambdas_fold = lambdas_fold.expand_dims(test_fold_ind=[test_fold_ind])
        W_fold = W_fold.expand_dims(test_fold_ind=[test_fold_ind])
        var_explained_train_fold = var_explained_train_fold.expand_dims(test_fold_ind=[test_fold_ind])
        var_explained_test_fold = var_explained_test_fold.expand_dims(test_fold_ind=[test_fold_ind])
        vr_test_train_ratio_fold = vr_test_train_ratio_fold.expand_dims(test_fold_ind=[test_fold_ind])
        
        if test_fold_ind == 0:
            lambdas_cv = lambdas_fold
            W_cv = W_fold
            var_explained_train_cv = var_explained_train_fold
            var_explained_test_cv = var_explained_test_fold
            vr_test_train_ratio_cv = vr_test_train_ratio_fold

        else:
            lambdas_cv = xr.concat([lambdas_cv, lambdas_fold], dim='test_fold_ind')
            W_cv = xr.concat([W_cv, W_fold], dim='test_fold_ind')
            var_explained_train_cv = xr.concat([var_explained_train_cv, var_explained_train_fold], dim='test_fold_ind')
            var_explained_test_cv = xr.concat([var_explained_test_cv, var_explained_test_fold], dim='test_fold_ind')
            vr_test_train_ratio_cv = xr.concat([vr_test_train_ratio_cv, vr_test_train_ratio_fold], dim='test_fold_ind')
                
    # Validation
    check_nan_weights(W_cv, run_params)
    
    return lambdas_cv, W_cv, var_explained_train_cv, var_explained_test_cv, vr_test_train_ratio_cv  
    

def collect_fold_results(run_params, fit_params, X_train_outer, X_test_outer, y_train_outer, y_test_outer,
                         nested_fold_inds, parallel=True, num_cores=None):
    ''' Collect fold results.
    
    Parameters
    ----------
    run_params : dict
        Run parameters, to be loaded from the design matrix results
    fit_params : dict
        Fitting parameters
    X_train_outer : xr.DataArray
        Design matrix for training. Outer fold (fitting GLM with lambda from innter folds)
    X_test_outer : xr.DataArray
        Design matrix for testing
    y_train_outer : xr.DataArray
        activity_trace matrix for training
    y_test_outer : xr.DataArray
        activity_trace matrix for testing
    nested_fold_inds : list
        List of indices for nested cross-validation
    parallel : bool (optional)
        Use dask for parallel processing, default is True
    num_cores : int (optional)
        Number of cores to use for parallel processing, default is None
        If None and use_dask is True, use all available cores
        
    Returns
    -------
    lambdas_fold : xr.DataArray
        Lambda values
    W_fold : xr.DataArray
        Weights
    var_explained_train_fold : xr.DataArray
        Variance ratio for training
    var_explained_test_fold : xr.DataArray
        Variance ratio for testing
    vr_test_train_ratio_fold : xr.DataArray
        Ratio of variance ratio for testing and training
    '''
        
    models = run_params['dropouts'].keys()
    if parallel:
        lambdas_fold, W_fold, var_explained_train_fold, var_explained_test_fold, \
            vr_test_train_ratio_fold = \
                collect_fold_results_parallel(run_params, fit_params, 
                                        X_train_outer, X_test_outer,
                                        y_train_outer, y_test_outer,
                                        nested_fold_inds, num_cores=num_cores)
    else:
        for mi, model_label in enumerate(models):
            model_results = collect_model_results(run_params, fit_params,
                                                  X_train_outer, X_test_outer,
                                                  y_train_outer, y_test_outer,
                                                  nested_fold_inds, model_label)
            (lambdas, W_model, var_explained_train, var_explained_test,
             vr_test_train_ratio) = model_results
            
            # Collect results
            if mi == 0:
                lambdas_fold = lambdas
                W_fold = W_model
                var_explained_train_fold = var_explained_train
                var_explained_test_fold = var_explained_test
                vr_test_train_ratio_fold = vr_test_train_ratio
            else:
                lambdas_fold = xr.concat([lambdas_fold, lambdas], dim='model')
                W_fold = xr.concat([W_fold, W_model], dim='model')
                var_explained_train_fold = xr.concat([var_explained_train_fold, var_explained_train], dim='model')
                var_explained_test_fold = xr.concat([var_explained_test_fold, var_explained_test], dim='model')
                vr_test_train_ratio_fold = xr.concat([vr_test_train_ratio_fold, vr_test_train_ratio], dim='model')
            
    return lambdas_fold, W_fold, var_explained_train_fold, var_explained_test_fold, vr_test_train_ratio_fold


def collect_fold_results_parallel(run_params, fit_params, X_train_outer, X_test_outer, y_train_outer, y_test_outer,
                              nested_fold_inds, num_cores=None):
    ''' Collect fold results using dask.
    
    Parameters
    ----------
    run_params : dict
        Run parameters, to be loaded from the design matrix results
    fit_params : dict
        Fitting parameters
    X_train_outer : xr.DataArray
        Design matrix for training. Outer fold (fitting GLM with lambda from innter folds)
    X_test_outer : xr.DataArray
        Design matrix for testing
    y_train_outer : xr.DataArray
        activity_trace matrix for training
    y_test_outer : xr.DataArray
        activity_trace matrix for testing
    nested_fold_inds : list
        List of indices for nested cross-validation
    num_cores : int (optional)
        Number of cores to use for parallel processing, default is None
        If None, use all available cores
        
    Returns
    -------
    lambdas_fold : xr.DataArray
        Lambda values
    W_fold : xr.DataArray
        Weights
    var_explained_train_fold : xr.DataArray
        Variance ratio from training
    var_explained_test_fold : xr.DataArray
        Variance ratio from testing
    vr_test_train_ratio_fold : xr.DataArray
         Ratio of variance ratio between testing and training (for overfitting check)
    '''
    models = list(run_params['dropouts'].keys())
    model_results = [
        collect_model_results(run_params, fit_params,
                              X_train_outer, X_test_outer,
                              y_train_outer, y_test_outer,
                              nested_fold_inds, model_label)
        for model_label in models
    ]

    for mi in range(len(models)):
        (lambdas, W_model, var_explained_train, var_explained_test, 
         vr_test_train_ratio) = model_results[mi]
        if mi == 0:
            lambdas_fold = lambdas
            W_fold = W_model
            var_explained_train_fold = var_explained_train
            var_explained_test_fold = var_explained_test
            vr_test_train_ratio_fold = vr_test_train_ratio

        else:
            lambdas_fold = xr.concat([lambdas_fold, lambdas], dim='model')
            W_fold = xr.concat([W_fold, W_model], dim='model')
            var_explained_train_fold = xr.concat([var_explained_train_fold, var_explained_train], dim='model')
            var_explained_test_fold = xr.concat([var_explained_test_fold, var_explained_test], dim='model')
            vr_test_train_ratio_fold = xr.concat([vr_test_train_ratio_fold, vr_test_train_ratio], dim='model')
            
    return lambdas_fold, W_fold, var_explained_train_fold, var_explained_test_fold, vr_test_train_ratio_fold


def collect_model_results(run_params, fit_params, X_train_outer, X_test_outer, y_train_outer, y_test_outer,
                          nested_fold_inds, model_label):
    ''' Collect fold results.
    
    Parameters
    ----------
    run_params : dict
        Run parameters, to be loaded from the design matrix results
    fit_params : dict
        Fitting parameters
    X_train_outer : xr.DataArray
        Design matrix for training. Outer fold (fitting GLM with lambda from innter folds)
    X_test_outer : xr.DataArray
        Design matrix for testing
    y_train_outer : xr.DataArray
        activity_trace matrix for training
    y_test_outer : xr.DataArray
        activity_trace matrix for testing
    nested_fold_inds : list
        List of indices for nested cross-validation
    model_label : str
        Model label
        
    Returns
    -------
    lambdas : xr.DataArray
        Lambda values
    W_model : xr.DataArray
        Weights
    var_explained_train : xr.DataArray
        Variance ratio from training
    var_explained_test : xr.DataArray
        Variance ratio from testing
    vr_test_train_ratio : xr.DataArray
        Ratio of variance ratio between testing and training (for overfitting check)
        Values close to 0 means overfitting. Values close to 1 is ideal.
    '''
    test_lams = np.geomspace(fit_params['L2_grid_range'][0], fit_params['L2_grid_range'][1], fit_params['L2_grid_num'])
    test_lams = xr.DataArray(test_lams, dims={'lam'})
    kernels = run_params['dropouts'][model_label]['kernels']
    weights = [w for w in X_train_outer.weights.values if np.any([(k in w) for k in kernels])]
    X_train_outer_model = X_train_outer.sel(weights=weights)
    x_test_outer_model = X_test_outer.sel(weights=weights)
    var_explained_xr_collected = collect_var_explained_across_lambdas_and_nested_folds(X_train_outer_model,
                                                                                       y_train_outer, nested_fold_inds, fit_params)
    
    best_lam_inds = var_explained_xr_collected.mean(dim="nested_fold_ind").argmax(dim="lam")
    lambdas = test_lams[best_lam_inds]
    
    # Train using the lambda
    assert len(lambdas) == len(y_train_outer.cell_roi_id)
    num_cell = len(lambdas)
    W_model = None
    for ci in range(num_cell):
        W_cell = fit_regularized(y_train_outer.isel(cell_roi_id=[ci]), X_train_outer_model, lambdas[ci].values)
        if W_model is None:
            W_model = W_cell
        else:
            W_model = xr.concat([W_model, W_cell], dim="cell_roi_id")
            
    # Calculate performance on train and test sets
    var_explained_train = compute_variance_explained(y_train_outer, W_model, X_train_outer_model)
    var_explained_test = compute_variance_explained(y_test_outer, W_model, x_test_outer_model)
    ve_test_train_ratio = var_explained_test / var_explained_train
    
    # Expand model dimension of the DataArrays
    lambdas = lambdas.expand_dims(model=[model_label])
    W_model = W_model.expand_dims(model=[model_label])
    var_explained_train = var_explained_train.expand_dims(model=[model_label])
    var_explained_test = var_explained_test.expand_dims(model=[model_label])
    vr_test_train_ratio = ve_test_train_ratio.expand_dims(model=[model_label])
    
    return lambdas, W_model, var_explained_train, var_explained_test, vr_test_train_ratio


############
# Checking and validation
def check_nan_weights(W, run_params):
    weights = W.weights.values
    model_labels = W.model.values

    assert set(model_labels) == set(run_params['dropouts'].keys()), 'Not all models in rum_params were run'
    for model_label in model_labels:
    # model_label = model_labels[3]
    # print(model_label)
        model_kernels = run_params['dropouts'][model_label]['kernels']    
        run_weights = [w for w in weights if np.any([mk in w for mk in model_kernels])]
        dropped_weights = [w for w in weights if np.all([mk not in w for mk in model_kernels])]

        # are all run_weights finite?
        assert np.all(np.isfinite(W.sel(weights=run_weights, model=model_label))), "Detected NaN for fit weights"
        # are all dropped_weights nan?
        assert np.all(np.isnan(W.sel(weights=dropped_weights, model=model_label))), "Detected a value(s) for dropped weights"
    return True


############################################################################################
# gathering session model traces
# one from splits, another from the mean model
def get_sessionwise_model_traces(fit_params, W_cv, X_trim, activity_trace_trim, stratified_frames, cv_inds_stratified):
    ''' Get sessionwise model traces - both from stitching each split fits and from mean coefficients across splits.
    These are too big so won't be saved in the results.
    Use this function to retrieve the sessionwise model traces.
    #TODO: make arguments to choose type of reconstruction (mean or from split) and specific models (via model name, e.g., 'Full')
    
    Parameters
    ----------
    fit_params : dict
        Fitting parameters
    W_cv : xr.DataArray
        Weights from cross-validation
    X_trim : xr.DataArray
        Trimmed design matrix
    activity_trace_trim : xr.DataArray
        Trimmed activity_trace matrix
    stratified_frames : list
        List of stratified frames
    cv_inds_stratified : list
        List of stratified indices for cross-validation
        
    Returns
    -------
    session_model_from_splits : xr.DataArray
        Session model from stitching each split fits
    session_model_from_mean_W : xr.DataArray
        Session model from mean coefficients across splits
    '''
    # getting session model from stitching each split fits
    models = W_cv.model.values
    session_model_from_splits = xr.DataArray(np.zeros((*activity_trace_trim.shape, len(models))),
                                            dims=['timestamps', 'cell_roi_id', 'model'],
                                            coords={'timestamps':activity_trace_trim.timestamps,
                                                    'cell_roi_id':activity_trace_trim.cell_roi_id,
                                                    'model':models})
    for test_fold_ind in range(fit_params['cv_fold']):
        train_frames, test_frames, nested_fold_inds = \
            get_train_test_inds(test_fold_ind, fit_params, stratified_frames, cv_inds_stratified)
        test_frames = np.sort(test_frames)

        X_test_outer = X_trim[test_frames, :]
        y_test_outer = activity_trace_trim[test_frames, :]
        
        W_fold = W_cv.sel(test_fold_ind=test_fold_ind)
        
        for mi, model in enumerate(models):
            W_fold_model = W_fold.sel(model=model)
            weights = W_fold_model.dropna(dim='weights').weights.values
            X_model = X_test_outer.sel(weights=weights)
            W_fold_model = W_fold_model.sel(weights=weights)
            
            # session_model_from_splits.sel(timestamps=test_timestamps, model=model)[:] = X_model.values @ W_fold_model.values
            # Somehow the above line does not work
            session_model_from_splits[test_frames, :, mi] = X_model.values @ W_fold_model.values

    # getting session model from mean coefficients across splits
    W_mean = W_cv.mean(dim='test_fold_ind') #TODO: check if nanmean is necessary
    session_model_from_mean_W = xr.full_like(session_model_from_splits, fill_value=0, dtype=float)
    for model in models:
        W_model = W_mean.sel(model=model)
        weights = W_model.dropna(dim='weights').weights.values
        X_model = X_trim.sel(weights=weights)
        W_model = W_model.sel(weights=weights)
        session_model_from_mean_W.sel(model=model)[:] = X_model.values @ W_model.values
        
    return session_model_from_splits, session_model_from_mean_W


def get_full_session_var_explained_from_mean_model(W_cv, X_trim, activity_trace_trim):
    ''' Get full session variance explained from mean model.
    
    Parameters
    ----------
    W_cv : xr.DataArray
        Weights from cross-validation
    X_trim : xr.DataArray
        Trimmed design matrix
    activity_trace_trim : xr.DataArray
        Trimmed activity_trace matrix
        
    Returns
    -------
    var_explained_mean_model : xr.DataArray
        Variance explained from the mean model
    '''
    # mean model and variance explained
    W_mean = W_cv.mean(dim='test_fold_ind') #TODO: check if nanmean is necessary
    for mi, model in enumerate(W_mean.model.values):
        W_model = W_mean.sel(model=model)
        weights = W_model.dropna(dim='weights').weights.values
        X_model = X_trim.sel(weights=weights)
        W_mean_model = W_model.sel(weights=weights)
        var_explained = compute_variance_explained(activity_trace_trim, W_mean_model, X_model)
        var_explained = var_explained.expand_dims(model=[model])
        if mi == 0:
            var_explained_mean_model = var_explained
        else:
            var_explained_mean_model = xr.concat([var_explained_mean_model, var_explained], dim='model')
    return var_explained_mean_model


############################################################################################
# Saving and loading the results
def save_glm_results(dm_version, session_key, data_type,
                     fit_params, use_indices, stratified_frames, cv_inds_stratified,
                     lambdas_cv, W_cv, 
                     var_explained_train_cv, var_explained_test_cv, 
                     vr_test_train_ratio_cv,
                     var_explained_mean_model,
                     save_dir):
    glm_results = {'fit_params': fit_params,
                'use_indices': use_indices,
                'stratified_frames': stratified_frames,
                'cv_inds_stratified': cv_inds_stratified,
                'lambdas_cv': lambdas_cv,
                'W_cv': W_cv,
                'var_explained_train_cv': var_explained_train_cv,
                'var_explained_test_cv': var_explained_test_cv,
                'vr_test_train_ratio_cv': vr_test_train_ratio_cv,
                'var_explained_mean_model': var_explained_mean_model,
                }
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fn = save_dir / f'glm_results_v{dm_version:02}_{session_key}_{data_type}.npy'
    if save_fn.exists():
        print(f'\n\n{save_fn} exists!\n\nAdding_suffix...')
        fn_base = f'glm_results_v{dm_version:02}_{session_key}_{data_type}'
        fn_list = [Path(fp).name for fp in glob(str(save_dir / fn_base) + '*')]
        suffixes = [fn.split(f'{fn_base}')[1].split('.')[0] for fn in fn_list]
        suffixes = [s.split('_')[1] for s in suffixes if len(s)>0]
        if len(suffixes) == 0:
            new_suffix = '_00'
        else:
            assert [s.isnumeric() for s in suffixes]
            max_suffix = np.max([int(s) for s in suffixes])
            new_suffix = f'_{max_suffix + 1:02}'
        save_fn = save_dir / f'{fn_base}{new_suffix}.npy'
        assert not save_fn.exists()
        print(f'New save filename  = {save_fn}')
    np.save(save_fn, glm_results)
    return save_fn
