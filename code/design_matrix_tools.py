import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
import scipy
import pandas as pd

def add_kernels(design, run_params, bod, response):
    '''
        Iterates through the kernels in run_params['kernels'] and adds
        each to the design matrix
        Each kernel must have fields:
            offset:
            length:
    
        design          the design matrix for this model
        run_params      the run_json for this model
        bod             COMB behaviorOphysPaneDataset object for this experiment
        response        dictionary about response arrays for this model
    '''
    run_params['failed_kernels'] = set()
    run_params['failed_dropouts'] = set()
    run_params['kernel_error_dict'] = dict()
    for kernel_name in run_params['kernels']:          
        if 'num_weights' not in run_params['kernels'][kernel_name]:
            run_params['kernels'][kernel_name]['num_weights'] = None
        if run_params['kernels'][kernel_name]['type'] == 'discrete':
            design = add_discrete_kernel_by_label(kernel_name, design, run_params, bod, response)
        else:
            design = add_continuous_kernel_by_label(kernel_name, design, run_params, bod, response)   

    clean_failed_kernels(run_params)
    return design


def clean_failed_kernels(run_params):
    '''
        Modifies the model definition to handle any kernels that failed to fit during the add_kernel process
        Removes the failed kernels from run_params['kernels'], and run_params['dropouts']
    '''
    if run_params['failed_kernels']:
        print('The following kernels failed to be added to the model: ')
        print(run_params['failed_kernels'])
        print()   
 
    # Iterate failed kernels
    for kernel in run_params['failed_kernels']:     
        # Remove the failed kernel from the full list of kernels
        if kernel in run_params['kernels'].keys():
            run_params['kernels'].pop(kernel)

        # Remove the dropout associated with this kernel
        if kernel in run_params['dropouts'].keys():
            run_params['dropouts'].pop(kernel)        
        
        # Remove the failed kernel from each dropout list of kernels
        for dropout in run_params['dropouts'].keys(): 
            # If the failed kernel is in this dropout, remove the kernel from the kernel list
            if kernel in run_params['dropouts'][dropout]['kernels']:
                run_params['dropouts'][dropout]['kernels'].remove(kernel) 
            # If the failed kernel is in the dropped kernel list, remove from dropped kernel list
            if kernel in run_params['dropouts'][dropout]['dropped_kernels']:
                run_params['dropouts'][dropout]['dropped_kernels'].remove(kernel) 

    # Iterate Dropouts, checking for empty dropouts
    drop_list = list(run_params['dropouts'].keys())
    for dropout in drop_list:
        if not (dropout == 'Full'):
            if len(run_params['dropouts'][dropout]['dropped_kernels']) == 0:
                run_params['dropouts'].pop(dropout)
                run_params['failed_dropouts'].add(dropout)
            elif len(run_params['dropouts'][dropout]['kernels']) == 1:
                run_params['dropouts'].pop(dropout)
                run_params['failed_dropouts'].add(dropout)

    if run_params['failed_dropouts']:
        print('The following dropouts failed to be added to the model: ')
        print(run_params['failed_dropouts'])
        print()


def add_continuous_kernel_by_label(kernel_name, design, run_params, bod, response):
    '''
        Adds the kernel specified by <kernel_name> to the design matrix
        kernel_name          <str> the label for this kernel, will raise an error if not implemented
        design          the design matrix for this model
        run_params      the run_json for this model
        bod             COMB behaviorOphysPaneDataset object for this experiment
        response        dictionary about response arrays for this model      
    ''' 
    print('    Adding kernel: '+kernel_name)
    try:
        feature = run_params['kernels'][kernel_name]['feature']
        unstd_timeseries = None

        if feature == 'intercept':
            timeseries = np.ones(len(response['timestamps']))
        elif feature == 'time':
            timeseries = np.array(range(1,len(response['timestamps'])+1))
            timeseries = timeseries/len(timeseries)
        elif feature == 'running':
            running_df = bod.running_speed
            running_df = running_df.rename(columns={'speed':'values'})
            timeseries = interpolate_to_ophys_timestamps(response, running_df)['values'].values
            #timeseries = standardize_inputs(timeseries, mean_center=False,unit_variance=False, max_value=run_params['max_run_speed'])
            unstd_timeseries = timeseries
            timeseries = standardize_inputs(timeseries)
            
        # elif feature.startswith('face_motion'):
        #     PC_number = int(feature.split('_')[-1])
        #     face_motion_df =  pd.DataFrame({
        #         'timestamps': bod.behavior_movie_timestamps,
        #         'values': bod.behavior_movie_pc_activations[:,PC_number]
        #     })
        #     timeseries = interpolate_to_ophys_timestamps(response, face_motion_df)['values'].values
        #     timeseries = standardize_inputs(timeseries, mean_center=run_params['mean_center_inputs'], unit_variance=run_params['unit_variance_inputs'])
        elif feature == 'population_mean':
            timeseries = np.mean(response['response_arr'],1).values
            timeseries = standardize_inputs(timeseries, mean_center=run_params['mean_center_inputs'], unit_variance=run_params['unit_variance_inputs'])
        elif feature == 'Population_Activity_PC1':
            pca = PCA()
            pca.fit(response['response_arr'].values)
            response_pca = pca.transform(response['response_arr'].values)
            timeseries = response_pca[:,0]
            timeseries = standardize_inputs(timeseries, mean_center=run_params['mean_center_inputs'], unit_variance=run_params['unit_variance_inputs'])
        # elif (len(feature) > 6) & ( feature[0:6] == 'model_'):
        #     bsid = bod.metadata['behavior_session_id']
        #     weight_name = feature[6:]
        #     weight = get_model_weight(bsid, weight_name, run_params)
        #     weight_df = pd.DataFrame()
        #     weight_df['timestamps'] = bod.stimulus_presentations.start_time.values
        #     weight_df['values'] = weight.values
        #     timeseries = interpolate_to_ophys_timestamps(response, weight_df)
        #     timeseries['values'].fillna(method='ffill',inplace=True) # TODO investigate where these NaNs come from
        #     timeseries = timeseries['values'].values
        #     timeseries = standardize_inputs(timeseries, mean_center=run_params['mean_center_inputs'],unit_variance=run_params['unit_variance_inputs'])
        elif feature == 'pupil':
            ophys_eye = process_eye_data(bod, ophys_timestamps=response['timestamps'])
            timeseries = ophys_eye['pupil_radius_zscore'].values
            unstd_timeseries = ophys_eyes['pupil_radius'].values
        # elif feature == 'lick_model' or feature == 'groom_model':
        #     if not hasattr(bod, 'lick_groom_model'):
        #         bod.lick_groom_model = process_behavior_predictions(bod, ophys_timestamps = response['timestamps'])
        #     timeseries = bod.lick_groom_model[feature.split('_')[0]].values
        else:
            raise Exception('Could not resolve kernel label')
    except Exception as e:
        print('\tError encountered while adding kernel for '+kernel_name+'. Attemping to continue without this kernel. ' )
        print(e)
        # Need to remove from relevant lists
        run_params['failed_kernels'].add(kernel_name)      
        run_params['kernel_error_dict'][kernel_name] = {
            'error_type': 'kernel', 
            'kernel_name': kernel_name, 
            'exception':e.args[0], 
            'oeid':bod.metadata['ophys_experiment_id'], 
            'glm_version':run_params['version']
        }
        # Logging errors due to mongo connection issues
        # # log error to mongo
        # gat.log_error(
        #     run_params['kernel_error_dict'][kernel_name], 
        #     keys_to_check = ['oeid', 'glm_version', 'kernel_name']
        # )
        return design
    else:
        #assert length of values is same as length of timestamps
        assert len(timeseries) == response['response_arr'].values.shape[0], 'Length of continuous regressor must match length of timestamps'

        # Add to design matrix
        design.add_kernel(
            timeseries, 
            run_params['kernels'][kernel_name]['length'], 
            kernel_name, 
            offset=run_params['kernels'][kernel_name]['offset'],
            num_weights=run_params['kernels'][kernel_name]['num_weights']
        )
        if unstd_timeseries is not None:
            design.add_unstd_features(unstd_timeseries, kernel_name)
        return design


def interpolate_to_ophys_timestamps(response, df):
    """ Interpolate timeseries onto ophys timestamps

    Parameters
    ----------
    response : dict
        ophys data, containing 'timestamps':<array of timestamps>
    df : pd.dataframe
        ith columns:
            timestamps (timestamps of signal)
            values  (signal of interest)

    Returns
    -------
    pd.dataFrame
        timestamps 
        values (values interpolated onto timestamps)
    """
    f = scipy.interpolate.interp1d(
        df['timestamps'],
        df['values'],
        bounds_error=False
    )

    interpolated = pd.DataFrame({
        'timestamps':response['timestamps'],
        'values':f(response['timestamps'])
    })

    return interpolated


def standardize_inputs(timeseries, mean_center=True, unit_variance=True, max_value=None):
    '''
        Performs three different input standarizations to the timeseries
    
        if mean_center, the timeseries is adjusted to have 0-mean. This can be performed with unit_variance. 

        if unit_variance, the timeseries is adjusted to have unit variance. This can be performed with mean_center.
    
        if max_value is given, then the timeseries is normalized by max_value. This cannot be performed with mean_center and unit_variance.

    '''
    if (max_value is not None ) & (mean_center or unit_variance):
        raise Exception('Cannot perform max_value standardization and mean_center or unit_variance standardizations together.')

    if mean_center:
        print('                 : '+'Mean Centering')
        timeseries = timeseries - np.mean(timeseries) # mean center
    if unit_variance:
        print('                 : '+'Standardized to unit variance')
        timeseries = timeseries / np.std(timeseries)
    if max_value is not None:
        print('                 : '+'Normalized by max value: '+str(max_value))
        timeseries = timeseries / max_value

    return timeseries


def add_discrete_kernel_by_label(kernel_name, design, run_params, bod, response):
    '''
        Adds the kernel specified by <kernel_name> to the design matrix
        kernel_name     <str> the label for this kernel, will raise an error if not implemented
        design          the design matrix for this model
        run_params      the run_json for this model
        bod             COMB behaviorOphysDataset object for this experiment
        response        dictionary about response arrays for this model      
    ''' 
    print('    Adding kernel: '+kernel_name)
    try:
        feature = run_params['kernels'][kernel_name]['feature']
        unstd_timeseries = None
        
        if feature == 'licks':
            feature_times = bod.licks.data['timestamps'].values
        elif feature == 'lick_bouts':
            licks = bod.licks.data
            licks['pre_ILI'] = licks['timestamps'] - licks['timestamps'].shift(fill_value=-10)
            licks['post_ILI'] = licks['timestamps'].shift(periods=-1,fill_value=5000) - licks['timestamps']
            licks['bout_start'] = licks['pre_ILI'] > run_params['lick_bout_ILI']
            licks['bout_end'] = licks['post_ILI'] > run_params['lick_bout_ILI']
            assert np.sum(licks['bout_start']) == np.sum(licks['bout_end']), "Lick bout splitting failed"
            
            # We are making an array of in-lick-bout-feature-times by tiling timepoints every <min_interval> seconds. 
            # If a lick is the end of a bout, the bout-feature-times continue <min_time_per_bout> after the lick
            # Otherwise, we tile the duration of the post_ILI
            feature_times = np.concatenate([np.arange(x[0],x[0]+run_params['min_time_per_bout'],run_params['min_interval']) if x[2] else
                                        np.arange(x[0],x[0]+x[1],run_params['min_interval']) for x in 
                                        zip(licks['timestamps'], licks['post_ILI'], licks['bout_end'])]) 
        elif feature == 'rewards':
            feature_times = bod.rewards['timestamps'].values
        elif feature == 'change':
            #feature_times = bod.trials.query('go')['change_time'].values # This method drops auto-rewarded changes
            feature_times = bod.stimulus_presentations.query('is_change')['start_time'].values
            feature_times = feature_times[~np.isnan(feature_times)]
        elif feature in ['hit', 'miss', 'false_alarm', 'correct_reject']:
            if feature == 'hit': # Includes auto-rewarded changes as hits, since they include a reward. 
                # feature_times = bod.trials.query('hit or auto_rewarded')['change_time'].values
                feature_times = bod.trials.query('hit')['change_time'].values
            else:
                feature_times = bod.trials.query(feature)['change_time'].values
            feature_times = feature_times[~np.isnan(feature_times)]
            if len(bod.rewards) < 5: ## HARD CODING THIS VALUE
                raise Exception('Trial type regressors arent defined for passive sessions (sessions with less than 5 rewards)')
        elif feature == 'passive_change':
            if len(bod.rewards) > 5: 
                raise Exception('\tPassive Change kernel cant be added to active sessions')               
            feature_times = bod.stimulus_presentations.query('is_change')['start_time'].values
            feature_times = feature_times[~np.isnan(feature_times)]
        elif feature == 'any-image':
            feature_times = bod.stimulus_presentations.query('not omitted')['start_time'].values
        elif feature == 'image_expectation':
            stimulus_presentations = bod.stimulus_presentations[~bod.stimulus_presentations.image_name.isna()]
            feature_times = bod.stimulus_presentations['start_time'].values
            # Append last image
            feature_times = np.concatenate([feature_times,[feature_times[-1]+.75]])
        elif feature == 'omissions':
            feature_times = bod.stimulus_presentations.query('omitted')['start_time'].values
        elif (len(feature)>5) & (feature[0:5] == 'image') & ('change' not in feature):
            feature_times = bod.stimulus_presentations.query('image_index == {}'.format(int(feature[-1])))['start_time'].values
        elif (len(feature)==5) & (feature[:2] == 'im') & (feature[2:].isnumeric()): # from 'each-image'
            feature_times = bod.stimulus_presentations.query('image_name == @feature')['start_time'].values
        elif (len(feature)>5) & (feature[0:5] == 'image') & ('change' in feature):
            feature_times = bod.stimulus_presentations.query('is_change & (image_index == {})'.format(int(feature[-1])))['start_time'].values
        else:
            raise Exception('\tCould not resolve kernel label')

        # Ensure minimum number of features
        if len(feature_times) < 5: # HARD CODING THIS VALUE HERE
            raise Exception('\tLess than minimum number of features: '+str(len(feature_times)) +' '+feature)

    except Exception as e:
        print('\tError encountered while adding kernel for '+kernel_name+'. Attemping to continue without this kernel.' )
        print(e)
        # Need to remove from relevant lists
        run_params['failed_kernels'].add(kernel_name)      
        run_params['kernel_error_dict'][kernel_name] = {
            'error_type': 'kernel', 
            'kernel_name': kernel_name, 
            'exception':e.args[0], 
            'opid':bod.metadata['ophys_plane_id'], 
        }
        # Logging does not work - an error with visual_behavior_data connection
        # # log error to mongo:
        # gat.log_error(
        #     run_params['kernel_error_dict'][kernel_name], 
        #     keys_to_check = ['oeid', 'glm_version', 'kernel_name']
        # )
        return design       
    else:
        features_vec, timestamps = np.histogram(feature_times, bins=response['time_bins'])
    
        if (feature == 'lick_bouts') or (feature == 'licks'):
            unstd_timeseries = features_vec.copy()
            # Force this to be 0 or 1, since we purposefully over-tiled the space. 
            features_vec[features_vec > 1] = 1

        if np.max(features_vec) > 1:
            raise Exception('Had multiple features in the same timebin, {}'.format(kernel_name))

        design.add_kernel(
            features_vec, 
            run_params['kernels'][kernel_name]['length'], 
            kernel_name, 
            offset=run_params['kernels'][kernel_name]['offset'],
            num_weights=run_params['kernels'][kernel_name]['num_weights']
        )
        if unstd_timeseries is not None:
            design.add_unstd_features(unstd_timeseries, kernel_name)

        return design


def process_eye_data(bod, ophys_timestamps):
    '''
        Returns a dataframe of eye tracking data with several processing steps
        1. All columns are interpolated onto ophys timestamps
        2. Likely blinks are removed with a threshold set by run_params['eye_blink_z']
        3. After blink removal, a second transient step removes outliers with threshold run_params['eye_tranisent_threshold']
        4. After interpolating onto the ophys timestamps, Z-scores the eye_width and pupil_radius
        
        Does not modifiy the original eye_tracking dataframe
    '''    

    # Set parameters for blink detection, and load data
    eye = bod.eye_tracking.copy(deep=True)

    # Compute pupil radius
    eye['pupil_radius'] = np.sqrt(eye['pupil_area']*(1/np.pi))
    
    # Remove likely blinks and interpolate
    eye.loc[eye['likely_blink'],:] = np.nan
    eye = eye.interpolate()

    # Interpolate everything onto ophys_timestamps
    ophys_eye = pd.DataFrame({'timestamps':ophys_timestamps})
    z_score = ['eye_width','pupil_radius']
    for column in eye.keys():
        if column != 'timestamps':
            f = scipy.interpolate.interp1d(eye['timestamps'], eye[column], bounds_error=False)
            ophys_eye[column] = f(ophys_eye['timestamps'])
            ophys_eye[column].fillna(method='ffill',inplace=True)
            if column in z_score:
                ophys_eye[column+'_zscore'] = scipy.stats.zscore(ophys_eye[column],nan_policy='omit')
    print('                 : '+'Mean Centering')
    print('                 : '+'Standardized to unit variance')
    return ophys_eye 

