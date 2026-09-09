from comb.behavior_ophys_dataset import BehaviorOphysDataset, BehaviorMultiplaneOphysDataset
from comb.behavior_session_dataset import BehaviorSessionDataset
from lamf_analysis import utils as lamf_utils
import os
import re
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import xarray as xr
import scipy
import pandas as pd


def load_plane_data(session_name, opid=None, opid_ind=None, data_dir='/root/capsule/data/'):
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
                print(f'Using plane {opid} for session {session_name}')
            else:
                raise ValueError(f'Processed data for session {session_name} has less than {opid_ind} planes')
    raw_path = Path(plane_path.split('_processed')[0])

    if not raw_path.exists():
        raise ValueError(f'No raw data found for session {session_name}')
    bod = BehaviorOphysDataset(plane_folder_path=plane_path,
                               raw_folder_path=raw_path,
                               eye_tracking_path=eye_path,
                               pipeline_version='v6')
    bod.metadata['ophys_plane_id'] = opid
    cell_specimen_table = get_roi_df_with_valid_roi(bod)
    
    return bod


def get_roi_df_with_valid_roi(bod, small_roi_radius_threshold_in_um=4):
    # This mutates bod.cell_specimen_table, so need to be run only once per bod loading
    cell_specimen_table = bod.cell_specimen_table
    if np.array([k in cell_specimen_table.columns for k in ['touching_motion_border', 'small_roi', 'valid_roi']]).all():
        return cell_specimen_table
    else:
        plane_path = bod.metadata['plane']['plane_path']
        range_y, range_x = lamf_utils.get_motion_correction_crop_xy_range(plane_path)
        range_y = [int(range_y[0]), -int(range_y[1])]
        range_x = [int(range_x[0]), -int(range_x[1])]
        
        on_mask = np.zeros((bod.metadata['plane']['fov_height'], bod.metadata['plane']['fov_width']), dtype=bool)
        on_mask[range_y[0]:range_y[1], range_x[0]:range_x[1]] = True
        motion_mask = ~on_mask

        def _touching_motion_border(row, motion_mask):
            if (row.mask_matrix * motion_mask).any():
                return True
            else:
                return False

        cell_specimen_table['touching_motion_border'] = cell_specimen_table.apply(_touching_motion_border, axis=1, motion_mask=motion_mask)
        
        small_roi_radius_threshold_in_pix = small_roi_radius_threshold_in_um / float(bod.metadata['plane']['fov_scale_factor'])
        area_threshold = np.pi * (small_roi_radius_threshold_in_pix**2)
        
        cell_specimen_table['small_roi'] = cell_specimen_table['mask_matrix'].apply(lambda x: len(np.where(x)[0]) < area_threshold)
        cell_specimen_table['valid_roi'] = ~cell_specimen_table['touching_motion_border'] & ~cell_specimen_table['small_roi']
    
    return cell_specimen_table


def add_ophys_plane_id(bod):
    if 'ophys_plane_id' not in bod.metadata:
        subject_id = bod.metadata['subject_id']
        session_date = bod.metadata['session_date']
        folder_name = bod.metadata['plane']['plane_path'].stem
        bod.metadata['ophys_plane_id'] = f'{subject_id}_{session_date}_{folder_name}'
    return bod


def load_behavior_data(session_name, data_dir='/root/capsule/data/'):
    ''' Load beahvrio data using COMB.

    Parameters
    ----------
    session_name : str
        name of the session (e.g., 'multiplane-ophys_721291_2024-05-08_08-05-54')
    data_dir : str (optional)
        path to the data directory
    
    Returns
    -------
    bsd : BehaviorSessionDataset
        COMB object containing the behavior data
    '''

    data_dir = Path(data_dir)
    raw_dir = data_dir / session_name
    eye_dir = glob.glob(str(data_dir / f'{session_name}*dlc-eye*'))[0]
    if raw_dir.exists():
        behavior_dataset = BehaviorSessionDataset(raw_folder_path=raw_dir,
                                                  eye_tracking_path=eye_dir)
    else:
        raise ValueError(f'Multiple processed data found for session {session_name}')

    return behavior_dataset


def extract_and_annotate_ophys_plane(bod, run_params, TESTING=False):
    '''
        Creates fit dictionary
        extracts dff_trace or events_trace from session object
        sets up the timestamps to be used
        sets up bins for binning times onto the ophys timestamps
    '''
    activity_trace = dict()
    activity_trace['activity_trace_arr'] = process_data(bod, run_params, TESTING=TESTING)
    activity_trace['timestamps'] = [float(round(ts, 4)) for ts in \
                                    activity_trace['activity_trace_arr']['timestamps'].values]
    step = np.mean(np.diff(activity_trace['timestamps']))
    step = float(round(step, 4))
    activity_trace['time_bins'] = np.concatenate([activity_trace['timestamps'],
                                                 [activity_trace['timestamps'][-1]+step]])-step*.5  
    # TODO: better to use previous frame end time and current frame end time. But for now just leave it as-is, 
    # because it's very minor and used for lick counts only.
    ophys_frame_rate = bod.ophys_plane_dataset.metadata['ophys_frame_rate']
    activity_trace['ophys_frame_rate'] = float(round(ophys_frame_rate, 3))
    
    # Interpolate onto stimulus 
    activity_trace, run_params = interpolate_to_stimulus(activity_trace, bod, run_params)

    # TODO: consider splitting on engagement (or using it as a feature)
    return activity_trace, run_params


def process_data(bod, activity_trace_params, TESTING=False):
    """ Processes activity traces by trimming off portions of recording session outside of the task period. These include:
        * a ~5 minute gray screen period before the task begins
        * a ~5 minute gray screen period after the task ends
        * a 5-10 minute movie following the second gray screen period

    Parameters
    ----------
    bod : behaviorOphysPlaneDataset object
        COMB object containing the ophys plane data
    activity_trace_params : dict
        dictionary containing the activity_trace parameters
    TESTING : bool, optional
        if True, only includes the first 6 cells of the experiment, by default False

    Returns
    -------
    xarray
        neuronal activity traces with dimensions [timestamps, cell_roi_ids] 
    """

    # clip off the grey screen periods
    timestamps_to_use = get_ophys_frames_to_use(bod)  # boolean mask

    if activity_trace_params['data_type'] == 'events':
        print('Using events traces')
        activity_trace_arr = get_events_xr(bod, timestamps_to_use, filtered=False)
    elif activity_trace_params['data_type'] == 'filtered_events':
        print('Using filtered events traces')
        activity_trace_arr = get_events_xr(bod, timestamps_to_use, filtered=True)
    elif activity_trace_params['data_type'] == 'dff':
        print('Using dff traces')
        activity_trace_arr = get_dff_xr(bod, timestamps_to_use)
    else:
        raise ValueError('Invalid data_type. Must be one of ["events", "filtered_events", "dff"]')

    # some assert statements to ensure that dimensions are correct
    assert np.sum(timestamps_to_use) == len(activity_trace_arr['timestamps'].values), 'length of `timestamps_to_use` must match length of `timestamps` in `activity_trace_arr`'
    assert np.sum(timestamps_to_use) == activity_trace_arr.values.shape[0], 'length of `timestamps_to_use` must match 0th dimension of `activity_trace_arr`'
    
    # Clip the array to just the first 6 cells
    if TESTING:
        activity_trace_arr = activity_trace_arr[:,0:6]
           
    return activity_trace_arr


def get_ophys_frames_to_use(bod, end_buffer=0.5, stim_dur=0.25):
    '''
    Trims out the grey period at start, end, and the fingerprint.

    Parameters
    ----------
    bod : behaviorOphysPlaneDataset object
        COMB object containing the ophys plane data
    end_buffer: float
        duration in seconds to extend beyond end of last stimulus presentation (default = 0.5)
    stim_dur: float
        duration in seconds of stimulus presentations
    
    Returns
    -------
    np.array of bool
        Boolean mask with which ophys frames to use
    '''
    # filter out omitted flashes to avoid omitted flashes at the start of the session from affecting analysis range
    filtered_stimulus_presentations = bod.stimulus_presentations[~bod.stimulus_presentations.image_name.isna()]
    while filtered_stimulus_presentations.iloc[0]['omitted'] == True:
        filtered_stimulus_presentations = filtered_stimulus_presentations.iloc[1:]
    ophys_timestamps = bod.ophys_timestamps.values

    ophys_frames_to_use = (
        (ophys_timestamps >= filtered_stimulus_presentations.iloc[0]['start_time']-end_buffer) 
        & (ophys_timestamps <= filtered_stimulus_presentations.iloc[-1]['start_time'] +stim_dur+ end_buffer)
    )
    return ophys_frames_to_use


def get_events_xr(bod, timestamps_to_use, filtered=False):
    '''
    Get the events traces from a session in xarray format (preserves cell ids and timestamps)

    timestamps_to_use is a boolean vector that contains which timestamps to use in the analysis
    '''
    # Get events and trim off ends
    valid_cell_ids = bod.cell_specimen_table[bod.cell_specimen_table.valid_roi].index.values
    valid_events = bod.events.loc[valid_cell_ids]
    if filtered:
        all_events = np.stack(valid_events['filtered_events'].values)
    else:
        all_events = np.stack(valid_events['events'].values)
    all_events_to_use = all_events[:, timestamps_to_use]

    # Get the timestamps
    events_trace_timestamps = bod.ophys_timestamps.values
    events_trace_timestamps_to_use = events_trace_timestamps[timestamps_to_use]

    # Note: it may be more efficient to get the xarrays directly, rather than extracting/building them from session.events_traces
    #       The dataframes are built from xarrays to start with, so we are effectively converting them twice by doing this
    #       But if there's no big time penalty to doing it this way, then maybe just leave it be.
    # Intentionally setting the name of the time axis to fit_trace_timestamps so it matches the fit_trace_xr
    bod = add_ophys_plane_id(bod)
    opid = bod.metadata['ophys_plane_id']    
    session_unique_csids = [f'{opid}_{int(rn):04}' for rn in valid_events.index.values]

    events_trace_xr = xr.DataArray(
            data = all_events_to_use.T,
            dims = ("timestamps", "cell_roi_id"),
            coords = {
                "timestamps": events_trace_timestamps_to_use,
                "cell_roi_id": session_unique_csids  # TODO: add checking valid ROIs. For now, assume all events from COMB are valid
            }
        )
    return events_trace_xr


def get_dff_xr(bod, timestamps_to_use):
    '''
    Get the dff traces from a session in xarray format (preserves cell ids and timestamps)

    timestamps_to_use is a boolean vector that contains which timestamps to use in the analysis
    '''
    # Get dff and trim off ends
    valid_cell_ids = bod.cell_specimen_table[bod.cell_specimen_table.valid_roi].index.values
    valid_dff = bod.dff_traces.loc[valid_cell_ids]
    all_dff = np.stack(valid_dff['dff'].values)
    all_dff_to_use = all_dff[:, timestamps_to_use]

    # Get the timestamps
    dff_trace_timestamps = bod.ophys_timestamps.values
    dff_trace_timestamps_to_use = dff_trace_timestamps[timestamps_to_use]

    # Note: it may be more efficient to get the xarrays directly, rather than extracting/building them from session.dff_traces
    #       The dataframes are built from xarrays to start with, so we are effectively converting them twice by doing this
    #       But if there's no big time penalty to doing it this way, then maybe just leave it be.
    bod = add_ophys_plane_id(bod)
    opid = bod.metadata['ophys_plane_id']    
    session_unique_csids = [f'{opid}_{int(rn):04}' for rn in valid_dff.index.values]

    dff_trace_xr = xr.DataArray(
            data = all_dff_to_use.T,
            dims = ("timestamps", "cell_roi_id"),
            coords = {
                "timestamps": dff_trace_timestamps_to_use,
                "cell_roi_id": session_unique_csids
            }
        )
    return dff_trace_xr


def interpolate_to_stimulus(activity_trace, bod, run_params, stimulus_interval=0.75):
    '''
        This function interpolates the neural signal (either dff or events) onto timestamps that are aligned to the stimulus.
        
        The new timestamps are aligned to the onset of each image presentation (or omission), and the last timebin in each 750ms image
        cycle is allowed to be variable to account for variability in image presentation start times, and the ophys timestamps not perfect
        dividing the image cycle. 
    '''
    # if ('interpolate_to_stimulus' not in run_params) or (not run_params['interpolate_to_stimulus']):
    #     print('Not interpolating onto stimulus aligned timestamps')
    #     return activity_trace, run_params
    # TODO: Is there a reason to not interpolate to stimulus aligned timestamps? 
    print('Interpolating neural signal onto stimulus aligned timestamps')
 
    # Find first non omitted stimulus and remove it (because it cannot be distinguished from initial gray screen period)
    filtered_stimulus_presentations = bod.stimulus_presentations[~bod.stimulus_presentations.image_name.isna()]
    while filtered_stimulus_presentations.iloc[0]['omitted'] == True:
        filtered_stimulus_presentations = filtered_stimulus_presentations.iloc[1:]

    # Make new timestamps by starting with each stimulus start time, and adding time points until we hit the next stimulus
    start_times = filtered_stimulus_presentations.start_time.values
    start_times = np.concatenate([start_times, [start_times[-1] + stimulus_interval]]) 
    mean_step = np.mean(np.diff(activity_trace['timestamps']))  #TODO: consider using ophys frame rate
    mean_step = float(round(mean_step, 4))
    # mean_step = 1 / bod.ophys_plane_dataset.metadata['ophys_frame_rate']
    sets_of_stimulus_timestamps = []
    for index, start in enumerate(start_times[0:-1]):
        sets_of_stimulus_timestamps.append(np.arange(start_times[index], start_times[index + 1] - mean_step / 2, mean_step)) 

    sets_of_stimulus_timestamps, mode = check_same_number_per_stimulus(sets_of_stimulus_timestamps, run_params)

    # Combine all the timestamps together
    new_timestamps = np.concatenate(sets_of_stimulus_timestamps)
    new_timestamps = np.array([float(round(ts, 4)) for ts in new_timestamps])
    new_bins = np.concatenate([new_timestamps, [new_timestamps[-1] + mean_step]]) - mean_step / 2

    # Check if it was already interpolated
    if np.array_equal(new_timestamps, activity_trace['timestamps']):
        print('Already interpolated onto stimulus aligned timestamps')
        return fit, run_params
    else:
        # Setup new variables 
        num_cells = np.size(activity_trace['activity_trace_arr'], 1)
        new_trace_arr = np.empty((len(new_timestamps), num_cells))
        new_trace_arr[:] = 0
        
        # Interpolate onto new timestamps
        for index in range(0,num_cells):
            # Fit array
            f = scipy.interpolate.interp1d(activity_trace['timestamps'],
                                           activity_trace['activity_trace_arr'][:,index],
                                           bounds_error=False, fill_value='extrapolate')
            new_trace_arr[:,index] = f(new_timestamps)

        # Convert into xarrays
        new_trace_arr = xr.DataArray(
            new_trace_arr, 
            dims = ('timestamps','cell_roi_id'), 
            coords = {
                'timestamps': new_timestamps,
                'cell_roi_id': activity_trace['activity_trace_arr']['cell_roi_id'].values
            }
        )

        # Save everything
        activity_trace['stimulus_interpolation'] = {
            'mean_step': mean_step,
            'timesteps_per_stimulus': mode,
            'original_activity_trace_arr': activity_trace['activity_trace_arr'],
            'original_timestamps': activity_trace['timestamps'],
            'original_bins': activity_trace['time_bins']
        }
        activity_trace['activity_trace_arr'] = new_trace_arr
        activity_trace['timestamps'] = new_timestamps
        activity_trace['time_bins'] = new_bins
    
        # Use the number of timesteps per stimulus to define the image kernel length so we get no overlap 
        # kernels_to_limit_per_image_cycle = ['image0','image1','image2','image3','image4','image5','image6','image7']
        kernels_to_limit_per_image_cycle = [k for k in run_params['kernels'].keys() if 'image' in k]
        if 'post-omissions' in run_params['kernels']:
            kernels_to_limit_per_image_cycle.append('omissions')
        if 'post-hits' in run_params['kernels']:
            kernels_to_limit_per_image_cycle.append('hits')
            kernels_to_limit_per_image_cycle.append('misses')
            kernels_to_limit_per_image_cycle.append('passive_change')
        for k in kernels_to_limit_per_image_cycle:
            if k in run_params['kernels']:
                run_params['kernels'][k]['num_weights'] = activity_trace['stimulus_interpolation']['timesteps_per_stimulus']    

        # Check to make sure there are no NaNs in the fit_trace
        assert np.isnan(activity_trace['activity_trace_arr']).sum() == 0, "Have NaNs in activity_trace_arr"

    return activity_trace, run_params


def check_same_number_per_stimulus(sets_of_stimulus_timestamps, run_params):
    """ Check to make sure we always have the same number of timestamps per stimulus

    Parameters
    ----------
    sets_of_stimulus_timestamps : list
        list of arrays of timestamps
    run_params : dict
        dictionary containing the run parameters

    Returns
    -------
    list
        list of arrays of timestamps
    """
    # Check to make sure we always have the same number of timestamps per stimulus
    lens = [len(x) for x in sets_of_stimulus_timestamps]
    mode = scipy.stats.mode(lens)[0]
    if len(np.unique(lens)) > 1:
        u,c = np.unique(lens, return_counts=True)
        for index, val in enumerate(u):
            print('   Stimuli with {} timestamps: {}'.format(u[index], c[index]))
        print('   This happens when the following stimulus is delayed creating a greater than 750ms duration')
        print('   I will truncate extra timestamps so that all stimuli have the same number of following timestamps')
    
        # Determine how many timestamps each stimuli most commonly has and trim off the extra
        sets_of_stimulus_timestamps = [x[0:mode] for x in sets_of_stimulus_timestamps]

        # Check again to make sure we always have the same number of timestamps
        # Note this can still fail if the stimulus duration is less than 750
        lens = [len(x) for x in sets_of_stimulus_timestamps]
        if len(np.unique(lens)) > 1:
            print('   Warning!!! uneven number of steps per stimulus interval')
            print('   This happens when the stimulus duration is much less than 750ms')
            print('   I will need to check for this happening when kernels are added to the design matrix')
            u,c = np.unique(lens, return_counts=True)
            overlaps = 0
            for index, val in enumerate(u):
                print('Stimuli with {} timestamps: {}'.format(u[index], c[index]))
                if u[index] < mode:
                    overlaps += (mode-u[index])*c[index]
            if run_params.get('image_kernel_overlap_tol', 0) > 0:
                print('checking to see if image kernel overlap is within tolerance ({})'.format(run_params['image_kernel_overlap_tol']))
                print('overlapping timestamps: {}'.format(overlaps))
                if overlaps > run_params['image_kernel_overlap_tol']:
                    raise Exception('Uneven number of steps per stimulus interval')
                else:
                    print('I think I am under the tolerance, continuing')
    return sets_of_stimulus_timestamps, mode


# ── capsule-specific session discovery and multi-plane loading ────────────────

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
            f'{[p.name for p in raw_dirs]}')
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


def is_plane_dir(p: Path) -> bool:
    return p.is_dir() and bool(re.match(r'^[A-Za-z]+_\d+$', p.name))


def merge_trials(bod):
    """Add hit/miss columns to bod.stimulus_presentations (mutates in place)."""
    stim = bod.stimulus_presentations
    stim['is_change'] = stim.is_change.astype(bool)
    stim['hit']  = False
    stim['miss'] = False
    try:
        trials = bod.trials
        stim.loc[stim.start_time.isin(trials.query('hit').change_time.values),  'hit']  = True
        stim.loc[stim.start_time.isin(trials.query('miss').change_time.values), 'miss'] = True
    except AttributeError:
        pass  # passive / training sessions without go-nogo task have no trials


def _load_plane(plane_dir, raw_dir, eye_dir):
    bod = BehaviorOphysDataset(
        plane_folder_path=plane_dir,
        raw_folder_path=raw_dir,
        eye_tracking_path=eye_dir,
        pipeline_version='v6',
    )
    bod.metadata['ophys_plane_id'] = plane_dir.name
    merge_trials(bod)
    return bod


def load_all_planes(proc_dir, raw_dir, eye_dir, n_workers=4):
    """Load all imaging planes from a processed session directory in parallel.

    Returns a list of BehaviorOphysDataset objects sorted by plane name.
    """
    plane_dirs = sorted([p for p in Path(proc_dir).iterdir() if is_plane_dir(p)])
    print(f'Loading {len(plane_dirs)} planes ({n_workers} threads): {[p.name for p in plane_dirs]}')
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_load_plane, pd, raw_dir, eye_dir): pd for pd in plane_dirs}
        bod_map = {}
        for fut, pd in futures.items():
            bod_map[pd.name] = fut.result()
            print(f'  loaded {pd.name}')
    return [bod_map[pd.name] for pd in plane_dirs]
