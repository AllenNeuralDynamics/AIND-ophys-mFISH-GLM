import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import skimage
    
from pathlib import Path
import json
from typing import Union, Dict
import h5py
from skimage import measure, io
from skimage.measure import regionprops


def add_trials_to_bod(bod, response_window=[0.15, 0.75]):
    """ Temporary fix to add trials to bod
    using stimulus_presentations and licks.
    No correct rejection for this.
    Columns: 'change_time', 'hit', 'miss'

    Parameters
    ----------
    bod : BehaviorOphysDataset
        The behavior ophys dataset object.

    Returns
    -------
    bod : BehaviorOphysDataset
        The behavior ophys dataset object with trials.
    """

    stimulus_presentations = bod.stimulus_presentations
    lick_times = bod.licks.timestamps.values
    trials = pd.DataFrame(columns=['change_time', 'hit', 'miss'])

    change_times = stimulus_presentations.query('is_change').start_time.values
    response_windows = np.array([change_times + response_window[0], change_times + response_window[1]]).T
    hit = np.zeros(len(change_times), 'bool')
    for i, window in enumerate(response_windows):
        if np.any((lick_times > window[0]) & (lick_times < window[1])):
            hit[i] = 1
    miss = ~hit
    trials = pd.DataFrame({'change_time': change_times, 'hit': hit, 'miss': miss})

    bod.trials = trials
    return bod


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
