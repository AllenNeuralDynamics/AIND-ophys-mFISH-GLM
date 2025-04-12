import pandas as pd
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm

from comb.behavior_ophys_dataset import BehaviorOphysDataset
from lamf_analysis.code_ocean import capsule_bod_utils as cbu

def load_roicat_results(roicat_dir, session_info_df):
    """ Load ROICaT results from a directory containing ROICaT results for multiple sessions.    
    Updates 03/21/2025
    - Add mouse_id to the unique_roi_name.
    - Change roi_name to session_roi_name.
    - Assign unique_roi_name to nonmatched ROIs

    Parameters
    ----------
    roicat_dir : str or Path
        Path to the directory containing ROICaT results for multiple sessions.

    Returns
    -------
    roicat_df : pd.DataFrame
        DataFrame containing ROICaT results for all cells in all sessions.
    """
    if isinstance(roicat_dir, str):
        roicat_dir = Path(roicat_dir)
    roicat_folder_names = np.arange(8)
    roicat_df_list = []
    for folder_name in roicat_folder_names:
        roicat_fp = roicat_dir / f'{folder_name}/ROICaT.tracking.results.csv'
        roicat_df = pd.read_csv(roicat_fp)
        # add new_ucid to nonmatched ROIs (instead of -1)
        max_ucid = roicat_df.ucid.max()
        nonmatched_inds = roicat_df[roicat_df['ucid'] == -1].index
        num_nonmatched = len(nonmatched_inds)
        new_ucid_nonmatched = np.arange(max_ucid + 1, max_ucid + 1 + num_nonmatched)
        roicat_df['new_ucid'] = roicat_df['ucid']
        roicat_df.loc[nonmatched_inds, 'new_ucid'] = new_ucid_nonmatched

        # assign session_key, session_roi_name, and unique_roi_name
        roicat_df['session_key'] = roicat_df['session_name'].apply(lambda x: '_'.join(x.split('_')[1:3]))
        roicat_df['mouse_id'] = roicat_df['session_key'].apply(lambda x: x.split('_')[0])
        roicat_df['session_roi_name'] = roicat_df.apply(lambda x: '_'.join(
            [x.session_key, x.fov_name, f'{x.roi_session_index:04}']), axis=1)
        roicat_df['session_ind'] = roicat_df.session_key.map(lambda x: session_info_df.loc[x, 'session_ind'])
        roicat_df['unique_roi_name'] = roicat_df.apply(lambda x: '_'.join(
            [x.mouse_id, x.fov_name, f'{x.new_ucid:04}']), axis=1)
        roicat_df_list.append(roicat_df)
    roicat_df = pd.concat(roicat_df_list)
    return roicat_df


def get_roi_df(session_info_df, data_dir=Path('/root/capsule/data')):
    """ Get ROI data for all cells in all sessions.
    Updates 03/21/2025:
    - Do not filter them, but leave valid_roi and other information (to be filtered afterwards).

    Parameters
    ----------
    session_info_df : pd.DataFrame
        DataFrame containing information about all sessions.
    data_dir : str or Path
        Path to the directory containing raw and processed data for all sessions.

    Returns
    -------
    roi_df : pd.DataFrame
        DataFrame containing ROI data for all cells in all sessions.
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    roi_df = pd.DataFrame()
    for session_key, row in tqdm(session_info_df.iterrows()):
        raw_path = row.raw_path
        session_name = row.session_name
        processed_path = str(list(data_dir.glob(f'{session_name}_processed*'))[0])

        plane_dirs = [d for d in glob.glob(processed_path + '/*') if Path(d).is_dir() and \
                    ('nwb' not in d.split('/')[-1]) and (d[-1].isnumeric())]
        for plane_dir in tqdm(plane_dirs):
            # bod = BehaviorOphysDataset(raw_folder_path=raw_path,
            #                         plane_folder_path=plane_dir,
            #                         pipeline_version='v6')
            # cell_specimen_table = cbu.get_roi_df_with_valid_roi(bod,
            cell_specimen_table = cbu.get_roi_df_with_valid_roi(plane_dir, 
                                    small_roi_radius_threshold_in_um=4)[
                                        ['cell_roi_id', 'valid_roi', 'exclusion_labels',
                                        'touching_motion_border', 'small_roi']].copy()
            cell_specimen_table['session_key'] = session_key
            cell_specimen_table['plane_session_key'] = session_key + '_' + Path(plane_dir).name
            cell_specimen_table['session_roi_name'] = cell_specimen_table.apply(lambda x: f'{x.plane_session_key}_{x.cell_roi_id:04}', axis=1)
            roi_df = pd.concat([roi_df, cell_specimen_table])
    return roi_df


def merge_roicat_df_and_roi_df(roicat_df, roi_df):
    """ Merge ROICat results and ROI dataframes.

    Parameters
    ----------
    roicat_df : pd.DataFrame
        DataFrame containing ROICat results.
    roi_df : pd.DataFrame
        DataFrame containing ROI data.

    Returns
    -------
    roicat_df : pd.DataFrame
        DataFrame containing ROICat results with additional ROI data.
    """
    if roi_df.index.name != 'session_roi_name':
        roi_df.set_index('session_roi_name', drop=True, inplace=True)
    if roicat_df.index.name != 'session_roi_name':
        roicat_df.set_index('session_roi_name', drop=True, inplace=True)
    unique_roi_df_columns = list(set(roi_df.columns) - set(roicat_df.columns))
    roicat_df = roicat_df.join(roi_df[unique_roi_df_columns], how='left', rsuffix='_roi_df')

    assert len(np.where(roicat_df.small_roi.isnull())[0]) == 0
    assert len(np.where(roicat_df.valid_roi.isnull())[0]) == 0
    assert len(np.where(roicat_df.touching_motion_border.isnull())[0]) == 0

    return roicat_df


def get_merged_roicat_df(session_info_df, roicat_dir):
    """ Get merged ROICaT results for all cells in all sessions.

    Parameters
    ----------
    session_info_df : pd.DataFrame
        DataFrame containing information about all sessions.
    roicat_dir : str or Path
        Path to the directory containing ROICaT results for multiple sessions.

    Returns
    -------
    roicat_df : pd.DataFrame
        DataFrame containing merged ROICaT results for all cells in all sessions.
    """
    roicat_df = load_roicat_results(roicat_dir, session_info_df)
    roi_df = get_roi_df(session_info_df)
    roicat_df = merge_roicat_df_and_roi_df(roicat_df, roi_df)
    return roicat_df