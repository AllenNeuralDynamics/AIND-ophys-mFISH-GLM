"""
aind_metadata_utils.py — AIND data-schema compliant metadata outputs.

Follows the pattern of lamf_analysis.code_ocean.json_utils (aind-data-schema==1.2.0).

Main entry point:
    write_metadata_files(session_name, proc_dir, save_dir, start_dt, end_dt,
                         run_parameters, process_name='glm')

Writes:
    processing.json        — via Processing.write_standard_file()
    data_description.json  — via DerivedDataDescription.from_data_description()
    session.json / subject.json / procedures.json / rig.json  (copied from data)
"""

import json
import shutil
import datetime
from pathlib import Path

import aind_data_schema
assert aind_data_schema.__version__ == '1.2.0', \
    f'Expected aind-data-schema==1.2.0, got {aind_data_schema.__version__}'

from aind_data_schema.core.data_description import (
    DataDescription,
    DerivedDataDescription,
    DataLevel,
    Organization,
    Modality,
    Platform,
    Funding,
)
from aind_data_schema.core.processing import DataProcess, Processing, PipelineProcess
from aind_data_schema_models.pid_names import PIDName

DATA_DIR    = Path('/data')
RESULTS_DIR = Path('/results')

INPUT_PROCESSING_DICT = {
    'name':             'Analysis',
    'software_version': '1.0.0',
    'code_url':         'https://github.com/AllenNeuralDynamics/AIND-ophys-mFISH-GLM',
    'notes':            'mFISH multiplane ophys encoding GLM fit',
}


def _copy_core_json(source_asset_name: str, data_dir: Path, results_dir: Path):
    for fname, dest_name in [
        ('session.json',    'session.json'),
        ('subject.json',    'subject.json'),
        ('procedures.json', 'procedures.json'),
        ('rig.json',        'rig.json'),
        ('instrument.json', 'instrument.json'),
    ]:
        src = next(data_dir.rglob(f'*{source_asset_name}*/{fname}'), None)
        if src:
            shutil.copy(src, results_dir / dest_name)
        else:
            print(f'  No {fname} found for {source_asset_name}')


def _base_data_description_dict(subject_id: str) -> dict:
    return {
        'institution':    Organization.AIND,
        'investigators':  [PIDName(name='Unknown')],
        'funding_source': [Funding(funder=Organization.AI)],
        'modality':       [Modality.POPHYS],
        'platform':       Platform.MULTIPLANE_OPHYS,
        'subject_id':     subject_id,
    }


def _data_description_dict(capture_name: str, source_asset_name: str,
                            processed_dd: dict) -> dict:
    copy_keys = ['institution', 'investigators', 'funding_source',
                 'modality', 'platform', 'subject_id']
    dd = {}
    for key in copy_keys:
        if key in processed_dd:
            dd[key] = processed_dd[key]
        else:
            print(f'  Warning: {key} not found in source data_description.json')
            dd[key] = None
    dd['creation_time'] = datetime.datetime.now()
    dd['name']          = capture_name
    dd['data_level']    = DataLevel.DERIVED
    return dd


def _processing_dict(start_dt: datetime.datetime, end_dt: datetime.datetime,
                     run_parameters: dict, input_processing_dict: dict,
                     data_dir: Path, results_dir: Path) -> dict:
    return {
        'name':             input_processing_dict['name'],
        'software_version': input_processing_dict['software_version'],
        'start_date_time':  str(start_dt),
        'end_date_time':    str(end_dt),
        'input_location':   data_dir.as_posix(),
        'output_location':  results_dir.as_posix(),
        'code_url':         input_processing_dict['code_url'],
        'parameters':       run_parameters,
        'notes':            input_processing_dict['notes'],
        'outputs':          {},
    }


def write_metadata_files(
        session_name: str,
        proc_dir: Path,
        save_dir: Path,
        start_dt: datetime.datetime,
        end_dt: datetime.datetime,
        run_parameters: dict,
        process_name: str = 'glm',
        processor_full_name: str = 'Jinho Kim',
        input_processing_dict: dict = INPUT_PROCESSING_DICT,
        data_dir: Path = DATA_DIR,
        results_dir: Path = RESULTS_DIR,
):
    """Write processing.json and data_description.json, and copy core JSON files.

    Parameters
    ----------
    session_name    Raw session folder name (e.g. 'multiplane-ophys_800792_…')
    proc_dir        Path to the processed session directory (source of data_description.json)
    save_dir        Directory where GLM outputs are written
    results_dir     Directory where processing.json and data_description.json are written
    start_dt / end_dt  Wall-clock times bracketing the GLM fit
    run_parameters  Dict of all CLI + kernel parameters to log
    process_name    String appended to the derived data description (default 'glm')
    """
    save_dir    = Path(save_dir)
    proc_dir    = Path(proc_dir)
    results_dir = Path(results_dir)

    # ── data_description.json ─────────────────────────────────────────────────
    source_asset_name = proc_dir.name          # e.g. 'multiplane-ophys_800792_…_processed_…'
    subject_id        = session_name.split('_')[1]
    capture_name      = save_dir.name          # e.g. '800792_2025-08-18_glm_v01'

    dd_path = proc_dir / 'data_description.json'
    if dd_path.exists():
        with dd_path.open('r') as f:
            processed_dd = json.load(f)
        dd_dict = _data_description_dict(capture_name, source_asset_name, processed_dd)
    else:
        print(f'  No data_description.json in {proc_dir.name} — using base template')
        dd_dict = _base_data_description_dict(subject_id)
        dd_dict.update({'creation_time': datetime.datetime.now(),
                        'name': capture_name, 'data_level': DataLevel.DERIVED})

    data_description = DataDescription(**dd_dict)
    derived_dd = DerivedDataDescription.from_data_description(
        data_description=data_description, process_name=process_name)
    dd_json = derived_dd.model_dump_json(indent=3)
    with (results_dir / 'data_description.json').open('w') as f:
        f.write(dd_json)
    print(f'data_description.json saved → {results_dir / "data_description.json"}')

    # ── processing.json ───────────────────────────────────────────────────────
    proc_dict = _processing_dict(start_dt, end_dt, run_parameters,
                                 input_processing_dict, data_dir, results_dir)
    processing_model    = DataProcess(**proc_dict)
    processing_pipeline = PipelineProcess(
        data_processes=[processing_model],
        processor_full_name=processor_full_name)
    processing = Processing(processing_pipeline=processing_pipeline)
    processing.write_standard_file(results_dir)
    print(f'processing.json saved → {results_dir / "processing.json"}')

    # ── copy core JSON files to results root ──────────────────────────────────
    _copy_core_json(session_name, data_dir, results_dir)
