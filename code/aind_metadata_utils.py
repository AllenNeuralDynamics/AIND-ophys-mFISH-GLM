"""
aind_metadata_utils.py — AIND data-schema compliant metadata outputs.

Writes two files per run:
  processing.json          — aind-data-schema Processing object
  data_description.json   — aind-data-schema DerivedDataDescription derived
                             from the processed folder's data_description.json
"""

import json
import platform
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# ── AIND schema imports (graceful fallback) ───────────────────────────────────
try:
    from aind_data_schema.core.processing import DataProcess, PipelineProcess, Processing
    _HAS_AIND_PROCESSING = True
except ImportError:
    _HAS_AIND_PROCESSING = False
    warnings.warn('aind-data-schema not installed — processing.json will use '
                  'plain JSON fallback schema.', RuntimeWarning)

try:
    from aind_data_schema.core.data_description import DerivedDataDescription
    _HAS_AIND_DD = True
except ImportError:
    _HAS_AIND_DD = False
    warnings.warn('aind-data-schema not installed — data_description.json will use '
                  'plain JSON fallback schema.', RuntimeWarning)


def save_processing_json(
        session_key,
        session_name,
        data_type,
        kernel_config_path,
        kernel_dict,
        fit_params,
        proc_dir,
        save_dir,
        start_time: datetime,
        end_time: datetime,
        code_url: str = 'https://github.com/AllenNeuralDynamics/AIND-ophys-mFISH-GLM',
):
    """Write an AIND-schema Processing object to save_dir/processing.json.

    Falls back to a plain JSON dict if aind-data-schema is not installed.
    """
    save_dir = Path(save_dir)

    params = {
        'session_key':    session_key,
        'session_name':   session_name,
        'data_type':      data_type,
        'output_dir':     str(save_dir),
        'kernel_config': {
            'file':    str(kernel_config_path),
            'kernels': {
                k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                for k, v in kernel_dict.items()
            },
        },
        'fit_params': {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in fit_params.items()
        },
        'environment': {
            'python':  platform.python_version(),
            'numpy':   np.__version__,
            'xarray':  xr.__version__,
            'platform': platform.platform(),
        },
    }

    out = save_dir / 'processing.json'

    if _HAS_AIND_PROCESSING:
        try:
            dp = DataProcess(
                name='Analysis',
                software_version='1.0.0',
                start_date_time=start_time,
                end_date_time=end_time,
                input_location=str(proc_dir),
                output_location=str(save_dir),
                code_url=code_url,
                parameters=params,
            )
            pp = PipelineProcess(
                data_processes=[dp],
                processor_full_name='mFISH-GLM capsule',
            )
            proc_obj = Processing(processing_pipeline=pp)
            out.write_text(proc_obj.model_dump_json(indent=2))
            print(f'processing.json (AIND schema) saved → {out}')
            return
        except Exception as e:
            warnings.warn(f'AIND schema serialisation failed ({e}); '
                          'falling back to plain JSON.', RuntimeWarning)

    # Fallback: plain JSON with same key structure
    doc = {
        'schema_version': '1.0',
        'start_time':     start_time.isoformat(),
        'end_time':       end_time.isoformat(),
        'duration_s':     round((end_time - start_time).total_seconds(), 1),
        **params,
    }
    with open(out, 'w') as f:
        json.dump(doc, f, indent=2, default=str)
    print(f'processing.json (plain JSON fallback) saved → {out}')


def save_data_description_json(proc_dir, save_dir, process_name='glm'):
    """Write data_description.json derived from the processed folder's copy.

    Reads <proc_dir>/data_description.json, then creates a DerivedDataDescription
    (schema v1.2.0) with input_data_name pointing at the processed folder and
    process_name='glm'.  Falls back to a patched plain-JSON copy if the schema
    library is not available.
    """
    proc_dir = Path(proc_dir)
    save_dir = Path(save_dir)

    src_path = proc_dir / 'data_description.json'
    if not src_path.exists():
        print(f'  Warning: {src_path} not found — data_description.json skipped.')
        return

    src = json.loads(src_path.read_text())
    out = save_dir / 'data_description.json'

    if _HAS_AIND_DD:
        try:
            derived = DerivedDataDescription(
                input_data_name=proc_dir.name,
                process_name=process_name,
                creation_time=datetime.utcnow(),
                institution=src.get('institution'),
                investigators=src.get('investigators', []),
                project_name=src.get('project_name'),
                subject_id=src.get('subject_id'),
                modality=src.get('modality', []),
                platform=src.get('platform'),
                license=src.get('license', 'CC-BY-4.0'),
            )
            out.write_text(derived.model_dump_json(indent=2))
            print(f'data_description.json (AIND schema) saved → {out}')
            return
        except Exception as e:
            warnings.warn(f'AIND DerivedDataDescription failed ({e}); '
                          'falling back to patched plain JSON.', RuntimeWarning)

    # Fallback: patch the source doc
    doc = dict(src)
    doc['data_level']        = 'derived'
    doc['input_data_name']   = proc_dir.name
    doc['process_name']      = process_name
    doc['creation_time']     = datetime.utcnow().isoformat()
    doc.pop('schema_version', None)   # let consumer handle version differences

    with open(out, 'w') as f:
        json.dump(doc, f, indent=2, default=str)
    print(f'data_description.json (plain JSON fallback) saved → {out}')
