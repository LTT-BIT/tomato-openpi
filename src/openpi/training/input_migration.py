"""Explicit audit for a pixel-equivalent reader change during checkpoint resume."""
import hashlib
import json
import os
from pathlib import Path


ALLOWED_CODE_CHANGES = {
    'src/openpi/training/data_loader.py',
    'src/openpi/training/frame_cache.py',
    'src/openpi/training/input_migration.py',
    'src/openpi/training/reproducibility.py',
    'scripts/build_rgb_frame_cache.py',
    'scripts/prepare_rgb_cache_resume.py',
    'scripts/tomato100_cached_resume.py',
    # These UI-only changes were made after the original CLI launch.
    'scripts/training_ui.py',
    # Separate offline ROI preparation added after this run; frozen training videos are unchanged.
    'scripts/tomato_full_block_roi.py',
    'scripts/tomato_offline_roi.py',
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_code_change(previous, current):
    if {k:v for k,v in previous.items() if k != 'code'} != {k:v for k,v in current.items() if k != 'code'}:
        raise ValueError('Input acceleration cannot change data, normalization, sampler, model or training settings')
    old, new = previous['code'], current['code']
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    if not changed or changed - ALLOWED_CODE_CHANGES:
        raise ValueError('Unapproved code changes for cached resume: ' + str(sorted(changed)))
    return sorted(changed)


def validate_migration(previous, current, report_path, *, cache_path=None):
    changed = check_code_change(previous, current)
    report = json.loads(Path(report_path).read_text())
    expected = {'status':'exact_input_acceleration_validated',
        'old_contract_sha256':fingerprint(previous), 'new_contract_sha256':fingerprint(current),
        'changed_code_files':changed, 'checkpoint_step':4000}
    if any(report.get(k) != v for k,v in expected.items()):
        raise ValueError('Cached-resume report does not match the original and current contracts')
    cache_path = Path(cache_path or os.environ['PI05_RGB_FRAME_CACHE']).resolve()
    manifest = cache_path/'manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != report['cache_manifest_sha256']:
        raise ValueError('Cached-resume manifest changed')
    checks = report.get('validation', {})
    if (checks.get('all_model_inputs_equal') is not True or checks.get('episodes_checked') != 100
            or checks.get('samples_checked', 0) < 300
            or checks.get('loader_batches', 0) < 24
            or checks.get('steady_batch_seconds', float('inf')) >= 5.5):
        raise ValueError('Exact full100 inputs and adequate loader throughput required')
    return report
