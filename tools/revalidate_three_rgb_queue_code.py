"""Rebind a completed input audit only when the sole change is the queue supervisor.

The trainer, transforms, configuration, decoder, data, statistics and cache must
be unchanged. Preserve the original accepted input audit and record its SHA.
"""
import argparse
import json
from pathlib import Path
import shutil


def changed_queue_only(old,new):
    changed=sorted(k for k in old.keys()|new.keys() if old.get(k)!=new.get(k))
    if changed!=['scripts/queue_in_hand204_three_rgb.py']:
        raise ValueError('Only queue supervisor code may be rebound; rerun the full input audit for any learner/input change')
    return changed


def main():
    from openpi.subtasks.prepare import digest
    from openpi.training import reproducibility
    from scripts.run_training import load_experiment,preflight
    from training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_three_rgb import MODULE
    from scripts.training_ui import atomic_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    old=json.loads(a.report.read_text());old_sha=digest(a.report)
    config=load_experiment(MODULE);inventory=reproducibility._code_inventory(config)
    changed=changed_queue_only(old['code_inventory'],inventory);inputs=preflight(config)
    if old.get('status')!='three_rgb204_accepted' or old.get('model_samples_checked')!=612:
        raise ValueError('A completed full input audit is required')
    if any(inputs[k]!=old['inputs'][k] for k in ['norm_stats_sha256','norm_audit_sha256','frames','episodes','dataset_root','dataset_repo_id','prompts']):
        raise ValueError('Inputs/norm changed')
    if (digest(Path(config.policy_metadata['dataset_root'])/'view.json')!=old['view_sha256']
            or digest(Path(config.policy_metadata['rgb_frame_cache'])/'manifest.json')!=old['cache_manifest_sha256']):
        raise ValueError('View/cache changed')
    archived=a.report.with_name('accepted_input_audit_0e151d6.json')
    if archived.exists():raise FileExistsError(archived)
    shutil.copyfile(a.report,archived)
    record=dict(source_report=str(archived),source_report_sha256=old_sha,changed_code_files=changed,
        old_code={k:old['code_inventory'][k] for k in changed},new_code={k:inventory[k] for k in changed},
        learner_and_input_code_unchanged=True,data_cache_norm_unchanged=True)
    atomic_json(a.report,{**old,'code_inventory':inventory,'inputs':inputs,'queue_supervisor_revalidation':record})
    print('PASS only queue supervisor changed; full input audit retained',old_sha)


if __name__=='__main__':main()
