"""Guarded check/start for the independent in-hand119 B run. Default is check."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from openpi.subtasks import tomato_in_hand as contract
from scripts.check_tomato_in_hand119 import MODULE
from scripts.tomato_ab_run import gpu_inventory,validate_gpus

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'logs/in_hand119_preflight/accepted_20261116.json'


def checked_config(bundle,cache,report_path):
    from openpi.subtasks.prepare import digest
    from scripts.run_training import load_experiment,preflight,validate_config
    report=json.loads(report_path.read_text())
    if (report.get('status')!='in_hand119_training_inputs_accepted'
            or report.get('episodes')!=contract.EPISODES or report.get('frames')!=contract.FRAMES
            or report.get('all_model_inputs_equal') is not True or report.get('original_sources_unchanged') is not True
            or report.get('model_samples_checked')!=3*contract.EPISODES
            or report.get('per_frame_prompts_checked')!=contract.FRAMES
            or report.get('loader_batches')!=32 or report.get('steady_batch_seconds',99)>=5.5
            or report.get('bundle')!=str(bundle.resolve()) or report.get('cache')!=str(cache.resolve())
            or report.get('bundle_sha256')!=digest(bundle/'bundle.json')
            or report.get('cache_manifest_sha256')!=digest(cache/'manifest.json')):
        raise ValueError('Full119 source/phase/input/cache acceptance is required')
    os.environ['PI05_IN_HAND119_BUNDLE']=str(bundle.resolve());os.environ['PI05_RGB_FRAME_CACHE']=str(cache.resolve())
    os.environ.pop('PI05_INPUT_EXECUTION_MIGRATION',None)
    config=load_experiment(MODULE);validate_config(config);inputs=preflight(config)
    from openpi.training import reproducibility
    if report.get('code_inventory')!=reproducibility._code_inventory(config):
        raise ValueError('Code changed after acceptance; rerun the input check')
    if (inputs['training_episodes']!=contract.EPISODES or inputs['training_frames']!=contract.FRAMES
            or inputs['norm_stats_sha256']!=report['norm']['norm_stats_sha256']):
        raise ValueError('Accepted inputs or norm changed')
    if config.resume or config.overwrite or config.checkpoint_dir.exists():
        raise ValueError('Fresh experiment required; never overwrite or implicitly resume')
    return config,inputs


def main():
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check','start'],nargs='?',default='check')
    p.add_argument('--bundle',type=Path,default=contract.DEFAULT_BUNDLE);p.add_argument('--cache',type=Path,default=contract.DEFAULT_CACHE)
    p.add_argument('--report',type=Path,default=REPORT);a=p.parse_args()
    config,inputs=checked_config(a.bundle,a.cache,a.report)
    selected=list(map(str,range(8)));devices=gpu_inventory()
    try:validate_gpus(devices,selected,fsdp_devices=8,batch_size=256);gpu_status='available_at_check_time'
    except (ValueError,RuntimeError) as error:gpu_status=str(error)
    print(json.dumps(dict(inputs=inputs,gpu_status=gpu_status,checkpoint_dir=str(config.checkpoint_dir),
        training_started=False),ensure_ascii=False),flush=True)
    if a.action=='check':return
    lock_path=ROOT/'.training-locks/normal93_gpus.lock';lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_gpus(gpu_inventory(),selected,fsdp_devices=8,batch_size=256)
        from scripts.training_environment import cuda_environment
        from scripts.training_ui import attach_cli_job
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
        console=ROOT/'logs'/f'in_hand119_b256_{stamp}.console.log'
        command=['bash',str(ROOT/'scripts/python.sh'),'-u','-m','scripts.run_training',MODULE,
            '--exp-name',config.exp_name,'--wandb-mode','offline']
        with console.open('xb') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=cuda_environment(selected),stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        import re
        for _ in range(180):
            if child.poll() is not None:raise RuntimeError('Training launch failed; see '+str(console))
            match=re.search(r'Run manifest: ([^\r\n]+)',console.read_text(errors='replace'))
            if match:
                job=attach_cli_job(Path(match.group(1)),console,child.pid)
                print(json.dumps(dict(pid=child.pid,console=str(console),ui_job_id=job['id'])),flush=True);break
            time.sleep(1)
        else:raise RuntimeError('Launch manifest pending; do not launch again')
        raise SystemExit(child.wait())


if __name__=='__main__':main()
