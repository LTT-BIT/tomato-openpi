"""Durable dependency queue: wait for our successful crop run, then launch the three-RGB control."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
import uuid
from zoneinfo import ZoneInfo

from training_configs import common
from training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_three_rgb import MODULE
from scripts.training_ui import atomic_json,process_identity,job_directory,now,submit_lock


def config_ready():
    from openpi.subtasks.prepare import digest
    from openpi.training import reproducibility
    from scripts.run_training import load_experiment,preflight,validate_config
    config=load_experiment(MODULE);validate_config(config)
    report_path=Path(config.policy_metadata['three_view_acceptance_report'])
    report=json.loads(report_path.read_text());inputs=preflight(config)
    if (report.get('status')!='three_rgb204_accepted' or report.get('episodes')!=204 or report.get('frames')!=588734
            or report.get('model_samples_checked')!=612 or report.get('phase_boundary_samples_checked')!=2040
            or not all(report.get(k) is True for k in ['all_original_pixels_equal','only_removed_interaction','normalization_values_equal'])
            or report.get('loader_batches')!=32 or report.get('steady_batch_seconds',99)>=5.5
            or report.get('code_inventory')!=reproducibility._code_inventory(config)
            or report.get('inputs',{}).get('norm_stats_sha256')!=inputs['norm_stats_sha256']
            or report.get('inputs',{}).get('norm_audit_sha256')!=inputs['norm_audit_sha256']
            or report.get('view_sha256')!=digest(Path(config.policy_metadata['dataset_root'])/'view.json')
            or report.get('cache_manifest_sha256')!=digest(Path(config.policy_metadata['rgb_frame_cache'])/'manifest.json')):
        raise ValueError('Three-camera acceptance/data/code/norm binding failed')
    if config.resume or config.overwrite or config.checkpoint_dir.exists():raise ValueError('Fresh three-camera run required')
    return config,report_path,digest(report_path)


def gpu_indices(devices, uuids):
    if len(devices)!=8 or {d['uuid'] for d in devices}!=set(uuids):
        raise ValueError('GPU UUID set changed')
    return [d['index'] for d in devices if d['uuid'] in uuids]


def original_code_unchanged(shared_root, report_path, expected_report_sha):
    from openpi.subtasks.prepare import digest
    if digest(report_path)!=expected_report_sha:raise ValueError('Original acceptance report changed')
    report=json.loads(Path(report_path).read_text())
    for relative,sha in report['code_inventory'].items():
        if digest(Path(shared_root)/relative)!=sha:raise ValueError('Running crop code changed: '+relative)


def predecessor_ready(state,binding,identity):
    for key in ['id','host','exp_name','checkpoint_dir','child_pid','child_start','gpus']:
        if state.get(key)!=binding.get(key):raise ValueError('Predecessor identity changed: '+key)
    if state['status'] in ['failed','stopped','interrupted','orphaned']:
        raise ValueError('Predecessor did not complete successfully; queued run will not start')
    if state['status']!='completed' or identity==binding['child_start']:return False
    root=Path(binding['checkpoint_dir'])/'20000'
    p=root/'_CHECKPOINT_METADATA'
    if not p.is_file():return False
    if not json.loads(p.read_text()).get('commit_timestamp_nsecs'):return False
    return all((root/name/'_METADATA').is_file() for name in ['params','train_state']) and (root/'assets').is_dir()


def enqueue(predecessor):
    from openpi.training.reproducibility import serialize_config
    from openpi.subtasks.prepare import digest
    config,report,report_sha=config_ready()
    old=json.loads((job_directory(predecessor)/'state.json').read_text())
    if old.get('host')!=socket.gethostname() or old.get('status')!='running':raise ValueError('Expected our running local crop UI job')
    command=Path(f"/proc/{old['child_pid']}/cmdline").read_bytes().split(b'\0')
    if (process_identity(old['child_pid'])!=old['child_start'] or b'scripts.run_training' not in command
            or old['exp_name'].encode() not in command or old['values']['num_train_steps']!=20000):
        raise ValueError('Predecessor PID/start/command/steps differs')
    root=common.repository_root();shared_root=(root/'logs').resolve().parent
    code_report=shared_root/'logs/in_hand204_preflight/accepted_20261120_s20000_c5000.json'
    code_report_sha=digest(code_report);original_code_unchanged(shared_root,code_report,code_report_sha)
    job_id=uuid.uuid5(uuid.NAMESPACE_URL,str(config.checkpoint_dir)).hex
    directory=job_directory(job_id)
    with submit_lock():
        if directory.exists():raise FileExistsError('This control is already queued; never enqueue twice')
        directory.mkdir()
        binding={k:old[k] for k in ['id','host','exp_name','checkpoint_dir','child_pid','child_start','gpus']}
        path=root/'training_configs/experiments/tomato_sharpa/subtasks/in_hand204_b_three_rgb.py'
        state=dict(id=job_id,host=socket.gethostname(),status='queued',stage='等待四路训练完成20000步并释放8卡',
            created_at=now(),updated_at=now(),exp_name=config.exp_name,checkpoint_dir=str(config.checkpoint_dir),
            module=MODULE,config_path=str(path),resolved_config=json.loads(serialize_config(config)),
            gpus=old['gpus'],wandb_mode='offline',resume=False,queue_dependency=binding,
            acceptance_report=str(report),acceptance_sha256=report_sha,runtime_root=str(root),
            config_sha256=digest(path),read_only=False,predecessor_code_root=str(shared_root),
            predecessor_code_report=str(code_report),predecessor_code_report_sha256=code_report_sha)
        atomic_json(directory/'state.json',state)
        with (directory/'console.log').open('ab') as stream:
            child=subprocess.Popen(['bash',str(root/'scripts/python.sh'),'-u','-m',__name__ if __name__!='__main__' else 'scripts.queue_in_hand204_three_rgb','worker',job_id],
                cwd=root,env={**os.environ,'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':''},
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(worker_pid=child.pid,worker_start=process_identity(child.pid))
        if not state['worker_start']:raise RuntimeError('Queue supervisor did not start')
        atomic_json(directory/'state.json',state);(directory/'ready').touch()
    print(json.dumps(dict(job_id=job_id,status='queued',predecessor=predecessor,training_started=False)))


def worker(job_id):
    from openpi.subtasks.prepare import digest
    from scripts.tomato_ab_run import gpu_inventory,validate_gpus
    from scripts.training_environment import cuda_environment
    directory=job_directory(job_id)
    for _ in range(100):
        if (directory/'ready').exists():break
        time.sleep(.1)
    else:raise RuntimeError('Queue registration handshake failed')
    state=json.loads((directory/'state.json').read_text());child=None
    def save(status,stage,**extra):
        state.update(status=status,stage=stage,updated_at=now(),**extra);atomic_json(directory/'state.json',state)
    def stop(*_):(directory/'stop').touch()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with (directory/'supervisor.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            config,report,sha=config_ready()
            if sha!=state['acceptance_sha256'] or digest(state['config_path'])!=state['config_sha256']:
                raise ValueError('Queued acceptance/config changed')
            while True:
                if (directory/'stop').exists():save('stopped','排队已取消；未启动三路训练');return
                old=json.loads((job_directory(state['queue_dependency']['id'])/'state.json').read_text())
                binding=state['queue_dependency']
                if predecessor_ready(old,binding,process_identity(binding['child_pid'])):
                    devices=gpu_inventory()
                    selected=gpu_indices(devices,state['gpus'])
                    try:validate_gpus(devices,selected,fsdp_devices=8,batch_size=256)
                    except (ValueError,RuntimeError):time.sleep(30);continue
                    break
                time.sleep(30)
            config,report,sha=config_ready()
            original_code_unchanged(state['predecessor_code_root'],state['predecessor_code_report'],state['predecessor_code_report_sha256'])
            if sha!=state['acceptance_sha256']:raise ValueError('Acceptance changed while queued')
            # Shared submit lock prevents this dashboard from allocating the same GPUs concurrently.
            with submit_lock():
                if (directory/'stop').exists():save('stopped','排队已取消');return
                devices=gpu_inventory();validate_gpus(devices,gpu_indices(devices,state['gpus']),fsdp_devices=8,batch_size=256)
                env=cuda_environment(state['gpus'])
                env['PI05_RGB_FRAME_CACHE']=config.policy_metadata['rgb_frame_cache']
                env.pop('PI05_INPUT_EXECUTION_MIGRATION',None)
                command=['bash',str(common.repository_root()/'scripts/python.sh'),'-u','-m','scripts.run_training',MODULE,
                    '--exp-name',config.exp_name,'--wandb-mode','offline']
                child=subprocess.Popen(command,cwd=common.repository_root(),env=env,stdin=subprocess.DEVNULL,start_new_session=True)
                save('running','三路RGB＋subtask训练中',child_pid=child.pid,child_start=process_identity(child.pid),training_started_at=now())
            while child.poll() is None:
                if (directory/'stop').exists():
                    os.killpg(child.pid,signal.SIGTERM);child.wait();save('stopped','三路训练已停止',returncode=child.returncode);return
                time.sleep(1)
            if child.returncode:raise RuntimeError('Three-view training failed: '+str(child.returncode))
            save('completed','三路RGB＋subtask训练完成',returncode=0)
    except Exception as error:
        traceback.print_exc()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM);child.wait()
        save('failed','三路队列/训练失败',error=str(error),read_only=True)


if __name__=='__main__':
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['enqueue','worker']);p.add_argument('job_id');a=p.parse_args()
    if a.action=='enqueue':enqueue(a.job_id)
    else:worker(a.job_id)
