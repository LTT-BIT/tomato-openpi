"""Check a cache migration, then explicitly hand off only our run after checkpoint4000."""
import argparse
from dataclasses import replace
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

OLD_PID = 575579


def checkpoint_ready(directory):
    import orbax.checkpoint as ocp
    path=Path(directory)/'4000'
    if not path.is_dir():return False
    with ocp.CheckpointManager(str(directory),options=ocp.CheckpointManagerOptions(read_only=True)) as manager:
        latest=manager.latest_step()
        if latest is not None and latest>4000:
            raise ValueError('A later checkpoint exists; do not roll back this run automatically')
        if latest!=4000:return False
    metadata=path/'_CHECKPOINT_METADATA'
    if not metadata.is_file():return False
    value=json.loads(metadata.read_text())
    if not value.get('commit_timestamp_nsecs'):return False
    return all((path/name/'_METADATA').is_file() for name in ['train_state','params'])


def validate_process(pid,identity,experiment):
    from scripts.training_ui import process_identity
    if process_identity(pid)!=identity:raise ValueError('Original training PID changed or exited')
    command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if b'scripts.run_training' not in command or experiment.encode() not in command:
        raise ValueError('Unexpected original training command')


def check(report_path):
    from scripts.tomato100_run import configure_inputs,DEFAULT_BUNDLE,PREPARATION,MODULES
    from scripts.run_training import load_experiment,preflight
    from openpi.training import reproducibility as rep
    from openpi.training.input_migration import validate_migration
    from openpi.training.data_loader import create_torch_dataset
    from scripts.training_ui import process_identity
    report=json.loads(report_path.read_text())
    configure_inputs(DEFAULT_BUNDLE,PREPARATION)
    os.environ['PI05_RGB_FRAME_CACHE']=report['cache_root']
    os.environ['PI05_INPUT_EXECUTION_MIGRATION']=str(report_path.resolve())
    config=replace(load_experiment(MODULES['b']),resume=True)
    preflight(config)
    data=config.data.create(config.assets_dirs,config.model)
    source=create_torch_dataset(data,config.model.action_horizon,config.model,sampling_mode=config.sampling_mode)
    original=json.loads((Path(config.checkpoint_dir)/'reproducibility/manifest.json').read_text())['contract']
    current={**report['new_contract'],'code':rep._code_inventory(config),
        'training':rep._training_contract(config,data)}
    validate_migration(original,current,report_path)
    if len(source)!=425556 or str(config.checkpoint_dir)!=report['checkpoint_dir']:
        raise ValueError('Cache handoff dataset or checkpoint identity changed')
    ready=checkpoint_ready(config.checkpoint_dir)
    return config,{'status':'ready' if ready else 'waiting_checkpoint4000',
        'old_pid':OLD_PID,'old_identity':report['old_identity'],
        'checkpoint_dir':str(config.checkpoint_dir),'cache_root':report['cache_root']}


def stop_signal_from_status(text):
    masks={line.split(':',1)[0]:int(line.split(':',1)[1].strip(),16)
        for line in text.splitlines() if line.startswith(('SigIgn:', 'SigCgt:'))}
    ignored=masks['SigIgn']
    if not ignored & (1 << (signal.SIGINT-1)):
        return signal.SIGINT
    # nohup may pass SIGINT=SIG_IGN to Python; SIGTERM remains a normal exit signal.
    if ignored & (1 << (signal.SIGTERM-1)):
        raise ValueError('Both normal stop signals are ignored; do not force kill')
    return signal.SIGTERM


def choose_stop_signal(pid):
    return stop_signal_from_status(Path(f'/proc/{pid}/status').read_text())


def handoff(config,status,report_path,*,recover_stopping=False):
    from scripts.tomato100_run import ROOT,MODULES
    from scripts.tomato_ab_run import gpu_inventory,validate_gpus
    from scripts.training_environment import cuda_environment
    from scripts.training_ui import atomic_json,attach_cli_job,process_identity
    if status['status']!='ready':
        print(json.dumps(status),flush=True);return
    directory=ROOT/'logs/complete100_preflight'
    receipt=directory/'rgb_cache_handoff_4000.json'
    lock_path=ROOT/'.training-locks/rgb_cache_handoff.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            if not recover_stopping:
                print(receipt.read_text(),flush=True);return
            if (previous.get('phase')!='stopping_original_after_finalized4000'
                    or previous.get('new_pid') is not None
                    or previous.get('old_identity')!=status['old_identity']
                    or previous.get('report')!=str(report_path)
                    or process_identity(previous['handoff_pid']) is not None):
                raise ValueError('Cannot recover an active, changed or already launched handoff')
        elif recover_stopping:
            raise ValueError('No stopped handoff to recover')
        validate_process(OLD_PID,status['old_identity'],config.exp_name)
        if not checkpoint_ready(config.checkpoint_dir):raise ValueError('Checkpoint4000 not finalized')
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        pids={int(value.strip()) for value in gpu_processes.splitlines() if value.strip()}
        if pids!={OLD_PID}:raise ValueError('Unexpected GPU processes; do not interrupt them')
        state={**status,'phase':'stopping_original_after_finalized4000','report':str(report_path),
            'handoff_pid':os.getpid(),'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
        stop_signal=choose_stop_signal(OLD_PID)
        state['stop_signal']=signal.Signals(stop_signal).name
        if previous is not None:
            state['previous_stop_attempt']=previous
        atomic_json(receipt,state)
        os.kill(OLD_PID,stop_signal)
        for _ in range(60):
            if process_identity(OLD_PID)!=status['old_identity']:break
            time.sleep(1)
        else:raise RuntimeError('Original process did not stop; no SIGKILL and no second training launch')
        for path in (ROOT/'logs/training_ui').glob('*/state.json'):
            old_job=json.loads(path.read_text())
            if old_job.get('child_pid')==OLD_PID and old_job.get('read_only'):
                old_job['external_original_manifest']=old_job.pop('external_manifest',None)
                old_job.update(status='stopped',stage='已从4000步切换缓存读取',handoff_receipt=str(receipt))
                atomic_json(path,old_job)
        selected=[str(i) for i in range(8)]
        for _ in range(30):
            try:
                validate_gpus(gpu_inventory(),selected,fsdp_devices=8,batch_size=256)
                break
            except RuntimeError:time.sleep(1)
        else:raise RuntimeError('GPUs not released; cached training was not launched')
        gpu_lock=(ROOT/'.training-locks/normal93_gpus.lock').open('a')
        for _ in range(20):
            try:
                fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.5)
        else:raise RuntimeError('Another launcher holds the GPU lock; no new training launched')
        validate_gpus(gpu_inventory(),selected,fsdp_devices=8,batch_size=256)
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
        console=ROOT/'logs'/f'complete100_b256_cached_resume_{stamp}.console.log'
        env=cuda_environment(selected)
        command=['bash',str(ROOT/'scripts/python.sh'),'-u','-m','scripts.run_training',MODULES['b'],
            '--exp-name',config.exp_name,'--resume','--wandb-mode','offline']
        with console.open('xb') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='cached_resume_started',new_pid=child.pid,console=str(console))
        atomic_json(receipt,state)
        for _ in range(180):
            if child.poll() is not None:
                state.update(phase='cached_resume_failed',returncode=child.returncode)
                atomic_json(receipt,state);raise RuntimeError('Cached resume failed; checkpoint4000 is preserved')
            text=console.read_text(errors='replace')
            import re
            match=re.search(r'Run manifest: ([^\r\n]+)',text)
            if match:
                manifest=Path(match.group(1))
                job=attach_cli_job(manifest,console,child.pid)
                state.update(ui_job_id=job['id'],manifest=str(manifest))
                atomic_json(receipt,state)
                break
            time.sleep(1)
        else:raise RuntimeError('Resume process is alive but its launch manifest is not ready; do not relaunch')
        print(json.dumps(state,ensure_ascii=False),flush=True)
        # Keep the supervisor alive; this also provides a clear final return code.
        result=child.wait()
        state.update(phase='training_completed' if result==0 else 'cached_resume_failed',returncode=result)
        atomic_json(receipt,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','handoff','recover-stopping'],default='check',nargs='?')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    config,status=check(args.report)
    if args.action=='check':print(json.dumps(status),flush=True)
    else:handoff(config,status,args.report,recover_stopping=args.action=='recover-stopping')


if __name__=='__main__':main()
