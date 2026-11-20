"""Exercise the deferred supervisor with fake jobs/GPUs and no real training process."""
import json
from types import SimpleNamespace
import pytest

from scripts import queue_in_hand204_three_rgb as q
from scripts import tomato_ab_run
from openpi.subtasks.prepare import write_json,digest


@pytest.mark.parametrize('case',['success','cancel','predecessor_failed','busy'])
def test_queue_worker_dependency_and_cancellation(tmp_path,monkeypatch,case):
    monkeypatch.setattr(q.common,'repository_root',lambda:tmp_path)
    monkeypatch.setattr(q.signal,'signal',lambda *args:None)
    monkeypatch.setattr(q,'original_code_unchanged',lambda *args:None)
    gpus=['GPU-'+str(i) for i in range(8)]
    old_id='a'*32;new_id='b'*32
    old=q.job_directory(old_id);new=q.job_directory(new_id);new.mkdir(parents=True)
    checkpoint=tmp_path/'crop_checkpoint'
    write_json(checkpoint/'20000/_CHECKPOINT_METADATA',dict(commit_timestamp_nsecs=1))
    for name in ['params','train_state']:write_json(checkpoint/'20000'/name/'_METADATA',{})
    (checkpoint/'20000/assets').mkdir()
    binding=dict(id=old_id,host='test',exp_name='crop',checkpoint_dir=str(checkpoint),child_pid=123,child_start='old-start',gpus=gpus)
    write_json(old/'state.json',{**binding,'status':'failed' if case=='predecessor_failed' else 'completed'})
    cfg_file=tmp_path/'config.py';cfg_file.write_text('frozen')
    report=tmp_path/'accepted.json';write_json(report,{})
    state=dict(id=new_id,status='queued',queue_dependency=binding,gpus=gpus,
        acceptance_sha256=digest(report),config_path=str(cfg_file),config_sha256=digest(cfg_file),
        predecessor_code_root=str(tmp_path),predecessor_code_report=str(report),predecessor_code_report_sha256=digest(report))
    write_json(new/'state.json',state);(new/'ready').touch()
    if case=='cancel':(new/'stop').touch()
    cfg=SimpleNamespace(exp_name='three',policy_metadata={'rgb_frame_cache':'/three-camera-cache'})
    monkeypatch.setattr(q,'config_ready',lambda:(cfg,report,digest(report)))
    monkeypatch.setattr(q,'process_identity',lambda pid:None if pid==123 else 'new-start')
    devices=[dict(index=str(i),uuid=gpus[i],used_mib=2000 if case=='busy' else 0,
        utilization=0,compute_pids=[]) for i in range(8)]
    monkeypatch.setattr(tomato_ab_run,'gpu_inventory',lambda:devices)
    calls=[]
    class Child:
        pid=456
        returncode=0
        def poll(self):return 0
    def popen(command,**kwargs):
        calls.append((command,kwargs));return Child()
    monkeypatch.setattr(q.subprocess,'Popen',popen)
    def sleep(_):(new/'stop').touch()
    monkeypatch.setattr(q.time,'sleep',sleep)
    before=(old/'state.json').read_bytes()
    q.worker(new_id)
    after=json.loads((new/'state.json').read_text())
    assert (old/'state.json').read_bytes()==before
    if case=='success':
        assert after['status']=='completed' and len(calls)==1
        command,kw=calls[0]
        assert q.MODULE in command and '--wandb-mode' in command and command[-1]=='offline'
        assert kw['env']['PI05_RGB_FRAME_CACHE']=='/three-camera-cache'
        assert kw['env']['JAX_PLATFORMS']=='cuda'
        assert kw['env']['CUDA_VISIBLE_DEVICES']==','.join(gpus)
    else:
        assert not calls
        assert after['status']==('failed' if case=='predecessor_failed' else 'stopped')


def test_audit_releases_episode_maps_and_file_descriptors(tmp_path):
    import os
    import numpy as np
    from scripts.check_in_hand204_three_rgb import release_frame_maps
    files=[]
    for i in range(12):
        p=tmp_path/f'{i}.npy';np.save(p,np.zeros((2,3),dtype=np.uint8));files.append(p)
    a=SimpleNamespace(_cache_maps={i:np.load(p,mmap_mode='r') for i,p in enumerate(files[:6])})
    b=SimpleNamespace(_cache_maps={i:np.load(p,mmap_mode='r') for i,p in enumerate(files[6:])})
    count=lambda:len(os.listdir('/proc/self/fd'))
    before=count();release_frame_maps(a,b)
    assert not a._cache_maps and not b._cache_maps
    assert count()<=before-12
