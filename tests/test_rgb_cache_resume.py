"""Validate exact cached pixels, immutable resume inputs and the checkpoint4000 barrier."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from openpi.training.input_migration import check_code_change,fingerprint,validate_migration


def contracts():
    old={'training':{'batch':256},'dataset_sha256':'data','norm_sha256':'norm',
         'plan_indices_sha256':'plan','code':{'src/openpi/training/data_loader.py':'old','src/openpi/models/pi0.py':'model'}}
    new={**old,'code':{**old['code'],'src/openpi/training/data_loader.py':'new','src/openpi/training/frame_cache.py':'cache'}}
    return old,new


def test_only_reader_code_can_change_during_input_migration(tmp_path):
    old,new=contracts();changed=check_code_change(old,new)
    cache=tmp_path/'cache';cache.mkdir();manifest=cache/'manifest.json';manifest.write_text('{}')
    from openpi.training.frame_cache import sha256
    report={'status':'exact_input_acceleration_validated','old_contract_sha256':fingerprint(old),
        'new_contract_sha256':fingerprint(new),'changed_code_files':changed,'checkpoint_step':4000,
        'cache_manifest_sha256':sha256(manifest),'validation':{'all_model_inputs_equal':True,
            'episodes_checked':100,'samples_checked':300,'loader_batches':32,'steady_batch_seconds':1.2}}
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    assert validate_migration(old,new,path,cache_path=cache)==report
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='manifest changed'):validate_migration(old,new,path,cache_path=cache)


@pytest.mark.parametrize('part',['training','dataset_sha256','norm_sha256','plan_indices_sha256'])
def test_training_data_and_sampler_cannot_be_changed(part):
    old,new=contracts();new[part]='changed'
    with pytest.raises(ValueError,match='cannot change'):check_code_change(old,new)


def test_model_code_change_cannot_be_hidden_as_reader_migration():
    old,new=contracts();new['code']['src/openpi/models/pi0.py']='changed'
    with pytest.raises(ValueError,match='Unapproved code'):check_code_change(old,new)


def test_real_video_cache_matches_decoder_and_preserves_timestamp(tmp_path):
    import av
    from scripts.build_rgb_frame_cache import build_video
    from openpi.training.frame_cache import CachedLeRobotDataset
    source=tmp_path/'source.mp4'
    with av.open(str(source),'w') as output:
        stream=output.add_stream('libx264',rate=30);stream.width=20;stream.height=16;stream.pix_fmt='yuv420p'
        for i in range(12):
            frame=av.VideoFrame.from_ndarray(np.full((16,20,3),[i*10,80,140],np.uint8),format='rgb24')
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    cache=tmp_path/'cache';destination=cache/'cam0/episode_000000.npy'
    entry=build_video((str(source),str(destination),0,'cam0',12,30))
    assert entry['exact_reference_samples']==5 and not destination.stat().st_mode & 0o222
    reader=object.__new__(CachedLeRobotDataset);reader.meta=SimpleNamespace(fps=30)
    reader.cache_root=cache;reader.cache_entries={(0,'cam0'):entry};reader._cache_maps={};reader.tolerance_s=1e-4
    value=reader._query_videos({'cam0':[float(np.float32(5/30))]},0)['cam0']
    np.testing.assert_array_equal(value,np.load(destination,mmap_mode='r')[5])
    assert value.shape==(224,224,3)
    assert reader.__getstate__()['_cache_maps']=={}
    with pytest.raises(ValueError,match='current image only'):reader._query_videos({'cam0':[0,.1]},0)
    with pytest.raises(ValueError,match='timeline'):reader._query_videos({'cam0':[-.1]},0)


def test_actual_orbax_checkpoint4000_is_finalized_and_restores_optimizer(tmp_path):
    import jax.numpy as jnp
    from openpi.training.checkpoints import initialize_checkpoint_dir
    from scripts.tomato100_cached_resume import checkpoint_ready
    directory=tmp_path/'run';manager,_=initialize_checkpoint_dir(directory,keep_period=4000,overwrite=False,resume=False)
    state={'step':jnp.array(4000),'optimizer':{'count':jnp.array(4000),'moment':jnp.array([1.,2.])}}
    assert not checkpoint_ready(directory)
    manager.save(4000,{'assets':lambda path:None,'train_state':state,'params':{'params':{'w':jnp.array(3.)}}})
    manager.wait_until_finished();manager.close()
    assert checkpoint_ready(directory)
    manager,resuming=initialize_checkpoint_dir(directory,keep_period=4000,overwrite=False,resume=True)
    try:
        restored=manager.restore(4000,items={'train_state':state,'params':{'params':{'w':jnp.array(0.)}}})
        assert resuming and int(restored['train_state']['step'])==4000
        np.testing.assert_array_equal(restored['train_state']['optimizer']['moment'],[1.,2.])
    finally:manager.close()


def test_handoff_never_stops_before_finalized_checkpoint(tmp_path,monkeypatch):
    from scripts import tomato100_cached_resume as handoff
    from scripts import tomato100_run
    monkeypatch.setattr(tomato100_run,'ROOT',tmp_path)
    monkeypatch.setattr(handoff,'checkpoint_ready',lambda path:False)
    monkeypatch.setattr(handoff,'validate_process',lambda *args:None)
    monkeypatch.setattr(handoff.os,'kill',lambda *args:pytest.fail('must not signal training'))
    with pytest.raises(ValueError,match='not finalized'):
        handoff.handoff(SimpleNamespace(checkpoint_dir=tmp_path/'run',exp_name='run'),
            {'status':'ready','old_identity':'expected'},tmp_path/'report.json')


def test_handoff_launches_one_resume_and_preserves_experiment(tmp_path,monkeypatch):
    from scripts import tomato100_cached_resume as handoff
    from scripts import tomato100_run,tomato_ab_run,training_ui
    import signal
    monkeypatch.setattr(tomato100_run,'ROOT',tmp_path)
    (tmp_path/'logs/complete100_preflight').mkdir(parents=True)
    monkeypatch.setattr(handoff,'checkpoint_ready',lambda path:True)
    monkeypatch.setattr(handoff,'validate_process',lambda *args:None)
    monkeypatch.setattr(training_ui,'process_identity',lambda pid:None)
    monkeypatch.setattr(handoff.subprocess,'check_output',lambda *a,**k:str(handoff.OLD_PID)+'\n')
    monkeypatch.setattr(tomato_ab_run,'gpu_inventory',lambda:[])
    monkeypatch.setattr(tomato_ab_run,'validate_gpus',lambda *a,**k:None)
    monkeypatch.setattr(training_ui,'attach_cli_job',lambda *a:{'id':'cachedjob'})
    signals=[];calls=[]
    monkeypatch.setattr(handoff.os,'kill',lambda *args:signals.append(args))
    class Child:
        pid=1234
        returncode=0
        def poll(self):return None
        def wait(self):return 0
    def launch(command,**kwargs):
        calls.append(command)
        kwargs['stdout'].write(b'Run manifest: /tmp/new-run.json\n');kwargs['stdout'].flush()
        return Child()
    monkeypatch.setattr(handoff.subprocess,'Popen',launch)
    config=SimpleNamespace(checkpoint_dir=tmp_path/'checkpoint',exp_name='same_experiment')
    status={'status':'ready','old_identity':'pinnedidentity'}
    handoff.handoff(config,status,tmp_path/'report.json')
    assert signals==[(handoff.OLD_PID,signal.SIGINT)]
    assert len(calls)==1 and '--resume' in calls[0]
    assert calls[0][calls[0].index('--exp-name')+1]=='same_experiment'
    handoff.handoff(config,status,tmp_path/'report.json')
    assert len(calls)==1 and len(signals)==1
