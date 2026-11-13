"""Reject changed offline data and keep the new launcher from starting implicitly."""
import json
from pathlib import Path
import sys

import pytest
from scripts import tomato100_run as run
from training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop import offline_preparation

RECORDS = Path(__file__).resolve().parents[1] / 'training_configs/experiments/tomato_sharpa/subtasks/records'


def preparation(tmp_path):
    # The real frozen metadata is small and included in the prior loader validation.
    validation = json.loads((RECORDS/'complete100_offline_local_20261112_validation_r11.json').read_text())
    report = json.loads(run.PREPARATION.read_text())
    return validation, report


def test_candidate_remains_closed_without_explicit_preparation(tmp_path):
    assert not offline_preparation(tmp_path, {}, 'any', None)


@pytest.mark.parametrize('change', [None, 'hash', 'selection', 'transfer', 'future'])
def test_only_exact_verified_offline_candidate_is_accepted(tmp_path, change):
    _, report = preparation(tmp_path)
    fixture = Path(__file__).parent/'fixtures/tomato_r11_crop_manifest.json'
    manifest = json.loads(fixture.read_text())
    (tmp_path/'crop_manifest.json').write_bytes(fixture.read_bytes())
    selection = report['selection_sha256']
    if change == 'hash':report['crop_manifest_sha256']='bad'
    if change == 'selection':report['selection_sha256']='bad'
    if change == 'transfer':report['transfer']['passed']=False
    if change == 'future':manifest['roi_generation_mode']='causal_prediction'
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    if change:
        with pytest.raises(ValueError):offline_preparation(tmp_path,manifest,selection,path)
    else:
        assert offline_preparation(tmp_path,manifest,selection,path)


def test_start_busy_stops_before_loading_or_launching(monkeypatch):
    inventory=[dict(index=str(i),used_mib=69000,total_mib=73000,utilization=100,compute_pids=[]) for i in range(8)]
    monkeypatch.setattr(run,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(sys,'argv',['tomato100_run','start'])
    monkeypatch.setattr(run,'cpu_preflight',lambda *a:pytest.fail('must stop before loading'))
    monkeypatch.setattr(run.subprocess,'call',lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(RuntimeError,match='not started'):run.main()


def test_default_only_checks_b(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(sys,'argv',['tomato100_run'])
    monkeypatch.setattr(run,'ROOT',tmp_path)
    monkeypatch.setattr(run,'gpu_inventory',lambda:[dict(index=str(i),used_mib=0,utilization=0,compute_pids=[]) for i in range(8)])
    variants=[]
    def preflight(variant,exp):
        variants.append(variant)
        return SimpleNamespace(fsdp_devices=8,batch_size=256),{'variant':variant}
    monkeypatch.setattr(run,'cpu_preflight',preflight)
    monkeypatch.setattr(run.subprocess,'call',lambda *a,**k:pytest.fail('must not launch'))
    run.main()
    assert variants==['b']
    assert len(list((tmp_path/'logs/complete100_preflight').glob('*.json')))==1


def test_actual_all100_recipe_saves_completed_steps_and_keeps_each_checkpoint(monkeypatch):
    from training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop import build_config
    from scripts.train import should_save_checkpoint
    bundle=Path('/home/user/datasets/derived/tomato_complete100_four_view_offline_local_candidate_r11')
    if not bundle.exists():pytest.skip('Actual prepared all100 data is unavailable')
    monkeypatch.setenv('PI05_TOMATO100_CROP_BUNDLE',str(bundle))
    monkeypatch.setenv('PI05_TOMATO100_CROP_A_BUNDLE',str(bundle))
    monkeypatch.setenv('PI05_TOMATO100_PREPARATION_REPORT',str(run.PREPARATION))
    for variant in ['a_goal','b_coarse']:
        config=build_config(variant=variant)
        assert (config.batch_size,config.num_train_steps,config.fsdp_devices)==(256,16000,8)
        assert config.save_interval==config.keep_period==4000
        assert [step+1 for step in range(16000) if should_save_checkpoint(config,step,0)]==[4000,8000,12000,16000]
        assert config.batch_size*config.num_train_steps==512*8000
        assert config.lr_schedule.warmup_steps==1000 and config.lr_schedule.decay_steps==30000
