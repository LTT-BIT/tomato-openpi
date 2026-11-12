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
        return SimpleNamespace(fsdp_devices=8,batch_size=512),{'variant':variant}
    monkeypatch.setattr(run,'cpu_preflight',preflight)
    monkeypatch.setattr(run.subprocess,'call',lambda *a,**k:pytest.fail('must not launch'))
    run.main()
    assert variants==['b']
    assert len(list((tmp_path/'logs/complete100_preflight').glob('*.json')))==1
