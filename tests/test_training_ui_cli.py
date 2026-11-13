"""Attach a running CLI experiment without starting, stopping or owning training."""
import json
from pathlib import Path

import pytest
from scripts import training_ui as ui


@pytest.fixture
def cli_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ui.common, 'repository_root', lambda: tmp_path)
    root=tmp_path/'logs';root.mkdir()
    manifest=root/'run.json';console=root/'console.log'
    record={'started_at':'20261113_104112_778551','status':'starting',
        'config':{'fields':{'exp_name':'current_b256','batch_size':256,'num_train_steps':16000}},
        'checkpoint_dir':str(tmp_path/'checkpoints/current_b256'),
        'experiment_source':{'original_path':'config.py'}}
    manifest.write_text(json.dumps(record));console.write_text('Run manifest: '+str(manifest)+'\nStep 10: loss=1.6\n')
    monkeypatch.setattr(ui,'process_identity',lambda pid:'identity1')
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda path:b'python\0-m\0scripts.run_training\0--exp-name\0current_b256\0' if str(path)=='/proc/123/cmdline' else original(path))
    return manifest,console,record


def test_attach_is_idempotent_and_reads_original_live_log(cli_run):
    manifest,console,_=cli_run
    state=ui.attach_cli_job(manifest,console,123)
    assert state['read_only'] and state['config_path']=='config.py'
    assert state['created_at']=='2026-11-13T10:41:12+08:00'
    assert ui.attach_cli_job(manifest,console,123)['id']==state['id']
    assert len(ui.jobs())==1 and ui.jobs()[0]['status']=='running'
    assert 'batch256' in ui.jobs()[0]['stage']
    link=ui.job_directory(state['id'])/'console.log'
    assert link.is_symlink() and link.resolve()==console
    with console.open('a') as stream:stream.write('Step 20: loss=1.5\n')
    assert 'Step 20' in link.read_text()


@pytest.mark.parametrize('status,expected',[('training_completed','completed'),('failed','failed')])
def test_external_completion_uses_real_manifest(cli_run,status,expected):
    manifest,console,record=cli_run;state=ui.attach_cli_job(manifest,console,123)
    record['status']=status;manifest.write_text(json.dumps(record))
    assert ui.read_job(state['id'])['status']==expected


def test_pid_reuse_or_exit_is_not_reported_running(cli_run,monkeypatch):
    manifest,console,_=cli_run;state=ui.attach_cli_job(manifest,console,123)
    monkeypatch.setattr(ui,'process_identity',lambda pid:'different')
    assert ui.read_job(state['id'])['status']=='interrupted'


def test_cli_attachment_cannot_stop_or_resume_training(cli_run):
    manifest,console,_=cli_run;state=ui.attach_cli_job(manifest,console,123)
    with pytest.raises(ValueError,match='仅供查看'):ui.stop_job(state['id'])
    with pytest.raises(ValueError,match='仅供查看'):ui.resume_job({'id':state['id']})
    assert not (ui.job_directory(state['id'])/'stop').exists()


def test_unrelated_console_is_rejected(cli_run):
    manifest,console,_=cli_run;console.write_text('unrelated training')
    with pytest.raises(ValueError,match='Console'):ui.attach_cli_job(manifest,console,123)
