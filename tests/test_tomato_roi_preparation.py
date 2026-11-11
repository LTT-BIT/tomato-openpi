import copy
import json
from pathlib import Path

import pytest

from scripts.prepare_tomato100_roi import worklist
from openpi.subtasks.rgb100 import derived_output_path, validate_source_plan


RECORDS=Path(__file__).resolve().parents[1]/'training_configs/experiments/tomato_sharpa/subtasks/records'


def test_real_worklist_does_not_fabricate_boxes_or_claim_full_coverage():
    s=json.loads((RECORDS/'complete100_rgb_crop_20261111_saved_labels_r1.json').read_text())
    p=json.loads((RECORDS/'normal93_20261110_interaction_preview_annotations.json').read_text())
    r=worklist(s,p,preview_sha256='test')
    assert len(r['episodes'])==100
    assert r['total_frames']==425556 and r['offline_preview_frames']==258 and r['remaining_frames']==425298
    assert sum(bool(e['roi_clips']) for e in r['episodes'])==1
    assert not r['training_ready'] and not r['source_writes']
    p=copy.deepcopy(p);p['source_sha256']['cam0']='changed'
    with pytest.raises(ValueError):worklist(s,p,preview_sha256='test')


def test_output_cannot_touch_source_or_symlink_to_source(tmp_path):
    source=tmp_path/'original';source.mkdir()
    s={'episodes':[{'dataset':str(source)}]}
    for path in (source,source/'crops',tmp_path):
        with pytest.raises(ValueError):derived_output_path(s,path)
    link=tmp_path/'alias';link.symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError):derived_output_path(s,link/'crops')
    assert derived_output_path(s,tmp_path/'derived')==tmp_path/'derived'
    (tmp_path/'derived').mkdir()
    with pytest.raises(FileExistsError):derived_output_path(s,tmp_path/'derived')
    collection=tmp_path/'raw_collection';(collection/'A'/'recording').mkdir(parents=True)
    with pytest.raises(ValueError):derived_output_path({'episodes':[{'dataset':str(collection/'A'/'recording')}]},collection/'new_crops')


def test_changed_plan_identity_is_rejected_before_preparation():
    s=json.loads((RECORDS/'complete100_rgb_crop_20261111_saved_labels_r1.json').read_text())
    p=json.loads((RECORDS/'complete100_rgb_crop_20261111_source_plan_r1.json').read_text())
    p['episodes'].reverse()
    with pytest.raises(ValueError):validate_source_plan(s,p,selection_sha256=p['selection_sha256'])


def test_full_training_recipe_rejects_base_or_missing_roi(monkeypatch,tmp_path):
    from training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop import build_config
    monkeypatch.delenv('PI05_TOMATO100_CROP_BUNDLE',raising=False)
    with pytest.raises(ValueError,match='full100'):build_config()
    monkeypatch.setenv('PI05_TOMATO100_CROP_BUNDLE',str(tmp_path))
    with pytest.raises(ValueError,match='crop_manifest'):build_config()
    (tmp_path/'crop_manifest.json').write_text(json.dumps({'full_roi_coverage':False}))
    with pytest.raises(ValueError,match='coverage'):build_config()


def test_metadata_only_recipe_uses_four_keys_all100_and_original_hyperparameters(monkeypatch,tmp_path):
    from training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop import build_config
    from openpi.subtasks.rgb100 import digest,IMAGE_KEYS
    from openpi.subtasks.tomato import PROMPTS
    # Synthetic metadata tests configuration wiring, never real dataset readiness.
    (tmp_path/'crop_manifest.json').write_text(json.dumps(dict(
        selection_sha256=digest(RECORDS/'complete100_rgb_crop_20261111_saved_labels_r1.json'),
        full_roi_coverage=True,episodes=100,frames=425556,
        roi_generation_mode='manual_offline_interpolated',roi_generator_signature='synthetic-test')))
    (tmp_path/'splits.json').write_text(json.dumps({'splits':{'train':list(range(100))}}))
    meta=tmp_path/'b_coarse/meta';meta.mkdir(parents=True)
    (meta/'tasks.jsonl').write_text(''.join(json.dumps({'task':p})+'\n' for p in dict.fromkeys(PROMPTS)))
    (meta/'info.json').write_text(json.dumps(dict(total_episodes=100,total_frames=425556,
        features={**{f'observation.images.{c}':{'dtype':'video'} for c in ('cam0','cam1','cam2','interaction')},
                  'observation.interaction_valid':{'dtype':'bool'}})))
    monkeypatch.setenv('PI05_TOMATO100_CROP_BUNDLE',str(tmp_path))
    config=build_config()
    assert config.model.image_keys==IMAGE_KEYS and config.model.max_token_len==256
    assert config.batch_size==512 and config.num_train_steps==8000 and config.fsdp_devices==8
    assert config.data.extra_delta_transform and config.data.base_config.task_view.split=='train'
    assert not config.policy_metadata['memory_to_pi05']
    assert config.lr_schedule.warmup_steps==1000 and config.lr_schedule.decay_steps==30000
    config.data.create_vector_transforms(config.model)
