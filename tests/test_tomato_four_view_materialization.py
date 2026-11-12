import gzip
import json
import pytest
from scripts.materialize_tomato100_four_view import read_candidate_index,collect_candidates


def row():return dict(source_frame=0,timestamp=0,available_at=0,valid=True,crop_box=[1,1,7,7],interaction_box=[2,2,6,6],source_size=[10,10])


def write(path,entry):
    with gzip.open(path,'wt') as stream:stream.write(json.dumps(entry)+'\n')


def test_index_preserves_causality_null_loss_and_native_containment(tmp_path):
    p=tmp_path/'index.jsonl.gz';entry=row();write(p,entry)
    assert read_candidate_index(p,1)[0]['valid']
    entry['available_at']=.1;write(p,entry)
    with pytest.raises(ValueError,match='causality'):read_candidate_index(p,1)
    entry=row();entry['valid']=False;write(p,entry)
    with pytest.raises(ValueError,match='null'):read_candidate_index(p,1)
    entry['crop_box']=entry['interaction_box']=None;write(p,entry)
    assert not read_candidate_index(p,1)[0]['valid']
    entry=row();entry['crop_box']=[3,3,5,5];write(p,entry)
    with pytest.raises(ValueError,match='contain'):read_candidate_index(p,1)


def test_partial_shards_and_both_training_entries_reject_candidate(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='Four finished'):collect_candidates(tmp_path)
    from training_configs.experiments.tomato_sharpa.subtasks.complete100_a_crop import build_config as a
    from training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop import build_config as b
    (tmp_path/'crop_manifest.json').write_text(json.dumps(dict(full_roi_coverage=False,frames=425556,episodes=100)))
    monkeypatch.setenv('PI05_TOMATO100_CROP_A_BUNDLE',str(tmp_path));monkeypatch.setenv('PI05_TOMATO100_CROP_BUNDLE',str(tmp_path))
    for build in (a,b):
        with pytest.raises(ValueError,match='coverage'):build()


@pytest.mark.parametrize('stream,suffix',[
    ('continuous_current_pixels_false_mask_on_loss_v1','_stable_r1'),
    ('continuous_current_pixels_state_guarded_v2','_state_guarded_r1'),
    ('current_rgb_local_or_full_fallback_v5','_available_r6')])
def test_stable_stream_selects_new_assets_and_explicit_training_ready_gate(tmp_path,monkeypatch,stream,suffix):
    from pathlib import Path
    from openpi.subtasks.rgb100 import digest
    from openpi.subtasks.tomato import PROMPTS
    from training_configs.embodiments.fr3_sharpa_crop import FR3_SHARPA_CROP
    from training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop import build_config
    selection=Path(__file__).resolve().parents[1]/'training_configs/experiments/tomato_sharpa/subtasks/records/complete100_rgb_crop_20261111_saved_labels_r1.json'
    manifest=dict(selection_sha256=digest(selection),full_roi_coverage=True,training_ready=True,
                  episodes=100,frames=425556,roi_generation_mode='causal_prediction',roi_generator_signature={'test':'metadata-only fixture'},
                  stream_contract=stream)
    if suffix=='_available_r6':
        manifest.update(full_roi_coverage=False,full_image_coverage=True,image_available_frames=425556,
                        interaction_mask_key='observation.interaction_available')
    (tmp_path/'crop_manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'splits.json').write_text(json.dumps(dict(splits={'train':list(range(100))})))
    meta=tmp_path/'b_coarse/meta';meta.mkdir(parents=True)
    features={key:dict(dtype='video') for key in FR3_SHARPA_CROP.camera_keys}
    features['observation.interaction_valid']=dict(dtype='bool')
    if suffix=='_available_r6':features['observation.interaction_available']=dict(dtype='bool')
    (meta/'info.json').write_text(json.dumps(dict(total_episodes=100,total_frames=425556,features=features)))
    (meta/'tasks.jsonl').write_text(''.join(json.dumps({'task':prompt})+'\n' for prompt in dict.fromkeys(PROMPTS)))
    monkeypatch.setenv('PI05_TOMATO100_CROP_BUNDLE',str(tmp_path))
    config=build_config()
    assert config.data.assets.asset_id.endswith(suffix+'_delta_30hz')
    assert config.data.repo_id.endswith(suffix)
    assert config.name.endswith(suffix)
    if suffix=='_available_r6':
        assert config.data.interaction_mask_key=='observation.interaction_available'
        manifest['image_available_frames']=425555
        (tmp_path/'crop_manifest.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError,match='coverage'):build_config()
        manifest['image_available_frames']=425556
    manifest['training_ready']=False
    (tmp_path/'crop_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='coverage'):build_config()
