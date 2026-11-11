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
