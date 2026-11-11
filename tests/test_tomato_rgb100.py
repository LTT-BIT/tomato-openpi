import copy
import json
from pathlib import Path

import pytest

from openpi.subtasks.rgb100 import freeze_selection, full_text, sequence_slots, write_new, audit_tokens
from openpi.subtasks.tomato import COARSE_SEQUENCE, SOURCE_ACTION_NAMES, SOURCE_STATE_NAMES


def manifest(tmp_path):
    rows=[]
    for i in range(100):
        root=tmp_path/str(i)
        rows.append(dict(episode_id=str(i),source_recording_id=f'source{i}',fps=30,length=7,
            episode_index=i,dataset_root=str(root),parquet=str(root/'state.parquet'),
            videos=[str(root/f'{c}.mp4') for c in range(3)],
            input_sha256={'state.parquet':'hash',**{f'{c}.mp4':'hash' for c in range(3)}},
            split='val' if i<8 else 'train',
            info={'fps':30,'features':{'observation.state':{'shape':[116],'names':list(SOURCE_STATE_NAMES)},
                'action':{'shape':[58],'names':list(SOURCE_ACTION_NAMES)}}},
            annotation={'path':str(root/'annotation.json'),'sha256':'hash','segments':[
                dict(label=c,start_frame=j,end_frame=j+1,reviewed=True) for j,c in enumerate(COARSE_SEQUENCE)]}))
    return {'episodes':rows}


def freeze(tmp_path,m):
    p=tmp_path/'manifest.json';p.write_text(json.dumps(m))
    return freeze_selection(p,verify_sources=False)


def test_all100_training_does_not_inherit_agent_holdout(tmp_path):
    r=freeze(tmp_path,manifest(tmp_path))
    assert r['splits']=={'train':list(range(100))}
    assert all(e['action_split']=='train' for e in r['episodes'])
    assert r['episodes'][0]['agent_split_provenance']=='val'
    assert not r['training_ready'] and not r['source_files_verified']
    assert all(e['phase_label_status']=='not_verified' for e in r['episodes'])


@pytest.mark.parametrize('change',['count','duplicate','three_stage','gap','schema','hash'])
def test_selection_rejects_identity_and_label_damage(tmp_path,change):
    m=manifest(tmp_path);e=m['episodes'][0]
    if change=='count':m['episodes'].pop()
    if change=='duplicate':m['episodes'][1]['source_recording_id']=e['source_recording_id']
    if change=='three_stage':e['annotation']['segments']=e['annotation']['segments'][:3]
    if change=='gap':e['annotation']['segments'][1]['start_frame']=2
    if change=='schema':e['info']['features']['action']['names'].reverse()
    if change=='hash':del e['input_sha256']['0.mp4']
    with pytest.raises(ValueError):freeze(tmp_path,m)


def test_saved_vs_agent_boundary_conflicts_are_explicit_not_silently_applied(tmp_path,monkeypatch):
    from openpi.subtasks import rgb100
    m=manifest(tmp_path)
    for e in m['episodes']:
        p=Path(e['annotation']['path']);p.parent.mkdir()
        saved=copy.deepcopy(e['annotation']['segments'])
        # Preserve seven contiguous phases while changing one effective boundary.
        for j,r in enumerate(saved):r['start_frame']=j*2;r['end_frame']=(j+1)*2
        e['length']=14
        e['annotation']['segments']=copy.deepcopy(saved)
        p.write_text(json.dumps({'segments':saved}))
        (p.parent/'meta').mkdir()
        (p.parent/'meta/info.json').write_text(json.dumps({**e['info'],'codebase_version':'v2.1'}))
        (p.parent/'meta/episodes.jsonl').write_text(json.dumps(dict(episode_index=e['episode_index'],length=14,source_recording_id=e['source_recording_id']))+'\n')
    m['episodes'][0]['annotation']['segments'][0]['end_frame']=3
    m['episodes'][0]['annotation']['segments'][1]['start_frame']=3
    monkeypatch.setattr(rgb100,'digest',lambda path:'hash')
    p=tmp_path/'manifest.json';p.write_text(json.dumps(m))
    r=freeze_selection(p)
    assert r['phase_label_resolution_required']==1
    assert len(r['episodes'][0]['phase_differences'])==2
    assert not r['training_ready']
    assert json.loads(Path(m['episodes'][0]['annotation']['path']).read_text())['segments'][0]['end_frame']==2


def test_sequence_lengths_separate_text_history_and_action_positions():
    assert sequence_slots(views=3)['prefix_plus_action_slots']==1074
    assert sequence_slots()['prefix_plus_action_slots']==1330
    assert sequence_slots(history_tokens=32)['prefix_plus_action_slots']==1362
    assert sequence_slots(history_tokens=64)['prefix_plus_action_slots']==1394
    assert sequence_slots(text_capacity=320)['prefix_plus_action_slots']==1394


def test_complete_text_preserves_state_and_memory_without_truncation():
    text=full_text('right_hand\nmove',[255]*58,'x '*1000)
    assert text.startswith('Task: right hand move Memory:')
    assert text.count('255')==58 and text.endswith(';\nAction: ')
    with pytest.raises(ValueError):full_text('move',[255]*57)
    with pytest.raises(ValueError):full_text('move',[256]*58)


def test_actual_distribution_requires_unique_observations_and_one_normalization():
    # Synthetic tokenizer only checks audit plumbing; real SP run is separate.
    def encode(text,add_bos=False):return list(text)+(list('B') if add_bos else [])
    row=dict(action='pickup',observation_id='e:1',normalization_sha256='new100',state_bins=[255]*58)
    r=audit_tokens(encode,tokenizer_sha256='synthetic',normalized_rows=[row])
    assert r['real_distribution']['rows']==1 and r['real_distribution']['exceeds_256']==1
    assert not r['truncation_performed']
    with pytest.raises(ValueError):audit_tokens(encode,tokenizer_sha256='synthetic',normalized_rows=[row,row])
    other={**row,'observation_id':'e:2','normalization_sha256':'old93'}
    with pytest.raises(ValueError):audit_tokens(encode,tokenizer_sha256='synthetic',normalized_rows=[row,other])


def test_receipts_cannot_overwrite_existing_files(tmp_path):
    p=tmp_path/'receipt.json';write_new(p,{'original':True})
    with pytest.raises(FileExistsError):write_new(p,{'replacement':True})
    assert json.loads(p.read_text())=={'original':True}
