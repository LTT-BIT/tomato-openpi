"""Keep five-phase prompts, source data and materialized frame supervision aligned."""
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.subtasks import tomato_in_hand as c
from openpi.subtasks.prepare import digest,write_json,write_jsonl
from scripts.prepare_tomato_in_hand119 import materialize


class Tokenizer:
    def encode(self,text,add_bos=True):return [1]*len(text.split())


def document(length=10):
    return dict(prompt_profile=c.PROFILE,stage='coarse',segments=[dict(label=k,text=c.ACTION_PROMPTS[k],
        start_frame=2*i,end_frame=2*i+2,reviewed=True) for i,k in enumerate(c.SEQUENCE)])


@pytest.mark.parametrize('mutation',['seven','old_prompt','gap','review','length','profile'])
def test_reject_incompatible_annotation(mutation):
    d=document();length=10
    if mutation=='seven':d['segments'].append(copy.deepcopy(d['segments'][-1]))
    if mutation=='old_prompt':d['segments'][2]['text']='Right fingers reposition the truss on the right plate'
    if mutation=='gap':d['segments'][2]['start_frame']+=1
    if mutation=='review':d['segments'][0]['reviewed']=False
    if mutation=='length':length=11
    if mutation=='profile':d['prompt_profile']='tomato-in-hand-five-phases-v1'
    with pytest.raises(ValueError):c.reviewed_segments(d,length)


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'EPISODES',2);monkeypatch.setattr(c,'FRAMES',20)
    root=tmp_path/'source'/'A'/'batch';features={
        'observation.state':dict(dtype='float32',shape=[116],names=list(c.SOURCE_STATE_NAMES)),
        'action':dict(dtype='float32',shape=[58],names=list(c.SOURCE_ACTION_NAMES)),
        **{cam:dict(dtype='video',shape=[224,224,3],names=['height','width','channels'],info={'video.fps':30}) for cam in c.CAMERAS},
        **{k:dict(dtype='int64',shape=[1],names=None) for k in ['index','frame_index','episode_index','task_index']},
        'timestamp':dict(dtype='float32',shape=[1],names=None)}
    info=dict(codebase_version='v2.1',fps=30,features=features,total_episodes=2,total_frames=20,chunks_size=1000,
        data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
    entries=[dict(episode_index=ep,length=10,source_recording_id=f'recording{ep}',tasks=['raw goal']) for ep in range(2)]
    write_json(root/'meta/info.json',info);write_jsonl(root/'meta/episodes.jsonl',entries)
    write_jsonl(root/'meta/tasks.jsonl',[dict(task_index=0,task='raw goal')]);write_json(root/'visual_review.json',{})
    receipts=[]
    for ep in range(2):
        data=dict(index=list(range(ep*10,(ep+1)*10)),frame_index=list(range(10)),episode_index=[ep]*10,task_index=[0]*10,
            timestamp=np.arange(10,dtype=np.float32)/30,
            **{'observation.state':np.full((10,116),ep,dtype=np.float32).tolist(),'action':np.full((10,58),ep+.5,dtype=np.float32).tolist()})
        p=root/f'data/chunk-000/episode_{ep:06d}.parquet';p.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.table(data),p)
        paths=['meta/info.json','meta/episodes.jsonl','meta/tasks.jsonl',str(p.relative_to(root))]
        for cam in c.CAMERAS:
            video=root/f'videos/chunk-000/{cam}/episode_{ep:06d}.mp4';video.parent.mkdir(parents=True,exist_ok=True);video.write_bytes(f'video{ep}{cam}'.encode());paths.append(str(video.relative_to(root)))
        d=document();d['source']=dict(episode_index=ep,length=10,fps=30,source_recording_id=f'recording{ep}',sha256={p:digest(root/p) for p in paths})
        write_json(root/f'annotations/subtasks/coarse/episode_{ep:06d}.json',d)
        receipts.append(dict(episode=ep,frames=10,output_sha256=d['source']['sha256'][paths[-1]],checks=dict(all_boxes_contained=True,uncovered_frames=[],center_steps_over_3pct=0,scale_steps_over_2pct=0)))
    roi=dict(current_rgb_pixels_only=True,state_action_unchanged=True,future_coordinates_used=True,
        semantic_review_status='sampled_coding_agent_review_complete_not_independent_per_frame_accuracy',
        visual_review_sha256=digest(root/'visual_review.json'),roi_receipts=receipts)
    write_json(root/'roi_manifest.json',roi)
    publication=tmp_path/'source';write_json(publication/'manifest.json',dict(packages=[dict(path=str(root),category='normal',manifest_sha256=digest(root/'roi_manifest.json'),episodes=2,frames=20)]))
    return c.freeze(publication),root


def test_full_materialization_preserves_raw_and_labels(tmp_path,frozen):
    selection,root=frozen;before={str(p):digest(p) for p in root.rglob('*') if p.is_file()}
    dest=tmp_path/'prepared';m=materialize(selection,dest,Tokenizer())
    assert m['physical_columns_equal'] and m['raw_sources_unchanged']
    assert before=={str(p):digest(p) for p in root.rglob('*') if p.is_file()}
    for ep in range(2):
        t=pq.read_table(dest/f'b_coarse/data/chunk-000/episode_{ep:06d}.parquet')
        assert t['task_index'].to_pylist()==[0,0,1,1,2,2,1,1,3,3]
        assert t['coarse_phase_index'].to_pylist()==[0,0,1,1,2,2,3,3,4,4]
        assert all(t['observation.interaction_available'].to_pylist())
        np.testing.assert_array_equal(t['action'].to_pylist(),np.full((10,58),ep+.5))
    with pytest.raises(FileExistsError):materialize(selection,dest,Tokenizer())


def test_materialization_rejects_changed_source(tmp_path,frozen):
    selection,root=frozen
    (root/'videos/chunk-000/observation.images.cam0/episode_000000.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):materialize(selection,tmp_path/'prepared',Tokenizer())
    assert not (tmp_path/'prepared').exists()


def test_materialization_rejects_source_destination(frozen):
    selection,root=frozen
    with pytest.raises(ValueError,match='separate'):materialize(selection,root/'prepared',Tokenizer())
