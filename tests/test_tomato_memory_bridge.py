from dataclasses import replace
import pytest
from openpi.subtasks.memory_bridge import *
from openpi.subtasks.agent_bridge import PredictionPromptProvider, bind_chunk
from test_tomato_agent_bridge import prediction


def stage(t=1.):
    p=PredictionPromptProvider();p.reset('s');return p.accept_prediction(prediction('pickup',t),observation_id=str(t),now=t+.1)


def memory(ctx,version=1,kind='truss_stably_lifted',mode='oracle'):
    return dict(protocol=PROTOCOL,session_id='s',episode_id='e',observation_id=ctx.observation_id,input_timestamp=ctx.input_timestamp,valid_until=ctx.input_timestamp+1.,memory_version=version,phase_index=ctx.phase_index,mode=mode,verified_facts=[dict(event_id='event1',event_type=kind,event_time=.1,evidence_time=.5,source='independent_operation_review',independent=True)],picked_count=None)


def encode(text,add_bos=True):return list(range(len(text.split())+int(add_bos)))


def test_none_is_exact_old_context_and_binding():
    c=stage();p=MemoryProvider();p.reset('s')
    assert p.accept(c,now=1.1) is c
    assert bind_memory_chunk(c,'x')==bind_chunk(c,'x')


def test_semantic_identity_avoids_churn_and_new_memory_invalidates_old_chunk():
    c=stage();p=MemoryProvider('oracle',encode=encode);p.reset('s');m=memory(c)
    a=p.accept(c,m,state_bins=[0]*58,now=1.1);old=bind_memory_chunk(a,'x')
    assert memory_chunk_matches(old,a,now=1.1)
    assert a.prompt.startswith(c.prompt) and 'Truss lift verified' in a.prompt
    b=p.accept(c,{**m,'memory_version':2},state_bins=[0]*58,now=1.1)
    assert b.version==a.version and b.language_cache_key==a.language_cache_key
    d=p.accept(c,memory(c,3,'truss_returned'),state_bins=[0]*58,now=1.1)
    assert d.requires_prefill and d.version==a.version+1
    assert not memory_chunk_matches(old,d,now=1.1)
    assert not memory_chunk_matches(bind_memory_chunk(d,'new'),d,now=3.)


def test_full_token_budget_drops_memory_never_state_or_original_prompt():
    c=stage();m=memory(c)
    base=len(encode(input_text(c.prompt,[255]*58)))
    prompt,r=condition_prompt(c.prompt,m,state_bins=[255]*58,encode=encode,max_tokens=base)
    assert prompt==c.prompt and r['memory_omitted'] and r['dropped_fact_ids']==['event1']
    with pytest.raises(ValueError):condition_prompt(c.prompt,m,state_bins=[255]*58,encode=encode,max_tokens=base-1)


@pytest.mark.parametrize('change',[
 {'input_timestamp':3.},{'valid_until':.5},{'session_id':'wrong'},
 {'picked_count':1},{'phase_index':4},{'mode':'predicted'}])
def test_invalid_memory_is_not_used(change):
    c=stage();p=MemoryProvider('oracle',encode=encode);p.reset('s')
    with pytest.raises(ValueError):p.accept(c,{**memory(c),**change},state_bins=[0]*58,now=1.1)


def test_future_outcomes_and_stage_pseudo_labels_rejected():
    c=stage();m=memory(c);m['verified_facts'][0]['evidence_time']=2.
    with pytest.raises(ValueError):validate_memory(m,session_id='s',observation_id='1.0',phase_index=0,now=1.1,mode='oracle')
    m=memory(c);m['verified_facts'][0]['source']='stage_prediction'
    with pytest.raises(ValueError):validate_memory(m,session_id='s',observation_id='1.0',phase_index=0,now=1.1,mode='oracle')


def test_training_and_inference_use_identical_expression():
    c=stage();m=memory(c);sample={'session_id':'s','episode_id':'e','observation_id':c.observation_id,'phase_index':0,'timestamp':1.,'prompt':c.prompt,'actions':[[1.]*58]*50}
    x,r=condition_training_sample(sample,m,state_bins=[0]*58,encode=encode)
    prompt,rr=condition_prompt(c.prompt,m,state_bins=[0]*58,encode=encode)
    assert x['prompt']==prompt and r==rr and x['actions'] is sample['actions']
    assert sample['prompt']==c.prompt


def test_restart_rollback_and_duplicate_event_rejected():
    c=stage();p=MemoryProvider('oracle',encode=encode);p.reset('s')
    p.accept(c,memory(c,3),state_bins=[0]*58,now=1.1)
    with pytest.raises(ValueError):p.accept(c,memory(c,2),state_bins=[0]*58,now=1.1)
    m=memory(c,4);m['verified_facts']*=2
    with pytest.raises(ValueError):p.accept(c,m,state_bins=[0]*58,now=1.1)


def test_same_content_refresh_does_not_extend_old_chunk_evidence_lifetime():
    c=stage();p=MemoryProvider('oracle',encode=encode);p.reset('s')
    a=p.accept(c,memory(c),state_bins=[0]*58,now=1.1)
    assert a.requires_prefill
    binding=bind_memory_chunk(a,'old')
    new_stage=replace(c,input_timestamp=3.,observation_id='3.0')
    fresh=memory(new_stage,2)
    b=p.accept(new_stage,fresh,state_bins=[0]*58,now=3.1)
    assert a.version==b.version
    assert not memory_chunk_matches(binding,b,now=3.1)
    assert memory_chunk_matches(bind_memory_chunk(b,'new'),b,now=3.1)


def test_training_future_label_guard():
    c=stage();m=memory(c);m['verified_facts'][0]['evidence_time']=2.
    sample=dict(session_id='s',episode_id='e',observation_id='1.0',phase_index=0,timestamp=1.,prompt=c.prompt)
    with pytest.raises(ValueError):condition_training_sample(sample,m,state_bins=[0]*58,encode=encode)


def test_no_memory_training_adapter_is_exact_data_noop():
    sample={'prompt':'unchanged','state':[0.]*58,'actions':[[0.]*58]*50}
    result,receipt=condition_training_sample(sample,None,state_bins=None,encode=None)
    assert result==sample and result['actions'] is sample['actions']
    assert receipt==dict(mode='none',memory_used=False)


def test_training_row_and_memory_must_match_episode_and_exact_timestamp():
    c=stage();m=memory(c)
    sample=dict(session_id='s',episode_id='e',observation_id=c.observation_id,phase_index=0,timestamp=1.,prompt=c.prompt)
    for changes in ({'episode_id':'other'}, {'input_timestamp':.9}):
        with pytest.raises(ValueError):condition_training_sample(sample,{**m,**changes},state_bins=[0]*58,encode=encode)


def test_cross_episode_memory_requires_new_session():
    c=stage();p=MemoryProvider('oracle',encode=encode);p.reset('s');m=memory(c)
    p.accept(c,m,state_bins=[0]*58,now=1.1)
    with pytest.raises(ValueError):p.accept(c,{**m,'episode_id':'other'},state_bins=[0]*58,now=1.1)
