"""Protocol failures and repeated-stage mapping, without any robot connections."""
import math
import pytest
from openpi.subtasks.agent_bridge import (
    LABELS, PROTOCOL, PredictionPromptProvider, bind_chunk, chunk_matches_prompt,
    named_agent_history, validate_prediction,
)
from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.subtasks.tomato import ACTION_PROMPTS, COARSE_SEQUENCE, PHASES


def prediction(action, t=1., future='harvest'):
    def values(label):
        logits = [4. if k == label else -4. for k in LABELS]
        ex = [math.exp(x-max(logits)) for x in logits]
        return logits,[x/sum(ex) for x in ex]
    z,p=values(action);fz,fp=values(future)
    return dict(protocol=PROTOCOL, labels=list(LABELS), valid=True,error=None,
                model_version='sha',observation_id=str(t),session_id='s',input_timestamp=t,
                result_timestamp=t+.05,future_horizon_s=1.,prediction_mode='joint',
                current_logits=z,current_probabilities=p,future_logits=fz,future_probabilities=fp)


def accept(provider, action, t, future='harvest'):
    return provider.accept_prediction(prediction(action,t,future), observation_id=str(t), now=t+.1)


def test_explicit_reset_then_seven_positions_four_prompts_and_no_auto_success():
    p=PredictionPromptProvider(confirmations=2)
    with pytest.raises(ValueError,match='Initialize'):
        accept(p,'pickup',1.)
    p.reset('s');ctx=accept(p,'pickup',1.)
    assert ctx.phase=='pickup_1'
    t=2.;contexts=[ctx]
    for action in COARSE_SEQUENCE[1:]:
        first=accept(p,action,t);ctx=accept(p,action,t+.5);t+=1
        assert not first.changed and ctx.changed
        contexts.append(ctx)
    assert [c.phase for c in contexts]==list(PHASES)
    assert len({c.prompt for c in contexts})==4
    assert contexts[2].prompt==contexts[6].prompt==ACTION_PROMPTS['return']
    assert not any(c.task_complete for c in contexts)


def test_future_warning_never_advances_prompt_and_chunk_version_changes():
    p=PredictionPromptProvider(confirmations=1);p.reset('s')
    a=accept(p,'pickup',1.,future='harvest')
    assert a.future_warning=='harvest' and a.phase_index==0
    binding=bind_chunk(a,'chunk1');assert chunk_matches_prompt(binding,a)
    b=accept(p,'harvest',2.)
    assert not chunk_matches_prompt(binding,b)
    assert b.version==a.version+1


@pytest.mark.parametrize('mutation',[
    lambda x:x.update(current_probabilities=[1,0,0,0,0,0,0]),
    lambda x:x.update(current_probabilities=[float('nan')]*4),
    lambda x:x.update(observation_id='wrong'),
    lambda x:x.update(input_timestamp=-100.),
    lambda x:x.update(current_logits=[0,0,0,0]),
    lambda x:x.update(labels=['return','pickup','harvest','adjust']),
])
def test_invalid_schema_rejected_without_progress(mutation):
    p=PredictionPromptProvider();p.reset('s');value=prediction('harvest')
    mutation(value)
    with pytest.raises(ValueError):
        p.accept_prediction(value,observation_id='1.0',now=1.1)
    assert p.phase==p.count==0 and p.last_timestamp is None


def test_illegal_skip_and_out_of_order_cannot_advance():
    p=PredictionPromptProvider(confirmations=1);p.reset('s')
    assert accept(p,'adjust',1.).phase_index==0
    with pytest.raises(ValueError,match='out-of-order'):
        accept(p,'harvest',1.)
    assert accept(p,'return',2.).phase_index==0


def test_joint_mapping_uses_names_not_shared_dimension():
    # Agent is arms-left/right then hands-left/right; policy is left arm+hand then right.
    agent=list(ORDER[:7])+list(ORDER[29:36])+list(ORDER[7:29])+list(ORDER[36:])
    mapped=named_agent_history([list(range(58)) for _ in range(16)],ORDER,[n+'.position' for n in agent])
    assert mapped[0]==[list(ORDER).index(n) for n in agent]
    assert mapped[0]!=list(range(58))
    with pytest.raises(ValueError,match='name mismatch'):
        named_agent_history([list(range(58))]*16,ORDER,[f'wrong{i}' for i in range(58)])


def test_gap_breaks_consecutive_confirmation():
    p=PredictionPromptProvider(confirmations=2,max_age_s=2.);p.reset('s')
    assert not accept(p,'harvest',1.).changed
    assert not accept(p,'harvest',10.).changed
    assert accept(p,'harvest',10.5).changed


def test_old_chunk_cannot_cross_session_reset_even_with_same_prompt_and_version():
    p=PredictionPromptProvider();p.reset('s');a=accept(p,'pickup',1.)
    old=bind_chunk(a,'old');p.reset('next-session')
    value=prediction('pickup',2.);value['session_id']='next-session'
    b=p.accept_prediction(value,observation_id='2.0',now=2.1)
    assert a.prompt==b.prompt and a.version==b.version
    assert not chunk_matches_prompt(old,b)
    with pytest.raises(ValueError,match='new session'):
        p.reset('next-session')
