"""Optional language-memory boundary. No RTC, robot or action-expert changes."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from openpi.subtasks.agent_bridge import bind_chunk, chunk_matches_prompt
from openpi.subtasks.tomato import input_text

PROTOCOL='tomato-memory-input-v1'
TEXT={
 'right_grasp_started':'Right grasp attempted',
 'truss_stably_lifted':'Truss lift verified',
 'left_pick_started':'Left picking attempted',
 'tomato_detached':'Tomato detachment verified',
 'truss_returned':'Truss return verified',
 'adjustment_ended':'Adjustment end verified',
 'right_regrasp_started':'Right regrasp attempted',
 'second_grasp_verified':'Second grasp verified',
}


def validate_memory(m,*,session_id,observation_id,phase_index,now,mode):
    if (m.get('protocol')!=PROTOCOL or m.get('mode')!=mode or m.get('session_id')!=session_id
        or m.get('observation_id')!=observation_id or type(m.get('phase_index')) is not int or m.get('phase_index')!=phase_index
        or not isinstance(m.get('episode_id'),str) or not m['episode_id']
        or type(m.get('memory_version')) is not int or m['memory_version']<0):raise ValueError('Memory identity/schema mismatch')
    t,end=m.get('input_timestamp'),m.get('valid_until')
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in (t,end,now)) or not 0<=now-t<=2 or not t<=now<=end<=t+2:raise ValueError('Memory stale/future/invalid validity')
    facts=m.get('verified_facts');seen=set()
    if not isinstance(facts,list) or len(facts)>128:raise ValueError('Bounded facts required')
    for f in facts:
        if (f.get('event_type') not in TEXT or not isinstance(f.get('event_id'),str) or not f['event_id']
            or f['event_id'] in seen or f.get('independent') is not True):raise ValueError('Unverified/duplicate/unknown result')
        a,b=f.get('event_time'),f.get('evidence_time')
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in (a,b)) or not a<=b<=t:raise ValueError('Future outcome leakage')
        source=f.get('source','')
        if mode=='oracle' and source!='independent_operation_review':raise ValueError('Oracle requires independently reviewed physical outcomes')
        if mode=='predicted' and not source.startswith('outcome_verifier:'):raise ValueError('Predicted results require operation verifier provenance')
        seen.add(f['event_id'])
    if m.get('picked_count') is not None:raise ValueError('Counting is not qualified in v1; Unknown required')
    return m


def condition_prompt(stage_prompt,memory,*,state_bins,encode,max_tokens=256):
    """Shared training/inference builder; measure *full* Pi05 input, never truncate.

    Only complete facts are retained. No shortened stage instruction or state.
    Excluded facts are explicit in the receipt, not silently treated as false.
    """
    def count(prompt):
        # Matches PaligemmaTokenizer cleanup including underscores/newlines.
        cleaned=prompt.strip().replace('_',' ').replace('\n',' ')
        return len(encode(input_text(cleaned,state_bins),add_bos=True))
    base=count(stage_prompt)
    if base>max_tokens:raise ValueError('Original stage/state exceeds tokenizer budget')
    facts=memory.get('verified_facts',[]) if memory else []
    # Most recent facts first, but canonical order independent of message order.
    facts=sorted(facts,key=lambda f:(-f['event_time'],f['event_type'],f['event_id']))
    phrases=[];kept=[];dropped=[];collapsed=[]
    for f in facts:
        text=TEXT[f['event_type']]
        if text in phrases:collapsed.append(f['event_id']);continue  # Not a count; same semantic fact does not churn KV.
        trial=stage_prompt+' Memory: '+'. '.join(phrases+[text])+'.'
        if count(trial)<=max_tokens:phrases.append(text);kept.append(f['event_id'])
        else:dropped.append(f['event_id'])
    prompt=stage_prompt+(' Memory: '+'. '.join(phrases)+'.' if phrases else '')
    digest=hashlib.sha256(prompt.encode()).hexdigest()
    return prompt,dict(token_count=count(prompt),base_token_count=base,max_tokens=max_tokens,kept_fact_ids=kept,dropped_fact_ids=dropped,collapsed_fact_ids=collapsed,semantic_digest=digest,memory_omitted=bool(facts) and not phrases)


@dataclass(frozen=True)
class MemoryPromptContext:
    stage: object
    prompt: str
    version: int
    memory_digest: str
    memory_version: int
    memory_mode: str
    memory_valid_until: float | None
    language_cache_key: str
    requires_prefill: bool
    receipt: dict
    changed: bool

    def __getattr__(self,name):return getattr(self.stage,name)

    @property
    def reason(self):
        return 'memory_semantics_changed' if self.changed and not self.stage.changed else self.stage.reason


class MemoryProvider:
    """Opt-in only. Caller applies changes at its existing replan boundary.

    An executing chunk is never modified here. Memory updates invalidate only
    future reuse bindings. Current pi0.sample_actions already prefills each call.
    """
    def __init__(self,mode='none',*,encode=None,max_tokens=256):
        if mode not in ('none','oracle','predicted'):raise ValueError('Unknown memory mode')
        if mode!='none' and encode is None:raise ValueError('Real tokenizer encoder required before enabling memory')
        self.mode,self.encode,self.max_tokens=mode,encode,max_tokens
        self.session=None;self.episode=None;self.key=None;self.version=0;self.last_memory_version=-1;self.last_timestamp=None

    def reset(self,session_id):
        if not session_id or session_id==self.session:raise ValueError('Use explicit new session identity')
        self.session=session_id;self.episode=None;self.key=None;self.version=0;self.last_memory_version=-1;self.last_timestamp=None

    def accept(self,stage,memory=None,*,state_bins=None,now):
        if stage.session_id!=self.session:raise ValueError('Explicit reset/session required')
        if self.mode=='none':
            # Exactly the old object, prompt, version and chunk binding contract.
            return stage
        validate_memory(memory,session_id=self.session,observation_id=stage.observation_id,phase_index=stage.phase_index,now=now,mode=self.mode)
        if memory['input_timestamp']!=stage.input_timestamp:raise ValueError('Memory/prediction observations differ')
        if self.episode is not None and memory['episode_id']!=self.episode:raise ValueError('Episode changed without session reset')
        if memory['memory_version']<self.last_memory_version or self.last_timestamp is not None and memory['input_timestamp']<self.last_timestamp:raise ValueError('Memory rollback; explicitly reset after restart')
        prompt,receipt=condition_prompt(stage.prompt,memory,state_bins=state_bins,encode=self.encode,max_tokens=self.max_tokens)
        key=(stage.phase_index,stage.model_version,receipt['semantic_digest'])
        changed=self.key is not None and key!=self.key
        prefill=self.key is None or changed
        if self.key is None:self.version=stage.version
        elif changed:self.version+=1
        self.episode=memory['episode_id'];self.key=key;self.last_memory_version=memory['memory_version'];self.last_timestamp=memory['input_timestamp']
        cache_key=hashlib.sha256(json.dumps([self.session,*key],separators=(',',':')).encode()).hexdigest()
        return MemoryPromptContext(stage,prompt,self.version,receipt['semantic_digest'],memory['memory_version'],self.mode,memory['valid_until'],cache_key,prefill,receipt,changed or stage.changed)


def bind_memory_chunk(context,chunk_id):
    binding=bind_chunk(context,chunk_id)
    if isinstance(context,MemoryPromptContext):binding.update(memory_digest=context.memory_digest,memory_mode=context.memory_mode,language_cache_key=context.language_cache_key,memory_valid_until=context.memory_valid_until)
    return binding


def memory_chunk_matches(binding,context,*,now):
    if not chunk_matches_prompt(binding,context):return False
    if isinstance(context,MemoryPromptContext):
        return (type(binding.get('memory_valid_until')) in (int,float) and context.memory_valid_until is not None and context.input_timestamp<=now<=min(context.memory_valid_until,binding['memory_valid_until'])
                and binding.get('memory_digest')==context.memory_digest and binding.get('memory_mode')==context.memory_mode
                and binding.get('language_cache_key')==context.language_cache_key)
    return 'memory_digest' not in binding


def condition_training_sample(sample,memory,*,state_bins,encode):
    """Materialize after the SAME normalization/binning as inference.

    No automatic data-loader injection: explicit row-aligned causal memory and
    normalized bins are mandatory. Returns a copy; actions/state are untouched.
    """
    if memory is None:return dict(sample),dict(mode='none',memory_used=False)
    if memory.get('input_timestamp')!=sample['timestamp'] or memory.get('episode_id')!=sample['episode_id']:raise ValueError('Training row/memory identity mismatch')
    validate_memory(memory,session_id=sample['session_id'],observation_id=sample['observation_id'],phase_index=sample['phase_index'],now=sample['timestamp'],mode=memory['mode'])
    prompt,receipt=condition_prompt(sample['prompt'],memory,state_bins=state_bins,encode=encode)
    return {**sample,'prompt':prompt},receipt
