"""Real tokenizer budget check, using full58D input; never loads action weights."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from openpi.subtasks.tomato import ACTION_PROMPTS,input_text,audit_token_budget
from openpi.subtasks.memory_bridge import condition_prompt,TEXT


def run(tokenizer,out):
    if out.exists():raise FileExistsError(out)
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=str(tokenizer));rng=np.random.default_rng(42);cases=[]
    facts=[dict(event_id=str(i),event_type=e,event_time=float(i)) for i,e in enumerate(TEXT)]
    for action,prompt in ACTION_PROMPTS.items():
        # Exhaustive uniform bins + reproducible mixed58D states.
        states=[[i]*58 for i in range(-1,256)]+rng.integers(-1,256,size=(200,58)).tolist()
        for index,bins in enumerate(states):
            p,r=condition_prompt(prompt,{'verified_facts':facts},state_bins=bins,encode=sp.encode)
            assert p.startswith(prompt) and len(sp.encode(input_text(p,bins),add_bos=True))<=256
            cases.append(dict(action=action,uniform_bin=index-1 if index<257 else None,**r))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),base_worst_case=audit_token_budget(sp),tested_states=len(cases),max_total_tokens=max(x['token_count'] for x in cases),all_memory_omitted=sum(x['memory_omitted'] for x in cases),any_fact_retained=sum(bool(x['kept_fact_ids']) for x in cases),note='Artificial state-bin budget audit and artificial past facts; not trained memory or action evaluation',cases=cases),indent=2)+'\n');print('states',len(cases),'max tokens',max(x['token_count'] for x in cases),'memory omitted',sum(x['memory_omitted'] for x in cases))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.tokenizer,a.out)
