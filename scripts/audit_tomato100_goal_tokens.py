"""Audit the A goal prompt against all real normalized state bins, without truncation."""
import argparse
import json
from pathlib import Path
from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path,digest,full_text
from openpi.subtasks.tomato import GOAL


def main():
    import numpy as np
    import sentencepiece
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--norm',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    selection=json.loads(a.selection.read_text());out=derived_output_path(selection,a.out)
    receipt=json.loads((a.norm/'receipt.json').read_text())
    if receipt['selection_sha256']!=digest(a.selection) or receipt['normalization_sha256']!=digest(a.norm/'norm_stats.json'):raise ValueError('Normalization identity changed')
    sp=sentencepiece.SentencePieceProcessor(model_file=str(a.tokenizer))
    with np.load(a.norm/'real_state_tokens.npz') as arrays:bins=arrays['state_bins']
    counts=[]
    for start in range(0,len(bins),1024):
        counts.extend(map(len,sp.encode([full_text(GOAL,row.tolist()) for row in bins[start:start+1024]],add_bos=True,num_threads=4)))
    counts=np.array(counts,np.int16)
    with new_artifact_directory(out) as staging:
        np.save(staging/'goal_token_lengths.npy',counts)
        result=dict(selection_sha256=digest(a.selection),normalization_sha256=digest(a.norm/'norm_stats.json'),
            tokenizer_sha256=digest(a.tokenizer),prompt=GOAL,rows=len(counts),p50=float(np.percentile(counts,50)),
            p95=float(np.percentile(counts,95)),maximum=int(counts.max()),exceeds_256=int((counts>256).sum()),
            truncation_performed=False,training_started=False,final_training_contract_bound=False)
        (staging/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
