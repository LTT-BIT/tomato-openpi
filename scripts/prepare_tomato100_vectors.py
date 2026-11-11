"""Compute image-independent all100 vectors and real token lengths; final train contract stays gated."""
import argparse
import json
from pathlib import Path
import time

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import IMAGE_KEYS, derived_output_path, digest, full_text


def window_indices(length,rows,horizon=50):
    import numpy as np
    return np.minimum(np.asarray(rows)[:,None]+np.arange(horizon),length-1)


def run(selection_path,tokenizer_path,out):
    import numpy as np
    import pyarrow.parquet as pq
    import sentencepiece
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import normalize
    from openpi.training.config import LeRobotFr3SharpaCropDataConfig
    selection=json.loads(selection_path.read_text())
    if selection.get('phase_label_resolution_required')!=0 or selection['splits']!={'train':list(range(100))} or len(selection['episodes'])!=100:
        raise ValueError('Resolved all100 selection required')
    out=derived_output_path(selection,out)
    config=LeRobotFr3SharpaCropDataConfig(extra_delta_transform=True)
    repack,vectors=config.create_vector_transforms(Pi0Config(pi05=True,action_dim=58,image_keys=IMAGE_KEYS))
    transform=transforms.compose([*repack.inputs,*vectors.inputs])
    image=np.zeros((1,1,3),np.uint8)
    def apply(state,actions):
        return transform({'observation.state':state,'action':actions,'prompt':'',
            **dict.fromkeys(('observation.images.cam0','observation.images.cam1','observation.images.cam2','observation.images.interaction'),image),
            'observation.interaction_valid':np.bool_(False)})
    stats={key:normalize.RunningStats() for key in ['state','actions']}
    totals={key:np.zeros(58,np.float64) for key in stats};squares={key:np.zeros(58,np.float64) for key in stats}
    counts=dict.fromkeys(stats,0);states=[];identities=[];prompts=[];sources=[]
    started=time.perf_counter()
    with new_artifact_directory(out) as staging:
        for row in selection['episodes']:
            rel=f"data/chunk-{row['source_episode_index']//1000:03d}/episode_{row['source_episode_index']:06d}.parquet"
            path=Path(row['dataset'])/rel;expected=row['source_sha256'][rel]
            if digest(path)!=expected or digest(row['annotation'])!=row['annotation_sha256']:raise ValueError('Source changed')
            table=pq.read_table(path,columns=['observation.state','action'])
            state=np.array(table['observation.state'].to_pylist(),np.float32);actions=np.array(table['action'].to_pylist(),np.float32)
            if state.shape!=(row['length'],116) or actions.shape!=(row['length'],58):raise ValueError('Vector schema mismatch')
            measured=apply(state,actions[:,None,:])['state'];states.append(measured)
            identities.extend((row['episode_index'],i) for i in range(row['length']))
            for segment in row['training_segments']:
                prompts.extend([segment['text']]*(segment['end_frame']-segment['start_frame']))
            for start in range(0,len(state),1024):
                rows=np.arange(start,min(start+1024,len(state)))
                batch=apply(state[rows],actions[window_indices(len(state),rows)])
                for key in stats:
                    values=np.asarray(batch[key],np.float64).reshape(-1,58)
                    if not np.isfinite(values).all():raise ValueError('Nonfinite vector')
                    stats[key].update(values);totals[key]+=values.sum(0);squares[key]+=(values**2).sum(0);counts[key]+=len(values)
            if digest(path)!=expected:raise ValueError('Source changed during vector preparation')
            sources.append(dict(episode_index=row['episode_index'],parquet_sha256=expected))
            print(f"Vector stats {len(sources)}/100",flush=True)
        n=sum(r['length'] for r in selection['episodes'])
        if counts!={'state':n,'actions':n*50} or len(prompts)!=n:raise ValueError('Incomplete vector coverage')
        result={key:s.get_statistics() for key,s in stats.items()}
        for key,s in result.items():
            np.testing.assert_allclose(s.mean,totals[key]/counts[key],rtol=1e-9,atol=1e-10)
            np.testing.assert_allclose(s.std,np.sqrt(np.maximum(0,squares[key]/counts[key]-s.mean**2)),rtol=1e-8,atol=1e-9)
            if not all(np.isfinite(getattr(s,f)).all() for f in ['mean','std','q01','q99']):raise ValueError('Nonfinite statistics')
        normalize.save(staging,result)
        norm_sha=digest(staging/'norm_stats.json')
        q=np.concatenate(states)
        normalized=transforms.Normalize({'state':result['state']},use_quantiles=True)({'state':q})['state']
        bins=np.digitize(normalized,np.linspace(-1,1,257)[:-1])-1
        sp=sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
        lengths=[]
        for start in range(0,n,1024):
            texts=[full_text(prompt,b.tolist()) for prompt,b in zip(prompts[start:start+1024],bins[start:start+1024],strict=True)]
            lengths.extend(map(len,sp.encode(texts,add_bos=True,num_threads=4)))
        lengths=np.array(lengths,np.int16)
        np.savez_compressed(staging/'real_state_tokens.npz',state_bins=bins.astype(np.int16),
                           observation_ids=np.array(identities,np.int32),token_lengths=lengths)
        per_prompt={p:dict(rows=int(mask.sum()),p50=float(np.percentile(lengths[mask],50)),p95=float(np.percentile(lengths[mask],95)),
                          maximum=int(lengths[mask].max()),exceeds_256=int((lengths[mask]>256).sum()))
                    for p in sorted(set(prompts)) for mask in [np.array(prompts)==p]}
        receipt=dict(selection_sha256=digest(selection_path),normalization_sha256=norm_sha,tokenizer_sha256=digest(tokenizer_path),
            episodes=100,frames=n,vector_counts=counts,action_horizon=50,action_semantics='delta14_arm_absolute44_hand',
            state_shape=[n,58],normalization='quantile q01/q99; same Normalize transform, no clipping',
            statistics_estimator='existing RunningStats, float64, 5000 histogram bins',sources=sources,
            token_distribution=dict(rows=n,p50=float(np.percentile(lengths,50)),p95=float(np.percentile(lengths,95)),
                maximum=int(lengths.max()),exceeds_256=int((lengths>256).sum()),by_prompt=per_prompt),
            image_independent_vectors=True,final_training_contract_bound=False,training_ready=False,
            note='Preparation statistics only: must bind and verify final four-view dataset contract before training; no old93 statistics reused.',
            source_writes=False,training_started=False,elapsed_seconds=time.perf_counter()-started)
        (staging/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.selection,a.tokenizer,a.out),indent=2))


if __name__=='__main__':main()
