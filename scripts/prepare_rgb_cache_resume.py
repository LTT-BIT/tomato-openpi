"""Verify exact image/vector/token inputs and benchmark the unchanged finite sampler."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batches',type=int,default=32)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    import jax
    import numpy as np
    import torch
    from scripts.tomato100_run import configure_inputs,DEFAULT_BUNDLE,PREPARATION,MODULES
    from scripts.run_training import load_experiment,preflight
    from scripts.training_ui import process_identity
    from openpi.training.data_loader import create_torch_dataset,transform_dataset,TorchDataLoader
    from openpi.training import reproducibility as rep
    from openpi.training.input_migration import fingerprint,check_code_change
    from openpi.training.frame_cache import sha256
    from openpi.training import sampling
    torch.set_num_threads(1)
    configure_inputs(DEFAULT_BUNDLE,PREPARATION)
    config=replace(load_experiment(MODULES['b']),resume=True)
    inputs=preflight(config)
    data=config.data.create(config.assets_dirs,config.model)
    os.environ.pop('PI05_RGB_FRAME_CACHE',None)
    original=create_torch_dataset(data,config.model.action_horizon,config.model,sampling_mode=config.sampling_mode)
    os.environ['PI05_RGB_FRAME_CACHE']=str(args.cache.resolve())
    cached=create_torch_dataset(data,config.model.action_horizon,config.model,sampling_mode=config.sampling_mode)
    old_inputs,new_inputs=transform_dataset(original,data),transform_dataset(cached,data)
    assert np.array_equal(original.ids,cached.ids)
    rows=[json.loads(line) for line in (Path(data.lerobot_datasets[0].root)/'meta/episodes.jsonl').read_text().splitlines()]
    offset=0;checked=0
    for ep,row in enumerate(rows):
        for frame in [0,row['length']//2,row['length']-1]:
            a,b=old_inputs[offset+frame],new_inputs[offset+frame]
            leaves_a,tree_a=jax.tree_util.tree_flatten(a);leaves_b,tree_b=jax.tree_util.tree_flatten(b)
            if tree_a != tree_b or any(not np.array_equal(x,y) for x,y in zip(leaves_a,leaves_b,strict=True)):
                raise ValueError(f'Model input changed at episode{ep} frame{frame}')
            checked+=1
        offset+=row['length']
        if (ep+1)%10==0:print('Exact inputs checked',ep+1,'/100',flush=True)
    loader=TorchDataLoader(new_inputs,256,num_workers=config.num_workers,seed=config.data_seed,
        framework='pytorch',sampling_mode=config.sampling_mode,num_train_steps=config.num_train_steps,
        num_batches=args.batches)
    loader.seek(4000)
    durations=[];last=time.monotonic()
    for index,_ in enumerate(loader):
        now=time.monotonic();durations.append(now-last);last=now
        print('Cached batch',index+1,'seconds',round(durations[-1],3),flush=True)
    if len(durations)<24:raise ValueError('Insufficient loader benchmark')
    steady=statistics.mean(durations[8:])
    if steady>=5.5:raise ValueError(f'Cached reader still too slow: {steady}s/batch')
    dataset,_=rep._source_identity(data,cached)
    plan=sampling.epoch_shuffle_indices(len(cached),seed=config.data_seed,num_samples=config.batch_size*config.num_train_steps)
    current={'sampler_version':sampling.SAMPLER_VERSION,'training':rep._training_contract(config,data),
        'dataset_sha256':__import__('hashlib').sha256(rep._json_bytes(dataset)).hexdigest(),
        'norm_sha256':__import__('hashlib').sha256(rep._json_bytes(data.norm_stats)).hexdigest(),
        'plan_indices_sha256':__import__('hashlib').sha256(plan.astype('<i8').tobytes()).hexdigest(),
        'code':rep._code_inventory(config)}
    previous=json.loads((Path(config.checkpoint_dir)/'reproducibility/manifest.json').read_text())['contract']
    changed=check_code_change(previous,current)
    report=dict(status='exact_input_acceleration_validated',checkpoint_step=4000,
        old_contract_sha256=fingerprint(previous),new_contract_sha256=fingerprint(current),
        changed_code_files=changed,cache_manifest_sha256=sha256(args.cache/'manifest.json'),
        cache_root=str(args.cache.resolve()),checkpoint_dir=str(config.checkpoint_dir),inputs=inputs,
        validation=dict(all_model_inputs_equal=True,episodes_checked=100,samples_checked=checked,
            loader_batches=len(durations),steady_batch_seconds=steady,batch_seconds=durations),
        previous_contract=previous,new_contract=current,training_started=False,
        old_pid=575579,old_identity=process_identity(575579))
    if report['old_identity'] is None:raise ValueError('Original process is no longer available')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS exact cached resume; steady seconds/batch',steady,flush=True)


if __name__=='__main__':main()
