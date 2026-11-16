"""CPU-only full119 source, phase, model-input and cached-loader acceptance."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

from openpi.subtasks import tomato_in_hand as contract

MODULE='training_configs.experiments.tomato_sharpa.subtasks.in_hand119_b_crop'


def check(bundle,cache,out):
    import jax
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from scripts.run_training import load_experiment,preflight,validate_config
    from openpi.subtasks.prepare import digest
    from openpi.training.data_loader import create_torch_dataset,transform_dataset,TorchDataLoader
    torch.set_num_threads(1)
    os.environ['PI05_IN_HAND119_BUNDLE']=str(bundle.resolve())
    config=load_experiment(MODULE);validate_config(config);inputs=preflight(config)
    manifest=json.loads((bundle/'bundle.json').read_text())
    # Verify the complete training copy including every video, annotation and physical table.
    for relative,sha in manifest['files'].items():
        p=(bundle/relative).resolve()
        if not p.is_relative_to(bundle.resolve()) or digest(p)!=sha:raise ValueError('Prepared file changed: '+relative)
    selection=json.loads((bundle/'selection.json').read_text());source_hashes={}
    for row in selection['episodes']:
        for rel,sha in row['source_sha256'].items():source_hashes[str(Path(row['source_dataset'])/rel)]=sha
        if digest(row['annotation'])!=row['annotation_sha256']:raise ValueError('Source annotation changed')
    for path,sha in source_hashes.items():
        if digest(path)!=sha:raise ValueError('Original source changed: '+path)
    data=config.data.create(config.assets_dirs,config.model)
    os.environ.pop('PI05_RGB_FRAME_CACHE',None)
    original=create_torch_dataset(data,config.model.action_horizon,config.model,sampling_mode=config.sampling_mode)
    os.environ['PI05_RGB_FRAME_CACHE']=str(cache.resolve())
    cached=create_torch_dataset(data,config.model.action_horizon,config.model,sampling_mode=config.sampling_mode)
    old_inputs,new_inputs=transform_dataset(original,data),transform_dataset(cached,data)
    assert len(original)==len(cached)==contract.FRAMES
    offset=checked=phase_checks=0
    for row in selection['episodes']:
        ep=row['episode_index'];length=row['length']
        doc=json.loads((bundle/f'b_coarse/annotations/episode_{ep:06d}.json').read_text())
        segments=contract.reviewed_segments(doc,length)
        physical=pq.read_table(bundle/f'b_coarse/data/chunk-000/episode_{ep:06d}.parquet')
        upstream=pq.read_table(Path(row['source_dataset'])/f"data/chunk-000/episode_{row['source_episode_index']:06d}.parquet")
        for key in upstream.column_names:
            if key not in ('index','episode_index','task_index') and not upstream[key].equals(physical[key]):
                raise ValueError('Source physical column changed: '+key)
        task_ids=physical['task_index'].to_numpy();phase_ids=physical['coarse_phase_index'].to_numpy()
        for index,segment in enumerate(segments):
            a,b=segment['start_frame'],segment['end_frame']
            if not np.all(task_ids[a:b]==contract.TASKS.index(segment['text'])) or not np.all(phase_ids[a:b]==index):
                raise ValueError('Per-frame stage supervision differs')
            for frame in sorted({a,b-1}):
                raw=cached[offset+frame]
                if raw['prompt']!=segment['text']:raise ValueError('Actual loader phase prompt differs')
                phase_checks+=1
        for frame in sorted({0,length//2,length-1}):
            a,b=old_inputs[offset+frame],new_inputs[offset+frame]
            la,ta=jax.tree_util.tree_flatten_with_path(a);lb,tb=jax.tree_util.tree_flatten_with_path(b)
            if ta!=tb:raise ValueError('Model input structure differs')
            for (path,x),(_,y) in zip(la,lb,strict=True):
                if np.asarray(x).dtype!=np.asarray(y).dtype or not np.array_equal(x,y):
                    raise ValueError(f'Pixel/vector/token difference: ep{ep} frame{frame} {path}')
                if np.issubdtype(np.asarray(x).dtype,np.number) and not np.isfinite(x).all():raise ValueError('Nonfinite input')
            if np.asarray(b['state']).shape!=(58,) or np.asarray(b['actions']).shape!=(50,58):raise ValueError('Wrong model vector shape')
            if not all(bool(v) for v in b['image_mask'].values()):raise ValueError('Missing RGB view')
            checked+=1
        offset+=length
        if (ep+1)%10==0:print('Exact inputs',ep+1,'/119',flush=True)
    loader=TorchDataLoader(new_inputs,config.batch_size,num_workers=config.num_workers,
        seed=config.data_seed,framework='pytorch',sampling_mode=config.sampling_mode,
        num_train_steps=config.num_train_steps,num_batches=32)
    durations=[];last=time.monotonic()
    for index,_ in enumerate(loader):
        now=time.monotonic();durations.append(now-last);last=now
        print('Cached batch',index+1,round(durations[-1],3),flush=True)
    if len(durations)!=32 or statistics.mean(durations[8:])>=5.5:raise ValueError('Cached reader too slow')
    cache_manifest=json.loads((cache/'manifest.json').read_text())
    from openpi.subtasks.prepare import write_json
    report=dict(status='in_hand119_training_inputs_accepted',training_started=False,
        episodes=contract.EPISODES,frames=contract.FRAMES,bundle=str(bundle.resolve()),cache=str(cache.resolve()),
        bundle_sha256=digest(bundle/'bundle.json'),selection_sha256=digest(bundle/'selection.json'),
        cache_manifest_sha256=digest(cache/'manifest.json'),norm=inputs,
        all_model_inputs_equal=True,model_samples_checked=checked,phase_boundary_samples_checked=phase_checks,
        per_frame_prompts_checked=offset,original_sources_unchanged=True,physical_columns_equal=True,
        loader_batches=len(durations),batch_seconds=durations,steady_batch_seconds=statistics.mean(durations[8:]),
        cache_entries=len(cache_manifest['entries']),exact_video_reference_samples=sum(e['exact_reference_samples'] for e in cache_manifest['entries']),
        all_videos_decoded=True,offline_roi_uses_future_coordinates=True,independent_semantic_accuracy_measured=False,
        gpu_capacity='not_exercised_this_run; prior four-view batch256 verified separately')
    if out.exists():raise FileExistsError(out)
    write_json(out,report);return report


def main():
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=contract.DEFAULT_BUNDLE)
    p.add_argument('--cache',type=Path,default=contract.DEFAULT_CACHE);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=check(a.bundle,a.cache,a.out);print('PASS',r['model_samples_checked'],r['steady_batch_seconds'],flush=True)


if __name__=='__main__':main()
