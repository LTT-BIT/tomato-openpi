"""Validate the three-camera subtask control against original pixels and the running crop design."""
import json
import os
from pathlib import Path
import statistics
import time

from openpi.subtasks import tomato_in_hand as c
from openpi.subtasks.prepare import digest,write_json
from training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_three_rgb import MODULE,REPORT


def release_frame_maps(*datasets):
    """Bound audit file descriptors while comparing multiple full camera sets."""
    from openpi.training.data_loader import TransformedDataset
    for dataset in datasets:
        while isinstance(dataset,TransformedDataset):dataset=dataset._dataset
        dataset._cache_maps.clear()


def check():
    import jax
    import numpy as np
    import torch
    from openpi.training.data_loader import create_torch_dataset,transform_dataset,TorchDataLoader
    from openpi.training import reproducibility
    from scripts.run_training import load_experiment,preflight,validate_config
    torch.set_num_threads(1)
    config=load_experiment(MODULE);validate_config(config);inputs=preflight(config)
    root=Path(config.policy_metadata['dataset_root']);view=json.loads((root/'view.json').read_text())
    bundle=Path(view['source_bundle']);manifest=json.loads((bundle/'bundle.json').read_text())
    for rel,sha in view['physical_sha256'].items():
        if digest(root/rel)!=sha or digest(bundle/'b_coarse'/rel)!=sha:raise ValueError('Physical/task data changed')
    for rel,sha in manifest['files'].items():
        if digest(bundle/rel)!=sha:raise ValueError('Original four-view source changed')
    from scripts.check_tomato_in_hand204 import MODULE as CROP_MODULE
    crop=load_experiment(CROP_MODULE);crop_inputs=preflight(crop)
    if inputs['norm_stats_sha256']!=crop_inputs['norm_stats_sha256']:
        raise ValueError('Removing RGB must not change the state/action normalization values')
    data=config.data.create(config.assets_dirs,config.model)
    os.environ.pop('PI05_RGB_FRAME_CACHE',None)
    raw=create_torch_dataset(data,50,config.model,sampling_mode=config.sampling_mode)
    cache=Path(config.policy_metadata['rgb_frame_cache'])
    os.environ['PI05_RGB_FRAME_CACHE']=str(cache)
    cached=create_torch_dataset(data,50,config.model,sampling_mode=config.sampling_mode)
    crop_data=crop.data.create(crop.assets_dirs,crop.model)
    os.environ['PI05_RGB_FRAME_CACHE']=crop.policy_metadata['rgb_frame_cache']
    four_cached=create_torch_dataset(crop_data,50,crop.model,sampling_mode=crop.sampling_mode)
    originals,three,four=transform_dataset(raw,data),transform_dataset(cached,data),transform_dataset(four_cached,crop_data)
    def equal(a,b):
        la,ta=jax.tree_util.tree_flatten(a);lb,tb=jax.tree_util.tree_flatten(b)
        if ta!=tb:raise ValueError('Input structure differs')
        for x,y in zip(la,lb,strict=True):
            if np.asarray(x).dtype!=np.asarray(y).dtype or not np.array_equal(x,y):raise ValueError('Pixel/state/action/token differs')
    offset=checked=endpoints=0;lengths=[]
    selection=json.loads((bundle/'selection.json').read_text())
    for row in selection['episodes']:
        ep,length=row['episode_index'],row['length']
        doc=json.loads((bundle/f'b_coarse/annotations/episode_{ep:06d}.json').read_text())
        for segment in c.reviewed_segments(doc,length):
            for frame in [segment['start_frame'],segment['end_frame']-1]:
                if cached[offset+frame]['prompt']!=segment['text']:raise ValueError('Subtask differs')
                endpoints+=1
        for frame in [0,length//2,length-1]:
            a=three[offset+frame];b=four[offset+frame]
            equal(a,originals[offset+frame])
            b={**b,'image':{k:v for k,v in b['image'].items() if k!='interaction_0_rgb'},
                'image_mask':{k:v for k,v in b['image_mask'].items() if k!='interaction_0_rgb'}}
            equal(a,b)
            if set(a['image'])!=set(config.model.image_keys) or len(a['image'])!=3:raise ValueError('Not exactly three RGB inputs')
            lengths.append(int(np.asarray(a['tokenized_prompt_mask']).sum()));checked+=1
        offset+=length
        release_frame_maps(cached,four_cached)
        if (ep+1)%10==0:print('Three RGB exact inputs',ep+1,'/204',flush=True)
    os.environ['PI05_RGB_FRAME_CACHE']=str(cache)
    loader=TorchDataLoader(three,config.batch_size,num_workers=config.num_workers,seed=config.data_seed,
        framework='pytorch',sampling_mode=config.sampling_mode,num_train_steps=config.num_train_steps,num_batches=32)
    durations=[];last=time.monotonic()
    for index,_ in enumerate(loader):
        now=time.monotonic();durations.append(now-last);last=now;print('Batch',index+1,durations[-1],flush=True)
    if len(durations)!=32 or statistics.mean(durations[8:])>=5.5:raise ValueError('Insufficient loader throughput')
    report=dict(status='three_rgb204_accepted',training_started=False,episodes=c.EPISODES,frames=offset,
        model_samples_checked=checked,phase_boundary_samples_checked=endpoints,
        all_original_pixels_equal=True,only_removed_interaction=True,normalization_values_equal=True,
        inputs=inputs,view_sha256=digest(root/'view.json'),cache_manifest_sha256=digest(cache/'manifest.json'),
        code_inventory=reproducibility._code_inventory(config),steady_batch_seconds=statistics.mean(durations[8:]),
        actual_token_lengths={'min':min(lengths),'max':max(lengths)},loader_batches=32,
        visual_tokens=768,text_limit=256,raw_files_deleted=False,raw_files_modified=False)
    out=Path(config.policy_metadata['three_view_acceptance_report'])
    if out.exists():raise FileExistsError(out)
    write_json(out,report);print('PASS three RGB subtask control',flush=True)


if __name__=='__main__':
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']='';check()
