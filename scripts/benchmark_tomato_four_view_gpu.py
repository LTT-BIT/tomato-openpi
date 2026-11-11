"""Profile full pi05 3/4-view architecture on real RGB/state; no optimization or quality claim.

Released base weights use32 actions. Only mismatched58D projection parameters
are initialized for this structural benchmark and recorded explicitly.
"""
import argparse
import json
import time
from pathlib import Path


def main():
    import av
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict,unflatten_dict
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pyarrow.parquet as pq
    import sentencepiece
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import nnx_utils,normalize
    from openpi.shared.local_storage import new_artifact_directory
    from openpi.subtasks.rgb100 import IMAGE_KEYS,derived_output_path,digest,full_text
    from openpi_client import image_tools
    from scripts.tomato_interaction_preview import interaction_at,native_crop,square_box
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--params',type=Path,required=True);p.add_argument('--norm',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=30);p.add_argument('--check-gradients',action='store_true');a=p.parse_args()
    if jax.default_backend()!='gpu':raise RuntimeError('Real CUDA backend required')
    selection=json.loads(a.selection.read_text());row=selection['episodes'][86];spec=json.loads(a.annotations.read_text())
    out=derived_output_path(selection,a.out);frame_id=83;images={}
    for camera,key in [('cam0','base_0_rgb'),('cam2','left_wrist_0_rgb'),('cam1','right_wrist_0_rgb')]:
        rel=f"videos/chunk-{row['source_episode_index']//1000:03d}/observation.images.{camera}/episode_{row['source_episode_index']:06d}.mp4"
        source=Path(row['dataset'])/rel
        if digest(source)!=row['source_sha256'][rel]:raise ValueError('Source changed')
        with av.open(str(source)) as c:
            for i,f in enumerate(c.decode(video=0)):
                if i==frame_id:images[key]=f.to_ndarray(format='rgb24');break
    clip=next(c for c in spec['clips'] if c['start']<=frame_id<c['end'])
    raw=images['base_0_rgb'];h,w=raw.shape[:2]
    images['interaction_0_rgb']=native_crop(raw,square_box(interaction_at(clip,frame_id,w,h),w,h))
    images={k:(image_tools.resize_with_pad(im,224,224).astype(np.float32)/127.5-1)[None] for k,im in images.items()}
    rel=f"data/chunk-{row['source_episode_index']//1000:03d}/episode_{row['source_episode_index']:06d}.parquet"
    q=np.array(pq.read_table(Path(row['dataset'])/rel,columns=['observation.state'])['observation.state'][frame_id].as_py(),np.float32)
    q=np.r_[q[:7],q[28:50],q[14:21],q[72:94]]
    stats=normalize.load(a.norm);state=transforms.Normalize({'state':stats['state']},use_quantiles=True)({'state':q})['state']
    bins=(np.digitize(state,np.linspace(-1,1,257)[:-1])-1).tolist()
    prompt=next(s['text'] for s in row['training_segments'] if s['start_frame']<=frame_id<s['end_frame'])
    sp=sentencepiece.SentencePieceProcessor(model_file=str(a.tokenizer));tokens=sp.encode(full_text(prompt,bins),add_bos=True)
    if len(tokens)>256:raise ValueError('Real prompt exceeds capacity')
    token_count=len(tokens);mask=np.arange(256)<token_count;tokens=np.pad(np.array(tokens,np.int32),(0,256-token_count))
    configs={str(n):Pi0Config(pi05=True,action_dim=58,max_token_len=256,image_keys=IMAGE_KEYS[:3] if n==3 else IMAGE_KEYS) for n in [3,4]}
    print('Loading full released base parameters',flush=True)
    params=model_lib.restore_params(a.params,restore_type=np.ndarray,dtype=jnp.bfloat16)
    shapes=nnx.state(nnx.eval_shape(configs['4'].create,jax.random.key(0))).to_pure_dict()
    reference=flatten_dict(shapes);loaded=flatten_dict(params);adapted=[];rng=np.random.default_rng(42)
    merged={}
    for key,shape in reference.items():
        value=loaded.get(key)
        if value is not None and value.shape==shape.shape:merged[key]=value
        elif any(name in '/'.join(key) for name in ['action_in_proj','action_out_proj','state_proj']):
            merged[key]=(rng.normal(0,.02,shape.shape) if len(shape.shape)>1 else np.zeros(shape.shape)).astype(shape.dtype)
            adapted.append(dict(key='/'.join(key),from_shape=list(value.shape) if value is not None else None,to_shape=list(shape.shape)))
        else:raise ValueError(f'Unexpected missing/mismatched full model parameter: {key}')
    params=jax.tree.map(jnp.asarray,unflatten_dict(merged));jax.block_until_ready(params)
    print('Full model parameters ready; profiling3 then4 views',flush=True)
    gradient_check=None
    results={};noise=jnp.asarray(rng.normal(size=(1,50,58)).astype(np.float32));key=jax.random.key(42)
    with new_artifact_directory(out) as staging:
        for n,config in configs.items():
            net=config.load(params);net.eval();predict=nnx_utils.module_jit(net.sample_actions)
            observation=model_lib.Observation(images={k:jnp.asarray(images[k]) for k in config.image_keys},
                image_masks={k:jnp.array([True]) for k in config.image_keys},state=jnp.asarray(state[None],dtype=jnp.float32),
                tokenized_prompt=jnp.array(tokens[None]),tokenized_prompt_mask=jnp.array(mask[None]))
            start=time.perf_counter();action=predict(key,observation,noise=noise);action.block_until_ready();cold=time.perf_counter()-start
            for _ in range(3):predict(key,observation,noise=noise).block_until_ready()
            times=[]
            for _ in range(a.samples):
                start=time.perf_counter();action=predict(key,observation,noise=noise);action.block_until_ready();times.append(time.perf_counter()-start)
            if action.shape!=(1,50,58) or not np.isfinite(np.asarray(action)).all():raise ValueError('Full sample shape/finite check failed')
            results[n]=dict(views=int(n),visual_tokens=int(n)*256,text_slots=256,action_shape=list(action.shape),
                denoising_steps=10,samples=len(times),cold_compile_and_first_call_seconds=cold,
                p50_ms=float(np.percentile(times,50))*1000,p95_ms=float(np.percentile(times,95))*1000,
                p99_ms=float(np.percentile(times,99))*1000,max_ms=float(max(times))*1000)
            print(json.dumps(results[n]),flush=True)
            if n=='4' and a.check_gradients:
                table=pq.read_table(Path(row['dataset'])/rel,columns=['action'])
                raw_actions=np.array(table['action'].to_pylist(),np.float32)[np.minimum(frame_id+np.arange(50),row['length']-1)]
                targets=np.c_[raw_actions[:,:7],raw_actions[:,14:36],raw_actions[:,7:14],raw_actions[:,36:58]]
                targets[:,:7]-=q[:7];targets[:,29:36]-=q[29:36]
                targets=transforms.Normalize({'actions':stats['actions']},use_quantiles=True)({'actions':targets})['actions']
                targets=jnp.asarray(targets[None],jnp.float32)
                graphdef,model_state=nnx.split(net)
                def loss_fn(st):
                    module=nnx.merge(graphdef,st)
                    return module.compute_loss(key,observation,targets,train=True).mean()
                def summarize(st):
                    loss,grad_state=jax.value_and_grad(loss_fn,allow_int=True)(st)
                    leaves=[g for g in jax.tree.leaves(grad_state) if jnp.issubdtype(g.dtype,jnp.inexact)]
                    finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in leaves]))
                    squared=sum(jnp.sum(g.astype(jnp.float32)**2) for g in leaves)
                    return loss,finite,jnp.sqrt(squared)
                def crop_loss(st,crop):
                    module=nnx.merge(graphdef,st)
                    obs=observation.replace(images={**observation.images,'interaction_0_rgb':crop})
                    # Image augmentation itself is not optimized. Test the network
                    # input gradient without differentiating HSV augmentation.
                    return module.compute_loss(key,obs,targets,train=False).mean()
                print('Full four-view parameter and crop backward diagnostic',flush=True)
                start=time.perf_counter();values=jax.jit(summarize)(model_state)
                jax.block_until_ready(values)
                loss,finite,norm=map(np.asarray,values)
                crop_gradient=jax.jit(jax.grad(crop_loss,argnums=1))(model_state,observation.images['interaction_0_rgb'])
                crop_gradient.block_until_ready();crop_gradient=np.asarray(crop_gradient)
                crop_l1=float(np.abs(crop_gradient).sum())
                if not finite or not np.isfinite(loss) or not np.isfinite(norm) or not np.isfinite(crop_gradient).all() or not np.isfinite(crop_l1) or crop_l1<=0:
                    raise ValueError('Full backward check failed')
                gradient_check=dict(batch_size=1,loss=float(loss),all_parameter_gradients_finite=bool(finite),
                    parameter_gradient_l2=float(norm),fourth_image_gradient_l1=crop_l1,
                    parameter_gradient_augmentation=True,crop_input_gradient_augmentation=False,
                    compile_plus_run_seconds=time.perf_counter()-start,optimizer_step_performed=False,
                    training_batch512_memory_accepted=False)
                print(json.dumps(gradient_check),flush=True)
            del net,predict
        receipt=dict(device=str(jax.devices()[0]),device_kind=jax.devices()[0].device_kind,jax_version=jax.__version__,
            selection_sha256=digest(a.selection),annotation_sha256=digest(a.annotations),normalization_sha256=digest(a.norm/'norm_stats.json'),
            tokenizer_sha256=digest(a.tokenizer),weights=str(a.params),actual_text_tokens=token_count,frame_id=frame_id,episode_index=86,
            adapted_projection_parameters=adapted,scope='Full architecture latency only.58D projections initialized; no claim of trained four-view behavior.',
            measurements=results,gradient_check=gradient_check,four_vs_three_p50_ratio=results['4']['p50_ms']/results['3']['p50_ms'],
            memory_stats={k:int(v) for k,v in jax.devices()[0].memory_stats().items()},
            includes_decode_or_roi=False,training_started=False,optimizer_step_performed=False,source_writes=False,
            deployment_cadence_accepted=False)
        (staging/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
