"""Repair offline local ROI coordinates from adjacent RGB; never substitutes pixels."""
import argparse
import collections
import concurrent.futures
import gzip
import json
from pathlib import Path
import numpy as np
from openpi.shared.local_storage import atomic_write_json,new_artifact_directory
from openpi.subtasks.rgb100 import digest,derived_output_path
from scripts.materialize_tomato100_four_view import collect_candidates,read_candidate_index
from scripts.tomato_offline_roi import track,union,plan_windows


def seed(detections,role,indices,width,height):
    for i in indices:
        items=[d for d in detections[i]['detections'] if d['label']==role and d['score']>=.75
               and min(np.asarray(d['box'])[2:]-np.asarray(d['box'])[:2])>=15
               and np.prod(np.asarray(d['box'])[2:]-np.asarray(d['box'])[:2])<=width*height*.13]
        if items:return i,max(items,key=lambda d:d['score'])['box']
    return None


def process(job):
    import av,cv2
    from scripts.tomato_stable_roi import red_regions,gap
    ep,row,folder,receipt,out,contract=job;cv2.setNumThreads(1)
    if digest(folder/'index.jsonl.gz')!=receipt['index_sha256'] or digest(folder/'detections.jsonl.gz')!=receipt['detections_sha256']:
        raise ValueError('Parent annotations changed')
    rows=read_candidate_index(folder/'index.jsonl.gz',row['length'])
    with gzip.open(folder/'detections.jsonl.gz','rt') as f:detections=[json.loads(l) for l in f]
    n=len(rows);w,h=rows[0]['source_size'];runs=[];start=None
    for i,r in enumerate(rows+[{'observed_interaction_box':True}]):
        if r.get('observed_interaction_box') is None:
            if start is None:start=i
        elif start is not None:runs.append((start,i));start=None
    original_roles=[r['roles'] for r in rows]
    for start,end in runs:
        # Short role glitches during detection loss must not redirect the crop
        # to a different hand when both observed endpoints agree.
        if end-start<=15 and start>0 and end<n and rows[start-1]['roles']==rows[end]['roles']:
            for i in range(start,end):rows[i]={**rows[i],'roles':rows[start-1]['roles']}
    needed=set()
    for start,end in runs:needed.update(range(max(0,start-30),min(n,end+30)))
    video_rel=f"videos/chunk-{row['source_episode_index']//1000:03d}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
    source=Path(row['dataset'])/video_rel
    if digest(source)!=row['source_sha256'][video_rel]:raise ValueError('Original source changed')
    rgb={};gray={}
    if needed:
        with av.open(str(source)) as src:
            for i,frame in enumerate(src.decode(video=0)):
                if i in needed:
                    pixels=frame.to_ndarray(format='rgb24');gray[i]=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
                    if rows[i]['observed_interaction_box'] is None:rgb[i]=pixels
                if i>=max(needed):break
    results={};pending=[];run_reports=[]
    for start,end in runs:
        roles=sorted(set(role for r in rows[start:end] for role in r['roles']))
        role_data={};support=[]
        for role in roles:
            left=seed(detections,role,range(start-1,max(-1,start-31),-1),w,h)
            right=seed(detections,role,range(end,min(n,end+30)),w,h)
            forward={} if left is None else track(gray,left[0],left[1],range(left[0]+1,end))
            backward={} if right is None else track(gray,right[0],right[1],range(right[0]-1,start-1,-1))
            support.extend([] if left is None else [left[0]]);support.extend([] if right is None else [right[0]])
            role_data[role]=(left,right,forward,backward)
        for i in range(start,end):
            boxes=[];methods=[];verified=True;disagreements=[]
            for role in rows[i]['roles']:
                left,right,forward,backward=role_data[role]
                current=seed(detections,role,[i],w,h)
                if current is not None and rows[i]['hand_evidence'].get(str(role))!='lost':
                    boxes.append(np.array(current[1]));methods.append('current_detector');continue
                f,b=forward.get(i),backward.get(i)
                if f is not None and b is not None:
                    disagreement=float(np.linalg.norm((f[:2]+f[2:]-b[:2]-b[2:])/2));disagreements.append(disagreement)
                    overlap=np.prod(np.maximum(0,np.minimum(f[2:],b[2:])-np.maximum(f[:2],b[:2])))
                    # Both tracks must describe substantially the same region;
                    # retain their union rather than shrink to their average.
                    verified &= overlap>=.5*min(np.prod(f[2:]-f[:2]),np.prod(b[2:]-b[:2]))
                    boxes.append(union([f,b]));methods.append('bidirectional_current_rgb_flow')
                elif f is not None and (i-left[0]<=15 or right is None):
                    boxes.append(f);methods.append('forward_current_rgb_flow')
                elif b is not None and (right[0]-i<=15 or left is None):
                    boxes.append(b);methods.append('backward_current_rgb_flow')
                else:
                    # Provisional coordinates only; never silently certify interpolation.
                    verified=False;methods.append('unverified_endpoint_interpolation')
                    if left is not None and right is not None:
                        alpha=(i-left[0])/(right[0]-left[0]);box=(1-alpha)*np.array(left[1])+alpha*np.array(right[1])
                    elif left is not None:box=np.array(left[1])
                    elif right is not None:box=np.array(right[1])
                    else:raise ValueError(f'No semantic anchor for ep{ep} frame{i} role{role}')
                    boxes.append(box)
            hands=boxes.copy()
            boxes += [b for b in red_regions(rgb[i]) if any(gap(b,hand)<.12*max(hand[2:]-hand[:2]) for hand in hands)]
            tight=union(boxes);tight[:2]=np.maximum(tight[:2],0);tight[2:]=np.minimum(tight[2:],[w,h])
            results[i]=dict(tight=tight.tolist(),repair_verified=bool(verified),repair_methods=methods,
                            support_frames=sorted(set(support)),bidirectional_center_disagreement_px=disagreements)
            if not verified:pending.append(i)
        run_reports.append(dict(start=start,end=end,frames=end-start,pending_frames=sum(start<=i<end for i in pending)))
    tight=np.array([results[i]['tight'] if i in results else r['observed_interaction_box'] for i,r in enumerate(rows)])
    try:windows,metrics=plan_windows(tight,w,h)
    except ValueError:
        atomic_write_json(out/f'geometry_failure_ep{ep}.json',dict(episode_index=ep,tight=tight.tolist(),pending=pending))
        raise
    target=out/f'shard{ep%4}'/f'episode_{ep:06d}'
    with new_artifact_directory(target) as staging:
        with gzip.open(staging/'index.jsonl.gz','wt') as f:
            for i,r in enumerate(rows):
                repair=results.get(i);support=repair['support_frames'] if repair else [i]
                record=dict(source_frame=i,timestamp=i/30,available_at=(n-1)/30,
                            source_camera='cam0',source_size=[w,h],generation_mode='offline_bidirectional_annotation',
                            review_status='independent_semantic_quality_unapproved',valid=i not in pending,
                            image_available=True,local_roi_valid=i not in pending,roles=r['roles'],
                            interaction_box=tight[i].tolist(),crop_box=windows[i].tolist(),display_crop_box=windows[i].tolist(),
                            input_mode='offline_local_crop',input_box=windows[i].tolist(),
                            original_valid=r['valid'],original_reason=r['reason'],original_roles=original_roles[i],
                            repair=repair,annotation_support_frames=support,window_support_range=[0,n-1],
                            uses_future_frames_for_roi=i<n-1)
                f.write(json.dumps(record)+'\n')
        receipt=dict(contract=contract,episode_index=ep,frames=n,algorithm_valid_frames=n-len(pending),
                     algorithm_invalid_frames=len(pending),image_available_frames=n,full_view_fallback_frames=0,
                     repaired_missing_frames=len(results),pending_frames=pending,missing_runs=run_reports,
                     geometry=metrics,index_sha256=digest(staging/'index.jsonl.gz'),source_video_sha256=digest(source),
                     parent_receipt_sha256=digest(folder/'receipt.json'),source_writes=False,training_ready=False)
        atomic_write_json(staging/'receipt.json',receipt)
    return receipt


def run(selection_path,parent,out,episodes,workers):
    selection=json.loads(selection_path.read_text());parent_contract,parents=collect_candidates(parent)
    out=derived_output_path(selection,out);out.mkdir(parents=True,exist_ok=False)
    contract=dict(stream_contract='offline_bidirectional_local_rgb_v6',selection_sha256=digest(selection_path),
                  generator_sha256=digest(Path(__file__)),geometry_sha256=digest(Path(plan_windows.__code__.co_filename)),
                  parent_contract=parent_contract,uses_future_frames_for_roi=True,offline_annotation_only=True,
                  pixel_source='same_current_native_frame',training_ready=False)
    atomic_write_json(out/'contract.json',contract)
    jobs=[(ep,selection['episodes'][ep],*parents[ep],out,contract) for ep in episodes]
    completed=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(process,jobs):
            completed.append(r);print(f"ep{r['episode_index']}: missing repair {r['repaired_missing_frames']}, pending {len(r['pending_frames'])}, geometry {r['geometry']}",flush=True)
    result=dict(contract=contract,episodes=len(completed),frames=sum(r['frames'] for r in completed),
                pending_frames=sum(len(r['pending_frames']) for r in completed),
                geometry_all_contained=all(r['geometry']['all_boxes_contained'] for r in completed),
                center_steps_over_3pct=sum(r['geometry']['center_steps_over_3pct'] for r in completed),
                scale_steps_over_2pct=sum(r['geometry']['scale_steps_over_2pct'] for r in completed),
                receipts=completed,training_ready=False,source_writes=False)
    atomic_write_json(out/'annotation_manifest.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',default='all');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    episodes=list(range(100)) if a.episodes=='all' else [int(i) for i in a.episodes.split(',')]
    run(a.selection,a.parent,a.out,episodes,a.workers)
