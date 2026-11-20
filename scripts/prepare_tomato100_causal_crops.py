"""Generate isolated causal ROI CANDIDATES for frozen100; never approves quality or starts training."""
import argparse
import gzip
import json
import time
from pathlib import Path

from openpi.shared.local_storage import atomic_write_json, new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.tomato_roi_locator import CausalRoleSelector, RgbLocator
from scripts.tomato_interaction_preview import native_crop


def run(selection_path,locator_root,norm_root,out,episodes,batch_size=16,resume=False):
    import av
    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image,ImageDraw
    torch.set_num_threads(4);cv2.setNumThreads(1)
    selection=json.loads(selection_path.read_text())
    if selection.get('phase_label_resolution_required')!=0 or len(selection['episodes'])!=100:
        raise ValueError('Resolved frozen100 required')
    contract=dict(selection_sha256=digest(selection_path),locator_sha256=digest(locator_root/'model.pt'),
                  normalization_sha256=digest(norm_root/'norm_stats.json'),
                  generator_sha256=digest(Path(__file__)),selector_sha256=digest(Path(CausalRoleSelector.select.__code__.co_filename)))
    if resume:
        out=out.expanduser().resolve()
        if json.loads((out/'contract.json').read_text())!=contract:raise ValueError('Resume contract changed')
        # Reuse the standard output protection even when resuming an existing artifact root.
        derived_output_path(selection,out/'_resume_path_guard')
    else:
        out=derived_output_path(selection,out);out.mkdir(parents=True,exist_ok=False)
        atomic_write_json(out/'contract.json',contract)
    norm=json.loads((norm_root/'norm_stats.json').read_text())['norm_stats']['state']
    scale=np.array(norm['q99'])-np.array(norm['q01'])
    locator=RgbLocator(locator_root);completed=[]
    for ep in episodes:
        row=selection['episodes'][ep];target=out/f'episode_{ep:06d}'
        if target.exists():
            receipt=json.loads((target/'receipt.json').read_text())
            if receipt['contract']!=contract or receipt['frames']!=row['length'] or digest(target/'interaction224.mp4')!=receipt['video_sha256']:
                raise ValueError('Existing candidate episode changed')
            completed.append(receipt);continue
        prefix=f"chunk-{row['source_episode_index']//1000:03d}"
        video_rel=f"videos/{prefix}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
        data_rel=f"data/{prefix}/episode_{row['source_episode_index']:06d}.parquet"
        source=Path(row['dataset'])/video_rel;data=Path(row['dataset'])/data_rel
        if digest(source)!=row['source_sha256'][video_rel] or digest(data)!=row['source_sha256'][data_rel]:raise ValueError('Original source changed')
        states=np.array(pq.read_table(data,columns=['observation.state'])['observation.state'].to_pylist(),np.float32)
        if states.shape!=(row['length'],116):raise ValueError('Measured state/frame mismatch')
        selector=CausalRoleSelector(scale);valid_count=0;side=[];role_switches=0;previous_roles=None;previous_center=None;jumps=[]
        qa=set(range(0,row['length'],max(1,row['length']//28)))|{row['length']-1}
        qa.update((s['start_frame']+s['end_frame']-1)//2 for s in row['training_segments']) # QA sampling only
        qa_tiles=[];invalid_examples=0;frame_count=0;started=time.perf_counter();infer_times=[]
        with new_artifact_directory(target) as staging:
            with av.open(str(source)) as src,av.open(str(staging/'interaction224.mp4'),'w') as dst, \
                    gzip.open(staging/'index.jsonl.gz','wt') as index:
                if float(src.streams.video[0].average_rate)!=row['fps']:raise ValueError('Source FPS changed')
                stream=dst.add_stream('libx264',rate=row['fps']);stream.width,stream.height,stream.pix_fmt=224,224,'yuv420p'
                stream.options={'crf':'16','preset':'fast'};stream.thread_count=2
                pending=[]
                def process(batch):
                    nonlocal valid_count,frame_count,role_switches,previous_roles,previous_center,invalid_examples
                    torch.cuda.synchronize();t=time.perf_counter();predictions=locator.predict([im for _,im in batch]);torch.cuda.synchronize()
                    infer_times.append((time.perf_counter()-t)/len(batch))
                    for (frame,image),detections in zip(batch,predictions,strict=True):
                        h,w=image.shape[:2];roi=selector.select(detections,states[frame],w,h,rgb=image)
                        crop=np.zeros((224,224,3),np.uint8)
                        if roi['valid']:
                            valid_count+=1;box=roi['crop_box'];side.append(box[2]-box[0]);center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
                            if previous_center is not None:jumps.append(float(np.linalg.norm(center-previous_center)))
                            previous_center=center
                            if previous_roles is not None and previous_roles!=roi['roles']:role_switches+=1
                            previous_roles=roi['roles']
                            crop=cv2.resize(native_crop(image,box),(224,224),interpolation=cv2.INTER_AREA)
                        else:previous_center=None
                        record=dict(source_frame=frame,timestamp=frame/row['fps'],available_at=frame/row['fps'],
                            source_camera='cam0',source_size=[w,h],generation_mode='causal_prediction_candidate',
                            review_status='semantic_quality_unapproved',**roi)
                        index.write(json.dumps(record)+'\n')
                        for packet in stream.encode(av.VideoFrame.from_ndarray(crop,format='rgb24')):dst.mux(packet)
                        if frame in qa or (not roi['valid'] and invalid_examples<3):
                            overlay=Image.fromarray(image);draw=ImageDraw.Draw(overlay)
                            if roi['valid']:
                                draw.rectangle(roi['interaction_box'],outline='#ffb000',width=2);draw.rectangle(roi['crop_box'],outline='#00e0d0',width=2)
                            tile=Image.new('RGB',(544,230),'#151515');overlay.thumbnail((320,200));tile.paste(overlay,(0,30));tile.paste(Image.fromarray(crop),(320,0))
                            ImageDraw.Draw(tile).text((5,5),f"ep{ep} f{frame} valid={roi['valid']} roles={roi['roles']}",fill='white')
                            qa_tiles.append(tile)
                            if not roi['valid']:invalid_examples+=1
                        frame_count+=1
                for i,frame in enumerate(src.decode(video=0)):
                    pending.append((i,frame.to_ndarray(format='rgb24')))
                    if len(pending)==batch_size:process(pending);pending=[]
                if pending:process(pending)
                for packet in stream.encode():dst.mux(packet)
            if frame_count!=row['length'] or digest(source)!=row['source_sha256'][video_rel] or digest(data)!=row['source_sha256'][data_rel]:raise ValueError('Source/frame recheck failed')
            with av.open(str(staging/'interaction224.mp4')) as verify:
                if sum(1 for _ in verify.decode(video=0))!=frame_count:raise ValueError('Encoded frame mismatch')
            for start in range(0,len(qa_tiles),12):
                page=qa_tiles[start:start+12];sheet=Image.new('RGB',(1088,230*((len(page)+1)//2)))
                for j,tile in enumerate(page):sheet.paste(tile,((j%2)*544,(j//2)*230))
                sheet.save(staging/f'qa_{start//12:02d}.jpg',quality=90)
            receipt=dict(contract=contract,episode_index=ep,frames=frame_count,algorithm_valid_frames=valid_count,
                algorithm_invalid_frames=frame_count-valid_count,valid_fraction=valid_count/frame_count,
                source_video_sha256=row['source_sha256'][video_rel],source_parquet_sha256=row['source_sha256'][data_rel],source_hashes_unchanged=True,
                video_sha256=digest(staging/'interaction224.mp4'),index_sha256=digest(staging/'index.jsonl.gz'),qa_images=len(qa_tiles),
                role_switches=role_switches,max_center_step_px=max(jumps,default=0),crop_side_p50=float(np.median(side)) if side else None,
                batched_detector_amortized_ms_p50=float(np.median(infer_times))*1000,
                detector_batch_size=batch_size,not_single_frame_latency=True,uses_phase_labels_for_roi=False,
                uses_future_frames_for_roi=False,independent_quality_approved=False,full_roi_coverage=False,training_ready=False,
                elapsed_seconds=time.perf_counter()-started,source_writes=False)
            atomic_write_json(staging/'receipt.json',receipt)
        completed.append(receipt)
        atomic_write_json(out/'progress.json',dict(contract=contract,episodes_completed=len(completed),frames_processed=sum(r['frames'] for r in completed),
            algorithm_valid_frames=sum(r['algorithm_valid_frames'] for r in completed),training_ready=False))
        print(f"ep{ep}: {frame_count} frames, valid {valid_count/frame_count:.1%}, {receipt['elapsed_seconds']:.1f}s",flush=True)
    final=dict(contract=contract,episodes=completed,episodes_completed=len(completed),frames=sum(r['frames'] for r in completed),
        all100_frames_enumerated=len(completed)==100,algorithm_valid_frames=sum(r['algorithm_valid_frames'] for r in completed),
        full_roi_coverage=False,independent_quality_approved=False,training_ready=False,training_started=False,source_writes=False,
        note='Every invalid frame has a black crop and false mask; current-only detector supports fresh reacquisition. Independent semantic/temporal QA and final training contract still required.')
    atomic_write_json(out/'candidate_manifest.json',final)
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--norm',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',default='all');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--resume',action='store_true')
    a=p.parse_args();episodes=list(range(100)) if a.episodes=='all' else [int(i) for i in a.episodes.split(',')]
    if not episodes or len(set(episodes))!=len(episodes) or any(not 0<=i<100 for i in episodes) or a.batch_size<1:p.error('Unique episodes 0..99 and positive batch size required')
    result=run(a.selection,a.locator,a.norm,a.out,episodes,a.batch_size,a.resume)
    print('Candidate frames',result['frames'],'quality approval remains false',flush=True)


if __name__=='__main__':main()
