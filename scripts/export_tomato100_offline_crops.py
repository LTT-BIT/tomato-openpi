"""Export reviewed offline temporal annotations using each frame's OWN RGB pixels."""
import argparse
import collections
import concurrent.futures
import gzip
import json
from pathlib import Path
import numpy as np
from openpi.shared.local_storage import atomic_write_json,new_artifact_directory
from openpi.subtasks.rgb100 import digest,derived_output_path
from scripts.materialize_tomato100_four_view import read_candidate_index
from scripts.tomato_offline_roi import plan_windows,union
from scripts.tomato_interaction_preview import native_crop


def finalize_rows(rows,reviews,right_regions):
    pending={r['source_frame'] for r in rows if not r['valid']}
    if set(reviews)!=pending:raise ValueError('Every exception requires exactly one explicit visual review')
    tight=np.array([r['interaction_box'] for r in rows],float)
    for i,review in reviews.items():
        if review.get('decision')!='crop_content_reviewed':raise ValueError('Unaccepted exception')
        if review['annotation_action']=='include_visible_right_hand_alongside_left_context':
            tight[i]=union([tight[i],review['additional_hand_box'],*right_regions.get(i,[])])
        elif review['annotation_action']!='retain_candidate_box':raise ValueError('Unknown review operation')
    w,h=rows[0]['source_size']
    windows,metrics=plan_windows(tight,w,h)
    if not metrics['all_boxes_contained'] or metrics['center_steps_over_3pct'] or metrics['scale_steps_over_2pct']:
        raise ValueError('Offline local geometry/continuity acceptance failed')
    for i,row in enumerate(rows):
        row={**row,'valid':True,'local_roi_valid':True,'image_available':True,
             'interaction_box':tight[i].tolist(),'crop_box':windows[i].tolist(),
             'display_crop_box':windows[i].tolist(),'input_box':windows[i].tolist(),
             'input_mode':'offline_local_crop',
             'local_evidence_source':'coding_agent_exception_crop_review' if i in reviews else ('bidirectional_RGB_tracking' if row['repair'] else 'original_current_visual_evidence')}
        if i in reviews:
            row['exception_review']=reviews[i]
            row['review_status']='coding_agent_visual_exception_review_not_independent_ground_truth'
            row['roles']=reviews[i].get('roles',row['roles'])
        rows[i]=row
    return rows,metrics


def episode(job):
    import av,cv2
    from scripts.tomato_stable_roi import red_regions,gap
    cv2.setNumThreads(1)
    ep,row,folder,review,out,contract=job
    parent_receipt=json.loads((folder/'receipt.json').read_text())
    if digest(folder/'index.jsonl.gz')!=parent_receipt['index_sha256']:raise ValueError('Offline parent changed')
    rows=read_candidate_index(folder/'index.jsonl.gz',row['length'],allow_offline=True)
    reviews={r['frame']:r for r in review['reviews'] if r['episode']==ep}
    video_rel=f"videos/chunk-{row['source_episode_index']//1000:03d}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
    source=Path(row['dataset'])/video_rel
    if digest(source)!=row['source_sha256'][video_rel]:raise ValueError('Original source changed')
    changed={i:r for i,r in reviews.items() if r['annotation_action']!='retain_candidate_box'}
    regions={}
    if changed:
        with av.open(str(source)) as src:
            for i,f in enumerate(src.decode(video=0)):
                if i in changed:
                    hand=np.array(changed[i]['additional_hand_box'])
                    regions[i]=[b for b in red_regions(f.to_ndarray(format='rgb24')) if gap(b,hand)<.12*max(hand[2:]-hand[:2])]
                if i>=max(changed):break
    rows,geometry=finalize_rows(rows,reviews,regions)
    target=out/f'shard{ep%4}'/f'episode_{ep:06d}';checks={}
    with new_artifact_directory(target) as staging:
        with av.open(str(source)) as src,av.open(str(staging/'interaction224.mp4'),'w') as dst,gzip.open(staging/'index.jsonl.gz','wt') as index:
            stream=dst.add_stream('libx264',rate=30);stream.width=224;stream.height=224;stream.pix_fmt='yuv420p'
            stream.options={'crf':'16','preset':'fast'};stream.thread_count=2;n=0
            for n,frame in enumerate(src.decode(video=0),1):
                i=n-1;r=rows[i]
                if abs(float(frame.pts*frame.time_base)-i/30)>1e-4:raise ValueError('Current source time mismatch')
                pixels=frame.to_ndarray(format='rgb24')
                crop=cv2.resize(native_crop(pixels,r['crop_box']),(224,224),interpolation=cv2.INTER_AREA)
                if not r['original_valid'] or i%200==0:checks[i]=crop
                index.write(json.dumps(r)+'\n')
                for packet in stream.encode(av.VideoFrame.from_ndarray(crop,format='rgb24')):dst.mux(packet)
            for packet in stream.encode():dst.mux(packet)
        if n!=len(rows):raise ValueError('Source frame count mismatch')
        black=0;decoded=0;maximum_mae=0.
        with av.open(str(staging/'interaction224.mp4')) as verify:
            for decoded,frame in enumerate(verify.decode(video=0),1):
                pixels=frame.to_ndarray(format='rgb24')
                if pixels.shape!=(224,224,3) or abs(float(frame.pts*frame.time_base)-(decoded-1)/30)>1e-4:
                    raise ValueError('Encoded frame alignment mismatch')
                black+=int(not pixels.any())
                if decoded-1 in checks:
                    mae=float(np.abs(pixels.astype(np.float32)-checks[decoded-1]).mean());maximum_mae=max(maximum_mae,mae)
                    if mae>12:raise ValueError('Encoded pixels are not the corresponding current crop')
        if decoded!=n or black or digest(source)!=row['source_sha256'][video_rel]:raise ValueError('Incomplete/black output or changed source')
        receipt=dict(contract=contract,episode_index=ep,frames=n,algorithm_valid_frames=n,algorithm_invalid_frames=0,
                     image_available_frames=n,full_view_fallback_frames=0,original_missing_repaired_frames=parent_receipt['repaired_missing_frames'],
                     visual_exception_reviewed_frames=len(reviews),right_hand_context_corrections=len(changed),
                     geometry=geometry,black_frames=black,all_frames_decoded=True,current_pixel_checks=len(checks),
                     current_pixel_max_mae=maximum_mae,video_sha256=digest(staging/'interaction224.mp4'),
                     index_sha256=digest(staging/'index.jsonl.gz'),source_video_sha256=digest(source),source_hashes_unchanged=True,
                     offline_annotation_receipt_sha256=digest(folder/'receipt.json'),source_writes=False,training_ready=False)
        atomic_write_json(staging/'receipt.json',receipt)
    return receipt


def run(selection_path,annotations,review_path,out,workers):
    selection=json.loads(selection_path.read_text());manifest=json.loads((annotations/'annotation_manifest.json').read_text());review=json.loads(review_path.read_text())
    if manifest['episodes']!=100 or manifest['frames']!=425556 or review['annotation_manifest_sha256']!=digest(annotations/'annotation_manifest.json'):
        raise ValueError('Complete hash-bound offline annotations and review required')
    if not review['all266_current_frame_tiles_viewed'] or len(review['reviews'])!=manifest['pending_frames']:
        raise ValueError('Complete exception review required')
    out=derived_output_path(selection,out);out.mkdir(parents=True,exist_ok=False)
    contract=dict(stream_contract='offline_bidirectional_local_rgb_v6',selection_sha256=digest(selection_path),
                  generator_sha256=digest(Path(__file__)),geometry_sha256=digest(Path(plan_windows.__code__.co_filename)),
                  annotation_contract=manifest['contract'],annotation_manifest_sha256=digest(annotations/'annotation_manifest.json'),
                  exception_review_sha256=digest(review_path),uses_future_frames_for_roi=True,offline_annotation_only=True,
                  pixel_source='same_current_native_frame',independent_semantic_quality_approved=False)
    atomic_write_json(out/'contract.json',contract)
    jobs=[(ep,row,annotations/f'shard{ep%4}'/f'episode_{ep:06d}',review,out,contract) for ep,row in enumerate(selection['episodes'])]
    receipts=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(episode,jobs):
            receipts.append(r);print(f"ep{r['episode_index']}: {r['frames']} current local frames, no global fallback; geometry and decode pass",flush=True)
    for shard in range(4):atomic_write_json(out/f'shard{shard}'/'candidate_manifest.json',dict(contract=contract,episodes=[r for r in receipts if r['episode_index']%4==shard]))
    result=dict(contract=contract,episodes=100,frames=sum(r['frames'] for r in receipts),image_available_frames=sum(r['image_available_frames'] for r in receipts),
                local_crop_frames=sum(r['frames'] for r in receipts),full_view_fallback_frames=0,black_frames=0,pending_frames=0,
                full_image_coverage=True,full_local_geometry_coverage=True,all_frames_decoded=True,
                center_steps_over_3pct=sum(r['geometry']['center_steps_over_3pct'] for r in receipts),
                scale_steps_over_2pct=sum(r['geometry']['scale_steps_over_2pct'] for r in receipts),
                max_center_step_px=max(r['geometry']['max_center_step_px'] for r in receipts),
                current_pixel_checks=sum(r['current_pixel_checks'] for r in receipts),
                visual_exception_reviewed_frames=sum(r['visual_exception_reviewed_frames'] for r in receipts),
                independent_semantic_accuracy_measured=False,training_ready=False,training_started=False,source_writes=False,
                scope='100% frame/pixel/geometry/continuity acceptance against prepared annotations, not100% independently measured semantic accuracy or causal locator performance')
    if result['frames']!=425556 or result['image_available_frames']!=425556:raise ValueError('Full100 coverage failed')
    atomic_write_json(out/'offline_export_manifest.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    run(a.selection,a.annotations,a.review,a.out,a.workers)
