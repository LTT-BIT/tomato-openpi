"""Create isolated ROI annotation worklists and export the EXISTING sparse preview.

Never invent boxes for uncovered frames or write into source datasets. The
existing preview is coding-agent/manual offline interpolation, not ground truth
or an online locator. Full coverage remains a separate readiness requirement.
"""
import argparse
import json
from pathlib import Path

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest


def worklist(selection, preview, *, preview_sha256):
    if selection.get('phase_label_resolution_required')!=0 or len(selection['episodes'])!=100:
        raise ValueError('Resolved100 selection required')
    rows=[]; matched=0
    for row in selection['episodes']:
        root=Path(row['dataset']); source=f"{root.parent.name}/{root.name}/episode_{row['source_episode_index']:06d}"
        clips=[]
        if source==preview['source_recording']:
            matched+=1
            if preview['fps']!=row['fps']:
                raise ValueError('ROI/source FPS mismatch')
            for cam,sha in preview['source_sha256'].items():
                relative=f"videos/chunk-{row['source_episode_index']//1000:03d}/observation.images.{cam}/episode_{row['source_episode_index']:06d}.mp4"
                if row['source_sha256'].get(relative)!=sha:
                    raise ValueError('Preview source SHA does not match selected recording')
            clips=preview['clips']
        intervals=[]
        for clip in clips:
            if not 0<=clip['start']<clip['end']<=row['length']:
                raise ValueError('ROI interval outside recording')
            intervals.append((clip['start'],clip['end']))
        ordered=sorted(intervals)
        if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
            raise ValueError('Overlapping ROI intervals')
        covered=sum(b-a for a,b in intervals)
        rows.append(dict(episode_index=row['episode_index'],source_recording=source,
            dataset=row['dataset'],source_episode_index=row['source_episode_index'],
            recording_group=row['recording_group'],length=row['length'],fps=row['fps'],
            annotation_sha256=row['annotation_sha256'],source_sha256=row['source_sha256'],
            training_segments=row['training_segments'],
            roi_clips=clips,offline_preview_frames=covered,remaining_frames=row['length']-covered,
            roi_generation='manual_offline_interpolated' if clips else 'not_annotated',
            roi_review_status='unreviewed_preview' if clips else 'not_annotated'))
    if matched!=1:
        raise ValueError('Existing preview must match exactly one selected recording')
    return dict(version='tomato-complete100-roi-worklist-v1',episodes=rows,
        preview_sha256=preview_sha256,total_frames=sum(r['length'] for r in rows),
        offline_preview_frames=sum(r['offline_preview_frames'] for r in rows),
        remaining_frames=sum(r['remaining_frames'] for r in rows),
        full_roi_coverage=False,training_ready=False,source_writes=False,
        note='Blank intervals are missing annotations, never implicit success/full-view ROI')


def export_preview(selection, preview, out):
    import av
    import cv2
    import numpy as np
    from scripts.tomato_interaction_preview import interaction_at, native_crop, square_box
    out=derived_output_path(selection,out)
    listing=worklist(selection,preview,preview_sha256='provided_by_caller')
    row=next(r for r in listing['episodes'] if r['roi_clips'])
    relative=f"videos/chunk-{row['source_episode_index']//1000:03d}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
    source=Path(row['dataset'])/relative; expected=row['source_sha256'][relative]
    if digest(source)!=expected:
        raise ValueError('Source RGB changed')
    wanted={i:clip for clip in preview['clips'] for i in range(clip['start'],clip['end'])}
    arrays=[]; index=[]
    with av.open(str(source)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in wanted:
                image=frame.to_ndarray(format='rgb24');h,w=image.shape[:2]
                if [w,h]!=preview['source_size']:
                    raise ValueError('Native source dimensions changed')
                clip=wanted[i];tight=interaction_at(clip,i,w,h);box=square_box(tight,w,h)
                arrays.append(cv2.resize(native_crop(image,box),(224,224),interpolation=cv2.INTER_AREA))
                future_key=next(k['frame'] for k in clip['keyframes'] if k['frame']>=i)
                index.append(dict(source_frame=i,timestamp=i/preview['fps'],interaction_box=tight,
                    crop_box=box,source_camera='cam0',valid=True,available_at=future_key/preview['fps'],
                    generation_mode='manual_offline_interpolated',review_status='unreviewed_preview'))
            if i>=max(wanted):break
    if len(index)!=len(wanted) or digest(source)!=expected:
        raise ValueError('Source/frame verification failed')
    with new_artifact_directory(out) as staging:
        np.savez_compressed(staging/'crops224.npz',rgb=np.asarray(arrays),source_frames=np.array(sorted(wanted)))
        (staging/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
        receipt=dict(episode_index=row['episode_index'],source_recording=row['source_recording'],
            source_video=str(source),source_sha256=expected,source_hash_unchanged=True,
            crop_sha256=digest(staging/'crops224.npz'),frames=len(index),size=224,
            automatic_roi=False,full_roi_coverage=False,source_writes=False,training_started=False)
        (staging/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--preview',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--export-existing-preview',action='store_true')
    args=parser.parse_args()
    s=json.loads(args.selection.read_text());p=json.loads(args.preview.read_text())
    out=derived_output_path(s,args.out)
    if args.export_existing_preview:
        result=export_preview(s,p,out)
        print('Existing offline preview exported:',result['frames'],'frames; not full100 ROI')
    else:
        result=worklist(s,p,preview_sha256=digest(args.preview))
        result['selection_sha256']=digest(args.selection)
        with new_artifact_directory(out) as staging:
            (staging/'worklist.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('100-entry ROI worklist; existing preview',result['offline_preview_frames'],
              'missing',result['remaining_frames'])


if __name__=='__main__':
    main()
