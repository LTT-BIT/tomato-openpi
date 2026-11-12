"""Offline phase consistency and manual-reference diagnostics, never ROI generation."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from openpi.subtasks.rgb100 import derived_output_path,digest,write_new
from scripts.audit_tomato_stable_roi import entries,find_episodes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--reference-annotations',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    selection=json.loads(args.selection.read_text())
    folders=find_episodes(args.candidates)
    if set(folders)!=set(range(100)):
        raise ValueError('All100 required for final diagnostics')
    flags=[];reasons=collections.Counter();evidence=collections.Counter();cases=[]
    reference_rows=None
    for ep,folder in sorted(folders.items()):
        rows=entries(folder)
        receipt=json.loads((folder/'receipt.json').read_text())
        if digest(folder/'index.jsonl.gz')!=receipt['index_sha256'] or receipt['contract']['selection_sha256']!=digest(args.selection):
            raise ValueError('Changed candidate/source identity')
        reasons.update(r['reason'] for r in rows if not r['valid'])
        for r in rows:
            evidence.update(r['hand_evidence'].values())
        for segment in selection['episodes'][ep]['training_segments']:
            if segment['label']=='harvest':
                continue
            for frame in range(segment['start_frame'],segment['end_frame']):
                r=rows[frame]
                if r['valid'] and 1 not in r['roles']:
                    flags.append(dict(episode_index=ep,frame=frame,phase=segment['label'],
                        boundary_distance=min(frame-segment['start_frame'],segment['end_frame']-1-frame)))
        for target in [(15,4988),(70,3116),(13,1929),(36,1851),(86,2820)]:
            if ep==target[0]:
                r=rows[target[1]]
                cases.append(dict(episode_index=ep,source_frame=target[1],valid=r['valid'],roles=r['roles'],
                    carry_evidence=r['carry_evidence'],grasp_state_confirmed=r.get('left_grasp_state_confirmed')))
        if ep==86:
            reference_rows=rows
    annotation=json.loads(args.reference_annotations.read_text())
    reference=collections.defaultdict(list)
    for clip in annotation['clips']:
        for key in clip['keyframes']:
            b=np.asarray(reference_rows[key['frame']]['display_crop_box'])
            for role in clip['roles']:
                if role not in ('right_hand','left_hand','left_hand_and_carried_fruit','truss','target','held_target'):
                    continue
                t=np.asarray(key['boxes'][role])
                intersection=np.maximum(0,np.minimum(b[2:],t[2:])-np.maximum(b[:2],t[:2]))
                reference[role].append(float(np.prod(intersection)/np.prod(t[2:]-t[:2])))
    distances=[r['boundary_distance'] for r in flags]
    result=dict(selection_sha256=digest(args.selection),episodes=100,
        potential_nonharvest_left_only_frames=len(flags),phase_counts=dict(collections.Counter(r['phase'] for r in flags)),
        boundary_diagnostic=dict(within30=sum(d<=30 for d in distances),within60=sum(d<=60 for d in distances),
            median=float(np.median(distances)) if distances else None,maximum=max(distances,default=0)),
        interior_examples=sorted(flags,key=lambda r:-r['boundary_distance'])[:10],
        invalid_reasons=dict(reasons),hand_evidence_counts=dict(evidence),previous_error_cases=cases,
        manual_reference_annotation_sha256=digest(args.reference_annotations),
        manual_reference_display_coverage={role:dict(boxes=len(values),mean=float(np.mean(values)),
            at_least90pct=sum(v>=.9 for v in values)) for role,values in reference.items()},
        phase_labels_used='Offline QA only; never used to generate or modify boxes/masks.',
        scope='Potential mismatches are not labeled errors. Manual reference is in-sample, used for locator/calibration; not independent accuracy.',
        semantic_quality_accepted=False,training_ready=False,training_started=False,source_writes=False)
    write_new(derived_output_path(selection,args.out),result)
    print('Potential consistency flags',len(flags),'(not an independently labeled error count)')


if __name__=='__main__':
    main()
