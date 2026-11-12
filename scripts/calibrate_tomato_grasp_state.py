"""Record raw measured MCP-flexion calibration from the bound manual ROI reference."""
import argparse
import json
from pathlib import Path

from openpi.subtasks.rgb100 import derived_output_path,digest,write_new
from scripts.prepare_tomato100_roi import worklist
from scripts.tomato_state_guarded_roi import LEFT_MCP_COLUMNS,LEFT_MCP_NAMES,GRASP_ENTER,GRASP_EXIT


def main():
    import numpy as np
    import pyarrow.parquet as pq
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--episode',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    selection=json.loads(args.selection.read_text())
    annotations=json.loads(args.annotations.read_text())
    listing=worklist(selection,annotations,preview_sha256=digest(args.annotations))
    if [r['episode_index'] for r in listing['episodes'] if r['roi_clips']]!=[args.episode]:
        raise ValueError('A single bound manual reference episode is required')
    row=selection['episodes'][args.episode]
    rel=f"data/chunk-{row['source_episode_index']//1000:03d}/episode_{row['source_episode_index']:06d}.parquet"
    path=Path(row['dataset'])/rel
    if digest(path)!=row['source_sha256'][rel]:
        raise ValueError('Raw measured states changed')
    info=json.loads((Path(row['dataset'])/'meta/info.json').read_text())
    names=info['features']['observation.state']['names']
    if tuple(names[i] for i in LEFT_MCP_COLUMNS)!=LEFT_MCP_NAMES:
        raise ValueError('Unexpected measured joint schema')
    states=np.asarray(pq.read_table(path,columns=['observation.state'])['observation.state'].to_pylist())
    curl=states[:,list(LEFT_MCP_COLUMNS)].mean(1)
    positive=[];negative=[]
    for clip in annotations['clips']:
        span=dict(start=clip['start'],end=clip['end'])
        if 'left_hand_and_carried_fruit' in clip['roles']:
            positive.append(span)
        elif all(not role.startswith('left_hand') for role in clip['roles']):
            negative.append(span)
    def distribution(spans):
        ids=[i for span in spans for i in range(span['start'],span['end'])]
        return dict(frames=len(ids),percentiles={str(p):float(np.percentile(curl[ids],p)) for p in [1,5,25,50,95,99]})
    result=dict(selection_sha256=digest(args.selection),annotation_sha256=digest(args.annotations),
        source_parquet_sha256=digest(path),episode_index=args.episode,native_source_episode_index=row['source_episode_index'],
        measured_state_columns=list(LEFT_MCP_COLUMNS),joint_names=list(LEFT_MCP_NAMES),
        score='mean of four raw measured MCP flexion positions; no command targets or pi05 normalization',
        enter=GRASP_ENTER,exit=GRASP_EXIT,consecutive_frames=3,
        positive_intervals=positive,negative_intervals=negative,
        positive=distribution(positive),negative=distribution(negative),
        labels='Manual ROI transport intervals and left-inactive intervals; these are weak pose references, not possession ground truth.',
        threshold_policy='Conservative fixed hand-shape hysteresis informed by reference distributions; independent accuracy not measured.',
        requires_current_visual_evidence=True,inference_uses_phase_labels=False,source_writes=False,training_started=False)
    if digest(path)!=row['source_sha256'][rel]:
        raise ValueError('Raw source changed during calibration')
    out=derived_output_path(selection,args.out)
    write_new(out,result)
    print('Recorded measured hand-state calibration',out)


if __name__=='__main__':
    main()
