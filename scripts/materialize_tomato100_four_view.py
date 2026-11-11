"""Materialize complete four-view CANDIDATE copies; quality gate remains false."""
import argparse
import copy
import gzip
import json
import shutil
from pathlib import Path

from openpi.shared.local_storage import atomic_write_json,new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path,digest


def collect_candidates(root):
    manifests=sorted(root.glob('shard*/candidate_manifest.json'))
    if len(manifests)!=4:raise ValueError('Four finished candidate shards required')
    episodes={};contract=None
    for path in manifests:
        spec=json.loads(path.read_text())
        if contract is None:contract=spec['contract']
        if spec['contract']!=contract:raise ValueError('Candidate generator identities differ')
        for row in spec['episodes']:
            ep=row['episode_index']
            if ep in episodes:raise ValueError('Duplicate candidate episode')
            episodes[ep]=(path.parent/f'episode_{ep:06d}',row)
    if set(episodes)!=set(range(100)):raise ValueError('All100 candidate episodes required')
    return contract,episodes


def read_candidate_index(path,length):
    import numpy as np
    rows=[]
    with gzip.open(path,'rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['source_frame']!=len(rows) or abs(row['timestamp']-len(rows)/30)>1e-6 or row['available_at']>row['timestamp']+1e-9:
                raise ValueError('ROI time/frame causality mismatch')
            if type(row['valid']) is not bool:raise ValueError('Explicit boolean required')
            if row['valid']:
                box=np.asarray(row['crop_box'])
                if box.shape!=(4,) or not np.isfinite(box).all() or box[2]<=box[0] or box[3]-box[1]!=box[2]-box[0]:raise ValueError('Invalid candidate square')
                tight=np.asarray(row['interaction_box']);w,h=row['source_size']
                if (tight.shape!=(4,) or not np.isfinite(tight).all() or not (0<=tight[0]<tight[2]<=w and 0<=tight[1]<tight[3]<=h)
                    or not (box[0]<=tight[0] and box[1]<=tight[1] and box[2]>=tight[2] and box[3]>=tight[3])):
                    raise ValueError('Crop must contain the finite native interaction box')
            elif row['crop_box'] is not None or row['interaction_box'] is not None:
                raise ValueError('Lost target must have null coordinates')
            rows.append(row)
    if len(rows)!=length:raise ValueError('Incomplete candidate index')
    return rows


def run(selection_path,base,candidates,out):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    selection=json.loads(selection_path.read_text());parent=json.loads((base/'bundle.json').read_text())
    contract,episodes=collect_candidates(candidates)
    if contract['selection_sha256']!=digest(selection_path) or parent['selection_sha256']!=digest(selection_path):raise ValueError('Selection identity mismatch')
    if json.loads((base/'splits.json').read_text())['splits']!={'train':list(range(100))}:raise ValueError('All100 split required')
    out=derived_output_path(selection,out);valid_total=0;count=0
    with new_artifact_directory(out) as staging:
        for mode in ('a_goal','b_coarse'):shutil.copytree(base/mode,staging/mode,copy_function=shutil.copy2)
        shutil.copy2(base/'splits.json',staging/'splits.json')
        roi_meta=staging/'roi_metadata';roi_meta.mkdir()
        all_validity={}
        for ep,(folder,receipt) in sorted(episodes.items()):
            row=selection['episodes'][ep]
            index=folder/'index.jsonl.gz';video=folder/'interaction224.mp4'
            if receipt['contract']!=contract or digest(video)!=receipt['video_sha256'] or digest(index)!=receipt['index_sha256']:
                raise ValueError('Candidate artifact changed')
            entries=read_candidate_index(index,row['length'])
            valid=np.array([e['valid'] for e in entries],bool);all_validity[ep]=valid
            if int(valid.sum())!=receipt['algorithm_valid_frames']:raise ValueError('Validity receipt mismatch')
            valid_total+=int(valid.sum());count+=len(valid)
            shutil.copy2(index,roi_meta/f'episode_{ep:06d}.jsonl.gz')
            for mode in ('a_goal','b_coarse'):
                data=staging/mode/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
                table=pq.read_table(data)
                if 'observation.interaction_valid' in table.column_names:raise ValueError('Base unexpectedly has crop validity')
                np.testing.assert_allclose(table['timestamp'].to_numpy(),np.arange(len(valid))/30,atol=1e-5,rtol=0)
                table=table.append_column('observation.interaction_valid',pa.array(valid,type=pa.bool_()))
                pq.write_table(table,data)
                # Every pre-existing column remains bit-exact in value.
                before=pq.read_table(base/mode/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet')
                for key in before.column_names:
                    if not before[key].equals(table[key]):raise ValueError('Base column altered')
                target=staging/mode/f'videos/chunk-{ep//1000:03d}/observation.images.interaction/episode_{ep:06d}.mp4'
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(video,target)
                if digest(target)!=receipt['video_sha256']:raise ValueError('Crop copy changed')
        for mode in ('a_goal','b_coarse'):
            meta=staging/mode/'meta';info=json.loads((meta/'info.json').read_text())
            feature=copy.deepcopy(info['features']['observation.images.cam0'])
            feature['shape']=[224,224,3];feature['info'].update({'video.height':224,'video.width':224,'video.fps':30})
            info['features']['observation.images.interaction']=feature
            info['features']['observation.interaction_valid']={'dtype':'bool','shape':[1],'names':None}
            info['total_videos']=400;atomic_write_json(meta/'info.json',info)
            stats=json.loads((meta/'stats.json').read_text());p=valid_total/count
            stats['observation.interaction_valid']=dict(min=[0 if valid_total<count else 1],max=[1 if valid_total else 0],mean=[p],std=[float(np.sqrt(p*(1-p)))],count=[count])
            atomic_write_json(meta/'stats.json',stats)
            records=[json.loads(line) for line in (meta/'episodes_stats.jsonl').read_text().splitlines() if line]
            for item in records:
                valid=all_validity[item['episode_index']];p=float(valid.mean())
                item['stats']['observation.interaction_valid']=dict(min=[int(valid.all())],max=[int(valid.any())],mean=[p],std=[float(np.sqrt(p*(1-p)))],count=[len(valid)])
            (meta/'episodes_stats.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        manifest=dict(selection_sha256=digest(selection_path),episodes=100,frames=count,algorithm_valid_frames=valid_total,
            algorithm_invalid_frames=count-valid_total,all_frames_enumerated=True,full_roi_coverage=False,
            roi_generation_mode='causal_prediction_candidate',roi_generator_signature=contract,
            training_ready=False,quality_review='pending_sampled_semantic_and_temporal_acceptance',
            source_writes=False,training_started=False,
            scope='Complete candidate input plumbing; current training recipe must reject until explicit quality acceptance.')
        atomic_write_json(staging/'crop_manifest.json',manifest)
        atomic_write_json(staging/'bundle.json',{**parent,'four_view_candidate':True,'training_ready':False,
            'parent_bundle_sha256':digest(base/'bundle.json'),'crop_manifest_sha256':digest(staging/'crop_manifest.json')})
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.selection,a.base,a.candidates,a.out),indent=2))


if __name__=='__main__':main()
