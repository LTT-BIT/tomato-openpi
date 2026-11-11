"""Read-only full base-copy and real four-view adapter checks; never trains."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.subtasks.rgb100 import digest, write_new


def check_base(selection, bundle):
    meta=json.loads((bundle/'bundle.json').read_text())
    split=json.loads((bundle/'splits.json').read_text())
    if split['splits']!={'train':list(range(100))}:
        raise ValueError('All100 training split required')
    frames=0; media=0
    for row,mapping in zip(selection['episodes'],meta['mapping'],strict=True):
        i=row['episode_index'];old=row['source_episode_index'];root=Path(row['dataset'])
        if (mapping['recording_group']!=row['recording_group']
                or mapping['annotation_sha256']!=row['annotation_sha256']):
            raise ValueError('Derived source mapping mismatch')
        relative=f'data/chunk-{old//1000:03d}/episode_{old:06d}.parquet'
        source=pq.read_table(root/relative)
        for mode in ('a_goal','b_coarse'):
            derived=pq.read_table(bundle/mode/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet')
            for key in source.column_names:
                if key not in ('index','episode_index','task_index','coarse_phase_index'):
                    if not source[key].equals(derived[key]):
                        raise ValueError(f'Physical source column changed in copy: {i} {mode} {key}')
            expected=np.empty(row['length'],dtype=np.int64)
            for phase,r in enumerate(row['training_segments']):expected[r['start_frame']:r['end_frame']]=phase
            np.testing.assert_array_equal(derived['coarse_phase_index'].to_numpy(),expected)
            if mode=='a_goal':np.testing.assert_array_equal(derived['task_index'].to_numpy(),np.zeros(len(source),dtype=np.int64))
            else:np.testing.assert_array_equal(derived['task_index'].to_numpy(),np.array([0,1,2,3,0,1,2])[expected])
            for cam in ('cam0','cam1','cam2'):
                old_rel=f'videos/chunk-{old//1000:03d}/observation.images.{cam}/episode_{old:06d}.mp4'
                new_path=bundle/mode/f'videos/chunk-{i//1000:03d}/observation.images.{cam}/episode_{i:06d}.mp4'
                if digest(new_path)!=row['source_sha256'][old_rel]:
                    raise ValueError('Copied video changed')
                media+=1
        # Original labels, media, metadata and physical vectors remain hash-equal.
        for relative,expected_hash in {**row['source_sha256'],**row['metadata_sha256']}.items():
            if digest(root/relative)!=expected_hash:raise ValueError('Original source changed')
        if digest(row['annotation'])!=row['annotation_sha256']:raise ValueError('Original annotation changed')
        frames+=len(source)
    return dict(recordings=100,frames=frames,derived_video_hashes_checked=media,
        original_sources_hash_unchanged=True,physical_columns_equal=True,
        phase_prompt_indices_verified=True,four_view_dataset_ready=False)


def check_real_adapter(selection, preview_dir, bundle):
    import av
    from openpi.models import model
    from openpi.policies.fr3_sharpa_crop_policy import Fr3SharpaCropInputs
    receipt=json.loads((preview_dir/'receipt.json').read_text());i=receipt['episode_index']
    row=selection['episodes'][i];old=row['source_episode_index'];root=Path(row['dataset'])
    cache=np.load(preview_dir/'crops224.npz');ids=cache['source_frames'].tolist()
    selected=(83,270);decoded=[]
    for cam in ('cam0','cam2','cam1'):
        path=root/f'videos/chunk-{old//1000:03d}/observation.images.{cam}/episode_{old:06d}.mp4'
        frames={}
        with av.open(str(path)) as container:
            for frame_id,frame in enumerate(container.decode(video=0)):
                if frame_id in selected:frames[frame_id]=frame.to_ndarray(format='rgb24')
                if frame_id>=max(selected):break
        decoded.append(frames)
    table=pq.read_table(bundle/'b_coarse'/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet')
    states=np.array(table['observation.state'].to_pylist(),dtype=np.float32)
    actions=np.array(table['action'].to_pylist(),dtype=np.float32)
    checked=[]
    for frame_id in selected:
        crop=cache['rgb'][ids.index(frame_id)]
        data={'observation/state':states[frame_id],
            'observation/image':decoded[0][frame_id],
            'observation/left_wrist_image':decoded[1][frame_id],
            'observation/right_wrist_image':decoded[2][frame_id],
            'observation/interaction_image':crop,'observation/interaction_valid':np.bool_(True),
            'actions':actions[frame_id:frame_id+50]}
        out=Fr3SharpaCropInputs(model.ModelType.PI05)(data)
        np.testing.assert_array_equal(out['image']['interaction_0_rgb'],crop)
        np.testing.assert_array_equal(out['state'],np.concatenate([states[frame_id,:7],states[frame_id,28:50],states[frame_id,14:21],states[frame_id,72:94]]))
        np.testing.assert_array_equal(out['actions'],np.concatenate([data['actions'][:,:7],data['actions'][:,14:36],data['actions'][:,7:14],data['actions'][:,36:58]],axis=-1))
        checked.append(dict(source_frame=frame_id,state_shape=list(out['state'].shape),action_shape=list(out['actions'].shape),images=list(out['image'])))
    return dict(samples=checked,scope='real pixels and measured/commanded vector adapter only; no normalized model query',
        weights_loaded=False,training_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--preview-dir',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    selection=json.loads(args.selection.read_text())
    result=dict(selection_sha256=digest(args.selection),bundle=str(args.bundle),base=check_base(selection,args.bundle))
    if args.preview_dir:result['real_adapter']=check_real_adapter(selection,args.preview_dir,args.bundle)
    write_new(args.out,result);print('100 base copies and original hashes verified; real adapter checked:',bool(args.preview_dir))


if __name__=='__main__':main()
