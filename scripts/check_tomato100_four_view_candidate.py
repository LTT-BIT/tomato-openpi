"""Read-only actual four-video LeRobot/vector/token checks; never bypasses the training quality gate."""
import argparse
import json
from pathlib import Path

from openpi.subtasks.rgb100 import IMAGE_KEYS,digest,write_new


def run(selection_path,bundle,norm_root,tokenizer_path):
    import numpy as np
    import sentencepiece
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared import normalize
    from openpi.training.config import LeRobotFr3SharpaCropDataConfig
    from openpi.training.task_view import TaskView,TaskViewDataset
    from scripts.check_tomato100_derived import check_base
    selection=json.loads(selection_path.read_text());crop=json.loads((bundle/'crop_manifest.json').read_text())
    if crop['selection_sha256']!=digest(selection_path) or crop['frames']!=425556 or crop['training_ready'] is not False:
        raise ValueError('Expected complete candidate, no quality approval')
    checked_base=check_base(selection,bundle)
    root=bundle/'b_coarse'
    dataset=LeRobotDataset(repo_id='fr3_sharpa/tomato_complete100_fourview_candidate_b',root=root,
        delta_timestamps={'action':[i/30 for i in range(50)]},video_backend='pyav')
    source=TaskViewDataset(dataset,TaskView(split_path=str(bundle/'splits.json'),split='train'),50)
    repack,vectors=LeRobotFr3SharpaCropDataConfig(extra_delta_transform=True).create_vector_transforms(
        Pi0Config(pi05=True,action_dim=58,image_keys=IMAGE_KEYS,max_token_len=256))
    transform=transforms.compose([*repack.inputs,*vectors.inputs])
    stats=normalize.load(norm_root);normalizer=transforms.Normalize(stats,use_quantiles=True)
    tokenizer=PaligemmaTokenizer.__new__(PaligemmaTokenizer);tokenizer._max_len=256
    tokenizer._tokenizer=sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    with np.load(norm_root/'real_state_tokens.npz') as cache:
        lengths=cache['token_lengths'];ids=cache['observation_ids']
    offset=0;checked=0;invalid_checked=0;observed=[]
    for row in selection['episodes']:
        source_path=Path(row['dataset'])/f"data/chunk-{row['source_episode_index']//1000:03d}/episode_{row['source_episode_index']:06d}.parquet"
        physical=pq.read_table(source_path,columns=['observation.state','action'])
        measured=np.array(physical['observation.state'].to_pylist(),np.float32)
        commanded=np.array(physical['action'].to_pylist(),np.float32)
        for frame in [0,row['length']//2,row['length']-1]:
            index=offset+frame;raw=source[index]
            if int(raw['episode_index'])!=row['episode_index'] or int(raw['frame_index'])!=frame:raise ValueError('Sample identity mismatch')
            value=transform(raw);expected=transform({**raw,**source.vectors(index)})
            for key in ('state','actions'):np.testing.assert_array_equal(value[key],expected[key])
            q=measured[frame];canonical=np.r_[q[:7],q[28:50],q[14:21],q[72:94]]
            window=commanded[np.minimum(frame+np.arange(50),row['length']-1)]
            target=np.c_[window[:,:7],window[:,14:36],window[:,7:14],window[:,36:58]]
            target[:,:7]-=canonical[:7];target[:,29:36]-=canonical[29:36]
            np.testing.assert_array_equal(value['state'],canonical)
            np.testing.assert_array_equal(value['actions'],target)
            if tuple(value['image'])!=IMAGE_KEYS or value['state'].shape!=(58,) or value['actions'].shape!=(50,58):raise ValueError('Actual four-view vector shape mismatch')
            valid=bool(raw['observation.interaction_valid'])
            if bool(value['image_mask']['interaction_0_rgb'])!=valid:raise ValueError('Validity mask lost')
            if not valid:invalid_checked+=1
            if value['image']['interaction_0_rgb'].shape!=(224,224,3):raise ValueError('Crop decode shape mismatch')
            normalized=normalizer(value);tokens,mask=tokenizer.tokenize(normalized['prompt'],normalized['state'])
            if int(mask.sum())!=int(lengths[index]) or list(ids[index])!=[row['episode_index'],frame]:raise ValueError('Actual tokenizer path differs from all100 audit')
            observed.append(int(mask.sum()));checked+=1
        offset+=row['length']
        if (row['episode_index']+1)%10==0:print('Actual four-view loader checked',row['episode_index']+1,'/100',flush=True)
    if offset!=425556 or len(source)!=425556:raise ValueError('Incomplete selected frame set')
    return dict(selection_sha256=digest(selection_path),crop_manifest_sha256=digest(bundle/'crop_manifest.json'),
        base=checked_base,real_four_view_loader=dict(episodes=100,samples=checked,invalid_mask_samples=invalid_checked,
            actual_crop_size=[224,224,3],state_size=58,action_window=[50,58],model_keys=list(IMAGE_KEYS),
            normalized_token_lengths_match=True,independent_original_vector_window_reference=True,sampled_text_maximum=max(observed),all100_action_training=True),
        final_normalization_contract_bound=False,roi_quality_accepted=False,training_ready=False,training_started=False,
        source_writes=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--norm',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=run(a.selection,a.bundle,a.norm,a.tokenizer);write_new(a.out,result);print('PASS actual four-view candidate loader and token checks')


if __name__=='__main__':main()
