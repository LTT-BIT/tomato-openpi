"""Three-current-RGB subtask control; all non-image training settings match crop B."""
from dataclasses import replace
import json
import os
from pathlib import Path

from openpi.subtasks import tomato_in_hand as contract
from openpi.subtasks.prepare import digest
from training_configs import common
from training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL
from training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_crop import build_config as crop_config

DATA_ROOT = '/mnt/data/users/mengxiang/workspace/training_data/sharpa/full-block-more/in_hand204_three_rgb_v3_r1'
CACHE_ROOT = '/mnt/data/users/mengxiang/workspace/training_data/sharpa/full-block-more/in_hand204_three_rgb224_cache_v3_r1'
ASSET_ID = 'tomato_in_hand204_three_rgb_v3_r1_delta_30hz'
MODULE = 'training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_three_rgb'
EXP_NAME = 'tomato_in_hand204_b_three_rgb_delta_v3_r1_s42_b256_s20000_c5000_after_crop_1120'
REPORT = 'logs/in_hand204_three_rgb/accepted.json'


def build_config():
    base = crop_config()
    root = Path(os.environ.get('PI05_IN_HAND204_THREE_VIEW', DATA_ROOT)).resolve()
    manifest = json.loads((root/'view.json').read_text())
    info = json.loads((root/'meta/info.json').read_text())
    if (manifest.get('episodes') != contract.EPISODES or manifest.get('frames') != contract.FRAMES
            or manifest.get('only_removed_interaction_view') is not True
            or manifest.get('source_bundle_sha256') != base.policy_metadata['bundle_sha256']
            or set(k for k,v in info['features'].items() if v.get('dtype') == 'video') != set(contract.CAMERAS[:3])):
        raise ValueError('Bound three-RGB view of the same complete204 recordings required')
    for relative, sha in manifest['metadata_sha256'].items():
        if digest(root/relative) != sha: raise ValueError('Three-view metadata changed')
    dataset = common.Dataset(version='tomato_in_hand204_three_rgb_v3_r1',
        repo_id='fr3_sharpa/tomato_in_hand204_three_rgb_v3_r1',root=str(root),
        assets_dir=str(common.repository_root()/'assets'),asset_id=ASSET_ID,
        sources=(*base.policy_metadata['source_datasets'],str(root/'view.json')),prompt_example=contract.TASKS[1])
    data = replace(FR3_SHARPA_MULTIMODAL.data_config(dataset),extra_delta_transform=True)
    metadata = {**base.policy_metadata, **FR3_SHARPA_MULTIMODAL.protocol_metadata,
        'embodiment_version':FR3_SHARPA_MULTIMODAL.version,'camera_keys':list(contract.CAMERAS[:3]),
        'dataset_version':dataset.version,'dataset_repo_id':dataset.repo_id,'dataset_root':str(root),
        'source_datasets':list(dataset.sources),'subtask_variant':'in_hand_b_three_rgb',
        'visual_tokens':768,'rgb_frame_cache':CACHE_ROOT,'three_view_acceptance_report':str(common.repository_root()/REPORT),
        'training_action_semantics':'delta_arm_joint_target_absolute_hand',
        'comparison':'same204_subtasks_state_actions_without_interaction_rgb',
        'three_view_manifest_sha256':digest(root/'view.json')}
    metadata.pop('input_acceptance_report',None)
    revision = common.repository_root()/'runtime_revision.json'
    if revision.exists():metadata['runtime_source_commit']=json.loads(revision.read_text())['git_commit']
    return replace(base,name='pi05_sharpa_in_hand204_b_three_rgb_fullft_delta_v3_r1',exp_name=EXP_NAME,
        model=replace(base.model,image_keys=base.model.image_keys[:3]),data=data,policy_metadata=metadata)
