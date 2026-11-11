"""All100 four-view B candidate. Refuses partial ROI/base three-view bundles.

No training starts by importing this module. Set PI05_TOMATO100_CROP_BUNDLE to
an independently materialized four-view bundle with a bound crop_manifest.json.
"""
from dataclasses import replace
import json
import os
from pathlib import Path

from openpi.subtasks.rgb100 import IMAGE_KEYS, digest
from openpi.subtasks.tomato import GOAL, PHASES, PROMPTS
from openpi.training.config import TaskDataConfig
from openpi.training.task_view import TaskView
from training_configs import common
from training_configs.embodiments.fr3_sharpa_crop import FR3_SHARPA_CROP
from training_configs.recipes.pi05_fullft_v1 import build_config as recipe


def build_config():
    value=os.environ.get('PI05_TOMATO100_CROP_BUNDLE')
    if not value:raise ValueError('Set PI05_TOMATO100_CROP_BUNDLE after full100 ROI preparation')
    root=Path(value).expanduser().resolve()
    crop=root/'crop_manifest.json'
    if not crop.is_file():raise ValueError('Missing crop_manifest.json; base three-view bundle is not ready')
    manifest=json.loads(crop.read_text())
    records=Path(__file__).parent/'records'
    selection_path=records/'complete100_rgb_crop_20261111_saved_labels_r1.json'
    selection=json.loads(selection_path.read_text())
    if (manifest.get('selection_sha256')!=digest(selection_path)
            or manifest.get('full_roi_coverage') is not True
            or manifest.get('episodes')!=100 or manifest.get('frames')!=425556
            or manifest.get('roi_generation_mode') not in ('manual_offline_interpolated','causal_prediction')
            or not manifest.get('roi_generator_signature')):
        raise ValueError('Complete hash-bound100 ROI coverage/provenance required')
    splits=json.loads((root/'splits.json').read_text())
    if splits['splits']!={'train':list(range(100))}:raise ValueError('All100 action training required')
    info=json.loads((root/'b_coarse/meta/info.json').read_text())
    tasks=[json.loads(line)['task'] for line in (root/'b_coarse/meta/tasks.jsonl').read_text().splitlines()]
    if (info['total_episodes']!=100 or info['total_frames']!=sum(r['length'] for r in selection['episodes'])
            or tasks!=list(dict.fromkeys(PROMPTS))
            or set(k for k,v in info['features'].items() if v.get('dtype')=='video')!=set(FR3_SHARPA_CROP.camera_keys)
            or info['features'].get('observation.interaction_valid',{}).get('dtype')!='bool'):
        raise ValueError('Four-view metadata/mask/tasks mismatch')
    dataset=common.Dataset(version='tomato_complete100_rgb_crop_saved_labels_r1',
        repo_id='fr3_sharpa/tomato_complete100_rgb_crop_b_saved_labels_r1',root=str(root/'b_coarse'),
        assets_dir=str(common.repository_root()/'assets'),asset_id='tomato_complete100_rgb_crop_b_saved_labels_r1_delta_30hz',
        sources=(str(crop),str(selection_path)),prompt_example=PROMPTS[1])
    config=recipe(name='pi05_sharpa_tomato_complete100_b_crop_fullft_delta_r1',
        exp_name='tomato_complete100_b_crop_delta_r1_s42',embodiment=FR3_SHARPA_CROP,dataset=dataset)
    return replace(config,model=replace(config.model,image_keys=IMAGE_KEYS,max_token_len=256),
        data=replace(config.data,extra_delta_transform=True,
            base_config=TaskDataConfig(prompt_from_task=True,task_view=TaskView(split_path=str(root/'splits.json'),split='train'))),
        batch_size=512,num_train_steps=8000,fsdp_devices=8,
        policy_metadata={**config.policy_metadata,'subtask_variant':'b_coarse',
            'training_action_semantics':'delta_arm_joint_target_absolute_hand',
            'selection_sha256':digest(selection_path),'crop_manifest_sha256':digest(crop),
            'roi_generation_mode':manifest['roi_generation_mode'],'model_image_keys':list(IMAGE_KEYS),
            'max_token_len':256,'visual_tokens':1024,'memory_to_pi05':False,
            'phases':list(PHASES),'goal':GOAL,'data_policy':'all_sources',
            'gpu_memory_preflight_required':True,'agent_enabled':False})
