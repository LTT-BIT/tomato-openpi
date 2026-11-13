"""All100 four-view B candidate. Refuses partial ROI/base three-view bundles.

No training starts by importing this module. Set PI05_TOMATO100_CROP_BUNDLE to
an independently materialized four-view bundle with a bound crop_manifest.json.
"""
from dataclasses import fields, replace
import json
import os
from pathlib import Path

from openpi.subtasks.rgb100 import IMAGE_KEYS, digest
from openpi.subtasks.tomato import GOAL, PHASES, PROMPTS
from openpi.training.config import ScheduledTrainConfig, TaskDataConfig
from openpi.training.task_view import TaskView
from training_configs import common
from training_configs.embodiments.fr3_sharpa_crop import FR3_SHARPA_CROP
from training_configs.recipes.pi05_fullft_v1 import build_config as recipe


def offline_preparation(root, manifest, selection_sha, report_path):
    """Accept the exact reviewed offline input without changing its frozen manifest."""
    if not report_path:
        return False
    report = json.loads(Path(report_path).expanduser().read_text())
    validation = Path(__file__).parent / 'records' / 'complete100_offline_local_20261112_validation_r11.json'
    expected = {
        'status': 'offline_training_inputs_accepted',
        'crop_manifest_sha256': digest(root / 'crop_manifest.json'),
        'selection_sha256': selection_sha,
        'validation_sha256': digest(validation),
        'episodes': 100, 'frames': 425556,
        'independent_semantic_accuracy_measured': False,
        'online_roi_validated': False,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('Offline preparation report does not match reviewed data')
    receipt = report.get('transfer', {})
    if (receipt.get('passed') is not True or receipt.get('errors') != []
            or receipt.get('unexpected_files') != [] or receipt.get('episodes') != 100
            or receipt.get('frames') != 425556):
        raise ValueError('Full server transfer verification required')
    review = json.loads(validation.read_text())
    acceptance = review['acceptance']
    if (manifest.get('stream_contract') != 'offline_bidirectional_local_rgb_v6'
            or manifest.get('roi_generation_mode') != 'offline_bidirectional_annotation_candidate'
            or any(acceptance.get(k) != v for k, v in {
                'frames': 425556, 'pending_frames': 0, 'black_frames': 0,
                'full_view_fallback_frames': 0, 'center_steps_over_3pct': 0,
                'scale_steps_over_2pct': 0, 'all_frames_decoded': True,
                'full_image_coverage': True, 'full_local_geometry_coverage': True,
            }.items())
            or review['actual_loader']['crop_manifest_sha256'] != expected['crop_manifest_sha256']):
        raise ValueError('Reviewed full offline input coverage required')
    return True


def build_config(*,variant='b_coarse'):
    if variant not in ('a_goal','b_coarse'):raise ValueError('Unknown all100 crop variant')
    short='a' if variant=='a_goal' else 'b'
    environment='PI05_TOMATO100_CROP_A_BUNDLE' if short=='a' else 'PI05_TOMATO100_CROP_BUNDLE'
    value=os.environ.get(environment)
    if not value:raise ValueError(f'Set {environment} after full100 ROI preparation')
    root=Path(value).expanduser().resolve()
    crop=root/'crop_manifest.json'
    if not crop.is_file():raise ValueError('Missing crop_manifest.json; base three-view bundle is not ready')
    manifest=json.loads(crop.read_text())
    records=Path(__file__).parent/'records'
    selection_path=records/'complete100_rgb_crop_20261111_saved_labels_r1.json'
    selection=json.loads(selection_path.read_text())
    stream_contract=manifest.get('stream_contract','black_crop_false_mask_on_loss_v1')
    full_input=stream_contract in ('current_rgb_local_or_full_fallback_v5','offline_bidirectional_local_rgb_v6')
    prepared=offline_preparation(root,manifest,digest(selection_path),os.environ.get('PI05_TOMATO100_PREPARATION_REPORT'))
    coverage_ok=(manifest.get('full_image_coverage') is True
                 and manifest.get('image_available_frames')==425556
                 and manifest.get('interaction_mask_key')=='observation.interaction_available') if full_input else manifest.get('full_roi_coverage') is True
    if (manifest.get('selection_sha256')!=digest(selection_path)
            or not coverage_ok
            or (manifest.get('training_ready') is not True and not prepared)
            or manifest.get('episodes')!=100 or manifest.get('frames')!=425556
            or (manifest.get('roi_generation_mode') not in ('manual_offline_interpolated','causal_prediction','offline_bidirectional_annotation') and not prepared)
            or not manifest.get('roi_generator_signature')):
        raise ValueError('Complete hash-bound100 ROI coverage/provenance required')
    if stream_contract=='offline_bidirectional_local_rgb_v6' and (
        (manifest.get('roi_generation_mode')!='offline_bidirectional_annotation' and not prepared)
        or manifest['roi_generator_signature'].get('offline_annotation_only') is not True):
        raise ValueError('Explicit offline future annotation provenance required')
    if stream_contract not in ('black_crop_false_mask_on_loss_v1','continuous_current_pixels_false_mask_on_loss_v1','continuous_current_pixels_state_guarded_v2','continuous_current_pixels_coverage_v3','continuous_current_pixels_transport_v4','current_rgb_local_or_full_fallback_v5','offline_bidirectional_local_rgb_v6'):
        raise ValueError('Unknown crop stream contract')
    suffix={'continuous_current_pixels_false_mask_on_loss_v1':'_stable_r1','continuous_current_pixels_state_guarded_v2':'_state_guarded_r1','continuous_current_pixels_coverage_v3':'_coverage_r4','continuous_current_pixels_transport_v4':'_transport_r5','current_rgb_local_or_full_fallback_v5':'_available_r6','offline_bidirectional_local_rgb_v6':'_offline_r11'}.get(stream_contract,'')
    splits=json.loads((root/'splits.json').read_text())
    if splits['splits']!={'train':list(range(100))}:raise ValueError('All100 action training required')
    info=json.loads((root/variant/'meta/info.json').read_text())
    tasks=[json.loads(line)['task'] for line in (root/variant/'meta/tasks.jsonl').read_text().splitlines()]
    expected_prompts=[GOAL] if short=='a' else list(dict.fromkeys(PROMPTS))
    if (info['total_episodes']!=100 or info['total_frames']!=sum(r['length'] for r in selection['episodes'])
            or tasks!=expected_prompts
            or set(k for k,v in info['features'].items() if v.get('dtype')=='video')!=set(FR3_SHARPA_CROP.camera_keys)
            or info['features'].get('observation.interaction_valid',{}).get('dtype')!='bool'):
        raise ValueError('Four-view metadata/mask/tasks mismatch')
    if full_input and info['features'].get('observation.interaction_available',{}).get('dtype')!='bool':
        raise ValueError('Full input coverage requires separate image availability')
    dataset=common.Dataset(version=f'tomato_complete100_rgb_crop_saved_labels_r1{suffix}',
        repo_id=f'fr3_sharpa/tomato_complete100_rgb_crop_{short}_saved_labels_r1{suffix}',root=str(root/variant),
        assets_dir=str(common.repository_root()/'assets'),asset_id=f'tomato_complete100_rgb_crop_{short}_saved_labels_r1{suffix}_delta_30hz',
        sources=(str(crop),str(selection_path)),prompt_example=GOAL if short=='a' else PROMPTS[1])
    config=recipe(name=f'pi05_sharpa_tomato_complete100_{short}_crop_fullft_delta_r1{suffix}',
        exp_name=f'tomato_complete100_{short}_crop_delta_r1{suffix}_s42',embodiment=FR3_SHARPA_CROP,dataset=dataset)
    config=replace(config,exp_name=config.exp_name+"_b256_s16000_c4000",model=replace(config.model,image_keys=IMAGE_KEYS,max_token_len=256),
        data=replace(config.data,extra_delta_transform=True,
            interaction_mask_key='observation.interaction_available' if full_input else 'observation.interaction_valid',
            base_config=TaskDataConfig(prompt_from_task=True,task_view=TaskView(split_path=str(root/'splits.json'),split='train'))),
        batch_size=256,num_train_steps=16000,fsdp_devices=8,save_interval=4000,keep_period=4000,
        policy_metadata={**config.policy_metadata,'subtask_variant':variant,
            'training_action_semantics':'delta_arm_joint_target_absolute_hand',
            'selection_sha256':digest(selection_path),'crop_manifest_sha256':digest(crop),
            'roi_generation_mode':manifest['roi_generation_mode'],'model_image_keys':list(IMAGE_KEYS),
            'crop_stream_contract':stream_contract,
            'interaction_mask_key':'observation.interaction_available' if full_input else 'observation.interaction_valid',
            'offline_roi_uses_future_annotation':stream_contract=='offline_bidirectional_local_rgb_v6',
            'max_token_len':256,'visual_tokens':1024,'memory_to_pi05':False,
            'phases':list(PHASES),'goal':GOAL,'data_policy':'all_sources',
            'offline_training_preparation_sha256':digest(Path(os.environ['PI05_TOMATO100_PREPARATION_REPORT'])) if prepared else None,
            'independent_semantic_accuracy_measured':False,
            'gpu_memory_preflight_required':True,'agent_enabled':False})

    # Completed-step numbering preserves all four requested checkpoints.
    return ScheduledTrainConfig(
        **{field.name:getattr(config,field.name) for field in fields(config)},
        save_before_step=1,save_steps_before=(),save_interval_after=4000,
    )
