"""Bind verified image-independent all100 statistics to actual four-view A/B data, not quality approval."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import IMAGE_KEYS,digest


def main():
    import numpy as np
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import AssetsConfig,LeRobotDatasetSpec,LeRobotFr3SharpaCropDataConfig,TaskDataConfig
    from openpi.training.norm_provenance import SCHEMA_VERSION,build_contract,validate_metadata
    from openpi.training.task_view import TaskView
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source-norm',type=Path,required=True)
    p.add_argument('--verification',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--variant-suffix',default='',help='New dataset/asset version, e.g. _stable_r1; never overwrites old assets')
    a=p.parse_args();verification=json.loads(a.verification.read_text());source=json.loads((a.source_norm/'receipt.json').read_text())
    if a.variant_suffix and (not a.variant_suffix.startswith('_') or any(c not in '_abcdefghijklmnopqrstuvwxyz0123456789' for c in a.variant_suffix)):
        p.error('Use an underscore-prefixed lowercase version suffix')
    crop=json.loads((a.bundle/'crop_manifest.json').read_text())
    if not (verification['base']['physical_columns_equal'] and verification['base']['original_sources_hash_unchanged']
        and verification['real_four_view_loader']['samples']==300 and verification['real_four_view_loader']['normalized_token_lengths_match']
        and verification['crop_manifest_sha256']==digest(a.bundle/'crop_manifest.json')
        and source['normalization_sha256']==digest(a.source_norm/'norm_stats.json')
        and source['selection_sha256']==crop['selection_sha256'] and source['frames']==425556):
        raise ValueError('Verified actual dataset/vector/token equality required before provenance binding')
    results=[]
    for mode,short in [('a_goal','a'),('b_coarse','b')]:
        repo=f'fr3_sharpa/tomato_complete100_rgb_crop_{short}_saved_labels_r1{a.variant_suffix}'
        asset=f'tomato_complete100_rgb_crop_{short}_saved_labels_r1{a.variant_suffix}_delta_30hz'
        data=LeRobotFr3SharpaCropDataConfig(repo_id=repo,assets=AssetsConfig(assets_dir=str(a.assets),asset_id=asset),
            extra_delta_transform=True,action_sequence_keys=('action',),action_hz=30,video_backend='pyav',
            lerobot_datasets=(LeRobotDatasetSpec(repo_id=repo,root=str(a.bundle/mode),weight=1.),),
            base_config=TaskDataConfig(prompt_from_task=True,task_view=TaskView(split_path=str(a.bundle/'splits.json'),split='train')))
        config=SimpleNamespace(data=data,model=Pi0Config(pi05=True,action_dim=58,image_keys=IMAGE_KEYS,max_token_len=256),policy_metadata={})
        contract=build_contract(config);out=a.assets/asset
        with new_artifact_directory(out) as staging:
            shutil.copy2(a.source_norm/'norm_stats.json',staging/'norm_stats.json')
            metadata=dict(schema_version=SCHEMA_VERSION,contract=contract,norm_stats_sha256=digest(staging/'norm_stats.json'),
                repo_id=repo,frames=425556,episodes=100,training_frames=425556,training_episodes=100,action_horizon=50,
                vector_counts=source['vector_counts'],lerobot_comparison_samples=300,accumulation_dtype='float64',
                datasets=[dict(repo_id=repo,frames=425556,episodes=100)],
                sampling='All100 frames equally weighted; forward50 windows padded at episode end; no task/phase arm holds.',
                numeric_source_receipt_sha256=digest(a.source_norm/'receipt.json'),verification_sha256=digest(a.verification),
                image_independence_verified=True,quality_approval=False,training_started=False)
            (staging/'norm_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            if build_contract(config)!=contract:raise ValueError('Dataset changed during contract binding')
            status=validate_metadata(config,staging,stats_sha256=digest(staging/'norm_stats.json'))
        results.append(dict(variant=mode,path=str(out),norm_stats_sha256=digest(out/'norm_stats.json'),
            norm_metadata_sha256=digest(out/'norm_metadata.json'),status=status))
    print(json.dumps(dict(assets=results,roi_quality_approved=False,training_started=False),indent=2))


if __name__=='__main__':main()
