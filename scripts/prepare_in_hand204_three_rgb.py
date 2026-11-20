"""Create a read-only three-image training view and exact cache without deleting sources."""
import json
import os
from pathlib import Path
import shutil

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks import tomato_in_hand as contract
from openpi.subtasks.prepare import digest,write_json
from training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_three_rgb import DATA_ROOT,CACHE_ROOT


def prepare(source_bundle, source_cache, output, cache_output):
    source_bundle, source_cache = Path(source_bundle).resolve(), Path(source_cache).resolve()
    output, cache_output = Path(output).resolve(), Path(cache_output).resolve()
    if output.exists() or cache_output.exists():raise FileExistsError('Three-view artifacts already exist')
    source = source_bundle/'b_coarse'
    bundle = json.loads((source_bundle/'bundle.json').read_text())
    old = json.loads((source_cache/'manifest.json').read_text())
    if (bundle['episodes'] != contract.EPISODES or bundle['frames'] != contract.FRAMES
            or old.get('complete') is not True or old['schema'] != 'rgb224_exact_pil_v2'):
        raise ValueError('Full exact204 source bundle/cache required')
    for rel,sha in bundle['files'].items():
        if digest(source_bundle/rel)!=sha:raise ValueError('Four-view source changed: '+rel)
    info = json.loads((source/'meta/info.json').read_text()); contract.validate_info(info)
    with new_artifact_directory(output) as stage:
        shutil.copytree(source/'meta',stage/'meta')
        shutil.copytree(source/'data',stage/'data')
        info['features'].pop('observation.images.interaction');info['total_videos']=3*contract.EPISODES
        write_json(stage/'meta/info.json',info)
        for camera in contract.CAMERAS[:3]:
            for video in (source/'videos').glob('*/'+camera+'/*.mp4'):
                target=stage/video.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
                # Source videos stay at the original path; this view only reads them.
                target.symlink_to(video)
        metadata={str(p.relative_to(stage)):digest(p) for p in (stage/'meta').iterdir() if p.is_file()}
        physical={str(p.relative_to(stage)):digest(p) for p in (stage/'data').rglob('*.parquet')}
        if any(sha!=digest(source/rel) for rel,sha in physical.items()):raise ValueError('State/action/subtask data differs')
        write_json(stage/'view.json',dict(episodes=contract.EPISODES,frames=contract.FRAMES,
            source_bundle=str(source_bundle),source_bundle_sha256=digest(source_bundle/'bundle.json'),
            source_cache_manifest_sha256=digest(source_cache/'manifest.json'),
            only_removed_interaction_view=True,metadata_sha256=metadata,physical_sha256=physical,
            source_files_deleted=False,source_files_modified=False))
    entries=[]
    with new_artifact_directory(cache_output) as stage:
        import numpy as np
        for index,item in enumerate(old['entries']):
            if item['camera'] not in contract.CAMERAS[:3]:continue
            p=(source_cache/item['cache_file']).resolve()
            if not p.is_relative_to(source_cache) or digest(p)!=item['cache_sha256'] or p.stat().st_mode&0o222:
                raise ValueError('Source cache content/mode changed')
            array=np.load(p,mmap_mode='r',allow_pickle=False)
            if array.dtype!=np.uint8 or array.shape!=(item['frames'],224,224,3):raise ValueError('Cache shape differs')
            del array
            target=stage/item['cache_file'];target.parent.mkdir(parents=True,exist_ok=True)
            os.link(p,target)  # Existing immutable arrays on the same training NAS; no source write.
            entries.append(dict(item))
            if len(entries)%50==0:print('Verified original-view cache',len(entries),'/612',flush=True)
        if len(entries)!=3*contract.EPISODES:raise ValueError('Incomplete three-camera coverage')
        write_json(stage/'manifest.json',{**old,'entries':entries,
            'source_metadata':{f'meta/{n}':digest(output/'meta'/n) for n in ['info.json','episodes.jsonl','tasks.jsonl']},
            'projection_source_manifest_sha256':digest(source_cache/'manifest.json')})
    # Recheck source payload after publishing both views.
    for rel,sha in bundle['files'].items():
        if digest(source_bundle/rel)!=sha:raise ValueError('Four-view source changed during preparation')
    print('PASS three-view data/cache; originals unchanged',flush=True)


if __name__=='__main__':
    prepare(contract.DEFAULT_BUNDLE,contract.DEFAULT_CACHE,DATA_ROOT,CACHE_ROOT)
