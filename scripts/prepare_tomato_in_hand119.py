"""Freeze and materialize a NEW full119 five-phase four-RGB B training bundle."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from openpi.subtasks import tomato_in_hand as contract


def materialize(selection, output, tokenizer):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from openpi.shared.local_storage import new_artifact_directory
    from openpi.subtasks.binding import bind_annotation, verify_annotation
    from openpi.subtasks.prepare import digest, safe_file, stats, write_json, write_jsonl
    if (selection.get('version') != contract.VERSION or not selection.get('source_files_verified')
            or len(selection['episodes']) != contract.EPISODES or selection['frames'] != contract.FRAMES):
        raise ValueError('Verified full119 frozen selection required')
    output = output.resolve()
    roots = [Path(r['source_dataset']).resolve() for r in selection['episodes']]
    if any(output.is_relative_to(r) or r.is_relative_to(output) for r in roots):
        raise ValueError('Training output must be separate from sources')
    budget = contract.audit_tokens(tokenizer)
    physical = {'observation.state': [], 'action': []}; mappings=[]; episodes=[]; episode_stats=[]; offset=0
    source_files = {}; annotation_snapshots=[]
    with new_artifact_directory(output) as stage:
        dest=stage/'b_coarse'
        for row in selection['episodes']:
            root=Path(row['source_dataset']); old=row['source_episode_index']; ep=row['episode_index']
            info=json.loads((root/'meta/info.json').read_text()); contract.validate_info(info)
            if digest(row['annotation']) != row['annotation_sha256'] or digest(root/'roi_manifest.json') != row['roi_manifest_sha256']:
                raise ValueError('Annotation/ROI changed after freeze')
            annotation=Path(row['annotation']).read_bytes(); doc=json.loads(annotation)
            segments=contract.reviewed_segments(doc,row['length']); verify_annotation(root,doc)
            for relative,sha in row['source_sha256'].items():source_files[str(safe_file(root,relative))]=sha
            path=info['data_path'].format(episode_chunk=old//1000,episode_index=old)
            original=pq.read_table(safe_file(root,path)); table=original
            if (len(table)!=row['length'] or table['frame_index'].to_pylist()!=list(range(len(table)))
                    or set(table['episode_index'].to_pylist())!={old}):raise ValueError('Invalid raw indexing')
            np.testing.assert_allclose(table['timestamp'].to_numpy(),np.arange(len(table))/30,atol=1e-5,rtol=0)
            phases=np.empty(len(table),dtype=np.int64);tasks=np.empty_like(phases)
            for i,r in enumerate(segments):
                phases[r['start_frame']:r['end_frame']]=i
                tasks[r['start_frame']:r['end_frame']]=contract.TASKS.index(r['text'])
            updates={'episode_index':np.full(len(table),ep),'index':np.arange(offset,offset+len(table)),
                'task_index':tasks,'coarse_phase_index':phases,'observation.interaction_available':np.ones(len(table),bool)}
            for k,v in updates.items():
                dtype=table.schema.field(k).type if k in table.column_names else pa.bool_() if k.endswith('available') else pa.int64()
                col=pa.array(v,type=dtype)
                table=table.set_column(table.schema.get_field_index(k),k,col) if k in table.column_names else table.append_column(k,col)
            for k in original.column_names:
                if k not in updates and not original[k].equals(table[k]):raise ValueError('Physical source column changed')
            target=dest/f'data/chunk-000/episode_{ep:06d}.parquet';target.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,target)
            for cam in contract.CAMERAS:
                old_video=info['video_path'].format(episode_chunk=old//1000,episode_index=old,video_key=cam)
                target=dest/f'videos/chunk-000/{cam}/episode_{ep:06d}.mp4';target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(safe_file(root,old_video),target)
                if digest(target)!=row['source_sha256'][old_video]:raise ValueError('Video copy changed')
            per_episode={}
            for k,arrays in physical.items():
                value=np.asarray(table[k].to_pylist(),dtype=np.float64)
                if not np.isfinite(value).all():raise ValueError('Nonfinite physical data')
                arrays.append(value);per_episode[k]=stats([value])
            episode_stats.append(dict(episode_index=ep,stats=per_episode))
            entry={**row['entry'],'episode_index':ep,'tasks':list(contract.TASKS)};entry.pop('task_index',None);episodes.append(entry)
            mappings.append({k:v for k,v in row.items() if k!='entry'})
            annotation_snapshots.append((doc,annotation));offset+=len(table)
            print(f'Materialized {ep+1}/{contract.EPISODES}',flush=True)
        info=copy.deepcopy(info);info.update(total_episodes=contract.EPISODES,total_frames=offset,total_tasks=len(contract.TASKS),
            total_videos=4*contract.EPISODES,total_chunks=1,splits={'train':f'0:{contract.EPISODES}'},
            data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
        info['features']['coarse_phase_index']=dict(dtype='int64',shape=[1],names=None)
        info['features']['observation.interaction_available']=dict(dtype='bool',shape=[1],names=None)
        write_json(dest/'meta/info.json',info);write_json(dest/'meta/stats.json',{k:stats(v) for k,v in physical.items()})
        write_jsonl(dest/'meta/episodes.jsonl',episodes);write_jsonl(dest/'meta/episodes_stats.jsonl',episode_stats)
        write_jsonl(dest/'meta/tasks.jsonl',[dict(task_index=i,task=t) for i,t in enumerate(contract.TASKS)])
        for ep,(doc,snapshot) in enumerate(annotation_snapshots):
            bound=bind_annotation(dest,output/'b_coarse',ep,doc,f'provenance/source_annotations/episode_{ep:06d}.json',snapshot_bytes=snapshot)
            write_json(dest/f'annotations/episode_{ep:06d}.json',bound)
        for path,sha in source_files.items():
            if digest(path)!=sha:raise ValueError('Source changed during preparation')
        for row in selection['episodes']:
            if digest(row['annotation'])!=row['annotation_sha256']:raise ValueError('Source annotation changed')
        write_json(stage/'selection.json',selection)
        files={str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        bundle=dict(version=contract.VERSION,profile=contract.PROFILE,episodes=contract.EPISODES,frames=offset,
            phases=contract.PHASES,sequence=contract.SEQUENCE,prompts=contract.TASKS,token_budget=budget,
            data_policy='all_sources',mapping=mappings,files=files,raw_sources_unchanged=True,
            physical_columns_equal=True,offline_roi_uses_future_coordinates=True,
            image_availability_means_local_rgb_exists=True,independent_semantic_accuracy_measured=False,
            input_preparation_complete=True,training_ready=False,training_started=False)
        write_json(stage/'bundle.json',bundle)
    return bundle


def main():
    from openpi.subtasks.prepare import write_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=contract.DEFAULT_SOURCE)
    p.add_argument('--out',type=Path,default=contract.DEFAULT_BUNDLE);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.selection.exists():raise FileExistsError('Use new output paths')
    frozen=contract.freeze(a.source);write_json(a.selection,frozen)
    import sentencepiece
    tokenizer=sentencepiece.SentencePieceProcessor(model_file=str(a.tokenizer))
    result=materialize(frozen,a.out,tokenizer);print(json.dumps({k:result[k] for k in ['episodes','frames','token_budget']},indent=2))


if __name__=='__main__':main()
