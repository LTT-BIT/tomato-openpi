"""Read-only preparation contracts for all100 RGB/crop training experiments.

This module does not materialize crops, normalize states, load action weights,
construct a model or start training. Legacy normal93 recipes are untouched.
"""
import hashlib
import json
from pathlib import Path
import random

from openpi.subtasks.tomato import ACTION_PROMPTS, COARSE_SEQUENCE, SOURCE_ACTION_NAMES, SOURCE_STATE_NAMES

VERSION = 'tomato-complete100-rgb-crop-preparation-v1'
IMAGE_KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb', 'interaction_0_rgb')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def freeze_selection(manifest_path, *, verify_sources=True):
    """Freeze exactly the existing100 identities, assign ALL to action training.

    Agent split is provenance only. No raw/annotation changes or directory scan.
    Input checksum/schema verification is separate from training readiness.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest['episodes']
    if len(episodes) != 100:
        raise ValueError('Exactly100 complete source recordings required')
    seen_ids, seen_sources, rows = set(), set(), []
    for i, e in enumerate(episodes):
        eid, source = e['episode_id'], e['source_recording_id']
        if not eid or not source or eid in seen_ids or source in seen_sources:
            raise ValueError('Duplicate or missing original recording identity')
        seen_ids.add(eid); seen_sources.add(source)
        info = e['info']
        if (e['fps'] != 30 or info.get('fps') != 30 or e['length'] <= 0
                or info['features']['observation.state'].get('shape') != [116]
                or info['features']['observation.state'].get('names') != list(SOURCE_STATE_NAMES)
                or info['features']['action'].get('shape') != [58]
                or info['features']['action'].get('names') != list(SOURCE_ACTION_NAMES)):
            raise ValueError('30Hz converter state116/action58 named schema required')
        annotation = e['annotation']
        segments = annotation['segments']
        if tuple(r['label'] for r in segments) != COARSE_SEQUENCE:
            raise ValueError('Seven complete ordered phases required')
        end = 0
        for r in segments:
            if (r.get('reviewed') is not True or type(r['start_frame']) is not int
                    or type(r['end_frame']) is not int or r['start_frame'] != end
                    or r['end_frame'] <= end):
                raise ValueError('Reviewed contiguous phase coverage required')
            end = r['end_frame']
        if end != e['length']:
            raise ValueError('Phase coverage does not match full episode')
        root = Path(e['dataset_root']).resolve()
        hashes = e['input_sha256']
        expected_files = [Path(e['parquet']), *(Path(p) for p in e['videos'])]
        if len(e['videos']) != 3 or len(set(e['videos'])) != 3:
            raise ValueError('Exactly three distinct original camera files required')
        for path in expected_files:
            relative = str(path.resolve().relative_to(root))
            if relative not in hashes:
                raise ValueError('Missing camera/parquet source checksum')
        phase_differences = None; metadata_sha256 = None
        if verify_sources:
            current_info=json.loads((root/'meta/info.json').read_text())
            current_entries=[json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
            matching=[r for r in current_entries if r['episode_index']==e['episode_index']]
            if (current_info.get('fps')!=30 or current_info.get('codebase_version')!='v2.1'
                    or any(current_info['features'][k]!=info['features'][k] for k in ('observation.state','action'))
                    or len(matching)!=1 or matching[0]['length']!=e['length']
                    or matching[0].get('source_recording_id')!=source):
                raise ValueError('Actual source metadata/schema differs from frozen recording')
            metadata_sha256={p:digest(root/p) for p in ('meta/info.json','meta/episodes.jsonl')}
            if digest(annotation['path']) != annotation['sha256']:
                raise ValueError('Annotation changed since Agent manifest freeze')
            actual = json.loads(Path(annotation['path']).read_text())
            # Agent experiments may use boundary proposals without rewriting
            # saved source labels. Record both; never silently choose a proposal.
            saved = actual['segments']
            if (tuple(r['label'] for r in saved) != COARSE_SEQUENCE
                    or any(r.get('reviewed') is not True for r in saved)
                    or saved[0]['start_frame'] != 0 or saved[-1]['end_frame'] != e['length']
                    or any(r['end_frame'] <= r['start_frame'] for r in saved)
                    or any(a['end_frame'] != b['start_frame'] for a,b in zip(saved,saved[1:]))):
                raise ValueError('Saved source must also contain reviewed complete phases')
            phase_differences = [dict(phase_index=j, saved=a, agent_effective=b)
                for j,(a,b) in enumerate(zip(saved,segments)) if a != b]
            for relative, expected in hashes.items():
                path = (root/relative).resolve()
                if not path.is_relative_to(root) or digest(path) != expected:
                    raise ValueError(f'Source checksum mismatch: {path}')
        rows.append(dict(episode_index=i, dataset=str(root), source_episode_index=e['episode_index'],
            agent_episode_id=eid, recording_group=source, length=e['length'], fps=30,
            annotation=annotation['path'], annotation_sha256=annotation['sha256'],
            source_sha256=hashes, agent_split_provenance=e['split'], action_split='train',
            metadata_sha256=metadata_sha256,
            phase_label_status=('not_verified' if phase_differences is None else
                'needs_resolution' if phase_differences else 'saved_matches_agent'),
            phase_differences=phase_differences,
            roi_coverage='not_prepared'))
    return dict(version=VERSION, name='tomato_complete100_rgb_crop_all_sources_v1',
        agent_manifest_sha256=digest(manifest_path), agent_manifest=str(manifest_path.resolve()),
        data_policy='all_sources', splits={'train':list(range(100))}, episodes=rows,
        source_files_verified=verify_sources, training_ready=False,
        phase_label_resolution_required=sum(r['phase_label_status']=='needs_resolution' for r in rows),
        phase_label_policy='resolve saved-vs-effective differences before materializing stage prompts',
        normalization='pending: recompute on all100 with new action/data identity',
        roi='pending: full per-frame coverage/provenance; existing preview is partial',
        agent_split_changed=False, deployment_modified=False)


def full_text(prompt, bins, memory_text=''):
    if len(bins) != 58 or any(type(v) is not int or not -1 <= v <= 255 for v in bins):
        raise ValueError('Exactly58 state bins in[-1,255] required')
    text = prompt + ((' Memory: '+memory_text) if memory_text else '')
    cleaned = text.strip().replace('_', ' ').replace('\n', ' ')
    return f"Task: {cleaned}, State: {' '.join(map(str,bins))};\nAction: "


def sequence_slots(*, text_capacity=256, history_tokens=0, views=4, action_horizon=50):
    """Architectural lengths, not compute/FLOP or active-mask estimates."""
    if any(type(v) is not int or v < 0 for v in (text_capacity, history_tokens, views, action_horizon)):
        raise ValueError('Nonnegative integer sequence dimensions required')
    if not text_capacity or not views or not action_horizon:
        raise ValueError('Text, views and horizon must be positive')
    visual = views*(224//14)**2
    return dict(visual_tokens=visual, text_slots=text_capacity, history_tokens=history_tokens,
        prefix_slots=visual+text_capacity+history_tokens, action_positions=action_horizon,
        prefix_plus_action_slots=visual+text_capacity+history_tokens+action_horizon,
        count_scope='token positions only; prefix/action experts and masks differ')


def audit_tokens(encode, *, tokenizer_sha256, normalized_rows=None):
    """Real tokenizer on synthetic bins; optionally exact normalized state-bin rows.

    No token truncation and no fabricated physical-result facts. Synthetic bin
    coverage cannot stand in for the all100 post-normalization distribution.
    """
    rng = random.Random(42)
    synthetic = [[v]*58 for v in range(-1,256)] + [[rng.randint(-1,255) for _ in range(58)] for _ in range(200)]
    reports=[]
    for action,prompt in ACTION_PROMPTS.items():
        counts=[len(encode(full_text(prompt,b),add_bos=True)) for b in synthetic]
        # Independently check state-number separability for the additive maximum.
        base=len(encode(full_text(prompt,[0]*58),add_bos=True)); increments=[]
        for value in range(-1,256):
            inc=len(encode(full_text(prompt,[value]+[0]*57),add_bos=True))-base
            uniform=len(encode(full_text(prompt,[value]*58),add_bos=True))
            if uniform != base+58*inc:
                raise ValueError('Tokenizer state boundaries changed; cannot use old bound')
            increments.append(inc)
        bound=base+58*max(increments)
        reports.append(dict(action=action, prompt_alone=len(encode(prompt)),
            full_text_upper_bound=bound, tested_max=max(counts), spare_at_bound=256-bound,
            within_256=bound<=256))
    real=[]; seen=set(); normalizations=set()
    for row in normalized_rows or []:
        if row.get('action') not in ACTION_PROMPTS:
            raise ValueError('Known action required')
        # Bins must come from the new normalization/tokenizer path, not raw angles.
        if not row.get('normalization_sha256') or not row.get('observation_id'):
            raise ValueError('Explicit normalization and observation identity required')
        if row['observation_id'] in seen:
            raise ValueError('Duplicate real observation identity')
        seen.add(row['observation_id']); normalizations.add(row['normalization_sha256'])
        n=len(encode(full_text(ACTION_PROMPTS[row['action']],row['state_bins']),add_bos=True))
        real.append(dict(observation_id=row['observation_id'],tokens=n,exceeds_256=n>256,
                         normalization_sha256=row['normalization_sha256']))
    if len(normalizations)>1:
        raise ValueError('Do not mix normalization identities in one token distribution')
    counts=sorted(r['tokens'] for r in real)
    def quantile(fraction):
        if not counts:return None
        index=(len(counts)-1)*fraction; lower=int(index); upper=min(lower+1,len(counts)-1)
        return counts[lower]+(counts[upper]-counts[lower])*(index-lower)
    return dict(version=VERSION,tokenizer_sha256=tokenizer_sha256,
        synthetic_cases_per_action=len(synthetic),base_prompt_audit=reports,
        real_normalized_state_rows=real,real_distribution_available=bool(real),
        real_distribution=dict(rows=len(real),p50=quantile(.5),p95=quantile(.95),
            maximum=max(counts) if counts else None,exceeds_256=sum(r['exceeds_256'] for r in real)),
        old_three_view=sequence_slots(views=3),
        scheme1_four_view=sequence_slots(),
        scheme2_candidates=[dict(kind='learned_history_unimplemented',**sequence_slots(history_tokens=n)) for n in (32,64)],
        text_memory_candidate=dict(status='unimplemented; requires reliable facts',**sequence_slots(text_capacity=320)),
        truncation_performed=False, training_started=False)
