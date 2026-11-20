"""Account for every recovered/regressed crop frame; phase labels are reporting only."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from openpi.subtasks.rgb100 import digest, write_new
from scripts.audit_tomato_stable_roi import entries, find_episodes


def profile(selection_path, before, after):
    selection = json.loads(selection_path.read_text())
    old_folders, new_folders = find_episodes(before), find_episodes(after)
    if set(old_folders)!=set(range(100)) or set(new_folders)!=set(range(100)):
        raise ValueError('All100 before and after required')
    total = collections.Counter()
    phases = collections.defaultdict(collections.Counter)
    episodes, runs, sizes = [], [], []
    for ep,folder in sorted(new_folders.items()):
        current,old = entries(folder),entries(old_folders[ep])
        row = selection['episodes'][ep]
        for source,rows in [(folder,current),(old_folders[ep],old)]:
            receipt = json.loads((source/'receipt.json').read_text())
            if (digest(source/'index.jsonl.gz')!=receipt['index_sha256']
                or receipt['contract']['selection_sha256']!=digest(selection_path)
                or len(rows)!=row['length']):
                raise ValueError('Changed ROI/source identity')
        counts = collections.Counter()
        start = None
        for i,(a,b) in enumerate(zip(old,current,strict=True)):
            if a['source_frame']!=i or b['source_frame']!=i:
                raise ValueError('Frame order mismatch')
            counts['frames'] += 1
            counts['before_valid'] += a['valid']
            counts['after_valid'] += b['valid']
            counts['recovered'] += not a['valid'] and b['valid']
            counts['regressed'] += a['valid'] and not b['valid']
            if not b['valid']:
                counts[b['reason']] += 1
                observed = b['observed_interaction_box']
                if observed is not None and max(np.asarray(observed)[2:]-np.asarray(observed)[:2])>.95*min(b['source_size']):
                    counts['observed_region_wider_than_maximum_window'] += 1
                if start is None:
                    start = i
            elif start is not None:
                runs.append(dict(episode_index=ep,start=start,end=i,frames=i-start))
                start = None
            sizes.append(b['display_crop_box'][2]-b['display_crop_box'][0])
        if start is not None:
            runs.append(dict(episode_index=ep,start=start,end=len(current),frames=len(current)-start))
        for segment in row['training_segments']:
            phase = phases[segment['label']]
            for b in current[segment['start_frame']:segment['end_frame']]:
                phase['frames'] += 1
                phase['valid'] += b['valid']
                if not b['valid']:
                    phase[b['reason']] += 1
        total.update(counts)
        episodes.append(dict(episode_index=ep,**counts))
    return dict(selection_sha256=digest(selection_path),before=str(before),after=str(after),
                totals=dict(total),episodes=episodes,phases={p:dict(v) for p,v in phases.items()},
                longest_remaining_invalid_runs=sorted(runs,key=lambda r:-r['frames'])[:30],
                display_side_quantiles=dict(zip(('min','p50','p95','max'),np.quantile(sizes,[0,.5,.95,1]).tolist(),strict=True)),
                phase_labels_used_only_for_reporting=True,algorithm_validity_is_not_accuracy=True,
                independent_semantic_quality_accepted=False,source_writes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    write_new(args.out,profile(args.selection,args.before,args.after))


if __name__=='__main__':
    main()
