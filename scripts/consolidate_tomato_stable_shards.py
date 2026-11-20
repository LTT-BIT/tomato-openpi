"""Rebuild four logical manifests from disjoint workers' completed hash-verified episodes."""
import argparse
import json
from pathlib import Path

from openpi.shared.local_storage import atomic_write_json
from openpi.subtasks.rgb100 import derived_output_path, digest


def consolidate(root,selection_path):
    selection = json.loads(selection_path.read_text())
    derived_output_path(selection,root/'_output_guard')
    all_rows = []
    common_contract = None
    for shard in range(4):
        folder = root/f'shard{shard}'
        contract = json.loads((folder/'contract.json').read_text())
        if contract['selection_sha256']!=digest(selection_path) or (common_contract is not None and contract!=common_contract):
            raise ValueError('Shard selection/generator mismatch')
        common_contract = contract
        rows = []
        for ep in range(shard,100,4):
            episode = folder/f'episode_{ep:06d}'
            row = json.loads((episode/'receipt.json').read_text())
            if row['episode_index']!=ep or row['contract']!=contract or row['frames']!=selection['episodes'][ep]['length']:
                raise ValueError('Episode identity/contract mismatch')
            for file,key in [('interaction224.mp4','video_sha256'),('index.jsonl.gz','index_sha256'),('detections.jsonl.gz','detections_sha256')]:
                if digest(episode/file)!=row[key]:
                    raise ValueError('Episode artifact hash mismatch')
            rows.append(row)
        result = dict(contract=contract,episodes=rows,episodes_completed=25,frames=sum(r['frames'] for r in rows),
                      all100_frames_enumerated=False,algorithm_valid_frames=sum(r['algorithm_valid_frames'] for r in rows),
                      full_roi_coverage=False,independent_quality_approved=False,training_ready=False,
                      training_started=False,source_writes=False,
                      note='Logical25-episode shard rebuilt after disjoint workers; all100 status belongs to the combined bundle.')
        atomic_write_json(folder/'candidate_manifest.json',result)
        atomic_write_json(folder/'progress.json',dict(contract=contract,episodes_completed=25,frames_processed=result['frames'],
                                                    algorithm_valid_frames=result['algorithm_valid_frames'],training_ready=False))
        all_rows.extend(rows)
    if {r['episode_index'] for r in all_rows}!=set(range(100)) or sum(r['frames'] for r in all_rows)!=425556:
        raise ValueError('Incomplete all100')
    return dict(episodes=100,frames=425556,algorithm_valid_frames=sum(r['algorithm_valid_frames'] for r in all_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.root,args.selection),indent=2))


if __name__=='__main__':
    main()
