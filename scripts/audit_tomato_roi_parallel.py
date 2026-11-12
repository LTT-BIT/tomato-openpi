"""Run the existing full video/pixel/geometry audit independently per recording."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.audit_tomato_stable_roi import check_index, entries, find_episodes, statistics, verify_pixels


def check_episode(args):
    ep,row,folder,baseline,selection_sha = args
    receipt = json.loads((folder/'receipt.json').read_text())
    if receipt['episode_index']!=ep or receipt['contract']['selection_sha256']!=selection_sha:
        raise ValueError('Source/episode identity mismatch')
    for name,key in [('index.jsonl.gz','index_sha256'),('interaction224.mp4','video_sha256'),('detections.jsonl.gz','detections_sha256')]:
        if digest(folder/name)!=receipt[key]:
            raise ValueError('Changed candidate artifact')
    current,old = entries(folder),entries(baseline)
    check_index(current,row['length'])
    return dict(episode_index=ep,before=statistics(old),after=statistics(current,True),video=verify_pixels(row,folder,current))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--wait-all',action='store_true')
    args = parser.parse_args()
    if not 1<=args.workers<=16:parser.error('Use1..16 audit processes')
    selection = json.loads(args.selection.read_text())
    selection_sha = digest(args.selection)
    out = derived_output_path(selection,args.out)
    baseline = find_episodes(args.baseline)
    deadline = time.monotonic()+1800
    while True:
        folders = find_episodes(args.candidates)
        if set(folders)==set(range(100)):break
        if not args.wait_all or time.monotonic()>deadline:raise ValueError('All100 complete candidates required')
        time.sleep(1)
    if set(baseline)!=set(range(100)):raise ValueError('All100 baseline required')
    jobs = [(ep,selection['episodes'][ep],folder,baseline[ep],selection_sha) for ep,folder in sorted(folders.items())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for result in pool.map(check_episode,jobs):
            results.append(result)
            print('Full audit passed',result['episode_index'],flush=True)
    if sum(r['after']['frames'] for r in results)!=425556:raise ValueError('Unexpected complete100 frame count')
    result = dict(selection_sha256=selection_sha,episodes=100,frames=425556,episodes_checked=results,
                  motion_and_current_pixel_checks_passed=True,semantic_quality_accepted=False,
                  training_ready=False,training_started=False,source_writes=False,workers=args.workers,
                  helper_sha256=digest(Path(verify_pixels.__code__.co_filename)),
                  scope='Same full geometry/causality/decode/PTS/current-pixel checks as the serial audit; no independent semantic accuracy claim.')
    with new_artifact_directory(out) as staging:
        (staging/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
