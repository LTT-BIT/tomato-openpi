"""Sequentially decode immutable videos to exact uint8 training-size frame arrays."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time


def worker_init(cpu_queue):
    os.sched_setaffinity(0, cpu_queue.get())
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'


def build_video(job):
    import av
    import numpy as np
    import torch
    import torchvision
    from openpi.policies.fr3_sharpa_policy import _parse_image
    from openpi_client.image_tools import resize_with_pad
    from openpi.training.frame_cache import sha256

    torch.set_num_threads(1)
    torchvision.set_video_backend('pyav')
    source, destination, episode, camera, frames, fps = job
    source, destination = Path(source), Path(destination)
    receipt = destination.with_suffix('.json')
    source_sha = sha256(source)
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if (old['source_sha256'] == source_sha and destination.exists()
                and sha256(destination) == old['cache_sha256']):
            return old
        raise ValueError('Conflicting previous cache: ' + str(destination))
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.partial.npy')
    if temp.exists():
        raise FileExistsError(temp)
    output = np.lib.format.open_memmap(temp, mode='w+', dtype=np.uint8, shape=(frames, 224, 224, 3))
    checks = sorted(set([0, frames // 4, frames // 2, 3 * frames // 4, frames - 1]))
    t = time.monotonic()
    count = 0
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index >= frames or abs(float(frame.time) - index / fps) > 1e-4:
                raise ValueError('Decode count/timestamp mismatch')
            # Reproduce torchvision float32 /255 then the existing adapter conversion.
            image = _parse_image(torch.from_numpy(frame.to_rgb().to_ndarray().copy()).permute(2, 0, 1).float() / 255)
            output[index] = np.asarray(resize_with_pad(image, 224, 224))
            count += 1
    if count != frames:
        raise ValueError('Missing decoded frames')
    output.flush()
    for index in checks:
        reader = torchvision.io.VideoReader(str(source), 'video')
        ts = float(np.float32(index / fps))
        reader.seek(ts, keyframes_only=True)
        reference = None
        best_distance = float('inf')
        for frame in reader:
            distance = abs(frame['pts'] - ts)
            if distance < best_distance:
                best_distance = distance
                reference = frame['data']
            if frame['pts'] >= ts:
                break
        if best_distance > 1e-4:
            raise ValueError('Reference timestamp mismatch')
        if reference is not None:
            reference = np.asarray(resize_with_pad(_parse_image(reference.float() / 255), 224, 224))
        reader.container.close()
        if reference is None or not np.array_equal(reference, output[index]):
            raise ValueError(f'Cached RGB differs from actual decoder at {source}:{index}')
    del output
    temp.rename(destination)
    if sha256(source) != source_sha:
        raise ValueError('Source changed during cache build')
    destination.chmod(0o444)
    stat = destination.stat()
    result = dict(episode=episode, camera=camera, frames=frames,
        cache_file=str(destination.relative_to(destination.parents[1])),
        source_sha256=source_sha, cache_sha256=sha256(destination),
        bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, exact_reference_samples=len(checks),
        seconds=time.monotonic()-t)
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--episodes', type=int, nargs='*')
    args = parser.parse_args()
    from openpi.training.frame_cache import sha256
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata('local/rgb-cache', root=args.source)
    from openpi.policies import fr3_sharpa_policy
    from openpi_client import image_tools
    pipeline={'parse_image':sha256(fr3_sharpa_policy.__file__),'resize':sha256(image_tools.__file__)}
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists():
        previous=json.loads((args.out/'manifest.json').read_text())
        if previous.get('complete') is True:
            raise FileExistsError(args.out/'manifest.json')
        if previous.get('schema')!='rgb224_exact_pil_v2' or previous.get('pixel_pipeline')!=pipeline:
            raise ValueError('Partial cache uses a different pixel pipeline; use a fresh directory')
    elif any(args.out.iterdir()):
        raise ValueError('Unbound partial cache; use a fresh output directory')
    selected = args.episodes if args.episodes is not None else sorted(meta.episodes)
    jobs = [(str(args.source/meta.get_video_file_path(ep, cam)),
             str(args.out/cam/f'episode_{ep:06d}.npy'),ep,cam,meta.episodes[ep]['length'],meta.fps)
            for ep in selected for cam in meta.video_keys]
    context = multiprocessing.get_context('spawn')
    cpu_queue = context.Queue()
    cpus = sorted(os.sched_getaffinity(0))
    if args.workers < 1 or 2*args.workers > len(cpus):
        raise ValueError('Insufficient CPUs for bounded cache workers')
    for i in range(args.workers):cpu_queue.put(set(cpus[2*i:2*i+2]))
    results=[]
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=worker_init, initargs=(cpu_queue,)) as pool:
        for future in as_completed([pool.submit(build_video, job) for job in jobs]):
            result=future.result();results.append(result)
            print(json.dumps({'done':len(results),'total':len(jobs),'episode':result['episode'],
                'camera':result['camera'],'seconds':round(result['seconds'],2)}),flush=True)
    from openpi.policies import fr3_sharpa_policy
    from openpi_client import image_tools
    manifest=dict(schema='rgb224_exact_pil_v2',complete=args.episodes is None,
        pixel_pipeline={'parse_image':sha256(fr3_sharpa_policy.__file__),'resize':sha256(image_tools.__file__)},
        builder_sha256=sha256(__file__),
        episodes=len(selected),frames=sum(meta.episodes[ep]['length'] for ep in selected),
        source_metadata={f'meta/{name}':sha256(args.source/'meta'/name) for name in ['info.json','episodes.jsonl','tasks.jsonl']},
        entries=sorted(results,key=lambda x:(x['episode'],x['camera'])))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if manifest['complete']:
        (args.out/'manifest.json').chmod(0o444)


if __name__=='__main__':main()
