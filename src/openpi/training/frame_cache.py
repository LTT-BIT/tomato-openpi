"""Opt-in, lossless cache of the existing 224-pixel training image inputs."""
import hashlib
import json
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class CachedLeRobotDataset(LeRobotDataset):
    def __init__(self, *args, cache_root, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_root = Path(cache_root).resolve()
        self.cache_manifest = json.loads((self.cache_root / 'manifest.json').read_text())
        if (self.cache_manifest.get('complete') is not True
                or self.cache_manifest.get('schema') != 'rgb224_exact_pil_v2'
                or self.image_transforms is not None):
            raise ValueError('Complete exact RGB cache without extra image transforms required')
        from openpi.policies import fr3_sharpa_policy
        from openpi_client import image_tools
        pipeline = {'parse_image': sha256(fr3_sharpa_policy.__file__), 'resize': sha256(image_tools.__file__)}
        if self.cache_manifest.get('pixel_pipeline') != pipeline:
            raise ValueError('Cache pixel pipeline changed')
        expected = self.cache_manifest['source_metadata']
        for name, digest in expected.items():
            if sha256(self.root / name) != digest:
                raise ValueError('Cache source metadata changed: ' + name)
        self.cache_entries = {}
        for item in self.cache_manifest['entries']:
            key = (item['episode'], item['camera'])
            if key in self.cache_entries:
                raise ValueError('Duplicate cached video')
            if item['frames'] != self.meta.episodes[key[0]]['length']:
                raise ValueError('Cached episode length differs')
            source = self.root / self.meta.get_video_file_path(*key)
            if sha256(source) != item['source_sha256']:
                raise ValueError('Cache source video changed: ' + str(source))
            cache = (self.cache_root / item['cache_file']).resolve()
            if not cache.is_relative_to(self.cache_root):
                raise ValueError('Invalid cached image path')
            stat = cache.stat()
            if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
                raise ValueError('Cached image file changed: ' + str(cache))
            if item.get('exact_reference_samples', 0) < 3:
                raise ValueError('Cache lacks exact reference validation')
            self.cache_entries[key] = item
        expected_keys = {(int(ep), cam) for ep in self.meta.episodes for cam in self.meta.video_keys}
        if set(self.cache_entries) != expected_keys:
            raise ValueError('Cache must cover every episode and camera')
        self._cache_maps = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache_maps'] = {}
        return state

    def _query_videos(self, query_timestamps, ep_idx):
        result = {}
        for camera, times in query_timestamps.items():
            if len(times) != 1:
                raise ValueError('RGB cache supports current image only')
            key = (int(ep_idx), camera)
            item = self.cache_entries[key]
            frame = round(times[0] * self.fps)
            if (not 0 <= frame < item['frames']
                    or abs(frame / self.fps - times[0]) >= self.tolerance_s):
                raise ValueError('Cached timestamp outside source timeline')
            if key not in self._cache_maps:
                value = np.load(self.cache_root / item['cache_file'], mmap_mode='r', allow_pickle=False)
                if value.shape != (item['frames'], 224, 224, 3) or value.dtype != np.uint8:
                    raise ValueError('Invalid RGB cache shape/dtype')
                self._cache_maps[key] = value
            result[camera] = torch.from_numpy(self._cache_maps[key][frame].copy())
        return result
