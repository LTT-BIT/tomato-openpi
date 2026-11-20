"""Read visual history without fetching the actions and state of older observations."""

import numpy as np

from openpi.subtasks.visual_memory import HISTORY
from openpi.subtasks.visual_memory import camera_frame
from openpi.subtasks.visual_memory import pack_history
from openpi.training.visual_memory import VisualMemoryDataset


class FastVisualMemoryDataset(VisualMemoryDataset):
    """Keep the native current sample and fetch older images together per camera."""

    def __init__(self, dataset, spec, mask_key="observation.interaction_available"):
        super().__init__(dataset, spec, mask_key)
        self._history_columns = None
        if hasattr(dataset, "_query_videos") and dataset.image_transforms is None and spec.lags:
            keys = ["timestamp"] if mask_key is None else ["timestamp", mask_key]
            self._history_columns = dataset.hf_dataset.select_columns(keys).with_format("numpy")[:]

    def __getitem__(self, index):
        columns = self._history_columns
        if columns is None:
            return super().__getitem__(index)
        index = index.__index__()
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample = self.dataset[index]
        current, _ = camera_frame(sample, self.mask_key)
        episode = self.episodes[index]
        lags = [lag for lag in self.spec.lags if index >= lag and self.episodes[index - lag] == episode]
        frames = {}
        if lags:
            indices = index - np.asarray(lags)
            timestamps = columns["timestamp"][indices].tolist()
            decoded = self.dataset._query_videos(dict.fromkeys(self.meta.video_keys, timestamps), int(episode))  # noqa: SLF001
            for slot, (lag, previous) in enumerate(zip(lags, indices, strict=True)):
                raw = {key: value if len(lags) == 1 else value[slot] for key, value in decoded.items()}
                if self.mask_key is not None:
                    raw[self.mask_key] = columns[self.mask_key][previous]
                frames[lag] = camera_frame(raw, self.mask_key)
        return {**sample, HISTORY: pack_history(self.spec, current, frames.get)}
