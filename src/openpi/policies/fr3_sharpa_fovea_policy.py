"""Fovea training labels are carried separately from the current-image prefix."""

from dataclasses import dataclass

import numpy as np

from openpi.policies.fr3_sharpa_crop_policy import Fr3SharpaCropInputs


@dataclass(frozen=True)
class FoveaInputs(Fr3SharpaCropInputs):
    def __call__(self, data):
        result = super().__call__(data)
        if "foveation_tokens" in data:
            tokens = np.asarray(data["foveation_tokens"])
            if (
                tokens.shape != (4,)
                or tokens.dtype.kind not in "iu"
                or np.any(tokens < 256000)
                or np.any(tokens > 257023)
            ):
                raise ValueError("Four valid future location tokens required")
            result["foveation_tokens"] = tokens.astype(np.int32)
        return result
