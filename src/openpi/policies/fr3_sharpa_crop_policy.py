"""Opt-in current-crop input. No state history or result facts enter pi05 here."""
import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model
from openpi.policies.fr3_sharpa_multimodal_policy import Fr3SharpaMultimodalInputs
from openpi.policies.fr3_sharpa_policy import _parse_image


@dataclasses.dataclass(frozen=True)
class Fr3SharpaCropInputs(transforms.DataTransformFn):
    model_type: model.ModelType
    vector_statistics_image_independent = True

    def __call__(self, data: dict) -> dict:
        result = Fr3SharpaMultimodalInputs(self.model_type)(data)
        crop = _parse_image(data["observation/interaction_image"])
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[-1] != 3 or min(crop.shape[:2]) < 1:
            raise ValueError("Explicit RGB interaction image required")
        valid = np.asarray(data["observation/interaction_valid"])
        if valid.shape != () or valid.dtype != np.bool_:
            raise ValueError("Explicit scalar boolean interaction validity required")
        result["image"]["interaction_0_rgb"] = crop
        result["image_mask"]["interaction_0_rgb"] = valid
        return result
