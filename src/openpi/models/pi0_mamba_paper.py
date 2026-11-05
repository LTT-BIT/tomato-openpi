"""Mamba with the existing T0/T1/T2/T3/Tg data/prior contracts.

Multiple inheritance shares the original Paper configuration identity, allowing
unmodified PaperSharpaData/statistics/provenance to serve both architectures.
RTC uses the same clean-prefix training semantics as the Transformer.
"""

from dataclasses import dataclass
import math

from flax import nnx
import jax.numpy as jnp

from openpi.models.pi0_delta_paper import PaperPi0
from openpi.models.pi0_delta_paper import PaperPi0Config
from openpi.models.pi0_mamba import MambaPi0
from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.shared import delta_paper as paper


@dataclass(frozen=True)
class PaperMambaPi0Config(MambaPi0Config, PaperPi0Config):
    stage: str = "paper_delta_fm"
    action_representation: str = "delta_arms_absolute_hands"
    normalization: str = "paper"

    def validate_algorithm(self):
        if (
            self.architecture != "pi05_mamba3_jax"
            or self.stage != "paper_delta_fm"
            or self.action_representation != "delta_arms_absolute_hands"
            or self.normalization != "paper"
        ):
            raise ValueError("Paper Mamba requires arm Delta and matching Paper normalization")
        if not math.isfinite(self.paper_std_floor) or self.paper_std_floor <= 0:
            raise ValueError("Paper standard deviation floor must be positive and finite")

    def create(self, rng):
        return PaperMambaPi0(self, rngs=nnx.Rngs(rng))

    def adapt_data_config(self, data):
        from openpi.training.delta_paper import PaperDataView
        from openpi.training.mamba_training import MambaPaperDataView
        from openpi.training.mamba_training import architecture_record

        if not isinstance(data, PaperDataView) or data.paper_bundle_sha256 != self.paper_asset_sha256:
            raise ValueError("Paper Mamba requires its matching validated PaperDataView")
        data = super().adapt_data_config(data)
        values = {k: v for k, v in vars(data).items() if k != "mamba_identity"}
        return MambaPaperDataView(**values, mamba_identity=architecture_record(self))


class PaperMambaPi0(MambaPi0):
    def __init__(self, config, *, rngs):
        if not config.paper_asset_sha256:
            raise ValueError("Compute matching Paper statistics and reload configuration before model creation")
        _, arrays = paper.load_bundle(
            config.paper_asset_dir,
            mode=config.paper_mode,
            beta=config.paper_beta,
            horizon=config.action_horizon,
            std_floor=config.paper_std_floor,
            expected_sha256=config.paper_asset_sha256,
        )
        super().__init__(config, rngs)
        self.paper_beta = config.paper_beta
        if self.paper_beta:
            self.paper_noise_cholesky = nnx.Variable(jnp.asarray(arrays["noise_cholesky"]))

    # Exact same time-major z @ L.T as the existing Transformer Paper model.
    noise = PaperPi0.noise
