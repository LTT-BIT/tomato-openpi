"""Build the existing TrainConfig without copying or changing the training loop."""

from dataclasses import dataclass
from dataclasses import replace
from typing import Protocol

from openpi.models import pi0_config
from openpi.training import config as cfg
from openpi.training import optimizer
from openpi.training import weight_loaders
from training_configs.datasets.common import Dataset
from training_configs.datasets.common import runtime_root


class Embodiment(Protocol):
    version: str
    action_dim: int
    raw_state_dims: tuple[int, ...]
    action_hz: float
    camera_keys: tuple[str, ...]

    def data_config(self, dataset: Dataset) -> cfg.DataConfigFactory: ...


@dataclass(frozen=True)
class Recipe:
    version: str
    source_baseline: str
    paligemma_variant: str
    action_expert_variant: str
    num_train_steps: int

    def build(self, *, name: str, exp_name: str, embodiment: Embodiment, dataset: Dataset) -> cfg.TrainConfig:
        for key, value in {"name": name, "exp_name": exp_name, **vars(dataset)}.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{key} must not be empty")
        model = pi0_config.Pi0Config(
            pi05=True,
            dtype="bfloat16",
            action_dim=embodiment.action_dim,
            action_horizon=50,
            max_token_len=256,
            discrete_state_input=True,
            paligemma_variant=self.paligemma_variant,
            action_expert_variant=self.action_expert_variant,
            pytorch_compile_mode="max-autotune",
        )
        config = cfg.TrainConfig(
            name=name,
            project_name="openpi",
            exp_name=exp_name,
            model=model,
            weight_loader=weight_loaders.PartialCheckpointWeightLoader(
                params_path="gs://openpi-assets/checkpoints/pi05_base/params",
                skip_on_mismatch_regex=".*(action_in_proj|action_out_proj|state_proj).*",
            ),
            freeze_filter=model.get_freeze_filter(),
            data=embodiment.data_config(dataset),
            lr_schedule=optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=2.5e-5,
                decay_steps=30_000,
                decay_lr=2.5e-6,
            ),
            optimizer=optimizer.AdamW(
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=1e-10,
                clip_gradient_norm=1.0,
            ),
            ema_decay=None,
            seed=42,
            model_seed=42,
            data_seed=42001,
            augmentation_seed=42002,
            sampling_mode="epoch_shuffle",
            batch_size=8,
            num_workers=8,
            num_train_steps=self.num_train_steps,
            fsdp_devices=1,
            log_interval=10,
            save_interval=2_000,
            keep_period=10_000,
            assets_base_dir=str(runtime_root() / "assets"),
            checkpoint_base_dir=str(runtime_root() / "checkpoints"),
            resume=False,
            overwrite=False,
            wandb_enabled=True,
            pytorch_weight_path=None,
            pytorch_training_precision="bfloat16",
        )
        return replace(
            config,
            policy_metadata={
                "recipe_version": self.version,
                "baseline_version": self.source_baseline,
                "embodiment_version": embodiment.version,
                "physical_action_dim": embodiment.action_dim,
                "raw_state_dims": list(embodiment.raw_state_dims),
                "camera_keys": list(embodiment.camera_keys),
                "dataset_version": dataset.version,
                "dataset_repo_id": dataset.repo_id,
                "dataset_root": dataset.root,
                "source_datasets": list(dataset.sources),
                "prompt_source": "dataset.meta/tasks.jsonl",
                "prompt_example": dataset.prompt_example,
                "expected_norm_sha256": dataset.expected_norm_sha256,
                "expected_norm_audit_sha256": dataset.expected_norm_audit_sha256,
            },
        )
