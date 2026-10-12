"""Self-contained personal experiment example; copy to your owner directory."""

from dataclasses import replace
import os

from training_configs.common import Dataset
from training_configs.common import runtime_root
from training_configs.embodiments.fr3_wuji import FR3_WUJI
from training_configs.recipes.pi05_lora_v1 import build_config as build_recipe

PROMPT = (
    "Pick up a tomato truss with the right hand and hold it steady between both hands, "
    "then pick a single cherry tomato from the truss with the left hand "
    "and place the picked tomato into the plate on the left."
)


def build_dataset() -> Dataset:
    runtime = runtime_root()
    root = os.environ.get("PI05_DATASET_ROOT", str(runtime / "data/tomato_fulltask_v1"))
    return Dataset(
        version="tomato_fulltask_v1_30hz",
        repo_id="fr3_wuji/tomato_fulltask_v1",
        root=root,
        assets_dir=os.environ.get("PI05_ASSETS_DIR", str(runtime / "assets/fulltask_c0")),
        asset_id="tomato_fulltask_v1_30hz",
        sources=(root,),
        prompt_example=PROMPT,
    )


def build_config():
    config = build_recipe(
        name="pi05_fr3_wuji_fulltask_lora_v1_30hz",
        exp_name="template_tomato_lora_lr1e5_s42_run01",
        embodiment=FR3_WUJI,
        dataset=build_dataset(),
    )

    return replace(
        config,
        lr_schedule=replace(config.lr_schedule, peak_lr=1e-5),
        policy_metadata={**config.policy_metadata, "owner": "template", "purpose": "learning_rate_comparison"},
    )
