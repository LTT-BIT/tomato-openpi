"""UI-only adapter: preserve data identity without changing normalization code."""

from dataclasses import replace
from pathlib import Path

from openpi.shared import delta_paper as paper


def adapt_ui_config(source, result):
    """Preserve sealed sources; keep model statistics aligned with UI storage."""
    data_source = source.data
    before = [str(Path(spec.root).expanduser().resolve()) for spec in data_source.lerobot_datasets]
    after = [str(Path(spec.root).expanduser().resolve()) for spec in result.data.lerobot_datasets]
    if after != before or result.data.repo_id != data_source.repo_id:
        raise ValueError("论文实验的数据目录、顺序和repo_id由个人配置固定,请在个人文件中定义新数据版本")
    for key in ("dataset_version", "source_datasets"):
        if result.policy_metadata[key] != source.policy_metadata[key]:
            raise ValueError("论文实验的数据版本和来源由个人配置固定,请定义新的数据配置")
    if not result.data.extra_delta_transform:
        raise ValueError("论文实验固定为双臂 delta、双手 absolute")
    if result.sampling_mode != "epoch_shuffle" or result.sampling_groups:
        raise ValueError("论文实验仅支持epoch_shuffle,不支持UI加权重复采样")
    data = replace(result.data, lerobot_datasets=data_source.lerobot_datasets)
    directory = data.asset_directory(result.assets_dirs)
    model = replace(
        result.model,
        paper_asset_dir=str(directory),
        paper_asset_sha256=paper.sha256(directory / paper.BUNDLE) if (directory / paper.BUNDLE).is_file() else None,
    )
    if (directory / paper.BUNDLE).is_file():
        # Fail before saving a UI snapshot that would reuse an incompatible asset.
        paper.load_bundle(
            directory,
            mode=model.paper_mode,
            beta=model.paper_beta,
            horizon=model.action_horizon,
            std_floor=model.paper_std_floor,
            expected_sha256=model.paper_asset_sha256,
        )
    return replace(
        result,
        data=data,
        model=model,
        policy_metadata={**result.policy_metadata, "training_action_semantics": "delta_arms_absolute_hands"},
    )
