"""Exercise growing datasets, immutable releases and shared task options."""

from dataclasses import replace
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.subtasks import tomato
from openpi.subtasks import tomato_in_hand
from scripts import training_ui_config as ui
from scripts.publish_tomato_dataset import publish
from training_configs.experiments.tomato_sharpa.tasks.common import build_config
from training_configs.experiments.tomato_sharpa.tasks.dataset import load_release
from training_configs.experiments.tomato_sharpa.tasks.profiles import PROFILES


def release(tmp_path, profile, *, count=2, views=4, version="release_r1"):
    root = tmp_path / version / "data"
    (root / "meta").mkdir(parents=True)
    task = PROFILES[profile]
    cameras = tomato_in_hand.CAMERAS[:views]
    features = {
        "observation.state": {"shape": [116], "names": list(tomato.SOURCE_STATE_NAMES)},
        "action": {"shape": [58], "names": list(tomato.SOURCE_ACTION_NAMES)},
        **{key: {"dtype": "video"} for key in cameras},
        "observation.interaction_available": {"dtype": "bool"},
    }
    sequence = tomato_in_hand.SEQUENCE if profile == "truss_in_hand" else tomato.COARSE_SEQUENCE
    length = len(sequence) * 2 if task.phases else 4
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "features": features,
        "total_episodes": count,
        "total_frames": count * length,
        "data_path": "data/episode_{episode_index:06d}.parquet",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps({"episode_index": i, "length": length}) for i in range(count))
    )
    prompts = task.prompts or (task.goal,)
    (root / "meta/tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": i, "task": prompt}) for i, prompt in enumerate(prompts))
    )
    if task.phases:
        (root / "annotations").mkdir()
        (root / "data").mkdir()
        for episode in range(count):
            document = {
                "stage": "coarse",
                "prompt_profile": tomato_in_hand.PROFILE,
                "segments": [
                    {
                        "label": label,
                        "text": task.action_prompts[label],
                        "reviewed": True,
                        "start_frame": 2 * i,
                        "end_frame": 2 * i + 2,
                    }
                    for i, label in enumerate(sequence)
                ],
            }
            (root / f"annotations/episode_{episode:06d}.json").write_text(json.dumps(document))
            indices = np.repeat([prompts.index(task.action_prompts[label]) for label in sequence], 2)
            pq.write_table(
                pa.table({"task_index": indices, "frame_index": range(length), "episode_index": [episode] * length}),
                root / f"data/episode_{episode:06d}.parquet",
            )
    provenance = root.parent / "preparation.json"
    provenance.write_text('{"source": "reviewed_fixture"}')
    manifest = root.parent / "release.json"
    publish(
        root=root,
        profile=profile,
        version=version,
        asset_id=f"tomato_{version}_delta_30hz",
        action_semantics="delta",
        sources=["reviewed_fixture"],
        provenance=[provenance],
        output=manifest,
    )
    return manifest, root


@pytest.mark.parametrize("profile", PROFILES)
def test_same_entry_accepts_growing_releases_and_preserves_old_version(tmp_path, profile):
    first, _ = release(tmp_path, profile, count=2)
    second, _ = release(tmp_path, profile, count=3, version="release_r2")
    a = build_config(profile, dataset_manifest=first)
    b = build_config(profile, dataset_manifest=second)
    assert a.name == b.name == f"pi05_sharpa_{profile}_fullft"
    assert (a.policy_metadata["training_episodes"], b.policy_metadata["training_episodes"]) == (2, 3)
    assert a.data.assets.asset_id != b.data.assets.asset_id
    assert a.checkpoint_dir != b.checkpoint_dir
    assert build_config(profile, dataset_manifest=first) == a
    assert ui.apply_overrides(a, ui.config_values(a)) == a
    assert a.policy_metadata["phases"] == list(PROFILES[profile].phases)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("method", ["fullft", "lora"])
@pytest.mark.parametrize("views", [3, 4])
@pytest.mark.parametrize("prompt", ["stage", "global"])
def test_shared_options_use_profile_prompts_and_correct_adapters(tmp_path, profile, method, views, prompt):
    path, _ = release(tmp_path, profile)
    original = build_config(profile, dataset_manifest=path)
    values = {
        **ui.config_values(original),
        "training_method": method,
        "video_views": views,
        "prompt_mode": prompt,
        "asset_id": "independent_trial",
    }
    actual = ui.apply_overrides(original, values)
    assert len(actual.model.image_keys) == views
    assert ("lora" in actual.model.paligemma_variant) == (method == "lora")
    assert actual.freeze_filter == actual.model.get_freeze_filter()
    repack, vectors = actual.data.create_vector_transforms(actual.model)
    assert repack.inputs
    assert vectors.inputs
    assert actual.data.base_config.global_prompt == (PROFILES[profile].goal if prompt == "global" else None)
    assert actual.policy_metadata["prompts"] == (
        [PROFILES[profile].goal] if prompt == "global" else list(PROFILES[profile].prompts or (PROFILES[profile].goal,))
    )
    assert ui.apply_overrides(actual, ui.config_values(actual)) == actual


def test_reject_missing_view_mutated_release_or_wrong_task(tmp_path):
    path, root = release(tmp_path, "single_pick", views=3)
    config = build_config("single_pick", dataset_manifest=path)
    with pytest.raises(ValueError, match="局部图"):
        ui.apply_overrides(config, {**ui.config_values(config), "video_views": 4, "asset_id": "new"})
    with pytest.raises(ValueError, match="profile"):
        load_release(path, "truss_in_hand")
    values = ui.config_values(config)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="release changed"):
        ui.apply_overrides(config, values)
    (root / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"changed"}')
    with pytest.raises(ValueError, match="changed"):
        load_release(path, "single_pick")


def test_reject_wrong_phase_supervision_before_publishing(tmp_path):
    path, root = release(tmp_path, "truss_in_hand")
    table = pq.read_table(root / "data/episode_000000.parquet")
    table = table.set_column(0, "task_index", pa.array([0] * len(table)))
    pq.write_table(table, root / "data/episode_000000.parquet")
    with pytest.raises(ValueError, match="task_index"):
        load_release(path, "truss_in_hand")


def test_ui_catalog_only_lists_three_sharpa_tasks():
    entries = ui.experiment_catalog()
    assert {entry["id"].split(".")[-1] for entry in entries} == set(PROFILES)
    assert {entry["task"] for entry in ui.baseline_catalog() + entries} == {"tomato_sharpa"}


def test_http_catalog_and_dataset_release_load(tmp_path):
    import http.client
    from http.server import ThreadingHTTPServer
    import threading

    from scripts import training_ui as service

    path, _ = release(tmp_path, "truss_in_hand")
    module = "training_configs.experiments.tomato_sharpa.tasks.truss_in_hand"
    server = ThreadingHTTPServer(("127.0.0.1", 0), service.handler_class("test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*server.server_address)
    headers = {"X-Training-Token": "test-token", "Content-Type": "application/json"}
    try:
        connection.request("GET", "/api/baselines", headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        assert len(json.loads(response.read())["baselines"]) == 3
        connection.request("POST", "/api/dataset-release", json.dumps({"id": module, "path": str(path)}), headers)
        response = connection.getresponse()
        assert response.status == 200
        data = json.loads(response.read())
        assert data["protocol"]["training_episodes"] == 2
        assert data["values"]["dataset_manifest_sha256"]
        original = ui.source_config(module)
        saved = ui.personal_source(module, "tester", data["values"])
        actual = ui.config_from_source(saved, module, "saved.py")
        assert actual.policy_metadata["dataset_manifest_sha256"] == data["values"]["dataset_manifest_sha256"]
        assert original.policy_metadata["dataset_manifest"] == ""
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_release_binding_cannot_be_bypassed_by_cli_roots(tmp_path):
    from scripts.run_training import preflight

    path, _ = release(tmp_path, "single_pick")
    config = build_config("single_pick", dataset_manifest=path)
    wrong = replace(
        config,
        data=replace(
            config.data, lerobot_datasets=(replace(config.data.lerobot_datasets[0], root=str(tmp_path / "other")),)
        ),
    )
    with pytest.raises(ValueError, match="differ from the selected release"):
        preflight(wrong, require_norm=False)


def test_saved_paper_recipe_can_return_to_lora_with_new_statistics(monkeypatch, tmp_path):
    monkeypatch.setenv("PI05_IN_HAND204_BUNDLE", str(tmp_path / "missing"))
    source = build_config("truss_in_hand")
    paper = ui.apply_overrides(source, {**ui.config_values(source), "paper_trial": "t3", "asset_id": "paper_trial"})
    lora = ui.apply_overrides(
        paper, {**ui.config_values(paper), "paper_trial": "none", "training_method": "lora", "asset_id": "lora_trial"}
    )
    assert "lora" in lora.model.paligemma_variant
    assert not hasattr(lora.data, "paper_contract")
    restored = ui.apply_overrides(
        lora, {**ui.config_values(lora), "paper_trial": "t3", "training_method": "fullft", "asset_id": "fullft_trial"}
    )
    assert restored.model.paper_mode == "timestamp_zscore_arms"
    assert "lora" not in restored.model.paligemma_variant
