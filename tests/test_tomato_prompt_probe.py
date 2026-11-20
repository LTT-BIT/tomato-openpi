import numpy as np

from scripts.tomato_prompt_probe import GROUPS
from scripts.tomato_prompt_probe import noise_for
from scripts.tomato_prompt_probe import paired_metrics
from scripts.tomato_prompt_probe import sample_plan


def test_sampling_is_prediction_independent_and_keeps_phase_pure_length():
    segments = [{"start_frame": 0, "end_frame": 60}, {"start_frame": 60, "end_frame": 90}]
    plan = sample_plan([segments])
    before = next(r for r in plan if r["frame"] == 45)
    assert before["phase"] == 0
    assert before["phase_remaining_frames"] == 15
    assert next(r for r in plan if r["frame"] == 60)["phase"] == 1
    assert len({(r["episode"], r["frame"]) for r in plan}) == len(plan)
    assert all(0 <= r["frame"] < 90 for r in plan)


def test_common_noise_is_reproducible_and_not_shared_between_anchors_or_seeds():
    np.testing.assert_array_equal(noise_for(42, 3, 10), noise_for(42, 3, 10))
    assert not np.array_equal(noise_for(42, 3, 10), noise_for(43, 3, 10))
    assert not np.array_equal(noise_for(42, 3, 10), noise_for(42, 3, 11))


def test_primary_errors_do_not_score_next_phase_target_as_current_phase():
    gt = np.zeros((50, 58))
    predictions = np.zeros((4, 50, 58))
    predictions[0, 10:] = 100
    result = paired_metrics(predictions, gt, 0, 10)
    assert set(GROUPS) == set(result)
    for group in result.values():
        assert group["50"]["evaluated_frames"] == 10
        assert group["50"]["mse"][0] == 0
    full = paired_metrics(predictions, gt, 0, 50)
    assert full["right_hand"]["50"]["mse"][0] > 0


def test_group_indices_and_counterfactual_labels():
    assert sorted(i for dims in GROUPS.values() for i in dims) == list(range(58))
    p = np.ones((4, 50, 58))
    p[2] = 0
    r = paired_metrics(p, np.zeros((50, 58)), 2, 50)
    assert r["left_arm"]["15"]["factual_best"]
    assert r["left_arm"]["15"]["best_wrong_minus_factual"] == 1


def test_extra_crop_requires_explicit_model_image_key_configuration():
    from openpi.models import model

    image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    extra = "right_contact_crop_rgb"
    obs = model.Observation(
        images={**dict.fromkeys(model.IMAGE_KEYS, image), extra: np.ones_like(image)},
        image_masks=dict.fromkeys((*model.IMAGE_KEYS, extra), np.ones(1, dtype=bool)),
        state=np.zeros((1, 58), dtype=np.float32),
    )
    default = model.preprocess_observation(None, obs, train=False)
    assert extra not in default.images
    configured = model.preprocess_observation(None, obs, train=False, image_keys=(*model.IMAGE_KEYS, extra))
    assert extra in configured.images
    np.testing.assert_array_equal(configured.images[extra], obs.images[extra])


def test_expert_action_is_not_a_model_observation():
    from openpi.models import model

    raw = {"image": {}, "image_mask": {}, "state": np.zeros((1, 58)), "actions": np.ones((1, 50, 58))}
    obs = model.Observation.from_dict(raw)
    assert not hasattr(obs, "actions")


def test_crop_box_is_native_resolution_and_bounded():
    from scripts.tomato_crop_audit import centered_box

    assert centered_box(640, 480) == (160, 120, 480, 360)
    assert centered_box(1, 1) == (0, 0, 1, 1)


def test_noise_effect_is_measured_separately_from_prompt_effect(tmp_path):
    from scripts.summarize_tomato_prompt_probe import paired_noise

    a = np.ones((4, 50, 58))
    a[2] = 3
    b = a + 1
    path = tmp_path / "pair.npz"
    np.savez(path, absolute_s42=a, absolute_s43=b)
    result = paired_noise([({"phase_remaining_frames": 5}, path)])
    for group in result.values():
        assert group["pickup_vs_return_rms_rad"]["median"] == 2
        assert group["noise_rms_rad"]["median"] == 1
        assert group["pickup_vs_return_to_noise_ratio"]["median"] == 2


def test_identical_prompt_outputs_are_ties_not_evidence_of_correct_conditioning():
    from scripts.summarize_tomato_prompt_probe import summarize_rows

    metrics = paired_metrics(np.zeros((4, 50, 58)), np.zeros((50, 58)), 0, 50)
    result = summarize_rows([{"metrics": metrics, "factual_prompt": 0}])
    assert result["right_arm"]["5"]["factual_strictly_lower_error_fraction"] == 0
    assert result["right_arm"]["5"]["factual_tied_fraction"] == 1


def test_full_summary_reconstructs_physical_metrics_from_frozen_arrays(tmp_path):
    import json

    from scripts.summarize_tomato_prompt_probe import CLASSES
    from scripts.summarize_tomato_prompt_probe import run
    from scripts.tomato_prompt_probe import sha

    root = tmp_path / "shards"
    for shard in range(4):
        folder = root / f"shard{shard}"
        folder.mkdir(parents=True)
        plan = [
            {
                "episode": ep,
                "frame": 0,
                "phase": ep % 7,
                "phase_remaining_frames": 50,
                "tags": [{"kind": "midpoint", "phase": ep % 7}],
            }
            for ep in range(shard, 93, 4)
        ]
        protocol = dict.fromkeys(
            (
                "checkpoint_metadata_sha256",
                "params_metadata_sha256",
                "probe_code_sha256",
                "norm_stats_sha256",
                "selection_sha256",
                "checkpoint",
                "scope",
                "host",
            ),
            "synthetic-unit-test",
        )
        protocol.update(prompt_order=["pickup", "harvest", "return", "adjust"], seeds=[42, 43], plan=plan)
        (folder / "protocol.json").write_text(json.dumps(protocol))
        rows = []
        for row in plan:
            pred = np.stack([np.full((50, 58), k + 1.0) for k in range(4)])
            target = np.full((50, 58), 0.25)
            np.savez(
                folder / f"ep{row['episode']:03d}_f000000.npz",
                absolute_s42=pred,
                absolute_s43=pred + 0.1,
                target_absolute=target,
            )
            for seed in (42, 43):
                factual = CLASSES[row["phase"]]
                rows.append(
                    {
                        **row,
                        "seed": seed,
                        "factual_prompt": factual,
                        "seconds": 0.1,
                        "token_counts": [240] * 4,
                        "metrics": paired_metrics(pred, target, factual, 50),
                    }
                )
        (folder / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (folder / "completed.json").write_text(
            json.dumps({"status": "completed", "calls": len(rows), "protocol_sha256": sha(folder / "protocol.json")})
        )
    output = tmp_path / "summary.json"
    run(root, output)
    result = json.loads(output.read_text())
    assert set(result["phase_matched_noise_comparison_h15"]) == {str(i) for i in range(7)}
    assert result["anchors"] == 93
    assert result["chunks"] == 744
    assert result["summary_physical_rad"]["right_hand"]["15"]["factual_mse"]["n"] == 186
    assert "physical_metrics" not in json.loads((root / "shard0/metrics.jsonl").read_text().splitlines()[0])
