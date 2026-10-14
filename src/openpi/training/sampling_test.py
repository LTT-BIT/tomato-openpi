import dataclasses

import numpy as np
import pytest

from openpi.training import config
from openpi.training import data_loader
from openpi.training import sampling


@pytest.mark.parametrize("size", [1, 3, 8, 11])
def test_full_epochs_and_budget_prefix(size):
    indices = sampling.epoch_shuffle_indices(size, seed=42001, num_samples=size * 3 + 2)
    for epoch in range(3):
        np.testing.assert_array_equal(np.sort(indices[epoch * size : (epoch + 1) * size]), np.arange(size))
    np.testing.assert_array_equal(
        indices[: size + 1], sampling.epoch_shuffle_indices(size, seed=42001, num_samples=size + 1)
    )


def test_seed_and_seek():
    plan = sampling.epoch_shuffle_indices(101, seed=42, num_samples=400)
    np.testing.assert_array_equal(plan, sampling.epoch_shuffle_indices(101, seed=42, num_samples=400))
    assert not np.array_equal(plan, sampling.epoch_shuffle_indices(101, seed=43, num_samples=400))
    sampler = sampling.PlanSampler(plan)
    sampler.seek(104)
    np.testing.assert_array_equal(list(sampler), plan[104:])
    sampler.seek(400)
    assert list(sampler) == []
    with pytest.raises(ValueError, match="offset"):
        sampler.seek(401)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("mode", sampling.PLAN_MODES)
def test_worker_prefetch_resume_and_tail(workers, mode):
    def make_loader():
        return data_loader.TorchDataLoader(
            list(range(11)),
            local_batch_size=4,
            num_workers=workers,
            sampling_mode=mode,
            sample_weights=np.arange(1, 12) if mode == "weighted_replacement" else None,
            num_train_steps=7,
            seed=42,
            framework="pytorch",
        )

    full = np.stack([batch.numpy() for batch in make_loader()])
    assert full.shape == (7, 4)
    if mode == "epoch_shuffle":
        np.testing.assert_array_equal(np.sort(full.ravel()[:11]), np.arange(11))
    loader = make_loader()
    iterator = iter(loader)
    prefix = np.stack([next(iterator).numpy() for _ in range(3)])
    # Workers may have already fetched the rest. Resume in a fresh loader anyway.
    resumed = make_loader()
    resumed.seek(3)
    suffix = np.stack([batch.numpy() for batch in resumed])
    np.testing.assert_array_equal(np.concatenate((prefix, suffix)), full)


def test_smaller_than_batch_is_still_complete():
    loader = data_loader.TorchDataLoader(
        [0, 1, 2], local_batch_size=8, sampling_mode="epoch_shuffle", num_train_steps=2, framework="pytorch"
    )
    assert [batch.shape for batch in loader] == [(8,), (8,)]


def test_legacy_weighted_behavior_unchanged():
    loader = data_loader.TorchDataLoader(
        list(range(11)), local_batch_size=4, sample_weights=np.ones(11), num_batches=3, framework="pytorch"
    )
    assert loader.torch_loader.sampler.replacement is True
    assert loader.torch_loader.drop_last is True
    assert len(list(loader)) == 3


def test_independent_data_seed():
    cfg = dataclasses.replace(
        config.get_config("debug"),
        sampling_mode="epoch_shuffle",
        seed=42,
        data_seed=42001,
        num_workers=0,
        num_train_steps=3,
    )
    first = data_loader.create_data_loader(cfg, skip_norm_stats=True)
    second = data_loader.create_data_loader(dataclasses.replace(cfg, seed=99, model_seed=8), skip_norm_stats=True)
    for (_, a), (_, b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_epoch_shuffle_rejects_mixtures_and_other_frameworks():
    cfg = dataclasses.replace(config.get_config("debug"), sampling_mode="epoch_shuffle")
    with pytest.raises(ValueError, match="JAX"):
        data_loader.create_data_loader(cfg, framework="pytorch")
    data = config.DataConfig(
        repo_id="mixture",
        lerobot_datasets=(
            config.LeRobotDatasetSpec(repo_id="a", root="/unused/a"),
            config.LeRobotDatasetSpec(repo_id="b", root="/unused/b"),
        ),
    )
    with pytest.raises(ValueError, match="exactly one"):
        data_loader.create_torch_dataset(data, 2, cfg.model, sampling_mode="epoch_shuffle")


def test_group_probability_is_independent_of_group_frame_count():
    # A has 2 frames, B has 18; weights describe group mass, not per-frame mass.
    episodes = np.array([5, 5] + [9] * 18)
    groups = (sampling.SamplingGroup("A", (5,), 7), sampling.SamplingGroup("B", (9,), 3))
    weights = sampling.group_frame_weights(episodes, groups)
    assert weights[:2].sum() == pytest.approx(0.7)
    assert weights[2:].sum() == pytest.approx(0.3)
    plan = sampling.weighted_replacement_indices(weights, seed=42001, num_samples=100_000)
    assert np.mean(episodes[plan] == 5) == pytest.approx(0.7, abs=0.005)
    assert np.mean(plan == 0) == pytest.approx(0.35, abs=0.005)
    assert len(np.unique(plan)) < len(plan)  # with replacement
    np.testing.assert_array_equal(
        plan[:100], sampling.weighted_replacement_indices(weights, seed=42001, num_samples=100)
    )
    assert not np.array_equal(plan[:100], sampling.weighted_replacement_indices(weights, seed=42, num_samples=100))


@pytest.mark.parametrize("episodes", [(0, 0), (), (-1,), (True,), (1.5,)])
def test_invalid_group_episode_ids(episodes):
    with pytest.raises(ValueError, match="episodes"):
        sampling.SamplingGroup("A", episodes, 7)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True])
def test_invalid_group_weights(weight):
    with pytest.raises(ValueError, match="weight"):
        sampling.SamplingGroup("A", (0,), weight)


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        ((), "requires sampling_groups"),
        ((sampling.SamplingGroup("A", (0,), 7), sampling.SamplingGroup("A", (1,), 3)), "name"),
        ((sampling.SamplingGroup("A", (0,), 7), sampling.SamplingGroup("B", (0, 1), 3)), "overlap"),
        ((sampling.SamplingGroup("A", (0,), 7),), "cover exactly"),
        ((sampling.SamplingGroup("A", (0, 1, 2), 7),), "cover exactly"),
    ],
)
def test_invalid_group_partitions(groups, match):
    with pytest.raises(ValueError, match=match):
        sampling.group_frame_weights([0, 1, 1], groups)


def test_group_mode_must_be_explicit():
    cfg = config.get_config("debug")
    with pytest.raises(ValueError, match="requires sampling_groups"):
        dataclasses.replace(cfg, sampling_mode="weighted_replacement")
    with pytest.raises(ValueError, match="requires weighted_replacement"):
        dataclasses.replace(cfg, sampling_groups=(sampling.SamplingGroup("A", (0,), 1),))


def test_replacement_coverage_counts_distinct_frames():
    sampler = sampling.PlanSampler(np.array([2, 2, 0, 2, 1, 0]))
    assert [sampler.unique_count(i) for i in range(7)] == [0, 1, 1, 2, 2, 3, 3]
    sampler.seek(3)
    assert sampler.unique_count(4) == 2
