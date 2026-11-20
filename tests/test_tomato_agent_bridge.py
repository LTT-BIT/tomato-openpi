"""Protocol failures and repeated-stage mapping, without any robot connections."""

import math

import pytest

from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.subtasks.agent_bridge import LABELS
from openpi.subtasks.agent_bridge import PROTOCOL
from openpi.subtasks.agent_bridge import PredictionPromptProvider
from openpi.subtasks.agent_bridge import bind_chunk
from openpi.subtasks.agent_bridge import chunk_matches_prompt
from openpi.subtasks.agent_bridge import named_agent_history
from openpi.subtasks.tomato import ACTION_PROMPTS
from openpi.subtasks.tomato import COARSE_SEQUENCE
from openpi.subtasks.tomato import PHASES


def prediction(action, t=1.0, future="harvest"):
    def values(label):
        logits = [4.0 if k == label else -4.0 for k in LABELS]
        ex = [math.exp(x - max(logits)) for x in logits]
        return logits, [x / sum(ex) for x in ex]

    z, p = values(action)
    fz, fp = values(future)
    return {
        "protocol": PROTOCOL,
        "labels": list(LABELS),
        "valid": True,
        "error": None,
        "model_version": "sha",
        "observation_id": str(t),
        "session_id": "s",
        "input_timestamp": t,
        "result_timestamp": t + 0.05,
        "future_horizon_s": 1.0,
        "prediction_mode": "joint",
        "current_logits": z,
        "current_probabilities": p,
        "future_logits": fz,
        "future_probabilities": fp,
    }


def accept(provider, action, t, future="harvest"):
    return provider.accept_prediction(prediction(action, t, future), observation_id=str(t), now=t + 0.1)


def test_explicit_reset_then_seven_positions_four_prompts_and_no_auto_success():
    p = PredictionPromptProvider(confirmations=2)
    with pytest.raises(ValueError, match="Initialize"):
        accept(p, "pickup", 1.0)
    p.reset("s")
    ctx = accept(p, "pickup", 1.0)
    assert ctx.phase == "pickup_1"
    t = 2.0
    contexts = [ctx]
    for action in COARSE_SEQUENCE[1:]:
        first = accept(p, action, t)
        ctx = accept(p, action, t + 0.5)
        t += 1
        assert not first.changed
        assert ctx.changed
        contexts.append(ctx)
    assert [c.phase for c in contexts] == list(PHASES)
    assert len({c.prompt for c in contexts}) == 4
    assert contexts[2].prompt == contexts[6].prompt == ACTION_PROMPTS["return"]
    assert not any(c.task_complete for c in contexts)


def test_future_warning_never_advances_prompt_and_chunk_version_changes():
    p = PredictionPromptProvider(confirmations=1)
    p.reset("s")
    a = accept(p, "pickup", 1.0, future="harvest")
    assert a.future_warning == "harvest"
    assert a.phase_index == 0
    binding = bind_chunk(a, "chunk1")
    assert chunk_matches_prompt(binding, a)
    b = accept(p, "harvest", 2.0)
    assert not chunk_matches_prompt(binding, b)
    assert b.version == a.version + 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(current_probabilities=[1, 0, 0, 0, 0, 0, 0]),
        lambda x: x.update(current_probabilities=[float("nan")] * 4),
        lambda x: x.update(observation_id="wrong"),
        lambda x: x.update(input_timestamp=-100.0),
        lambda x: x.update(current_logits=[0, 0, 0, 0]),
        lambda x: x.update(labels=["return", "pickup", "harvest", "adjust"]),
    ],
)
def test_invalid_schema_rejected_without_progress(mutation):
    p = PredictionPromptProvider()
    p.reset("s")
    value = prediction("harvest")
    mutation(value)
    with pytest.raises(ValueError, match=".+"):
        p.accept_prediction(value, observation_id="1.0", now=1.1)
    assert p.phase == p.count == 0
    assert p.last_timestamp is None


def test_illegal_skip_and_out_of_order_cannot_advance():
    p = PredictionPromptProvider(confirmations=1)
    p.reset("s")
    assert accept(p, "adjust", 1.0).phase_index == 0
    with pytest.raises(ValueError, match="out-of-order"):
        accept(p, "harvest", 1.0)
    assert accept(p, "return", 2.0).phase_index == 0


def test_joint_mapping_uses_names_not_shared_dimension():
    # Agent is arms-left/right then hands-left/right; policy is left arm+hand then right.
    agent = list(ORDER[:7]) + list(ORDER[29:36]) + list(ORDER[7:29]) + list(ORDER[36:])
    mapped = named_agent_history([list(range(58)) for _ in range(16)], ORDER, [n + ".position" for n in agent])
    assert mapped[0] == [list(ORDER).index(n) for n in agent]
    assert mapped[0] != list(range(58))
    with pytest.raises(ValueError, match="name mismatch"):
        named_agent_history([list(range(58))] * 16, ORDER, [f"wrong{i}" for i in range(58)])


def test_gap_breaks_consecutive_confirmation():
    p = PredictionPromptProvider(confirmations=2, max_age_s=2.0)
    p.reset("s")
    assert not accept(p, "harvest", 1.0).changed
    assert not accept(p, "harvest", 10.0).changed
    assert accept(p, "harvest", 10.5).changed


def test_old_chunk_cannot_cross_session_reset_even_with_same_prompt_and_version():
    p = PredictionPromptProvider()
    p.reset("s")
    a = accept(p, "pickup", 1.0)
    old = bind_chunk(a, "old")
    p.reset("next-session")
    value = prediction("pickup", 2.0)
    value["session_id"] = "next-session"
    b = p.accept_prediction(value, observation_id="2.0", now=2.1)
    assert a.prompt == b.prompt
    assert a.version == b.version
    assert not chunk_matches_prompt(old, b)
    with pytest.raises(ValueError, match="new session"):
        p.reset("next-session")


def test_unsupported_five_phase_profile_cannot_silently_use_seven_phase_provider():
    with pytest.raises(ValueError, match="five-phase"):
        PredictionPromptProvider(phase_profile="tomato-in-hand-five-phase-v3")


def test_shipped_agent_config_matches_provider_and_remains_disabled():
    import json
    from pathlib import Path

    from openpi.subtasks.agent_bridge import PHASE_PROFILE

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/tomato_vjepa_agent.json").read_text())
    assert config["enabled"] is False
    assert config["agent_checkpoint"] is None
    assert config["execution"]["robot_control_enabled"] is False
    assert config["phase_profile"] == PHASE_PROFILE
    assert config["phases"] == list(PHASES)
    provider = PredictionPromptProvider(
        phase_profile=config["phase_profile"],
        **{key: config["decision"][key] for key in ("threshold", "confirmations", "max_age_s")},
    )
    provider.reset("s")
    assert accept(provider, "pickup", 1.0).phase == config["decision"]["initial_phase"]
    memory = json.loads((root / "configs/tomato_memory_experiment.json").read_text())
    assert memory["enabled"] is False
    assert memory["phase_profile"] == PHASE_PROFILE
    assert memory["memory_kind"] == "language_operation_results"
