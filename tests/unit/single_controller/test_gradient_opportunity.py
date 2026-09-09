"""Unit tests for exact GRPO coefficient opportunity and its scalar ledger."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.async_utils.gradient_opportunity import (
    GRPOOpportunityInputs,
    GradientOpportunityRecorder,
    GroupOpportunitySummary,
    compute_grpo_gradient_opportunity,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.algorithms.single_controller_utils.utils import (
    compute_advantages_from_data,
    prepare_advantage_inputs_from_data,
)


class _ConfiguredProductionGRPOEstimator:
    """Small configured-estimator stand-in for the module's injection boundary."""

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del prompt_ids, kwargs
        if rewards.shape != (2,):
            raise ValueError("the focused fixture requires exactly two siblings")
        leave_one_out_baseline = rewards.flip(0)
        return (rewards - leave_one_out_baseline).unsqueeze(-1).expand(mask.shape)

    def compute_single_prompt_scalar_advantage(
        self, *, prompt_ids: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        del prompt_ids
        if rewards.shape != (2,):
            raise ValueError("the focused fixture requires exactly two siblings")
        return rewards - rewards.flip(0)


def _estimator() -> _ConfiguredProductionGRPOEstimator:
    return _ConfiguredProductionGRPOEstimator()


def _prepare_inputs(train_batch: Mapping[str, Any]) -> GRPOOpportunityInputs:
    rewards = train_batch["rewards"].float()
    token_mask = train_batch["token_mask"].float()
    sample_mask = train_batch["sample_mask"].float()
    return GRPOOpportunityInputs(
        prompt_ids=train_batch["prompt_ids"],
        rewards=rewards,
        actor_mask=token_mask * sample_mask.unsqueeze(-1),
        repeated_batch={"total_reward": rewards},
        estimator_kwargs={},
    )


def _batch(
    *,
    rewards: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "prompt_ids": torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
        "rewards": rewards if rewards is not None else torch.tensor([1.0, 0.0]),
        "token_mask": token_mask
        if token_mask is not None
        else torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.float32),
        "sample_mask": sample_mask
        if sample_mask is not None
        else torch.ones(2, dtype=torch.float32),
    }


def _summary(train_batch: Mapping[str, Any] | None = None) -> GroupOpportunitySummary:
    return compute_grpo_gradient_opportunity(
        train_batch or _batch(),
        group_id="group",
        sample_ids=("group_g0", "group_g1"),
        start_weight_version=3,
        truncation=(False, True),
        estimator=_estimator(),
        prepare_inputs=_prepare_inputs,
    )


def test_exact_loss_aligned_float64_opportunity_math() -> None:
    summary = _summary()

    assert summary.opportunity == 3.0
    assert summary.l2_coefficient_mass == pytest.approx(3**0.5)
    assert summary.signed_coefficient_mass == 1.0
    assert summary.positive_coefficient_mass == 2.0
    assert summary.negative_coefficient_mass == 1.0
    assert summary.valid_actor_tokens == 3
    assert summary.nonzero_advantage_siblings == 2
    assert summary.nonzero_advantage_tokens == 3
    assert [sibling.scalar_advantage for sibling in summary.siblings] == [1.0, -1.0]
    assert [sibling.valid_actor_tokens for sibling in summary.siblings] == [2, 1]
    assert [sibling.opportunity for sibling in summary.siblings] == [2.0, 1.0]
    assert [sibling.truncated for sibling in summary.siblings] == [False, True]


def test_vectorized_sibling_reductions_match_rowwise_float64_reference() -> None:
    generator = torch.Generator().manual_seed(20260909)
    rewards = torch.randn(8, generator=generator, dtype=torch.float32)
    token_mask = torch.randint(
        0, 2, (8, 2048), generator=generator, dtype=torch.int64
    ).float()
    sample_mask = torch.randint(0, 2, (8,), generator=generator).float()
    actor_mask = token_mask * sample_mask.unsqueeze(-1)

    class FixedEstimator:
        def compute_advantage(self, prompt_ids, rewards, mask, **kwargs):
            del prompt_ids, rewards, kwargs
            scalars = torch.randn(8, generator=generator, dtype=torch.float32)
            return scalars.unsqueeze(-1).expand(mask.shape)

        def compute_single_prompt_scalar_advantage(self, *, prompt_ids, rewards):
            del prompt_ids, rewards
            return torch.randn(8, generator=generator, dtype=torch.float32)

    inputs = GRPOOpportunityInputs(
        prompt_ids=torch.ones((8, 64), dtype=torch.long),
        rewards=rewards,
        actor_mask=actor_mask,
        repeated_batch={"total_reward": rewards},
        estimator_kwargs={},
    )
    summary = compute_grpo_gradient_opportunity(
        {},
        group_id="group",
        sample_ids=tuple(f"group_g{index}" for index in range(8)),
        start_weight_version=3,
        truncation=(False,) * 8,
        estimator=FixedEstimator(),
        prepare_inputs=lambda _: inputs,
    )

    aligned_mask = actor_mask[:, 1:].to(dtype=torch.float64)
    for index, sibling in enumerate(summary.siblings):
        scalar = sibling.scalar_advantage
        coefficients = aligned_mask[index] * scalar
        assert sibling.valid_actor_tokens == int(
            aligned_mask[index].sum(dtype=torch.float64).item()
        )
        assert sibling.opportunity == coefficients.abs().sum(dtype=torch.float64).item()


@pytest.mark.parametrize("normalize_rewards", [False, True])
@pytest.mark.parametrize("use_leave_one_out_baseline", [False, True])
@pytest.mark.parametrize("sibling_count", [1, 2, 8])
def test_single_prompt_scalar_advantages_are_bit_exact(
    normalize_rewards: bool,
    use_leave_one_out_baseline: bool,
    sibling_count: int,
) -> None:
    generator = torch.Generator().manual_seed(
        20260910 + sibling_count + int(normalize_rewards) * 10
    )
    prompt = torch.randint(0, 32000, (1, 64), generator=generator)
    prompt_ids = prompt.expand(sibling_count, -1).clone()
    rewards = torch.randn(sibling_count, generator=generator, dtype=torch.float32)
    mask = torch.randint(
        0,
        2,
        (sibling_count, 2048),
        generator=generator,
        dtype=torch.int64,
    ).float()
    estimator = GRPOAdvantageEstimator(
        AdvEstimatorConfig(
            normalize_rewards=normalize_rewards,
            use_leave_one_out_baseline=use_leave_one_out_baseline,
        ),
        ClippedPGLossConfig(),
    )

    expanded = estimator.compute_advantage(prompt_ids, rewards, mask)[:, 0]
    scalar = estimator.compute_single_prompt_scalar_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
    )

    assert torch.equal(scalar, expanded)


def test_single_prompt_scalar_advantage_rejects_multiple_prompts() -> None:
    estimator = GRPOAdvantageEstimator(AdvEstimatorConfig(), ClippedPGLossConfig())
    prompt_ids = torch.tensor([[1, 2], [1, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="multiple prompts"):
        estimator.compute_single_prompt_scalar_advantage(
            prompt_ids=prompt_ids,
            rewards=torch.tensor([1.0, 0.0]),
        )


def test_zero_variance_and_zero_valid_siblings_are_retained() -> None:
    zero_variance = _summary(_batch(rewards=torch.tensor([1.0, 1.0])))
    assert zero_variance.opportunity == 0.0
    assert len(zero_variance.siblings) == 2
    assert [sibling.opportunity for sibling in zero_variance.siblings] == [0.0, 0.0]

    masked = _summary(
        _batch(
            token_mask=torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=torch.float32),
            sample_mask=torch.tensor([1.0, 0.0]),
        )
    )
    assert masked.opportunity == 2.0
    assert masked.valid_actor_tokens == 2
    assert masked.siblings[1].scalar_advantage == -1.0
    assert masked.siblings[1].valid_actor_tokens == 0
    assert masked.siblings[1].opportunity == 0.0

    no_valid_tokens = _summary(
        _batch(token_mask=torch.zeros((2, 4), dtype=torch.float32))
    )
    assert no_valid_tokens.opportunity == 0.0
    assert no_valid_tokens.valid_actor_tokens == 0
    assert no_valid_tokens.nonzero_advantage_siblings == 2
    assert no_valid_tokens.nonzero_advantage_tokens == 0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_inputs(bad_value: float) -> None:
    train_batch = _batch(rewards=torch.tensor([1.0, bad_value]))

    with pytest.raises(ValueError, match="nonfinite"):
        _summary(train_batch)


def test_rejects_nonfinite_estimator_output() -> None:
    class NonfiniteEstimator:
        def compute_advantage(self, prompt_ids, rewards, mask, **kwargs):
            del prompt_ids, rewards, kwargs
            return torch.full_like(mask, float("nan"))

        def compute_single_prompt_scalar_advantage(self, *, prompt_ids, rewards):
            del prompt_ids
            return torch.full_like(rewards, float("nan"))

    with pytest.raises(ValueError, match="advantages contain a nonfinite"):
        compute_grpo_gradient_opportunity(
            _batch(),
            group_id="group",
            sample_ids=("group_g0", "group_g1"),
            start_weight_version=3,
            truncation=(False, False),
            estimator=NonfiniteEstimator(),
            prepare_inputs=_prepare_inputs,
        )


def test_opportunity_computation_does_not_mutate_train_batch() -> None:
    train_batch = _batch()
    before = {name: tensor.clone() for name, tensor in train_batch.items()}

    _summary(train_batch)

    for name, tensor in train_batch.items():
        assert torch.equal(tensor, before[name])


def test_shared_production_helper_matches_audit_estimator_and_alignment() -> None:
    train_batch = {
        "prompt_ids_for_adv": torch.tensor([[7], [7]], dtype=torch.long),
        "total_reward": torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        "token_mask": torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.int64),
        "sample_mask": torch.tensor([[1], [1]], dtype=torch.int64),
    }
    advantage_config = AdvantageConfig()
    estimator = GRPOAdvantageEstimator(
        AdvEstimatorConfig(),
        ClippedPGLossConfig(),
    )
    production = compute_advantages_from_data(
        train_batch,
        advantage_config=advantage_config,
        advantage_estimator=estimator,
        policy_logprobs_required=False,
        reference_logprobs_required=False,
    )

    def _prepare_shared(data: Mapping[str, Any]) -> GRPOOpportunityInputs:
        prepared = prepare_advantage_inputs_from_data(
            data,
            advantage_config=advantage_config,
            policy_logprobs_required=False,
            reference_logprobs_required=False,
        )
        return GRPOOpportunityInputs(
            prompt_ids=prepared.prompt_ids,
            rewards=prepared.rewards,
            actor_mask=prepared.actor_mask,
            repeated_batch=prepared.repeated_batch,
            estimator_kwargs=prepared.estimator_kwargs,
        )

    summary = compute_grpo_gradient_opportunity(
        train_batch,
        group_id="group",
        sample_ids=("group_g0", "group_g1"),
        start_weight_version=0,
        truncation=(False, False),
        estimator=estimator,
        prepare_inputs=_prepare_shared,
    )

    expected = (
        (production.advantages[:, 1:] * production.actor_mask[:, 1:])
        .abs()
        .double()
        .sum()
    )
    assert summary.opportunity == expected.item()
    assert [s.scalar_advantage for s in summary.siblings] == pytest.approx(
        production.advantages[:, 0].tolist()
    )
    assert production.rewards.dtype == torch.float32
    assert production.actor_mask.dtype == torch.float32


class _Sequencer:
    def __init__(self) -> None:
        self._sequence = 10

    def __call__(self) -> tuple[int, int]:
        current = self._sequence
        self._sequence += 1
        return current, 123


def _write_ledger(path: Path) -> bytes:
    recorder = GradientOpportunityRecorder(
        run_id="run",
        clock_domain_id="controller",
        sequencer=_Sequencer(),
    )
    recorder.append_header(
        estimator_name="GRPOAdvantageEstimator",
        estimator_settings={
            "normalize_rewards": True,
            "use_leave_one_out_baseline": True,
        },
        loss_settings={"token_level_loss": True, "use_cispo": False},
    )
    recorder.append_group(_summary())
    recorder.append_train_step_completed(
        previous_learner_version=3,
        learner_version=4,
        sample_ids=("group_g0", "group_g1"),
    )
    recorder.flush_jsonl(path)
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    return path.read_bytes()


def test_canonical_atomic_ledger_is_byte_deterministic(tmp_path: Path) -> None:
    first = _write_ledger(tmp_path / "first.jsonl")
    second = _write_ledger(tmp_path / "second.jsonl")

    assert first == second
    assert b"NaN" not in first
    assert b" " not in first
    rows = [json.loads(line) for line in first.splitlines()]
    assert [row["event_type"] for row in rows] == [
        "header",
        "group",
        "train_step_completed",
    ]
    assert [row["controller_sequence"] for row in rows] == [10, 11, 12]
    assert {row["timestamp_ns"] for row in rows} == {123}
    assert rows[2]["sample_ids"] == ["group_g0", "group_g1"]
    assert "prompt_ids" not in rows[1]
    assert "token_mask" not in rows[1]


def test_completed_step_rejects_unknown_reused_and_invalid_ids() -> None:
    recorder = GradientOpportunityRecorder(
        run_id="run",
        clock_domain_id="controller",
        sequencer=_Sequencer(),
    )
    recorder.append_header(
        estimator_name="GRPOAdvantageEstimator",
        estimator_settings={},
        loss_settings={},
    )
    recorder.append_group(_summary())

    with pytest.raises(ValueError, match="lack opportunity"):
        recorder.append_train_step_completed(
            previous_learner_version=3,
            learner_version=4,
            sample_ids=("unknown",),
        )

    recorder.append_train_step_completed(
        previous_learner_version=3,
        learner_version=4,
        sample_ids=("group_g0", "group_g1"),
    )
    with pytest.raises(ValueError, match="already belong"):
        recorder.append_train_step_completed(
            previous_learner_version=4,
            learner_version=5,
            sample_ids=("group_g0",),
        )
