from tools.m4_llama_v5_analysis import (
    M4LlamaV5AnalysisError,
    ReplicateEndpoint,
    ReplicateInference,
    combine_replicates,
    groups_by_start_version,
)
from tools.opportunity_ledger_join import JoinedOpportunityAssignment


def endpoint(estimate: float, shift: float = 0.0) -> ReplicateEndpoint:
    return ReplicateEndpoint(estimate, 0.02, (shift,) * 100)


def replicate(name: str, estimate: float) -> ReplicateInference:
    return ReplicateInference(name, endpoint(estimate), endpoint(estimate), 0.0, 0.0)


def test_group_counter_uses_joined_dataclass_field() -> None:
    rows = [
        JoinedOpportunityAssignment("a", 0, 8, "control", 1.0, True),
        JoinedOpportunityAssignment("b", 1, 8, "d5", 2.0, False),
        JoinedOpportunityAssignment("c", 2, 9, "control", 3.0, True),
    ]
    assert groups_by_start_version(rows) == {8: 2, 9: 1}


def test_equal_weight_independent_combination() -> None:
    result = combine_replicates(
        {"r1": replicate("r1", 0.30), "r2": replicate("r2", 0.34)},
        expected_bootstrap_draws=100,
    )
    assert result.identification_interval == (0.32, 0.32)
    assert result.conclusion == "MATERIAL"
    assert result.coverage_gate_passed


def test_missingness_fails_closed_per_replicate_and_arm() -> None:
    bad = ReplicateInference("r2", endpoint(0.4), endpoint(0.4), 0.011, 0.0)
    result = combine_replicates(
        {"r1": replicate("r1", 0.4), "r2": bad}, expected_bootstrap_draws=100
    )
    assert result.conclusion == "INSUFFICIENT_TERMINAL_COVERAGE"


def test_rejects_wrong_window() -> None:
    bad = ReplicateInference(
        "r2", endpoint(0.4), endpoint(0.4), 0.0, 0.0, primary_end_version=408
    )
    try:
        combine_replicates(
            {"r1": replicate("r1", 0.4), "r2": bad},
            expected_bootstrap_draws=100,
        )
    except M4LlamaV5AnalysisError:
        pass
    else:
        raise AssertionError("wrong window accepted")
