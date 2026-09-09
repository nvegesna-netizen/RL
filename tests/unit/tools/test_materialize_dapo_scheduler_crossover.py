from tools import materialize_dapo_operational_latency_discovery as discovery
from tools import materialize_dapo_scheduler_crossover as materializer


def _prompts(count: int = 160) -> list[discovery.UniquePrompt]:
    return [
        discovery.UniquePrompt(
            canonical_sha256=f"{index:064x}",
            prompt=f"problem {index}",
            ground_truth=str(index),
            first_source_index=index,
            source_extra_index=f"source-{index}",
            duplicate_count=100,
        )
        for index in range(count)
    ]


def test_holdout_selection_is_deterministic_disjoint_and_excludes_discovery() -> None:
    prompts = _prompts()

    first, first_discovery = materializer._select_holdout_pools(prompts)
    second, second_discovery = materializer._select_holdout_pools(prompts)

    assert first == second
    assert first_discovery == second_discovery
    assert len(first_discovery) == 48
    holdout = [item.canonical_sha256 for pool in first.values() for item in pool]
    assert len(holdout) == 64
    assert len(set(holdout)) == 64
    assert not first_discovery.intersection(holdout)
    assert tuple(first) == (48001, 48002, 48003, 48004)


def test_candidate_and_confirmation_hashes_are_exact() -> None:
    assert materializer.PROTOCOL_SHA256 == (
        "c49f9604225db847eded1d298c592938299fb3c3d508cf60d2b598f1e8b604c7"
    )
    assert materializer.CONFIRMATION_SHA256 == (
        "a73a4b56744cc31fe5a50f4cccb2b1dd985357ca1879b8377df19c6b1622e91e"
    )
    assert materializer.POOL_SPECS == (
        (48001, 58001, 68001),
        (48002, 58002, 68002),
        (48003, 58003, 68003),
        (48004, 58004, 68004),
    )
