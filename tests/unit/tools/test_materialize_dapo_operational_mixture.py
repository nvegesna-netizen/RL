from tools import materialize_dapo_operational_latency_discovery as discovery
from tools import materialize_dapo_operational_mixture as materializer


def _prompts(count: int = 400) -> list[discovery.UniquePrompt]:
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


def test_fresh_selection_excludes_all_prior_and_intervening_identities() -> None:
    prompts = _prompts()
    prior_discovery, prior_crossover = materializer._prior_ids(prompts)
    intervening = {prompts[-1].canonical_sha256}

    first, first_discovery, first_crossover = materializer._select_fresh_pools(
        prompts, intervening
    )
    second, _, _ = materializer._select_fresh_pools(prompts, intervening)

    assert first == second
    assert first_discovery == prior_discovery
    assert first_crossover == prior_crossover
    selected = {item.canonical_sha256 for pool in first.values() for item in pool}
    assert len(selected) == 96
    assert not selected.intersection(prior_discovery | prior_crossover | intervening)


def test_confirmed_bindings_and_counterbalanced_orders_are_exact() -> None:
    assert materializer.PROTOCOL_SHA256 == (
        "e9942c4afbec0b3b622079d1208eaba6a7c83a7759fc307cfd47293e797a9f63"
    )
    assert materializer.CONFIRMATION_SHA256 == (
        "f6a55f89f641d46b3942fbc05eac46f071665eb21435211f035dc3ad99faf969"
    )
    expected_arms = {
        f"{level}_{sampler}"
        for level in ("l0", "l1", "l3")
        for sampler in ("in_order", "ready_first")
    }
    assert all(
        len(order) == 6 and set(order) == expected_arms
        for order in materializer.ARM_EXECUTION_ORDERS.values()
    )
