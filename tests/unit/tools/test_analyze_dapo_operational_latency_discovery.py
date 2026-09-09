import pytest

from tools import analyze_dapo_operational_latency_discovery as analyzer
from tools import materialize_dapo_operational_latency_discovery as materializer


def _observations(*, reverse_latency: bool = False):
    output = []
    for pool_seed, _ in materializer.POOL_SPECS:
        for index in range(16):
            load = 100 + index * 10
            latency = 1000 + (15 - index if reverse_latency else index) * 100
            item = analyzer.DesignItem(
                source_pool_ordinal=index,
                source_prompt_id=f"prompt-{pool_seed}-{index}",
                task_name=materializer.SOURCE_IDS[index % 2],
                canonical_prompt_sha256=f"{pool_seed:05d}{index:02d}".ljust(64, "0"),
                source_dataset_index=index,
                source_duplicate_count=100,
                rendered_prompt_tokens=100,
            )
            output.append(
                analyzer.Observation(
                    pool_seed=pool_seed,
                    item=item,
                    dispatch_ns=index * 10,
                    ready_ns=index * 10 + latency,
                    reward_mean=0.5,
                    reward_min=0.0,
                    reward_max=1.0,
                    mean_generated_tokens=float(load),
                    min_generated_tokens=load - 10,
                    max_generated_tokens=load,
                    length_terminations=0,
                )
            )
    return tuple(output)


def test_pooled_within_pool_spearman_uses_all_three_centered_pools() -> None:
    assert analyzer._pooled_within_pool_spearman(_observations()) == pytest.approx(1.0)
    assert analyzer._pooled_within_pool_spearman(
        _observations(reverse_latency=True)
    ) == pytest.approx(-1.0)


def test_percentile_uses_linear_interpolation() -> None:
    assert analyzer._percentile([0.0, 10.0, 20.0, 30.0], 0.1) == pytest.approx(3.0)
    assert analyzer._percentile([0.0, 10.0, 20.0, 30.0], 0.9) == pytest.approx(27.0)
