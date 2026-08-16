"""Per-layer delta aggregation tests (no GPU required)."""

from tinygrad_kv_worker.harness import _aggregate_layer_deltas


def test_aggregate_layer_deltas_keeps_worst_case_across_prompts():
    prompt0 = [
        {
            "max_K": 0.1,
            "mean_K": 0.01,
            "max_V": 0.2,
            "mean_V": 0.02,
            "over_tolerance": False,
        }
    ]
    prompt1 = [
        {
            "max_K": 0.3,
            "mean_K": 0.005,
            "max_V": 0.1,
            "mean_V": 0.04,
            "over_tolerance": True,
        }
    ]

    aggregate = _aggregate_layer_deltas([], prompt0)
    aggregate = _aggregate_layer_deltas(aggregate, prompt1)

    assert aggregate == [
        {
            "max_K": 0.3,
            "mean_K": 0.01,
            "max_V": 0.2,
            "mean_V": 0.04,
            "over_tolerance": True,
        }
    ]
