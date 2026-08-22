"""No-hardware contracts for native Llama O projection and gated MLP sources."""

from pathlib import Path


O_SOURCE = Path("native_r9700/kernels/llama_o_projection_f16.cpp")
MLP_SOURCE = Path("native_r9700/kernels/llama_gated_mlp_f16.cpp")


def test_o_projection_uses_device_context_weight_residual_and_fp32_accumulation() -> None:
    source = O_SOURCE.read_text(encoding="utf-8")
    for parameter in ("context", "o_projection_weight", "residual", "post_attention_hidden", "sequence_length"):
        assert parameter in source
    assert "kHiddenSize = 2048U" in source
    assert "float accumulator" in source
    assert "accumulator + skip" in source
    assert "__builtin_amdgcn_workgroup_id_x" in source


def test_gated_mlp_keeps_rmsnorm_silu_gate_up_down_and_residual_on_device() -> None:
    source = MLP_SOURCE.read_text(encoding="utf-8")
    for parameter in (
        "post_attention_hidden",
        "post_attention_layernorm_weight",
        "gate_projection_weight",
        "up_projection_weight",
        "down_projection_weight",
        "hidden",
        "sequence_length",
    ):
        assert parameter in source
    assert "kIntermediateSize = 8192U" in source
    assert "sum_of_squares" in source
    assert "silu_gate" in source
    assert "silu_gate * up * down" in source
    assert "accumulator + residual" in source


def test_o_mlp_sources_exclude_host_and_fixture_compute() -> None:
    forbidden = ("fixture", "archive", "cpu", "numpy", "tinygrad", "mlx", "hiplaunch", "main(")
    for path in (O_SOURCE, MLP_SOURCE):
        source = path.read_text(encoding="utf-8").lower()
        assert not any(marker in source for marker in forbidden)
