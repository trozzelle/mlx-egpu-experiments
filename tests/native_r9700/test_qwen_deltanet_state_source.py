"""Source-only contract for the bounded Qwen ArraysCache DeltaNet update."""

from pathlib import Path


DELTANET = Path("native_r9700/kernels/qwen_deltanet_state.cpp")


def test_deltanet_consumes_one_bounded_arrays_cache_window_on_device() -> None:
    source = DELTANET.read_text(encoding="utf-8")

    for token in (
        "qwen_deltanet_state",
        "const unsigned short* hidden_input",
        "unsigned short* hidden_output",
        "unsigned short* convolution_state",
        "float* recurrent_state",
        "unsigned int position",
        "hidden_capacity_elements",
        "convolution_state_capacity_elements",
        "recurrent_state_capacity_elements",
        "output_capacity_elements",
    ):
        assert token in source

    assert "position % convolution_width" in source
    assert "convolution_state[convolution_index]" in source
    assert "recurrent_state[recurrent_index]" in source
    assert "__builtin_amdgcn_workgroup_id_x" in source

    normalized = source.lower()
    for forbidden in (
        "numpy",
        "tinygrad",
        "mlx",
        "cpu",
        "host",
        "fixture",
        "archive",
        "class ",
        "struct ",
        "main(",
    ):
        assert forbidden not in normalized
