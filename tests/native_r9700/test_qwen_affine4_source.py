"""No-hardware source contract for the text-only Qwen affine4 linear kernel."""

from pathlib import Path


QWEN_AFFINE4_LINEAR_SOURCE = Path("native_r9700/kernels/qwen_affine4_linear.cpp")


def test_qwen_affine4_linear_source_is_text_only_group64_device_dequantization() -> None:
    """The affine4 source decodes one raw group-64 text linear window on device."""
    assert QWEN_AFFINE4_LINEAR_SOURCE.is_file(), "missing Qwen affine4 linear HSA source"
    source = QWEN_AFFINE4_LINEAR_SOURCE.read_text()

    for declaration in (
        'extern "C" __attribute__((global)) void qwen_affine4_linear(',
        "const unsigned short* input",
        "const unsigned char* packed_weight",
        "const unsigned short* scales",
        "const unsigned short* biases",
        "unsigned short* output",
        "unsigned long long input_features",
        "unsigned long long output_features",
        "unsigned long long input_capacity_elements",
        "unsigned long long packed_weight_capacity_bytes",
        "unsigned long long affine_group_capacity",
        "unsigned long long output_capacity_elements",
    ):
        assert declaration in source

    assert "input_index / 64ULL" in source
    assert "packed_weight[element_index >> 1U]" in source
    assert "(packed >> nibble_shift) & 0x0fU" in source
    assert "input_features % 64ULL != 0ULL" in source
    assert "__builtin_mul_overflow(output_features, input_features, &weight_elements)" in source
    assert "__builtin_mul_overflow(output_features, groups_per_output, &group_count)" in source
    assert "input_capacity_elements < input_features" in source
    assert "packed_weight_capacity_bytes < packed_bytes" in source
    assert "affine_group_capacity < group_count" in source
    assert "output_capacity_elements < output_features" in source
    assert "const float dequantized = (float)quantized * scale + bias" in source
    assert "float accumulated = 0.0f" in source
    assert "output[output_index] = __builtin_bit_cast(unsigned short, (_Float16)accumulated)" in source

    forbidden = (
        "llama",
        "cache",
        "cpu",
        "host",
        "fallback",
        "fixture",
        "quantizer",
        "registry",
        "framework",
        "tinygrad",
        "numpy",
        "mlx",
        "multimodal",
        "vision",
        "deltanet",
        "attention",
        "full_model",
    )
    assert not any(term in source.lower() for term in forbidden)
