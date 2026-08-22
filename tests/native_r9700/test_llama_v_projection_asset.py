"""Static no-hardware contract for the Llama V projection kernel source."""

import ctypes
from pathlib import Path


V_PROJECTION_SOURCE = Path("native_r9700/kernels/llama_v_projection_f16.cpp")




class FrozenVProjectionKernargs(ctypes.Structure):
    _fields_ = [
        ("normalized", ctypes.c_uint64),
        ("v_projection_weight", ctypes.c_uint64),
        ("fresh_v", ctypes.c_uint64),
        ("sequence_length", ctypes.c_uint32),
    ]


def test_v_projection_kernargs_are_the_frozen_32_byte_stage_layout() -> None:
    """The device source ABI has three pointers then N at byte 24."""
    assert ctypes.sizeof(FrozenVProjectionKernargs) == 32
    assert FrozenVProjectionKernargs.normalized.offset == 0
    assert FrozenVProjectionKernargs.v_projection_weight.offset == 8
    assert FrozenVProjectionKernargs.fresh_v.offset == 16
    assert FrozenVProjectionKernargs.sequence_length.offset == 24

def source_text() -> str:
    assert V_PROJECTION_SOURCE.is_file(), (
        "missing capability: Llama V projection HSA source is not checked in"
    )
    return V_PROJECTION_SOURCE.read_text(encoding="utf-8")


def test_v_projection_source_matches_the_frozen_32_byte_stage_abi() -> None:
    """The source parameter order is normalized, V weight, fresh V, then N."""
    source = source_text()

    signature = """extern \"C\" __attribute__((global)) void llama_v_projection_f16(
    const unsigned short* normalized,
    const unsigned short* v_projection_weight,
    unsigned short* fresh_v,
    unsigned int sequence_length)"""
    assert signature in source
    assert "const unsigned short* normalized" in source
    assert "const unsigned short* v_projection_weight" in source
    assert "unsigned short* fresh_v" in source
    assert "unsigned int sequence_length" in source


def test_v_projection_source_projects_one_fp16_weight_row_to_unrotated_fresh_v() -> None:
    """Each 64-lane workgroup maps one token/KV-head tile into (1,8,N,64)."""
    source = source_text()

    assert "constexpr unsigned int kHiddenSize = 2048U;" in source
    assert "constexpr unsigned int kKvHeadCount = 8U;" in source
    assert "constexpr unsigned int kHeadDimension = 64U;" in source
    assert "const unsigned long long token_index = workgroup / kKvHeadCount;" in source
    assert "if (token_index >= (unsigned long long)sequence_length) return;" in source
    assert "if (lane >= kHeadDimension) return;" in source
    assert "const unsigned int output_channel = kv_head * kHeadDimension + lane;" in source
    assert "for (unsigned int column = 0U; column < kHiddenSize; ++column)" in source
    assert "float accumulator = 0.0f;" in source
    assert "accumulator += (float)input * (float)weight;" in source
    assert "fresh_v[output_offset] = __builtin_bit_cast(unsigned short, (_Float16)accumulator);" in source


def test_v_projection_source_has_no_cache_rotation_or_host_fallback() -> None:
    """V source is only the direct device projection; later stages own cache and RoPE."""
    source = source_text().lower()

    for forbidden in ("rope", "cache", "fixture", "numpy", "tinygrad", "mlx", "cpu"):
        assert forbidden not in source
