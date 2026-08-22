"""No-hardware source contract for the native Llama K projection kernel."""

import re
from pathlib import Path


K_PROJECTION_SOURCE = Path("native_r9700/kernels/llama_k_projection_f16.cpp")


def test_llama_k_projection_source_uses_the_frozen_device_only_abi() -> None:
    """K projection consumes one fp16 device window and emits token-local fresh K."""
    assert K_PROJECTION_SOURCE.is_file(), (
        "missing capability: frozen Llama K projection HSA source is not checked in"
    )
    source_text = K_PROJECTION_SOURCE.read_text(encoding="utf-8")

    signature = re.search(
        r'extern\s+"C"\s+[^\n]*\bllama_k_projection_f16\s*\(([^)]*)\)',
        source_text,
    )
    assert signature is not None, "source must expose the frozen C-linkage K kernel"
    parameters = [parameter.strip() for parameter in signature.group(1).split(",")]
    assert len(parameters) == 4, "K ABI must contain exactly three pointers and sequence length"
    assert all("*" in parameter for parameter in parameters[:3]), (
        "K ABI must pass normalized input, K weight, and fresh K by pointer"
    )
    assert "normalized" in parameters[0]
    assert "k_projection_weight" in parameters[1]
    assert "fresh_k" in parameters[2]
    assert "sequence_length" in parameters[3] and "*" not in parameters[3]
    assert "unsigned int" in parameters[3], "sequence length must occupy the 32-bit tail"

    assert "const unsigned int token = workgroup / 8U;" in source_text
    assert "if (token >= sequence_length || head_dimension >= 64U) return;" in source_text
    assert "const unsigned int projection_row = kv_head * 64U + head_dimension;" in source_text
    assert "((unsigned long long)kv_head * sequence_length + token) * 64ULL" in source_text
    assert "2048ULL" in source_text
    assert "float accumulator" in source_text
    assert "unsigned short" in source_text, "all K source buffers must use fp16 storage"
    assert "__builtin_amdgcn_workgroup_id_x" in source_text
    assert "__builtin_amdgcn_workitem_id_x" in source_text
    assert "main(" not in source_text, "device source must not contain host logic"
    for forbidden in (
        "__shared__",
        "__builtin_amdgcn_lds",
        "fixture",
        "archive",
        "hiplaunch",
        "hipmalloc",
        "hipfree",
        "hipmemcpy",
        "std::",
        "cpu",
        "numpy",
        "torch",
        "rope",
        "cache",
        "query",
        "attention",
    ):
        assert forbidden not in source_text.lower(), (
            f"fresh K source must not depend on forbidden {forbidden!r} machinery"
        )
