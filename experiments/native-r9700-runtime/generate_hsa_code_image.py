#!/usr/bin/env python3
"""Generate a page-layout-preserving HSA code image from fresh HIP source.

This generation-only tool compiles the checked-in Llama embedding kernel through
COMGR, validates the resulting ELF code object, resolves its admitted REL64
relocations, and publishes an image plus a digest-bound manifest.  It neither
loads the product runtime nor creates a device or dispatches GPU work.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import secrets
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

KERNEL_NAME = "llama_embed_row_f16"
TARGET = "gfx1201"
KERNARG_SCHEMA = {
    "name": "llama-embed-row-f16-v1",
    "bytes": 24,
    "fields": [
        {"name": "embedding_rows", "offset": 0, "type": "uint64"},
        {"name": "hidden_output", "offset": 8, "type": "uint64"},
        {"name": "selected_row", "offset": 16, "type": "uint64"},
    ],
}
RMSNORM_KERNEL_NAME = "llama_rmsnorm_f16"
RMSNORM_KERNARG_SCHEMA = {
    "name": "llama-rmsnorm-f16-v1",
    "bytes": 32,
    "fields": [
        {"name": "hidden_input", "offset": 0, "type": "uint64"},
        {"name": "scale", "offset": 8, "type": "uint64"},
        {"name": "hidden_output", "offset": 16, "type": "uint64"},
        {"name": "epsilon", "offset": 24, "type": "float32"},
    ],
}
RMSNORM_ZERO_STORE_KERNEL_NAME = "llama_rmsnorm_zero_store_f16"
RMSNORM_ZERO_STORE_KERNARG_SCHEMA = RMSNORM_KERNARG_SCHEMA
RMSNORM_EPSILON_ARITHMETIC_KERNEL_NAME = "llama_rmsnorm_epsilon_arithmetic_f16"
RMSNORM_EPSILON_ARITHMETIC_KERNARG_SCHEMA = RMSNORM_KERNARG_SCHEMA



K_PROJECTION_KERNEL_NAME = "llama_k_projection_f16"
K_PROJECTION_KERNARG_SCHEMA = {
    "name": "llama-k-projection-f16-v1",
    "bytes": 32,
    "fields": [
        {"name": "normalized", "offset": 0, "type": "uint64"},
        {"name": "k_projection_weight", "offset": 8, "type": "uint64"},
        {"name": "fresh_k", "offset": 16, "type": "uint64"},
        {"name": "sequence_length", "offset": 24, "type": "uint32"},
    ],
}
V_PROJECTION_KERNEL_NAME = "llama_v_projection_f16"
V_PROJECTION_KERNARG_SCHEMA = {
    "name": "llama-v-projection-f16-v1",
    "bytes": 32,
    "fields": [
        {"name": "normalized", "offset": 0, "type": "uint64"},
        {"name": "v_projection_weight", "offset": 8, "type": "uint64"},
        {"name": "fresh_v", "offset": 16, "type": "uint64"},
        {"name": "sequence_length", "offset": 24, "type": "uint32"},
    ],
}
QWEN_AFFINE4_KERNEL_NAME = "qwen_affine4_linear"
QWEN_AFFINE4_KERNARG_SCHEMA = {
    "name": "qwen-affine4-linear-v1",
    "bytes": 88,
    "fields": [
        {"name": "input", "offset": 0, "type": "uint64"},
        {"name": "packed_weight", "offset": 8, "type": "uint64"},
        {"name": "scales", "offset": 16, "type": "uint64"},
        {"name": "biases", "offset": 24, "type": "uint64"},
        {"name": "output", "offset": 32, "type": "uint64"},
        {"name": "input_features", "offset": 40, "type": "uint64"},
        {"name": "output_features", "offset": 48, "type": "uint64"},
        {"name": "input_capacity_elements", "offset": 56, "type": "uint64"},
        {"name": "packed_weight_capacity_bytes", "offset": 64, "type": "uint64"},
        {"name": "affine_group_capacity", "offset": 72, "type": "uint64"},
        {"name": "output_capacity_elements", "offset": 80, "type": "uint64"},
    ],
}

ROPE_KV_KERNEL_NAME = "llama_rope_kv_f16"
ROPE_KV_KERNARG_SCHEMA = {
    "name": "llama-rope-kv-f16-v1",
    "bytes": 48,
    "fields": [
        {"name": "fresh_k", "offset": 0, "type": "uint64"},
        {"name": "fresh_v", "offset": 8, "type": "uint64"},
        {"name": "k_cache", "offset": 16, "type": "uint64"},
        {"name": "v_cache", "offset": 24, "type": "uint64"},
        {"name": "sequence_length", "offset": 32, "type": "uint32"},
        {"name": "position", "offset": 36, "type": "uint32"},
        {"name": "cache_capacity_tokens", "offset": 40, "type": "uint32"},
    ],
}
ATTENTION_SCORE_KERNEL_NAME = "llama_causal_attention_score_f16"
ATTENTION_SCORE_KERNARG_SCHEMA = {
    "name": "llama-causal-attention-score-f16-v1",
    "bytes": 48,
    "fields": [
        {"name": "normalized", "offset": 0, "type": "uint64"},
        {"name": "q_projection_weight", "offset": 8, "type": "uint64"},
        {"name": "k_cache", "offset": 16, "type": "uint64"},
        {"name": "attention_scores", "offset": 24, "type": "uint64"},
        {"name": "sequence_length", "offset": 32, "type": "uint32"},
        {"name": "position", "offset": 36, "type": "uint32"},
        {"name": "cache_capacity_tokens", "offset": 40, "type": "uint32"},
    ],
}
ATTENTION_SOFTMAX_KERNEL_NAME = "llama_causal_attention_softmax_f32"
ATTENTION_SOFTMAX_KERNARG_SCHEMA = {
    "name": "llama-causal-attention-softmax-f32-v1",
    "bytes": 32,
    "fields": [
        {"name": "attention_scores", "offset": 0, "type": "uint64"},
        {"name": "attention_probabilities", "offset": 8, "type": "uint64"},
        {"name": "sequence_length", "offset": 16, "type": "uint32"},
        {"name": "position", "offset": 20, "type": "uint32"},
        {"name": "cache_capacity_tokens", "offset": 24, "type": "uint32"},
    ],
}
ATTENTION_CONTEXT_KERNEL_NAME = "llama_causal_attention_context_f16"
ATTENTION_CONTEXT_KERNARG_SCHEMA = {
    "name": "llama-causal-attention-context-f16-v1",
    "bytes": 40,
    "fields": [
        {"name": "attention_probabilities", "offset": 0, "type": "uint64"},
        {"name": "v_cache", "offset": 8, "type": "uint64"},
        {"name": "context", "offset": 16, "type": "uint64"},
        {"name": "sequence_length", "offset": 24, "type": "uint32"},
        {"name": "position", "offset": 28, "type": "uint32"},
        {"name": "cache_capacity_tokens", "offset": 32, "type": "uint32"},
    ],
}
O_PROJECTION_KERNEL_NAME = "llama_o_projection_f16"
O_PROJECTION_KERNARG_SCHEMA = {
    "name": "llama-o-projection-f16-v1",
    "bytes": 40,
    "fields": [
        {"name": "context", "offset": 0, "type": "uint64"},
        {"name": "o_projection_weight", "offset": 8, "type": "uint64"},
        {"name": "residual", "offset": 16, "type": "uint64"},
        {"name": "post_attention_hidden", "offset": 24, "type": "uint64"},
        {"name": "sequence_length", "offset": 32, "type": "uint32"},
    ],
}
GATED_MLP_KERNEL_NAME = "llama_gated_mlp_f16"
GATED_MLP_KERNARG_SCHEMA = {
    "name": "llama-gated-mlp-f16-v1",
    "bytes": 56,
    "fields": [
        {"name": "post_attention_hidden", "offset": 0, "type": "uint64"},
        {"name": "post_attention_layernorm_weight", "offset": 8, "type": "uint64"},
        {"name": "gate_projection_weight", "offset": 16, "type": "uint64"},
        {"name": "up_projection_weight", "offset": 24, "type": "uint64"},
        {"name": "down_projection_weight", "offset": 32, "type": "uint64"},
        {"name": "hidden", "offset": 40, "type": "uint64"},
        {"name": "sequence_length", "offset": 48, "type": "uint32"},
    ],
}
GATE_UP_PROJECTION_KERNEL_NAME = "llama_gate_up_projection_f16"
GATE_UP_PROJECTION_KERNARG_SCHEMA = {
    "name": "llama-gate-up-projection-f16-v1",
    "bytes": 56,
    "fields": [
        {"name": "post_attention_hidden", "offset": 0, "type": "uint64"},
        {"name": "post_attention_layernorm_weight", "offset": 8, "type": "uint64"},
        {"name": "gate_projection_weight", "offset": 16, "type": "uint64"},
        {"name": "up_projection_weight", "offset": 24, "type": "uint64"},
        {"name": "gate_output", "offset": 32, "type": "uint64"},
        {"name": "up_output", "offset": 40, "type": "uint64"},
        {"name": "sequence_length", "offset": 48, "type": "uint32"},
    ],
}
MLP_DOWN_KERNEL_NAME = "llama_mlp_down_f16"
MLP_DOWN_KERNARG_SCHEMA = {
    "name": "llama-mlp-down-f16-v1",
    "bytes": 48,
    "fields": [
        {"name": "gate_input", "offset": 0, "type": "uint64"},
        {"name": "up_input", "offset": 8, "type": "uint64"},
        {"name": "down_projection_weight", "offset": 16, "type": "uint64"},
        {"name": "residual", "offset": 24, "type": "uint64"},
        {"name": "hidden", "offset": 32, "type": "uint64"},
        {"name": "sequence_length", "offset": 40, "type": "uint32"},
    ],
}
QWEN_DELTANET_KERNEL_NAME = "qwen_deltanet_state"
QWEN_DELTANET_KERNARG_SCHEMA = {
    "name": "qwen-deltanet-state-v1",
    "bytes": 80,
    "fields": [
        {"name": "q", "offset": 0, "type": "uint64"},
        {"name": "k", "offset": 8, "type": "uint64"},
        {"name": "v", "offset": 16, "type": "uint64"},
        {"name": "decay", "offset": 24, "type": "uint64"},
        {"name": "beta", "offset": 32, "type": "uint64"},
        {"name": "state", "offset": 40, "type": "uint64"},
        {"name": "output", "offset": 48, "type": "uint64"},
        {"name": "value_heads", "offset": 56, "type": "uint32"},
        {"name": "key_heads", "offset": 60, "type": "uint32"},
        {"name": "key_dimension", "offset": 64, "type": "uint32"},
        {"name": "value_dimension", "offset": 68, "type": "uint32"},
        {"name": "state_capacity_elements", "offset": 72, "type": "uint64"},
    ],
}
QWEN_FULL_ATTENTION_KERNEL_NAME = "qwen_full_attention"
QWEN_FULL_ATTENTION_KERNARG_SCHEMA = {
    "name": "qwen-full-attention-v1",
    "bytes": 56,
    "fields": [
        {"name": "query", "offset": 0, "type": "uint64"},
        {"name": "k_cache", "offset": 8, "type": "uint64"},
        {"name": "v_cache", "offset": 16, "type": "uint64"},
        {"name": "output", "offset": 24, "type": "uint64"},
        {"name": "query_heads", "offset": 32, "type": "uint32"},
        {"name": "kv_heads", "offset": 36, "type": "uint32"},
        {"name": "head_dimension", "offset": 40, "type": "uint32"},
        {"name": "query_length", "offset": 44, "type": "uint32"},
        {"name": "position", "offset": 48, "type": "uint32"},
        {"name": "cache_capacity_tokens", "offset": 52, "type": "uint32"},
    ],
}

ELFCLASS64 = 2
ELFDATA2LSB = 1
ET_DYN = 3
EM_AMDGPU = 224
SHT_PROGBITS = 1
SHT_STRTAB = 3
SHT_RELA = 4
SHT_NOBITS = 8
SHT_REL = 9
SHT_DYNSYM = 11
SHT_SYMTAB = 2
SHF_ALLOC = 0x2
SHF_TLS = 0x400
R_AMDGPU_REL64 = 5
DESCRIPTOR_SIZE = 64
KERNEL_CODE_PROPERTIES = 0x408
GATE_UP_EXPECTED_GROUP_SEGMENT_BYTES = 2048 * 2 + 4
MAX_ELF_BYTES = 8 * 1024 * 1024
MAX_ELF_SECTIONS = 1024
MAX_ELF_SYMBOLS = 1024
MAX_ELF_STRING_TABLE_BYTES = 64 * 1024
MAX_ELF_NAME_BYTES = 256
MAX_IMAGE_BYTES = 4 * 1024 * 1024
PM4_PROGRAM_ENTRY_ALIGNMENT = 256
CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/llama_embed_row_f16.cpp")
RMSNORM_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/llama_rmsnorm_f16.cpp")
RMSNORM_ZERO_STORE_CANONICAL_SOURCE_PATH = Path(
    "native_r9700/kernels/llama_rmsnorm_zero_store_f16.cpp"
)
RMSNORM_EPSILON_ARITHMETIC_CANONICAL_SOURCE_PATH = Path(
    "native_r9700/kernels/llama_rmsnorm_epsilon_arithmetic_f16.cpp"
)


K_PROJECTION_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/llama_k_projection_f16.cpp")
V_PROJECTION_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/llama_v_projection_f16.cpp")
QWEN_AFFINE4_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/qwen_affine4_linear.cpp")
ROPE_KV_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/llama_rope_kv_f16.cpp")
ATTENTION_SCORE_CANONICAL_SOURCE_PATH = Path(
    "native_r9700/kernels/llama_causal_attention_score_f16.cpp"
)
ATTENTION_SOFTMAX_CANONICAL_SOURCE_PATH = Path(
    "native_r9700/kernels/llama_causal_attention_softmax_f32.cpp"
)
ATTENTION_CONTEXT_CANONICAL_SOURCE_PATH = Path(
    "native_r9700/kernels/llama_causal_attention_context_f16.cpp"
)
O_PROJECTION_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/llama_o_projection_f16.cpp")
GATED_MLP_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/llama_gated_mlp_f16.cpp")
GATE_UP_PROJECTION_CANONICAL_SOURCE_PATH = Path(
    "native_r9700/kernels/llama_gate_up_projection_f16.cpp"
)
MLP_DOWN_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/llama_mlp_down_f16.cpp")
QWEN_DELTANET_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/qwen_deltanet_state.cpp")
QWEN_FULL_ATTENTION_CANONICAL_SOURCE_PATH = Path("native_r9700/kernels/qwen_full_attention.cpp")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE = REPOSITORY_ROOT / CANONICAL_SOURCE_PATH
RMSNORM_CANONICAL_SOURCE = REPOSITORY_ROOT / RMSNORM_CANONICAL_SOURCE_PATH
RMSNORM_ZERO_STORE_CANONICAL_SOURCE = (
    REPOSITORY_ROOT / RMSNORM_ZERO_STORE_CANONICAL_SOURCE_PATH
)
RMSNORM_EPSILON_ARITHMETIC_CANONICAL_SOURCE = (
    REPOSITORY_ROOT / RMSNORM_EPSILON_ARITHMETIC_CANONICAL_SOURCE_PATH
)

K_PROJECTION_CANONICAL_SOURCE = REPOSITORY_ROOT / K_PROJECTION_CANONICAL_SOURCE_PATH
V_PROJECTION_CANONICAL_SOURCE = REPOSITORY_ROOT / V_PROJECTION_CANONICAL_SOURCE_PATH
QWEN_AFFINE4_CANONICAL_SOURCE = REPOSITORY_ROOT / QWEN_AFFINE4_CANONICAL_SOURCE_PATH
ROPE_KV_CANONICAL_SOURCE = REPOSITORY_ROOT / ROPE_KV_CANONICAL_SOURCE_PATH
ATTENTION_SCORE_CANONICAL_SOURCE = REPOSITORY_ROOT / ATTENTION_SCORE_CANONICAL_SOURCE_PATH
ATTENTION_SOFTMAX_CANONICAL_SOURCE = REPOSITORY_ROOT / ATTENTION_SOFTMAX_CANONICAL_SOURCE_PATH
ATTENTION_CONTEXT_CANONICAL_SOURCE = REPOSITORY_ROOT / ATTENTION_CONTEXT_CANONICAL_SOURCE_PATH
O_PROJECTION_CANONICAL_SOURCE = REPOSITORY_ROOT / O_PROJECTION_CANONICAL_SOURCE_PATH
GATED_MLP_CANONICAL_SOURCE = REPOSITORY_ROOT / GATED_MLP_CANONICAL_SOURCE_PATH
GATE_UP_PROJECTION_CANONICAL_SOURCE = REPOSITORY_ROOT / GATE_UP_PROJECTION_CANONICAL_SOURCE_PATH
MLP_DOWN_CANONICAL_SOURCE = REPOSITORY_ROOT / MLP_DOWN_CANONICAL_SOURCE_PATH
QWEN_DELTANET_CANONICAL_SOURCE = REPOSITORY_ROOT / QWEN_DELTANET_CANONICAL_SOURCE_PATH
QWEN_FULL_ATTENTION_CANONICAL_SOURCE = REPOSITORY_ROOT / QWEN_FULL_ATTENTION_CANONICAL_SOURCE_PATH
REVIEWED_ASSETS = (
    (
        CANONICAL_SOURCE_PATH,
        CANONICAL_SOURCE,
        KERNEL_NAME,
        KERNARG_SCHEMA,
        ("embedding_rows", "hidden_output", "selected_row"),
        (),
        24,
    ),
    (
        RMSNORM_CANONICAL_SOURCE_PATH,
        RMSNORM_CANONICAL_SOURCE,
        RMSNORM_KERNEL_NAME,
        RMSNORM_KERNARG_SCHEMA,
        ("hidden_input", "scale", "hidden_output"),
        (("epsilon", "float"),),
        28,
    ),
    (
        RMSNORM_ZERO_STORE_CANONICAL_SOURCE_PATH,
        RMSNORM_ZERO_STORE_CANONICAL_SOURCE,
        RMSNORM_ZERO_STORE_KERNEL_NAME,
        RMSNORM_ZERO_STORE_KERNARG_SCHEMA,
        ("hidden_input", "scale", "hidden_output"),
        (("epsilon", "float"),),
        28,
    ),
    (
        RMSNORM_EPSILON_ARITHMETIC_CANONICAL_SOURCE_PATH,
        RMSNORM_EPSILON_ARITHMETIC_CANONICAL_SOURCE,
        RMSNORM_EPSILON_ARITHMETIC_KERNEL_NAME,
        RMSNORM_EPSILON_ARITHMETIC_KERNARG_SCHEMA,
        ("hidden_input", "scale", "hidden_output"),
        (("epsilon", "float"),),
        28,
    ),
    (
        K_PROJECTION_CANONICAL_SOURCE_PATH,
        K_PROJECTION_CANONICAL_SOURCE,
        K_PROJECTION_KERNEL_NAME,
        K_PROJECTION_KERNARG_SCHEMA,
        ("normalized", "k_projection_weight", "fresh_k"),
        (("sequence_length", "unsigned int"),),
        28,
    ),
    (
        V_PROJECTION_CANONICAL_SOURCE_PATH,
        V_PROJECTION_CANONICAL_SOURCE,
        V_PROJECTION_KERNEL_NAME,
        V_PROJECTION_KERNARG_SCHEMA,
        ("normalized", "v_projection_weight", "fresh_v"),
        (("sequence_length", "unsigned int"),),
        28,
    ),
    (
        QWEN_AFFINE4_CANONICAL_SOURCE_PATH,
        QWEN_AFFINE4_CANONICAL_SOURCE,
        QWEN_AFFINE4_KERNEL_NAME,
        QWEN_AFFINE4_KERNARG_SCHEMA,
        ("input", "packed_weight", "scales", "biases", "output"),
        (
            ("input_features", "unsigned long long"),

            ("output_features", "unsigned long long"),
            ("input_capacity_elements", "unsigned long long"),
            ("packed_weight_capacity_bytes", "unsigned long long"),
            ("affine_group_capacity", "unsigned long long"),
            ("output_capacity_elements", "unsigned long long"),
        ),
        88,
    ),
    (
        ROPE_KV_CANONICAL_SOURCE_PATH, ROPE_KV_CANONICAL_SOURCE, ROPE_KV_KERNEL_NAME,
        ROPE_KV_KERNARG_SCHEMA, ("fresh_k", "fresh_v", "k_cache", "v_cache"),
        (("sequence_length", "unsigned int"), ("position", "unsigned int"),
         ("cache_capacity_tokens", "unsigned int")), 44,
    ),
    (
        ATTENTION_SCORE_CANONICAL_SOURCE_PATH, ATTENTION_SCORE_CANONICAL_SOURCE,
        ATTENTION_SCORE_KERNEL_NAME, ATTENTION_SCORE_KERNARG_SCHEMA,
        ("normalized", "q_projection_weight", "k_cache", "attention_scores"),
        (("sequence_length", "unsigned int"), ("position", "unsigned int"),
         ("cache_capacity_tokens", "unsigned int")), 44,
    ),
    (
        ATTENTION_SOFTMAX_CANONICAL_SOURCE_PATH, ATTENTION_SOFTMAX_CANONICAL_SOURCE,
        ATTENTION_SOFTMAX_KERNEL_NAME, ATTENTION_SOFTMAX_KERNARG_SCHEMA,
        ("attention_scores", "attention_probabilities"),
        (("sequence_length", "unsigned int"), ("position", "unsigned int"),
         ("cache_capacity_tokens", "unsigned int")), 28,
    ),
    (
        ATTENTION_CONTEXT_CANONICAL_SOURCE_PATH, ATTENTION_CONTEXT_CANONICAL_SOURCE,
        ATTENTION_CONTEXT_KERNEL_NAME, ATTENTION_CONTEXT_KERNARG_SCHEMA,
        ("attention_probabilities", "v_cache", "context"),
        (("sequence_length", "unsigned int"), ("position", "unsigned int"),
         ("cache_capacity_tokens", "unsigned int")), 36,
    ),
    (
        O_PROJECTION_CANONICAL_SOURCE_PATH, O_PROJECTION_CANONICAL_SOURCE,
        O_PROJECTION_KERNEL_NAME, O_PROJECTION_KERNARG_SCHEMA,
        ("context", "o_projection_weight", "residual", "post_attention_hidden"),
        (("sequence_length", "unsigned int"),), 36,
    ),
    (
        GATED_MLP_CANONICAL_SOURCE_PATH, GATED_MLP_CANONICAL_SOURCE,
        GATED_MLP_KERNEL_NAME, GATED_MLP_KERNARG_SCHEMA,
        ("post_attention_hidden", "post_attention_layernorm_weight",
         "gate_projection_weight", "up_projection_weight", "down_projection_weight", "hidden"),
        (("sequence_length", "unsigned int"),), 52,
    ),
    (
        GATE_UP_PROJECTION_CANONICAL_SOURCE_PATH, GATE_UP_PROJECTION_CANONICAL_SOURCE,
        GATE_UP_PROJECTION_KERNEL_NAME, GATE_UP_PROJECTION_KERNARG_SCHEMA,
        ("post_attention_hidden", "post_attention_layernorm_weight",
         "gate_projection_weight", "up_projection_weight", "gate_output", "up_output"),
        (("sequence_length", "unsigned int"),), 52,
    ),
    (
        MLP_DOWN_CANONICAL_SOURCE_PATH, MLP_DOWN_CANONICAL_SOURCE,
        MLP_DOWN_KERNEL_NAME, MLP_DOWN_KERNARG_SCHEMA,
        ("gate_input", "up_input", "down_projection_weight", "residual", "hidden"),
        (("sequence_length", "unsigned int"),), 44,
    ),
    (
        QWEN_DELTANET_CANONICAL_SOURCE_PATH, QWEN_DELTANET_CANONICAL_SOURCE,
        QWEN_DELTANET_KERNEL_NAME, QWEN_DELTANET_KERNARG_SCHEMA,
        ("q", "k", "v", "decay", "beta", "state", "output"),
        (("value_heads", "unsigned int"), ("key_heads", "unsigned int"),
         ("key_dimension", "unsigned int"), ("value_dimension", "unsigned int"),
         ("state_capacity_elements", "unsigned long long")), 80,
    ),
    (
        QWEN_FULL_ATTENTION_CANONICAL_SOURCE_PATH, QWEN_FULL_ATTENTION_CANONICAL_SOURCE,
        QWEN_FULL_ATTENTION_KERNEL_NAME, QWEN_FULL_ATTENTION_KERNARG_SCHEMA,
        ("query", "k_cache", "v_cache", "output"),
        (("query_heads", "unsigned int"), ("kv_heads", "unsigned int"),
         ("head_dimension", "unsigned int"), ("query_length", "unsigned int"),
         ("position", "unsigned int"), ("cache_capacity_tokens", "unsigned int")), 56,
    ),
)
BSS_SENTINEL_NAME = ".bss"
BSS_SENTINEL_SIZE = 1
HIP_CUID_SYMBOL = re.compile(r"__hip_cuid_[0-9a-f]+")

RENAME_EXCL = 0x00000004
_RENAMEATX_NP = ctypes.CDLL(None, use_errno=True).renameatx_np
_RENAMEATX_NP.argtypes = (
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_uint,
)
_RENAMEATX_NP.restype = ctypes.c_int


# These are the only loadable sections emitted by the V1 direct-COMGR profile.
ADMITTED_ALLOCATED_SECTIONS = frozenset(
    {
        ".note",
        ".dynsym",
        ".gnu.hash",
        ".hash",
        ".dynstr",
        ".rodata",
        ".text",
        ".dynamic",
        ".relro_padding",
        BSS_SENTINEL_NAME,
    }
)


class GenerationError(RuntimeError):
    """The fresh source or COMGR object cannot form a V1 HSA image."""


@dataclass(frozen=True)
class ElfSection:
    """A range-validated ELF64 section header and its payload."""

    index: int
    name: str
    section_type: int
    flags: int
    address: int
    file_offset: int
    size: int
    content: bytes
    link: int
    info: int
    alignment: int
    entry_size: int


def _range_within(offset: int, size: int, limit: int, description: str) -> None:
    if offset < 0 or size < 0 or offset > limit or size > limit - offset:
        raise GenerationError(f"{description} exceeds the ELF file")


def _require_alignment(value: int, alignment: int, description: str) -> None:
    if alignment <= 0 or alignment & (alignment - 1) or value % alignment:
        raise GenerationError(f"{description} has invalid alignment")


def _validate_schema(schema: Any, expected_schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict) or schema != expected_schema:
        raise GenerationError("kernarg schema must exactly match the reviewed Llama asset ABI")
    return schema


def _source_without_comments(source_text: str) -> str:
    source = list(source_text)
    index = 0
    while index < len(source):
        if source[index] in ("'", '"'):
            quote = source[index]
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                elif source[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
            continue
        if source[index:index + 2] == ["/", "/"]:
            end = source_text.find("\n", index + 2)
            end = len(source) if end < 0 else end
        elif source[index:index + 2] == ["/", "*"]:
            end = source_text.find("*/", index + 2)
            if end < 0:
                raise GenerationError("source has an unterminated block comment")
            end += 2
        else:
            index += 1
            continue
        for offset in range(index, end):
            if source[offset] not in "\r\n":
                source[offset] = " "
        index = end
    return "".join(source)


def _expected_group_segment_bytes(kernel_name: str) -> int:
    return (
        GATE_UP_EXPECTED_GROUP_SEGMENT_BYTES
        if kernel_name == GATE_UP_PROJECTION_KERNEL_NAME
        else 0
    )


def validate_source_profile(
    source_text: str,
    kernel_name: str = KERNEL_NAME,
    pointer_parameters: tuple[str, ...] = (
        "embedding_rows",
        "hidden_output",
        "selected_row",
    ),
    scalar_parameters: tuple[tuple[str, str], ...] = (),
    *,
    expected_group_segment_bytes: int = 0,
) -> None:
    """Require one fresh, freestanding reviewed HIP kernel source."""
    source = _source_without_comments(source_text)
    if re.search(r"^[ \t]*(?:#|%:|\?\?=)", source, flags=re.MULTILINE):
        raise GenerationError("source profile forbids every preprocessor directive")
    lower = source.lower()
    forbidden = {
        "archive": "archived content",
        "fixture": "fixture content",
        "c0": "legacy C0 content",
        ".incbin": "embedded binary content",
        "hiplaunch": "HIP host launch API",
        "hipmalloc": "HIP allocation API",
        "hipfree": "HIP allocation API",
        "hipmemcpy": "HIP copy API",
        "__shared__": "shared storage",
        "__constant__": "constant storage",
        "main(": "host entry point",
    }
    for marker, description in forbidden.items():
        if marker in lower:
            raise GenerationError(f"source profile forbids {description}: {marker}")
    uses_shared_attribute = re.search(
        r"__attribute__\s*\(\(\s*shared\s*\)\)", source
    ) is not None
    if uses_shared_attribute and expected_group_segment_bytes == 0:
        raise GenerationError("source profile forbids shared storage without explicit LDS admission")
    if expected_group_segment_bytes < 0:
        raise GenerationError("expected group segment bytes must be nonnegative")
    signature = re.findall(
        rf'extern\s+"C"\s+__attribute__\s*\(\(\s*global\s*\)\)\s+void\s+'
        rf'{kernel_name}\s*\(([^)]*)\)',
        source,
        flags=re.MULTILINE,
    )
    if len(signature) != 1 or source.count('extern "C"') != 1:
        raise GenerationError("source must expose exactly one C-linkage reviewed kernel")
    parameters = [parameter.strip() for parameter in signature[0].split(",")]
    expected_count = len(pointer_parameters) + len(scalar_parameters)
    if len(parameters) != expected_count:
        raise GenerationError("kernel ABI has an unexpected parameter count")
    for parameter, name in zip(
        parameters[:len(pointer_parameters)], pointer_parameters, strict=True
    ):
        if "*" not in parameter or not re.search(rf"\b{re.escape(name)}\b", parameter):
            raise GenerationError(f"kernel ABI is missing pointer argument {name!r}")
    for parameter, (name, type_name) in zip(
        parameters[len(pointer_parameters):], scalar_parameters, strict=True
    ):
        if (
            "*" in parameter
            or not re.search(rf"\b{re.escape(name)}\b", parameter)
            or not re.search(rf"\b{re.escape(type_name)}\b", parameter)
        ):
            raise GenerationError(f"kernel ABI is missing {type_name} scalar argument {name!r}")


def _load_direct_comgr(tinygrad_root: Path) -> Callable[..., bytes]:
    """Import Tinygrad's proven direct-COMGR compiler without loading a device."""
    if not tinygrad_root.is_dir() or not (tinygrad_root / "tinygrad").is_dir():
        raise GenerationError(f"--tinygrad-root is not a Tinygrad checkout: {tinygrad_root}")
    sys.path.insert(0, str(tinygrad_root.resolve()))
    try:
        from tinygrad.runtime.support.compiler_amd import compile_hip
    except Exception as exc:
        raise GenerationError(
            f"cannot load Tinygrad direct COMGR compiler from {tinygrad_root}: {exc}"
        ) from exc
    finally:
        del sys.path[0]
    return compile_hip


def _parse_elf(hsaco: bytes) -> list[ElfSection]:
    if len(hsaco) > MAX_ELF_BYTES:
        raise GenerationError(
            f"COMGR ELF exceeds {MAX_ELF_BYTES}-byte raw-file limit"
        )
    if len(hsaco) < 64 or hsaco[:4] != b"\x7fELF":
        raise GenerationError("direct COMGR did not return an ELF HSACO")
    if hsaco[4] != ELFCLASS64 or hsaco[5] != ELFDATA2LSB:
        raise GenerationError("COMGR HSACO is not a little-endian ELF64 file")
    try:
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", hsaco)
    except struct.error as exc:
        raise GenerationError(f"cannot parse COMGR ELF header: {exc}") from exc
    if header[1] != ET_DYN or header[2] != EM_AMDGPU or header[3] != 1:
        raise GenerationError("COMGR ELF must be ELF64 LE ET_DYN EM_AMDGPU")
    section_offset, elf_header_size, section_entry_size, section_count, string_index = (
        header[6], header[8], header[11], header[12], header[13]
    )
    if (
        elf_header_size != 64
        or section_entry_size != 64
        or section_count == 0
        or string_index >= section_count
    ):
        raise GenerationError("COMGR ELF has an invalid section table")
    if section_count > MAX_ELF_SECTIONS:
        raise GenerationError(
            f"COMGR ELF section count exceeds {MAX_ELF_SECTIONS}-section limit"
        )
    _range_within(
        section_offset,
        section_entry_size * section_count,
        len(hsaco),
        "COMGR ELF section table",
    )
    headers: list[tuple[int, ...]] = []
    for index in range(section_count):
        try:
            headers.append(
                struct.unpack_from(
                    "<IIQQQQIIQQ", hsaco, section_offset + index * section_entry_size
                )
            )
        except struct.error as exc:
            raise GenerationError(f"cannot parse COMGR ELF section {index}: {exc}") from exc

    payload_ranges: list[tuple[int, int, int]] = []
    for index, raw in enumerate(headers):
        section_type, file_offset, size = raw[1], raw[4], raw[5]
        if section_type == SHT_NOBITS:
            continue
        _range_within(file_offset, size, len(hsaco), f"COMGR ELF section {index}")
        if section_type == SHT_STRTAB and size > MAX_ELF_STRING_TABLE_BYTES:
            raise GenerationError(
                "COMGR ELF string table exceeds "
                f"{MAX_ELF_STRING_TABLE_BYTES}-byte limit"
            )
        if size:
            payload_ranges.append((file_offset, file_offset + size, index))
    payload_ranges.sort()
    payload_bytes = 0
    previous_end = 0
    for file_offset, end, index in payload_ranges:
        if file_offset < previous_end:
            raise GenerationError(
                f"COMGR ELF non-NOBITS sections overlap at section {index}"
            )
        payload_bytes += end - file_offset
        if payload_bytes > MAX_IMAGE_BYTES:
            raise GenerationError(
                f"COMGR ELF non-NOBITS payloads exceed {MAX_IMAGE_BYTES}-byte limit"
            )
        previous_end = end

    names_header = headers[string_index]
    if names_header[1] != SHT_STRTAB:
        raise GenerationError("COMGR ELF has an invalid section-name table")
    names = hsaco[names_header[4]:names_header[4] + names_header[5]]

    name_cache: dict[int, str] = {}

    def name_at(offset: int) -> str:
        cached = name_cache.get(offset)
        if cached is not None:
            return cached
        if offset >= len(names):
            raise GenerationError("COMGR ELF has an invalid section-name offset")
        end = names.find(b"\0", offset)
        if end < 0:
            raise GenerationError("COMGR ELF has an unterminated section name")
        if end - offset > MAX_ELF_NAME_BYTES:
            raise GenerationError("COMGR ELF section name exceeds byte limit")
        try:
            name = names[offset:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GenerationError("COMGR ELF has a non-UTF-8 section name") from exc
        name_cache[offset] = name
        return name

    sections: list[ElfSection] = []
    for index, raw in enumerate(headers):
        name_offset, section_type, flags, address, file_offset, size, link, info, alignment, entry_size = raw
        if alignment and alignment & (alignment - 1):
            raise GenerationError(f"COMGR ELF section {index} has invalid alignment")
        if section_type == SHT_NOBITS:
            content = b""
        else:
            content = hsaco[file_offset:file_offset + size]
        sections.append(ElfSection(index, name_at(name_offset), section_type, flags, address, file_offset, size, content, link, info, alignment, entry_size))
    return sections


def _admit_allocated_sections(sections: list[ElfSection]) -> list[ElfSection]:
    allocated = [section for section in sections if section.flags & SHF_ALLOC]
    for section in sections:
        if section.name == ".relro_padding" and section.section_type != SHT_NOBITS:
            raise GenerationError("COMGR ELF .relro_padding must be NOBITS")
    names: set[str] = set()
    for section in allocated:
        if section.name in names:
            raise GenerationError(f"COMGR ELF duplicates allocated section {section.name!r}")
        names.add(section.name)
        if section.name == BSS_SENTINEL_NAME and (
            section.section_type != SHT_NOBITS or section.size != BSS_SENTINEL_SIZE
        ):
            raise GenerationError("COMGR ELF .bss must be the one-byte NOBITS sentinel")
        if section.section_type == SHT_NOBITS and section.name not in (
            ".relro_padding",
            BSS_SENTINEL_NAME,
        ):
            raise GenerationError(f"COMGR ELF allocated section {section.name!r} is NOBITS")
        if section.flags & SHF_TLS:
            raise GenerationError(f"COMGR ELF allocated section {section.name!r} is TLS")
        if section.name not in ADMITTED_ALLOCATED_SECTIONS:
            raise GenerationError(f"COMGR ELF contains unadmitted allocated section {section.name!r}")
        if section.size == 0:
            raise GenerationError(f"COMGR ELF allocated section {section.name!r} is empty")
        if section.alignment:
            _require_alignment(section.address, section.alignment, f"COMGR ELF allocated section {section.name!r}")
    text = [section for section in allocated if section.name == ".text"]
    rodata = [section for section in allocated if section.name == ".rodata"]
    if len(text) != 1 or len(rodata) != 1:
        raise GenerationError("COMGR ELF must have exactly one allocated .text and .rodata section")
    if text[0].section_type != SHT_PROGBITS or rodata[0].section_type != SHT_PROGBITS:
        raise GenerationError("COMGR ELF .text and .rodata must be PROGBITS")
    return allocated


def _image_layout(allocated: list[ElfSection]) -> tuple[bytearray, dict[int, int]]:
    end = 0
    ranges: list[tuple[int, int, ElfSection]] = []
    offsets: dict[int, int] = {}
    for section in allocated:
        offset = section.address
        if offset < 0 or section.size > (1 << 63) - offset:
            raise GenerationError(f"COMGR ELF allocated section {section.name!r} has an invalid address range")
        section_end = offset + section.size
        ranges.append((offset, section_end, section))
        offsets[section.index] = offset
        end = max(end, section_end)
    ordered_ranges = sorted(ranges)
    for (_, previous_end, previous), (next_start, _, following) in zip(
        ordered_ranges, ordered_ranges[1:]
    ):
        if next_start < previous_end:
            raise GenerationError(f"COMGR ELF allocated sections {previous.name!r} and {following.name!r} overlap")
    if end == 0:
        raise GenerationError("COMGR ELF image is empty")
    if end > MAX_IMAGE_BYTES:
        raise GenerationError(
            f"COMGR ELF image span {end} exceeds {MAX_IMAGE_BYTES}-byte limit"
        )
    image = bytearray(end)
    for section in allocated:
        if section.section_type == SHT_NOBITS:
            # bytearray() zero-fills this exact non-overlapping image range.
            continue
        start = offsets[section.index]
        image[start:start + section.size] = section.content
    return image, offsets


def _string_at(
    table: ElfSection, offset: int, description: str, cache: dict[int, str]
) -> str:
    cached = cache.get(offset)
    if cached is not None:
        return cached
    if offset >= len(table.content):
        raise GenerationError(f"{description} name offset exceeds its string table")
    end = table.content.find(b"\0", offset)
    if end < 0:
        raise GenerationError(f"{description} has an unterminated name")
    if end - offset > MAX_ELF_NAME_BYTES:
        raise GenerationError(f"{description} name exceeds byte limit")
    try:
        name = table.content[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenerationError(f"{description} has a non-UTF-8 name") from exc
    cache[offset] = name
    return name


def _is_compiler_marker_symbol(
    name: str, section_index: int, value: int, size: int, bss_sentinel: ElfSection | None
) -> bool:
    return (
        bss_sentinel is not None
        and section_index == bss_sentinel.index
        and bss_sentinel.name == BSS_SENTINEL_NAME
        and bss_sentinel.section_type == SHT_NOBITS
        and bss_sentinel.size == BSS_SENTINEL_SIZE
        and HIP_CUID_SYMBOL.fullmatch(name) is not None
        and value == bss_sentinel.address
        and size <= BSS_SENTINEL_SIZE
    )


def _symbol_tables(
    sections: list[ElfSection], bss_sentinel: ElfSection | None
) -> dict[int, list[tuple[str, int, int, int]]]:
    tables: dict[int, list[tuple[str, int, int, int]]] = {}
    name_caches: dict[int, dict[int, str]] = {}
    symbol_count = 0
    for table in sections:
        if table.section_type not in (SHT_SYMTAB, SHT_DYNSYM):
            continue
        if table.entry_size != 24 or table.size % table.entry_size:
            raise GenerationError("COMGR ELF has an invalid symbol table")
        entry_count = table.size // table.entry_size
        if entry_count > MAX_ELF_SYMBOLS - symbol_count:
            raise GenerationError(
                f"COMGR ELF symbol count exceeds {MAX_ELF_SYMBOLS}-entry limit"
            )
        symbol_count += entry_count
        if table.link >= len(sections) or sections[table.link].section_type != SHT_STRTAB:
            raise GenerationError("COMGR ELF symbol table has no valid string table")
        strings = sections[table.link]
        if strings.size > MAX_ELF_STRING_TABLE_BYTES:
            raise GenerationError(
                "COMGR ELF symbol string table exceeds "
                f"{MAX_ELF_STRING_TABLE_BYTES}-byte limit"
            )
        names = name_caches.setdefault(strings.index, {})
        records: list[tuple[str, int, int, int]] = []
        for offset in range(0, table.size, table.entry_size):
            name_offset, _info, _other, section_index, value, size = struct.unpack_from("<IBBHQQ", table.content, offset)
            if section_index >= len(sections) and section_index < 0xFF00:
                raise GenerationError("COMGR ELF symbol has an invalid section index")
            name = _string_at(strings, name_offset, "COMGR ELF symbol", names)
            if (
                bss_sentinel is not None
                and section_index == bss_sentinel.index
                and not _is_compiler_marker_symbol(
                    name, section_index, value, size, bss_sentinel
                )
            ):
                raise GenerationError("COMGR ELF symbol references the .bss sentinel")
            records.append((name, section_index, value, size))
        tables[table.index] = records
    return tables


def _kernel_symbol(
    sections: list[ElfSection],
    tables: dict[int, list[tuple[str, int, int, int]]],
    allocated: list[ElfSection],
    offsets: dict[int, int],
    kernel_name: str,
) -> tuple[int, int, int]:
    records = [record for table in tables.values() for record in table]
    by_index = {section.index: section for section in allocated}
    matches = [record for record in records if record[0] == kernel_name]
    targets = {(section_index, value) for _, section_index, value, _ in matches}
    if len(targets) != 1:
        raise GenerationError(
            f"expected exactly one ELF symbol target {kernel_name!r}, found {len(targets)}"
        )
    section_index, value = next(iter(targets))
    section = by_index.get(section_index)
    if section is None or section.name != ".text":
        raise GenerationError("kernel ELF symbol does not refer to allocated .text")
    if value < section.address or value >= section.address + section.size:
        raise GenerationError("kernel ELF symbol is outside .text")
    entry_offset = offsets[section.index] + value - section.address
    return entry_offset, len(matches), len(targets)


def _apply_relocations(
    image: bytearray,
    sections: list[ElfSection],
    allocated: list[ElfSection],
    offsets: dict[int, int],
    tables: dict[int, list[tuple[str, int, int, int]]],
    bss_sentinel_index: int | None,
) -> int:
    allocated_by_index = {section.index: section for section in allocated}
    sections_by_index = {section.index: section for section in sections}
    count = 0
    for relocation in sections:
        if relocation.section_type not in (SHT_REL, SHT_RELA):
            continue
        entry_size = 24 if relocation.section_type == SHT_RELA else 16
        if relocation.entry_size != entry_size or relocation.size % entry_size:
            raise GenerationError("COMGR ELF has an invalid relocation section")
        target = allocated_by_index.get(relocation.info)
        relocation_target = sections_by_index.get(relocation.info)
        if relocation_target is not None and relocation_target.name == ".relro_padding":
            raise GenerationError("COMGR ELF .relro_padding must not be a relocation target")
        if target is None:
            raise GenerationError("COMGR ELF relocation target is not an admitted allocated section")
        if target.index == bss_sentinel_index:
            raise GenerationError("COMGR ELF relocation references the .bss sentinel")
        symbols = tables.get(relocation.link)
        if symbols is None:
            raise GenerationError("COMGR ELF relocation section has no valid symbol table")
        for offset in range(0, relocation.size, entry_size):
            if relocation.section_type == SHT_RELA:
                place, info, addend = struct.unpack_from("<QQq", relocation.content, offset)
            else:
                place, info = struct.unpack_from("<QQ", relocation.content, offset)
                addend = None
            relocation_type = info & 0xFFFFFFFF
            symbol_index = info >> 32
            if relocation_type != R_AMDGPU_REL64:
                raise GenerationError(f"unsupported relocation type {relocation_type}; only REL64 is admitted")
            if symbol_index >= len(symbols):
                raise GenerationError("COMGR ELF relocation symbol index is invalid")
            _name, symbol_section_index, symbol_value, _symbol_size = symbols[symbol_index]
            if symbol_section_index == bss_sentinel_index:
                raise GenerationError("COMGR ELF relocation references the .bss sentinel")
            relocation_source = sections_by_index.get(symbol_section_index)
            if relocation_source is not None and relocation_source.name == ".relro_padding":
                raise GenerationError("COMGR ELF .relro_padding must not be a relocation source")
            symbol_section = allocated_by_index.get(symbol_section_index)
            if (
                symbol_section is None
                or symbol_value < symbol_section.address
                or symbol_value >= symbol_section.address + symbol_section.size
            ):
                raise GenerationError("COMGR ELF relocation has an unresolved symbol")
            if place < target.address or place > target.address + target.size - 8:
                raise GenerationError("COMGR ELF relocation place is outside its target section")
            image_offset = offsets[target.index] + place - target.address
            if addend is None:
                addend = struct.unpack_from("<q", image, image_offset)[0]
            value = symbol_value + addend - place
            if not -(1 << 63) <= value < (1 << 63):
                raise GenerationError("COMGR ELF REL64 relocation result overflows")
            struct.pack_into("<q", image, image_offset, value)
            count += 1
    return count


def _descriptor(
    image: bytearray,
    rodata: ElfSection,
    descriptor_offset: int,
    entry_offset: int,
    kernarg_schema: dict[str, Any],
    compiler_kernarg_bytes: int,
    *,
    expected_group_segment_bytes: int = 0,
) -> dict[str, int]:
    if rodata.size != DESCRIPTOR_SIZE:
        raise GenerationError(".rodata must contain exactly one 64-byte AMDHSA kernel descriptor")
    if descriptor_offset + DESCRIPTOR_SIZE > len(image):
        raise GenerationError("kernel descriptor exceeds the image")
    group, private, kernarg = struct.unpack_from("<IIQ", image, descriptor_offset)
    delta = struct.unpack_from("<q", image, descriptor_offset + 16)[0]
    descriptor_rsrc3 = struct.unpack_from("<I", image, descriptor_offset + 44)[0]
    descriptor_rsrc1 = struct.unpack_from("<I", image, descriptor_offset + 48)[0]
    descriptor_rsrc2 = struct.unpack_from("<I", image, descriptor_offset + 52)[0]
    properties, preload = struct.unpack_from("<HH", image, descriptor_offset + 56)
    if kernarg != compiler_kernarg_bytes:
        raise GenerationError("AMDHSA descriptor kernarg size disagrees with the reviewed ABI")
    if group != expected_group_segment_bytes:
        raise GenerationError(
            "AMDHSA descriptor group segment does not match the explicit LDS admission"
        )
    if private or preload:
        raise GenerationError("AMDHSA descriptor must not use private or preload storage")
    if properties != KERNEL_CODE_PROPERTIES:
        raise GenerationError("AMDHSA descriptor has unexpected kernel-code properties")
    if descriptor_offset + delta != entry_offset:
        raise GenerationError("AMDHSA descriptor entry delta disagrees with the kernel symbol")
    if any(
        value <= 0
        for value in (descriptor_rsrc1, descriptor_rsrc2, descriptor_rsrc3)
    ):
        raise GenerationError("AMDHSA descriptor resources must be positive")
    if kernarg != kernarg_schema["bytes"]:
        struct.pack_into("<I", image, descriptor_offset + 8, kernarg_schema["bytes"])
    # Match Tinygrad AMDProgram: PM4 needs the 512-byte LDS allocation count in
    # COMPUTE_PGM_RSRC2 even though COMGR leaves that field clear in the descriptor.
    lds_size = ((group + 511) // 512) & 0x1FF
    dispatch_rsrc2 = descriptor_rsrc2 | (lds_size << 15)
    return {
        "group_segment_bytes": group,
        "private_segment_bytes": private,
        "kernarg_bytes": kernarg_schema["bytes"],
        "kernel_code_properties": properties,
        "kernarg_preload_bytes": preload,
        "descriptor_rsrc1": descriptor_rsrc1,
        "descriptor_rsrc2": descriptor_rsrc2,
        "descriptor_rsrc3": descriptor_rsrc3,
        "rsrc1": descriptor_rsrc1,
        "rsrc2": dispatch_rsrc2,
        "rsrc3": descriptor_rsrc3,
    }


def _reviewed_asset(
    source: Path,
) -> tuple[Path, Path, str, dict[str, Any], tuple[str, ...], tuple[tuple[str, str], ...], int]:
    if source.is_symlink():
        raise GenerationError(f"--source must name a real HIP source file: {source}")
    for asset in REVIEWED_ASSETS:
        source_path, canonical_source, *_rest = asset
        if source in (source_path, canonical_source):
            return asset
    raise GenerationError("--source must name a checked-in reviewed Llama HIP source")

def _read_reviewed_source(source: Path, source_path: Path, canonical_source: Path) -> bytes:
    """Read only a lexically and physically reviewed HIP source file."""
    if source.is_symlink():
        raise GenerationError(f"--source must name a real HIP source file: {source}")
    if source not in (source_path, canonical_source):
        raise GenerationError(
            f"--source must name the checked-in reviewed HIP source: {source_path}"
        )
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as exc:
        raise GenerationError(f"--source must name a real HIP source file: {source}") from exc
    if resolved_source != canonical_source:
        raise GenerationError(
            f"--source must resolve to the checked-in reviewed HIP source: {source_path}"
        )
    try:
        source_status = source.stat()
    except OSError as exc:
        raise GenerationError(f"--source must name a real HIP source file: {source}") from exc
    if not stat.S_ISREG(source_status.st_mode):
        raise GenerationError(f"--source must name a real HIP source file: {source}")
    try:
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise GenerationError(f"cannot open checked-in V1 HIP source: {source}") from exc
    try:
        opened_status = os.fstat(source_fd)
        if not stat.S_ISREG(opened_status.st_mode) or not _same_inode(
            source_status, opened_status
        ):
            raise GenerationError("checked-in V1 HIP source changed while opening")
        chunks: list[bytes] = []
        while chunk := os.read(source_fd, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise GenerationError(f"cannot read checked-in V1 HIP source: {source}") from exc
    finally:
        os.close(source_fd)


def _write_all(file_descriptor: int, content: bytes) -> None:
    written = 0
    while written < len(content):
        try:
            count = os.write(file_descriptor, content[written:])
        except InterruptedError:
            continue
        if count <= 0:
            raise OSError("short write while publishing HSA image output")
        written += count


def _write_private_leaf(directory_fd: int, prefix: str, content: bytes) -> str:
    """Write one durable, no-replace leaf inside a private staging directory."""
    file_descriptor = os.open(
        prefix,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        _write_all(file_descriptor, content)
        os.fsync(file_descriptor)
    except BaseException:
        try:
            os.unlink(prefix, dir_fd=directory_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(file_descriptor)
    return prefix


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _cleanup_staging_output(
    parent_fd: int | None,
    staging_fd: int | None,
    staging_name: str,
    staging_identity: os.stat_result | None,
    leaves: tuple[str, ...],
) -> None:
    if parent_fd is None or staging_identity is None:
        return
    try:
        current = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        if not _same_inode(current, staging_identity):
            return
        if staging_fd is not None:
            descriptor_identity = os.fstat(staging_fd)
            if not _same_inode(descriptor_identity, staging_identity):
                return
            for leaf in leaves:
                try:
                    os.unlink(leaf, dir_fd=staging_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
        os.rmdir(staging_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError:
        pass


def _open_private_staging_directory(
    parent_fd: int, final_name: str
) -> tuple[str, int, os.stat_result]:
    for _ in range(32):
        staging_name = f".{final_name}.staging-{secrets.token_hex(16)}"
        try:
            os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        staging_fd: int | None = None
        staging_identity: os.stat_result | None = None
        try:
            staging_fd = os.open(
                staging_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            staging_identity = os.fstat(staging_fd)
            current = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(staging_identity.st_mode) or not _same_inode(
                current, staging_identity
            ):
                raise GenerationError("private HSA image staging directory was replaced")
            return staging_name, staging_fd, staging_identity
        except BaseException:
            _cleanup_staging_output(
                parent_fd, staging_fd, staging_name, staging_identity, ()
            )
            if staging_fd is not None:
                os.close(staging_fd)
            raise
    raise GenerationError("cannot reserve a private HSA image staging directory")


def _renameatx_np(
    source_name: str,
    destination_name: str,
    *,
    src_dir_fd: int,
    dst_dir_fd: int,
    flags: int,
) -> None:
    ctypes.set_errno(0)
    if _RENAMEATX_NP(
        src_dir_fd,
        os.fsencode(source_name),
        dst_dir_fd,
        os.fsencode(destination_name),
        flags,
    ):
        error_number = ctypes.get_errno()
        raise OSError(
            error_number, os.strerror(error_number), destination_name
        )


def _publish_output(
    out_dir: Path, image: bytes, metadata: dict[str, Any], kernel_name: str
) -> None:
    final_name = out_dir.name
    if final_name in ("", ".", ".."):
        raise GenerationError(f"output directory must name a direct child: {out_dir}")
    image_name = metadata["image_path"]
    manifest_name = f"{kernel_name}.json"
    parent_fd: int | None = None
    staging_fd: int | None = None
    final_fd: int | None = None
    staging_name = ""
    staging_identity: os.stat_result | None = None
    image_leaf = ""
    manifest_leaf = ""
    staging_published = False
    published = False
    try:
        parent_fd = os.open(
            out_dir.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        staging_name, staging_fd, staging_identity = _open_private_staging_directory(
            parent_fd, final_name
        )
        image_leaf = _write_private_leaf(staging_fd, image_name, image)
        manifest_leaf = _write_private_leaf(
            staging_fd,
            manifest_name,
            (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        os.fsync(staging_fd)
        _renameatx_np(
            staging_name,
            final_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            flags=RENAME_EXCL,
        )
        staging_published = True
        final_fd = os.open(
            final_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        final_identity = os.fstat(final_fd)
        if not stat.S_ISDIR(final_identity.st_mode):
            raise GenerationError("published HSA image output is not a directory")
        if not _same_inode(final_identity, staging_identity):
            raise GenerationError("published HSA image output was replaced")
        os.fsync(parent_fd)
        published = True
    except (GenerationError, OSError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError(f"cannot publish HSA image output: {exc}") from exc
    finally:
        if not published and not staging_published:
            _cleanup_staging_output(
                parent_fd,
                staging_fd,
                staging_name,
                staging_identity,
                (image_leaf, manifest_leaf),
            )
        if final_fd is not None:
            os.close(final_fd)
        if staging_fd is not None:
            os.close(staging_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def generate(
    source: Path, target: str, schema: Any, tinygrad_root: Path, out_dir: Path
) -> dict[str, Any]:
    if target != TARGET:
        raise GenerationError(f"target must be {TARGET!r}, not {target!r}")
    (
        source_path,
        canonical_source,
        kernel_name,
        kernarg_schema,
        pointer_parameters,
        scalar_parameters,
        compiler_kernarg_bytes,
    ) = _reviewed_asset(source)
    _validate_schema(schema, kernarg_schema)
    source_bytes = _read_reviewed_source(source, source_path, canonical_source)
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenerationError("checked-in reviewed HIP source is not UTF-8") from exc
    expected_group_segment_bytes = _expected_group_segment_bytes(kernel_name)
    validate_source_profile(
        source_text,
        kernel_name,
        pointer_parameters,
        scalar_parameters,
        expected_group_segment_bytes=expected_group_segment_bytes,
    )

    compile_hip = _load_direct_comgr(tinygrad_root)
    try:
        hsaco = compile_hip(source_text, target, asm=False)
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(f"direct COMGR HIP compilation failed: {exc}") from exc
    sections = _parse_elf(hsaco)
    allocated = _admit_allocated_sections(sections)
    image, offsets = _image_layout(allocated)
    bss_sentinel = next(
        (section for section in allocated if section.name == BSS_SENTINEL_NAME), None
    )
    bss_sentinel_index = None if bss_sentinel is None else bss_sentinel.index
    tables = _symbol_tables(sections, bss_sentinel)
    relocation_count = _apply_relocations(
        image, sections, allocated, offsets, tables, bss_sentinel_index
    )
    entry_offset, symbol_record_count, symbol_target_count = _kernel_symbol(
        sections, tables, allocated, offsets, kernel_name
    )
    text = next(section for section in allocated if section.name == ".text")
    rodata = next(section for section in allocated if section.name == ".rodata")
    descriptor_offset = offsets[rodata.index]
    if not offsets[text.index] <= entry_offset < offsets[text.index] + text.size:
        raise GenerationError("kernel entry is outside the image .text range")
    _require_alignment(entry_offset, PM4_PROGRAM_ENTRY_ALIGNMENT, "kernel entry")
    resources = _descriptor(
        image,
        rodata,
        descriptor_offset,
        entry_offset,
        kernarg_schema,
        compiler_kernarg_bytes,
        expected_group_segment_bytes=expected_group_segment_bytes,
    )

    layout = [
        {"name": section.name, "address": section.address, "image_offset": offsets[section.index], "size": section.size}
        for section in allocated
    ]
    metadata: dict[str, Any] = {
        "name": kernel_name,
        "target": TARGET,
        "kernarg_schema": kernarg_schema,
        "image_path": f"{kernel_name}.image",
        "source_path": source_path.as_posix(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "image_size": len(image),
        "descriptor_offset": descriptor_offset,
        "entry_offset": entry_offset,
        **resources,
        "elf_admission": {
            "section_count": len(allocated),
            "symbol_record_count": symbol_record_count,
            "symbol_target_count": symbol_target_count,
            "relocation_count": relocation_count,
            "admitted_allocated_sections": [section.name for section in allocated],
        },
        "image_layout": layout,
    }

    _publish_output(out_dir, image, metadata, kernel_name)

def _schema_argument(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--schema must be JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise argparse.ArgumentTypeError("--schema must decode to a JSON object")
    return decoded


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--schema", required=True, type=_schema_argument)
    parser.add_argument("--tinygrad-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        generate(
            arguments.source,
            arguments.target,
            arguments.schema,
            arguments.tinygrad_root,
            arguments.out_dir,
        )
    except (GenerationError, OSError, UnicodeError) as exc:
        print(f"generation failed: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
