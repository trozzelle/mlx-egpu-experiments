"""RED contracts for the selected Qwen3.8-27B affine-4bit text adapter.

These contracts inspect only the selected snapshot's metadata sidecars and
compile the standalone raw-byte binder. They do not load safetensor payloads,
generate fixtures, or select any device/hardware path.
"""

import subprocess
from pathlib import Path

import pytest

from native_r9700.config import Llama32Config
from native_r9700.qwen_text_adapter import (
    CANONICAL_QWEN_TEXT_SNAPSHOT,
    QwenTextConfig,
    QwenTextConfigError,
    QwenTextIndexError,
    QwenTextSpecialTokenError,
    load_qwen_text_adapter,
)


CANONICAL_SNAPSHOT = (
    "${HOME}/Development/ml/models/hub/"
    "models--mlx-community--Qwen3.8-27B-4bit/snapshots/"
    "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
)
QWEN_BINDER_HEADER = Path("native_r9700/qwen_weight_binder.h")
QWEN_BINDER_SOURCE = Path("native_r9700/qwen_weight_binder.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


def test_binder_rejects_cross_file_and_overlapping_affine_raw_spans_before_device_allocation(
    tmp_path: Path,
) -> None:
    """Caller-owned raw metadata must name one file and disjoint bounded spans."""
    assert QWEN_BINDER_HEADER.is_file() and QWEN_BINDER_SOURCE.is_file()
    probe = tmp_path / "qwen_weight_binder_probe.cpp"
    probe.write_text(
        r'''
#include <string>

#include "qwen_weight_binder.h"

int main() {
  native_r9700::QwenAffineBinding binding;
  binding.layer_index = 0;
  binding.mode = "affine";
  binding.bits = 4;
  binding.group_size = 64;
  binding.window_size_bytes = 96;
  binding.weight = {"language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
                    "model-00001.safetensors", 0, 32};
  binding.scales = {"language_model.model.layers.0.linear_attn.in_proj_qkv.scales",
                    "model-00001.safetensors", 32, 32};
  binding.biases = {"language_model.model.layers.0.linear_attn.in_proj_qkv.biases",
                    "model-00001.safetensors", 64, 32};

  native_r9700::QwenWeightBinder binder;
  std::string error;
  if (!binder.validate(binding, &error)) return 1;

  binding.scales.source_file = "model-00002.safetensors";
  if (binder.validate(binding, &error)) return 2;
  if (error.find("source file") == std::string::npos) return 3;

  binding.scales.source_file = "model-00001.safetensors";
  binding.scales.offset_bytes = 16;
  if (binder.validate(binding, &error)) return 4;
  return error.find("overlap") == std::string::npos ? 5 : 0;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "qwen_weight_binder_probe"
    completed = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(QWEN_BINDER_SOURCE),
            str(probe),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    completed = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def load_selected_adapter():
    return load_qwen_text_adapter(CANONICAL_QWEN_TEXT_SNAPSHOT)


def test_selected_snapshot_path_is_the_canonical_qwen_text_source() -> None:
    """The adapter is pinned to the reviewed local Qwen snapshot, not a fallback."""
    assert CANONICAL_QWEN_TEXT_SNAPSHOT == CANONICAL_SNAPSHOT


def test_adapter_reads_qwen_geometry_from_nested_text_config() -> None:
    """Qwen text geometry comes from ``text_config``, never the VLM top level."""
    adapter = load_selected_adapter()

    assert isinstance(adapter.text_config, QwenTextConfig)
    assert not isinstance(adapter.text_config, Llama32Config)
    assert adapter.text_config.model_type == "qwen3_5_text"
    assert adapter.text_config.num_hidden_layers == 64
    assert adapter.text_config.hidden_size == 5120
    assert adapter.text_config.intermediate_size == 17408
    assert adapter.text_config.num_attention_heads == 24
    assert adapter.text_config.num_key_value_heads == 4
    assert adapter.text_config.head_dim == 256
    assert adapter.text_config.full_attention_interval == 4

def test_adapter_requires_selected_affine_4bit_quantization_metadata() -> None:
    """The Qwen adapter cannot reinterpret this snapshot as an fp16 Llama model."""
    adapter = load_selected_adapter()

    assert adapter.quantization.mode == "affine"
    assert adapter.quantization.bits == 4
    assert adapter.quantization.group_size == 64


def test_adapter_preserves_affine_weight_scales_and_biases_names() -> None:
    """Each selected quantized tensor retains its MLX affine triplet names."""
    adapter = load_selected_adapter()

    for stem in (
        "language_model.model.layers.0.linear_attn.in_proj_qkv",
        "language_model.model.layers.3.self_attn.q_proj",
    ):
        tensor = adapter.affine_tensors[stem]
        assert tensor.weight_name == f"{stem}.weight"
        assert tensor.scales_name == f"{stem}.scales"
        assert tensor.biases_name == f"{stem}.biases"


def test_adapter_rejects_multimodal_special_token_ids_in_text_only_mode() -> None:
    """Image, video, and vision-control tokens cannot enter a text-only adapter."""
    adapter = load_selected_adapter()

    for token_id in (248053, 248054, 248056, 248057):
        with pytest.raises(QwenTextSpecialTokenError, match="text-only"):
            adapter.validate_text_token_ids((248044, token_id))


def test_adapter_does_not_fall_back_to_llama_config_or_weight_names() -> None:
    """Qwen metadata uses its own parser and ``language_model`` tensor namespace."""
    adapter = load_selected_adapter()

    assert adapter.text_config.model_type == "qwen3_5_text"
    assert all(name.startswith("language_model.") for name in adapter.affine_tensors)
    assert "model.layers.0.self_attn.q_proj.weight" not in adapter.weight_index


def test_adapter_normalizes_invalid_utf8_config_to_config_error(tmp_path: Path) -> None:
    """A malformed config sidecar cannot leak the decoder implementation."""
    (tmp_path / "config.json").write_bytes(b"\x80")

    with pytest.raises(QwenTextConfigError):
        load_qwen_text_adapter(tmp_path)


def test_adapter_normalizes_invalid_utf8_index_to_index_error(tmp_path: Path) -> None:
    """A malformed index sidecar cannot leak the decoder implementation."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_bytes(b"\x80")

    with pytest.raises(QwenTextIndexError):
        load_qwen_text_adapter(tmp_path)


def test_adapter_rejects_multimodal_tokens_before_invoking_device_bound_binder() -> None:
    """Text-only validation gates the device binder, not merely later model work."""
    adapter = load_selected_adapter()
    binder_calls = 0

    def device_bound_binder() -> None:
        nonlocal binder_calls
        binder_calls += 1

    def prepare_text_device_binding(token_ids: tuple[int, ...]) -> None:
        adapter.validate_text_token_ids(token_ids)
        device_bound_binder()

    with pytest.raises(QwenTextSpecialTokenError, match="text-only"):
        prepare_text_device_binding((248044, 248056))

    assert binder_calls == 0
