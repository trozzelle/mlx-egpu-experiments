"""No-hardware contracts for the narrow Llama fp16 safetensors weight binder."""

from pathlib import Path
import json
import struct
import subprocess


BINDER_HEADER = Path("native_r9700/model_weight_binder.h")
BINDER_SOURCE = Path("native_r9700/model_weight_binder.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


_NAMES_AND_SHAPES = (
    ("model.embed_tokens.weight", [8, 16]),
    ("model.layers.0.input_layernorm.weight", [16]),
    ("model.layers.0.post_attention_layernorm.weight", [16]),
    ("model.layers.0.self_attn.q_proj.weight", [16, 16]),
    ("model.layers.0.self_attn.k_proj.weight", [8, 16]),
    ("model.layers.0.self_attn.v_proj.weight", [8, 16]),
    ("model.layers.0.self_attn.o_proj.weight", [16, 16]),
    ("model.layers.0.mlp.gate_proj.weight", [32, 16]),
    ("model.layers.0.mlp.up_proj.weight", [32, 16]),
    ("model.layers.0.mlp.down_proj.weight", [16, 32]),
)


def write_safetensors(
    path: Path,
    tensors: tuple[tuple[str, list[int]], ...],
    *,
    bad_dtype: str | None = None,
    bad_offsets_name: str | None = None,
    bad_shape_name: str | None = None,
    overlap_offsets_name: str | None = None,
) -> None:
    """Write a minimal fp16 safetensors file without relying on Python safetensors."""
    offset = 0
    header: dict[str, dict[str, object]] = {}
    for name, expected_shape in tensors:
        shape = [expected_shape[0] - 1, *expected_shape[1:]] if name == bad_shape_name else expected_shape
        byte_count = 2
        for dimension in shape:
            byte_count *= dimension
        end = offset + byte_count
        header[name] = {
            "dtype": bad_dtype if name == "model.layers.0.self_attn.k_proj.weight" and bad_dtype else "F16",
            "shape": shape,
            "data_offsets": (
                [0, byte_count]
                if name == overlap_offsets_name
                else [offset, end - 2 if name == bad_offsets_name else end]
            ),
        }
        offset = end
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + bytes(offset))


def write_index(model_dir: Path, tensors: tuple[tuple[str, list[int]], ...], *, missing: str | None = None) -> None:
    weight_map = {
        name: "model-00001-of-00002.safetensors" if index < 5 else "model-00002-of-00002.safetensors"
        for index, (name, _) in enumerate(tensors)
        if name != missing
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, separators=(",", ":")), encoding="utf-8"
    )


def write_valid_sharded_model(
    model_dir: Path,
    *,
    missing: str | None = None,
    bad_dtype: str | None = None,
    bad_offsets_name: str | None = None,
    bad_shape_name: str | None = None,
    overlap_offsets_name: str | None = None,
) -> None:
    model_dir.mkdir()
    first = _NAMES_AND_SHAPES[:5]
    second = _NAMES_AND_SHAPES[5:]
    write_safetensors(
        model_dir / "model-00001-of-00002.safetensors",
        first,
        bad_dtype=bad_dtype,
        bad_offsets_name=bad_offsets_name,
        bad_shape_name=bad_shape_name,
        overlap_offsets_name=overlap_offsets_name,
    )
    write_safetensors(
        model_dir / "model-00002-of-00002.safetensors",
        second,
        bad_dtype=bad_dtype,
        bad_offsets_name=bad_offsets_name,
        bad_shape_name=bad_shape_name,
        overlap_offsets_name=overlap_offsets_name,
    )
    write_index(model_dir, _NAMES_AND_SHAPES, missing=missing)


def compile_binder_probe(tmp_path: Path) -> Path:
    """Compile the public byte-span API against real on-disk safetensors files."""
    assert BINDER_HEADER.is_file() and BINDER_SOURCE.is_file(), "model-weight binder sources are missing"
    probe_source = tmp_path / "model_weight_binder_probe.cpp"
    probe_source.write_text(
        r'''
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "model_weight_binder.h"

namespace {

bool requires_text(const std::string& error, const char* text) {
  return error.find(text) != std::string::npos;
}

native_r9700::LlamaModelGeometry test_geometry() {
  native_r9700::LlamaModelGeometry geometry;
  geometry.hidden_size = 16;
  geometry.intermediate_size = 32;
  geometry.n_kv_heads = 2;
  geometry.head_dim = 4;
  geometry.vocab_size = 8;
  return geometry;
}

int valid(const std::filesystem::path& model_dir) {
  native_r9700::ModelWeightBinder binder;
  std::string error;
  if (!binder.open(model_dir.string(), &error)) return 1;

  native_r9700::LlamaLayer0WeightSpans weights;
  if (!binder.bind_llama_layer0(test_geometry(), &weights, &error)) return 2;
  const native_r9700::Fp16WeightSpan* spans[] = {
      &weights.embed_tokens,
      &weights.input_layernorm,
      &weights.post_attention_layernorm,
      &weights.q_proj,
      &weights.k_proj,
      &weights.v_proj,
      &weights.o_proj,
      &weights.gate_proj,
      &weights.up_proj,
      &weights.down_proj,
  };
  const char* names[] = {
      "model.embed_tokens.weight",
      "model.layers.0.input_layernorm.weight",
      "model.layers.0.post_attention_layernorm.weight",
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.0.self_attn.k_proj.weight",
      "model.layers.0.self_attn.v_proj.weight",
      "model.layers.0.self_attn.o_proj.weight",
      "model.layers.0.mlp.gate_proj.weight",
      "model.layers.0.mlp.up_proj.weight",
      "model.layers.0.mlp.down_proj.weight",
  };
  for (std::size_t index = 0; index < sizeof(spans) / sizeof(spans[0]); ++index) {
    if (spans[index]->name != names[index] || spans[index]->shard_path.empty() ||
        spans[index]->byte_length == 0 ||
        spans[index]->data_offset < spans[index]->payload_offset) {
      return 3;
    }
  }
  if (weights.embed_tokens.byte_length != 256) return 4;
  if (weights.k_proj.byte_length != 256 || weights.gate_proj.byte_length != 1024) return 5;
  if (weights.k_proj.data_offset < weights.k_proj.payload_offset) return 6;
  if (weights.k_proj.shape.size() != 2 || weights.k_proj.shape[0] != 8 ||
      weights.k_proj.shape[1] != 16) return 7;
  if (weights.k_proj.shard_path.filename() != "model-00001-of-00002.safetensors") return 8;
  if (weights.v_proj.shard_path.filename() != "model-00002-of-00002.safetensors") return 9;
  return 0;
}

int missing(const std::filesystem::path& model_dir) {
  native_r9700::ModelWeightBinder binder;
  std::string error;
  if (!binder.open(model_dir.string(), &error)) return 1;
  native_r9700::LlamaLayer0WeightSpans weights;
  if (binder.bind_llama_layer0(test_geometry(), &weights, &error)) return 2;
  return requires_text(error, "required Llama tensor missing: model.layers.0.self_attn.v_proj.weight") ? 0 : 3;
}

int dtype(const std::filesystem::path& model_dir) {
  native_r9700::ModelWeightBinder binder;
  std::string error;
  if (!binder.open(model_dir.string(), &error)) return 1;
  native_r9700::LlamaLayer0WeightSpans weights;
  if (binder.bind_llama_layer0(test_geometry(), &weights, &error)) return 2;
  return requires_text(error, "unsupported dtype F32 for tensor model.layers.0.self_attn.k_proj.weight; expected F16") ? 0 : 3;
}

int offsets(const std::filesystem::path& model_dir) {
  native_r9700::ModelWeightBinder binder;
  std::string error;
  if (!binder.open(model_dir.string(), &error)) return 1;
  native_r9700::LlamaLayer0WeightSpans weights;
  if (binder.bind_llama_layer0(test_geometry(), &weights, &error)) return 2;
  return requires_text(error, "fp16 byte span does not match shape for tensor model.layers.0.mlp.down_proj.weight") ? 0 : 3;
}

int shape(const std::filesystem::path& model_dir) {
  native_r9700::ModelWeightBinder binder;
  std::string error;
  if (!binder.open(model_dir.string(), &error)) return 1;
  native_r9700::LlamaLayer0WeightSpans weights;
  if (binder.bind_llama_layer0(test_geometry(), &weights, &error)) return 2;
  return requires_text(error, "unexpected shape for tensor model.layers.0.mlp.up_proj.weight; expected [32,16], got [31,16]") ? 0 : 3;
}

int overlap(const std::filesystem::path& model_dir) {
  native_r9700::ModelWeightBinder binder;
  std::string error;
  if (!binder.open(model_dir.string(), &error)) return 1;
  native_r9700::LlamaLayer0WeightSpans weights;
  if (binder.bind_llama_layer0(test_geometry(), &weights, &error)) return 2;
  return error.empty() ? 3 : 0;
}

int external_shard(const std::filesystem::path& model_dir) {
  native_r9700::ModelWeightBinder binder;
  std::string error;
  return binder.open(model_dir.string(), &error) ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 64;
  const std::string mode = argv[1];
  const std::filesystem::path model_dir(argv[2]);
  if (mode == "valid") return valid(model_dir);
  if (mode == "missing") return missing(model_dir);
  if (mode == "dtype") return dtype(model_dir);
  if (mode == "offsets") return offsets(model_dir);
  if (mode == "shape") return shape(model_dir);
  if (mode == "overlap") return overlap(model_dir);
  if (mode == "external_shard") return external_shard(model_dir);
  return 65;
}
'''.lstrip(),
        encoding="utf-8",
    )
    exe = tmp_path / "model_weight_binder_probe"
    completed = subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2", "-Wall", "-Wextra",
            str(BINDER_SOURCE), str(probe_source), "-I", str(NATIVE_INCLUDE_DIR), "-o", str(exe),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return exe


def run_binder_probe(exe: Path, mode: str, model_dir: Path) -> None:
    completed = subprocess.run([str(exe), mode, str(model_dir)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_binder_returns_real_sharded_fp16_layer0_byte_spans(tmp_path: Path) -> None:
    """The dispatcher receives safe file offsets, not host-loaded tensor values."""
    model_dir = tmp_path / "valid"
    write_valid_sharded_model(model_dir)
    run_binder_probe(compile_binder_probe(tmp_path), "valid", model_dir)


def test_binder_rejects_missing_required_llama_tensor_before_upload(tmp_path: Path) -> None:
    """An index cannot omit a layer-0 weight and still become an upload source."""
    model_dir = tmp_path / "missing"
    write_valid_sharded_model(model_dir, missing="model.layers.0.self_attn.v_proj.weight")
    run_binder_probe(compile_binder_probe(tmp_path), "missing", model_dir)


def test_binder_rejects_non_fp16_required_tensor_before_upload(tmp_path: Path) -> None:
    """F32 metadata cannot be reinterpreted as an fp16 model byte span."""
    model_dir = tmp_path / "dtype"
    write_valid_sharded_model(model_dir, bad_dtype="F32")
    run_binder_probe(compile_binder_probe(tmp_path), "dtype", model_dir)


def test_binder_rejects_offset_span_inconsistent_with_fp16_shape(tmp_path: Path) -> None:
    """Offset metadata must describe exactly the declared fp16 tensor shape."""
    model_dir = tmp_path / "offsets"
    write_valid_sharded_model(model_dir, bad_offsets_name="model.layers.0.mlp.down_proj.weight")
    run_binder_probe(compile_binder_probe(tmp_path), "offsets", model_dir)


def test_binder_rejects_shape_inconsistent_with_llama_geometry(tmp_path: Path) -> None:
    """An fp16 tensor with a valid span still cannot bypass Llama geometry validation."""
    model_dir = tmp_path / "shape"
    write_valid_sharded_model(model_dir, bad_shape_name="model.layers.0.mlp.up_proj.weight")
    run_binder_probe(compile_binder_probe(tmp_path), "shape", model_dir)


def test_binder_rejects_overlapping_same_shape_required_tensor_payloads_before_upload(tmp_path: Path) -> None:
    """Required tensors in one shard must not alias the same fp16 payload bytes."""
    model_dir = tmp_path / "overlap"
    write_valid_sharded_model(model_dir, overlap_offsets_name="model.layers.0.self_attn.k_proj.weight")
    run_binder_probe(compile_binder_probe(tmp_path), "overlap", model_dir)


def test_binder_rejects_indexed_shard_symlinked_outside_model_directory(tmp_path: Path) -> None:
    """An index must not make an external file a valid Llama weight upload source."""
    model_dir = tmp_path / "external-shard"
    write_valid_sharded_model(model_dir)
    external_shard = tmp_path / "external.safetensors"
    write_safetensors(external_shard, _NAMES_AND_SHAPES[:5])
    indexed_shard = model_dir / "model-00001-of-00002.safetensors"
    indexed_shard.unlink()
    indexed_shard.symlink_to(external_shard)

    run_binder_probe(compile_binder_probe(tmp_path), "external_shard", model_dir)
