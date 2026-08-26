"""Hardware-free RED contracts for the concrete native resource backend.

These contracts inspect the runner's concrete backend rather than opening a
TinyGPU device.  The worker seam already proves protocol state transitions;
this file binds the backend implementation to the persistent execution and
identity requirements that a fake backend cannot observe.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_DIR = REPO_ROOT / "native_r9700"
RUNNER_SOURCE = NATIVE_DIR / "runner.cpp"
WORKER_HEADER = NATIVE_DIR / "native_resource_worker.h"
WORKER_SOURCE = NATIVE_DIR / "native_resource_worker.cpp"
EXECUTOR_SOURCE = NATIVE_DIR / "llama_layer_executor.cpp"
SESSION_SOURCE = NATIVE_DIR / "amdev_session.cpp"
NPZ_SOURCE = NATIVE_DIR / "prefill_npz.cpp"
NPZ_HEADER = NATIVE_DIR / "prefill_npz.h"

LAYER_WEIGHT_FIELDS = (
    "input_layernorm",
    "post_attention_layernorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _read(path: Path) -> str:
    assert path.is_file(), f"required native source is missing: {path}"
    return path.read_text(encoding="utf-8")


def _class_source(class_name: str) -> str:
    source = _read(RUNNER_SOURCE)
    start = source.index(f"class {class_name}")
    end = source.index("\n};", start) + len("\n};")
    return source[start:end]


def _backend_source() -> str:
    return _class_source("RunnerNativeResourceBackend")


def _execution_source() -> str:
    return _class_source("NativePersistentExecution")


def _method_source(class_source: str, signature: str, next_signature: str) -> str:
    start = class_source.index(signature)
    end = class_source.index(next_signature, start + len(signature))
    return class_source[start:end]


def _prepare_result_serializer() -> str:
    source = _read(WORKER_SOURCE)
    match = re.search(
        r"std::string\s+prepare_result_json\s*\([^)]*\)\s*\{.*?\n\}",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, "native worker must serialize Prepare results"
    return match.group(0)


_NPZ_PROBE_SOURCE = r'''#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "prefill_npz.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  native_r9700::NativePrefillNpzPayload payload;
  payload.model = "zero-prefix";
  payload.n_prefix = 0;
  payload.cache_capacity_tokens = 128;
  const std::size_t cache_bytes =
      static_cast<std::size_t>(payload.cache_capacity_tokens) * 8U * 64U *
      sizeof(std::uint16_t);
  payload.kv_readback_bytes.assign(
      32U, std::vector<std::uint8_t>(cache_bytes, 0U));
  std::string error;
  if (!native_r9700::write_native_prefill_npz(
          payload, argv[1], &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  return 0;
}
'''


def _persistent_owner_name(class_source: str) -> str:
    private_start = class_source.rfind("private:")
    assert private_start >= 0, "concrete backend must have private persistent state"
    private = class_source[private_start:]
    owner = re.search(
        r"^\s*(?:(?:std::(?:unique_ptr|shared_ptr|optional)\s*<[^;\n]+>)|"
        r"(?:[A-Za-z_]\w*(?:Execution|Session|Handle|Owner|Resources|Context)))\s*[*&]?\s*"
        r"(?P<name>[A-Za-z_]\w*)\s*(?:=(?:[^;]|\n)*|\{(?:[^;]|\n)*\})?;",
        private,
        flags=re.MULTILINE,
    )
    assert owner is not None, (
        "concrete backend must own a persistent prepared/committed execution "
        "object, not only generation flags and synthetic allocations"
    )
    return owner.group("name")


def _compile_npz_probe(tmp_path: Path) -> Path:
    assert NPZ_HEADER.is_file(), f"required native source is missing: {NPZ_HEADER}"
    assert NPZ_SOURCE.is_file(), f"required native source is missing: {NPZ_SOURCE}"
    source = tmp_path / "native_prefill_npz_zero_probe.cpp"
    source.write_text(_NPZ_PROBE_SOURCE, encoding="utf-8")
    executable = tmp_path / "native_prefill_npz_zero_probe"
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
            str(source),
            str(NPZ_SOURCE),
            "-I",
            str(NATIVE_DIR),
            "-o",
            str(executable),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def _persistent_dispatch_source() -> str:
    source = _read(EXECUTOR_SOURCE)
    start = source.index("bool build_llama_persistent_dispatch(")
    end = source.index("\nbool validate_layer_execution_evidence(", start)
    return source[start:end]


def _session_close_source() -> str:
    source = _read(SESSION_SOURCE)
    start = source.index("bool ResidentHsaSession::close(")
    end = source.index("\n\nconst PhaseTimers& ResidentHsaSession::phase_timers(", start)
    return source[start:end]


def test_prepare_materializes_all_sixteen_layers_nine_weight_sets_once() -> None:
    source = _persistent_dispatch_source()
    weight_start = source.index("candidate.layer_weight_metadata = weights;")
    weight_end = source.index("  auto append_scratch", weight_start)
    weight_region = source[weight_start:weight_end]

    assert re.search(
        r"for\s*\([^)]*(?:weights\.layers|weights\.layers\.size\(\))",
        weight_region,
    ), "Prepare must iterate every bound layer when creating resident weights"
    assert "weights.layers.front()" not in weight_region, (
        "resident weight preparation must not alias every layer to layer zero"
    )
    for field in LAYER_WEIGHT_FIELDS:
        assert field in weight_region, (
            f"resident preparation must include the layer-local {field} weight span"
        )
    assert "allow_post_prepare_upload = true" not in weight_region, (
        "model weights must be resident after Prepare, not request-uploadable windows"
    )


def test_each_layer_dispatch_uses_its_own_resident_weight_indices() -> None:
    source = _persistent_dispatch_source()
    assert not re.search(
        r"layer_buffers\.assign\s*\(\s*kLlamaStageLayerCount\s*,\s*window_indices",
        source,
    ), "per-layer dispatch must not alias one layer's weight indices"
    assert re.search(
        r"layer_buffers\.(?:push_back|emplace_back)\s*\(",
        source,
    ), "persistent dispatch must retain one resident index tuple per layer"
    stage_start = source.rfind(
        "  for (uint32_t layer = 0; layer < kLlamaStageLayerCount; ++layer) {"
    )
    assert stage_start >= 0
    stage_region = source[stage_start:]
    assert "candidate.layer_buffers[layer]" in stage_region, (
        "each layer's stages must bind that layer's resident weight indices"
    )


def test_warm_prefill_never_uploads_layer_weight_windows() -> None:
    execution = _execution_source()
    prefill = _method_source(execution, "bool prefill(", "\n private:")
    assert not re.search(
        r"(?:upload_named|upload_weight|upload_weights)\s*\([^;]*"
        r"(?:weight|layer|proj|norm)",
        prefill,
        flags=re.IGNORECASE | re.DOTALL,
    ), "warm Prefill must not upload model-weight windows per request"


def test_concrete_backend_owns_one_execution_object_from_prepare_through_release() -> None:
    backend = _backend_source()
    owner = _persistent_owner_name(backend)
    prepare = _method_source(backend, "bool prepare(", "\n  bool commit(")
    commit = _method_source(backend, "bool commit(", "\n  bool rollback(")
    rollback = _method_source(backend, "bool rollback(", "\n  bool release(")
    release = _method_source(backend, "bool release(", "\n  bool prefill(")
    prefill = _method_source(backend, "bool prefill(", "\n  bool health(")

    for operation, method in (
        ("Prepare", prepare),
        ("Commit", commit),
        ("Rollback", rollback),
        ("Release", release),
        ("Prefill", prefill),
    ):
        assert owner in method, (
            f"{operation} must operate on the same persistent execution owner {owner!r}"
        )

    for operation, method in (("Rollback", rollback), ("Release", release)):
        assert re.search(
            rf"(?:{re.escape(owner)}\s*(?:->|\.)\s*"
            rf"(?:close|release|reset|teardown|destroy|shutdown)\w*\s*\(|"
            rf"(?:close|release|reset|teardown|destroy|shutdown)\w*\s*\([^)]*"
            rf"{re.escape(owner)})",
            method,
            flags=re.IGNORECASE,
        ), f"{operation} must tear down the persistent execution owner"


def test_concrete_backend_prefill_reuses_owner_without_one_shot_execution_or_reload() -> None:
    backend = _backend_source()
    prefill = _method_source(backend, "bool prefill(", "\n  bool health(")

    assert "run_native_prefill(" not in prefill, (
        "warm Prefill must not call the one-shot runner path"
    )
    owner = _persistent_owner_name(backend)
    assert not re.search(r"\bbinder_\s*\.\s*open\s*\(", prefill), (
        "warm Prefill must not reopen and rebind model weights"
    )
    assert not re.search(r"\brelease_all\s*\(", prefill), (
        "warm Prefill must not release resident device state"
    )
    assert not re.search(r"\.\s*close\s*\(", prefill), (
        "warm Prefill must not close the resident device session"
    )
    assert re.search(rf"{re.escape(owner)}\s*(?:->|\.)", prefill), (
        "warm Prefill must dispatch through the persistent execution owner"
    )


def test_prepare_publishes_actual_runner_binary_sha256_in_result() -> None:
    header = _read(WORKER_HEADER)
    worker = _read(WORKER_SOURCE)
    runner = _read(RUNNER_SOURCE)
    backend = _backend_source()
    prepare = _method_source(backend, "bool prepare(", "\n  bool commit(")

    assert re.search(
        r"struct\s+NativePrepareResult\s*\{.*?\brunner_binary_sha256\s*;",
        header,
        flags=re.DOTALL,
    ), "NativePrepareResult must carry runner_binary_sha256"
    assert "runner_binary_sha256" in _prepare_result_serializer(), (
        "Prepare JSON must publish runner_binary_sha256"
    )
    assignment = re.search(
        r"\brunner_binary_sha256\s*=\s*(?P<value>[^;]+);", prepare
    )
    assert assignment is not None, "Prepare must assign the concrete runner hash"
    value = assignment.group("value").strip()
    assert not re.fullmatch(r"\"(?:sha256:)?0{64}\"", value), (
        "Prepare must not fingerprint a fictitious all-zero runner binary"
    )
    assert not re.fullmatch(r"\"0{64}\"", value), (
        "Prepare must not publish an all-zero runner binary digest"
    )
    assert "sha256_executable_file(" in runner
    assert "running_executable_path(" in runner
    assert "runner_binary_sha256" in worker


def test_prepare_derives_ordered_pack_digests_from_selected_concrete_assets() -> None:
    backend = _backend_source()
    prepare = _method_source(backend, "bool prepare(", "\n  bool commit(")

    assert "image_sha256" in backend, (
        "concrete Prepare must inspect selected HsaCodeImageAsset digests"
    )
    assert "ordered_kernel_pack_sha256" in backend
    assert not re.search(
        r"ordered_kernel_pack_sha256\s*=\s*spec\.kernel_pack\.digests\s*;",
        backend,
    ), "producer identity must not copy client-declared pack digests directly"
    assert re.search(
        r"(?:push_back|emplace_back)\s*\([^)]{0,160}image_sha256",
        backend,
        flags=re.DOTALL,
    ), "ordered identity digests must be collected from selected asset descriptors"
    assert re.search(
        r"(?:selected|asset|pack|kernel)[^;\n]{0,160}(?:digest|image|descriptor)",
        prepare,
        flags=re.IGNORECASE | re.DOTALL,
    ), "Prepare must select concrete pack/assets before fingerprinting"


def test_prepare_rejects_declared_pack_digest_mismatch_and_zero_identity() -> None:
    backend = _backend_source()
    prepare = _method_source(backend, "bool prepare(", "\n  bool commit(")

    assert re.search(
        r"(?:kernel_pack\.digests|spec\.kernel_pack\.digests)[\s\S]{0,240}?"
        r"(?:!=|compare|equal|mismatch)|"
        r"(?:!=|compare|equal|mismatch)[\s\S]{0,240}?"
        r"(?:kernel_pack\.digests|spec\.kernel_pack\.digests)",
        prepare,
        flags=re.IGNORECASE | re.DOTALL,
    ), "declared ResourceSpec pack digests must exactly match selected assets"
    assert re.search(
        r"(?:zero|nonzero|all[_ -]?zero|00000000|is_zero)",
        backend,
        flags=re.IGNORECASE,
    ), "zero/client-only pack digests must be rejected before acceptance"


def test_prepare_retains_full_embedding_tensor_for_the_warm_generation() -> None:
    execution = _execution_source()
    prepare = _method_source(
        execution,
        "bool prepare(",
        "\n  const native_r9700::LlamaPersistentDispatch& dispatch()",
    )
    prefill = _method_source(execution, "bool prefill(", "\n private:")
    private_start = execution.rfind("private:")
    assert private_start >= 0
    private = execution[private_start:]

    assert re.search(
        r"std::vector\s*<[^;]+>\s+\w*(?:embed|embedding)\w*\s*;",
        private,
        flags=re.IGNORECASE,
    ), "the full embedding tensor must have persistent generation-owned storage"
    assert re.search(
        r"(?:embed_tokens|embedding)[^;]{0,240}"
        r"(?:read|upload|copy|resident|bytes)",
        prepare,
        flags=re.IGNORECASE | re.DOTALL,
    ), "Prepare must materialize the full embedding tensor once"
    assert "select_llama_embedding_row" not in prefill, (
        "warm Prefill must not reopen or select embedding rows from a shard"
    )
    assert "read_span_bytes" not in prefill
    assert "shard_path" not in prefill


def test_prepare_checks_planned_resident_and_scratch_bytes_before_session_prepare() -> None:
    backend = _backend_source()
    prepare = _method_source(backend, "bool prepare(", "\n  bool commit(")
    session_prepare = prepare.index("execution_->prepare_resident(")
    budget_region = prepare[:session_prepare]

    assert re.search(
        r"(?:planned|required|actual)[\s\S]{0,240}"
        r"(?:resident|scratch|total)",
        budget_region,
        flags=re.IGNORECASE,
    ), "Prepare must calculate actual planned resource usage"
    for field in (
        "spec.resource_budget.resident_bytes_max",
        "spec.resource_budget.scratch_bytes_max",
        "spec.resource_budget.total_bytes_max",
    ):
        assert field in budget_region, f"Prepare must check {field} before session setup"
    assert re.search(
        r"(?:allocation_byte_count|request\.buffers|dispatch\(\)\.request\.buffers)",
        budget_region,
    ), "resource checks must use planned concrete buffer bytes"


def test_concrete_zero_prefix_emits_empty_npz_and_log_without_dispatch() -> None:
    execution = _execution_source()
    prefill = _method_source(execution, "bool prefill(", "\n private:")
    empty_branch = re.search(
        r"if\s*\(\s*request\.token_ids\.empty\(\)\s*\)\s*\{(?P<body>.*?)\n\s*\}",
        prefill,
        flags=re.DOTALL,
    )
    assert empty_branch is not None, (
        "concrete Prefill must define an explicit zero-prefix path"
    )
    body = empty_branch.group("body")
    assert "write_native_prefill_npz" in body
    assert "write_native_prefill_log" in body or "hardware_log_path" in body
    assert "dispatch_batch" not in body
    assert "upload_named" not in body
    assert not re.search(r"\breadback(?:_named)?\s*\(", body)


def test_cleanup_phase_diagnostics_do_not_pollute_private_jsonl_stdout() -> None:
    close = _session_close_source()
    assert "phase_timer" in close and "phase_counter" in close, (
        "cleanup must retain phase diagnostics"
    )
    assert not re.search(r"\b(?:std::)?printf\s*\(", close), (
        "Release/Rollback cleanup diagnostics must not write to stdout"
    )
    assert not re.search(r"(?:std::cout|stdout)\s*(?:<<|,)", close), (
        "cleanup diagnostics must not use the private JSONL stdout stream"
    )
    assert re.search(r"(?:stderr|log|diagnostic)", close, flags=re.IGNORECASE), (
        "cleanup phase diagnostics must be routed to stderr or a log"
    )


def test_native_prefill_npz_writes_canonical_zero_prefix_arrays(tmp_path: Path) -> None:
    output = tmp_path / "zero-prefix.npz"
    executable = _compile_npz_probe(tmp_path)
    completed = subprocess.run(
        [str(executable), str(output)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file()
    with zipfile.ZipFile(output) as archive:
        assert b"'shape': (1, 8, 0, 64)" in archive.read("layer0_K.npy")
        assert b"'shape': (1, 8, 0, 64)" in archive.read("layer15_V.npy")
