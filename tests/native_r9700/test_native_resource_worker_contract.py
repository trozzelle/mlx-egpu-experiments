"""Hardware-free RED contracts for the persistent native resource worker.

The probe links only ``native_resource_worker.cpp`` and injects a deterministic
backend.  It never opens TinyGPU, loads numerical model data, or creates a
runner process.  The source-list checks intentionally remain here (rather than
centralizing the existing test lists) so every runner-linked closure must add
the worker translation unit independently.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_DIR = REPO_ROOT / "native_r9700"
WORKER_HEADER = NATIVE_DIR / "native_resource_worker.h"
WORKER_SOURCE = NATIVE_DIR / "native_resource_worker.cpp"
RUNNER_SOURCE = NATIVE_DIR / "runner.cpp"
LEDGER = REPO_ROOT / "docs/tasks/native-r9700-producer/validation-commands.md"

WORKER_RELATIVE = "native_r9700/native_resource_worker.cpp"
RUNNER_RELATIVE = "native_r9700/runner.cpp"
PRIVATE_PROTOCOL = "r9700_native_resource_v1"
PRIVATE_OPERATIONS = (
    "Prepare",
    "Commit",
    "Rollback",
    "Release",
    "Prefill",
    "Health",
    "Shutdown",
)

RUNNER_CLOSURES = (
    ("tests/native_r9700/test_block_prefill_runtime_contract.py", "RUNNER_SOURCES"),
    ("tests/native_r9700/test_compute_barrier_policy.py", "RUNNER_SOURCES"),
    ("tests/native_r9700/test_native_hsa_prefill_contract.py", "RUNNER_SOURCES"),
    ("tests/native_r9700/test_runtime_lifecycle.py", "RUNNER_SOURCES"),
    ("tests/native_r9700/test_runtime_llama_embed_contract.py", "RUNNER_SOURCES"),
    ("tests/native_r9700/test_runtime_protocol.py", "RUNNER_SOURCES"),
    ("tests/native_r9700/test_runtime_vram_contract.py", "RUNNER_SOURCES"),
    ("tests/native_r9700/test_gpu_stage_profile_contract.py", "FORMAT_PROBE_SOURCES"),
)
ACTIVE_LEDGER_RUNNER_SECTIONS = (
    "Current native runner build and no-model smokes",
    "P3 schema",
    "P3 scalar migration",
)




# This is intentionally a narrow test-facing C++ seam: production owns the
# concrete backend, while tests supply fake resources and no device/runtime.
_PROBE_SOURCE = r'''#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "native_resource_worker.h"

namespace {

constexpr uint64_t kGeneration = 17;
constexpr const char* kFingerprint =
    "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr const char* kModelUri = "/tmp/f1-native-resource-isolated-model";
constexpr const char* kModelDigest =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr const char* kPackDigest =
    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr const char* kRunnerSha =
    "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";


const char* kSpecJson =
    R"JSON({"model_uri":"/tmp/f1-native-resource-isolated-model","model_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","model_fingerprint":{"model_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","format":"safetensors","quantization":"fp16","model_family":"llama","model_type":"llama","architectures":["LlamaForCausalLM"],"geometry":{"num_layers":16,"num_heads":32,"n_kv_heads":8,"head_dim":64,"hidden_size":2048,"intermediate_size":8192,"vocab_size":128256,"max_position_embeddings":131072},"rms_norm_eps":0.00001,"rope_theta":500000,"rope_scaling":{"rope_type":"llama3","factor":32,"high_freq_factor":4,"low_freq_factor":1,"original_max_position_embeddings":8192}},"cache_capacity":{"batch":1,"prefix_positions":128},"kernel_pack":{"name":"f1-test-pack","version":"v1","digests":["sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]},"resource_budget":{"resident_bytes_max":1048576,"scratch_bytes_max":262144,"total_bytes_max":1310720}})JSON";

const char* kPrefillBody =
    R"JSON({"resource_generation":17,"request_id":"prefill-request-1","token_ids":[128000,128001],"prefill_npz_path":"/tmp/f1-native-resource-prefill.npz","hardware_log_path":"/tmp/f1-native-resource-prefill.log"})JSON";

std::string prefill_body(const char* request_id, unsigned int token_count) {
  std::string body = std::string("{\"resource_generation\":17,\"request_id\":\"") +
                     request_id + "\",\"token_ids\":[";
  for (unsigned int index = 0; index < token_count; ++index) {
    if (index != 0U) body.push_back(',');
    body += std::to_string(128000U + index);
  }
  body += "],\"prefill_npz_path\":\"/tmp/f1-native-resource-prefill.npz\"";
  body += ",\"hardware_log_path\":\"/tmp/f1-native-resource-prefill.log\"}";
  return body;
}

bool require(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "%s\n", message);
  return condition;
}

void set_error(native_r9700::NativeResourceError* error,
               const char* domain, const char* message, const char* stage) {
  if (error == nullptr) return;
  error->domain = domain;
  error->message = message;
  error->failure_stage = stage;
}

std::string frame(const char* request_id, const char* operation,
                  const std::string& body) {
  return std::string("{\"protocol_version\":\"") +
         "r9700_native_resource_v1" + "\",\"request_id\":\"" +
         request_id + "\",\"operation\":\"" + operation +
         "\",\"body\":" + body + "}\n";
}

struct FakeBackend final : native_r9700::NativeResourceBackend {
  bool fail_prepare = false;
  bool fail_release_once = false;
  bool fault_prefill = false;
  bool accept_boundary_prefill = false;
  bool release_failed = false;
  bool resident_ready = false;
  int prepare_calls = 0;
  int prepare_self_cleanup_calls = 0;
  int commit_calls = 0;
  int rollback_calls = 0;
  int release_calls = 0;
  int prefill_calls = 0;
  int health_calls = 0;
  int shutdown_calls = 0;
  native_r9700::NativeResourceSpec observed_spec{};
  native_r9700::NativeResourcePrefillRequest observed_prefill{};

  bool prepare(const native_r9700::NativeResourceSpec& spec,
               native_r9700::NativePrepareResult* result,
               native_r9700::NativeResourceError* error) override {
    ++prepare_calls;
    observed_spec = spec;
    if (fail_prepare) {
      // A failed native Prepare owns and rolls back its own partial work.  The
      // worker must not invent a cleanup token and call rollback afterward.
      ++prepare_self_cleanup_calls;
      set_error(error, "resource_exhaustion", "fake prepare allocation failure",
                "prepare_allocate");
      return false;
    }
    if (result == nullptr) return false;
    result->resource_generation = kGeneration;
    result->state = "prepared";
    result->producer_fingerprint = kFingerprint;
    result->runner_binary_sha256 = kRunnerSha;
    return true;
  }

  bool commit(uint64_t generation, native_r9700::NativeCommitResult* result,
              native_r9700::NativeResourceError* error) override {
    ++commit_calls;
    if (generation != kGeneration) {
      set_error(error, "invalid_request", "unexpected generation", "commit_generation");
      return false;
    }
    if (result == nullptr) return false;
    resident_ready = true;
    result->resource_generation = generation;
    result->state = "resident-ready";
    result->producer_fingerprint = kFingerprint;
    return true;
  }

  bool rollback(uint64_t generation, native_r9700::NativeCleanupResult* result,
                native_r9700::NativeResourceError* error) override {
    ++rollback_calls;
    if (generation != kGeneration) {
      set_error(error, "invalid_request", "unexpected generation", "rollback_generation");
      return false;
    }
    if (result == nullptr) return false;
    result->resource_generation = generation;
    result->state = "released";
    result->already_released = rollback_calls > 1;
    resident_ready = false;
    return true;
  }

  bool release(uint64_t generation, native_r9700::NativeCleanupResult* result,
               native_r9700::NativeResourceError* error) override {
    ++release_calls;
    if (generation != kGeneration) {
      set_error(error, "invalid_request", "unexpected generation", "release_generation");
      return false;
    }
    if (fail_release_once && release_calls == 1) {
      release_failed = true;
      set_error(error, "device_lost_or_faulted", "fake release failure",
                "release_unmap");
      return false;
    }
    if (result == nullptr) return false;
    result->resource_generation = generation;
    result->state = "released";
    result->already_released = release_calls > 1;
    release_failed = false;
    resident_ready = false;
    return true;
  }

  bool prefill(const native_r9700::NativeResourcePrefillRequest& request,
               native_r9700::NativeResourcePrefillResult* result,
               native_r9700::NativeResourceError* error) override {
    ++prefill_calls;
    observed_prefill = request;
    if (request.resource_generation != kGeneration ||
        (!accept_boundary_prefill && request.request_id != "prefill-request-1") ||
        (!accept_boundary_prefill && request.token_ids.size() != 2U) ||
        request.prefill_npz_path != "/tmp/f1-native-resource-prefill.npz" ||
        request.hardware_log_path != "/tmp/f1-native-resource-prefill.log") {
      set_error(error, "invalid_request", "prefill request was changed",
                "prefill_request");
      return false;
    }
    if (fault_prefill) {
      set_error(error, "device_lost_or_faulted", "fake device fault",
                "prefill_dispatch");
      return false;
    }
    if (result == nullptr) return false;
    result->resource_generation = request.resource_generation;
    result->producer_fingerprint = kFingerprint;
    result->native_prefill_acceptance = "pass";
    result->native_prefill_full_layer_loop_status = "pass";
    result->runtime_substrate = "TinyGPU.app/APLRemotePCIDevice/PCIIface";
    result->hardware_log_path = request.hardware_log_path;
    result->compute_completion_policy = "terminal";
    result->compute_barrier_policy = "full";
    result->prefill_npz_path = request.prefill_npz_path;
    result->kernel_count = 16;
    result->transfer_bytes = 128;
    result->block_tokens = accept_boundary_prefill ? request.token_ids.size() : 4;
    result->block_count = 1;
    result->failure_stage = "none";
    result->exit_status = 0;
    result->failure_text = "";
    return true;
  }

  bool health(native_r9700::NativeHealthResult* result,
              native_r9700::NativeResourceError* error) override {
    (void)error;
    ++health_calls;
    if (result == nullptr) return false;
    *result = native_r9700::NativeHealthResult{};
    result->child_state = fault_prefill ? "faulted" : "ready";
    if (release_failed) {
      result->resource_state = "release-failed";
      result->resource_generation = kGeneration;
      result->producer_fingerprint = kFingerprint;
    } else if (resident_ready) {
      result->resource_state = "resident-ready";
      result->resource_generation = kGeneration;
      result->producer_fingerprint = kFingerprint;
    } else {
      result->resource_state = "none";
    }
    return true;
  }

  bool shutdown(native_r9700::NativeShutdownResult* result,
                 native_r9700::NativeResourceError* error) override {
    ++shutdown_calls;
    (void)error;
    if (result == nullptr) return false;
    result->state = "shutdown";
    return true;
  }
};

bool run_worker(const std::string& input, FakeBackend* backend,
                std::string* output) {
  std::istringstream input_stream(input);
  std::ostringstream output_stream;
  const int status = native_r9700::run_native_resource_worker(
      input_stream, output_stream, *backend);
  if (!require(status == 0, "worker JSONL loop must exit normally")) return false;
  if (output != nullptr) *output = output_stream.str();
  return true;
}

bool emit(const std::string& output) {
  std::cout << output;
  return true;
}

int protocol_mode() {
  FakeBackend backend;
  std::string oversized(65536, 'x');
  oversized.push_back('\n');
  std::string invalid_utf8("\xff\n", 2);
  const std::string duplicate =
      "{\"protocol_version\":\"r9700_native_resource_v1\","
      "\"protocol_version\":\"r9700_native_resource_v1\","
      "\"request_id\":\"duplicate\",\"operation\":\"Health\","
      "\"body\":{}}\n";
  const std::string unclosed =
      "{\"protocol_version\":\"r9700_native_resource_v1\",\"request_id\":\"unclosed\","
      "\"operation\":\"Health\",\"body\":{\n";
  const std::string input =
      oversized + "not-json\n" + invalid_utf8 + duplicate + unclosed +
      frame("health-1", "Health", "{}");
  std::string output;
  if (!run_worker(input, &backend, &output)) return 1;
  if (!require(backend.health_calls == 1,
               "pre-decode rejects must discard and continue to the next frame")) {
    return 1;
  }
  return emit(output) ? 0 : 1;
}

int prefill_bounds_mode() {
  FakeBackend backend;
  backend.accept_boundary_prefill = true;
  const std::string prepare =
      frame("prepare-bounds", "Prepare",
            std::string("{\"resource_spec\":") + kSpecJson + "}");
  const std::string commit =
      frame("commit-bounds", "Commit", "{\"resource_generation\":17}");
  const std::string empty =
      frame("prefill-empty", "Prefill", prefill_body("prefill-empty", 0));
  const std::string maximum =
      frame("prefill-maximum", "Prefill", prefill_body("prefill-maximum", 128));
  const std::string over =
      frame("prefill-over", "Prefill", prefill_body("prefill-over", 129));
  std::string output;
  if (!run_worker(prepare + commit + empty + maximum + over, &backend, &output)) return 1;
  return emit(output) ? 0 : 1;
}

int resource_spec_mode() {
  FakeBackend backend;
  std::string bad_spec = kSpecJson;
  bad_spec.pop_back();
  bad_spec += ",\"unknown_resource_field\":true}";
  const std::string input = frame("bad-spec", "Prepare", std::string("{\"resource_spec\":") +
                                             bad_spec + "}") +
                            frame("good-spec", "Prepare",
                                  std::string("{\"resource_spec\":") + kSpecJson + "}");
  std::string output;
  if (!run_worker(input, &backend, &output)) return 1;
  if (!require(backend.prepare_calls == 1,
               "unknown ResourceSpec fields must be rejected before backend work")) {
    return 1;
  }
  if (!require(backend.observed_spec.model_uri == kModelUri &&
                   backend.observed_spec.model_digest == kModelDigest &&
                   backend.observed_spec.cache_capacity.batch == 1 &&
                   backend.observed_spec.cache_capacity.prefix_positions == 128 &&
                   backend.observed_spec.kernel_pack.name == "f1-test-pack" &&
                   backend.observed_spec.kernel_pack.version == "v1" &&
                   backend.observed_spec.kernel_pack.digests.size() == 1U &&
                   backend.observed_spec.kernel_pack.digests[0] == kPackDigest &&
                   backend.observed_spec.resource_budget.resident_bytes_max == 1048576 &&
                   backend.observed_spec.resource_budget.scratch_bytes_max == 262144 &&
                   backend.observed_spec.resource_budget.total_bytes_max == 1310720,
               "Prepare must pass the immutable ResourceSpec fields without mutation")) {
    return 1;
  }
  return emit(output) ? 0 : 1;
}

int lifecycle_mode() {
  FakeBackend backend;
  const std::string prepare =
      frame("prepare-1", "Prepare",
            std::string("{\"resource_spec\":") + kSpecJson + "}");
  const std::string commit = frame(
      "commit-1", "Commit", "{\"resource_generation\":17}");
  const std::string prefill_1 = frame("prefill-1", "Prefill", kPrefillBody);
  const std::string prefill_2 = frame("prefill-2", "Prefill", kPrefillBody);
  const std::string health = frame("health-1", "Health", "{}");
  std::string output;
  if (!run_worker(prepare + commit + prefill_1 + prefill_2 + health,
                  &backend, &output)) return 1;
  if (!require(backend.prepare_calls == 1 && backend.commit_calls == 1 &&
                   backend.prefill_calls == 2 && backend.health_calls == 1,
               "one resident generation must serve repeated Prefill calls")) {
    return 1;
  }
  if (!require(backend.observed_prefill.resource_generation == kGeneration &&
                   backend.observed_prefill.request_id == "prefill-request-1",
               "Prefill must reuse the committed generation and request fields")) {
    return 1;
  }
  return emit(output) ? 0 : 1;
}

int prepare_failure_mode() {
  FakeBackend backend;
  backend.fail_prepare = true;
  const std::string input =
      frame("prepare-fail", "Prepare",
            std::string("{\"resource_spec\":") + kSpecJson + "}") +
      frame("health-after-fail", "Health", "{}");
  std::string output;
  if (!run_worker(input, &backend, &output)) return 1;
  if (!require(backend.prepare_calls == 1 && backend.prepare_self_cleanup_calls == 1 &&
                   backend.rollback_calls == 0,
               "Prepare failure must self-clean and return no rollback token")) {
    return 1;
  }
  return emit(output) ? 0 : 1;
}

int rollback_mode() {
  FakeBackend backend;
  const std::string input =
      frame("prepare-rollback", "Prepare",
            std::string("{\"resource_spec\":") + kSpecJson + "}") +
      frame("rollback-1", "Rollback", "{\"resource_generation\":17}") +
      frame("rollback-2", "Rollback", "{\"resource_generation\":17}");
  std::string output;
  if (!run_worker(input, &backend, &output)) return 1;
  if (!require(backend.prepare_calls == 1 && backend.rollback_calls == 2,
               "Rollback retries must target the same one-generation cleanup")) {
    return 1;
  }
  return emit(output) ? 0 : 1;
}

int release_mode() {
  FakeBackend backend;
  const std::string input =
      frame("prepare-release", "Prepare",
            std::string("{\"resource_spec\":") + kSpecJson + "}") +
      frame("commit-release", "Commit", "{\"resource_generation\":17}") +
      frame("release-1", "Release", "{\"resource_generation\":17}") +
      frame("release-2", "Release", "{\"resource_generation\":17}");
  std::string output;
  if (!run_worker(input, &backend, &output)) return 1;
  if (!require(backend.prepare_calls == 1 && backend.commit_calls == 1 &&
                   backend.release_calls == 2,
               "Release retries must be idempotent for one committed generation")) {
    return 1;
  }
  return emit(output) ? 0 : 1;
}

int release_failed_mode() {
  FakeBackend backend;
  backend.fail_release_once = true;
  const std::string input =
      frame("prepare-release-failed", "Prepare",
            std::string("{\"resource_spec\":") + kSpecJson + "}") +
      frame("commit-release-failed", "Commit", "{\"resource_generation\":17}") +
      frame("release-error", "Release", "{\"resource_generation\":17}") +
      frame("health-release-failed", "Health", "{}") +
      frame("prefill-release-failed", "Prefill", kPrefillBody) +
      frame("shutdown-release-failed", "Shutdown", "{}") +
      frame("release-retry", "Release", "{\"resource_generation\":17}") +
      frame("shutdown-after-release", "Shutdown", "{}");
  std::string output;
  if (!run_worker(input, &backend, &output)) return 1;
  if (!require(backend.release_calls == 2 && backend.prefill_calls == 0 &&
                   backend.shutdown_calls == 1 && backend.health_calls == 1,
               "release-failed must allow only Health and matching cleanup retry")) {
    return 1;
  }
  return emit(output) ? 0 : 1;
}

int fault_mode() {
  FakeBackend backend;
  backend.fault_prefill = true;
  const std::string input =
      frame("prepare-fault", "Prepare",
            std::string("{\"resource_spec\":") + kSpecJson + "}") +
      frame("commit-fault", "Commit", "{\"resource_generation\":17}") +
      frame("prefill-fault", "Prefill", kPrefillBody) +
      frame("health-fault", "Health", "{}");
  std::string output;
  if (!run_worker(input, &backend, &output)) return 1;
  if (!require(backend.prepare_calls == 1 && backend.commit_calls == 1 &&
                   backend.prefill_calls == 1 && backend.health_calls == 1,
               "a child/device fault must not allocate or launch a replacement generation")) {
    return 1;
  }
  return emit(output) ? 0 : 1;
}

int shutdown_mode() {
  FakeBackend backend;
  const std::string input =
      frame("prepare-shutdown", "Prepare",
            std::string("{\"resource_spec\":") + kSpecJson + "}") +
      frame("commit-shutdown", "Commit", "{\"resource_generation\":17}") +
      frame("release-shutdown", "Release", "{\"resource_generation\":17}") +
      frame("shutdown", "Shutdown", "{}");
  std::string output;
  if (!run_worker(input, &backend, &output)) return 1;
  if (!require(backend.shutdown_calls == 1,
               "Shutdown must be explicit and complete after native cleanup")) return 1;
  return emit(output) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "one probe mode is required\n");
    return 2;
  }
  const std::string mode = argv[1];
  if (mode == "protocol") return protocol_mode();
  if (mode == "prefill-bounds") return prefill_bounds_mode();
  if (mode == "resource-spec") return resource_spec_mode();
  if (mode == "lifecycle") return lifecycle_mode();
  if (mode == "prepare-failure") return prepare_failure_mode();
  if (mode == "rollback") return rollback_mode();
  if (mode == "release") return release_mode();
  if (mode == "release-failed") return release_failed_mode();
  if (mode == "fault") return fault_mode();
  if (mode == "shutdown") return shutdown_mode();
  std::fprintf(stderr, "unknown probe mode\n");
  return 2;
}
'''


def _write_probe_source(tmp_path: Path) -> Path:
    source = tmp_path / "native_resource_worker_probe.cpp"
    source.write_text(_PROBE_SOURCE, encoding="utf-8")
    return source


def compile_worker_probe(tmp_path: Path) -> Path:
    """Compile the worker against a fake backend; no runner/device is linked."""
    assert WORKER_HEADER.is_file(), (
        "native_r9700/native_resource_worker.h is missing; add the private worker DTO seam"
    )
    assert WORKER_SOURCE.is_file(), (
        "native_r9700/native_resource_worker.cpp is missing; add the private JSONL worker"
    )
    probe_source = _write_probe_source(tmp_path)
    executable = tmp_path / "native_resource_worker_probe"
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
            str(probe_source),
            str(WORKER_SOURCE),
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


def run_probe(executable: Path, mode: str) -> list[dict[str, object]]:
    completed = subprocess.run(
        [str(executable), mode],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line]
    assert lines, "native worker probe must emit at least one JSONL response"
    try:
        responses = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise AssertionError(completed.stdout) from exc
    assert all(isinstance(response, dict) for response in responses)
    return responses


def assert_private_envelope(response: dict[str, object]) -> None:
    assert set(response) == {
        "protocol_version",
        "request_id",
        "operation",
        "status",
        "result",
        "error",
    }
    assert response["protocol_version"] == PRIVATE_PROTOCOL
    assert isinstance(response["result"], dict)


def assert_error_response(
    response: dict[str, object], *, stage: str | None = None, status: str | None = None
) -> None:
    assert_private_envelope(response)
    assert response["status"] == (status or "error")
    assert response["result"] == {}
    error = response["error"]
    assert isinstance(error, dict)
    assert set(error) == {"domain", "message", "failure_stage"}
    assert error["domain"] in {
        "invalid_request",
        "resource_exhaustion",
        "device_lost_or_faulted",
    }
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error["failure_stage"], str) and error["failure_stage"]
    if stage is not None:
        assert error["failure_stage"] == stage


def assert_success_response(response: dict[str, object], operation: str) -> dict[str, object]:
    assert_private_envelope(response)
    assert response["operation"] == operation
    assert response["status"] == "pass"
    assert response["error"] is None
    result = response["result"]
    assert isinstance(result, dict)
    return result


def _assignment_string_constants(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        value = None
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if value is None or not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        return [
            child.value
            for child in ast.walk(value)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        ]
    raise AssertionError(f"{name} assignment is missing from {path}")


def _ledger_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^### {re.escape(heading)}\n(.*?)(?=^### |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"active validation ledger section is missing: {heading}"
    return match.group(1)


def test_private_worker_prefill_accepts_empty_and_maximum_prefix_rejects_129(
    tmp_path: Path,
) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "prefill-bounds")
    assert len(responses) == 5
    assert_success_response(responses[0], "Prepare")
    assert_success_response(responses[1], "Commit")
    empty = assert_success_response(responses[2], "Prefill")
    assert empty["block_tokens"] == 0
    maximum = assert_success_response(responses[3], "Prefill")
    assert maximum["block_tokens"] == 128
    rejected = responses[4]
    assert_error_response(rejected, status="blocked", stage="operation_validation")
    assert rejected["request_id"] == "prefill-over"
    assert rejected["operation"] == "Prefill"


def test_private_worker_predecode_envelope_discards_and_continues(tmp_path: Path) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "protocol")
    assert len(responses) == 6
    expected_stages = ("frame_size", "frame_decode", "frame_decode", "frame_decode", "frame_decode")
    for response, expected_stage in zip(responses[:5], expected_stages):
        assert_error_response(response, stage=expected_stage)
        assert response["request_id"] is None
        assert response["operation"] is None
        error = response["error"]
        assert isinstance(error, dict)
        assert error["domain"] == "invalid_request"
        assert error["message"] == "raw frame rejected before decode"
        assert len(error["message"].encode("utf-8")) <= 16 * 1024
    health = responses[5]
    result = assert_success_response(health, "Health")
    assert result["resource_state"] == "none"
    assert "evidence" not in health


def test_private_worker_rejects_unknown_resource_spec_fields_before_backend_work(
    tmp_path: Path,
) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "resource-spec")
    assert len(responses) == 2
    assert_error_response(responses[0], status="blocked")
    assert responses[0]["request_id"] == "bad-spec"
    assert responses[0]["operation"] == "Prepare"
    error = responses[0]["error"]
    assert isinstance(error, dict)
    assert error["domain"] == "invalid_request"
    assert error["failure_stage"] == "operation_validation"
    result = assert_success_response(responses[1], "Prepare")
    assert result == {
        "resource_generation": 17,
        "state": "prepared",
        "producer_fingerprint":
            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "runner_binary_sha256":
            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    }


def test_private_worker_prepare_commit_and_repeated_prefill_reuse_one_generation(
    tmp_path: Path,
) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "lifecycle")
    assert len(responses) == 5
    prepare = assert_success_response(responses[0], "Prepare")
    commit = assert_success_response(responses[1], "Commit")
    assert prepare["resource_generation"] == commit["resource_generation"] == 17
    assert prepare["producer_fingerprint"] == commit["producer_fingerprint"]
    for response in responses[2:4]:
        result = assert_success_response(response, "Prefill")
        assert result["resource_generation"] == 17
        assert result["producer_fingerprint"] == prepare["producer_fingerprint"]
        assert set(result) == {
            "resource_generation",
            "producer_fingerprint",
            "native_prefill_acceptance",
            "native_prefill_full_layer_loop_status",
            "runtime_substrate",
            "hardware_log_path",
            "compute_completion_policy",
            "compute_barrier_policy",
            "prefill_npz_path",
            "kernel_count",
            "transfer_bytes",
            "block_tokens",
            "block_count",
            "failure_stage",
            "exit_status",
            "failure_text",
        }
    health = assert_success_response(responses[4], "Health")
    assert health["resource_generation"] == 17
    assert health["resource_state"] == "resident-ready"
    assert health["producer_fingerprint"] == prepare["producer_fingerprint"]


def test_private_worker_prepare_failure_self_cleans_without_cleanup_token(
    tmp_path: Path,
) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "prepare-failure")
    assert len(responses) == 2
    assert_error_response(responses[0])
    assert responses[0]["request_id"] == "prepare-fail"
    assert responses[0]["operation"] == "Prepare"
    health = assert_success_response(responses[1], "Health")
    assert health["resource_generation"] is None
    assert health["resource_state"] == "none"
    assert health["producer_fingerprint"] is None


def test_private_worker_rollback_is_idempotent_for_same_generation(tmp_path: Path) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "rollback")
    assert len(responses) == 3
    assert_success_response(responses[0], "Prepare")
    first = assert_success_response(responses[1], "Rollback")
    repeat = assert_success_response(responses[2], "Rollback")
    assert first == {
        "resource_generation": 17,
        "state": "released",
        "already_released": False,
    }
    assert repeat == {
        "resource_generation": 17,
        "state": "released",
        "already_released": True,
    }


def test_private_worker_release_is_idempotent_for_same_generation(tmp_path: Path) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "release")
    assert len(responses) == 4
    assert_success_response(responses[0], "Prepare")
    assert_success_response(responses[1], "Commit")
    first = assert_success_response(responses[2], "Release")
    repeat = assert_success_response(responses[3], "Release")
    assert first == {
        "resource_generation": 17,
        "state": "released",
        "already_released": False,
    }
    assert repeat == {
        "resource_generation": 17,
        "state": "released",
        "already_released": True,
    }


def test_private_worker_release_failed_health_retry_gate_and_shutdown_order(
    tmp_path: Path,
) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "release-failed")
    assert len(responses) == 8
    assert_success_response(responses[0], "Prepare")
    assert_success_response(responses[1], "Commit")
    assert_error_response(responses[2])
    assert responses[2]["operation"] == "Release"
    health = assert_success_response(responses[3], "Health")
    assert health["resource_generation"] == 17
    assert health["resource_state"] == "release-failed"
    assert isinstance(health["error_summary"], dict)
    assert set(health["error_summary"]) == {"domain", "message", "failure_stage"}
    assert health["error_summary"]["domain"] == "device_lost_or_faulted"
    assert_error_response(responses[4], status="blocked")
    assert_error_response(responses[5], status="blocked")
    retry = assert_success_response(responses[6], "Release")
    assert retry == {
        "resource_generation": 17,
        "state": "released",
        "already_released": False,
    }
    shutdown = assert_success_response(responses[7], "Shutdown")
    assert shutdown == {"state": "shutdown"}


def test_private_worker_fault_is_error_and_does_not_replace_generation(tmp_path: Path) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "fault")
    assert len(responses) == 4
    assert_success_response(responses[0], "Prepare")
    assert_success_response(responses[1], "Commit")
    assert_error_response(responses[2])
    assert responses[2]["operation"] == "Prefill"
    assert responses[2]["error"]["domain"] == "device_lost_or_faulted"
    health = assert_success_response(responses[3], "Health")
    assert health["child_state"] == "faulted"
    assert health["resource_generation"] == 17
    assert health["resource_state"] == "resident-ready"


def test_private_worker_shutdown_is_post_cleanup_and_exact(tmp_path: Path) -> None:
    responses = run_probe(compile_worker_probe(tmp_path), "shutdown")
    assert len(responses) == 4
    assert_success_response(responses[0], "Prepare")
    assert_success_response(responses[1], "Commit")
    assert_success_response(responses[2], "Release")
    shutdown = assert_success_response(responses[3], "Shutdown")
    assert shutdown == {"state": "shutdown"}


def test_producer_fingerprint_source_pins_exact_jcs_identity_preimage() -> None:
    assert WORKER_HEADER.is_file() and WORKER_SOURCE.is_file(), (
        "native resource worker sources are required for producer fingerprint contract"
    )
    source = WORKER_SOURCE.read_text(encoding="utf-8")
    for field in (
        '"domain"',
        '"protocol_version"',
        '"runner_binary_sha256"',
        '"ordered_kernel_pack_sha256"',
        '"target"',
        '"runtime_substrate"',
        '"completion_policy"',
        '"barrier_policy"',
        '"device_identity"',
        '"vendor_id"',
        '"device_id"',
    ):
        assert field in source, f"producer fingerprint preimage is missing {field}"
    assert "r9700-producer-fingerprint-v1" in source
    assert "r9700_native_resource_v1" in source
    assert re.search(r"(?:JCS|jcs|canonical).*SHA|SHA.*(?:JCS|jcs|canonical)", source)


def test_runner_mode_dispatches_worker_and_keeps_one_entrypoint() -> None:
    assert RUNNER_SOURCE.is_file(), "native_r9700/runner.cpp is missing"
    assert WORKER_SOURCE.is_file(), "native_r9700/native_resource_worker.cpp is missing"
    runner = RUNNER_SOURCE.read_text(encoding="utf-8")
    worker = WORKER_SOURCE.read_text(encoding="utf-8")
    assert "--model-service-worker" in runner
    assert "run_native_resource_worker" in runner
    assert len(re.findall(r"\bint\s+main\s*\(", runner)) == 1
    assert not re.search(r"\bint\s+main\s*\(", worker)
    for operation in PRIVATE_OPERATIONS:
        assert operation in worker
    assert "Decode" not in worker


def test_every_frozen_runner_closure_includes_worker_and_single_runner_entrypoint() -> None:
    for relative_path, assignment in RUNNER_CLOSURES:
        path = REPO_ROOT / relative_path
        values = _assignment_string_constants(path, assignment)
        assert WORKER_RELATIVE in values or WORKER_RELATIVE.removeprefix("native_r9700/") in values, (
            f"{relative_path}::{assignment} must compile native_resource_worker.cpp"
        )
        runner_values = [value for value in values if value == RUNNER_RELATIVE or value == "runner.cpp"]
        assert len(runner_values) == 1, (
            f"{relative_path}::{assignment} must retain runner.cpp as its sole entrypoint"
        )


def test_active_ledger_runner_clang_blocks_include_worker_source() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    for heading in ACTIVE_LEDGER_RUNNER_SECTIONS:
        section = _ledger_section(text, heading)
        clang_blocks = [
            block
            for block in re.findall(r"```sh\n(.*?)```", section, flags=re.DOTALL)
            if "clang++" in block
        ]
        assert len(clang_blocks) == 1, f"{heading} must have one runner clang block"
        block = clang_blocks[0]
        assert WORKER_RELATIVE in block
        assert block.count(RUNNER_RELATIVE) == 1
        outputs = re.findall(r"(?:^|\s)-o\s+(\S+native_r9700_runner)(?:\s|$)", block, flags=re.MULTILINE)
        assert len(outputs) == 1, f"{heading} must build one native_r9700_runner output"
