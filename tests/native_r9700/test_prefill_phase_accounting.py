"""No-hardware contract for exclusive and complete native prefill phase accounting."""

import importlib
import subprocess
from pathlib import Path
from types import SimpleNamespace


import numpy as np

import pytest
REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_INCLUDE_DIR = REPO_ROOT / "native_r9700"
AMDEV_SESSION_SOURCE = REPO_ROOT / "native_r9700" / "amdev_session.cpp"

CLOSURE_SOURCES = (
    REPO_ROOT / "native_r9700/kernel_catalog.cpp",
    REPO_ROOT / "native_r9700/amdev_packets.cpp",
    REPO_ROOT / "native_r9700/hardware_lock.cpp",
    REPO_ROOT / "native_r9700/vram_layout.cpp",
    REPO_ROOT / "native_r9700/vram_allocator.cpp",
    REPO_ROOT / "native_r9700/dynamic_page_table.cpp",
    REPO_ROOT / "native_r9700/resident_memory.cpp",
    REPO_ROOT / "native_r9700/vram_smoke_asset.cpp",
)

PROBE_SOURCE = r"""
#include "amdev_session.cpp"
#include <cstdio>

int main() {
  native_r9700::PhaseTimers timers;
  timers.sdma_submit_inclusive_usec = 100;
  timers.sdma_fence_wait_usec = 80;
  timers.model_bind_inclusive_usec = 10;
  timers.dispatch_build_inclusive_usec = 20;
  timers.device_prepare_inclusive_usec = 30;
  timers.embedding_upload_inclusive_usec = 40;
  timers.weight_upload_inclusive_usec = 50;
  timers.compute_loop_inclusive_usec = 60;
  timers.kv_readback_inclusive_usec = 70;
  timers.session_close_inclusive_usec = 80;
  timers.npz_serialization_inclusive_usec = 90;
  native_r9700::finalize_phase_accounting(500, &timers);

  if (timers.sdma_submit_exclusive_usec != 20) return 1;
  if (timers.measured_exclusive_total_usec != 450) return 2;
  if (timers.unattributed_usec != 50) return 3;

  native_r9700::PhaseTimers saturating_timers;
  saturating_timers.sdma_submit_inclusive_usec = 10;
  saturating_timers.sdma_fence_wait_usec = 80;
  native_r9700::finalize_phase_accounting(100, &saturating_timers);
  if (saturating_timers.sdma_submit_exclusive_usec != 0) return 4;

  std::printf("status: pass\n");
  return 0;
}
""".lstrip()


def test_prefill_phase_accounting_probe(tmp_path: Path) -> None:
    source = tmp_path / "prefill_phase_accounting_probe.cpp"
    source.write_text(PROBE_SOURCE, encoding="utf-8")
    executable = tmp_path / "prefill_phase_accounting_probe"
    compiled = subprocess.run(
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
            *map(str, CLOSURE_SOURCES),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    completed = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == "status: pass\n"


def test_close_preserves_pre_resident_timer_snapshot() -> None:
    source = AMDEV_SESSION_SOURCE.read_text(encoding="utf-8")
    close_start = source.index("bool ResidentHsaSession::close(")
    close_end = source.index("const PhaseTimers& ResidentHsaSession::phase_timers()", close_start)
    close_source = source[close_start:close_end]
    null_branch_start = close_source.index("if (state.resident == nullptr)")
    null_branch_end = close_source.index("  }\n", null_branch_start)
    null_branch = close_source[null_branch_start:null_branch_end]

    assert close_source.index("state.phase_timers.socket_rpc_count") < null_branch_start
    assert null_branch.index("state.final_timers = state.phase_timers") < null_branch.index(
        "state.reset_after_close()"
    )


_F1_MODEL_DIGEST = "sha256:" + "1" * 64
_F1_PRODUCER_FINGERPRINT = "sha256:" + "2" * 64
_F1_KERNEL_PACK_DIGEST = "sha256:" + "3" * 64
_F1_MODEL_FINGERPRINT = {
    "model_digest": _F1_MODEL_DIGEST,
    "format": "safetensors",
    "quantization": "fp16",
    "model_family": "llama",
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "geometry": {
        "num_layers": 16,
        "num_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
    },
    "rms_norm_eps": 0.00001,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    },
}


def _require_model_service():
    try:
        module = importlib.import_module("native_r9700.model_service")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED: task-set-4 phase accounting requires native_r9700.model_service: {exc}",
            pytrace=False,
        )
    missing = [
        name
        for name in ("ModelRegistry", "ResourceSpec")
        if not callable(getattr(module, name, None))
    ]
    if missing:
        pytest.fail(
            "RED: native_r9700.model_service is missing task-set-2 API: "
            + ", ".join(missing),
            pytrace=False,
        )
    return module


def _service_request(request_id: str, operation: str, body: dict[str, object]):
    return {
        "protocol_version": "r9700_prefill_service_v1",
        "request_id": request_id,
        "operation": operation,
        "body": body,
    }


def _service_prefill_body(handle: str, token_ids: list[int]) -> dict[str, object]:
    return {
        "model_handle": handle,
        "token_ids": token_ids,
        "cache_spec": {
            "schema_version": "mlx_lm_prompt_cache_v1",
            "cache_class": "KVCache",
            "transport": "file",
        },
        "request_options": {"timeout_ms": 300_000},
    }


class _WarmResourceClient:
    """Persistent private-child stand-in; no model bytes or hardware are used."""

    def __init__(self) -> None:
        self.generation = 41
        self.child_pid = 73_003
        self.process_pids: list[int] = []
        self.calls: list[tuple[str, object]] = []
        self.model_uri: str | None = None
    def prepare(self, resource_spec):
        self.process_pids.append(self.child_pid)
        self.calls.append(("Prepare", resource_spec))
        self.model_uri = resource_spec.model_uri
        return {
            "resource_generation": self.generation,
            "state": "prepared",
            "producer_fingerprint": _F1_PRODUCER_FINGERPRINT,
        }

    def commit(self, resource_generation: int):
        self.calls.append(("Commit", resource_generation))
        return {
            "resource_generation": resource_generation,
            "state": "resident-ready",
            "producer_fingerprint": _F1_PRODUCER_FINGERPRINT,
        }

    def prefill(
        self,
        resource_generation: int,
        request_id: str,
        token_ids: list[int],
        prefill_npz_path: str,
        hardware_log_path: str,
    ):
        self.process_pids.append(self.child_pid)
        self.calls.append(
            (
                "Prefill",
                {
                    "resource_generation": resource_generation,
                    "request_id": request_id,
                    "token_ids": list(token_ids),
                    "prefill_npz_path": prefill_npz_path,
                    "hardware_log_path": hardware_log_path,
                },
            )
        )
        n_prefix = len(token_ids)
        arrays: dict[str, object] = {
            "n_prefix": np.array(n_prefix, dtype=np.int64),
            "producer_kind": np.array("r9700_native"),
            "model": np.array(self.model_uri),
            "num_layers": np.array(16, dtype=np.int64),
        }
        shape = (1, 8, n_prefix, 64)
        for layer_index in range(16):
            arrays[f"layer{layer_index}_K"] = np.zeros(shape, dtype=np.float16)
            arrays[f"layer{layer_index}_V"] = np.zeros(shape, dtype=np.float16)
        np.savez(prefill_npz_path, **arrays)
        Path(hardware_log_path).write_text("native evidence\n", encoding="utf-8")
        return {
            "resource_generation": resource_generation,
            "producer_fingerprint": _F1_PRODUCER_FINGERPRINT,
            "native_prefill_acceptance": "pass",
            "native_prefill_full_layer_loop_status": "pass",
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "hardware_log_path": hardware_log_path,
            "compute_completion_policy": "terminal",
            "compute_barrier_policy": "full",
            "prefill_npz_path": prefill_npz_path,
            "kernel_count": 1,
            "transfer_bytes": 4096,
            "prefill_elapsed_usec": 12500,
            "kernel_elapsed_usec": 8000,
            "transfer_elapsed_usec": 2500,
            "transfer_h2d_bytes": 3072,
            "transfer_d2h_bytes": 1024,
            "block_tokens": len(token_ids),
            "block_count": 1,
            "failure_stage": "",
            "exit_status": 0,
            "failure_text": "",
        }

    def release(self, resource_generation: int):
        self.calls.append(("Release", resource_generation))
        return {
            "resource_generation": resource_generation,
            "state": "released",
            "already_released": False,
        }

    def rollback(self, resource_generation: int):
        self.calls.append(("Rollback", resource_generation))
        return {
            "resource_generation": resource_generation,
            "state": "released",
            "already_released": False,
        }

    def health(self):
        self.calls.append(("Health", None))
        return {
            "child_state": "ready",
            "resource_generation": self.generation,
            "resource_state": "resident-ready",
            "producer_fingerprint": _F1_PRODUCER_FINGERPRINT,
            "error_summary": None,
        }

    def shutdown(self):
        self.calls.append(("Shutdown", None))
        return {"state": "shutdown"}


def test_cold_and_warm_phase_accounting_projects_public_evidence_without_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second request uses one resident generation and only warm phases."""
    model_service = _require_model_service()
    monkeypatch.setattr(
        model_service,
        "verify_model_identity",
        lambda model_uri, supplied_digest: SimpleNamespace(
            canonical_uri="/models/llama",
            digest=_F1_MODEL_DIGEST,
            fingerprint=_F1_MODEL_FINGERPRINT,
            resident_bytes=4096,
        ),
    )
    client = _WarmResourceClient()
    registry = model_service.ModelRegistry(
        resource_client=client,
        artifact_dir=tmp_path / "artifacts",
        kernel_pack={
            "name": "r9700-llama-fp16",
            "version": "v1",
            "digests": [_F1_KERNEL_PACK_DIGEST],
        },
        resource_budget={
            "resident_bytes_max": 100_000,
            "scratch_bytes_max": 20_000,
            "total_bytes_max": 120_000,
        },
    )
    load = registry.dispatch(
        _service_request(
            "cold-load",
            "LoadModel",
            {
                "model_uri": "/models/llama",
                "model_digest": _F1_MODEL_DIGEST,
                "format": "safetensors",
                "quantization": "fp16",
            },
        )
    )
    assert load["status"] == "pass"
    handle = load["result"]["model_handle"]
    assert handle.startswith("mh_")

    first = registry.dispatch(
        _service_request(
            "warm-1",
            "Prefill",
            _service_prefill_body(handle, [11, 12, 13]),
        )
    )
    calls_after_first = len(client.calls)
    second = registry.dispatch(
        _service_request(
            "warm-2",
            "Prefill",
            _service_prefill_body(handle, [21, 22, 23]),
        )
    )

    assert first["status"] == "pass"
    assert second["status"] == "pass"
    warm_calls = client.calls[calls_after_first:]
    assert [name for name, _ in warm_calls] == ["Prefill"]
    prefill_calls = [
        payload for name, payload in client.calls if name == "Prefill"
    ]
    assert [payload["resource_generation"] for payload in prefill_calls] == [
        client.generation,
        client.generation,
    ]
    assert [payload["token_ids"] for payload in prefill_calls] == [
        [11, 12],
        [21, 22],
    ]
    assert len({payload["prefill_npz_path"] for payload in prefill_calls}) == 2
    assert set(client.process_pids) == {client.child_pid}

    for response in (first, second):
        evidence = response["evidence"]
        cache = response["result"]["cache"]
        assert evidence["producer_kind"] == "r9700_native"
        assert evidence["producer_fingerprint"] == _F1_PRODUCER_FINGERPRINT
        assert "resource_generation" not in evidence
        assert evidence["prefill_npz_path"] == cache["prefill_npz_path"]
        assert evidence["hardware_log_path"] == cache["prefill_log_path"]
        assert (
            cache["metadata"]["producer_fingerprint"]
            == _F1_PRODUCER_FINGERPRINT
        )
        assert cache["metadata"]["request_id"] in {
            "warm-1",
            "warm-2",
        }

    metrics = registry.dispatch(_service_request("metrics-1", "GetMetrics", {}))
    assert metrics["status"] == "pass"
    snapshot = metrics["result"]["metrics"]
    assert snapshot["load_preparation_count"] == 1
    assert snapshot["warm_prefill_weight_reload_count"] == 0
    assert snapshot["prefill_count"] == 2
    names_before_close = [name for name, _ in client.calls]
    assert names_before_close[:4] == ["Prepare", "Commit", "Prefill", "Prefill"]
    assert names_before_close.count("Prepare") == 1
    assert names_before_close.count("Commit") == 1
    assert snapshot["resident_bytes_baseline"] > 0
    assert snapshot["resource_drift_bytes"] == 0
    registry.close()
    assert [name for name, _ in client.calls][-2:] == ["Release", "Shutdown"]
