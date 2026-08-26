"""C2 task set 3 RED contract for the mlx-lm imported-cache serving wrapper.

These tests define the future ``native_r9700.serving`` API before production
code lands. The module is imported lazily so pytest collection succeeds; the
current RED should be a clear missing module/API failure, not a syntax or
collection error.

Contract: short prompts stay on the native mlx-lm full-prompt path; long prompts
use the C1 local subprocess/file handoff, validate the complete prompt-cache ABI
before acceptance, decode from the imported S-1 cache with only the final prompt
token, and never recompute the full prompt after accepting imported cache.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = "${HOME}/.pyenv/versions/3.12.8/bin/python3"
_EXPECTED_NUM_LAYERS = 16
_EXPECTED_N_KV_HEADS = 8
_EXPECTED_HEAD_DIM = 64
_DEFAULT_THRESHOLD = 128
_DEFAULT_TIMEOUT_S = 300
_C2_REPORT_HEADING = "## Path C2 — mlx-lm imported-cache serving wrapper"
_CPU_REFERENCE_PRODUCER_KIND = "cpu_reference"
_R9700_NATIVE_PRODUCER_KIND = "r9700_native"


_PUBLIC_API = (
    "NativePrefillError",
    "NativePrefillConfig",
    "PersistentPrefillSession",
    "generate_with_native_prefill",
    "append_or_replace_path_c2_report",
    "main",
)


# Production mutation caught: deleting/renaming the C2 serving module or public
# entry points should fail here while pytest collection remains healthy.
def _serving_module():
    try:
        module = importlib.import_module("native_r9700.serving")
    except ModuleNotFoundError as exc:
        if exc.name == "native_r9700.serving":
            pytest.fail(
                "native_r9700.serving module missing; implement the C2 mlx-lm "
                "imported-cache serving wrapper API"
            )
        raise

    for api_name in _PUBLIC_API:
        assert hasattr(module, api_name), f"native_r9700.serving missing public API: {api_name}"
    assert issubclass(module.NativePrefillError, Exception)
    assert callable(module.NativePrefillConfig)
    assert callable(module.PersistentPrefillSession)
    assert callable(module.generate_with_native_prefill)
    assert callable(module.append_or_replace_path_c2_report)
    assert callable(module.main)
    return module


class FakeTokenizer:
    def __init__(self, prompt_tokens: dict[str, list[int]]):
        self._prompt_tokens = {prompt: list(tokens) for prompt, tokens in prompt_tokens.items()}

    def encode(self, prompt: str) -> list[int]:
        return list(self._prompt_tokens[prompt])


class KVCache:
    """Test double whose type name and state mirror mlx-lm's KVCache contract."""

    def __init__(
        self, n_prefix: int, *, layer_index: int, bad_shape: bool = False,
        bad_offset: bool = False, nonfinite: bool = False
    ):
        shape = (1, _EXPECTED_N_KV_HEADS, n_prefix, _EXPECTED_HEAD_DIM)
        if bad_shape and layer_index == 0:
            shape = (1, _EXPECTED_N_KV_HEADS - 1, n_prefix, _EXPECTED_HEAD_DIM)
        self.keys = np.zeros(shape, dtype=np.float16)
        self.values = np.ones(shape, dtype=np.float16)
        if nonfinite and layer_index == 0:
            self.keys.reshape(-1)[0] = np.float16(np.nan)
        self.state = (self.keys, self.values)
        self.offset = n_prefix - 1 if bad_offset and layer_index == 0 else n_prefix
        self.size = self.offset


def _native_config(serving, tmp_path: Path, **overrides):
    values = {
        "producer_model_dir": "producer-model-dir",
        "python_executable": _PYTHON,
        "threshold_tokens": _DEFAULT_THRESHOLD,
        "producer_timeout_s": _DEFAULT_TIMEOUT_S,
        "artifacts_dir": tmp_path,
        "request_id": "req-test",
        "producer_kind": _CPU_REFERENCE_PRODUCER_KIND,
    }
    values.update(overrides)
    return serving.NativePrefillConfig(**values)


def _as_int_list(prompt) -> list[int]:
    if hasattr(prompt, "tolist"):
        raw = prompt.tolist()
    else:
        raw = list(prompt)
    if isinstance(raw, int):
        return [raw]
    return [int(token) for token in raw]


def _valid_cache(n_prefix: int, **layer_kwargs) -> list[KVCache]:
    return [KVCache(n_prefix, layer_index=i, **layer_kwargs) for i in range(_EXPECTED_NUM_LAYERS)]


def _valid_metadata(n_prefix: int) -> dict[str, str]:
    return {
        "offset": str(n_prefix),
        "num_layers": str(_EXPECTED_NUM_LAYERS),
        "n_kv_heads": str(_EXPECTED_N_KV_HEADS),
        "head_dim": str(_EXPECTED_HEAD_DIM),
    }


def _completed(cmd, returncode: int = 0, stdout: str = "ok", stderr: str = ""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _cache_rejection(message: str, failure_stage: str) -> dict[str, object]:
    return {
        "status": "blocked",
        "error": {
            "domain": "cache_rejection",
            "message": message,
            "failure_stage": failure_stage,
        },
    }


def _install_producer_run(monkeypatch, serving, *, fail_prefill: bool = False, omit_cache_artifact: bool = False, native_evidence: bool = False):
    calls: list[dict] = []

    def fake_run(cmd, *, cwd=None, capture_output=True, text=True, timeout=None, check=False, **kwargs):
        calls.append(
            {
                "cmd": list(cmd),
                "cwd": cwd,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
                "check": check,
                "kwargs": kwargs,
            }
        )
        assert cmd[0] == _PYTHON
        assert cmd[1] == "-m"
        assert timeout is not None
        assert capture_output is True
        assert text is True
        assert check is False

        if cmd[2] == "native_r9700.prefill":
            if fail_prefill:
                return _completed(cmd, returncode=7, stderr="producer boom")
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"prefill-npz")
            if native_evidence:
                log_path = Path(cmd[cmd.index("--log") + 1])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    "\n".join(
                        (
                            "producer_kind: r9700_native",
                            "native_prefill_acceptance: pass",
                            f"hardware_log_path: {log_path}",
                            f"prefill_npz_path: {out_path}",
                            "kernel_count: 3",
                            "transfer_bytes: 4096",
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
            return _completed(cmd, stdout="prefill ok")

        if cmd[2] == "native_r9700.kv_cache":
            if not omit_cache_artifact:
                out_path = Path(cmd[cmd.index("--out") + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"prompt-cache")
            return _completed(cmd, stdout="kv ok")

        raise AssertionError(f"unexpected producer command: {cmd}")

    monkeypatch.setattr(serving, "subprocess", SimpleNamespace(run=fake_run), raising=False)
    return calls


def _assert_prefill_command(cmd: list[str], *, native, prompt_tokens: list[int], request_id: str):
    assert cmd[:3] == [_PYTHON, "-m", "native_r9700.prefill"]
    assert cmd[cmd.index("--model") + 1] == native.producer_model_dir
    assert json.loads(cmd[cmd.index("--token-ids-json") + 1]) == prompt_tokens
    assert "--producer-kind" in cmd
    assert cmd[cmd.index("--producer-kind") + 1] == native.producer_kind
    assert "--fixtures-dir" not in cmd
    assert "--prompt-name" not in cmd
    out_path = Path(cmd[cmd.index("--out") + 1])
    log_path = Path(cmd[cmd.index("--log") + 1])
    assert out_path.name == f"{request_id}.prefill.npz"
    assert log_path.name == f"{request_id}.prefill.log"
    assert out_path.parent == Path(native.artifacts_dir)
    assert log_path.parent == Path(native.artifacts_dir)


def _assert_kv_cache_command(cmd: list[str], *, native, request_id: str, prefill_out: Path):
    assert cmd[:3] == [_PYTHON, "-m", "native_r9700.kv_cache"]
    assert Path(cmd[cmd.index("--prefill-npz") + 1]) == prefill_out
    out_path = Path(cmd[cmd.index("--out") + 1])
    log_path = Path(cmd[cmd.index("--log") + 1])
    assert out_path.name == f"{request_id}.prompt-cache.safetensors"
    assert log_path.name == f"{request_id}.kv-cache.log"
    assert out_path.parent == Path(native.artifacts_dir)
    assert log_path.parent == Path(native.artifacts_dir)


def _write_prompts_fixture(fixtures_dir: Path, prompts: dict[str, dict]) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")


def _write_baseline_fixture(fixtures_dir: Path, baselines: dict[str, list[int]]) -> None:
    payload = {
        name: {"r_tokens": list(tokens), "max_new_tokens": 4, "S": len(tokens)}
        for name, tokens in baselines.items()
    }
    (fixtures_dir / "baseline_r_tokens.json").write_text(json.dumps(payload), encoding="utf-8")

def test_r9700_native_request_falls_back_but_cannot_pass_without_hardware_evidence(tmp_path, monkeypatch):
    serving = _serving_module()
    prompt = "explicit native producer"
    prompt_tokens = [101, 102, 103, 104, 105, 106]
    native = _native_config(
        serving,
        tmp_path,
        threshold_tokens=2,
        request_id="native-missing-evidence",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    service = _LiveServiceDispatcher(
        tmp_path / "service-artifacts",
        prefill_error=_cache_rejection(
            "native evidence missing",
            "native_evidence_validation",
        ),
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    observed = {"generate_calls": 0}

    def fake_generate(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(900)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "persistent native evidence rejection must not call subprocess.run"
        ),
    )
    with session:
        result = serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({prompt: prompt_tokens}),
            prompt,
            native=native,
            max_tokens=1,
            generate_step_fn=fake_generate,
            service_session=session,
        )

    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]
    assert service.last_prefill_response is not None
    assert service.last_prefill_response["status"] == "blocked"
    assert service.last_prefill_response["error"] == {
        "domain": "cache_rejection",
        "message": "native evidence missing",
        "failure_stage": "native_evidence_validation",
    }
    assert observed["generate_calls"] == 1
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == "cache_validation_failed"
    assert result["fallback_detail"] == "native evidence missing"
    assert result["accepted_cache"] is False
    assert result["status"] == "blocked"
    assert result["exit_status"] == 2
    assert result["producer_kind"] is None


def test_below_threshold_uses_native_mlx_full_prompt_without_producer(tmp_path, monkeypatch):
    serving = _serving_module()
    prompt = "tiny prompt"
    prompt_tokens = [101, 102, 103, 104, 105, 106]
    observed = {"generate_calls": 0, "producer_calls": 0}

    native = _native_config(serving, tmp_path, threshold_tokens=128, request_id="below-threshold")

    def unexpected_run(*args, **kwargs):
        observed["producer_calls"] += 1
        raise AssertionError("producer subprocess must not run for below-threshold prompts")

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        observed["generate_prompt"] = _as_int_list(prompt_arg)
        observed["model"] = model
        observed["kwargs"] = kwargs
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(11)
        yield np.int64(12)

    monkeypatch.setattr(serving, "subprocess", SimpleNamespace(run=unexpected_run), raising=False)

    result = serving.generate_with_native_prefill(
        "resident-model",
        FakeTokenizer({prompt: prompt_tokens}),
        prompt,
        native=native,
        max_tokens=2,
        generate_step_fn=fake_generate_step,
        temperature=0.0,
    )

    assert observed["producer_calls"] == 0
    assert observed["generate_calls"] == 1
    assert observed["model"] == "resident-model"
    assert observed["kwargs"]["max_tokens"] == 2
    assert observed["kwargs"]["temperature"] == 0.0
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == "below_threshold"
    assert result["accepted_cache"] is False
    assert result["prompt_token_count"] == len(prompt_tokens)
    assert result["prompt_cache_path"] in {None, ""}
    assert result["decoded_tokens"] == [11, 12]


def test_above_threshold_invokes_c1_subprocesses_accepts_full_abi_and_decodes_final_token_only(
    tmp_path, monkeypatch
):
    serving = _serving_module()
    prompt = "large prompt"
    prompt_tokens = list(range(1000, 1000 + 130))
    n_prefix = len(prompt_tokens) - 1
    request_id = "above-threshold"
    native = _native_config(
        serving,
        tmp_path,
        producer_model_dir="producer-override",
        producer_timeout_s=17,
        request_id=request_id,
    )
    producer_calls = _install_producer_run(monkeypatch, serving)
    cache_objects = _valid_cache(n_prefix)
    observed = {"generate_calls": 0}

    def fake_load_prompt_cache(path, return_metadata=False):
        observed["loaded_cache_path"] = Path(path)
        assert return_metadata is True
        assert Path(path).is_file()
        return cache_objects, _valid_metadata(n_prefix)

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        observed["generate_prompt"] = _as_int_list(prompt_arg)
        observed["model"] = model
        observed["kwargs"] = kwargs
        assert _as_int_list(prompt_arg) == [prompt_tokens[-1]]
        assert _as_int_list(prompt_arg) != prompt_tokens
        assert kwargs["prompt_cache"] is cache_objects
        yield np.int64(501)
        yield np.int64(502)

    result = serving.generate_with_native_prefill(
        "resident-model",
        FakeTokenizer({prompt: prompt_tokens}),
        prompt,
        native=native,
        max_tokens=2,
        generate_step_fn=fake_generate_step,
        load_prompt_cache_fn=fake_load_prompt_cache,
        top_p=0.7,
    )

    assert len(producer_calls) == 2
    prefill_cmd = producer_calls[0]["cmd"]
    kv_cmd = producer_calls[1]["cmd"]
    _assert_prefill_command(prefill_cmd, native=native, prompt_tokens=prompt_tokens, request_id=request_id)
    prefill_out = Path(prefill_cmd[prefill_cmd.index("--out") + 1])
    _assert_kv_cache_command(kv_cmd, native=native, request_id=request_id, prefill_out=prefill_out)
    assert [call["timeout"] for call in producer_calls] == [17, 17]
    assert observed["loaded_cache_path"].name == f"{request_id}.prompt-cache.safetensors"
    assert observed["generate_calls"] == 1
    assert observed["model"] == "resident-model"
    assert observed["kwargs"]["max_tokens"] == 2
    assert observed["kwargs"]["top_p"] == 0.7
    assert result["route"] == "native_producer"
    assert result["fallback_reason"] is None
    assert result["accepted_cache"] is True
    assert result["prompt_token_count"] == len(prompt_tokens)
    assert result["n_prefix"] == n_prefix
    assert result["metadata"] == _valid_metadata(n_prefix)
    assert result["decoded_tokens"] == [501, 502]
    assert result["requested_producer_kind"] == _CPU_REFERENCE_PRODUCER_KIND
    assert result["producer_kind"] == _CPU_REFERENCE_PRODUCER_KIND

def test_r9700_native_accepts_cache_only_with_hardware_evidence(tmp_path, monkeypatch):
    serving = _serving_module()
    prompt = "native evidence prompt"
    prompt_tokens = list(range(1200, 1200 + 130))
    n_prefix = len(prompt_tokens) - 1
    request_id = "native-evidence"
    native = _native_config(
        serving,
        tmp_path,
        request_id=request_id,
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    service = _LiveServiceDispatcher(
        tmp_path,
        evidence_override={"kernel_count": 3},
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    cache_objects = _valid_cache(n_prefix)

    def fake_load_prompt_cache(path, return_metadata=False):
        assert return_metadata is True
        return cache_objects, _task4_metadata(n_prefix, request_id)

    def fake_generate_step(prompt_arg, model, **kwargs):
        assert _as_int_list(prompt_arg) == [prompt_tokens[-1]]
        assert kwargs["prompt_cache"] is cache_objects
        yield np.int64(601)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "persistent native evidence acceptance must not call subprocess.run"
        ),
    )
    with session:
        result = serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({prompt: prompt_tokens}),
            prompt,
            native=native,
            max_tokens=1,
            generate_step_fn=fake_generate_step,
            load_prompt_cache_fn=fake_load_prompt_cache,
            service_session=session,
        )

    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]
    assert service.last_prefill_response is not None
    assert service.last_prefill_response["evidence"]["hardware_log_path"] == str(
        tmp_path / f"{request_id}.prefill.log"
    )
    assert service.last_prefill_response["evidence"]["native_prefill_acceptance"] == "pass"
    assert service.last_prefill_response["evidence"]["kernel_count"] == 3
    assert service.last_prefill_response["evidence"]["transfer_bytes"] == 4096
    assert result["route"] == "native_producer"
    assert result["accepted_cache"] is True
    assert result["status"] == "pass"
    assert result["exit_status"] == 0
    assert result["requested_producer_kind"] == _R9700_NATIVE_PRODUCER_KIND
    assert result["producer_kind"] == _R9700_NATIVE_PRODUCER_KIND
    assert result["native_prefill_acceptance"] == "pass"
    assert result["hardware_log_path"] == str(
        tmp_path / f"{request_id}.prefill.log"
    )
    assert result["kernel_count"] == 3
    assert result["transfer_bytes"] == 4096

@pytest.mark.parametrize("case", ("missing", "unbound"), ids=("missing", "unbound"))
def test_r9700_native_hardware_log_must_exist_and_bind_to_requested_prefill_log(
    tmp_path, monkeypatch, case
):
    """A native cache is unsafe until its hardware log is this request's readable log."""
    serving = _serving_module()
    prompt = "native hardware log safety prompt"
    prompt_tokens = list(range(1300, 1300 + 130))
    native = _native_config(
        serving,
        tmp_path,
        request_id=f"hardware-log-{case}",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    failure_messages = {
        "missing": "hardware_log_path is missing from native evidence",
        "unbound": "hardware_log_path is not bound to the requested prefill log",
    }
    service = _LiveServiceDispatcher(
        tmp_path / "service-artifacts",
        prefill_error=_cache_rejection(
            failure_messages[case],
            "hardware_log_validation",
        ),
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    observed = {"generate_calls": 0}

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(701)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "persistent hardware-log rejection must not call subprocess.run"
        ),
    )
    with session:
        result = serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({prompt: prompt_tokens}),
            prompt,
            native=native,
            max_tokens=1,
            generate_step_fn=fake_generate_step,
            load_prompt_cache_fn=lambda *args, **kwargs: pytest.fail(
                "hardware-log rejection must precede cache loading"
            ),
            service_session=session,
        )

    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]
    assert service.last_prefill_response is not None
    assert service.last_prefill_response["status"] == "blocked"
    assert service.last_prefill_response["error"] == {
        "domain": "cache_rejection",
        "message": failure_messages[case],
        "failure_stage": "hardware_log_validation",
    }
    assert observed == {"generate_calls": 1}
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == "cache_validation_failed"
    assert result["fallback_detail"] == failure_messages[case]
    assert result["accepted_cache"] is False
    assert result["status"] == "blocked"
    assert result["exit_status"] == 2

def test_r9700_native_unreadable_hardware_log_falls_back_before_cache_acceptance(
    tmp_path, monkeypatch
):
    serving = _serving_module()
    prompt = "native unreadable hardware log prompt"
    prompt_tokens = list(range(1500, 1500 + 130))
    native = _native_config(
        serving,
        tmp_path,
        request_id="unreadable-hardware-log",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    failure_message = "hardware_log_path is unreadable"
    service = _LiveServiceDispatcher(
        tmp_path / "service-artifacts",
        prefill_error=_cache_rejection(
            failure_message,
            "hardware_log_validation",
        ),
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    observed = {"generate_calls": 0}

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(702)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "persistent hardware-log rejection must not call subprocess.run"
        ),
    )
    with session:
        result = serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({prompt: prompt_tokens}),
            prompt,
            native=native,
            max_tokens=1,
            generate_step_fn=fake_generate_step,
            load_prompt_cache_fn=lambda *args, **kwargs: pytest.fail(
                "unreadable hardware-log rejection must precede cache loading"
            ),
            service_session=session,
        )

    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]
    assert service.last_prefill_response is not None
    assert service.last_prefill_response["status"] == "blocked"
    assert service.last_prefill_response["error"] == {
        "domain": "cache_rejection",
        "message": failure_message,
        "failure_stage": "hardware_log_validation",
    }
    assert observed == {"generate_calls": 1}
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == "cache_validation_failed"
    assert result["fallback_detail"] == failure_message
    assert result["accepted_cache"] is False
    assert result["status"] == "blocked"
    assert result["exit_status"] == 2


@pytest.mark.parametrize(
    "token_ids",
    (
        [1, -1],
        [1, 2.0],
        [1, True],
        [1, "2"],
        [1, 0x1_0000_0000],
    ),
    ids=("negative", "float", "bool", "string", "uint32-overflow"),
)
def test_r9700_native_rejects_non_uint32_tokens_before_producer_or_fallback(
    tmp_path, monkeypatch, token_ids
):
    serving = _serving_module()
    native = _native_config(
        serving,
        tmp_path,
        threshold_tokens=2,
        request_id="invalid-native-tokens",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    service = _LiveServiceDispatcher(tmp_path / "service-artifacts")
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    monkeypatch.setattr(
        serving,
        "subprocess",
        SimpleNamespace(
            run=lambda *args, **kwargs: pytest.fail(
                "invalid native tokens must not invoke the producer"
            )
        ),
        raising=False,
    )

    with session:
        with pytest.raises(serving.NativePrefillError, match="unsigned 32-bit integer"):
            serving.generate_with_native_prefill(
                "resident-model",
                None,
                token_ids,
                native=native,
                generate_step_fn=lambda *args, **kwargs: pytest.fail(
                    "invalid native tokens must not use fallback generation"
                ),
                service_session=session,
            )

    assert not list(tmp_path.glob("*.prefill.npz"))
    assert not list(tmp_path.glob("*.prompt-cache.safetensors"))
    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "UnloadModel",
    ]


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        pytest.param("nonzero_prefill", "producer_failed", id="prefill-nonzero"),
        pytest.param("missing_cache_artifact", "producer_artifact_missing", id="missing-cache-artifact"),
    ],
)
def test_producer_failure_before_acceptance_falls_back_to_native_full_prompt_and_logs_reason(
    tmp_path, monkeypatch, case, expected_reason
):
    serving = _serving_module()
    prompt = "producer fallback prompt"
    prompt_tokens = list(range(200, 200 + 130))
    native = _native_config(serving, tmp_path, request_id=f"fallback-{case}")
    log_path = tmp_path / f"{case}.log"
    _install_producer_run(
        monkeypatch,
        serving,
        fail_prefill=case == "nonzero_prefill",
        omit_cache_artifact=case == "missing_cache_artifact",
    )
    observed = {"generate_calls": 0}

    def unexpected_load_prompt_cache(*args, **kwargs):
        raise AssertionError("missing/nonzero producer output must fallback before cache loading")

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        observed["generate_prompt"] = _as_int_list(prompt_arg)
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(901)

    result = serving.generate_with_native_prefill(
        "resident-model",
        FakeTokenizer({prompt: prompt_tokens}),
        prompt,
        native=native,
        max_tokens=1,
        log_path=log_path,
        generate_step_fn=fake_generate_step,
        load_prompt_cache_fn=unexpected_load_prompt_cache,
    )

    assert observed["generate_calls"] == 1
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == expected_reason
    assert result["accepted_cache"] is False
    assert result["prompt_token_count"] == len(prompt_tokens)
    log_text = log_path.read_text(encoding="utf-8")
    assert "route: native_mlx_fallback" in log_text
    assert f"fallback_reason: {expected_reason}" in log_text
    assert "accepted_cache: false" in log_text


@pytest.mark.parametrize(
    ("metadata", "cache_layers"),
    [
        pytest.param(
            {"offset": "128", "num_layers": "16", "n_kv_heads": "8", "head_dim": "64"},
            None,
            id="metadata-offset-mismatch",
        ),
        pytest.param(None, "bad_shape", id="layer-shape-mismatch"),
        pytest.param(None, "bad_offset", id="layer-offset-mismatch"),
        pytest.param(None, "nonfinite", id="layer-nonfinite"),
    ],
)
def test_malformed_cache_before_acceptance_falls_back_without_accepting_cache(
    tmp_path, monkeypatch, metadata, cache_layers
):
    serving = _serving_module()
    prompt = "malformed cache prompt"
    prompt_tokens = list(range(300, 300 + 130))
    n_prefix = len(prompt_tokens) - 1
    native = _native_config(serving, tmp_path, request_id="malformed-cache")
    _install_producer_run(monkeypatch, serving)
    observed = {"generate_calls": 0}

    def fake_load_prompt_cache(path, return_metadata=False):
        assert return_metadata is True
        if cache_layers == "bad_shape":
            return _valid_cache(n_prefix, bad_shape=True), _valid_metadata(n_prefix)
        if cache_layers == "bad_offset":
            return _valid_cache(n_prefix, bad_offset=True), _valid_metadata(n_prefix)
        if cache_layers == "nonfinite":
            return _valid_cache(n_prefix, nonfinite=True), _valid_metadata(n_prefix)
        return _valid_cache(n_prefix), metadata

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        observed["generate_prompt"] = _as_int_list(prompt_arg)
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(77)

    result = serving.generate_with_native_prefill(
        "resident-model",
        FakeTokenizer({prompt: prompt_tokens}),
        prompt,
        native=native,
        max_tokens=1,
        generate_step_fn=fake_generate_step,
        load_prompt_cache_fn=fake_load_prompt_cache,
    )

    assert observed["generate_calls"] == 1
    assert result["route"] == "native_mlx_fallback"
    assert "cache" in result["fallback_reason"]
    assert result["accepted_cache"] is False
    assert result["prompt_token_count"] == len(prompt_tokens)
    assert result["prompt_cache_path"] in {None, ""}


def test_generate_failure_after_cache_acceptance_does_not_retry_full_prompt_fallback(tmp_path, monkeypatch):
    serving = _serving_module()
    prompt = "accepted cache prompt"
    prompt_tokens = list(range(400, 400 + 130))
    n_prefix = len(prompt_tokens) - 1
    native = _native_config(
        serving,
        tmp_path,
        request_id="accepted-cache-failure",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    service = _LiveServiceDispatcher(
        tmp_path / "service-artifacts",
        evidence_override={"kernel_count": 3},
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    observed = {"generate_calls": 0}

    def fake_load_prompt_cache(path, return_metadata=False):
        assert return_metadata is True
        assert service.last_metadata is not None
        return _valid_cache(n_prefix), dict(service.last_metadata)

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        observed["generate_prompt"] = _as_int_list(prompt_arg)
        if kwargs.get("prompt_cache") is None:
            raise AssertionError("must not retry native full-prompt generation after cache acceptance")
        assert _as_int_list(prompt_arg) == [prompt_tokens[-1]]
        raise RuntimeError("decode exploded after cache acceptance")
        yield 0
    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "persistent accepted-cache decode must not call subprocess.run"
        ),
    )

    with session:
        try:
            result = serving.generate_with_native_prefill(
                "resident-model",
                FakeTokenizer({prompt: prompt_tokens}),
                prompt,
                native=native,
                max_tokens=1,
                generate_step_fn=fake_generate_step,
                load_prompt_cache_fn=fake_load_prompt_cache,
                service_session=session,
            )
        except serving.NativePrefillError as exc:
            result = getattr(exc, "result", None)
            assert result is not None, "NativePrefillError must carry a result dict after accepted-cache decode failure"
            assert "decode exploded" in result["error"]["message"]
        else:
            assert result["status"] in {"error", "blocked"}
            assert result["exit_status"] != 0
            assert "decode exploded" in result["error"]["message"]

    assert observed["generate_calls"] == 1
    assert result["route"] == "native_producer"
    assert result["accepted_cache"] is True
    assert result["fallback_reason"] is None
    assert result["prompt_token_count"] == len(prompt_tokens)
    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]
    assert service.last_prefill_response is not None
    assert service.last_prefill_response["status"] == "pass"




def test_bad_python_executable_before_acceptance_falls_back_to_native_full_prompt(tmp_path):
    serving = _serving_module()
    prompt = "missing python prompt"
    prompt_tokens = list(range(500, 500 + 130))
    native = _native_config(
        serving,
        tmp_path,
        python_executable=str(tmp_path / "does-not-exist-python"),
        request_id="bad-python",
    )
    observed = {"generate_calls": 0}

    def unexpected_load_prompt_cache(*args, **kwargs):
        raise AssertionError("producer process creation failure must fallback before cache loading")

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(88)

    result = serving.generate_with_native_prefill(
        "resident-model",
        FakeTokenizer({prompt: prompt_tokens}),
        prompt,
        native=native,
        max_tokens=1,
        generate_step_fn=fake_generate_step,
        load_prompt_cache_fn=unexpected_load_prompt_cache,
    )

    assert observed["generate_calls"] == 1
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == "producer_failed"
    assert result["accepted_cache"] is False
    assert result["producer_commands"][0]["returncode"] is None
    assert result["producer_commands"][0]["error"]
    assert result["prompt_cache_path"] in {None, ""}


def test_bad_artifacts_dir_before_acceptance_falls_back_to_native_full_prompt(tmp_path):
    serving = _serving_module()
    prompt = "bad artifacts prompt"
    prompt_tokens = list(range(600, 600 + 130))
    bad_artifacts_dir = tmp_path / "not-a-directory"
    bad_artifacts_dir.write_text("occupied", encoding="utf-8")
    native = _native_config(
        serving,
        tmp_path,
        artifacts_dir=bad_artifacts_dir,
        request_id="bad-artifacts",
    )
    observed = {"generate_calls": 0}

    def unexpected_load_prompt_cache(*args, **kwargs):
        raise AssertionError("artifact path failure must fallback before cache loading")

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(89)

    result = serving.generate_with_native_prefill(
        "resident-model",
        FakeTokenizer({prompt: prompt_tokens}),
        prompt,
        native=native,
        max_tokens=1,
        generate_step_fn=fake_generate_step,
        load_prompt_cache_fn=unexpected_load_prompt_cache,
    )

    assert observed["generate_calls"] == 1
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == "producer_failed"
    assert result["accepted_cache"] is False
    assert result["producer_commands"][0]["returncode"] is None
    assert result["producer_commands"][0]["error"]
    assert result["prompt_cache_path"] in {None, ""}


def test_request_id_with_path_separator_is_rejected_before_artifact_writes(tmp_path, monkeypatch):
    serving = _serving_module()
    artifacts_dir = tmp_path / "artifacts"
    prompt = "traversal prompt"
    prompt_tokens = list(range(700, 700 + 130))
    native = _native_config(serving, artifacts_dir, request_id="../outside/request")
    observed = {"producer_calls": 0}

    def unexpected_run(*args, **kwargs):
        observed["producer_calls"] += 1
        raise AssertionError("unsafe request_id must be rejected before subprocess writes")

    monkeypatch.setattr(serving, "subprocess", SimpleNamespace(run=unexpected_run), raising=False)

    with pytest.raises(serving.NativePrefillError, match="request_id"):
        serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({prompt: prompt_tokens}),
            prompt,
            native=native,
            max_tokens=1,
            generate_step_fn=lambda *args, **kwargs: iter(()),
        )

    assert observed["producer_calls"] == 0
    assert not (tmp_path / "outside").exists()


def test_logged_producer_command_redacts_prompt_token_json(tmp_path, monkeypatch):
    serving = _serving_module()
    prompt = "redaction prompt"
    prompt_tokens = list(range(9000, 9000 + 130))
    n_prefix = len(prompt_tokens) - 1
    native = _native_config(serving, tmp_path, request_id="redaction")
    _install_producer_run(monkeypatch, serving)

    def fake_load_prompt_cache(path, return_metadata=False):
        assert return_metadata is True
        return _valid_cache(n_prefix), _valid_metadata(n_prefix)

    def fake_generate_step(prompt_arg, model, **kwargs):
        assert _as_int_list(prompt_arg) == [prompt_tokens[-1]]
        yield np.int64(90)

    result = serving.generate_with_native_prefill(
        "resident-model",
        FakeTokenizer({prompt: prompt_tokens}),
        prompt,
        native=native,
        max_tokens=1,
        generate_step_fn=fake_generate_step,
        load_prompt_cache_fn=fake_load_prompt_cache,
    )

    producer_command = result["producer_commands"][0]["command"]
    assert "<redacted>" in producer_command
    assert json.dumps(prompt_tokens) not in producer_command
    assert "9000" not in producer_command

def test_main_parses_threshold_timeout_producer_model_and_writes_json_log_report_paths(
    tmp_path, monkeypatch
):
    serving = _serving_module()
    artifacts_dir = tmp_path / "artifacts"
    json_path = tmp_path / "nested" / "result.json"
    log_path = tmp_path / "logs" / "run.log"
    report_path = tmp_path / "path-a-validation-results.md"
    report_path.write_text("# Existing report\n\n## Path A — Baseline\nkeep\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_load_model(model_dir):
        observed["load_model_dir"] = model_dir
        return "resident-model", FakeTokenizer({})

    def fake_generate_with_native_prefill(model, tokenizer, prompt, *, native, max_tokens, **kwargs):
        observed["model"] = model
        observed["tokenizer"] = tokenizer
        observed["prompt"] = prompt
        observed["native"] = native
        observed["max_tokens"] = max_tokens
        observed["kwargs"] = kwargs
        return {
            "schema_version": 1,
            "status": "pass",
            "route": "native_producer",
            "fallback_reason": None,
            "accepted_cache": True,
            "prompt_token_count": len(prompt),
            "decoded_tokens": [42],
            "producer_commands": [{"command": "prefill", "returncode": 0}],
            "prefill_npz_path": "artifacts/request.prefill.npz",
            "prefill_log_path": "artifacts/request.prefill.log",
            "kv_cache_log_path": "artifacts/request.kv-cache.log",
            "prompt_cache_path": "artifacts/request.prompt-cache.safetensors",
            "metadata": {"offset": "3", "num_layers": "16", "n_kv_heads": "8", "head_dim": "64"},
            "exit_status": 0,
        }

    def fake_append_report(path, result):
        observed["report_path"] = Path(path)
        observed["report_result"] = result
        text = Path(path).read_text(encoding="utf-8")
        Path(path).write_text(f"{text}\n{_C2_REPORT_HEADING}\nroute: {result['route']}\n", encoding="utf-8")

    monkeypatch.setattr(serving, "load_model", fake_load_model, raising=False)
    monkeypatch.setattr(serving, "generate_with_native_prefill", fake_generate_with_native_prefill)
    monkeypatch.setattr(serving, "append_or_replace_path_c2_report", fake_append_report)

    rc = serving.main(
        [
            "--model",
            "consumer-model-dir",
            "--producer-model",
            "producer-model-dir",
            "--token-ids-json",
            "[10, 11, 12, 13]",
            "--max-new-tokens",
            "3",
            "--threshold-tokens",
            "144",
            "--producer-timeout-s",
            "9",
            "--artifacts-dir",
            str(artifacts_dir),
            "--json",
            str(json_path),
            "--log",
            str(log_path),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    assert observed["load_model_dir"] == "consumer-model-dir"
    assert observed["model"] == "resident-model"
    assert observed["prompt"] == [10, 11, 12, 13]
    assert observed["max_tokens"] == 3
    native = observed["native"]
    assert native.producer_model_dir == "producer-model-dir"
    assert native.python_executable == _PYTHON
    assert native.threshold_tokens == 144
    assert native.producer_timeout_s == 9
    assert Path(native.artifacts_dir) == artifacts_dir
    result_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert result_json["route"] == "native_producer"
    assert "<redacted>" in result_json["command"]
    assert "[10, 11, 12, 13]" not in result_json["command"]
    log_text = log_path.read_text(encoding="utf-8")
    assert "<redacted>" in log_text
    assert "[10, 11, 12, 13]" not in log_text
    assert "--threshold-tokens 144" in log_text
    assert "producer_model_dir: producer-model-dir" in log_text
    assert "producer_timeout_s: 9" in log_text
    assert "producer_commands:" in log_text
    assert "prefill_npz_path: artifacts/request.prefill.npz" in log_text
    assert "prefill_log_path: artifacts/request.prefill.log" in log_text
    assert "kv_cache_log_path: artifacts/request.kv-cache.log" in log_text
    assert "prompt_cache_path: artifacts/request.prompt-cache.safetensors" in log_text
    assert "metadata:" in log_text
    assert "exit_status: 0" in log_text
    assert observed["report_path"] == report_path
    assert "## Path A — Baseline\nkeep" in report_path.read_text(encoding="utf-8")
    assert _C2_REPORT_HEADING in report_path.read_text(encoding="utf-8")


def test_main_blocks_r9700_native_without_hardware_evidence_and_writes_json_log_report(
    tmp_path, monkeypatch
):
    serving = _serving_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    model_dir = tmp_path / "consumer-model"
    runner_path = tmp_path / "native_r9700_runner"
    artifacts_dir = tmp_path / "service-artifacts"
    json_path = tmp_path / "result.json"
    log_path = tmp_path / "run.log"
    report_path = tmp_path / "path-a-validation-results.md"
    report_path.write_text("# Existing report\n", encoding="utf-8")
    service = _LiveServiceDispatcher(
        artifacts_dir,
        prefill_error=_cache_rejection(
            "native evidence missing",
            "native_evidence_validation",
        ),
    )
    events: list[object] = []

    class Registry:
        dispatch = service.dispatch

        def close(self):
            events.append("registry.close")

    registry = Registry()
    model_digest = _TASK4_MODEL_DIGEST
    observed = {"load_model_calls": 0, "generate_calls": 0}

    def fake_build_registry(*, runner_path, artifact_dir):
        events.append(("build_registry", str(runner_path), str(artifact_dir)))
        return registry

    def fake_verify_model_identity(model_uri, supplied_digest=None):
        events.append(("verify_model_identity", str(model_uri), supplied_digest))
        assert supplied_digest is None
        return SimpleNamespace(
            canonical_uri=str(model_uri),
            digest=model_digest,
            fingerprint=dict(_TASK4_MODEL_FINGERPRINT),
            resident_bytes=4096,
        )

    def fake_load_model(path):
        observed["load_model_calls"] += 1
        return (
            {"model_uri": str(path), "model_digest": model_digest},
            FakeTokenizer({}),
        )

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        assert _as_int_list(prompt_arg) == [10, 11, 12, 13]
        assert kwargs.get("prompt_cache") is None
        yield np.int64(91)

    monkeypatch.setattr(native_worker, "build_registry", fake_build_registry, raising=False)
    monkeypatch.setattr(serving, "native_worker", native_worker, raising=False)
    monkeypatch.setattr(serving, "build_registry", fake_build_registry, raising=False)
    monkeypatch.setattr(serving, "verify_model_identity", fake_verify_model_identity, raising=False)
    monkeypatch.setattr(serving, "load_model", fake_load_model)
    monkeypatch.setattr(serving, "generate_step", fake_generate_step)
    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "r9700_native main must not call one-shot subprocess.run"
        ),
    )

    rc = serving.main(
        [
            "--model",
            str(model_dir),
            "--producer-model",
            str(model_dir),
            "--native-runner",
            str(runner_path),
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--token-ids-json",
            "[10, 11, 12, 13]",
            "--threshold-tokens",
            "2",
            "--max-new-tokens",
            "1",
            "--json",
            str(json_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--log",
            str(log_path),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 2
    assert observed["load_model_calls"] == 1
    assert observed["generate_calls"] == 1
    assert events[:3] == [
        ("verify_model_identity", str(model_dir), None),
        ("verify_model_identity", str(model_dir), None),
        ("build_registry", str(runner_path), str(artifacts_dir)),
    ]
    assert events[-1] == "registry.close"
    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]
    assert service.last_prefill_response is not None
    assert service.last_prefill_response["status"] == "blocked"
    assert service.last_prefill_response["error"] == {
        "domain": "cache_rejection",
        "message": "native evidence missing",
        "failure_stage": "native_evidence_validation",
    }
    result_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert result_json["status"] == "blocked"
    assert result_json["gate_result"] == "blocked"
    assert result_json["route"] == "native_mlx_fallback"
    assert result_json["fallback_reason"] == "cache_validation_failed"
    assert result_json["fallback_detail"] == "native evidence missing"
    assert result_json["requested_producer_kind"] == _R9700_NATIVE_PRODUCER_KIND
    assert result_json["producer_kind"] is None
    assert result_json["accepted_cache"] is False
    assert result_json["native_prefill_acceptance"] is None
    assert result_json["kernel_count"] == 0
    assert result_json["transfer_bytes"] == 0
    assert result_json["exit_status"] == 2
    log_text = log_path.read_text(encoding="utf-8")
    assert "gate_result: blocked" in log_text
    assert "requested_producer_kind: r9700_native" in log_text
    assert "exit_status: 2" in log_text
    report_text = report_path.read_text(encoding="utf-8")
    assert _C2_REPORT_HEADING in report_text
    assert "status: blocked" in report_text


def test_main_runs_all_fixture_prompts_by_default_when_no_prompt_name(tmp_path, monkeypatch):
    serving = _serving_module()
    fixtures_dir = tmp_path / "fixtures"
    _write_prompts_fixture(
        fixtures_dir,
        {
            "prompt-a": {"text": "alpha", "token_ids": [1, 2, 3], "S": 3},
            "prompt-b": {"text": "beta", "token_ids": [4, 5, 6, 7], "S": 4},
        },
    )
    json_path = tmp_path / "result.json"
    log_path = tmp_path / "run.log"
    observed: dict[str, object] = {"prompts": []}

    def fake_load_model(model_dir):
        observed["load_model_dir"] = model_dir
        return "resident-model", FakeTokenizer({})

    def fake_generate_with_native_prefill(model, tokenizer, prompt, *, native, max_tokens, **kwargs):
        observed["prompts"].append(list(prompt))
        observed.setdefault("prompt_names", []).append(kwargs.get("prompt_name"))
        return {
            "schema_version": 1,
            "status": "pass",
            "prompt_name": kwargs.get("prompt_name"),
            "route": "native_mlx_fallback",
            "fallback_reason": "below_threshold",
            "accepted_cache": False,
            "prompt_token_count": len(prompt),
            "decoded_tokens": [len(prompt)],
            "exit_status": 0,
        }

    monkeypatch.setattr(serving, "load_model", fake_load_model, raising=False)
    monkeypatch.setattr(serving, "generate_with_native_prefill", fake_generate_with_native_prefill)

    rc = serving.main(
        [
            "--model",
            "consumer-model-dir",
            "--fixtures-dir",
            str(fixtures_dir),
            "--max-new-tokens",
            "2",
            "--threshold-tokens",
            "128",
            "--producer-timeout-s",
            "300",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--json",
            str(json_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 0
    assert observed["prompts"] == [[1, 2, 3], [4, 5, 6, 7]]
    assert observed["prompt_names"] == ["prompt-a", "prompt-b"]
    result = json.loads(json_path.read_text(encoding="utf-8"))
    assert result["prompt_count"] == 2
    assert [entry["prompt_name"] for entry in result["prompt_results"]] == ["prompt-a", "prompt-b"]
    log_text = log_path.read_text(encoding="utf-8")
    assert "prompt_count: 2" in log_text
    assert "prompt-a" in log_text
    assert "prompt-b" in log_text


def test_main_fixture_suite_fails_when_decoded_tokens_do_not_match_committed_baseline(
    tmp_path, monkeypatch
):
    serving = _serving_module()
    fixtures_dir = tmp_path / "fixtures"
    _write_prompts_fixture(
        fixtures_dir,
        {
            "prompt-a": {"text": "alpha", "token_ids": [1, 2, 3], "S": 3},
            "prompt-b": {"text": "beta", "token_ids": [4, 5, 6, 7], "S": 4},
        },
    )
    _write_baseline_fixture(fixtures_dir, {"prompt-a": [99], "prompt-b": [77]})
    json_path = tmp_path / "result.json"
    log_path = tmp_path / "run.log"

    def fake_load_model(model_dir):
        return "resident-model", FakeTokenizer({})

    def fake_generate_with_native_prefill(model, tokenizer, prompt, *, native, max_tokens, **kwargs):
        prompt_name = kwargs.get("prompt_name")
        return {
            "schema_version": 1,
            "status": "pass",
            "prompt_name": prompt_name,
            "route": "native_mlx_fallback",
            "fallback_reason": "below_threshold",
            "accepted_cache": False,
            "prompt_token_count": len(prompt),
            "S": len(prompt),
            "n_prefix": len(prompt) - 1,
            "decoded_tokens": [1],
            "exit_status": 0,
        }

    monkeypatch.setattr(serving, "load_model", fake_load_model, raising=False)
    monkeypatch.setattr(serving, "generate_with_native_prefill", fake_generate_with_native_prefill)

    rc = serving.main(
        [
            "--model",
            "consumer-model-dir",
            "--fixtures-dir",
            str(fixtures_dir),
            "--max-new-tokens",
            "4",
            "--threshold-tokens",
            "128",
            "--producer-timeout-s",
            "300",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--json",
            str(json_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 1
    result = json.loads(json_path.read_text(encoding="utf-8"))
    assert result["status"] == "fail"
    assert result["gate_result"] == "fail"
    assert [entry["comparison"]["exact_match"] for entry in result["prompt_results"]] == [False, False]
    assert "gate_result: fail" in log_path.read_text(encoding="utf-8")


_TASK4_PRODUCER_FINGERPRINT = "sha256:" + "b" * 64
_TASK4_MODEL_DIGEST = "sha256:" + "c" * 64
_TASK4_MODEL_FINGERPRINT = {
    "model_digest": _TASK4_MODEL_DIGEST,
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


def _task4_metadata(
    n_prefix: int, request_id: str, *, producer_fingerprint: str = _TASK4_PRODUCER_FINGERPRINT
) -> dict[str, object]:
    return {
        "schema_version": "mlx_lm_prompt_cache_v1",
        "producer_fingerprint": producer_fingerprint,
        "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
        "model_digest": _TASK4_MODEL_DIGEST,
        "num_layers": 16,
        "batch": 1,
        "n_kv_heads": 8,
        "offset": n_prefix,
        "sequence_length": n_prefix,
        "head_dim": 64,
        "physical_layout": "B,H,S,D",
        "dtype": "float16",
        "absolute_start_position": 0,
        "absolute_end_position": n_prefix,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
        "cache_class": "KVCache",
        "cache_variant": "llama3.2_1b_fp16",
        "request_id": request_id,
        "meta_state": ["" for _ in range(_EXPECTED_NUM_LAYERS)],
    }


class _LiveServiceDispatcher:
    """A live public-dispatch double; the real registry owns private state."""

    def __init__(
        self,
        artifacts_dir: Path,
        *,
        metadata_override: dict[str, object] | None = None,
        prefill_error: dict[str, object] | None = None,
        evidence_override: dict[str, object] | None = None,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.calls: list[dict[str, object]] = []
        self.responses: list[dict[str, object]] = []
        self.last_prefill_response: dict[str, object] | None = None
        self.last_prefill_result: dict[str, object] | None = None
        self.child_pid = 73_002
        self.launch_count = 1
        self.generation = 19
        self.handle = "mh_" + "d" * 32
        self.metadata_override = metadata_override or {}
        self.prefill_error = prefill_error
        self.evidence_override = evidence_override or {}
        self.last_metadata: dict[str, object] | None = None
        self.model_uri: str | None = None


    def _response(
        self,
        request: dict[str, object],
        *,
        status: str = "pass",
        result: dict[str, object] | None = None,
        error: dict[str, object] | None = None,
        evidence: dict[str, object] | None = None,
    ) -> dict[str, object]:
        response = {
            "protocol_version": "r9700_prefill_service_v1",
            "request_id": request["request_id"],
            "operation": request["operation"],
            "status": status,
            "result": {} if result is None else result,
            "error": error,
            "evidence": evidence,
        }
        self.responses.append(response)
        if request["operation"] == "Prefill":
            self.last_prefill_response = response
            self.last_prefill_result = dict(response["result"])
        return response

    def __call__(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(request)
        operation = request["operation"]
        if operation == "LoadModel":
            self.model_uri = str(request["body"]["model_uri"])
            return self._response(
                request,
                result={
                    "model_handle": self.handle,
                    "model_state": "resident-ready",
                    "model_fingerprint": _TASK4_MODEL_FINGERPRINT,
                    "kernel_pack_digests": ["sha256:" + "e" * 64],
                },
            )
        if operation == "Prefill":
            body = request["body"]
            token_ids = list(body["token_ids"])
            request_id = str(request["request_id"])
            n_prefix = len(token_ids) - 1
            if self.prefill_error is not None:
                return self._response(
                    request,
                    status=str(self.prefill_error.get("status", "error")),
                    error=dict(self.prefill_error["error"]),
                )
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            npz_path = self.artifacts_dir / f"{request_id}.prefill.npz"
            prefill_log_path = self.artifacts_dir / f"{request_id}.prefill.log"
            cache_path = self.artifacts_dir / f"{request_id}.prompt-cache.safetensors"
            cache_log_path = self.artifacts_dir / f"{request_id}.kv-cache.log"
            arrays: dict[str, object] = {
                "model": np.array(self.model_uri),
                "n_prefix": np.array(n_prefix, dtype=np.int64),
                "num_layers": np.array(16, dtype=np.int64),
                "producer_kind": np.array("r9700_native"),
            }
            shape = (1, 8, n_prefix, 64)
            for layer_index in range(16):
                arrays[f"layer{layer_index}_K"] = np.zeros(shape, dtype=np.float16)
                arrays[f"layer{layer_index}_V"] = np.zeros(shape, dtype=np.float16)
            np.savez(npz_path, **arrays)
            prefill_log_path.write_text("native evidence\n", encoding="utf-8")
            cache_path.write_bytes(b"persistent-cache")
            cache_log_path.write_text("cache emitted\n", encoding="utf-8")
            metadata = _task4_metadata(n_prefix, request_id)
            metadata.update(self.metadata_override)
            self.last_metadata = metadata
            cache = {
                "prompt_cache_path": str(cache_path),
                "metadata": metadata,
                "prefill_npz_path": str(npz_path),
                "prefill_log_path": str(prefill_log_path),
                "kv_cache_log_path": str(cache_log_path),
                "payload_digest": "sha256:" + "f" * 64,
                "payload_length_bytes": npz_path.stat().st_size,
            }
            evidence = {
                "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
                "producer_fingerprint": _TASK4_PRODUCER_FINGERPRINT,
                "native_prefill_acceptance": "pass",
                "native_prefill_full_layer_loop_status": "pass",
                "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
                "hardware_log_path": str(prefill_log_path),
                "compute_completion_policy": "terminal",
                "compute_barrier_policy": "full",
                "prefill_npz_path": str(npz_path),
                "kernel_count": 1,
                "transfer_bytes": 4096,
                "block_tokens": n_prefix,
                "block_count": 1,
                "failure_stage": "none",
                "exit_status": 0,
                "failure_text": "none",
            }
            evidence.update(self.evidence_override)
            return self._response(
                request,
                result={
                    "model_handle": self.handle,
                    "request_state": "produced",
                    "prompt_token_count": len(token_ids),
                    "prefix_token_count": n_prefix,
                    "cache": cache,
                },
                evidence=evidence,
            )
        if operation == "UnloadModel":
            return self._response(
                request,
                result={"model_handle": self.handle, "model_state": "unloaded"},
            )
        raise AssertionError(f"unexpected public operation: {operation}")

    def dispatch(self, request: dict[str, object]) -> dict[str, object]:
        return self(request)


def test_persistent_dispatch_routes_load_prefill_unload_and_injects_s_minus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live dispatcher replaces one-shot producer commands and preserves S-1 decode."""
    serving = _serving_module()
    service = _LiveServiceDispatcher(tmp_path / "service-artifacts")
    prompt_tokens = [101, 102, 103, 104]
    native = _native_config(
        serving,
        tmp_path,
        threshold_tokens=2,
        request_id="persistent-request",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    decoded_inputs: list[tuple[list[int], object]] = []

    def no_one_shot(*args, **kwargs):
        pytest.fail("RED: persistent warm Prefill must not call subprocess.run")

    def fake_load_prompt_cache(path, *, return_metadata=False):
        assert Path(path).name == "persistent-request.prompt-cache.safetensors"
        cache = _valid_cache(len(prompt_tokens) - 1)
        metadata = _task4_metadata(len(prompt_tokens) - 1, "persistent-request")
        return (cache, metadata) if return_metadata else cache

    def fake_generate(prompt_arg, model, **kwargs):
        decoded_inputs.append((_as_int_list(prompt_arg), kwargs.get("prompt_cache")))
        yield np.int64(900)

    monkeypatch.setattr(serving.subprocess, "run", no_one_shot)
    with session:
        result = serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({}),
            prompt_tokens,
            native=native,
            max_tokens=2,
            generate_step_fn=fake_generate,
            load_prompt_cache_fn=fake_load_prompt_cache,
            service_session=session,
        )

    operations = [call["operation"] for call in service.calls]
    assert operations == ["LoadModel", "Prefill", "UnloadModel"]
    prefill = service.calls[1]
    assert prefill["body"]["token_ids"] == prompt_tokens
    assert prefill["body"]["token_ids"][:-1] == prompt_tokens[:-1]
    assert result["route"] == "native_producer"
    assert result["accepted_cache"] is True
    assert result["producer_kind"] == _R9700_NATIVE_PRODUCER_KIND
    assert result["metadata"]["producer_fingerprint"] == _TASK4_PRODUCER_FINGERPRINT
    assert result["producer_fingerprint"] == _TASK4_PRODUCER_FINGERPRINT
    assert result["prefill_npz_path"].endswith("persistent-request.prefill.npz")
    assert result["prefill_log_path"].endswith("persistent-request.prefill.log")
    assert result["prompt_cache_path"].endswith(
        "persistent-request.prompt-cache.safetensors"
    )
    assert Path(result["prefill_npz_path"]).parent == tmp_path / "service-artifacts"
    assert decoded_inputs and decoded_inputs[0][0] == [prompt_tokens[-1]]
    assert decoded_inputs[0][1] is not None
    assert service.launch_count == 1


def test_persistent_dispatch_rejects_fingerprint_mismatch_before_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown/mismatched producer identity may fall back only before cache acceptance."""
    serving = _serving_module()
    service = _LiveServiceDispatcher(
        tmp_path / "service-artifacts",
        metadata_override={"producer_fingerprint": "sha256:" + "9" * 64},
    )
    prompt_tokens = [201, 202, 203]
    native = _native_config(
        serving,
        tmp_path,
        threshold_tokens=2,
        request_id="identity-mismatch",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    generate_inputs: list[list[int]] = []

    def fake_generate(prompt_arg, model, **kwargs):
        generate_inputs.append(_as_int_list(prompt_arg))
        yield np.int64(901)

    def fake_load_prompt_cache(path, *, return_metadata=False):
        cache = _valid_cache(len(prompt_tokens) - 1)
        metadata = _task4_metadata(len(prompt_tokens) - 1, "identity-mismatch")
        metadata["producer_fingerprint"] = "sha256:" + "9" * 64
        return (cache, metadata) if return_metadata else cache

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("RED: persistent path used one-shot subprocess"),
    )
    with session:
        result = serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({}),
            prompt_tokens,
            native=native,
            max_tokens=1,
            generate_step_fn=fake_generate,
            load_prompt_cache_fn=fake_load_prompt_cache,
            service_session=session,
        )

    assert result["accepted_cache"] is False
    assert result["route"] == "native_mlx_fallback"
    assert generate_inputs == [prompt_tokens]
    assert len([call for call in service.calls if call["operation"] == "Prefill"]) == 1


def test_persistent_dispatch_child_fault_is_terminal_without_prefix_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private child/device fault cannot silently recompute an accepted prefix."""
    serving = _serving_module()
    service = _LiveServiceDispatcher(
        tmp_path / "service-artifacts",
        prefill_error={
            "status": "error",
            "error": {
                "domain": "device_lost_or_faulted",
                "message": "native resource child became unavailable",
                "failure_stage": "child_eof",
            },
        },
    )
    native = _native_config(
        serving,
        tmp_path,
        threshold_tokens=2,
        request_id="child-fault",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    generate_inputs: list[list[int]] = []

    def fake_generate(prompt_arg, model, **kwargs):
        generate_inputs.append(_as_int_list(prompt_arg))
        yield np.int64(902)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("RED: persistent path used one-shot subprocess"),
    )
    with session:
        with pytest.raises(serving.NativePrefillError):
            serving.generate_with_native_prefill(
                "resident-model",
                FakeTokenizer({}),
                [301, 302, 303],
                native=native,
                max_tokens=1,
                generate_step_fn=fake_generate,
                service_session=session,
            )

    assert generate_inputs == []
    assert len([call for call in service.calls if call["operation"] == "Prefill"]) == 1


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        pytest.param("model_digest", "sha256:" + "8" * 64, id="model-digest"),
        pytest.param("rope_theta", 500001.0, id="rope-theta"),
        pytest.param("absolute_end_position", 1, id="absolute-end-position"),
        pytest.param("sequence_length", 1, id="sequence-length"),
        pytest.param("meta_state", [""] * 15, id="meta-state-cardinality"),
    ],
)
def test_persistent_dispatch_rejects_cache_identity_before_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: object,
) -> None:
    """Model/RoPE/position/meta-state mismatches are adapter rejection, not acceptance."""
    serving = _serving_module()
    service = _LiveServiceDispatcher(
        tmp_path / "service-artifacts",
        metadata_override={field: bad_value},
    )
    prompt_tokens = [401, 402, 403]
    native = _native_config(
        serving,
        tmp_path,
        threshold_tokens=2,
        request_id=f"identity-{field.replace('_', '-')}",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    generate_inputs: list[list[int]] = []

    def fake_generate(prompt_arg, model, **kwargs):
        generate_inputs.append(_as_int_list(prompt_arg))
        yield np.int64(903)

    def fake_load_prompt_cache(path, *, return_metadata=False):
        cache = _valid_cache(len(prompt_tokens) - 1)
        assert service.last_metadata is not None
        metadata = dict(service.last_metadata)
        return (cache, metadata) if return_metadata else cache

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("RED: persistent path used one-shot subprocess"),
    )
    with session:
        result = serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({}),
            prompt_tokens,
            native=native,
            max_tokens=1,
            generate_step_fn=fake_generate,
            load_prompt_cache_fn=fake_load_prompt_cache,
            service_session=session,
        )

    assert result["accepted_cache"] is False
    assert result["route"] == "native_mlx_fallback"
    assert generate_inputs == [prompt_tokens]
    assert result["fallback_reason"] == "cache_validation_failed"
    assert field in result["fallback_detail"]
    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]


def test_persistent_dispatch_decode_failure_is_terminal_after_cache_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once imported cache is accepted, decode failure cannot trigger prefix repair."""
    serving = _serving_module()
    service = _LiveServiceDispatcher(tmp_path / "service-artifacts")
    prompt_tokens = [501, 502, 503]
    native = _native_config(
        serving,
        tmp_path,
        threshold_tokens=2,
        request_id="accepted-decode-failure",
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri="producer-model-dir",
        model_digest=_TASK4_MODEL_DIGEST,
    )
    generate_inputs: list[list[int]] = []

    def fake_load_prompt_cache(path, *, return_metadata=False):
        cache = _valid_cache(len(prompt_tokens) - 1)
        assert service.last_metadata is not None
        return (
            (cache, dict(service.last_metadata))
            if return_metadata
            else cache
        )

    def failing_generate(prompt_arg, model, **kwargs):
        generate_inputs.append(_as_int_list(prompt_arg))
        assert kwargs.get("prompt_cache") is not None
        raise RuntimeError("decode failed after acceptance")
        yield np.int64(904)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("RED: persistent path used one-shot subprocess"),
    )
    with session:
        with pytest.raises(serving.NativePrefillError) as caught:
            serving.generate_with_native_prefill(
                "resident-model",
                FakeTokenizer({}),
                prompt_tokens,
                native=native,
                max_tokens=1,
                generate_step_fn=failing_generate,
                load_prompt_cache_fn=fake_load_prompt_cache,
                service_session=session,
            )

    assert "decode failed after acceptance" in str(caught.value)
    assert generate_inputs == [[prompt_tokens[-1]]]
    assert len([call for call in service.calls if call["operation"] == "Prefill"]) == 1

    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]

class _EmittingServiceDispatcher(_LiveServiceDispatcher):
    """Public dispatcher double that writes an actual mlx-lm cache artifact."""

    def __call__(self, request: dict[str, object]) -> dict[str, object]:
        response = super().__call__(request)
        if request["operation"] != "Prefill" or response["status"] != "pass":
            return response

        from native_r9700 import kv_cache

        body = request["body"]
        n_prefix = len(body["token_ids"]) - 1
        projection = response["result"]["cache"]
        metadata = dict(projection["metadata"])
        shape = (1, _EXPECTED_N_KV_HEADS, n_prefix, _EXPECTED_HEAD_DIM)
        layers = [
            {
                "layer": layer_index,
                "K": np.full(shape, layer_index, dtype=np.float16),
                "V": np.full(shape, layer_index + 1, dtype=np.float16),
            }
            for layer_index in range(_EXPECTED_NUM_LAYERS)
        ]
        kv_cache.emit_prompt_cache(
            {
                "model": "verified-llama",
                "n_prefix": n_prefix,
                "layers": layers,
                "metadata": metadata,
            },
            projection["prompt_cache_path"],
        )
        return response


def _load_emitted_prompt_cache(path: Path, *, return_metadata: bool = False):
    """Use mlx-lm's loader when available, with a byte-level equivalent fallback."""

    try:
        from mlx_lm.models.cache import load_prompt_cache as mlx_load_prompt_cache
    except ImportError:
        mlx_load_prompt_cache = None
    if mlx_load_prompt_cache is not None:
        return mlx_load_prompt_cache(str(path), return_metadata=return_metadata)

    from safetensors import safe_open

    with safe_open(str(path), framework="np") as handle:
        raw_metadata = dict(handle.metadata())
        tensors = {
            key: handle.get_tensor(key)
            for key in handle.keys()
        }

    offset = int(raw_metadata["1.offset"])
    cache = []
    for layer_index in range(_EXPECTED_NUM_LAYERS):
        layer = KVCache(offset, layer_index=layer_index)
        layer.keys = tensors[f"{layer_index}.0"]
        layer.values = tensors[f"{layer_index}.1"]
        layer.state = (layer.keys, layer.values)
        layer.offset = offset
        layer.size = offset
        layer.meta_state = raw_metadata[f"0.{layer_index}"]
        cache.append(layer)

    metadata = {
        key[2:]: value
        for key, value in raw_metadata.items()
        if key.startswith("1.")
    }
    return (cache, metadata) if return_metadata else cache


def test_persistent_prefill_session_reuses_one_loaded_model_for_two_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two warm generations share one explicit LoadModel/UnloadModel lifetime."""

    serving = _serving_module()
    service = _LiveServiceDispatcher(tmp_path / "service-artifacts")
    model_uri = str(tmp_path / "verified-model")
    model = {"model_uri": model_uri, "model_digest": _TASK4_MODEL_DIGEST}
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri=model_uri,
        model_digest=_TASK4_MODEL_DIGEST,
    )
    decode_inputs: list[tuple[list[int], object]] = []

    def fake_load_prompt_cache(path, *, return_metadata=False):
        assert return_metadata is True
        assert Path(path).is_file()
        assert service.last_metadata is not None
        n_prefix = int(service.last_metadata["offset"])
        cache = _valid_cache(n_prefix)
        metadata = dict(service.last_metadata)
        return (cache, metadata) if return_metadata else cache

    def fake_generate(prompt_arg, model_arg, **kwargs):
        decode_inputs.append((_as_int_list(prompt_arg), kwargs.get("prompt_cache")))
        yield np.int64(900 + len(decode_inputs) - 1)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "RED: a loaded PersistentPrefillSession must not use one-shot subprocess.run"
        ),
    )

    prompts = ([101, 102, 103], [201, 202])
    results = []
    with session as loaded_session:
        assert loaded_session is session
        for index, prompt_tokens in enumerate(prompts):
            native = _native_config(
                serving,
                tmp_path,
                threshold_tokens=2,
                request_id=f"warm-session-{index}",
                producer_model_dir=model_uri,
                producer_kind=_R9700_NATIVE_PRODUCER_KIND,
            )
            results.append(
                serving.generate_with_native_prefill(
                    model,
                    FakeTokenizer({}),
                    prompt_tokens,
                    native=native,
                    max_tokens=1,
                    generate_step_fn=fake_generate,
                    load_prompt_cache_fn=fake_load_prompt_cache,
                    service_session=session,
                )
            )

    # __exit__ performs the only unload; an explicit repeated close is idempotent.
    session.close()
    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "Prefill",
        "UnloadModel",
    ]
    assert [call["operation"] for call in service.calls].count("LoadModel") == 1
    assert [call["operation"] for call in service.calls].count("Prefill") == 2
    assert service.calls[0]["body"] == {
        "model_uri": model_uri,
        "model_digest": _TASK4_MODEL_DIGEST,
        "format": "safetensors",
        "quantization": "fp16",
    }
    prefill_calls = service.calls[1:3]
    assert [call["body"]["token_ids"] for call in prefill_calls] == [
        list(prompts[0]),
        list(prompts[1]),
    ]
    assert all(
        call["body"]["model_handle"] == service.handle
        for call in prefill_calls
    )
    assert [result["route"] for result in results] == [
        "native_producer",
        "native_producer",
    ]
    assert [result["decoded_tokens"] for result in results] == [[900], [901]]
    assert [entry[0] for entry in decode_inputs] == [[103], [202]]
    assert all(entry[1] is not None for entry in decode_inputs)
    assert service.launch_count == 1


def test_r9700_native_main_wires_verified_registry_and_session_without_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production r9700_native main owns one registry/session lifecycle."""

    serving = _serving_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    model_dir = tmp_path / "consumer-model"
    runner_path = tmp_path / "native_r9700_runner"
    artifacts_dir = tmp_path / "service-artifacts"
    model_digest = _TASK4_MODEL_DIGEST
    service = _EmittingServiceDispatcher(artifacts_dir)
    events: list[object] = []

    class Registry:
        dispatch = service.dispatch

        def close(self):
            events.append("registry.close")

    registry = Registry()

    def fake_build_registry(*, runner_path, artifact_dir):
        events.append(
            ("build_registry", str(runner_path), str(artifact_dir))
        )
        return registry

    def fake_verify_model_identity(model_uri, supplied_digest=None):
        events.append(("verify_model_identity", str(model_uri), supplied_digest))
        assert supplied_digest is None
        return SimpleNamespace(
            canonical_uri=str(model_uri),
            digest=model_digest,
            fingerprint=dict(_TASK4_MODEL_FINGERPRINT),
            resident_bytes=4096,
        )

    def fake_load_model(path):
        events.append(("load_model", str(path)))
        return (
            {"model_uri": str(path), "model_digest": model_digest},
            FakeTokenizer({}),
        )


    monkeypatch.setattr(
        native_worker,
        "build_registry",
        fake_build_registry,
        raising=False,
    )
    monkeypatch.setattr(serving, "native_worker", native_worker, raising=False)
    monkeypatch.setattr(
        serving,
        "build_registry",
        fake_build_registry,
        raising=False,
    )
    monkeypatch.setattr(
        serving,
        "verify_model_identity",
        fake_verify_model_identity,
        raising=False,
    )
    monkeypatch.setattr(serving, "load_model", fake_load_model)
    monkeypatch.setattr(serving, "load_prompt_cache", _load_emitted_prompt_cache)
    monkeypatch.setattr(
        serving,
        "generate_step",
        lambda prompt, model, **kwargs: iter([np.int64(907)]),
    )
    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "RED: r9700_native production main must not use one-shot subprocess.run"
        ),
    )

    rc = serving.main(
        [
            "--model",
            str(model_dir),
            "--producer-model",
            str(model_dir),
            "--native-runner",
            str(runner_path),
            "--token-ids-json",
            "[11, 12, 13]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--threshold-tokens",
            "2",
            "--max-new-tokens",
            "1",
            "--artifacts-dir",
            str(artifacts_dir),
            "--json",
            str(tmp_path / "result.json"),
            "--log",
            str(tmp_path / "run.log"),
        ]
    )

    assert rc == 0
    build_events = [event for event in events if isinstance(event, tuple) and event[0] == "build_registry"]
    assert build_events == [
        ("build_registry", str(runner_path), str(artifacts_dir))
    ]
    verify_events = [
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "verify_model_identity"
    ]
    assert verify_events == [
        ("verify_model_identity", str(model_dir), None),
        ("verify_model_identity", str(model_dir), None),
    ]
    load_call = next(call for call in service.calls if call["operation"] == "LoadModel")
    assert load_call["body"]["model_digest"] == model_digest
    assert load_call["body"]["model_digest"] != ""
    assert load_call["body"]["model_digest"].startswith("sha256:")
    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]
    assert events[-1] == "registry.close"


def test_persistent_serving_consumes_emitted_cache_state_and_validates_reconstructed_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An emitted file reconstructs 16 KVCache layers; returned metadata is 1.* only."""

    serving = _serving_module()
    service = _EmittingServiceDispatcher(tmp_path / "service-artifacts")
    model_uri = str(tmp_path / "verified-model")
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri=model_uri,
        model_digest=_TASK4_MODEL_DIGEST,
    )
    prompt_tokens = [301, 302, 303]
    decode_inputs: list[tuple[list[int], object]] = []

    def fake_generate(prompt_arg, model, **kwargs):
        decode_inputs.append((_as_int_list(prompt_arg), kwargs.get("prompt_cache")))
        yield np.int64(905)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "RED: emitted-cache session must not use one-shot subprocess.run"
        ),
    )

    with session:
        result = serving.generate_with_native_prefill(
            {"model_uri": model_uri, "model_digest": _TASK4_MODEL_DIGEST},
            FakeTokenizer({}),
            prompt_tokens,
            native=_native_config(
                serving,
                tmp_path,
                threshold_tokens=2,
                request_id="emitted-cache",
                producer_model_dir=model_uri,
                producer_kind=_R9700_NATIVE_PRODUCER_KIND,
            ),
            max_tokens=1,
            generate_step_fn=fake_generate,
            load_prompt_cache_fn=_load_emitted_prompt_cache,
            service_session=session,
        )

    assert result["accepted_cache"] is True
    assert result["route"] == "native_producer"
    from safetensors import safe_open

    with safe_open(str(result["prompt_cache_path"]), framework="np") as handle:
        emitted_metadata = dict(handle.metadata())
    assert {f"0.{index}" for index in range(_EXPECTED_NUM_LAYERS)} <= set(
        emitted_metadata
    )
    assert all(
        emitted_metadata[f"0.{index}"] == ""
        for index in range(_EXPECTED_NUM_LAYERS)
    )
    assert emitted_metadata["1.offset"] == "2"
    assert emitted_metadata["1.num_layers"] == "16"
    returned_metadata = result["metadata"]
    assert "meta_state" not in returned_metadata
    assert not any(
        str(key).startswith(("0.", "2."))
        for key in returned_metadata
    )
    assert returned_metadata.get("offset", returned_metadata.get("1.offset")) == "2"
    assert returned_metadata.get(
        "num_layers", returned_metadata.get("1.num_layers")
    ) == "16"
    assert len(decode_inputs) == 1
    assert decode_inputs[0][0] == [prompt_tokens[-1]]
    reconstructed_cache = decode_inputs[0][1]
    assert reconstructed_cache is not None
    assert len(reconstructed_cache) == _EXPECTED_NUM_LAYERS
    for layer in reconstructed_cache:
        assert type(layer).__name__ == "KVCache"
        assert tuple(np.asarray(layer.keys).shape) == (
            1,
            _EXPECTED_N_KV_HEADS,
            len(prompt_tokens) - 1,
            _EXPECTED_HEAD_DIM,
        )
        assert tuple(np.asarray(layer.values).shape) == tuple(
            np.asarray(layer.keys).shape
        )
        assert int(layer.offset) == len(prompt_tokens) - 1
        size = layer.size() if callable(getattr(layer, "size", None)) else layer.size
        assert int(size) == len(prompt_tokens) - 1
        assert getattr(layer, "meta_state", "") == ""


def test_persistent_serving_accepts_single_token_zero_prefix_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S=1 is a valid native request with an empty N=0 prefix cache."""

    serving = _serving_module()
    service = _LiveServiceDispatcher(
        tmp_path / "service-artifacts",
        evidence_override={
            "kernel_count": 0,
            "transfer_bytes": 0,
            "block_tokens": 0,
            "block_count": 0,
        },
    )
    model_uri = str(tmp_path / "verified-model")
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri=model_uri,
        model_digest=_TASK4_MODEL_DIGEST,
    )
    decode_inputs: list[list[int]] = []

    def fake_load_prompt_cache(path, *, return_metadata=False):
        assert return_metadata is True
        assert service.last_metadata is not None
        cache = _valid_cache(0)
        metadata = dict(service.last_metadata)
        return (cache, metadata) if return_metadata else cache

    def fake_generate(prompt_arg, model, **kwargs):
        decode_inputs.append(_as_int_list(prompt_arg))
        assert kwargs.get("prompt_cache") is not None
        yield np.int64(906)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "RED: S=1 native serving must not use one-shot subprocess.run"
        ),
    )

    with session:
        result = serving.generate_with_native_prefill(
            {"model_uri": model_uri, "model_digest": _TASK4_MODEL_DIGEST},
            FakeTokenizer({}),
            [777],
            native=_native_config(
                serving,
                tmp_path,
                threshold_tokens=1,
                request_id="zero-prefix",
                producer_model_dir=model_uri,
                producer_kind=_R9700_NATIVE_PRODUCER_KIND,
            ),
            max_tokens=1,
            generate_step_fn=fake_generate,
            load_prompt_cache_fn=fake_load_prompt_cache,
            service_session=session,
        )

    assert result["accepted_cache"] is True
    assert result["route"] == "native_producer"
    assert result["n_prefix"] == 0
    assert result["decoded_tokens"] == [906]
    assert decode_inputs == [[777]]
    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]
    assert service.calls[1]["body"]["token_ids"] == [777]
    assert service.last_prefill_result is not None
    assert service.last_prefill_result["prefix_token_count"] == 0
    for layer in _valid_cache(0):
        assert tuple(layer.keys.shape) == (1, _EXPECTED_N_KV_HEADS, 0, _EXPECTED_HEAD_DIM)
 
 
def test_r9700_native_main_rejects_consumer_producer_model_digest_mismatch_before_load_or_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native main verifies both model identities before loading either model."""

    serving = _serving_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    model_dir = tmp_path / "consumer-model"
    producer_model_dir = tmp_path / "producer-model"
    consumer_digest = "sha256:" + "a" * 64
    producer_digest = "sha256:" + "b" * 64
    events: list[tuple[object, ...]] = []

    def fake_verify_model_identity(model_uri, supplied_digest=None):
        path = str(model_uri)
        events.append(("verify_model_identity", path, supplied_digest))
        digest = consumer_digest if path == str(model_dir) else producer_digest
        fingerprint = dict(_TASK4_MODEL_FINGERPRINT)
        fingerprint["model_digest"] = digest
        return SimpleNamespace(
            canonical_uri=path,
            digest=digest,
            fingerprint=fingerprint,
            resident_bytes=4096,
        )

    def fail_load_model(path):
        events.append(("load_model", str(path)))
        pytest.fail("digest mismatch must be rejected before loading the consumer model")

    def fail_build_registry(*, runner_path, artifact_dir):
        events.append(("build_registry", str(runner_path), str(artifact_dir)))
        pytest.fail("digest mismatch must be rejected before constructing a session/registry")

    monkeypatch.setattr(serving, "verify_model_identity", fake_verify_model_identity)
    monkeypatch.setattr(native_worker, "verify_model_identity", fake_verify_model_identity, raising=False)
    monkeypatch.setattr(serving, "load_model", fail_load_model)
    monkeypatch.setattr(serving, "build_registry", fail_build_registry, raising=False)
    monkeypatch.setattr(native_worker, "build_registry", fail_build_registry, raising=False)

    json_path = tmp_path / "digest-mismatch.json"
    log_path = tmp_path / "digest-mismatch.log"
    rc = serving.main(
        [
            "--model",
            str(model_dir),
            "--producer-model",
            str(producer_model_dir),
            "--native-runner",
            str(tmp_path / "native_r9700_runner"),
            "--token-ids-json",
            "[11, 12, 13]",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--threshold-tokens",
            "2",
            "--max-new-tokens",
            "1",
            "--artifacts-dir",
            str(tmp_path / "service-artifacts"),
            "--json",
            str(json_path),
            "--log",
            str(log_path),
        ]
    )

    assert rc == 2
    assert [event[0] for event in events] == [
        "verify_model_identity",
        "verify_model_identity",
    ]
    assert {event[1] for event in events} == {
        str(model_dir),
        str(producer_model_dir),
    }
    assert not any(event[0] in {"load_model", "build_registry"} for event in events)
    result = json.loads(json_path.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["exit_status"] == 2
    assert "digest" in json.dumps(result["error"]).lower()


class _PersistentEvidenceMutationDispatcher(_LiveServiceDispatcher):
    """Persistent service double with one isolated native-evidence mutation."""

    def __init__(
        self,
        artifacts_dir: Path,
        *,
        model_uri: str,
        mutation: str,
    ) -> None:
        super().__init__(artifacts_dir)
        self.model_uri = model_uri
        self.mutation = mutation

    def _write_valid_npz(self, path: Path, n_prefix: int) -> None:
        shape = (1, _EXPECTED_N_KV_HEADS, n_prefix, _EXPECTED_HEAD_DIM)
        arrays: dict[str, np.ndarray] = {
            "model": np.asarray(self.model_uri),
            "producer_kind": np.asarray(_R9700_NATIVE_PRODUCER_KIND),
            "num_layers": np.asarray(_EXPECTED_NUM_LAYERS, dtype=np.int64),
            "n_prefix": np.asarray(n_prefix, dtype=np.int64),
        }
        for layer_index in range(_EXPECTED_NUM_LAYERS):
            arrays[f"layer{layer_index}_K"] = np.zeros(shape, dtype=np.float16)
            arrays[f"layer{layer_index}_V"] = np.ones(shape, dtype=np.float16)
        np.savez(path, **arrays)

    def __call__(self, request: dict[str, object]) -> dict[str, object]:
        response = super().__call__(request)
        if request["operation"] != "Prefill" or response["status"] != "pass":
            return response

        result = response["result"]
        evidence = response["evidence"]
        assert isinstance(result, dict)
        assert isinstance(evidence, dict)
        cache = result["cache"]
        assert isinstance(cache, dict)
        npz_path = Path(cache["prefill_npz_path"])
        prefill_log_path = Path(cache["prefill_log_path"])
        n_prefix = len(request["body"]["token_ids"]) - 1
        self._write_valid_npz(npz_path, n_prefix)

        if self.mutation == "blocked-acceptance":
            evidence["native_prefill_acceptance"] = "blocked"
        elif self.mutation == "failed-full-layer":
            evidence["native_prefill_full_layer_loop_status"] = "blocked"
        elif self.mutation == "wrong-substrate":
            evidence["runtime_substrate"] = "Linux/ROCm/HIP"
        elif self.mutation == "wrong-completion-policy":
            evidence["compute_completion_policy"] = "best-effort"
        elif self.mutation == "wrong-barrier-policy":
            evidence["compute_barrier_policy"] = "none"
        elif self.mutation == "failure-stage":
            evidence["failure_stage"] = "native_prefill_failed"
        elif self.mutation == "failure-text":
            evidence["failure_text"] = "native prefill failed"
        elif self.mutation == "nonzero-exit":
            evidence["exit_status"] = 1
        elif self.mutation == "missing-hardware-log":
            evidence.pop("hardware_log_path", None)
        elif self.mutation == "unreadable-hardware-log":
            prefill_log_path.write_bytes(b"\xff\xfe\xfd")
        elif self.mutation == "unbound-hardware-log":
            other_log = self.artifacts_dir / "other-request.prefill.log"
            other_log.write_text("other request\n", encoding="utf-8")
            evidence["hardware_log_path"] = str(other_log)
        elif self.mutation == "missing-npz":
            evidence.pop("prefill_npz_path", None)
        elif self.mutation == "invalid-npz":
            npz_path.write_bytes(b"not-a-valid-native-prefill-npz")
        elif self.mutation == "zero-kernel":
            evidence["kernel_count"] = 0
        elif self.mutation == "zero-transfer":
            evidence["transfer_bytes"] = 0
        elif self.mutation == "zero-prefix":
            evidence["kernel_count"] = 0
            evidence["transfer_bytes"] = 0
            evidence["block_tokens"] = 0
            evidence["block_count"] = 0
        else:
            raise AssertionError(f"unknown evidence mutation: {self.mutation}")
        return response


_PERSISTENT_EVIDENCE_MUTATIONS = (
    pytest.param("blocked-acceptance", id="blocked-acceptance"),
    pytest.param("failed-full-layer", id="failed-full-layer"),
    pytest.param("wrong-substrate", id="wrong-substrate"),
    pytest.param("wrong-completion-policy", id="wrong-completion-policy"),
    pytest.param("wrong-barrier-policy", id="wrong-barrier-policy"),
    pytest.param("failure-stage", id="nonempty-failure-stage"),
    pytest.param("failure-text", id="nonempty-failure-text"),
    pytest.param("nonzero-exit", id="nonzero-exit"),
    pytest.param("missing-hardware-log", id="missing-hardware-log"),
    pytest.param("unreadable-hardware-log", id="unreadable-hardware-log"),
    pytest.param("unbound-hardware-log", id="unbound-hardware-log"),
    pytest.param("missing-npz", id="missing-npz"),
    pytest.param("invalid-npz", id="invalid-npz"),
    pytest.param("zero-kernel", id="zero-kernel-on-positive-prefix"),
    pytest.param("zero-transfer", id="zero-transfer-on-positive-prefix"),
)


@pytest.mark.parametrize("mutation", _PERSISTENT_EVIDENCE_MUTATIONS)
def test_persistent_prefill_rejects_semantically_invalid_evidence_before_cache_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Every failed native-evidence gate stays on the pre-acceptance boundary."""

    serving = _serving_module()
    model_uri = str(tmp_path / "verified-model")
    service = _PersistentEvidenceMutationDispatcher(
        tmp_path / "service-artifacts",
        model_uri=model_uri,
        mutation=mutation,
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri=model_uri,
        model_digest=_TASK4_MODEL_DIGEST,
    )
    prompt_tokens = [601, 602, 603]
    native = _native_config(
        serving,
        tmp_path,
        threshold_tokens=2,
        request_id=f"evidence-{mutation}",
        producer_model_dir=model_uri,
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )
    generated: list[tuple[list[int], object]] = []

    def fake_load_prompt_cache(path, *, return_metadata=False):
        assert return_metadata is True
        assert service.last_metadata is not None
        cache = _valid_cache(len(prompt_tokens) - 1)
        metadata = dict(service.last_metadata)
        return (cache, metadata) if return_metadata else cache

    def fake_generate(prompt_arg, model, **kwargs):
        generated.append((_as_int_list(prompt_arg), kwargs.get("prompt_cache")))
        yield np.int64(910)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "persistent evidence rejection must not use one-shot subprocess"
        ),
    )
    with session:
        try:
            result = serving.generate_with_native_prefill(
                "resident-model",
                FakeTokenizer({}),
                prompt_tokens,
                native=native,
                max_tokens=1,
                generate_step_fn=fake_generate,
                load_prompt_cache_fn=fake_load_prompt_cache,
                service_session=session,
            )
        except serving.NativePrefillError as exc:
            assert exc.result is not None, "pre-acceptance errors must retain a result"
            result = exc.result

    assert result["accepted_cache"] is False
    assert result.get("prompt_cache_path") in {None, ""}
    assert result["route"] in {"native_mlx_fallback", "native_producer"}
    assert result["status"] in {"blocked", "error"}
    if result["route"] == "native_mlx_fallback":
        assert result["status"] == "blocked"
        assert result["exit_status"] == 2
        assert result["fallback_reason"] == "cache_validation_failed"
        assert generated == [(prompt_tokens, None)]
    else:
        assert result["status"] == "error"
        assert result["exit_status"] == 1
        assert generated == []


def test_persistent_prefill_accepts_declared_zero_prefix_without_positive_work_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N=0 may report no work while retaining complete accepted evidence."""

    serving = _serving_module()
    model_uri = str(tmp_path / "verified-model")
    service = _PersistentEvidenceMutationDispatcher(
        tmp_path / "service-artifacts",
        model_uri=model_uri,
        mutation="zero-prefix",
    )
    session = serving.PersistentPrefillSession(
        service.dispatch,
        model_uri=model_uri,
        model_digest=_TASK4_MODEL_DIGEST,
    )
    generated: list[tuple[list[int], object]] = []

    def fake_load_prompt_cache(path, *, return_metadata=False):
        assert return_metadata is True
        assert service.last_metadata is not None
        cache = _valid_cache(0)
        metadata = dict(service.last_metadata)
        return (cache, metadata) if return_metadata else cache

    def fake_generate(prompt_arg, model, **kwargs):
        generated.append((_as_int_list(prompt_arg), kwargs.get("prompt_cache")))
        yield np.int64(911)

    monkeypatch.setattr(
        serving.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "zero-prefix persistent serving must not use one-shot subprocess"
        ),
    )
    with session:
        result = serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({}),
            [777],
            native=_native_config(
                serving,
                tmp_path,
                threshold_tokens=1,
                request_id="zero-prefix-evidence",
                producer_model_dir=model_uri,
                producer_kind=_R9700_NATIVE_PRODUCER_KIND,
            ),
            max_tokens=1,
            generate_step_fn=fake_generate,
            load_prompt_cache_fn=fake_load_prompt_cache,
            service_session=session,
        )

    assert result["accepted_cache"] is True
    assert result["route"] == "native_producer"
    assert result["status"] == "pass"
    assert result["n_prefix"] == 0
    assert result["kernel_count"] == 0
    assert result["transfer_bytes"] == 0
    assert service.last_prefill_response is not None
    evidence = service.last_prefill_response["evidence"]
    assert evidence["native_prefill_acceptance"] == "pass"
    assert evidence["native_prefill_full_layer_loop_status"] == "pass"
    assert evidence["runtime_substrate"] == "TinyGPU.app/APLRemotePCIDevice/PCIIface"
    assert evidence["compute_completion_policy"] == "terminal"
    assert evidence["compute_barrier_policy"] == "full"
    assert evidence["block_tokens"] == 0
    assert evidence["block_count"] == 0
    assert evidence["failure_stage"] == "none"
    assert evidence["failure_text"] == "none"
    assert generated and generated[0][0] == [777]
    assert generated[0][1] is not None
    assert [call["operation"] for call in service.calls] == [
        "LoadModel",
        "Prefill",
        "UnloadModel",
    ]
