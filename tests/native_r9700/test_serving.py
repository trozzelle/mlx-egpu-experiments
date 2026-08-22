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

    def __init__(self, n_prefix: int, *, layer_index: int, bad_shape: bool = False, bad_offset: bool = False):
        shape = (1, _EXPECTED_N_KV_HEADS, n_prefix, _EXPECTED_HEAD_DIM)
        if bad_shape and layer_index == 0:
            shape = (1, _EXPECTED_N_KV_HEADS - 1, n_prefix, _EXPECTED_HEAD_DIM)
        self.keys = np.zeros(shape, dtype=np.float16)
        self.values = np.ones(shape, dtype=np.float16)
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
    producer_calls = _install_producer_run(monkeypatch, serving)
    observed = {"generate_calls": 0}

    def fake_generate(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(900)

    result = serving.generate_with_native_prefill(
        "resident-model",
        FakeTokenizer({prompt: prompt_tokens}),
        prompt,
        native=native,
        max_tokens=1,
        generate_step_fn=fake_generate,
    )

    assert len(producer_calls) == 1
    assert observed["generate_calls"] == 1
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == "native_evidence_missing"
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
    producer_calls = _install_producer_run(monkeypatch, serving, native_evidence=True)
    cache_objects = _valid_cache(n_prefix)

    def fake_load_prompt_cache(path, return_metadata=False):
        assert return_metadata is True
        return cache_objects, _valid_metadata(n_prefix)

    def fake_generate_step(prompt_arg, model, **kwargs):
        assert _as_int_list(prompt_arg) == [prompt_tokens[-1]]
        assert kwargs["prompt_cache"] is cache_objects
        yield np.int64(601)

    result = serving.generate_with_native_prefill(
        "resident-model",
        FakeTokenizer({prompt: prompt_tokens}),
        prompt,
        native=native,
        max_tokens=1,
        generate_step_fn=fake_generate_step,
        load_prompt_cache_fn=fake_load_prompt_cache,
    )

    assert len(producer_calls) == 2
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
    observed = {"producer_calls": 0, "generate_calls": 0}

    def fake_run(cmd, **kwargs):
        observed["producer_calls"] += 1
        assert cmd[2] == "native_r9700.prefill"
        prefill_path = Path(cmd[cmd.index("--out") + 1])
        prefill_log_path = Path(cmd[cmd.index("--log") + 1])
        hardware_log_path = tmp_path / f"{case}.hardware.log"
        prefill_path.parent.mkdir(parents=True, exist_ok=True)
        prefill_path.write_bytes(b"prefill-npz")
        if case == "unbound":
            hardware_log_path.write_text("stale hardware evidence\n", encoding="utf-8")
        prefill_log_path.write_text(
            "\n".join(
                (
                    "producer_kind: r9700_native",
                    "native_prefill_acceptance: pass",
                    f"hardware_log_path: {hardware_log_path}",
                    f"prefill_npz_path: {prefill_path}",
                    "kernel_count: 3",
                    "transfer_bytes: 4096",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return _completed(cmd, stdout="prefill ok")

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(701)

    monkeypatch.setattr(serving, "subprocess", SimpleNamespace(run=fake_run), raising=False)
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
    )

    assert observed == {"producer_calls": 1, "generate_calls": 1}
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == "native_evidence_missing"
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
    observed = {"producer_calls": 0, "generate_calls": 0}

    def fake_run(cmd, **kwargs):
        observed["producer_calls"] += 1
        assert cmd[2] == "native_r9700.prefill"
        prefill_path = Path(cmd[cmd.index("--out") + 1])
        prefill_log_path = Path(cmd[cmd.index("--log") + 1])
        prefill_path.parent.mkdir(parents=True, exist_ok=True)
        prefill_path.write_bytes(b"prefill-npz")
        prefill_log_path.write_text(
            "\n".join(
                (
                    "producer_kind: r9700_native",
                    "native_prefill_acceptance: pass",
                    f"hardware_log_path: {prefill_log_path}",
                    f"prefill_npz_path: {prefill_path}",
                    "kernel_count: 3",
                    "transfer_bytes: 4096",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return _completed(cmd, stdout="prefill ok")

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        assert _as_int_list(prompt_arg) == prompt_tokens
        assert kwargs.get("prompt_cache") is None
        yield np.int64(702)

    original_read_text = Path.read_text

    def unreadable_read_text(path, *args, **kwargs):
        if path.name == "unreadable-hardware-log.prefill.log":
            raise PermissionError("hardware log is unreadable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(serving, "subprocess", SimpleNamespace(run=fake_run), raising=False)
    monkeypatch.setattr(Path, "read_text", unreadable_read_text)
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
    )

    assert observed == {"producer_calls": 1, "generate_calls": 1}
    assert result["route"] == "native_mlx_fallback"
    assert result["fallback_reason"] == "native_evidence_missing"
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

    with pytest.raises(serving.NativePrefillError, match="unsigned 32-bit integer"):
        serving.generate_with_native_prefill(
            "resident-model",
            None,
            token_ids,
            native=native,
            generate_step_fn=lambda *args, **kwargs: pytest.fail(
                "invalid native tokens must not use fallback generation"
            ),
        )

    assert not list(tmp_path.glob("*.prefill.npz"))
    assert not list(tmp_path.glob("*.prompt-cache.safetensors"))


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
    _install_producer_run(monkeypatch, serving, native_evidence=True)
    observed = {"generate_calls": 0}

    def fake_load_prompt_cache(path, return_metadata=False):
        assert return_metadata is True
        return _valid_cache(n_prefix), _valid_metadata(n_prefix)

    def fake_generate_step(prompt_arg, model, **kwargs):
        observed["generate_calls"] += 1
        observed["generate_prompt"] = _as_int_list(prompt_arg)
        if kwargs.get("prompt_cache") is None:
            raise AssertionError("must not retry native full-prompt generation after cache acceptance")
        assert _as_int_list(prompt_arg) == [prompt_tokens[-1]]
        raise RuntimeError("decode exploded after cache acceptance")
        yield 0

    try:
        result = serving.generate_with_native_prefill(
            "resident-model",
            FakeTokenizer({prompt: prompt_tokens}),
            prompt,
            native=native,
            max_tokens=1,
            generate_step_fn=fake_generate_step,
            load_prompt_cache_fn=fake_load_prompt_cache,
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
    json_path = tmp_path / "result.json"
    log_path = tmp_path / "run.log"
    report_path = tmp_path / "path-a-validation-results.md"
    report_path.write_text("# Existing report\n", encoding="utf-8")
    observed = {"load_model_calls": 0}

    def fake_load_model(model_dir):
        observed["load_model_calls"] += 1
        return "resident-model", FakeTokenizer({})

    def fake_generate_with_native_prefill(model, tokenizer, prompt, *, native, max_tokens, **kwargs):
        assert native.producer_kind == _R9700_NATIVE_PRODUCER_KIND
        return {
            "schema_version": 1,
            "status": "blocked",
            "route": "native_mlx_fallback",
            "fallback_reason": "native_evidence_missing",
            "accepted_cache": False,
            "prompt_token_count": len(prompt),
            "S": len(prompt),
            "n_prefix": len(prompt) - 1,
            "decoded_tokens": [91],
            "requested_producer_kind": _R9700_NATIVE_PRODUCER_KIND,
            "producer_kind": None,
            "native_prefill_acceptance": None,
            "hardware_log_path": None,
            "kernel_count": 0,
            "transfer_bytes": 0,
            "exit_status": 2,
        }

    monkeypatch.setattr(serving, "load_model", fake_load_model, raising=False)
    monkeypatch.setattr(serving, "generate_with_native_prefill", fake_generate_with_native_prefill)

    rc = serving.main(
        [
            "--model",
            "consumer-model-dir",
            "--producer-kind",
            _R9700_NATIVE_PRODUCER_KIND,
            "--token-ids-json",
            "[10, 11, 12, 13]",
            "--json",
            str(json_path),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--log",
            str(log_path),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 2
    assert observed["load_model_calls"] == 1
    result_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert result_json["status"] == "blocked"
    assert result_json["gate_result"] == "blocked"
    assert result_json["route"] == "native_mlx_fallback"
    assert result_json["fallback_reason"] == "native_evidence_missing"
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
