"""C1 task set 9 RED contract for native/R token parity harness.

These tests define the future ``native_r9700.parity`` public API before the
production harness lands. The module is imported lazily so pytest collection
succeeds; the expected RED is a clear missing module/API failure, not a syntax
or collection-time import error.

Contract: load the committed prompt/R fixtures, run native S-1 prefill/cache
emission, inject the prompt cache into mlx-lm final-token decode, compare P/R
tokens exactly, write JSON/log/report artifacts, and keep Qwen/C2/C++ runtime
and semantic-equivalence fallback out of this C1 Llama ladder.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "native_r9700" / "fixtures"
_PYTHON = "${HOME}/.pyenv/versions/3.12.8/bin/python3"
_LLAMA_MLX_MODEL_DIR = (
    _REPO_ROOT
    / ".."
    / "tinygrad-kv-worker-phase0"
    / "mlx_models"
    / "meta-Llama-3.2-1B-Instruct"
).resolve()
_PATH_C_HEADING = "## Path C – C1 CPU reference / prompt-cache ABI results (reclassified)"
_PROMPT0_TOKENS = [128000, 791, 6864, 315, 9822, 374]
_BASELINE_R_TOKENS = {
    "prompt-0": [12366, 13, 578, 469],
    "prompt-1": [128009, 128006, 78191, 271],
    "prompt-2": [128009, 128006, 128006, 128006],
}
_CPU_REFERENCE_PRODUCER_KIND = "cpu_reference"
_R9700_NATIVE_PRODUCER_KIND = "r9700_native"


_PARITY_PUBLIC_API = (
    "ParityError",
    "PromptCase",
    "load_prompt_cases",
    "load_fixture_r_tokens",
    "compare_tokens",
    "run_parity_suite",
    "write_result_json",
    "append_or_replace_path_c_report",
    "main",
)


def _parity_module():
    try:
        module = importlib.import_module("native_r9700.parity")
    except ModuleNotFoundError as exc:
        if exc.name == "native_r9700.parity":
            pytest.fail(
                "native_r9700.parity module missing; implement the C1 task "
                "set 9 native-vs-mlx token parity harness API"
            )
        raise

    missing = [name for name in _PARITY_PUBLIC_API if not hasattr(module, name)]
    assert not missing, f"native_r9700.parity missing public APIs: {missing}"
    for name in _PARITY_PUBLIC_API:
        value = getattr(module, name)
        if name not in ("ParityError", "PromptCase"):
            assert callable(value), f"native_r9700.parity.{name} must be callable"
    return module


def _write_prompts_fixture(fixtures_dir: Path, prompts: dict) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")


def _write_baseline_fixture(fixtures_dir: Path, payload: dict) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / "baseline_r_tokens.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _case_by_name(cases, name: str):
    for case in cases:
        if case.name == name:
            return case
    raise AssertionError(f"missing prompt case {name!r}")


def _prompt_result(result: dict, name: str) -> dict:
    for prompt_result in result["prompt_results"]:
        if prompt_result["prompt_name"] == name:
            return prompt_result
    raise AssertionError(f"missing result for {name!r}: {result}")


def _as_int_list(tokens) -> list[int]:
    return [int(token) for token in tokens]


def _native_decode_evidence(
    tmp_path: Path,
    *,
    p_tokens: list[int],
    producer_kind: str = _R9700_NATIVE_PRODUCER_KIND,
    failure_stage: str = "",
    failure_text: str = "",
) -> dict:
    """Return accepted native-boundary evidence with real local artifact paths."""
    hardware_log_path = tmp_path / "native-prefill.hardware.log"
    prompt_cache_path = tmp_path / "native-prompt-cache.safetensors"
    hardware_log_path.write_text("native hardware evidence\n", encoding="utf-8")
    prompt_cache_path.write_bytes(b"native prompt-cache evidence")
    return {
        "p_tokens": list(p_tokens),
        "producer_kind": producer_kind,
        "native_prefill_acceptance": "pass",
        "hardware_log_path": str(hardware_log_path),
        "prompt_cache_path": str(prompt_cache_path),
        "kernel_count": 2,
        "transfer_bytes": 2048,
        "failure_stage": failure_stage,
        "failure_text": failure_text,
    }


def test_parity_module_exports_public_api():
    parity = _parity_module()

    assert issubclass(parity.ParityError, Exception)
    assert parity.PromptCase is not None
    assert callable(parity.load_prompt_cases)
    assert callable(parity.load_fixture_r_tokens)
    assert callable(parity.compare_tokens)
    assert callable(parity.run_parity_suite)
    assert callable(parity.write_result_json)
    assert callable(parity.append_or_replace_path_c_report)
    assert callable(parity.main)


def test_load_prompt_cases_preserves_fixture_order_and_s_minus_one_split():
    parity = _parity_module()

    cases = parity.load_prompt_cases(_FIXTURE_DIR)

    assert [case.name for case in cases] == ["prompt-0", "prompt-1", "prompt-2"]
    assert [case.S for case in cases] == [6, 222, 661]
    for case in cases:
        assert isinstance(case, parity.PromptCase)
        assert len(case.token_ids) == case.S
        assert case.n_prefix == case.S - 1
        assert case.prefix_token_ids == case.token_ids[:-1]
        assert case.final_token_id == case.token_ids[-1]

    prompt0 = _case_by_name(cases, "prompt-0")
    assert prompt0.token_ids == _PROMPT0_TOKENS
    assert prompt0.S == 6
    assert prompt0.n_prefix == 5
    assert prompt0.prefix_token_ids == [128000, 791, 6864, 315, 9822]
    assert prompt0.final_token_id == 374


def test_load_prompt_cases_rejects_s_shorter_than_two(tmp_path):
    parity = _parity_module()
    fixtures_dir = tmp_path / "fixtures"
    _write_prompts_fixture(
        fixtures_dir,
        {"prompt-bad": {"text": "too short", "token_ids": [128000], "S": 1}},
    )

    with pytest.raises(parity.ParityError, match="(?i)prompt-bad|S|at least 2"):
        parity.load_prompt_cases(fixtures_dir)


def test_load_fixture_r_tokens_returns_committed_arrays_and_validates_contract():
    parity = _parity_module()
    cases = parity.load_prompt_cases(_FIXTURE_DIR)

    r_tokens = parity.load_fixture_r_tokens(_FIXTURE_DIR, cases, max_new_tokens=4)

    assert r_tokens == _BASELINE_R_TOKENS
    assert all(isinstance(token, int) for tokens in r_tokens.values() for token in tokens)


def test_load_fixture_r_tokens_rejects_max_new_tokens_mismatch():
    parity = _parity_module()
    cases = parity.load_prompt_cases(_FIXTURE_DIR)

    with pytest.raises(parity.ParityError, match="(?i)max_new_tokens|4|3"):
        parity.load_fixture_r_tokens(_FIXTURE_DIR, cases, max_new_tokens=3)


def test_load_fixture_r_tokens_rejects_missing_r_tokens_and_wrong_s(tmp_path):
    parity = _parity_module()
    fixtures_dir = tmp_path / "fixtures"
    _write_prompts_fixture(
        fixtures_dir,
        {"prompt-0": {"text": "ok", "token_ids": _PROMPT0_TOKENS, "S": 6}},
    )
    cases = parity.load_prompt_cases(fixtures_dir)

    _write_baseline_fixture(
        fixtures_dir,
        {"prompt-0": {"tokens": [1, 2, 3, 4], "max_new_tokens": 4, "S": 6}},
    )
    with pytest.raises(parity.ParityError, match="(?i)prompt-0|r_tokens"):
        parity.load_fixture_r_tokens(fixtures_dir, cases, max_new_tokens=4)

    _write_baseline_fixture(
        fixtures_dir,
        {"prompt-0": {"r_tokens": [1, 2, 3, 4], "max_new_tokens": 4, "S": 5}},
    )
    with pytest.raises(parity.ParityError, match="(?i)prompt-0|S|6|5"):
        parity.load_fixture_r_tokens(fixtures_dir, cases, max_new_tokens=4)


def test_compare_tokens_reports_exact_value_and_length_mismatches():
    parity = _parity_module()

    exact = parity.compare_tokens([1, 2, 3], [1, 2, 3])
    assert exact["exact_match"] is True
    assert exact["mismatch_indices"] == []
    assert exact["length_mismatch"] is False
    assert exact["p_len"] == 3
    assert exact["r_len"] == 3

    value_mismatch = parity.compare_tokens([1, 2, 3], [1, 99, 3])
    assert value_mismatch["exact_match"] is False
    assert value_mismatch["length_mismatch"] is False
    assert value_mismatch["mismatch_indices"] == [1]
    assert value_mismatch["p_tokens"] == [1, 2, 3]
    assert value_mismatch["r_tokens"] == [1, 99, 3]

    length_mismatch = parity.compare_tokens([1, 2], [1, 2, 3])
    assert length_mismatch["exact_match"] is False
    assert length_mismatch["length_mismatch"] is True
    assert length_mismatch["mismatch_indices"] == []
    assert length_mismatch["p_len"] == 2
    assert length_mismatch["r_len"] == 3


def test_p_decode_injects_s_minus_one_cache_and_final_token_only(tmp_path, monkeypatch):
    parity = _parity_module()
    observed = {"generate_calls": 0}
    cache_objects = [object(), object()]

    def fake_prefill_prompt_prefix(model_dir, prefix_token_ids, *, producer_kind=_CPU_REFERENCE_PRODUCER_KIND):
        observed["prefill_model_dir"] = model_dir
        observed["prefill_prefix_ids"] = list(prefix_token_ids)
        observed["producer_kind"] = producer_kind
        return {"model": model_dir, "n_prefix": 5, "layers": [], "producer_kind": producer_kind}

    def fake_emit_prompt_cache(prefill_result, out_path):
        observed["emitted_n_prefix"] = prefill_result["n_prefix"]
        observed["cache_path"] = Path(out_path)
        Path(out_path).write_text("fake cache", encoding="utf-8")

    def fake_load_prompt_cache(path, return_metadata=False):
        observed["loaded_cache_path"] = Path(path)
        assert return_metadata is True
        return cache_objects, {"offset": "5", "num_layers": "16"}

    def fake_load_model(model_dir):
        observed["loaded_model_dir"] = model_dir
        return "fake-model", "fake-tokenizer"

    def fake_generate_step(prompt, *args, **kwargs):
        observed["generate_calls"] += 1
        observed["generate_prompt"] = _as_int_list(prompt)
        observed["generate_args"] = args
        observed["generate_kwargs"] = kwargs
        assert _as_int_list(prompt) == [374]
        assert _PROMPT0_TOKENS != _as_int_list(prompt)
        prompt_cache = kwargs.get("prompt_cache", kwargs.get("cache"))
        assert prompt_cache is cache_objects
        yield np.int64(12366)
        yield np.int64(13)
        yield np.int64(578)
        yield np.int64(469)

    monkeypatch.setattr(parity, "prefill_prompt_prefix", fake_prefill_prompt_prefix, raising=False)
    monkeypatch.setattr(parity, "emit_prompt_cache", fake_emit_prompt_cache, raising=False)
    monkeypatch.setattr(parity, "load_prompt_cache", fake_load_prompt_cache, raising=False)
    monkeypatch.setattr(parity, "load_model", fake_load_model, raising=False)
    monkeypatch.setattr(parity, "generate_step", fake_generate_step, raising=False)

    result = parity.run_parity_suite(
        model_dir="fake-model-dir",
        fixtures_dir=_FIXTURE_DIR,
        r_source="fixture",
        max_new_tokens=4,
        artifacts_dir=tmp_path,
        prompt_names=("prompt-0",),
    )

    prompt0 = _prompt_result(result, "prompt-0")
    assert prompt0["p_tokens"] == _BASELINE_R_TOKENS["prompt-0"]
    assert prompt0["r_tokens"] == _BASELINE_R_TOKENS["prompt-0"]
    assert prompt0["comparison"]["exact_match"] is True
    assert observed["prefill_model_dir"] == "fake-model-dir"
    assert observed["prefill_prefix_ids"] == [128000, 791, 6864, 315, 9822]
    assert observed["emitted_n_prefix"] == 5
    assert observed["loaded_cache_path"] == observed["cache_path"]
    assert observed["loaded_model_dir"] == "fake-model-dir"
    assert observed["generate_calls"] == 1
    assert observed["generate_prompt"] == [374]
    assert observed["producer_kind"] == _CPU_REFERENCE_PRODUCER_KIND
    assert result["producer_kind"] == _CPU_REFERENCE_PRODUCER_KIND


def test_p_decode_rejects_prompt_cache_offset_mismatch(tmp_path, monkeypatch):
    parity = _parity_module()

    monkeypatch.setattr(
        parity,
        "prefill_prompt_prefix",
        lambda model_dir, prefix_token_ids: {"model": model_dir, "n_prefix": 5, "layers": []},
        raising=False,
    )
    monkeypatch.setattr(
        parity,
        "emit_prompt_cache",
        lambda prefill_result, out_path: Path(out_path).write_text("fake", encoding="utf-8"),
        raising=False,
    )
    monkeypatch.setattr(
        parity,
        "load_prompt_cache",
        lambda path, return_metadata=False: ([object()], {"offset": "4"}),
        raising=False,
    )
    monkeypatch.setattr(
        parity,
        "load_model",
        lambda model_dir: ("fake-model", "fake-tokenizer"),
        raising=False,
    )

    def unexpected_generate_step(*args, **kwargs):
        raise AssertionError("generate_step must not run after prompt-cache offset mismatch")
        yield 0

    monkeypatch.setattr(parity, "generate_step", unexpected_generate_step, raising=False)

    with pytest.raises(parity.ParityError, match="(?i)prompt-cache|offset|5|4"):
        parity.run_parity_suite(
            model_dir="fake-model-dir",
            fixtures_dir=_FIXTURE_DIR,
            r_source="fixture",
            max_new_tokens=4,
            artifacts_dir=tmp_path,
            prompt_names=("prompt-0",),
        )


def test_run_parity_suite_fixture_r_passes_when_all_prompt_tokens_match(tmp_path, monkeypatch):
    parity = _parity_module()

    def fake_decode_p_tokens_for_case(model_dir, case, max_new_tokens, artifacts_dir, *, producer_kind=_CPU_REFERENCE_PRODUCER_KIND):
        assert model_dir == "fake-model-dir"
        assert max_new_tokens == 4
        assert Path(artifacts_dir) == tmp_path
        assert producer_kind == _CPU_REFERENCE_PRODUCER_KIND
        return [np.int64(token) for token in _BASELINE_R_TOKENS[case.name]]

    monkeypatch.setattr(
        parity, "decode_p_tokens_for_case", fake_decode_p_tokens_for_case, raising=False
    )

    result = parity.run_parity_suite(
        model_dir="fake-model-dir",
        fixtures_dir=_FIXTURE_DIR,
        r_source="fixture",
        max_new_tokens=4,
        artifacts_dir=tmp_path,
    )

    assert result["status"] == "pass"
    assert result["gate_result"] == "pass"
    assert result["r_source"] == "fixture"
    assert result["producer_kind"] == _CPU_REFERENCE_PRODUCER_KIND
    assert [entry["prompt_name"] for entry in result["prompt_results"]] == [
        "prompt-0",
        "prompt-1",
        "prompt-2",
    ]
    for prompt_name, expected_tokens in _BASELINE_R_TOKENS.items():
        prompt_result = _prompt_result(result, prompt_name)
        assert prompt_result["p_tokens"] == expected_tokens
        assert prompt_result["r_tokens"] == expected_tokens
        assert prompt_result["comparison"]["exact_match"] is True
        assert prompt_result["status"] == "pass"



def _write_native_prefill_npz(path: Path, n_prefix: int) -> None:
    arrays = {
        "model": np.asarray("fake-model-dir"),
        "n_prefix": np.asarray(n_prefix, dtype=np.int64),
        "num_layers": np.asarray(16, dtype=np.int64),
        "producer_kind": np.asarray(_R9700_NATIVE_PRODUCER_KIND),
    }
    for layer_index in range(16):
        arrays[f"layer{layer_index}_K"] = np.zeros((1, 8, n_prefix, 64), dtype=np.float16)
        arrays[f"layer{layer_index}_V"] = np.zeros((1, 8, n_prefix, 64), dtype=np.float16)
    np.savez(path, **arrays)


def test_run_parity_suite_routes_r9700_native_through_native_worker(tmp_path, monkeypatch):
    """The r9700_native P path must go through the fail-closed native worker."""
    parity = _parity_module()
    native_worker = importlib.import_module("native_r9700.native_worker")
    worker_calls = []

    def fake_run_native_prefill(model_dir, token_ids, out_npz, log_path):
        worker_calls.append((model_dir, list(token_ids), str(out_npz), str(log_path)))
        _write_native_prefill_npz(Path(out_npz), n_prefix=len(token_ids))
        Path(log_path).write_text("native hardware evidence\n", encoding="utf-8")
        return {
            "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
            "native_prefill_acceptance": "pass",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_npz),
            "kernel_count": 2,
            "transfer_bytes": 2048,
            "failure_stage": "",
            "failure_text": "",
            "exit_status": 0,
        }

    def fake_load_model(model_dir):
        return object(), None

    def fake_collect(model, token_ids, max_new_tokens, *, prompt_cache=None):
        assert list(token_ids) == [_PROMPT0_TOKENS[-1]], (
            "P decode must receive only the final prompt token"
        )
        assert prompt_cache is not None
        return list(_BASELINE_R_TOKENS["prompt-0"])

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)
    monkeypatch.setattr(parity, "load_model", fake_load_model)
    monkeypatch.setattr(parity, "_collect_generated_tokens", fake_collect)

    result = parity.run_parity_suite(
        model_dir="fake-model-dir",
        fixtures_dir=_FIXTURE_DIR,
        r_source="fixture",
        max_new_tokens=4,
        artifacts_dir=tmp_path,
        prompt_names=("prompt-0",),
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )

    assert worker_calls == [
        (
            "fake-model-dir",
            _PROMPT0_TOKENS[:-1],
            str(tmp_path / "prompt-0-native-prefill.npz"),
            str(tmp_path / "prompt-0-native-prefill.log"),
        )
    ]
    prompt_result = _prompt_result(result, "prompt-0")
    assert result["gate_result"] == "pass"
    assert result["producer_kind"] == _R9700_NATIVE_PRODUCER_KIND
    assert prompt_result["native_prefill_acceptance"] == "pass"
    assert prompt_result["comparison"]["exact_match"] is True
    assert Path(prompt_result["hardware_log_path"]).is_file()
    assert Path(prompt_result["prompt_cache_path"]).is_file()


def test_run_parity_suite_native_worker_rejection_fails_closed(tmp_path, monkeypatch):
    parity = _parity_module()
    native_worker = importlib.import_module("native_r9700.native_worker")

    def fake_run_native_prefill(model_dir, token_ids, out_npz, log_path):
        Path(log_path).write_text("native hardware evidence\n", encoding="utf-8")
        return {
            "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
            "native_prefill_acceptance": "open",
            "hardware_log_path": str(log_path),
            "prefill_npz_path": str(out_npz),
            "kernel_count": 0,
            "transfer_bytes": 0,
            "failure_stage": "resident_prepare",
            "failure_text": "synthetic rejection",
            "exit_status": 1,
        }

    monkeypatch.setattr(native_worker, "run_native_prefill", fake_run_native_prefill)

    with pytest.raises(parity.ParityError, match="r9700_native producer failed"):
        parity.run_parity_suite(
            model_dir="fake-model-dir",
            fixtures_dir=_FIXTURE_DIR,
            r_source="fixture",
            max_new_tokens=4,
            artifacts_dir=tmp_path,
            prompt_names=("prompt-0",),
            producer_kind=_R9700_NATIVE_PRODUCER_KIND,
        )

def test_run_parity_suite_normalizes_native_prefill_error_to_parity_error(tmp_path, monkeypatch):
    parity = _parity_module()
    prefill_error = importlib.import_module("native_r9700.prefill").PrefillError

    def fail_native_decode(*args, **kwargs):
        raise prefill_error("native worker unavailable")

    monkeypatch.setattr(parity, "decode_p_tokens_for_case", fail_native_decode)

    with pytest.raises(
        parity.ParityError, match="r9700_native producer failed.*native worker unavailable"
    ) as exc_info:
        parity.run_parity_suite(
            model_dir="fake-model-dir",
            fixtures_dir=_FIXTURE_DIR,
            r_source="fixture",
            max_new_tokens=4,
            artifacts_dir=tmp_path,
            prompt_names=("prompt-0",),
            producer_kind=_R9700_NATIVE_PRODUCER_KIND,
        )

    assert isinstance(exc_info.value.__cause__, prefill_error)

def test_run_parity_suite_rejects_r9700_native_tokens_without_hardware_evidence(tmp_path, monkeypatch):
    parity = _parity_module()

    def fake_decode_p_tokens_for_case(model_dir, case, max_new_tokens, artifacts_dir, *, producer_kind=_CPU_REFERENCE_PRODUCER_KIND):
        return {
            "p_tokens": _BASELINE_R_TOKENS[case.name],
            "producer_kind": _CPU_REFERENCE_PRODUCER_KIND,
        }

    monkeypatch.setattr(parity, "decode_p_tokens_for_case", fake_decode_p_tokens_for_case, raising=False)

    with pytest.raises(parity.ParityError, match="producer_kind=r9700_native|hardware"):
        parity.run_parity_suite(
            model_dir="fake-model-dir",
            fixtures_dir=_FIXTURE_DIR,
            r_source="fixture",
            max_new_tokens=4,
            artifacts_dir=tmp_path,
            prompt_names=("prompt-0",),
            producer_kind=_R9700_NATIVE_PRODUCER_KIND,
        )


def test_run_parity_suite_accepts_r9700_native_only_with_hardware_evidence(tmp_path, monkeypatch):
    parity = _parity_module()

    def fake_decode_p_tokens_for_case(
        model_dir,
        case,
        max_new_tokens,
        artifacts_dir,
        *,
        producer_kind=_CPU_REFERENCE_PRODUCER_KIND,
    ):
        return _native_decode_evidence(
            tmp_path,
            p_tokens=_BASELINE_R_TOKENS[case.name],
        )

    monkeypatch.setattr(parity, "decode_p_tokens_for_case", fake_decode_p_tokens_for_case, raising=False)

    result = parity.run_parity_suite(
        model_dir="fake-model-dir",
        fixtures_dir=_FIXTURE_DIR,
        r_source="fixture",
        max_new_tokens=4,
        artifacts_dir=tmp_path,
        prompt_names=("prompt-0",),
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )

    prompt_result = _prompt_result(result, "prompt-0")
    assert result["gate_result"] == "pass"
    assert result["producer_kind"] == _R9700_NATIVE_PRODUCER_KIND
    assert prompt_result["native_prefill_acceptance"] == "pass"
    assert Path(prompt_result["hardware_log_path"]).is_file()
    assert Path(prompt_result["prompt_cache_path"]).is_file()
    assert prompt_result["kernel_count"] == 2
    assert prompt_result["transfer_bytes"] == 2048


@pytest.mark.parametrize("missing_field", ("hardware_log_path", "prompt_cache_path"))
def test_run_parity_suite_rejects_native_pass_without_required_hardware_artifacts(
    tmp_path, monkeypatch, missing_field
):
    parity = _parity_module()

    def fake_decode_p_tokens_for_case(
        model_dir,
        case,
        max_new_tokens,
        artifacts_dir,
        *,
        producer_kind=_CPU_REFERENCE_PRODUCER_KIND,
    ):
        decoded = _native_decode_evidence(
            tmp_path,
            p_tokens=_BASELINE_R_TOKENS[case.name],
        )
        del decoded[missing_field]
        return decoded

    monkeypatch.setattr(
        parity, "decode_p_tokens_for_case", fake_decode_p_tokens_for_case, raising=False
    )

    with pytest.raises(parity.ParityError, match=missing_field):
        parity.run_parity_suite(
            model_dir="fake-model-dir",
            fixtures_dir=_FIXTURE_DIR,
            r_source="fixture",
            max_new_tokens=4,
            artifacts_dir=tmp_path,
            prompt_names=("prompt-0",),
            producer_kind=_R9700_NATIVE_PRODUCER_KIND,
        )


def test_run_parity_suite_rejects_cpu_producer_identity_claiming_native_evidence(
    tmp_path, monkeypatch
):
    parity = _parity_module()

    def fake_decode_p_tokens_for_case(
        model_dir,
        case,
        max_new_tokens,
        artifacts_dir,
        *,
        producer_kind=_CPU_REFERENCE_PRODUCER_KIND,
    ):
        assert producer_kind == _R9700_NATIVE_PRODUCER_KIND
        return _native_decode_evidence(
            tmp_path,
            p_tokens=_BASELINE_R_TOKENS[case.name],
            producer_kind=_CPU_REFERENCE_PRODUCER_KIND,
        )

    monkeypatch.setattr(
        parity, "decode_p_tokens_for_case", fake_decode_p_tokens_for_case, raising=False
    )

    with pytest.raises(parity.ParityError, match="producer_kind=r9700_native"):
        parity.run_parity_suite(
            model_dir="fake-model-dir",
            fixtures_dir=_FIXTURE_DIR,
            r_source="fixture",
            max_new_tokens=4,
            artifacts_dir=tmp_path,
            prompt_names=("prompt-0",),
            producer_kind=_R9700_NATIVE_PRODUCER_KIND,
        )


@pytest.mark.parametrize(
    ("failure_stage", "failure_text", "required_field"),
    (
        ("native_dispatch", "", "failure_stage"),
        ("", "dispatch did not reach R9700", "failure_text"),
    ),
)
def test_run_parity_suite_rejects_native_pass_with_failure_evidence(
    tmp_path, monkeypatch, failure_stage, failure_text, required_field
):
    parity = _parity_module()

    def fake_decode_p_tokens_for_case(
        model_dir,
        case,
        max_new_tokens,
        artifacts_dir,
        *,
        producer_kind=_CPU_REFERENCE_PRODUCER_KIND,
    ):
        return _native_decode_evidence(
            tmp_path,
            p_tokens=_BASELINE_R_TOKENS[case.name],
            failure_stage=failure_stage,
            failure_text=failure_text,
        )

    monkeypatch.setattr(
        parity, "decode_p_tokens_for_case", fake_decode_p_tokens_for_case, raising=False
    )

    with pytest.raises(parity.ParityError, match=required_field):
        parity.run_parity_suite(
            model_dir="fake-model-dir",
            fixtures_dir=_FIXTURE_DIR,
            r_source="fixture",
            max_new_tokens=4,
            artifacts_dir=tmp_path,
            prompt_names=("prompt-0",),
            producer_kind=_R9700_NATIVE_PRODUCER_KIND,
        )


def test_run_parity_suite_native_evidence_preserves_s_minus_one_final_token_handoff(
    tmp_path, monkeypatch
):
    parity = _parity_module()
    observed = {}

    def fake_decode_p_tokens_for_case(
        model_dir,
        case,
        max_new_tokens,
        artifacts_dir,
        *,
        producer_kind=_CPU_REFERENCE_PRODUCER_KIND,
    ):
        observed["prefix_token_ids"] = case.prefix_token_ids
        observed["final_token_id"] = case.final_token_id
        observed["producer_kind"] = producer_kind
        return _native_decode_evidence(
            tmp_path,
            p_tokens=_BASELINE_R_TOKENS[case.name],
        )

    monkeypatch.setattr(
        parity, "decode_p_tokens_for_case", fake_decode_p_tokens_for_case, raising=False
    )

    result = parity.run_parity_suite(
        model_dir="fake-model-dir",
        fixtures_dir=_FIXTURE_DIR,
        r_source="fixture",
        max_new_tokens=4,
        artifacts_dir=tmp_path,
        prompt_names=("prompt-0",),
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )

    prompt_result = _prompt_result(result, "prompt-0")
    assert observed == {
        "prefix_token_ids": _PROMPT0_TOKENS[:-1],
        "final_token_id": _PROMPT0_TOKENS[-1],
        "producer_kind": _R9700_NATIVE_PRODUCER_KIND,
    }
    assert prompt_result["S"] == len(_PROMPT0_TOKENS)
    assert prompt_result["n_prefix"] == len(_PROMPT0_TOKENS) - 1
    assert prompt_result["final_token_id"] == _PROMPT0_TOKENS[-1]
    assert prompt_result["p_tokens"] == _BASELINE_R_TOKENS["prompt-0"]
    assert prompt_result["comparison"]["exact_match"] is True


def test_run_parity_suite_native_token_mismatch_fails_without_native_pass_claim(
    tmp_path, monkeypatch
):
    parity = _parity_module()

    def fake_decode_p_tokens_for_case(
        model_dir,
        case,
        max_new_tokens,
        artifacts_dir,
        *,
        producer_kind=_CPU_REFERENCE_PRODUCER_KIND,
    ):
        p_tokens = list(_BASELINE_R_TOKENS[case.name])
        p_tokens[2] = 999
        return _native_decode_evidence(tmp_path, p_tokens=p_tokens)

    monkeypatch.setattr(
        parity, "decode_p_tokens_for_case", fake_decode_p_tokens_for_case, raising=False
    )

    result = parity.run_parity_suite(
        model_dir="fake-model-dir",
        fixtures_dir=_FIXTURE_DIR,
        r_source="fixture",
        max_new_tokens=4,
        artifacts_dir=tmp_path,
        prompt_names=("prompt-0",),
        producer_kind=_R9700_NATIVE_PRODUCER_KIND,
    )

    prompt_result = _prompt_result(result, "prompt-0")
    assert result["status"] == "fail"
    assert result["gate_result"] == "fail"
    assert result["producer_kind"] == _R9700_NATIVE_PRODUCER_KIND
    assert prompt_result["status"] == "fail"
    assert prompt_result["comparison"]["exact_match"] is False
    assert prompt_result["comparison"]["mismatch_indices"] == [2]
    assert prompt_result["native_prefill_acceptance"] == "pass"


def test_run_parity_suite_fixture_r_fails_on_one_token_mismatch_without_pass_claim(
    tmp_path, monkeypatch
):
    parity = _parity_module()

    def fake_decode_p_tokens_for_case(model_dir, case, max_new_tokens, artifacts_dir, *, producer_kind=_CPU_REFERENCE_PRODUCER_KIND):
        tokens = list(_BASELINE_R_TOKENS[case.name])
        if case.name == "prompt-1":
            tokens[2] = 999
        return tokens

    monkeypatch.setattr(
        parity, "decode_p_tokens_for_case", fake_decode_p_tokens_for_case, raising=False
    )

    result = parity.run_parity_suite(
        model_dir="fake-model-dir",
        fixtures_dir=_FIXTURE_DIR,
        r_source="fixture",
        max_new_tokens=4,
        artifacts_dir=tmp_path,
    )

    assert result["status"] == "fail"
    assert result["gate_result"] == "fail"
    mismatch = _prompt_result(result, "prompt-1")
    assert mismatch["status"] == "fail"
    assert mismatch["comparison"]["exact_match"] is False
    assert mismatch["comparison"]["mismatch_indices"] == [2]
    assert mismatch["p_tokens"] == [128009, 128006, 999, 271]
    assert mismatch["r_tokens"] == _BASELINE_R_TOKENS["prompt-1"]
    assert all(entry["status"] != "pass" or entry["comparison"]["exact_match"] for entry in result["prompt_results"])


def test_append_or_replace_path_c_report_preserves_path_a_and_replaces_only_path_c(tmp_path):
    parity = _parity_module()
    report_path = tmp_path / "path-a-validation-results.md"
    report_path.write_text(
        "# Validation Results\n\n"
        "## Path A — Tinygrad producer baseline\n"
        "path-a content must stay\n\n"
        f"{_PATH_C_HEADING}\n"
        "old path-c content must be replaced\n\n"
        "## Path D — Future serving integration\n"
        "path-d content must stay\n",
        encoding="utf-8",
    )
    result = {
        "status": "pass",
        "gate_result": "pass",
        "producer_kind": _CPU_REFERENCE_PRODUCER_KIND,
        "r_source": "fixture",
        "prompt_results": [
            {
                "prompt_name": "prompt-0",
                "status": "pass",
                "p_tokens": _BASELINE_R_TOKENS["prompt-0"],
                "r_tokens": _BASELINE_R_TOKENS["prompt-0"],
                "comparison": {"exact_match": True, "mismatch_indices": [], "length_mismatch": False},
            }
        ],
    }

    parity.append_or_replace_path_c_report(report_path, result)

    text = report_path.read_text(encoding="utf-8")
    assert "## Path A — Tinygrad producer baseline\npath-a content must stay" in text
    assert "## Path D — Future serving integration\npath-d content must stay" in text
    assert text.count(_PATH_C_HEADING) == 1
    assert "old path-c content must be replaced" not in text
    assert "gate_result: pass" in text
    assert "producer_kind: cpu_reference" in text
    assert "REFERENCE PASS; NATIVE R9700 C1 OPEN" in text
    assert "prompt-0" in text

def test_path_c_report_includes_required_log_provenance_and_rope_grounding(tmp_path):
    parity = _parity_module()
    report_path = tmp_path / "path-a-validation-results.md"
    result = {
        "status": "pass",
        "gate_result": "pass",
        "r_source": "both",
        "model_dir": "mlx_models/meta-Llama-3.2-1B-Instruct",
        "config_path": "mlx_models/meta-Llama-3.2-1B-Instruct/config.json",
        "json_path": "logs/c1-parity/result.json",
        "log_path": "logs/c1-parity/run.log",
        "weight_provenance": "official fp16 meta-llama/Llama-3.2-1B-Instruct MLX safetensors",
        "rope_config_note": "Llama-3 rope_scaling loaded from config.json sidecar",
        "prompt_results": [],
        "producer_kind": _CPU_REFERENCE_PRODUCER_KIND,
        "per_layer": [{"layer": 0, "max_K": 0.0, "mean_K": 0.0, "max_V": 0.0, "mean_V": 0.0}],
    }

    parity.append_or_replace_path_c_report(report_path, result)

    text = report_path.read_text(encoding="utf-8")
    assert "log_path: logs/c1-parity/run.log" in text
    assert "json_path: logs/c1-parity/result.json" in text
    assert "weight_provenance: official fp16 meta-llama/Llama-3.2-1B-Instruct MLX safetensors" in text
    assert "rope_config_note: Llama-3 rope_scaling loaded from config.json sidecar" in text
    assert "config_path: mlx_models/meta-Llama-3.2-1B-Instruct/config.json" in text
    assert "producer_kind: cpu_reference" in text



def test_main_writes_json_log_report_and_returns_zero_for_pass(tmp_path, monkeypatch):
    parity = _parity_module()
    artifacts_dir = tmp_path / "artifacts"
    json_path = artifacts_dir / "result.json"
    log_path = artifacts_dir / "run.log"
    report_path = tmp_path / "path-a-validation-results.md"
    report_path.write_text("# Existing report\n\n## Path A — Baseline\nkeep\n", encoding="utf-8")
    fake_result = {
        "status": "pass",
        "gate_result": "pass",
        "r_source": "both",
        "prompt_results": [],
        "producer_kind": _CPU_REFERENCE_PRODUCER_KIND,
    }
    observed = {}

    def fake_run_parity_suite(**kwargs):
        observed.update(kwargs)
        return fake_result

    monkeypatch.setattr(parity, "run_parity_suite", fake_run_parity_suite)

    rc = parity.main(
        [
            "--model",
            str(_LLAMA_MLX_MODEL_DIR),
            "--fixtures-dir",
            "tests/native_r9700/fixtures",
            "--r-source",
            "both",
            "--max-new-tokens",
            "4",
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

    assert observed["model_dir"] == str(_LLAMA_MLX_MODEL_DIR)
    assert Path(observed["fixtures_dir"]) == Path("tests/native_r9700/fixtures")
    assert observed["r_source"] == "both"
    assert observed["max_new_tokens"] == 4
    assert observed["producer_kind"] == _CPU_REFERENCE_PRODUCER_KIND
    assert Path(observed["artifacts_dir"]) == artifacts_dir
    assert json.loads(json_path.read_text(encoding="utf-8")) == fake_result
    log_text = log_path.read_text(encoding="utf-8")
    assert "command:" in log_text
    assert _PYTHON in log_text
    assert "--r-source both" in log_text
    assert "gate_result: pass" in log_text
    assert "producer_kind: cpu_reference" in log_text
    assert "exit_status: 0" in log_text
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Path A — Baseline\nkeep" in report_text
    assert _PATH_C_HEADING in report_text
    assert "gate_result: pass" in report_text


def test_main_returns_one_for_token_fail_and_still_writes_artifacts(tmp_path, monkeypatch):
    parity = _parity_module()
    artifacts_dir = tmp_path / "artifacts"
    json_path = artifacts_dir / "result.json"
    log_path = artifacts_dir / "run.log"
    report_path = tmp_path / "path-a-validation-results.md"
    fake_result = {
        "status": "fail",
        "gate_result": "fail",
        "r_source": "both",
        "prompt_results": [
            {
                "prompt_name": "prompt-1",
                "status": "fail",
                "p_tokens": [128009, 128006, 999, 271],
                "r_tokens": _BASELINE_R_TOKENS["prompt-1"],
                "comparison": {"exact_match": False, "mismatch_indices": [2], "length_mismatch": False},
            }
        ],
    }

    monkeypatch.setattr(parity, "run_parity_suite", lambda **kwargs: fake_result)

    rc = parity.main(
        [
            "--model",
            str(_LLAMA_MLX_MODEL_DIR),
            "--fixtures-dir",
            "tests/native_r9700/fixtures",
            "--r-source",
            "both",
            "--max-new-tokens",
            "4",
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

    assert rc == 1
    assert json.loads(json_path.read_text(encoding="utf-8")) == fake_result
    assert "gate_result: fail" in log_path.read_text(encoding="utf-8")
    assert _PATH_C_HEADING in report_path.read_text(encoding="utf-8")


def test_main_writes_blocked_json_log_report_and_replaces_stale_pass_on_error(
    tmp_path, monkeypatch
):
    parity = _parity_module()
    artifacts_dir = tmp_path / "artifacts"
    json_path = artifacts_dir / "result.json"
    log_path = artifacts_dir / "run.log"
    report_path = tmp_path / "path-a-validation-results.md"
    report_path.write_text(
        "# Existing report\n\n"
        "## Path A — Baseline\nkeep\n\n"
        f"{_PATH_C_HEADING}\n\n"
        "Status: **PASS**\n"
        "gate_result: pass\n"
        "stale token evidence must be removed\n",
        encoding="utf-8",
    )

    def fail_run_parity_suite(**kwargs):
        raise parity.ParityError("prompt-cache metadata offset '4' != expected '5'")

    monkeypatch.setattr(parity, "run_parity_suite", fail_run_parity_suite)

    rc = parity.main(
        [
            "--model",
            str(_LLAMA_MLX_MODEL_DIR),
            "--fixtures-dir",
            "tests/native_r9700/fixtures",
            "--r-source",
            "both",
            "--max-new-tokens",
            "4",
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

    assert rc == 2
    blocked = json.loads(json_path.read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked"
    assert blocked["gate_result"] == "blocked"
    assert blocked["error"]["type"] == "ParityError"
    assert "metadata offset" in blocked["error"]["message"]
    log_text = log_path.read_text(encoding="utf-8")
    assert "gate_result: blocked" in log_text
    assert "exit_status: 2" in log_text
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Path A — Baseline\nkeep" in report_text
    assert report_text.count(_PATH_C_HEADING) == 1
    assert "Status: **BLOCKED**" in report_text
    assert "gate_result: pass" not in report_text
    assert "stale token evidence must be removed" not in report_text
    assert "prompt-cache metadata offset" in report_text
