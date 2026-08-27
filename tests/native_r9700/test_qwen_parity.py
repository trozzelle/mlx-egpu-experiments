"""No-hardware contracts for restoring Qwen hybrid state before final-token decode."""

import json
import shutil
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from native_r9700 import qwen_parity
from native_r9700.qwen_spill import QwenHybridState, QwenStateEntry, QwenStateLeaf


class ArraysCache:
    def __init__(self) -> None:
        self.state = None


class KVCache:
    def __init__(self) -> None:
        self.state = None
        self.offset = -1


class Cache:
    def __init__(self) -> None:
        self.layers = [KVCache() if index % 4 == 3 else ArraysCache() for index in range(64)]


class LanguageModel:
    def __init__(self) -> None:
        self.cache = Cache()


class Model:
    def __init__(self) -> None:
        self.language_model = LanguageModel()


def state(position: int = 9) -> QwenHybridState:
    entries = []
    for index in range(64):
        left_payload = bytes((index,))
        right_payload = bytes((index + 1,))
        leaves = (
            QwenStateLeaf((1,), "bfloat16", left_payload, sha256(left_payload).hexdigest()),
            QwenStateLeaf((1,), "bfloat16", right_payload, sha256(right_payload).hexdigest()),
        )
        cache_class = "KVCache" if index % 4 == 3 else "ArraysCache"
        entries.append(QwenStateEntry(index, cache_class, position if cache_class == "KVCache" else None, leaves))
    return QwenHybridState("qwen-text", position, tuple(entries))


def test_restores_the_existing_interleaved_state_through_task3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Model()
    restored = state()
    expected_cache = model.language_model.cache
    observed: dict[str, object] = {}

    def fake_restore(model_arg, state_arg, *args, **kwargs):
        observed.update(model=model_arg, state=state_arg, args=args, kwargs=kwargs)
        return expected_cache

    monkeypatch.setattr(qwen_parity, "restore_qwen_hybrid_cache_into_mlx", fake_restore)
    assert qwen_parity.restore_qwen_hybrid_state_into_model(model, restored) is expected_cache
    assert observed == {"model": model, "state": restored, "args": (), "kwargs": {}}


def test_decodes_with_only_the_final_token_after_restoring_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Model()
    restored_cache = model.language_model.cache
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        qwen_parity,
        "restore_qwen_hybrid_cache_into_mlx",
        lambda *args, **kwargs: restored_cache,
    )

    def fake_generate_step(prompt, passed_model, **kwargs):
        observed["prompt"] = prompt
        observed["model"] = passed_model
        observed["cache"] = kwargs["prompt_cache"]
        yield 987

    monkeypatch.setattr(qwen_parity, "generate_step", fake_generate_step)
    generated = qwen_parity.generate_qwen_from_hybrid_state(model, state(), (248044, 12, 13))

    assert list(generated) == [987]
    assert list(observed["prompt"]) == [13]
    assert observed["model"] is model
    assert observed["cache"] is restored_cache


def test_rejects_non_qwen_language_cache_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Model()
    monkeypatch.setattr(
        qwen_parity,
        "restore_qwen_hybrid_cache_into_mlx",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("layer 3 must be KVCache")),
    )
    monkeypatch.setattr(
        qwen_parity,
        "generate_step",
        lambda *args, **kwargs: pytest.fail("invalid cache must not call generate_step"),
    )

    with pytest.raises(qwen_parity.QwenParityError, match="layer 3|KVCache"):
        qwen_parity.generate_qwen_from_hybrid_state(model, state(), (248044,))


QWEN_PROBE_TOKEN_IDS = (760, 6511, 314, 9338, 369)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_QWEN_MODEL_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"


def test_qwen_parity_uses_task3_mlx_restore_and_final_token_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixture parity must call executable MLX restore, never opaque leaf assignment."""
    model = object()
    state = object()
    restored_cache = object()
    observed: dict[str, object] = {}

    def fake_restore(model_arg, state_arg, *args, **kwargs):
        observed["restore_model"] = model_arg
        observed["restore_state"] = state_arg
        observed["restore_args"] = args
        observed["restore_kwargs"] = kwargs
        return restored_cache

    def fake_generate_step(prompt, passed_model, **kwargs):
        observed["prompt"] = prompt
        observed["model"] = passed_model
        observed["cache"] = kwargs["prompt_cache"]
        yield 987

    monkeypatch.setattr(
        qwen_parity,
        "restore_qwen_hybrid_cache_into_mlx",
        fake_restore,
        raising=False,
    )
    monkeypatch.setattr(qwen_parity, "generate_step", fake_generate_step)

    assert list(
        qwen_parity.generate_qwen_from_hybrid_state(
            model,
            state,
            QWEN_PROBE_TOKEN_IDS,
        )
    ) == [987]
    assert observed["restore_model"] is model
    assert observed["restore_state"] is state
    assert list(observed["prompt"]) == [369]
    assert observed["model"] is model
    assert observed["cache"] is restored_cache


def test_qwen_parity_exposes_fixture_comparison_and_rejects_native_evidence():
    compare = getattr(qwen_parity, "compare_qwen_fixtures", None)
    assert callable(compare), (
        "native_r9700.qwen_parity must expose compare_qwen_fixtures for "
        "--compare-fixtures"
    )
    validate_evidence = getattr(qwen_parity, "validate_qwen_fixture_evidence", None)
    assert callable(validate_evidence), (
        "native_r9700.qwen_parity must expose fixture evidence validation"
    )

    validate_evidence({"producer_kind": "cpu_reference", "native_evidence": False})
    with pytest.raises(qwen_parity.QwenParityError, match="native|producer_kind"):
        validate_evidence({"producer_kind": "r9700_native", "native_evidence": True})
    with pytest.raises(qwen_parity.QwenParityError, match="native|native_evidence"):
        validate_evidence({"producer_kind": "cpu_reference", "native_evidence": True})


def test_qwen_parity_rejects_expected_basename_with_mismatched_source_identity(
    tmp_path: Path,
) -> None:
    """A revision-looking directory must not bypass source-pin validation."""
    model_dir = tmp_path / _QWEN_MODEL_REVISION
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type":"not-the-pinned-qwen-model"}',
        encoding="utf-8",
    )
    assert model_dir.name == _QWEN_MODEL_REVISION

    with pytest.raises(
        qwen_parity.QwenParityError,
        match="model|source|identity|revision|fingerprint|config",
    ):
        qwen_parity.compare_qwen_fixtures(
            _REPO_ROOT / "tests" / "native_r9700" / "fixtures",
            model_dir=model_dir,
            inventory=_REPO_ROOT / "logs" / "q1-qwen-tensor-inventory.json",
            token_ids=QWEN_PROBE_TOKEN_IDS,
        )


@pytest.mark.parametrize(
    "mutation",
    ("remove_tensors", "empty_shards", "remove_affine_classification", "mutate_provenance"),
)
def test_qwen_parity_rejects_inventory_structure_drift_with_frozen_scalar_identity(
    mutation: str,
) -> None:
    """Inventory contents and CPU provenance are bound, not just scalar digests."""
    schema_path = _REPO_ROOT / "tests" / "native_r9700" / "fixtures" / "qwen_fixtures_schema.json"
    inventory_path = _REPO_ROOT / "logs" / "q1-qwen-tensor-inventory.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    assert inventory["schema_version"] == 2
    assert inventory["model_fingerprint"] == schema["model_fingerprint"]
    assert inventory["inventory_sha256"] == schema["inventory_sha256"]
    if mutation == "remove_tensors":
        inventory.pop("tensors")
    elif mutation == "empty_shards":
        inventory["shards"] = []
    elif mutation == "remove_affine_classification":
        inventory.pop("affine_classification")
    else:
        inventory["producer_kind"] = "r9700_native"
        inventory["native_evidence"] = True

    with pytest.raises(
        qwen_parity.QwenParityError,
        match="inventory|tensor|shard|affine|provenance|producer|native",
    ):
        qwen_parity.compare_qwen_fixtures(
            _REPO_ROOT / "tests" / "native_r9700" / "fixtures",
            inventory=inventory,
            token_ids=QWEN_PROBE_TOKEN_IDS,
        )


def _empty_qwen_fixture_package(tmp_path: Path) -> Path:
    fixture_dir = tmp_path / "fixtures"
    shutil.copytree(
        _REPO_ROOT / "tests" / "native_r9700" / "fixtures",
        fixture_dir,
    )
    schema_path = fixture_dir / "qwen_fixtures_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    artifact_names = (
        "qwen_affine_windows.npz",
        "qwen_hybrid_state_samples.npz",
        "qwen_oracle_trace.npz",
    )
    for name in artifact_names:
        artifact_path = fixture_dir / name
        np.savez(artifact_path)
        schema["files"][name]["arrays"] = {}
        schema["files"][name]["sha256"] = sha256(artifact_path.read_bytes()).hexdigest()

    determinism_preimage = {
        "model_fingerprint": schema["model_fingerprint"],
        "inventory_sha256": schema["inventory_sha256"],
        "source_revisions": schema["source_revisions"],
        "shards": schema["shards"],
        "fixture_file_sha256": {
            name: schema["files"][name]["sha256"] for name in sorted(schema["files"])
        },
    }
    schema["determinism_digest"] = sha256(
        json.dumps(
            determinism_preimage,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    schema_path.write_text(
        json.dumps(schema, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return fixture_dir


def test_qwen_parity_rejects_self_consistent_empty_required_fixture_arrays(
    tmp_path: Path,
) -> None:
    """Recomputed file/determinism hashes cannot bless an empty oracle package."""
    fixture_dir = _empty_qwen_fixture_package(tmp_path)

    with pytest.raises(
        qwen_parity.QwenParityError,
        match="fixture|array|affine|state|trace|boundary|component",
    ):
        qwen_parity.compare_qwen_fixtures(
            fixture_dir,
            inventory=_REPO_ROOT / "logs" / "q1-qwen-tensor-inventory.json",
            token_ids=QWEN_PROBE_TOKEN_IDS,
        )
