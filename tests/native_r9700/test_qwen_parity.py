"""No-hardware contracts for restoring Qwen hybrid state before final-token decode."""

from hashlib import sha256

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


def test_restores_the_existing_interleaved_state_into_language_model_cache() -> None:
    model = Model()
    restored = state()

    cache = qwen_parity.restore_qwen_hybrid_state_into_model(model, restored)

    assert cache is model.language_model.cache
    for index, entry in enumerate(restored.entries):
        layer = cache.layers[index]
        assert type(layer).__name__ == entry.class_name
        assert layer.state is entry.leaves
        if entry.class_name == "KVCache":
            assert layer.offset == restored.committed_position


def test_decodes_with_only_the_final_token_after_restoring_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Model()
    observed: dict[str, object] = {}

    def fake_generate_step(prompt, passed_model, **kwargs):
        observed["prompt"] = prompt
        observed["model"] = passed_model
        observed["cache"] = kwargs["prompt_cache"]
        yield 987

    monkeypatch.setattr(qwen_parity, "generate_step", fake_generate_step)

    generated = qwen_parity.generate_qwen_from_hybrid_state(model, state(), (248044, 12, 13))

    assert list(generated) == [987]
    assert observed == {
        "prompt": [13],
        "model": model,
        "cache": model.language_model.cache,
    }


def test_rejects_non_qwen_language_cache_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Model()
    model.language_model.cache.layers[3] = ArraysCache()
    monkeypatch.setattr(
        qwen_parity,
        "generate_step",
        lambda *args, **kwargs: pytest.fail("invalid cache must not call generate_step"),
    )

    with pytest.raises(qwen_parity.QwenParityError, match="layer 3|KVCache"):
        qwen_parity.generate_qwen_from_hybrid_state(model, state(), (248044,))
