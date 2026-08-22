"""Native prefill NPZ serialization contracts (no hardware).

The C++ serializer converts raw full-capacity resident K/V readback buffers
(head-major [kv_head][capacity][head_dim] fp16) into the strict prefill NPZ
schema consumed by native_worker.validate_native_prefill_npz and kv_cache.py:
fp16 arrays shaped (1, 8, n_prefix, 64) plus model/n_prefix/num_layers/
producer_kind scalars, written atomically via a temp sibling and rename.
"""

from pathlib import Path
import subprocess

import numpy as np
import pytest

from native_r9700 import native_worker


SERIALIZER_SOURCE = Path("native_r9700/prefill_npz.cpp")
SERIALIZER_HEADER = Path("native_r9700/prefill_npz.h")

_CAPACITY = 128
_KV_HEADS = 8
_HEAD_DIM = 64
_NUM_LAYERS = 16
_N_PREFIX = 2

_PROBE_SOURCE = r"""
#include "prefill_npz.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kCapacity = 128;
constexpr uint64_t kCacheBytes =
    static_cast<uint64_t>(kCapacity) * 8 * 64 * sizeof(uint16_t);

void fill_cache(uint32_t slot, std::vector<uint8_t>* bytes) {
  bytes->resize(kCacheBytes);
  for (uint64_t element = 0; element < kCacheBytes / 2; ++element) {
    const uint16_t value = static_cast<uint16_t>((element + slot * 1009) & 0xFFFFu);
    (*bytes)[element * 2] = static_cast<uint8_t>(value & 0xFFu);
    (*bytes)[element * 2 + 1] = static_cast<uint8_t>(value >> 8);
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) return 2;
  native_r9700::NativePrefillNpzPayload payload;
  payload.model = "synthetic-model";
  payload.n_prefix = 2;
  payload.cache_capacity_tokens = kCapacity;
  for (uint32_t slot = 0; slot < 32; ++slot) {
    std::vector<uint8_t> cache;
    fill_cache(slot, &cache);
    payload.kv_readback_bytes.push_back(std::move(cache));
  }
  if (argc == 3) payload.kv_readback_bytes.back().resize(kCacheBytes - 2);
  std::string error;
  if (!native_r9700::write_native_prefill_npz(payload, argv[1], &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  return 0;
}
"""


def _compile_probe(tmp_path: Path) -> Path:
    assert SERIALIZER_SOURCE.exists(), "native prefill NPZ serializer source missing"
    assert SERIALIZER_HEADER.exists(), "native prefill NPZ serializer header missing"
    source = tmp_path / "prefill_npz_probe.cpp"
    source.write_text(_PROBE_SOURCE, encoding="utf-8")
    exe = tmp_path / "prefill_npz_probe"
    subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(SERIALIZER_SOURCE),
            str(source),
            "-I",
            "native_r9700",
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return exe


def _run_probe(exe: Path, out_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(exe), str(out_path), *extra],
        check=False,
        capture_output=True,
        text=True,
    )


def _expected_cache_bits(slot: int, n_prefix: int) -> np.ndarray:
    heads = np.arange(_KV_HEADS, dtype=np.uint64).reshape(_KV_HEADS, 1, 1)
    tokens = np.arange(n_prefix, dtype=np.uint64).reshape(1, n_prefix, 1)
    dims = np.arange(_HEAD_DIM, dtype=np.uint64).reshape(1, 1, _HEAD_DIM)
    flat = (heads * _CAPACITY + tokens) * _HEAD_DIM + dims
    return ((flat + slot * 1009) & 0xFFFF).astype(np.uint16)


def test_npz_roundtrip_validates_with_strict_schema(tmp_path: Path) -> None:
    exe = _compile_probe(tmp_path)
    out_path = tmp_path / "prefill.npz"
    completed = _run_probe(exe, out_path)
    assert completed.returncode == 0, completed.stderr

    assert native_worker.validate_native_prefill_npz(out_path, _N_PREFIX, "synthetic-model") == []


def test_npz_layer_kv_permutation_is_head_major_prefix_slice(tmp_path: Path) -> None:
    exe = _compile_probe(tmp_path)
    out_path = tmp_path / "prefill.npz"
    completed = _run_probe(exe, out_path)
    assert completed.returncode == 0, completed.stderr

    with np.load(out_path, allow_pickle=False) as npz:
        assert set(npz.files) == {
            "model",
            "n_prefix",
            "num_layers",
            "producer_kind",
            *(
                f"layer{layer}_{kind}"
                for layer in range(_NUM_LAYERS)
                for kind in ("K", "V")
            ),
        }
        assert int(npz["n_prefix"]) == _N_PREFIX
        assert int(npz["num_layers"]) == _NUM_LAYERS
        assert str(npz["producer_kind"]) == "r9700_native"
        assert str(npz["model"]) == "synthetic-model"
        for layer in range(_NUM_LAYERS):
            for kind, slot in (("K", 2 * layer), ("V", 2 * layer + 1)):
                array = npz[f"layer{layer}_{kind}"]
                assert array.dtype == np.float16
                assert array.shape == (1, _KV_HEADS, _N_PREFIX, _HEAD_DIM)
                bits = np.ascontiguousarray(array).view(np.uint16).reshape(
                    _KV_HEADS, _N_PREFIX, _HEAD_DIM
                )
                np.testing.assert_array_equal(
                    bits, _expected_cache_bits(slot, _N_PREFIX),
                    err_msg=f"layer{layer}_{kind} permutation mismatch",
                )


def test_npz_write_is_atomic_and_fail_closed(tmp_path: Path) -> None:
    exe = _compile_probe(tmp_path)
    out_path = tmp_path / "prefill.npz"
    completed = _run_probe(exe, out_path)
    assert completed.returncode == 0, completed.stderr
    assert [path.name for path in tmp_path.iterdir() if path.name != out_path.name and path.suffix != ".cpp" and path.name != exe.name] == []

    bad_path = tmp_path / "bad.npz"
    rejected = _run_probe(exe, bad_path, "truncate")
    assert rejected.returncode == 1
    assert not bad_path.exists()
    assert [path for path in tmp_path.glob("*.npz*") if path.name not in {out_path.name}] == []
