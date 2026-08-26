# Repository Guidelines

## Project Overview

This repo now develops two co-equal products for the AMD Radeon AI PRO R9700 eGPU on Apple Silicon: a persistent **R9700 Prefill Service** for mlx-lm/oMLX consumers and a **Portable Inference Device Platform** built around TinyGPU ownership, an inference-shaped HAL, and admitted Kernel Packs. The durable compatibility artifact remains the mlx-lm prompt cache; the broader logical boundary is the Canonical KV Description plus Engine Adapters.

Baseline B0 is complete: native C0 kernel/transfer/resident-VRAM proof, 16-layer Llama 3.2 1B C1R token parity through prompt-128, and C2R imported-cache serving through the actual hardware producer with no fallback. Do not reopen those gates or relabel CPU/NumPy evidence as native acceptance.

Current parallel-ready work is F1 persistent warm worker, F2 gfx1201 WMMA foundation, P1 TinyGPU device-owner hardening, P3 Kernel Packs, and Q1 Qwen contract/oracle research. Prefer the shortest evidence-producing slice; do not build exhaustive proof ladders or speculative platform abstractions without a phase gate.

Qwen3.8-27B remains a separate target-expansion contract because it uses MLX-VLM Qwen3.5-family quantization and hybrid recurrent/full-attention cache state. Q1 may proceed in parallel, but native Qwen performance promotion waits for the selected shared matrix/attention prerequisites.

## Architecture & Data Flow

- Canonical vocabulary lives in `CONTEXT.md`: distinguish in-memory **KV cache** from serialized **prompt cache**; the **Prefill Producer** owns KV truth; the **Prefill Consumer** must not recompute an accepted prefix.
- Accepted native flow: `native_r9700.prefill` validates the request, `native_r9700.native_worker` invokes the TinyGPU/AMDev C++ runner, the hardware path emits a validated `r9700_native` NPZ/evidence set, `native_r9700.kv_cache` writes the mlx-lm prompt cache, and `native_r9700.serving` imports it for final-token decode.
- R9700 Prefill Service target: persistent process, resident/prepacked model handles, reusable buffers, warm-request evidence, and Engine Adapters; F1 is not yet complete.
- Portable Inference Device Platform target: the TinyGPU Device Owner remains the sole macOS hardware authority; the Inference HAL, Kernel Packs, conformance, and service adoption advance through P1–P4 gates.
- Path A is a historical correctness control: `tinygrad_kv_worker.harness` runs tinygrad prefill on R9700, exports an `S-1` `.safetensors` prompt cache, and passes only the final prompt token to mlx-lm.
- CPU Reference Producer flow: `native_r9700.prefill` may run the NumPy oracle and emit the same NPZ schema, but `producer_kind=cpu_reference` is never native hardware acceptance.
- Acceptance gate: producer path `P` must match native mlx-lm baseline `R` token-for-token. Optimized WMMA/attention intermediates may use reviewed bounded tolerances; semantic similarity alone is not acceptance.
- `producer_kind` is load-bearing. `r9700_native` is accepted only with validated, request-bound hardware evidence; missing, stale, mismatched, or malformed evidence fails closed.
- Serving fallback is allowed only before cache acceptance. After acceptance, decode failures must not silently recompute or repair the offloaded prefix.

## Key Directories

- `native_r9700/` — current service, producer/oracle, cache, parity, benchmark, runtime, model-binding, Kernel Pack foundation, and Qwen contract code.
- `tinygrad_kv_worker/` — historical Path A harness/exporter retained as a correctness control.
- `tests/native_r9700/` — regression and contract tests for accepted C1R/C2R behavior plus current runtime, benchmark, kernel, and Qwen work.
- `tests/native_r9700/fixtures/` — committed Llama/Qwen oracle fixture data and schemas.
- `tests/` — Path A controls plus native AMDev/runtime contract coverage.
- `docs/` — current architecture, design, roadmap, implementation plan, references, ADRs, validation results, and active command ledger.
- `docs/archive/` — completed/superseded task packets, implementation plans, design specs, and diagnostic handoffs; historical evidence only.
- `.superpowers/swarm/` — swarm ledger/reports. Use `progress.md` for freshest status when it conflicts with older reports.
- `experiments/native-r9700-runtime/` — native runtime probes and stale negative controls.
- `logs/`, `build/`, `mlx_models/`, `artifacts/` — local/generated; do not commit unless a task explicitly promotes a fixture/report.

## Development Commands

Use the pinned interpreter from `docs/tasks/native-r9700-producer/validation-commands.md`; do not rely on `python3` from `PATH`.

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
```

Common checks:

```sh
$PY -m pytest tests -v
$PY -m pytest tests/native_r9700 -v
$PY -m py_compile tinygrad_kv_worker/harness.py
git diff --check
```

Focused examples:

```sh
$PY -m pytest tests/native_r9700/test_serving.py -v
$PY -m pytest tests/native_r9700/test_runtime_contract.py -q
$PY -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Native Python CLIs:

```sh
$PY -m native_r9700.loader --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
$PY -m native_r9700.prefill --model <mlx-model-dir> --token-ids-json '[...]' --producer-kind cpu_reference --out <prefill.npz> --log <prefill.log>
$PY -m native_r9700.kv_cache --prefill-npz <prefill.npz> --out <prompt-cache.safetensors> --log <kv-cache.log>
$PY -m native_r9700.serving --model <mlx-model-dir> --fixtures-dir tests/native_r9700/fixtures --threshold-tokens 128 --max-new-tokens 4 --artifacts-dir logs/c2-serving --json logs/c2-serving/result.json --log logs/c2-serving/run.log
```

Current C++ runner build shape (kept in sync with `RUNNER_SOURCES` in `tests/native_r9700/test_block_prefill_runtime_contract.py`):

```sh
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp native_r9700/runtime_contract.cpp \
  native_r9700/prefill_npz.cpp native_r9700/vram_layout.cpp \
  native_r9700/vram_allocator.cpp native_r9700/dynamic_page_table.cpp \
  native_r9700/resident_memory.cpp native_r9700/vram_smoke_asset.cpp \
  native_r9700/hsa_code_image_asset.cpp native_r9700/model_weight_binder.cpp \
  native_r9700/amdev_session.cpp native_r9700/kernel_catalog.cpp \
  native_r9700/device_memory.cpp native_r9700/hardware_lock.cpp \
  native_r9700/llama_stage_layout.cpp native_r9700/llama_layer_executor.cpp \
  native_r9700/kernel_assets.cpp native_r9700/runtime.cpp \
  native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run
build/native-r9700-runtime/native_r9700_runner --kernel-proof
build/native-r9700-runtime/native_r9700_runner --transfer-proof --bytes 20480
build/native-r9700-runtime/native_r9700_runner --vram-smoke
build/native-r9700-runtime/native_r9700_runner --native-prefill-proof \
  --model <mlx-model-dir> --token-ids-json '[...]' \
  --out <prefill.npz> --log <prefill.log>
```

Phase 0 GPU parity control command pattern:

```sh
DEV=AMD JITBEAM=2 HF_HOME=${HOME}/Development/ml/models \
  $PY -m tinygrad_kv_worker.harness \
  --gguf mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf \
  --mlx mlx_models/meta-Llama-3.2-1B-Instruct \
  --out docs/path-a-validation-results.md --run-tag meta-f16-final
```

## Code Conventions & Common Patterns

- Prefer small, explicit modules and fail-loud validation over generic abstractions.
- Keep `native_r9700` tinygrad-free except for explicitly labeled comparison/control commands.
- Preserve the mlx-lm serialized-adapter contract: prompt cache contains the `S-1` prefix and the final prompt token is passed to `generate_step`.
- Validate shapes/dtypes/geometry at boundaries. Accepted Llama control: 16 layers, 8 KV heads, head dim 64, fp16 K/V shape `(1, 8, N, 64)`; do not apply it to Qwen hybrid state.
- Use custom error classes with precise messages (`ConfigError`, `PrefillError`, `KVCacheError`, `ParityError`, `NativePrefillError`, `PrimitiveError`).
- Write cache artifacts atomically: validate in memory, write temp sibling, `os.replace`, and clean temp/output on failure.
- Redact sensitive CLI inputs in logs (`--prompt`, `--token-ids-json`).
- C++ logs should use reviewable `key: value` fields: command, timestamp, substrate/device identity, model/config or no-model note, input shape/prompt length, comparison/digest, `failure_stage`, `failure_text`, `exit_status`.
- For C++ register/runtime work, source-ground offsets and bitfields from tinygrad/generated AMD headers or documented traces; do not guess.
- Prefer direct `xcrun clang++` build commands already documented here over inventing CMake/Make/Ninja infrastructure.
- No network/TCP transport before focused security/transport review. Prefer local subprocess/file handoff, Unix-socket JSON, or stdio JSON.

## Important Files

- `CONTEXT.md` — canonical cache, product, platform, and measurement language.
- `docs/ARCHITECTURE.md` — two-product boundaries, ownership, layers, and current accepted baseline.
- `docs/DESIGN.md` — TinyGPU/HAL/Kernel Pack/service/cache/numerical/security contracts.
- `docs/ROADMAP.md` — current F1–F6, P1–P5, Q1 capability ordering and G0–G3 integration gates.
- `docs/IMPLEMENTATION_PLAN.md` — current two-track high-level implementation plan.
- `docs/tasks/r9700-products/README.md` — current supervisor/swarm task packet index and concurrency map.
- `docs/REFERENCES.md` and `docs/upstream-reference-manifest.yaml` — source-reuse roles and immutable upstream pins.
- `docs/adr/0006-two-products-independent-tracks.md` — co-equal product and independent-track decision.
- `docs/adr/0007-tinygpu-owner-portable-hal.md` — TinyGPU ownership and portable HAL decision.
- `docs/tasks/native-r9700-producer/validation-commands.md` — active F/P/Q command ledger; historical exact commands are linked from it.
- `docs/archive/README.md` — completed/superseded task packets, plans, specs, and diagnostic history.
- `docs/path-a-validation-results.md` — Path A validation and native/reference acceptance record.
- `.superpowers/swarm/progress.md` — freshest current status and ready/blocked work.
- `native_r9700/prefill.py` — Llama CPU oracle plus producer-kind validation and native-worker CLI orchestration.
- `native_r9700/native_worker.py` — fail-closed hardware runner invocation and acceptance-evidence validation.
- `native_r9700/kv_cache.py` — NPZ-to-mlx-lm prompt-cache emitter.
- `native_r9700/parity.py` — producer/baseline parity harness.
- `native_r9700/serving.py` — mlx-lm imported-cache adapter, acceptance, and fallback boundary.
- `native_r9700/benchmark.py` — evidence-gated native benchmark rows plus CPU-reference baseline and Path A control rows.
- `native_r9700/runtime.h`, `native_r9700/runtime.cpp`, `native_r9700/runner.cpp` — native C++ runtime, model execution, and proof commands.

## Runtime/Tooling Preferences

- Python: `${HOME}/.pyenv/versions/3.12.8/bin/python3`.
- Python dependencies are source/import-grounded, not lockfile-grounded: `pytest`, `numpy`, `safetensors`, `mlx`, `mlx_lm`, `tinygrad` for specific harness paths.
- No package/build metadata is present (`pyproject.toml`, `requirements*.txt`, `setup.py`, `Makefile`, `CMakeLists.txt` absent). Run modules directly from repo root.
- C++: macOS `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra`.
- Tinygrad comparison/control env: `DEV=AMD`, `JITBEAM=2`, `HF_HOME=${HOME}/Development/ml/models`; TinyGPU discovery may need `PYTHONPATH=${HOME}/Development/ml/tools/tinygrad`.
- Hardware target: AMD Radeon AI PRO R9700, `1002:7551`, `gfx1201`, via TinyGPU.app `APLRemotePCIDevice` / `PCIIface`. Stale libusb/`USBIface` probes are negative controls, not acceptance evidence.
- `.gitignore` excludes `.worktrees/`, `__pycache__/`, `*.pyc`, `mlx_models/`, `logs/`, and `build/`.

## Testing & QA

- Test framework: pytest only.
- Most tests are hardware-free. Hardware proof commands are supervisor/manual commands, not normal pytest runs.
- C++ runtime tests compile into pytest temp dirs and use fake bridge scripts/marker logs for wrapper contracts.
- Fixture-heavy tests may skip if optional local model/fixture files are absent; committed fixtures live under `tests/native_r9700/fixtures/`.
- RED/GREEN pattern is common: add or update a focused failing contract first, verify the failure is meaningful, implement, rerun focused tests, then run broader native/full regressions as scope warrants.
- For docs-only changes, `git diff --check` is the required minimum verification.
- For source changes, run the narrow focused pytest command covering the changed behavior, then a broader relevant suite (`tests/native_r9700 -v` or `tests -v`) before claiming completion.
- For native/GPU claims, require a fresh log under `logs/` with hardware identity and `exit_status: 0`; never infer native acceptance from CPU-reference tests.
- Swarm agents may research or edit narrow slices, but supervisor owns verification, review gates, ledger updates, and checkpoint commits.
