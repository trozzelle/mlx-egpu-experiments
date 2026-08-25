# Repository Guidelines

## Project Overview

This repo builds a prefill-offload path for AMD Radeon AI PRO R9700 eGPU + mlx-lm/oMLX decode on Apple Silicon. The durable boundary is the mlx-lm-compatible prompt-cache artifact: a producer emits KV state, a consumer imports that cache, then decodes from the final prompt token.

Current priority: shortest working vertical slice to a user-usable native R9700 prefill worker. Do not default to exhaustive primitive proofs, full proof ladders, or proof-complete hardware claims unless a specific production blocker requires them. Record honest limitations; never relabel CPU/NumPy evidence as native R9700 acceptance.

Qwen3.8-27B is a product goal, but it is a separate target-expansion slice from the Llama C1/C2 acceptance path because the local Qwen target uses a different MLX-VLM/quantized/hybrid-cache ABI.

## Architecture & Data Flow

- Canonical vocabulary lives in `CONTEXT.md`: distinguish in-memory **KV cache** from serialized **prompt cache**; the **prefill producer** owns KV truth; the **prefill consumer** must not recompute the prefilled prefix.
- Path A validation: `tinygrad_kv_worker.harness` runs tinygrad prefill on R9700, `tinygrad_kv_worker.exporter` writes `.safetensors`, mlx-lm imports the S-1 prompt cache, then `generate_step` receives only the final prompt token.
- Path C target: `native_r9700` should run model-forward prefill tensor work on macOS TinyGPU.app / `APLRemotePCIDevice` / `PCIIface` (`pci_id 1002:7551`, `gfx1201`) without tinygrad, then emit the same prompt-cache ABI.
- CPU-reference flow: `native_r9700.prefill` emits NPZ K/V (`layer{i}_K`, `layer{i}_V`, `n_prefix`, `producer_kind`), `native_r9700.kv_cache` converts it to mlx-lm `.safetensors`, and `native_r9700.serving` validates/imports that cache before mlx-lm decode.
- Acceptance gate: producer path `P` must match native mlx-lm baseline `R` token-for-token. Semantic similarity is not acceptance.
- `producer_kind` is load-bearing. `cpu_reference` is oracle/regression evidence only; `r9700_native` must fail closed until it emits a validated hardware-backed cache.
- Serving fallback is allowed only before cache acceptance. After a prompt cache is accepted, decode failures must not silently recompute or repair the offloaded prefix.

## Key Directories

- `native_r9700/` — Path C Python/C++ implementation. Includes model config/loading, NumPy oracle primitives, prefill/cache/parity/serving CLIs, and native runtime shell.
- `tinygrad_kv_worker/` — Phase 0 / Path A harness and exporter for tinygrad-to-mlx-lm prompt-cache validation.
- `tests/native_r9700/` — staged C1/C2 tests for loader, primitives, attention, prefill, cache emitter, parity, serving, and runtime contracts.
- `tests/native_r9700/fixtures/` — committed oracle fixture data and schemas used by C1/C2 tests.
- `tests/` — Phase 0 harness/exporter tests plus native AMDev C++ probe contracts.
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

C++ runtime build/run shape:

```sh
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp \
  native_r9700/amdev_session.cpp native_r9700/device_memory.cpp \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run
build/native-r9700-runtime/native_r9700_runner --kernel-proof
build/native-r9700-runtime/native_r9700_runner --transfer-proof --bytes 20480
build/native-r9700-runtime/native_r9700_runner --native-prefill-proof \
  --model <mlx-model-dir> --token-ids-json '[...]' --out <prefill.npz> --log <prefill.log>
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
- Preserve the S-1 prompt-cache contract: cache contains prefix tokens; final prompt token is passed to mlx-lm `generate_step`.
- Validate shapes/dtypes/geometry at boundaries. Llama 3.2 1B first target: 16 layers, 8 KV heads, head dim 64, fp16 K/V shape `(1, 8, N, 64)`.
- Use custom error classes with precise messages (`ConfigError`, `PrefillError`, `KVCacheError`, `ParityError`, `NativePrefillError`, `PrimitiveError`).
- Write cache artifacts atomically: validate in memory, write temp sibling, `os.replace`, and clean temp/output on failure.
- Redact sensitive CLI inputs in logs (`--prompt`, `--token-ids-json`).
- C++ logs should use reviewable `key: value` fields: command, timestamp, substrate/device identity, model/config or no-model note, input shape/prompt length, comparison/digest, `failure_stage`, `failure_text`, `exit_status`.
- For C++ register/runtime work, source-ground offsets and bitfields from tinygrad/generated AMD headers or documented traces; do not guess.
- Prefer direct `xcrun clang++` build commands already documented here over inventing CMake/Make/Ninja infrastructure.
- No network/TCP transport before focused security/transport review. Prefer local subprocess/file handoff, Unix-socket JSON, or stdio JSON.

## Important Files

- `CONTEXT.md` — canonical terms and project language.
- `docs/ARCHITECTURE.md` — producer/consumer boundary and high-level flows.
- `docs/DESIGN.md` — KV interchange, producer, validation, runtime-discovery, and serving contracts.
- `docs/ROADMAP.md` — current F1–F6, P1–P5, Q1 capability ordering and G0–G3 integration gates.
- `docs/adr/0004-macos-substrate-selection.md` — accepted macOS TinyGPU/AMDev substrate.
- `docs/adr/0005-cpu-reference-is-not-native-r9700-producer.md` — critical native-acceptance correction.
- `docs/pinned-upstream-interfaces.md` — mlx-lm prompt-cache ABI, TinyGPU/tinygrad AMD facts, oMLX notes.
- `docs/IMPLEMENTATION_PLAN.md` — current two-track high-level implementation plan; archived phase packets are indexed by `docs/archive/README.md`.
- `docs/tasks/native-r9700-producer/validation-commands.md` — exact command ledger; add discovered commands here, not placeholders.
- `docs/path-a-validation-results.md` — validation report and reclassified reference/native status.
- `.superpowers/swarm/progress.md` — freshest swarm status ledger.
- `native_r9700/config.py` — strict Llama config validation.
- `native_r9700/loader.py` — MLX safetensors metadata/provenance loader.
- `native_r9700/prefill.py` — CPU/NumPy full-layer reference producer; `r9700_native` remains fail-closed until implemented.
- `native_r9700/kv_cache.py` — NPZ-to-mlx-lm prompt-cache emitter.
- `native_r9700/parity.py` — C1 P/R parity harness.
- `native_r9700/serving.py` — C2 imported-cache serving wrapper.
- `native_r9700/runtime.h`, `native_r9700/runtime.cpp`, `native_r9700/runner.cpp` — native C++ runtime shell and proof commands.

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
