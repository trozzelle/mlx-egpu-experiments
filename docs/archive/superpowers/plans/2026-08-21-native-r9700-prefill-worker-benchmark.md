# Native R9700 Prefill Worker and Real Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a tinygrad-free `r9700_native` Llama 3.2 1B fp16 prefill worker that emits the existing mlx-lm prompt-cache artifact, passes token-exact `P == R`, routes through serving for large prompts, and produces honest benchmark evidence.

**Architecture:** Keep the existing KV interchange boundary. CPU code may load weights, allocate buffers, schedule kernels, emit NPZ/safetensors, and verify logs; accepted model-forward tensor math for `producer_kind=r9700_native` must run on the AMD Radeon AI PRO R9700 through the selected macOS TinyGPU.app/APLRemotePCIDevice/PCIIface native AMDev runtime. The shortest path is to convert the proven bounded primitive chains into a resident layer/full-prefill worker, not to keep expanding proof-only ladders.

**Tech Stack:** Python 3.12.8, pytest, NumPy/MLX safetensors reference loaders, mlx-lm prompt-cache import, C++17 native AMDev runtime (`xcrun --sdk macosx clang++`), local hardware logs under `logs/`, generated benchmark artifacts under `artifacts/`.

> **Superseded implementation mechanics:** Tasks 2–3 reference the retired `native_r9700/c1_primitive_bridge.cpp` source-as-data proof lane. Do not execute those bridge-dependent steps. Use `docs/archive/superpowers/plans/2026-08-21-native-r9700-product-worker-rearchitecture.md` and its `phase-c1r-product-worker-*.md` task packets for the C1R product-worker path. The goal, acceptance gates, C2R dependency, and benchmark-after-C2 sequence remain active.

## Global Constraints

- Use `${PY}` for Python commands.
- First accepted model: `../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`.
- C1 acceptance requires `producer_kind=r9700_native`, no production tinygrad import/call, R9700/eGPU model-forward tensor math, existing `S-1` KV cache artifact, and `P == R` across the Phase 0 prompt suite.
- CPU reference (`producer_kind=cpu_reference`) stays useful only as oracle/reference evidence.
- C2 acceptance requires large prompts to route through the accepted `r9700_native` producer; CPU-reference serving remains a reference wrapper.
- Qwen3.8-27B is a product goal but remains a separate target-expansion phase because the current local Qwen target has incompatible MLX-VLM/quantized/hybrid-cache ABI.
- Do not record final hardware validation commands in `docs/tasks/native-r9700-producer/validation-commands.md` until the task implements the command and captures real output.
- Do not commit generated logs, model files, build outputs, or artifacts. Do not commit at all unless the user explicitly asks.

---

## Grounded Current State

- C0 is complete: macOS TinyGPU.app/APLRemotePCIDevice/PCIIface native AMDev is selected; C0A25 kernel proof and host-device transfer pass are recorded.
- C1 CPU reference is complete as an ABI oracle only: `native_r9700.prefill` emits 16-layer fp16 K/V NPZ, `native_r9700.kv_cache` emits mlx-lm-compatible `.safetensors`, and `native_r9700.parity` can check `P == R`.
- C2 CPU-reference wrapper is complete as serving/reference evidence: `native_r9700.serving` and `tinygrad_kv_worker.harness --c2-serving` can route a Llama prompt through imported-cache serving with explicit `producer_kind=cpu_reference`.
- Native hardware proof coverage is broad but still proof-boundary-based: primitive chains cover integrated attention, softmax/context, MLP down output cols0:2048, attention heads0:31/context cols0:2048, and O-proj partial follow-up bands; full native prefill is still open because those chains consume fixture/oracle boundaries instead of one resident model-forward dataflow.
- The next product gap is not another arbitrary O-proj band. It is a worker that takes real tokens/model weights, keeps intermediate state resident across stages, writes real K/V, and exposes the same `prefill -> kv_cache -> serving` route under `producer_kind=r9700_native`.

---

## Files and Responsibilities

- Modify `native_r9700/prefill.py`: route `producer_kind=r9700_native` to the native worker, preserve CPU reference, and keep CLI output/log identity honest.
- Create `native_r9700/native_worker.py`: Python orchestration boundary for native worker commands, artifact paths, JSON parsing, and fail-closed status handling.
- Modify `native_r9700/runtime.h`, `native_r9700/runtime.cpp`: reusable resident-buffer lifecycle, stage timing, transfer counters, and explicit failure-stage reporting for model-prefill runs.
- Modify `native_r9700/runner.cpp`: add narrow worker/proof commands (`--native-layer0-proof`, `--native-prefill-proof`, later `--benchmark-prefill`) that call the same runtime as production.
- Modify `native_r9700/c1_primitive_bridge.cpp`: move from fixture-specific proof wrappers toward parameterized layer/full-prefill execution using uploaded model/prompt buffers.
- Modify `native_r9700/ref_fixtures.py` and fixture tests only when a new oracle slice is required to verify layer/full-prefill output.
- Modify `native_r9700/kv_cache.py` only if the native worker emits a new NPZ field needed for provenance; do not change the KV tensor schema without an ADR.
- Modify `native_r9700/parity.py`: add native-producer result fields and gate output for real C1.
- Modify `native_r9700/serving.py`: enforce native producer identity for real C2 and benchmark routes.
- Modify `tinygrad_kv_worker/harness.py`: keep `--c2-serving` as delegation only; do not duplicate serving logic.
- Test under `tests/native_r9700/` and `tests/test_harness_c2_serving.py`.
- Update `docs/tasks/native-r9700-producer/validation-commands.md`, `docs/path-a-validation-results.md`, and `.superpowers/swarm/progress.md` only with observed commands/results.

---

### Task 1: Native worker command contract and fail-closed spine

**Files:**
- Create: `native_r9700/native_worker.py`
- Modify: `native_r9700/prefill.py`
- Modify: `tests/native_r9700/test_prefill.py`
- Test: `tests/native_r9700/test_runtime_contract.py`

**Interfaces:**
- Consumes: existing `producer_kind` values `cpu_reference` and `r9700_native`.
- Produces: `native_worker.run_native_prefill(model_dir: str, token_ids: Sequence[int], out_npz: Path, log_path: Path) -> dict[str, object]` with fields `producer_kind`, `native_prefill_acceptance`, `runtime_substrate`, `hardware_log_path`, `prefill_npz_path`, `kernel_count`, `transfer_bytes`, `failure_stage`, `exit_status`.

- [ ] **Step 1: Write fail-closed tests**

```python
def test_r9700_native_prefill_fails_closed_until_worker_accepts(model_dir, tmp_path):
    result = run_prefill_cli(
        "--model", str(model_dir),
        "--token-ids-json", "[1,2,3,4,5,6]",
        "--producer-kind", "r9700_native",
        "--out", str(tmp_path / "prefill.npz"),
        "--log", str(tmp_path / "prefill.log"),
        check=False,
    )
    assert result.returncode != 0
    assert "producer_kind: r9700_native" in (tmp_path / "prefill.log").read_text()
    assert "native_prefill_acceptance: open" in (tmp_path / "prefill.log").read_text()
```

- [ ] **Step 2: Add `native_worker.py`**

Implement only the orchestration shell: build runner argv, run it with `subprocess.run(check=False)`, parse JSON/log output, and reject missing `producer_kind=r9700_native`, missing `native_prefill_acceptance=pass`, missing `prefill_npz_path`, or nonzero exit.

- [ ] **Step 3: Wire `prefill.py` routing**

`cpu_reference` keeps the current NumPy implementation. `r9700_native` calls `run_native_prefill(...)`; until Task 5 passes, it returns a loud error and does not write an accepted NPZ.

- [ ] **Step 4: Verify**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_runtime_contract.py -q
```

Expected: pass, with `r9700_native` fail-closed behavior covered.

---

### Task 2: Resident layer0 dataflow proof, no fixture boundaries

**Files:**
- Modify: `native_r9700/runtime.h`
- Modify: `native_r9700/runtime.cpp`
- Modify: `native_r9700/runner.cpp`
- Modify: `native_r9700/c1_primitive_bridge.cpp`
- Modify: `tests/native_r9700/test_runtime_contract.py`
- Optional fixture extension: `native_r9700/ref_fixtures.py`, `tests/native_r9700/test_ref_fixtures.py`

**Interfaces:**
- Consumes: model directory, prompt token ids, existing layer0 oracle fixtures.
- Produces: runner mode `--native-layer0-proof` that emits JSON/log fields `acceptance_scope=hardware_layer0_resident_dataflow`, `native_prefill_acceptance=open`, `layer_index=0`, `resident_boundary_count`, `kernel_count`, `transfer_bytes`, `k_shape`, `v_shape`, `hidden_shape`, `failure_stage`, `exit_status`.

- [ ] **Step 1: Write runtime contract tests**

Assert the fake bridge fails if `--native-layer0-proof` reports fixture-sourced stage inputs, missing transfer counters, missing kernel counts, or `native_prefill_acceptance=pass` before full prefill exists.

- [ ] **Step 2: Add runner mode**

Add `--native-layer0-proof --model <mlx-model-dir> --token-ids-json '[...]' --json <path> --log <path>` to `runner.cpp`. The mode must use the same runtime allocation/dispatch/readback path as production, not a test-only code path.

- [ ] **Step 3: Replace fixture seams with resident stage outputs for layer0**

For prompt-0 first, upload embeddings and layer0 weights, run attention and MLP stages against resident buffers, write layer0 K/V plus post-layer hidden output, and compare against CPU fixture/oracle after readback. CPU comparison is allowed only after GPU computation completes.

- [ ] **Step 4: Verify on hardware**

Build runner and execute the new layer0 proof. Acceptance for this task is not C1; it is `layer0_resident_dataflow_status: pass` with `native_prefill_acceptance: open` and no fixture-sourced intermediate boundaries.

- [ ] **Step 5: Regression**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -q
```

Expected: pass.

---

### Task 3: Native prefill NPZ writer for one prompt

**Files:**
- Modify: `native_r9700/runner.cpp`
- Modify: `native_r9700/c1_primitive_bridge.cpp`
- Modify: `native_r9700/native_worker.py`
- Modify: `native_r9700/prefill.py`
- Modify: `tests/native_r9700/test_prefill.py`
- Modify: `tests/native_r9700/test_runtime_contract.py`

**Interfaces:**
- Consumes: Task 2 resident layer0 dataflow.
- Produces: runner mode `--native-prefill-proof` and accepted NPZ schema identical to CPU reference: `layer_{i}_K` and `layer_{i}_V`, fp16, `(1, 8, S-1, 64)` for all 16 layers, plus metadata `producer_kind=r9700_native` in logs/JSON.

- [ ] **Step 1: Add tests for NPZ schema and identity**

Use a fake native worker result to assert `prefill.py --producer-kind r9700_native` writes/returns only an NPZ with 16 K/V layers, fp16 geometry, `producer_kind=r9700_native`, and hardware evidence fields. Assert `cpu_reference` cannot be relabeled.

- [ ] **Step 2: Implement layer loop**

Extend the resident dataflow from layer0 to layers 0..15. Stream weights per layer if full residency is slower or riskier; log each upload chunk and per-layer kernel count. The hidden state passed from layer `i` to `i+1` must stay GPU-computed; CPU may read back for verification only, not to feed the next accepted layer.

- [ ] **Step 3: Emit NPZ from native output**

Write K/V tensors in the existing `native_r9700.kv_cache` input layout. Reject partial layer output, dtype mismatch, shape mismatch, or missing hardware evidence before writing success.

- [ ] **Step 4: Run one-prompt native proof**

Use prompt-0 (`S=6`) first because it already has compact oracle fixtures and accepted CPU-reference serving output. Expected task result: `native_prefill_acceptance: pass` for prompt-0 only, with all 16 layers present and all model-forward tensor math on R9700.

- [ ] **Step 5: Verify cache conversion**

Run the produced NPZ through:

```sh
${PY} -m native_r9700.kv_cache --prefill-npz <native-prefill.npz> --out <native-prompt-cache.safetensors> --log <kv-cache.log>
```

Expected: cache loads with 32 tensors, `offset=S-1`, 16 layers, `(1, 8, S-1, 64)` K/V.

---

### Task 4: C1 native parity gate across Phase 0 prompts

**Files:**
- Modify: `native_r9700/parity.py`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Modify: `docs/path-a-validation-results.md`
- Modify: `.superpowers/swarm/progress.md`
- Test: `tests/native_r9700/test_parity.py` or existing parity tests

**Interfaces:**
- Consumes: Task 3 native prefill NPZ/cache route.
- Produces: C1 report fields `producer_kind=r9700_native`, `P`, `R`, `gate_result`, per-prompt token exactness, prefill log paths, cache paths, kernel/transfer timing, and failure details.

- [ ] **Step 1: Add report-schema tests**

Assert parity reports distinguish `cpu_reference` and `r9700_native`, reject missing hardware log paths for native runs, and fail the gate on any `P != R` token.

- [ ] **Step 2: Run native parity for prompt-0**

Use the exact model directory from Global Constraints and compare native `P` against `baseline_r_tokens.json`/mlx-lm `R`. Fix only real K/V geometry, RoPE, position, precision, or layer-order defects surfaced by deltas.

- [ ] **Step 3: Extend to Phase 0 prompt suite**

Run short, ~200-token, and ~1000-token prompts. If runtime time is high, keep the command single-prompt addressable but produce one aggregate C1 report over all prompts.

- [ ] **Step 4: Record exact commands only after success**

Update `validation-commands.md` with the real commands and observed result. Update `docs/path-a-validation-results.md` with native Path C C1 evidence only if every prompt passes `P == R`.

- [ ] **Step 5: Verification**

Run focused parity tests and the native C1 command(s). Expected C1 acceptance: `gate_result=pass`, `producer_kind=r9700_native`, all prompts token-exact, hardware logs prove model-forward kernels.

---

### Task 5: C2 serving route uses accepted native producer

**Files:**
- Modify: `native_r9700/serving.py`
- Modify: `tinygrad_kv_worker/harness.py` only if delegation needs a new pass-through argument
- Modify: `tests/native_r9700/test_serving.py`
- Modify: `tests/test_harness_c2_serving.py`
- Modify: `docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md`

**Interfaces:**
- Consumes: Task 4 accepted native producer command/cache.
- Produces: serving result fields `route=native_producer`, `accepted_cache=true`, `producer_kind=r9700_native`, `fallback_reason`, `hardware_log_path`, `prefill_elapsed_sec`, `cache_emit_elapsed_sec`, `cache_import_elapsed_sec`, `decode_elapsed_sec`.

- [ ] **Step 1: Add serving enforcement tests**

Assert a native-serving request fails closed or falls back before cache acceptance if producer output says `cpu_reference`, omits hardware evidence, produces malformed cache, or exits nonzero.

- [ ] **Step 2: Route large prompts through native producer**

For `--producer-kind r9700_native`, serving must invoke the accepted native `prefill.py` path, then `kv_cache.py`, validate cache metadata, import with mlx-lm, and decode using the final prompt token only.

- [ ] **Step 3: Preserve fallback semantics**

Below-threshold prompts, unavailable native producer before cache acceptance, and malformed native output must fall back to native mlx-lm prefill with redacted logs. After cache acceptance, decode errors are errors, not silent prefill recompute.

- [ ] **Step 4: Run real C2 smoke**

Use at least prompt-0 and one larger Phase 0 prompt through `native_r9700.serving --producer-kind r9700_native`. Expected: imported cache accepted, decoded tokens match `R`, and JSON/log status says native R9700 C2 pass only for the native route.

- [ ] **Step 5: Harness smoke**

Run `tinygrad_kv_worker.harness --c2-serving --producer-kind r9700_native` as a thin delegate. Expected: same C2 result as direct serving, no duplicate cache validation logic in the harness.

---

### Task 6: Real benchmark phase

**Files:**
- Create: `native_r9700/benchmark.py`
- Modify: `native_r9700/runner.cpp` if a C++ timing mode is needed
- Create or modify tests: `tests/native_r9700/test_benchmark.py`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Modify: benchmark report under `docs/path-a-validation-results.md` or a task-specific report named from the actual run

**Interfaces:**
- Consumes: Task 5 native C2 route.
- Produces: benchmark JSON rows with `prompt_name`, `prompt_tokens`, `producer_kind`, `gate_result`, `prefill_elapsed_sec`, `kernel_elapsed_usec`, `transfer_h2d_bytes`, `transfer_d2h_bytes`, `transfer_elapsed_sec`, `cache_emit_elapsed_sec`, `cache_import_elapsed_sec`, `decode_elapsed_sec`, `total_elapsed_sec`, `tokens_per_sec_prefill`, `tokens_per_sec_end_to_end`, `baseline_name`, and `speedup_vs_baseline`.

- [ ] **Step 1: Add benchmark schema tests**

Assert benchmark rows require `producer_kind=r9700_native`, token-exact gate pass, timing fields, byte counters, model identity, prompt name, and hardware log path. Assert CPU-reference rows may be included only as baseline rows, never as native benchmark rows.

- [ ] **Step 2: Implement benchmark CLI**

`native_r9700.benchmark` should run the accepted native serving route plus baselines: CPU reference prefill/cache, native mlx-lm prefill/decode, and optional Path A tinygrad control when `DEV=AMD JITBEAM=2` is explicitly provided. It writes JSON and a concise Markdown report.

- [ ] **Step 3: Benchmark prompt set**

Use the Phase 0 prompt suite first: prompt-0 for smoke, prompt-1 (~200 tokens), prompt-2 (~1000 tokens). Report warmup policy, model path, command, timestamps, and whether weights were reloaded per request.

- [ ] **Step 4: Run real benchmark**

Run the benchmark only after C2 native route is accepted. Expected: every native benchmark row has `gate_result=pass` and `producer_kind=r9700_native`. If speedup is absent, report the bottleneck honestly from timing counters instead of changing acceptance criteria.

- [ ] **Step 5: Final QA**

Run:

```sh
${PY} -m pytest tests/native_r9700 -q
git diff --check
```

Expected: tests pass and whitespace check emits no output.

---

## Promotion Gates

1. **Worker spine gate:** `r9700_native` fails closed until native hardware evidence is present.
2. **Layer0 dataflow gate:** layer0 runs resident R9700 dataflow without fixture intermediate boundaries; acceptance remains open.
3. **Native prefill gate:** all 16 layers emit native hardware-backed K/V NPZ for at least prompt-0.
4. **C1 gate:** all Phase 0 prompts pass token-exact `P == R` using `producer_kind=r9700_native`.
5. **C2 gate:** large prompts route through the accepted native producer and imported-cache serving, with fallback behavior intact.
6. **Benchmark gate:** benchmark rows are produced only from accepted native C2 runs; CPU reference and Path A rows are labeled baselines/controls.

## Stop Conditions

- Stop and record a blocker if macOS AMDev runtime cannot allocate/upload/dispatch/download for model-sized buffers, if the worker cannot keep layer output resident without CPU recompute, if `P != R` persists after geometry/RoPE/layout fixes, or if hardware logs cannot prove model-forward tensor execution.
- Do not stop because CPU-reference serving passes. That is already known reference evidence.
- Do not continue primitive proof expansion when it does not shorten the path to resident full-prefill execution.
