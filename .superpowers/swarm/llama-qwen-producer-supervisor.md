# Swarm Supervisor Plan: Llama/Qwen Native Producer Delivery

## Source and resume state

- Master plan: `docs/archive/superpowers/plans/2026-08-22-llama-qwen-native-producer-delivery.md`.
- Execution packets: `docs/archive/tasks/native-r9700-producer/phase-c1r-native-llama-delivery.md`, `phase-qwen3-8-native-text-delivery.md`, and `phase-native-producer-swarm-integration.md`.
- Durable ledger: `.superpowers/swarm/progress.md`.
- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on branch `feature/native-r9700-producer` (current feature branch; every executor stays in this checkout).
- Preserved evidence: standalone C0 kernel proof, lower-BAR resident VRAM smoke, and real Llama safetensors embedding HSA smoke pass on `1002:7551` / `gfx1201`.

## Orchestration map

- Sequential blockers: shared stage ABI → source assets → serial manifest/catalog integration → stage hardware proof → complete producer artifact → token-exact parity/serving.
- Parallel waves: source-only Llama and Qwen files with disjoint targets. No more than one owner changes catalog/loader/session/runner per wave.
- Shared contracts: selected TinyGPU substrate; lower-BAR window limits; Llama `(1,8,N,64)` fp16 K/V and S-1 ABI; Qwen 64 runtime-ordered hybrid entries with `KVCache` at layers 3, 7, …, 63 and `ArraysCache` otherwise; `producer_kind=r9700_native` fail-closed identity.
- Hardware rule: supervisor alone runs one changed-premise device command after accepted source integration. Never run Llama and Qwen hardware commands concurrently.
- Quality rule: reject generic runtimes, transport/build systems, archive dependencies, CPU model math, broad proof ladders, benchmark campaigns, and unrelated cleanup.
- **Commit discipline:** Supervisor makes a small local checkpoint commit immediately after every reviewed and verified wave. Commit only that wave's source/tests/assets/reports/ledger rows; agents never commit or push. Do not defer all commits to the end. Push remains user-owned.

## Repeated wave loop

1. Read first unblocked row in both phase packets and freeze the wave interface in one task-batch context.
2. Dispatch all independent source workers together. They do not run test/lint/format/git/hardware commands and write one report each.
3. Inspect reports/diffs. Dispatch independent Llama/Qwen/common reviewers together.
4. Route only verified Critical/Important findings into one bounded fix wave. Ledger minor observations; do not open side quests.
5. Run the named focused supervisor tests once. Only then run the single appropriate hardware command.
6. Update `.superpowers/swarm/progress.md`, write exact log evidence, and make the local checkpoint commit for the reviewed wave. Push remains user-owned.
7. Repeat until a full Llama or Qwen native artifact reaches its exact parity gate.

## Wave status

| Wave | Status | Gate |
|---|---|---|
| 0. ABI/binder freeze | Done | Llama stage descriptor and Qwen affine-window/cache contracts accepted by focused tests and independent review/rereview. |
| 1. First source assets | Not started | Llama RMS/K/V and Qwen affine/state source contracts. |
| 2. Second source assets | Not started | Llama RoPE/attention/O-MLP and Qwen DeltaNet/full-attention source contracts. |
| 3. Serial asset/executor integration | Not started | One model stage hardware proof per lane. |
| 4. Complete native producers | Not started | 16-layer Llama NPZ; Qwen 64-entry hybrid artifact. |
| 5. Consumer parity/serving | Not started | Final-token exact parity; no accepted-cache recompute. |
| 6. Final review | Not started | Exact focused suites, fresh logs, diff hygiene. |

## Wave 0: ABI and binder freeze

### Shared context

- Llama stage names: `rmsnorm`, `k_projection`, `v_projection`, `rope_kv`, `attention_score`, `attention_softmax`, `attention_context`, `o_projection`, `gated_mlp`.
- Each Llama descriptor declares source/HSA identity, fixed kernarg schema, workgroup geometry, and named resident spans; values cannot come from CPU model math.
- Qwen binds only raw affine4 `(weight, scales, biases)` spans with mode 4-bit/group-64 validation and preserves runtime layer order: `KVCache` at `layer_index % 4 == 3`, `ArraysCache` otherwise.
- Agents skip tests, formatters, linters, package managers, git, and hardware. Reports go to `.superpowers/swarm/reports/lq-w0-*.md`.

### Agents

| Agent | Ledger row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| LlamaAbiOwner / LlamaAbiFix / LlamaCacheExtentFix | LQ-W0-1 | Llama layout, catalog contract, weight binding tests | Current C0/VRAM/embed evidence | `.superpowers/swarm/reports/lq-w0-llama-abi.md`; `lq-w0-llama-cache-extent-rereview.md` | Done: 3 focused tests passed; assets remain unadmitted/fail-closed. |
| QwenAbiOwner / QwenAbiFix | LQ-W0-2 | Qwen raw affine-window binder and hybrid-cache contract | Current Qwen adapter/spill evidence | `.superpowers/swarm/reports/lq-w0-qwen-abi.md`; `lq-w0-qwen-rereview.md` | Done: 15 focused tests passed. |

### Supervisor gates

- Inspect executor reports and diffs, then dispatch an independent Llama/Qwen review wave.
- Quality bar: direct metadata/span validation, no generic registry/cache abstraction, no CPU model math, and no Llama/Qwen cache crossover.
- Focused verification: `test_llama_stage_layout.py`, `test_kernel_catalog.py`, `test_qwen_text_adapter.py`, `test_qwen_hybrid_state_spill.py`.

### Wave 0 result

- Correctness: exact stage metadata now rejects undersized absolute-position cache spans, caller-controlled/unadmitted assets, malformed kernarg layouts, and invalid Qwen raw windows/state order.
- Maintainability/architecture: direct static Llama layouts and caller-owned Qwen metadata reuse existing model/spill seams; neither added a registry, transport, cache framework, or CPU model path.
- Simplicity: catalog admission remains deferred until actual reviewed assets exist; the Q projection is fused into the existing attention-score stage rather than adding a tenth stage.

## Wave 1: first source assets

### Shared context

- K/V source workers use the accepted static Llama ABI and create fresh fp16 `(1,8,N,64)` values only; cache materialization/RoPE is deferred.
- Qwen affine source performs group-64 affine4 decoding on-device only; DeltaNet state and full attention are deferred.
- Existing `llama_rmsnorm_f16.cpp` is reused unchanged. Generated HSA assets/catalog entries remain single-owner Wave 3 work.
- All three tasks are source-only and mutually disjoint. Executors skip validation, git, and hardware.

### Agents

| Agent | Ledger row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| LlamaKProjectionSource | LQ-W1 | `llama_k_projection_f16.cpp` and focused K contract | Wave 0 | `.superpowers/swarm/reports/lq-w1-llama-k-projection.md` | In progress |
| LlamaVProjectionSource | LQ-W1 | `llama_v_projection_f16.cpp` and focused V contract | Wave 0 | `.superpowers/swarm/reports/lq-w1-llama-v-projection.md` | In progress |
| QwenAffine4Source | LQ-W1 | `qwen_affine4_linear.cpp` and focused affine source contract | Wave 0 | `.superpowers/swarm/reports/lq-w1-qwen-affine4.md` | In progress |

### Supervisor gates

- Inspect reports and current source, then review K/V/Qwen lanes in parallel.
- Quality bar: direct device-only arithmetic, exact ABI, bounded windows, no fixture/CPU path, no generic quantization/runtime layer.
- Focused verification: `test_llama_kv_projection_asset.py`, `test_llama_v_projection_asset.py`, and `test_qwen_hsa_kernel_assets.py`.

## Wave LN-1: bounded numerical localization

### Shared context

- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer` on `feature/native-r9700-producer`.
- Source plans: `2026-08-23-llama-numerical-debug-plan.md`, `phase-llama-numerical-trace.md`, `phase-llama-numerical-remediation.md`.
- Native two-token prefill is structurally accepted but prompt-0 C1R decodes zero tokens and reports NaN K/V comparisons. No native parity/cache acceptance is claimed.
- Oracle and native trace JSON schema: `token_index`, `layer_index`, `stage`, `buffer`, `shape`, `dtype`, `byte_count`, `sha256`, `finite_count`, optional `raw_path`; native adds `kernarg_hex`, image digest, GPU VA, scalars.
- Executors do not run tests, hardware, formatters, linters, package managers, or git. Reports go in `.superpowers/swarm/reports/`.

### Agents

| Agent | Ledger row | Target | Depends on | Report | Status |
|---|---|---|---|---|---|
| LlamaStageOracle | LN-1A | Python layer-0/token-0 oracle and tests | Schema frozen | `ln-1a-oracle.md` | In progress |
| LlamaNativeTrace | LN-1B | C++ bounded resident trace and tests | Schema frozen | `ln-1b-native-trace.md` | In progress |

### Supervisor gates

- Inspect reports/diffs, then dispatch independent reviews for LN-1A and LN-1B.
- Quality bar: request-scoped/bounded artifacts, no CPU values into accepted outputs, no generic trace framework, unchanged K/V cache/serving.
- Focused verification: oracle test, trace runtime contracts, C0 kernel proof, VRAM smoke, then one layer-0/token-0 `hidden` trace comparison.
