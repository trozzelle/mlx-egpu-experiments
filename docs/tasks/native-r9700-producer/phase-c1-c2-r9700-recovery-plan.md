# Phase C1R/C2R: R9700/eGPU producer recovery plan

## Purpose

This plan restores the original objective after ADR 0005: build and integrate a producer whose Llama 3.2 1B prefill model-forward tensor work runs on the AMD Radeon AI PRO R9700/eGPU. The existing CPU/NumPy producer remains a reference oracle and ABI fixture generator only.

## Grounded facts

- Selected local substrate: macOS TinyGPU.app / `APLRemotePCIDevice` / `PCIIface` native AMDev, PCI `1002:7551`, arch `gfx1201` (ADR 0004).
- C0 hardware transfer proof is complete: `logs/c0b-native-amdev-sdma-transfer.log` records SDMA host-device transfer pass, CPU comparison pass, `failure_stage: none`, and `exit_status: 0`.
- C0 minimal kernel proof is complete: `logs/c0p-native-amdev-kernel-load-fix.log` records `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `exit_status: 0`, and exact `out[i]=in[i]+1` readback.
- Reusable runtime shell exists in `native_r9700/runtime.{h,cpp}` and `native_r9700/runner.cpp`, but its comments and implementation still say real socket/BAR/SDMA/compute mechanics are deferred.
- CPU reference producer exists in `native_r9700/prefill.py`. It loads MLX safetensors, runs all 16 Llama layers in NumPy, emits per-layer fp16 K/V NPZ, and is covered by tests under `tests/native_r9700/`.
- Cache emitter exists in `native_r9700/kv_cache.py`; it converts the NPZ shape into mlx-lm-compatible prompt-cache safetensors.
- Parity harness exists in `native_r9700/parity.py`; it can compare produced tokens against mlx-lm baseline `R` and write Path C results.
- Serving wrapper exists in `native_r9700/serving.py`; it proves imported-cache consumption, threshold fallback, redaction, and local-file/subprocess safety against the CPU reference producer.
- Qwen3.8-27B local candidate is an MLX `mlx-vlm` Qwen3VL 4-bit model with hybrid linear/full attention and mRoPE. It is not compatible with the C1 Llama prompt-cache ABI and stays out of C1R/C2R acceptance.

## Corrected status

| Area | State |
|---|---|
| C0 substrate | Done; selected for native work. |
| C1 CPU reference | Done as reference/ABI oracle only. |
| C1R Native R9700 producer | Open. |
| C2 CPU reference wrapper | Done as reference wrapper/security evidence only. |
| C2R R9700 serving integration | Blocked on C1R. |
| C3 backend decision/prototype | Blocked on C2R evidence. |

## Acceptance definitions

### C1R accepted

C1R is accepted only when all are true:

1. The producer route is identified as `r9700_native` in command output, JSON result, and logs.
2. Llama 3.2 1B fp16 prefill model-forward tensor work runs on the R9700/eGPU through the selected macOS native substrate.
3. CPU code may orchestrate, load weights, allocate buffers, dispatch kernels, and verify results, but it must not perform the accepted model-forward tensor math for the `r9700_native` route.
4. The route emits the existing `S-1` prompt-cache artifact consumed by mlx-lm.
5. `P == R` token-for-token across the Phase 0 prompt suite.
6. Logs prove hardware execution: substrate, PCI id, arch, producer kind, kernel count or named kernel stages, transfer bytes, CPU comparison fields for diagnostic kernels, failure stage/text, and exit status.
7. No production producer path imports or calls tinygrad.

### C2R accepted

C2R is accepted only when all are true:

1. Large prompts route through the accepted `r9700_native` C1R producer.
2. The wrapper still falls back to native mlx-lm prefill before cache acceptance for small prompts, unavailable producer, and malformed producer output.
3. After cache acceptance, decode never silently recomputes prompt prefill.
4. Serving results match the producer-swap parity/quality gate for the Phase 0 prompt set.
5. Logs and JSON preserve redaction and local-only transport assumptions.

## Execution strategy

### 2026-08-21 product-smoke pivot

User steering narrowed the immediate acceptance path after the C1R primitive-chain confidence work: stop default exhaustive O-proj expansion, accept a real Llama imported-cache product smoke with explicit CPU-reference labeling, and move C2 forward through the external harness/serving seam.

This creates two tracked statuses:

- Product/reference path accepted: Llama `native_r9700.prefill` CPU-reference NPZ -> `native_r9700.kv_cache` safetensors -> `native_r9700.serving` imported cache; `tinygrad_kv_worker.harness --c2-serving` delegates to that wrapper and passed prompt-0 smoke.
- Original native R9700 C1R/C2R remains open under the acceptance definitions above; `r9700_native` must continue to fail closed until it emits a validated cache and hardware evidence.

Qwen3.8-27B remains outside this Llama path and requires a separate target-expansion phase.


Use supervised waves. Each implementation task must skip project-wide validation while running; the supervisor runs focused tests and full gates after review. Agents may edit overlapping files; contracts below define shared interfaces.

Shared implementation contract:

- Add explicit producer identity fields before hardware work can pass: `producer_kind` values are `cpu_reference` and `r9700_native`.
- Keep CPU reference APIs available, but labels must prevent using them as native acceptance.
- Reuse existing NPZ and safetensors cache artifacts; do not change the KV interchange schema unless a new ADR is written.
- Use the C0A25 kernel/load/store/kernarg facts as source grounding; do not re-derive the architecture from a wrong RDNA premise. Target is RDNA4/gfx1201.
- No TCP or non-local transport in C1R/C2R.
- Every hardware run writes under `logs/` and stays uncommitted.

## Wave 1 — Correct labels and reusable hardware boundary

### Task C1R-1: Producer identity guardrail

**Target:** `native_r9700/prefill.py`, `native_r9700/parity.py`, `native_r9700/serving.py`, tests under `tests/native_r9700/`, docs in this task set.

**Change:**

1. Add explicit producer identity to results/logs: `producer_kind=cpu_reference` for the current NumPy route.
2. Add CLI/report fields that make `cpu_reference` visible in parity and serving outputs.
3. Add tests proving CPU reference output cannot be counted as `r9700_native` acceptance.
4. Do not add fake `r9700_native` mode; unavailable hardware mode must fail closed until implemented.

**Acceptance:** focused tests pass; existing CPU-reference parity still passes as reference; docs and JSON distinguish `cpu_reference` from `r9700_native`.

### Task C1R-2: Promote proven C0 hardware path into reusable runtime

**Target:** `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, `native_r9700/runtime.{h,cpp}`, `native_r9700/runner.cpp`, `tests/native_r9700/test_runtime_contract.py`.

**Change:**

1. Port or wrap the proven C0A25 socket/BAR/SDMA/compute-dispatch mechanics into `native_r9700::RuntimeSession` without mutating the frozen C0 probe.
2. Replace the current `--kernel-proof` unavailable path with a real hardware proof mode that executes the known `out[i]=in[i]+1` kernel through the reusable runtime.
3. Preserve the no-hardware `--lifecycle-dry-run` contract.
4. Log the same C0A25 acceptance fields plus `producer_kind: hardware_probe` or equivalent non-model label.

**Acceptance:** hardware proof build/run command exits 0 and writes a log equivalent to C0A25 pass; focused runtime tests pass; no tinygrad import/call.

## Wave 2 — Native tensor primitive ladder

### Task C1R-3: R9700 tensor fixture seam

**Target:** `native_r9700/prefill.py`, new or existing fixture helpers under `native_r9700/ref_fixtures.py`, `tests/native_r9700/fixtures/`, `tests/native_r9700/test_ref_fixtures.py`.

**Change:**

1. Generate small deterministic per-primitive and per-layer fixtures from the CPU reference path: embeddings, RMSNorm input/output, Q/K/V projections, RoPE-applied Q/K, attention scores/probabilities/context, O projection, MLP gate/up/down, residuals, final K/V.
2. Keep fixture files small enough for the repo; large hardware logs/artifacts stay under `logs/`.
3. Record exact tensor names, shapes, dtypes, tolerances, and digest metadata.

**Acceptance:** fixture tests prove determinism and schema; fixtures cover at least layer 0 and one later layer for Llama 3.2 1B geometry.

### Task C1R-4: Hardware memory residency and transfer manager

**Target:** `native_r9700/runtime.{h,cpp}`, optional narrow C++ helper files, tests.

**Change:**

1. Implement reusable buffer allocation, upload, download, and teardown APIs backed by the C0 AMDev memory-manager path.
2. Support weight/input/output buffers large enough for Llama 3.2 1B layer slices; if full model residency is not feasible in one allocation, implement explicit streaming with logged chunks.
3. Log allocation sizes, transfer bytes, and failure stages.

**Acceptance:** hardware transfer tests round-trip fixture-sized fp16/fp32 buffers with exact byte comparison; failure paths are loud and leave no partial accepted model output.

### Task C1R-5: Hardware primitive kernels

**Target:** native R9700 kernel blobs/source-generation utilities, `native_r9700/runtime.{h,cpp}`, tests.

**Change:**

Implement and verify the minimum kernels needed for Llama 3.2 1B prefill:

1. fp16/fp32 elementwise add/residual and scaling.
2. RMSNorm reductions and normalization.
3. Dense projection kernels for `x @ W.T` shapes used by Q/K/V/O and MLP gate/up/down.
4. RoPE split-half application using Llama-3 sidecar scaling.
5. Causal attention score computation, masked softmax, and V-weighted context.
6. SiLU and gated MLP multiply.

Use the CPU reference fixtures as oracle. Start with small shapes; then run layer-sized slices. Handwritten/embedded gfx1201 kernels are allowed; tinygrad may be used only as external source-grounding during development, never as a production import/call or runtime dependency.

**Acceptance:** each primitive has a hardware run log, fixture comparison within documented tolerance, focused tests, and no CPU fallback hidden inside the `r9700_native` primitive route.

## Wave 3 — Model-layer and full prefill assembly

### Task C1R-6: Layer-0 hardware forward pass

**Target:** new R9700 producer implementation module(s), `native_r9700/prefill.py` routing, fixtures/tests.

**Change:**

1. Compose hardware primitives into a complete layer-0 prefill pass for the `S-1` prefix.
2. Compare emitted layer-0 K/V and post-layer hidden state against CPU reference fixtures.
3. Log every hardware kernel stage and transfer boundary.

**Acceptance:** layer-0 R9700 output matches CPU reference tolerance; log proves all model-forward tensor operations in the layer used the hardware route.

### Task C1R-7: Full 16-layer R9700 prefill producer

**Target:** R9700 producer implementation, `native_r9700/prefill.py`, `native_r9700/kv_cache.py`, `native_r9700/parity.py`.

**Change:**

1. Extend the layer assembly to all 16 Llama layers.
2. Emit the same NPZ K/V schema consumed by `native_r9700.kv_cache`.
3. Add CLI selection for `--producer-kind r9700_native` or an equivalent explicit option.
4. Reject `r9700_native` if hardware prerequisites or exact geometry are unavailable.
5. Preserve `cpu_reference` as the default only for reference tests if changing defaults would break existing harnesses; parity/serving acceptance commands must explicitly request `r9700_native`.

**Acceptance:** focused full-prefill tests pass on a fixture prompt; logs prove hardware model-forward execution; generated prompt-cache loads in mlx-lm.

### Task C1R-8: Native parity gate and C1R review

**Target:** `native_r9700/parity.py`, `docs/path-a-validation-results.md`, `.superpowers/swarm/reports/`.

**Change:**

1. Run the C1R parity command against all Phase 0 prompts with `producer_kind=r9700_native`.
2. Record per-prompt tokens, exactness, mismatches, cache paths, R9700 hardware logs, timing, and deltas.
3. Request focused review of geometry, RoPE/position semantics, K/V layout, transfer boundaries, kernel execution evidence, and failure behavior.
4. Fix confirmed findings.

**Acceptance:** C1R is Done only when parity reports `P == R` for every prompt and review approves without Critical/Important findings.

## Wave 4 — C2R serving integration

### Task C2R-1: Serving wrapper producer-kind enforcement

**Target:** `native_r9700/serving.py`, tests.

**Change:**

1. Route large prompts through `r9700_native` only when C1R acceptance artifacts are available.
2. Log requested producer kind, actual producer kind, hardware log path, cache path, and fallback reason.
3. Fail closed if a command requested native serving but receives a `cpu_reference` producer artifact.
4. Preserve below-threshold and pre-acceptance fallback behavior.

**Acceptance:** focused serving tests prove CPU reference cannot masquerade as native C2 and existing security/redaction behavior remains intact.

### Task C2R-2: R9700 serving run and report

**Target:** `native_r9700/serving.py`, `docs/path-a-validation-results.md`, logs.

**Change:**

1. Run serving integration with large prompts through the accepted R9700 producer.
2. Run below-threshold and producer-unavailable fallback smokes.
3. Record timings: producer kernel time, transfer time, cache emit/import validation, final-token decode, total latency.
4. Update Path C2 report as native accepted only if the producer route is `r9700_native` and exactness passes.

**Acceptance:** C2R serving status is pass with R9700/eGPU producer route for large prompts and fallback/security semantics intact.

### Task C2R-3: Security and final review

**Target:** changed C1R/C2R files and docs.

**Change:**

1. Request focused security/transport review.
2. Request final code/design review.
3. Fix confirmed findings and rerun affected exact commands.
4. Update ledgers, validation command list, reports, and handoff.

**Acceptance:** no open Critical/Important findings; final verification commands pass; C3 remains blocked or reopened based only on real C2R evidence.

## Qwen3.8-27B target-expansion ladder

Qwen is not part of C1R/C2R acceptance. Do not fake it through the Llama ABI. If Qwen remains desired after Llama C2R:

1. Create a separate Qwen target phase.
2. Decide model family: MLX `mlx-vlm` Qwen3VL 4-bit snapshot vs a GGUF text-only Qwen replacement.
3. Discover and document cache ABI: layer count, hybrid linear/full attention cache state, mRoPE sections, partial RoPE dimensions, VLM processor effects, and quantized weight/dequant path.
4. Build Qwen-specific fixtures and baseline tokens.
5. Only then implement Qwen R9700 producer kernels and parity gates.

## Validation command policy

Do not record fake exact commands before implementation supplies them. Until C1R task sets discover concrete commands, `validation-commands.md` should state the required fields and owning task set.

Known final gate shape after implementation:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

Additional final hardware gates must be added by the implementing task sets and must include:

- reusable runtime hardware proof command;
- C1R `r9700_native` parity command;
- C2R `r9700_native` serving integration command;
- `git diff --check` over touched docs/tests/source.

## Stop conditions

Stop and record a blocker only when the next failure is outside repo/actionable control, for example:

- TinyGPU.app/AMDev hardware access regresses from C0A25 pass and cannot be restored locally;
- R9700 kernel dispatch works for C0A25 but a required model primitive has an unexplained hardware mismatch after CPU fixture and ISA review;
- memory residency/streaming cannot fit required layer shapes and needs a new architecture decision;
- token-exact parity fails after K/V deltas are understood and a product decision is needed to change acceptance.

Do not stop because the CPU reference path passes. That is the oracle, not the target.
