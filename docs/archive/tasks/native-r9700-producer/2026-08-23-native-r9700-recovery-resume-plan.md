# Native R9700 Producer — Recovery & Resume Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a healthy TinyGPU/R9700 control path after the power cycle, re-establish the frozen C0 + resident-VRAM health gate, then repair layer-0 RMSNorm arithmetic and drive the native producer through token-exact C1R/C2R parity.

**Architecture:** Phase 0 recovers stock AMDev discovery (current blocker). Phase 1 re-proves the native C0 queue control. Phase 2 re-baselines the bounded numerical traces. Phase 3 repairs RMSNorm with a source-grounded hypothesis and a hardware-free failing contract first. Phases 4–7 run the numerical recurrence and acceptance gates that were blocked on RMSNorm.

**Tech Stack:** macOS TinyGPU.app / `APLRemotePCIDevice` / `PCIIface`; pinned Python `${PY}`; `xcrun --sdk macosx clang++ -std=c++17`; pytest (hardware-free) for contracts.

## Global Constraints

- Interpreter: `${PY}` (`$PY`). Never `python3` from `PATH`.
- Worktree: `<former-native-r9700-worktree>`; branch `feature/native-r9700-producer`.
- Native product math stays tinygrad-free. Tinygrad allowed only for device reset/bootstrap/control.
- CPU/NumPy is oracle-only; never feed CPU-generated model/KV values into an accepted native artifact.
- Preserve S-1 prompt-cache semantics: cache holds prefix tokens; mlx-lm receives the final prompt token.
- Do not change `native_r9700/kv_cache.py` for this work.
- Stop at the first non-finite or out-of-tolerance stage; do not run later stages while an earlier one fails.
- Hardware dispatch order: stock discovery → frozen C0 add-one control → resident VRAM smoke → bounded traces. Do not send a model or experimental HSA doorbell while C0 is unhealthy.
- Do not force-kill active native runners; queued interrupts have previously left HQD/MEC state unhealthy.
- `producer_kind=r9700_native` must fail closed until it emits a validated hardware-backed cache.
- Commits are supervisor-owned after reviewed/verified waves. Executors do not commit or push.
- Pre-existing uncommitted state (417 paths, mostly `.superpowers/swarm/` → `docs/archive/` moves) is prior work; leave it untouched unless a task explicitly touches those files.

## Root Cause of Current Blocker (read-only evidence)

AMDev discovery fails at `amdev.py:299` (`discovery signatures mismatch`) — the very first VRAM read of the IP discovery table at `vram_size - 64 KiB`.

Observed in a read-only probe:

- `mmRCC_CONFIG_MEMSIZE` (0xde3) reads `0x7f70` → `vram_size = 0x7f7000000` (≈ 32 GiB). Correct.
- BAR0 = 256 MiB, so `large_bar = False` → discovery uses the indirect `_read_vram` path via `regBIF_BX_PF0_RSMU_INDEX/DATA`.
- The discovery table's first 8 words read back as `[0x00000000, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0xffffffff, 0xffffffff, 0xffffffff]` — erased/uninitialized, not a valid `binary_signature`.

Interpretation: PCI config, BAR mapping, and MMIO register reads all work; the PSP has not (re)written the IP discovery table into VRAM after the power cycle. The handoff's documented working recovery sequence was `reset()` **then** AMDev boot; the prior step skipped `reset()`. Phase 0 tests this hypothesis minimally.

---

### Task 0: Restore stock discovery

**Files:**
- None (read-only bridge/AMDev commands).

**Interfaces:**
- Produces: repeatable `tinygpu_reset: pass` + `tinygpu_amdev_full_boot: pass`, and a populated discovery table (word 0 nonzero `binary_signature`).

- [ ] **Step 0.1: Run the non-destructive bridge reset**

```sh
PYTHONPATH=<tinygrad-checkout> \
  $PY -c "from tinygrad.runtime.support.system import APLRemotePCIDevice; APLRemotePCIDevice('AMD','usb4').reset(); print('tinygpu_reset: pass')"
```

Expected: `tinygpu_reset: pass`, exit 0.

- [ ] **Step 0.2: Re-run discovery-only AMDev full boot**

```sh
PYTHONPATH=<tinygrad-checkout> \
  $PY -c "from tinygrad.runtime.support.am.amdev import AMDev; from tinygrad.runtime.support.system import APLRemotePCIDevice; pci=APLRemotePCIDevice('AMD','usb4'); am=AMDev(pci); am.fini(); pci.sock.close(); print('tinygpu_amdev_full_boot: pass')"
```

Expected: `tinygpu_amdev_full_boot: pass`, exit 0. No `AssertionError: discovery signatures mismatch`.

- [ ] **Step 0.3: Confirm discovery table is populated**

Re-run the `/tmp/vram_diag.py` probe (or equivalent read-only word dump at `vram_size - 64 KiB`). Expected: word 0 equals a nonzero `binary_signature` (no `0xffffffff` fill).

- [ ] **Step 0.4: Gate check**

If Step 0.1–0.2 pass and the table is populated, proceed to Task 1. If `reset()` alone does not populate the table, do not dispatch further native work; record the exact table bytes and return to root-cause investigation (the failure is below our native runtime, in bridge/PSP boot, not in the native queue path).

---

### Task 1: Restore frozen C0 health gate

**Files:**
- Modify (only if the gate fails): `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`, `native_r9700/amdev_session.cpp` (MQD construction/readback audit).

**Interfaces:**
- Consumes: `tinygpu_amdev_full_boot: pass` (Task 0).
- Produces: frozen C0 `--kernel-proof` exit 0 with `cpu_comparison_status: pass`, `failure_stage: none`; resident VRAM smoke exit 0.

- [ ] **Step 1.1: Run the frozen C0 add-one control**

```sh
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --kernel-proof
```

Expected: exit 0, `cpu_comparison_status: pass`, `failure_stage: none`. No `compute_ring_setup` MQD readback mismatch.

- [ ] **Step 1.2: Run resident VRAM smoke**

Use the current runner's resident-VRAM smoke command (see `docs/tasks/native-r9700-producer/validation-commands.md` for the exact invocation). Expected: exit 0, `failure_stage: none`.

- [ ] **Step 1.3: If the C0 gate fails, audit MQD construction before another doorbell**

Read `native_r9700/amdev_session.cpp` MQD page write, VRAM zeroing, and readback sequence against the last known-good frozen C0 commit (`0292878`). Focus on the mismatch at physical `0x02003000`: expected `0xc0310800`, observed `0x00300000`. Identify the single source-grounded discrepancy; do not send another doorbell until it is fixed.

- [ ] **Step 1.4: Gate check**

Proceed to Task 2 only after `--kernel-proof` and the resident VRAM smoke both finish normally (exit 0). Let both complete; do not cancel them.

---

### Task 2: Re-baseline bounded numerical traces

**Files:**
- None (invoke existing `--llama-stage-trace` CLI).

**Interfaces:**
- Consumes: healthy C0 (Task 1).
- Produces: `hidden` trace SHA-256 exact match to `4d2c5ceaca8ace6263af0d595b6d47040dc4a91b6abf1b72edbe89418129b808`; `normalized` trace nonfinite diagnostic metadata recorded.

- [ ] **Step 2.1: Re-run layer-0/token-0 `hidden` trace**

```sh
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --llama-stage-trace \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-id 128000 --layer 0 --position 0 --stage hidden \
  --trace-dir logs/ln-resume-native
```

Expected: SHA-256 of `layer0-token0-hidden.bin` == `4d2c5ceaca8ace6263af0d595b6d47040dc4a91b6abf1b72edbe89418129b808`.

- [ ] **Step 2.2: Re-run layer-0/token-0 `normalized` trace**

Same command with `--stage normalized`. Expected: `failure_stage: trace_nonfinite` (reproduced), finite diagnostic metadata recorded.

- [ ] **Step 2.3: Gate check**

`hidden` must remain exact. `normalized` is the known first failure; record its metadata and proceed to Task 3.

---

### Task 3: Repair layer-0 RMSNorm arithmetic

**Files:**
- Modify: `native_r9700/kernels/llama_rmsnorm_f16.cpp` (original arithmetic asset).
- Test: `tests/native_r9700/test_rmsnorm_arithmetic_contract.py` (new, hardware-free).
- Reference (read-only): `native_r9700/kernels/llama_rmsnorm_zero_store_f16.cpp` (proven store-path control).

**Interfaces:**
- Consumes: `normalized` nonfinite diagnosis (Task 2) and the arithmetic-decode report `.superpowers/swarm/reports/ln-2-rmsnorm-arithmetic-decode.md`.
- Produces: a source-grounded hypothesis, a failing hardware-free contract, one minimal kernel change, and a finite `normalized` trace matching the oracle.

- [ ] **Step 3.1: Select a source-grounded hypothesis**

Re-read `.superpowers/swarm/reports/ln-2-rmsnorm-arithmetic-decode.md` (inverse-root lowering at `.text 0x1838–0x1944`: `V_S_SQRT_F32` / `V_DIV_SCALE_F32` / `V_RCP_F32` / refinement FMAs / `V_DIV_FIXUP_F32`). Diff the arithmetic asset against the proven zero-store asset to isolate the exact arithmetic block. State the hypothesis in one sentence with a source offset.

- [ ] **Step 3.2: Write a failing hardware-free contract**

Add a test asserting the fixed-point behavior of the suspected arithmetic path on a finite fp16 input (e.g. a known nonzero vector with a known RMSNorm result), so the bug fails without hardware.

```python
def test_rmsnorm_finite_output_for_known_input():
    inp = [2.0, 2.0, 2.0, 2.0]  # rms = 2.0 -> normalized all 1.0
    out = rmsnorm_f16(inp, eps=1e-5)
    assert all(v == 1.0 for v in out)
```

- [ ] **Step 3.3: Run the test to verify it fails**

```sh
$PY -m pytest tests/native_r9700/test_rmsnorm_arithmetic_contract.py -v
```

Expected: FAIL for the current arithmetic.

- [ ] **Step 3.4: Make one minimal kernel change**

Apply the single change matching the hypothesis (e.g. correct the reciprocal/inverse-sqrt lowering or its fp16 rounding). No bundled refactoring.

- [ ] **Step 3.5: Rebuild and re-run the bounded `normalized` trace**

Rebuild the runner, then re-run the `--stage normalized` trace from Step 2.2. Expected: finite output, no `trace_nonfinite`.

- [ ] **Step 3.6: Compare `normalized` to oracle numerically**

Run the `native_r9700/llama_stage_oracle.py` `normalized` oracle and compare against the finite native trace. Expected: exact fp16 match (same SHA-256 policy as `hidden`).

- [ ] **Step 3.7: Gate check**

Proceed to Task 4 only after `normalized` is finite and oracle-exact.

---

### Task 4: Layer-0 recurrence across all nine boundaries

**Files:**
- None (existing `--llama-stage-trace` CLI).

**Interfaces:**
- Consumes: finite, oracle-exact `normalized` (Task 3).
- Produces: all nine boundaries (`hidden` → `post_attention_hidden`) finite and oracle-exact for layer 0, at prefix lengths 2, 6, 16, 64, 128.

- [ ] **Step 4.1: Walk the remaining boundaries one at a time**

After `normalized`, in order: `fresh_k`, `fresh_v`, `k_cache`, `v_cache`, `attention_scores`, `attention_probabilities`, `context`, `post_attention_hidden`. For each, run the bounded trace and compare to the oracle. Stop at the first non-finite or mismatched stage; do not skip ahead.

- [ ] **Step 4.2: Run layer-0 recurrence at prefix lengths 2, 6, 16, 64, 128**

For each length, run the full layer-0 stage walk. Expected: finite and oracle-exact at every length.

- [ ] **Step 4.3: Gate check**

Proceed to Task 5 only after layer-0 passes at all five lengths.

---

### Task 5: All-layer recurrence and full native artifact

**Files:**
- None (existing native prefill proof path).

**Interfaces:**
- Consumes: layer-0 recurrence pass (Task 4).
- Produces: full 16-layer finite K/V and a schema-valid `r9700_native` NPZ via the existing `--native-prefill-proof` path.

- [ ] **Step 5.1: Repeat stage localization/repair through layers 1–15**

Apply the Task 3/4 method layer by layer. Stop at the first failing layer/stage.

- [ ] **Step 5.2: Emit a full native artifact**

```sh
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --native-prefill-proof \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-ids-json '[128000,128001]' \
  --out logs/full-native-prefill.npz \
  --log logs/full-native-prefill.log
```

Expected: `native_prefill_acceptance: pass`, finite K/V across all layers.

- [ ] **Step 5.3: Validate NPZ schema**

```sh
$PY -c "from native_r9700.native_worker import validate_native_prefill_npz; problems = validate_native_prefill_npz('logs/full-native-prefill.npz', 2, '../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct'); print(problems); raise SystemExit(bool(problems))"
```

Expected: `[]`.

- [ ] **Step 5.4: Gate check**

Proceed to Task 6 only after a schema-valid, finite native artifact exists.

---

### Task 6: Token-exact C1R parity and C2R serving

**Files:**
- None (existing `native_r9700/parity.py` and `native_r9700/serving.py`).

**Interfaces:**
- Consumes: valid finite native NPZ (Task 5) and unchanged `native_r9700/kv_cache.py`.
- Produces: C1R `gate_result=pass` (P == R token-for-token), then C2R imported-cache serving pass.

- [ ] **Step 6.1: Convert the native NPZ with the unchanged emitter**

```sh
$PY -m native_r9700.kv_cache \
  --prefill-npz logs/full-native-prefill.npz \
  --out logs/full-native-prompt-cache.safetensors \
  --log logs/full-native-kv-cache.log
```

Expected: `wrote prompt cache ... (n_prefix=2, num_layers=16)`.

- [ ] **Step 6.2: Run C1R native parity (prompt-0)**

```sh
NATIVE_R9700_PREFILL_RUNNER=build/native-r9700-runtime/native_r9700_runner \
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  $PY -m native_r9700.parity \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --r-source both --producer-kind r9700_native \
  --prompt-name prompt-0 --max-new-tokens 4 \
  --artifacts-dir logs/c1r-native-parity \
  --json logs/c1r-native-parity/result.json \
  --log logs/c1r-native-parity/run.log \
  --report docs/path-a-validation-results.md
```

Expected: `gate_result=pass`, P tokens == R tokens (`[12366, 13, 578, 469]`).

- [ ] **Step 6.3: Run imported-cache C2R serving**

Use the existing `native_r9700/serving.py` C2R command from `validation-commands.md`. Expected: cache accepted, decode produces the reference continuation (no recompute/repair of the offloaded prefix).

- [ ] **Step 6.4: Gate check**

Llama C1R and C2R both pass → Llama native acceptance is met.

---

### Task 7: Qwen expansion (gated)

**Files:**
- TBD after Llama C1R/C2R passes.

**Interfaces:**
- Consumes: Llama C1R + C2R pass (Task 6).

- [ ] **Step 7.1: Only after Llama passes, start Qwen3.8-27B native producer work**

Do not begin before Task 6.4. Qwen uses a different MLX-VLM/quantized/hybrid-cache ABI; it is a separate target-expansion slice.

---

## Self-Review Notes

- Spec coverage: every stage in the handoff's "Pending" list maps to a task (C0 health → Task 1; stage localization/recurrence → Tasks 2/4; all-layer → Task 5; native artifact → Task 5; C1R/C2R → Task 6; Qwen gated → Task 7).
- The RMSNorm task (Task 3) is intentionally hypothesis-first: the exact code fix is not known until Step 3.1's source-grounded diff selects it. This is correct under the systematic-debugging rule (no fix before root cause), not a placeholder.
- Type/command consistency: `$PY`, socket path, model path, and runner path are copied verbatim from `validation-commands.md` / the handoff.
