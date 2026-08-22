<handoff-context>
## Goal

Continue the C1 native producer parity work after the completed Wave 2 (C1 task sets 3 and 5). Resuming from a fresh session: verify the current tree state, then dispatch the next sequential kernel-ladder wave. The natural next step is **C1 task set 6 (Attention/RoPE/KV writer path)**, which is now unblocked (its dependencies C1-1/2/3/4/5 are all Done). Follow the `execute-subagent-swarm` skill: establish the shared work boundary, dispatch executor lanes, supervise, verify, review, checkpoint-commit (never push), update the durable ledger, and re-present options.

## Constraints & Preferences

- Work boundary: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`, branch `feature/native-r9700-producer` (current feature branch; shared swarm boundary; NOT `main`; no fallback worktree). Confirm the branch before any edit/dispatch.
- Supervisor (main controller) owns verification, ledger, review gates, and checkpoint commits. Subagents (task/reviewer/fix) NEVER run git, tests, builds, linters, package managers, or hardware commands.
- Push to `origin https://github.com/<account>/mlx-egpu-experiments` remains the USER's responsibility — do not push.
- Do NOT touch the frozen C0 probe `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` — it is the byte-stable reference. Refactor = create NEW files under `native_r9700/`, never edit the probe.
- Do NOT modify `docs/adr/*`, `docs/ROADMAP.md` frozen contract text, or `phase-c1-native-producer-parity.md` frozen contract content.
- C1 contract is FROZEN ("the frozen contract"): macOS TinyGPU.app/APLRemotePCIDevice/PCIIface native AMDev substrate; source root `native_r9700/`, test root `tests/native_r9700/`; 24-byte kernarg layout `{output_va@0, input_va@8, scalar_va@16, scalar:u32@24}`; KV interchange = mlx-lm prompt-cache `.safetensors` (per-layer `KVCache`, empty `meta_state`, global `offset`, fp16, `(1,8,N,64)`); `P == R` token-exact gate over Phase 0 suite; no tinygrad dependency in producer path; RoPE from MLX config sidecar; `S-1` + final-token injection.
- Python: `${HOME}/.pyenv/versions/3.12.8/bin/python3`. C++ build: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra`.
- Every GPU/native run writes a reviewable local log under `logs/` (git-ignored; do not commit).
- Model weights dir: `.worktrees/tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct` (relative to the worktree; `mlx_models/` is gitignored).

## Progress

### Done (all committed on `feature/native-r9700-producer`)
- [x] **C0 complete**: C0A25 `--kernel-proof` PASS (`45d7b95`, `65414e0`, `33a7036`); ADR 0004 macOS substrate (`dd90595`).
- [x] **C1 contract freeze**: `34a09ce` (docs-only).
- [x] **Wave 1 (commit `861c794`)**: C1 task set 2 (weight/config loader, `C1WeightLoader`, 19 tests) + C1 task set 4 (native runner runtime shell, `C1RunnerScaffold`→fix agents→`C1RunnerRereview` APPROVE). Runtime shell ports C0 SDMA copy+fence and 59-dword/12-packet PM4 dispatch byte-faithfully; PM4=59 verified.
- [x] **Wave 2 (commit `268671f`)**: C1 task set 3 (CPU/MLX reference fixtures, `C1RefFixtures`) + C1 task set 5 (native tensor primitives, `C1Primitives`), parallel lanes. Combined `tests/native_r9700` = 57 passed; full `tests` = 97 passed. Wave 2 reviewer `C1Wave2Review` -> APPROVE both lanes (0 Critical/Important; Minor note recorded).

### Current C1 ledger state (from `.superpowers/swarm/progress.md`)
- C1-1 contract freeze → **Done**
- C1-2 weight/config loader → **Done**
- C1-3 CPU/MLX reference fixtures → **Done** (Wave 2 Lane B2)
- C1-4 runtime wrapper/logging shell → **Done**
- C1-5 native tensor primitives → **Done** (Wave 2 Lane A2)
- C1-6 attention/RoPE/KV writer → **Blocked** (stale blocker text "missing C0-selected substrate"; TRUE deps C1-1/2/3/4 now Done — unblocked, ready to dispatch)
- C1-7 full layer stack prefill → **Blocked** (deps C1-5, C1-6)
- C1-8 KV interchange emitter → **Blocked** (dep C1-7)
- C1-9 parity harness/report writer → **Blocked** (dep C1-8)
- C1-10 C1 review/handoff → **Blocked** (dep C1-9)
- C2-* / C3-* → **Blocked** (depend on C1 parity gate)

## Wave 1 & 2 Artifacts (committed, do not regress)

### C1 task set 2 — weight loader (`native_r9700/`)
- `__init__.py`, `config.py`, `loader.py`; `tests/native_r9700/test_loader.py` (19 tests).
- Container decision = MLX safetensors dir (fp16 weights + `config.json` sidecar with Llama-3 `rope_scaling`; GGUF lacks `rope_scaling`).
- `Llama32Config`/`ModelData` dataclasses; geometry 16/8/64/2048, rope_theta 500000, rope_scaling llama3, weight_dtype F16.
- Lane B reviewer `C1LoaderReviewer` 3 Minor notes recorded in ledger (shard/index machinery generality; `max_position_embeddings` not validated; `format_report` hardcoded geometry).

### C1 task set 4 — runtime shell (`native_r9700/`)
- `runtime.h`, `runtime.cpp`, `runner.cpp`; `tests/native_r9700/test_runtime_contract.py` (8 tests).
- `native_r9700::RuntimeSession` with `initialize/allocate_buffers/copy_input/load_kernel/write_kernargs/dispatch_and_poll/readback_and_compare/cleanup/dry_run`.
- `--lifecycle-dry-run` = hardware-free contract mode (exit 0; PM4=59; SDMA header 0x000001; PM4 first dword 0xc0065800); `--kernel-proof` = hardware-gated (exits 2 without HW).
- Hardware stages are honest state-machine advances; real TinyGPU socket/BAR/SDMA/compute wiring deferred to task sets 5-8 (NOT implemented).

### C1 task set 5 — primitives (`native_r9700/primitives.py`, `tests/native_r9700/test_primitives.py`)
- `cast_fp32_to_fp16`/`cast_fp16_to_fp32`, `matmul` (fp16×fp16→fp16, fp32 accumulator, single rounding), `rms_norm` (eps=1e-5, fp32 internal, per-row), `silu` (fp32 internal) — loud `UnsupportedDtypeError`/`UnsupportedShapeError`.
- `TestPrimitiveFixtureSeam` (4 tests) consumes Lane B2 `primitives_fixtures.npz` (cast/matmul bit-exact, rms/silu ≤1-fp16-ulp; pytest.skip when fixtures absent).
- Substrate decision: CPU/numpy-host fp32-accumulate reference (the C++ RuntimeSession performs no tensor math — correct not to use it for compute).

### C1 task set 3 — reference fixtures (`native_r9700/ref_fixtures.py`, `tests/native_r9700/test_ref_fixtures.py`, `tests/native_r9700/fixtures/`)
- Deterministic, small, committed fixtures: `prompts.json` (S=6/222/661), `baseline_r_tokens.json` (mlx-lm greedy R), `kv_state.npz` (per-layer K/V (1,8,5,64) fp16, 16 layers, S-1 + final_token_id=374), `primitives_fixtures.npz` (11-key seam schema), `fixtures_schema.json` (sha256 digests).
- Regenerate: `python3 -m native_r9700.ref_fixtures --generate --model <mlx-model-dir> --fixtures-dir tests/native_r9700/fixtures`. Verified byte-for-byte deterministic by supervisor.
- Wave 2 Minor note (recorded, owner C1RefFixtures, evidence `c1k-wave2-review.md`): `rms_eps` stored fp32 vs fp64 ground-truth eps (`ref_fixtures.py:146-156`) — harmless (bit-exact), left as-is.

## Reports (all under `.superpowers/swarm/reports/`, committed)
- `c1k-task-2-weight-loader.md`, `c1k-task-2-review.md`
- `c1k-task-4-runner-scaffold.md`, `c1k-task-4-runner-fix.md`, `c1k-task-4-runner-review-fix.md`, `c1k-task-4-runner-rereview.md` (NOTE: `c1k-task-4-review.md` does NOT exist — Lane A reviewer's CHANGES_REQUIRED were only in `agent://C1RunnerReviewer`)
- `c1k-task-5-primitives.md`, `c1k-task-3-reference-fixtures.md`, `c1k-wave2-review.md`

## Confirmed supervisor verification (as of end of Wave 2)
- `pytest tests/native_r9700 -v` → **57 passed** (19 loader + 8 runtime + 19 primitives-focus + 4 seam + 7 fixtures).
- `pytest tests/test_native_amdev_transfer_contract.py -q` → **23 passed**.
- `pytest tests -v` → **97 passed, 2 warnings** (pre-existing swig DeprecationWarnings, unrelated).
- Build: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp` → exit 0, **warning-free**.
- Dry-run: `./build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run` → exit 0.
- Fixture regeneration byte-for-byte deterministic (supervisor verified with sha256).
- Probe untouched (`git diff --stat experiments/...probe.cpp` empty); `git diff --check` clean.

## Working tree (current, safe to leave)
- Pre-existing untracked (leave alone): `docs.zip`, `docs/research/`, `docs/tasks/amdev-doorbell-delivery/phase-9-cp-mec-rs64-handoff.md`.
- Gitlog HEAD: `268671f` (Wave 2) → `861c794` (Wave 1) → `34a09ce` (contract freeze) → `dd90595` (ADR 0004) → `33a7036` → `65414e0` → `45d7b95` → `120ef29`.

## Next Steps (recommended next wave = C1 task set 6, Attention/RoPE/KV writer)

1. **Confirm boundary**: `git branch --show-current` == `feature/native-r9700-producer`; confirm tree state matches this handoff (no uncommitted changes beyond the 3 pre-existing untracked files).
2. **Read task set 6 spec** in `docs/tasks/native-r9700-producer/phase-c1-native-producer-parity.md` (§Task set 6, lines ~225-273). It requires: attention input projection / head shaping for single-layer K/V; Llama-3 RoPE scaling exactly per the MLX sidecar config; emit K/V in temporal order shape `(1,8,N,64)`; compare vs MLX reference fixtures (from C1-3!); record per-layer max/mean deltas; `N == S-1` explicit.
3. **Dispatch Wave 3**: C1-6 is the next sequential lane (NOT parallel with C1-7, which depends on C1-6). Treat the "stale blocker" on the C1-6 ledger row as cleared when dispatching — update the row from `Blocked` to `In progress` first alarmits blocker text is stale ("missing C0-selected substrate" — false; substrate is selected and all deps Done).
   - Lane must consume the existing primitives (`native_r9700/primitives.py`) and reference fixtures (`native_r9700/ref_fixtures.py`, `tests/native_r9700/fixtures/kv_state.npz`) — reuses not reinvents.
   - RoPE scaling must come from the MLX config sidecar (the loader already validates rope_theta 500000 + llama3 rope_scaling).
   - File-overlap contract: new files only (e.g. `native_r9700/attention*.py`, `tests/native_r9700/test_attention*.py`); do NOT edit committed loader/runtime/primitives/ref_fixtures or fixtures.
4. **Verify + review + checkpoint-commit + ledger update** exactly as Wave 1/2. Add a C1 task-set-6 command row to `docs/tasks/native-r9700-producer/validation-commands.md`.
5. **Re-present options** to the user (A continue / B checkpoint / C stop) after the wave, per the established pattern.

## Key Decisions (preserve)
- C0 substrate = macOS TinyGPU/AMDev native for C1 (ADR 0004); Linux ROCm/HIP = reference/deferred.
- Weight container = MLX safetensors dir (not F16 GGUF) because it carries both fp16 weights and `config.json` with Llama-3 `rope_scaling`.
- Runtime shell packet encodings must remain byte-faithful to the proven C0 probe (already done; do not regress).
- Primitive matmul substrate = CPU/numpy-host fp32-accumulate reference (C++ shell performs no math).
- Reference fixtures are the oracle for primitive + KV comparison; regenerate deterministically.
- Quality bar: simplest adequate design; no speculative generality; reuse C1 vocabulary/contracts; probe stays untouched; no tinygrad in producer path.
- Push remains the user's responsibility; local checkpoint commits only after reviewed/verified waves.
</handoff-context>
