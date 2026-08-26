# Native R9700 Producer — Validation Commands

This is the active shared command ledger for current F1–F6, P1–P5, and Q1 task packets. The complete committed C0–C3 command history is preserved verbatim in [`validation-commands-c0-c3.md`](../../archive/tasks/native-r9700-producer/validation-commands-c0-c3.md); new task packets must add concrete commands here before execution.

## Fixed environment

Use this Python for Python-side validation in this repo:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3
```

Do not rely on `python3` from `PATH`.

For AMD eGPU/tinygrad comparison runs that intentionally use tinygrad:

```sh
DEV=AMD
JITBEAM=2
HF_HOME=${HOME}/Development/ml/models
```

Native R9700 producer commands must not import or call tinygrad unless explicitly running a labeled comparison/control command outside the producer path.

## Exact commands known now

### Existing Python regression suite

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v
```

### Current native runner build and no-model smokes

The translation-unit list is maintained by `RUNNER_SOURCES` in `tests/native_r9700/test_block_prefill_runtime_contract.py`:

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
  native_r9700/native_resource_worker.cpp \
  native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run
build/native-r9700-runtime/native_r9700_runner --kernel-proof
build/native-r9700-runtime/native_r9700_runner --transfer-proof --bytes 20480
build/native-r9700-runtime/native_r9700_runner --vram-smoke
```

### Existing harness syntax check

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m py_compile tinygrad_kv_worker/harness.py
```

### Existing Phase 0 GPU parity command

This is a regression/control command for the validated tinygrad producer path, not a Native R9700 producer command:

```sh
DEV=AMD JITBEAM=2 HF_HOME=${HOME}/Development/ml/models \
  ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m tinygrad_kv_worker.harness \
  --gguf mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf \
  --mlx mlx_models/meta-Llama-3.2-1B-Instruct \
  --out docs/path-a-validation-results.md \
  --run-tag meta-f16-final
```

Historical Phase 0 acceptance evidence lives in `docs/path-a-validation-results.md` and `docs/archive/tasks/tinygrad-kv-worker/phase-0-parity.md`.

### Documentation whitespace check

Use this after task-doc or design-doc edits:

```sh
git diff --check
```

## Command discovery policy

Each new task packet must record its exact focused test, broader regression, native build, hardware smoke, log path, and expected observable result here before the command is used as promotion evidence. Cite the current roadmap phase and task document; do not attach new commands to archived C0–C3 packets.

## Log requirements for all GPU/native runs

Every GPU/native run must write a reviewable local log under `logs/` or record an explicit remote log artifact path. Logs must include:

- command line;
- timestamp;
- runtime substrate and device identity if discoverable;
- model/config path or note that no model is used;
- prompt length or input shape;
- output comparison result or digest;
- exit status;
- failure traceback/error text when failing.

Logs and model files must not be committed.

## Gate reminders

- Producer-swap acceptance is token-exact `P == R`, not semantic equivalence.
- mlx-lm injected decode uses an imported `S-1` prefix cache plus the final prompt token.
- Llama-3 RoPE scaling must match the MLX sidecar config.
- Native R9700 producer code must not depend on tinygrad.

## F1

### F1 persistent process smoke

```sh
mkdir -p logs/f1-persistent-worker/process-smoke
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.native_worker \
  --smoke-load-unload-reload \
  --model mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --prompt-name prompt-128 \
  --samples 10 \
  --producer-kind r9700_native \
  --native-runner build/native-r9700-runtime/native_r9700_runner \
  --artifacts-dir logs/f1-persistent-worker/process-smoke/artifacts \
  --json logs/f1-persistent-worker/process-smoke/result.json \
  --log logs/f1-persistent-worker/process-smoke/run.log \
  --trace logs/f1-persistent-worker/process-smoke/trace.json
```

Expected evidence: one public Python service process launches exactly one `native_r9700_runner --model-service-worker` child at service startup and keeps that child PID alive through final shutdown; the public stdin/stdout and private child pipes are distinct. The child performs `Prepare → Commit` for each explicit load generation and ten `Prefill` operations in the resident generation, then `Release → released` before the public `draining → unloaded` transition; there is no per-request launch, `subprocess.run`, socket, or one-shot `--native-prefill-proof` call. The child log/result records `r9700_native_resource_v1`, one-in-flight correlation, the generation, `runner_binary_sha256`, ordered pack digests, and the computed `producer_fingerprint`. The smoke also performs an initial `LoadModel → validating → preparing → resident-ready`, ten independent `prompt-128` requests with `S=129` and `N=128`, `UnloadModel → draining → unloaded`, a second explicit `LoadModel → validating → preparing → resident-ready`, and final unload. The result must show `load_preparation_count=2` (the initial load plus the explicit reload), `raw_warm_sample_count=10`, `warm_prefill_weight_reload_count=0` across those ten warm Prefills, no resource drift, no fallback after acceptance, and no stale request/model association. Every accepted request must have `producer_kind=r9700_native`, `native_prefill_acceptance=pass`, `native_prefill_full_layer_loop_status=pass`, the exact runtime substrate, an existing request-bound hardware log, nonzero kernel/transfer evidence, strict S-1 cache metadata including the 16 empty `meta_state` values, an exact handle/evidence/cache `producer_fingerprint` match, empty `failure_stage`, and `exit_status=0`.

### F1 warm benchmark promotion

```sh
mkdir -p logs/f1-persistent-worker/warm
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.native_worker \
  --warm-prefill-samples \
  --model mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --prompt-name prompt-128 \
  --samples 10 \
  --producer-kind r9700_native \
  --native-runner build/native-r9700-runtime/native_r9700_runner \
  --artifacts-dir logs/f1-persistent-worker/warm/artifacts \
  --json logs/f1-persistent-worker/warm/serving.json \
  --log logs/f1-persistent-worker/warm/worker.log \
  --trace logs/f1-persistent-worker/warm/trace.json
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.benchmark \
  --model mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --artifacts-dir logs/f1-persistent-worker/warm/artifacts \
  --json logs/f1-persistent-worker/warm/benchmark.json \
  --report logs/f1-persistent-worker/warm/benchmark.md \
  --log logs/f1-persistent-worker/warm/benchmark.log \
  --producer-kind r9700_native \
  --serving-result logs/f1-persistent-worker/warm/serving.json
```

Expected evidence: one public Python service process and one persistent private `native_r9700_runner --model-service-worker` child provide all ten warm requests; the child is not relaunched between requests, its private pipes are not the public service pipes, and the warm path never invokes one-shot `--native-prefill-proof`, `subprocess.run`, TCP, or a fallback after acceptance. The worker JSON has exactly ten raw warm samples for the concrete `prompt-128` fixture, each with `S=129` and `N=128`, all accepted with `route=native_producer`, `accepted_cache=true`, empty `fallback_reason`, token-exact evidence, request-bound hardware logs, `warm_prefill_weight_reload_count=0`, and exact equality among committed-handle, private-Prefill evidence, and cache-metadata `producer_fingerprint`. The child process evidence records `r9700_native_resource_v1`, one generation, ten `Prefill` operations, the immutable runner binary SHA, ordered kernel-pack SHA values, and the JCS-derived fingerprint. The benchmark JSON/report/log has ten raw warm sample records plus exactly three aggregate scope records (`cold_process`, `warm_prefill`, and `gpu_compute`), for `raw_warm_sample_count=10`, `scope_aggregate_count=3`, and `total_record_count=13`; with the raw samples labeled `warm_prefill`, `records_by_scope` is `{"cold_process":1,"warm_prefill":11,"gpu_compute":1}`. No one-time load time is included in a warm-prefill timing. Every record retains the existing `native_r9700_benchmark_v1` required timing/transfer/correctness fields and `row_role=native_benchmark`; aggregate records carry their explicit scope and aggregate identity. The 2026-08-25 B4 observation (18.012 seconds / 7.11 prefix tok/s for prompt-128) is recorded as the first comparator, not an automatic threshold; promotion requires no warm reload, no resource drift, cache integrity, fail-closed behavior, exact-token evidence, and persistent-child/fingerprint evidence.

Shared F1 child-boundary requirements for both command observations: task set 3's existing `RUNNER_SOURCES`/validation source list must compile `native_resource_worker.cpp` into the one `native_r9700_runner` binary. Task set 2 starts that runner once with `--model-service-worker` via `subprocess.Popen` and private pipes; the child alone owns the native generation from service startup through shutdown. The private protocol is exactly `r9700_native_resource_v1` with `Prepare`, `Commit`, `Rollback`, `Release`, `Prefill`, `Health`, and `Shutdown`; its pipes are distinct from public stdio, it permits one in-flight request, and mismatched/duplicate IDs, child EOF, or child exit fail closed. The child `Health` result must report `child_state:"ready"`, the current `resource_generation`, `resource_state` in `none|prepared|resident-ready|release-failed`, the generation's `producer_fingerprint`, and nullable bounded `error_summary` (`null` outside failure; the retained cleanup failure summary in `release-failed`).

Rollback and Release pass only with `{resource_generation:uint64,state:"released",already_released:bool}`; the first pass is `already_released:false` and an idempotent same-operation repeat is `true`. A cleanup error returns `{domain,message,failure_stage}` in `error`, retains child ownership as `release-failed`, leaves Python `draining`, blocks new `LoadModel`, and reaches `unloaded` only after a pass. While `release-failed`, allowed operations are exactly read-only `Health` and the same cleanup operation/generation retry; all others, including `Shutdown`, reject. Child crash or device loss faults the service and disallows accepted-prefix repair or fallback until process restart. The runner computes `producer_fingerprint` as `sha256:` of SHA-256 over UTF-8 RFC 8785 JCS of the exact `r9700-producer-fingerprint-v1` preimage; its binary SHA is included and published during `Prepare`, Python binds it to the model handle, and every native Prefill evidence/cache metadata repeats it. The consumer requires exact equality across the handle, private Prefill evidence, and cache metadata and rejects unknown, missing, non-finite, or mismatched identity. These private-child PID, protocol/generation, runner-SHA, cleanup, health, and fingerprint observations are requirements for both command records, not claims that they have been run.

## F2

**F2 physical WMMA layout proof (task set 3-owned; required before task set 4):**

```sh
/bin/bash -o pipefail -c '
  set -u
  : "${ROCWMMA_CHECKOUT:?set ROCWMMA_CHECKOUT to the exact f7f2aee8e764e612f49f2dc030b7e1639fb30d34 checkout}"
  : "${AITER_CHECKOUT:?set AITER_CHECKOUT to the exact 35c652ed3bd34e5d5828954e1545babc9255a69a checkout}"
  mkdir -p build/f2-wmma logs/f2
  layout_spec=build/f2-wmma/f2-wmma-physical-layout-spec.json
  inverse_fixture=build/f2-wmma/f2-wmma-physical-layout-inverse.npz
  log=logs/f2/wmma-physical-layout-proof.log
  {
    printf "%s\n" "command: tools/f2-wmma-layout-proof --source-layout-version f16-row-major-nk-source-v1 --physical-layout-version f2-wmma-physical-tile-v1 --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/samples/simple_hgemm.cpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/rocwmma.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/io_config.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/io_layout.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/mapping_util.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/accessors_impl.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/layout/matrix_layout_traits_impl.hpp --rocwmma-symbols matrix_b,col_major,fragment,load_matrix_sync,IOConfig,GetMappingUtil --aiter-source \$AITER_CHECKOUT/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py --calculator-source ${HOME}/Development/ml/tools/amd_matrix_instruction_calculator-2ef91896bcdc4d26624f952e5c905c787cd9bc9e/matrix_calculator.py --local-source native_r9700/kernels/llama_gate_up_projection_f16.cpp --layout-spec build/f2-wmma/f2-wmma-physical-layout-spec.json --inverse-fixture build/f2-wmma/f2-wmma-physical-layout-inverse.npz --output logs/f2/wmma-physical-layout-proof.json"
    date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"
    tools/f2-wmma-layout-proof \
      --source-layout-version f16-row-major-nk-source-v1 \
      --physical-layout-version f2-wmma-physical-tile-v1 \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/samples/simple_hgemm.cpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/rocwmma.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/io_config.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/io_layout.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/mapping_util.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/accessors_impl.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/layout/matrix_layout_traits_impl.hpp" \
      --rocwmma-symbols matrix_b,col_major,fragment,load_matrix_sync,IOConfig,GetMappingUtil \
      --aiter-source "$AITER_CHECKOUT/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py" \
      --calculator-source ${HOME}/Development/ml/tools/amd_matrix_instruction_calculator-2ef91896bcdc4d26624f952e5c905c787cd9bc9e/matrix_calculator.py \
      --local-source native_r9700/kernels/llama_gate_up_projection_f16.cpp \
      --layout-spec "$layout_spec" \
      --inverse-fixture "$inverse_fixture" \
      --output logs/f2/wmma-physical-layout-proof.json
    status=$?
    printf "wrapper_exit_status: %d\n" "$status"
    exit "$status"
  } 2>&1 | tee "$log"
'
```

The required output is a concrete `EvidenceRef` with `record_kind: offline_review`, `evidence_slot: layout_proof`, `record_id: f2-wmma-physical-layout-proof-v1`, nonempty record/source/tool/spec/fixture/target/image/pack/input/output digests, and exactly empty `producer_kind`. It also requires nonempty `source_tensor_layout_version`, `physical_layout_version`, `layout_spec_path`, `layout_spec_sha256`, `inverse_fixture_path`, and `inverse_fixture_sha256`; exact source-element-to-physical-byte and 16x16 B-tile/LDS mapping; strides/alignment/padding/swizzle; `layout_origin: pinned_header|reviewed_local_v1`; inverse/conformance fixture input/output digests; and `layout_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`. Its `tool_digest` identifies the exact review tool/version or signed manual-review record digest. A `reviewed_local_v1` origin carries the task-set-5 hardware numerical record key/digest before G0; a missing source checkout, absent or unreviewed spec, or failed inverse fixture rejects task set 3 and keeps task set 4 blocked.

### F2 lane-map proof

```sh
/bin/bash -o pipefail -c '
  set -u
  mkdir -p build/f2-wmma logs/f2 \
    build/upstream/amd-matrix-instruction-calculator build/upstream/python
  PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
  CALC=build/upstream/amd-matrix-instruction-calculator/matrix_calculator.py
  CALC_URL=https://raw.githubusercontent.com/ROCm/amd_matrix_instruction_calculator/2ef91896bcdc4d26624f952e5c905c787cd9bc9e/matrix_calculator.py
  CALC_SHA=53b027855ca44120401eeff69f41961821d1a393b163e112f7aa4d2a313e185d
  if [ ! -f "$CALC" ]; then
    "$PY" -c "import pathlib,sys,urllib.request; pathlib.Path(sys.argv[2]).write_bytes(urllib.request.urlopen(sys.argv[1]).read())" "$CALC_URL" "$CALC"
  fi
  actual_calc_sha=$("$PY" -c "import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())" "$CALC")
  [ "$actual_calc_sha" = "$CALC_SHA" ] || { printf "calculator digest mismatch\n" >&2; exit 2; }
  if ! PYTHONPATH=build/upstream/python "$PY" -c "import tabulate" 2>/dev/null; then
    "$PY" -m pip install --quiet --target build/upstream/python tabulate==0.9.0
  fi
  export PYTHONPATH=build/upstream/python
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
    native_r9700/amdev_packets.cpp native_r9700/runtime_contract.cpp \
    native_r9700/prefill_npz.cpp native_r9700/vram_layout.cpp \
    native_r9700/vram_allocator.cpp native_r9700/dynamic_page_table.cpp \
    native_r9700/resident_memory.cpp native_r9700/vram_smoke_asset.cpp \
    native_r9700/hsa_code_image_asset.cpp native_r9700/model_weight_binder.cpp \
    native_r9700/llama_stage_layout.cpp native_r9700/llama_layer_executor.cpp \
    native_r9700/kernel_assets.cpp native_r9700/amdev_session.cpp \
    native_r9700/kernel_catalog.cpp native_r9700/device_memory.cpp \
    native_r9700/hardware_lock.cpp native_r9700/runtime.cpp \
    native_r9700/native_resource_worker.cpp native_r9700/wmma_lane_map_runner.cpp \
    -I native_r9700 -o build/f2-wmma/wmma_lane_map_gfx1201
  detail=logs/f2/wmma-calculator-detail.txt
  a_map=logs/f2/wmma-calculator-a.csv
  b_map=logs/f2/wmma-calculator-b.csv
  d_map=logs/f2/wmma-calculator-d.csv
  observed=logs/f2/wmma-lane-map-proof.json
  conformance=logs/f2/wmma-lane-map-conformance.json
  log=logs/f2/wmma-lane-map-proof.log
  {
    printf "%s\n" "command: pinned calculator outputs; tools/native-r9700-hardware-run build/f2-wmma/wmma_lane_map_gfx1201; python -m native_r9700.wmma_lane_map"
    date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"
    "$PY" "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --detail-instruction > "$detail"
    "$PY" "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --register-layout --A-matrix --csv > "$a_map"
    "$PY" "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --register-layout --B-matrix --csv > "$b_map"
    "$PY" "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --register-layout --D-matrix --csv > "$d_map"
    tools/native-r9700-hardware-run \
      build/f2-wmma/wmma_lane_map_gfx1201 \
      --asset-root native_r9700/kernels/wmma-lane-map-gfx1201-hsa-assets \
      --log "$observed"
    status=$?
    if [ "$status" -eq 0 ]; then
      "$PY" -m native_r9700.wmma_lane_map \
        --calculator-detail "$detail" --calculator-a "$a_map" \
        --calculator-b "$b_map" --calculator-d "$d_map" \
        --observed "$observed" \
        --asset-root native_r9700/kernels/wmma-lane-map-gfx1201-hsa-assets \
        --out "$conformance"
      status=$?
    fi
    printf "wrapper_exit_status: %d\n" "$status"
    exit "$status"
  } 2>&1 | tee "$log"
'
```

Expected log observations, all required: `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `wave_size: 32`, `instruction: v_wmma_f32_16x16x16_f16`, the pinned calculator source revision and layout digest, exact source/image/manifest paths and SHA-256 values, observed A/B/D register/lane/bit records matching the equations in §2, `record_kind: target_conformance`, `evidence_slot: conformance`, nonempty target/image/pack/producer/input/output digests, exactly `producer_kind: r9700_native`, exactly empty `tool_digest`, `lane_map_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`. A mismatch is a failed proof and must update this contract before task set 4; no compensating transpose is allowed.

### F2 standalone WMMA

This named section includes the mandatory NumPy oracle, accepted scalar/native projection, and WMMA comparisons over identical inputs; they are separate EvidenceRefs and acceptance outcomes.

```sh
/bin/bash -o pipefail -c '
  set -u
  mkdir -p build/f2-wmma logs/f2
  input_record=build/f2-wmma/f2-projection-inputs-fp16.npz
  layout_record=logs/f2/wmma-physical-layout-proof.json
  numpy_record=logs/f2/numpy-oracle.json
  native_record=logs/f2/native-projection.json
  native_log=logs/f2/native-projection.log
  log=logs/f2/standalone-wmma.log
  {
    printf "%s\n" "command: tools/f2-wmma-numpy-oracle and scalar_native_projection_gfx1201 and standalone_wmma_gfx1201 --input-record build/f2-wmma/f2-projection-inputs-fp16.npz --m 128 --k 2048 --n 8192 --tail-m 13 --tail-policy masked/padded --geometry-rule f2-wmma-64x64-m-tail-v1 --source-layout-version f16-row-major-nk-source-v1 --packing-version f2-wmma-physical-tile-v1 --layout-record logs/f2/wmma-physical-layout-proof.json --numpy-oracle-record logs/f2/numpy-oracle.json --native-projection-record logs/f2/native-projection.json"
    date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"
    tools/f2-wmma-numpy-oracle \
      --input-record "$input_record" \
      --m 128 --k 2048 --n 8192 --tail-m 13 \
      --tail-policy masked/padded \
      --geometry-rule f2-wmma-64x64-m-tail-v1 \
      --source-layout-version f16-row-major-nk-source-v1 \
      --record "$numpy_record"
    numpy_status=$?
    if [ "$numpy_status" -eq 0 ]; then
      tools/native-r9700-hardware-run \
        build/f2-wmma/scalar_native_projection_gfx1201 \
        --input-record "$input_record" \
        --m 128 --k 2048 --n 8192 --tail-m 13 \
        --tail-policy masked/padded \
        --geometry-rule f2-wmma-64x64-m-tail-v1 \
        --source-layout-version f16-row-major-nk-source-v1 \
        --evidence-record "$native_record" \
        --log "$native_log"
      native_status=$?
    else
      native_status=$numpy_status
    fi
    if [ "$native_status" -eq 0 ]; then
      tools/native-r9700-hardware-run \
        build/f2-wmma/standalone_wmma_gfx1201 \
        --asset-root native_r9700/kernels/linear-wmma-f16-hsa-assets \
        --input-record "$input_record" \
        --m 128 --k 2048 --n 8192 --tail-m 13 \
        --tail-policy masked/padded \
        --geometry-rule f2-wmma-64x64-m-tail-v1 \
        --source-layout-version f16-row-major-nk-source-v1 \
        --packing-version f2-wmma-physical-tile-v1 \
        --layout-record "$layout_record" \
        --numpy-oracle-record "$numpy_record" \
        --native-projection-record "$native_record" \
        --numerical-policy F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1 \
        --log logs/f2/standalone-wmma.json
      status=$?
    else
      status=$native_status
    fi
    printf "wrapper_exit_status: %d\n" "$status"
    exit "$status"
  } 2>&1 | tee "$log"
'
```

Expected observations, all required: both requests use the same canonical input record and emit the same `input_digest` in the NumPy, scalar/native, and WMMA evidence; request/shape `M=128,K=2048,N=8192`; explicit bounded tail `M=13`; `input_dtype: fp16`, `accumulator_dtype: fp32`, `output_dtype: fp16`; `wave_size: 32`; `instruction: v_wmma_f32_16x16x16_f16`; `source_tensor_layout_version: f16-row-major-nk-source-v1`; `packing_version: f2-wmma-physical-tile-v1` bound to the passing `record_kind: offline_review`, `evidence_slot: layout_proof` EvidenceRef; separate `numpy_oracle_record_id` (`record_kind: offline_oracle`, `evidence_slot: numpy_oracle`) and `native_projection_record_id` (`record_kind: target_conformance`, `evidence_slot: scalar_native_projection`, `producer_kind: r9700_native`); native subject target/image/pack/record/input/output digests; finite valid outputs; unchanged padding/sentinel rows; separate NumPy and scalar/native max/mean error fields within the reviewed policy; exact manifest resource/descriptor fields; `standalone_wmma_status: pass`; `failure_stage: none`; `exit_status: 0`; and `wrapper_exit_status: 0`. The record must identify `benchmark_scope: gpu_compute`/standalone, not warm service throughput. A CPU/scalar-only run, mismatched input digest, missing native record, or request-unbound log is failure.

### F2 G0 publication — canonical full invocation (P3 copies verbatim)

P3's G0 migration MUST copy the complete hardware invocation in the block below, including every argument after `--g0`, and may append only P3 pack-consumption outputs/observations after that exact call. It MUST NOT omit an argument, add a default, or reinterpret an EvidenceRef; this block is the sole F2 G0 CLI authority.

```sh
/bin/bash -o pipefail -c '
  set -u
  mkdir -p logs/f2
  log=logs/f2/g0-publication.log
  {
    printf "%s\n" "command: F2 G0 HSA/asset/catalog/numerical/evidence gates and hardware publication"
    date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"
    status=0
    if [ "$status" -eq 0 ]; then
      test -s .superpowers/swarm/reports/g0-wmma-conformance.md || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
        tests/native_r9700/test_wmma_lane_map_asset.py \
        tests/native_r9700/test_linear_wmma_f16_asset.py \
        tests/native_r9700/test_hsa_code_image_generator.py \
        tests/native_r9700/test_hsa_code_image_loader.py \
        tests/native_r9700/test_kernel_assets.py \
        tests/native_r9700/test_kernel_catalog.py -v || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      tools/native-r9700-hardware-run \
        build/f2-wmma/standalone_wmma_gfx1201 \
        --g0 \
        --asset-root native_r9700/kernels/linear-wmma-f16-hsa-assets \
        --source-layout-version f16-row-major-nk-source-v1 \
        --tail-policy masked/padded \
        --geometry-rule f2-wmma-64x64-m-tail-v1 \
        --packing-version f2-wmma-physical-tile-v1 \
        --layout-record logs/f2/wmma-physical-layout-proof.json \
        --numpy-oracle-record logs/f2/numpy-oracle.json \
        --native-projection-record logs/f2/native-projection.json \
        --g0-report .superpowers/swarm/reports/g0-wmma-conformance.md \
        --log logs/f2/g0-wmma-dispatch.log || status=$?
    fi
    printf "wrapper_exit_status: %d\n" "$status"
    exit "$status"
  } 2>&1 | tee "$log"
'
```

Expected observations, all required: F2 HSA/asset/catalog/numerical/evidence gates exit 0 (no P3 pack contract or pack-manifest test is a G0 prerequisite); the report contains `g0_status: pass`, the exact calculator expectation and independent hardware lane-map result, target `1002:7551`/`gfx1201`, source/image/manifest SHA-256 values, the passing `record_kind: offline_review`, `evidence_slot: layout_proof` physical-layout EvidenceRef, descriptor/kernarg/wave/resource/ISA records, canonical fixed `K=2048,N=8192` family with runtime `1<=M<=128` under `tail_policy: masked/padded` and `geometry_rule: f2-wmma-64x64-m-tail-v1`, `M=13` tail result, `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`, `source_tensor_layout_version: f16-row-major-nk-source-v1`, `weight_packing_version: f2-wmma-physical-tile-v1`, separate NumPy (`record_kind: offline_oracle`, `evidence_slot: numpy_oracle`) and request-bound native projection (`record_kind: target_conformance`, `evidence_slot: scalar_native_projection`, `producer_kind: r9700_native`) EvidenceRefs sharing each `input_digest`, standalone GPU-compute performance evidence, reviewer result, explicit replacement/supersession rules, and `pack_sha256` equal to the canonical RFC8785 JCS preimage digest. The dispatch log must contain matching G0 image/source/pack/layout/evidence digests, `standalone_wmma_status: pass`, separate NumPy/native comparison passes, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`. No G0 record may be published from `cpu_reference`, a stale image, a source-layout-as-pack alias, or a request-unbound log. P3 later validates exact immutable G0 consumption; it does not gate this command or regenerate any evidence.

## P1

The fixed downstream `TGPUConformanceClient` binary and subcommands below are the exact P1 CLI contract. Xcode 26.6 build `17F113` and DriverKit SDK 25.5 are selected; production distribution credentials remain a separate promotion blocker.

### SDK/build/install preflight and local install

```sh
xcode-select -p
xcrun --sdk driverkit --show-sdk-version
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -target TGPUConformanceClient -configuration Debug \
  CONFIGURATION_BUILD_DIR=${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug
./install_nosip.sh
```

Verified preflight on 2026-08-26: active developer directory `/Applications/Xcode.app/Contents/Developer`; Xcode 26.6 build `17F113`; DriverKit SDK `25.5` at `/Applications/Xcode.app/Contents/Developer/Platforms/DriverKit.platform/Developer/SDKs/DriverKit25.5.sdk`. The local installer may use only the NoSIP development entitlement. No command in this section launches or links `Shared/server.c`. Apple PCI distribution entitlement, profiles, Developer ID/notarization credentials, and approved external signing invocation remain promotion-only inputs, separate from the cleared SDK gate.

### P1 cold lifecycle

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug/tgpu-conformance-client \
  cold-lifecycle --service org.tinygrad.tinygpu.driver2 \
  --pci-id 1002:7551 --architecture gfx1201 \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/cold-lifecycle.log
```

Required observations are a fresh DEXT cold attach, ordered lifecycle stages, `TGPU_QUERY_CAPABILITIES` with `abi_major: 1`, `abi_minor: 0`, exact PCI/architecture identity, and a ready health record. The command uses the DriverKit user client directly and never the legacy proxy. A missing DriverKit SDK or DEXT is a blocked prerequisite, not a pass or a reason to fall back to a raw socket.

### P1 malformed submission, stale/client-death, queue reset, and bounded fault query

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug/tgpu-conformance-client \
  malformed-submit --service org.tinygrad.tinygpu.driver2 \
  --cases wrong-record-size,absolute-address,unbound-binding,stale-handle \
  --expect-status TGPU_STATUS_SUBMISSION_REJECTED \
  --expect-no-queue-mutation \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/malformed-submit.log

${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug/tgpu-conformance-client \
  client-death --service org.tinygrad.tinygpu.driver2 \
  --close-with-live-resources --reopen --replay-handles \
  --expect-status TGPU_STATUS_INVALID_HANDLE \
  --expect-empty-new-namespace \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/client-death.log

${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug/tgpu-conformance-client \
  queue-reset --service org.tinygrad.tinygpu.driver2 \
  --owner-only --expect-pending-fence-status TGPU_STATUS_CANCELED \
  --expect-no-replay \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/queue-reset.log

${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug/tgpu-conformance-client \
  fault-query --service org.tinygrad.tinygpu.driver2 \
  --scope client --max-text-bytes 192 --expect-redacted \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/fault-query.log
```

The malformed cases must return status 12 before queue mutation or signal allocation; stale/cross-client handles must return status 5; queue reset must be accepted only for the queue owner and must explicitly fail pending fences; fault text must be at most 192 bytes and must not contain raw registers, addresses, prompts, or tokens. No invocation uses a socket/proxy command.

### P1 device recovery

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug/tgpu-conformance-client \
  device-recovery --service org.tinygrad.tinygpu.driver2 \
  --recovery-service org.tinygrad.tinygpu.recovery \
  --preflight-normal-reset-denied --fault-source physical \
  --expect-device-epoch-increment --expect-stale-handle-rejection \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/device-recovery.log
```

The fixed client first proves that a normal inference connection receives `TGPU_STATUS_PERMISSION_DENIED` for device reset, then uses the exact recovery role and entitlement. If physical fault injection is unavailable, the command must record `physical_fault_injection: unavailable` and `status: blocked`; it must not claim recovery success and must not substitute the old kernel-proof/raw-proxy control. Once hardware injection exists, the command requires a bounded fault, serialized recovery, incremented `device_epoch`, stale-handle rejection, and a clean new-client capability query. The fixed CLI remains the task-5 implementation target even while the physical injector is unavailable.

### P1 exact G0 binding

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-tinygpu-device-owner/extra/usbgpu/tbgpu/installer/build/Debug/tgpu-conformance-client \
  g0-binding --service org.tinygrad.tinygpu.driver2 \
  --g0-report ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/.superpowers/swarm/reports/g0-wmma-conformance.md \
  --require-status-field g0_status=pass \
  --require-record-id-field g0_record_id \
  --require-image-sha256-field g0_image_sha256 \
  --require-target-field g0_target \
  --require-entry-field g0_entry \
  --require-pci-id 1002:7551 --require-architecture gfx1201 \
  --expect-recomputed-digest-match --expect-no-fallback \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/g0-binding.log
```

The client reads exactly the accepted report fields `g0_record_id`, `g0_image_sha256`, `g0_target`, and `g0_entry`, passes only opaque audit metadata plus image bytes to the TGPU client, and requires the executable-admission response's recomputed digest/target/entry to match those values. The DriverKit implementation does not parse a P3 Kernel Pack. Missing report, non-pass status, digest/target/entry mismatch, target other than `1002:7551`/`gfx1201`, or fallback is fail-closed. No command regenerates G0.

## P3

### P3 schema

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp native_r9700/runtime_contract.cpp \
  native_r9700/prefill_npz.cpp native_r9700/vram_layout.cpp \
  native_r9700/vram_allocator.cpp native_r9700/dynamic_page_table.cpp \
  native_r9700/resident_memory.cpp native_r9700/vram_smoke_asset.cpp \
  native_r9700/hsa_code_image_asset.cpp native_r9700/model_weight_binder.cpp \
  native_r9700/amdev_session.cpp native_r9700/kernel_pack.cpp \
  native_r9700/kernel_catalog.cpp native_r9700/device_memory.cpp \
  native_r9700/hardware_lock.cpp native_r9700/llama_stage_layout.cpp \
  native_r9700/llama_layer_executor.cpp native_r9700/kernel_assets.cpp \
  native_r9700/runtime.cpp native_r9700/native_resource_worker.cpp \
  native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run
"$PY" -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_pack_manifest.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py -v
```

Expected observations:

- The direct C++ build succeeds with `kernel_pack.cpp` linked into the existing runner shape; `--lifecycle-dry-run` remains hardware-free and passes.
- Pack contract tests prove exact identity/key lookup, no implicit version upgrade, no plugin/runtime-document path, output-preserving rejection, and reuse of existing descriptor/image admission.
- Manifest tests prove canonical schema validation and deterministic generated-record output. Malformed-pack cases (unknown key, missing/unknown license, missing source/image digest, wrong target, duplicate symbol, malformed kernarg/resource/geometry/tail padding, contradictory dtype/shape/layout/packing/numerics/reference-set kind, missing or unresolved `EvidenceRef`, unknown or mismatched kind/slot, missing physical-layout proof, invalid `pack_sha256` preimage digest, missing required matrix fields or nonempty exact-empty fields, both/neither benchmark outcome fields, `native_run` collapsed into `target_conformance`, and `cpu_reference` substituted for native evidence) reject with a nonempty reason before any generated output is published; the input/output sentinel remains unchanged.
- No runtime process opens `docs/upstream-reference-manifest.yaml` or a `.pack.json` file.

### P3 malformed-pack rejection (focused observation)

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
"$PY" -m pytest tests/native_r9700/test_kernel_pack_manifest.py -v
```

Expected: every malformed record exits through offline validation with a named rejection; no malformed record produces a generated C++ initializer or becomes visible to the runtime catalog. In particular, a component license of `unknown` is a hard rejection, not a warning or pending state.

### P3 scalar migration

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
"$PY" -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_pack_manifest.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py \
  tests/native_r9700/test_runtime_contract.py -v
mkdir -p build/native-r9700-runtime logs
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp native_r9700/runtime_contract.cpp \
  native_r9700/prefill_npz.cpp native_r9700/vram_layout.cpp \
  native_r9700/vram_allocator.cpp native_r9700/dynamic_page_table.cpp \
  native_r9700/resident_memory.cpp native_r9700/vram_smoke_asset.cpp \
  native_r9700/hsa_code_image_asset.cpp native_r9700/model_weight_binder.cpp \
  native_r9700/amdev_session.cpp native_r9700/kernel_pack.cpp \
  native_r9700/kernel_catalog.cpp native_r9700/device_memory.cpp \
  native_r9700/hardware_lock.cpp native_r9700/llama_stage_layout.cpp \
  native_r9700/llama_layer_executor.cpp native_r9700/kernel_assets.cpp \
  native_r9700/runtime.cpp native_r9700/native_resource_worker.cpp \
  native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
/bin/bash -o pipefail -c 'mkdir -p logs; log=logs/p3-scalar-migration-native-prefill.log; { printf "%s\\n" "command: tools/native-r9700-hardware-run env APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock build/native-r9700-runtime/native_r9700_runner --native-prefill-proof --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --token-ids-json [128000,128001] --out logs/p3-scalar-migration.npz --log logs/p3-scalar-migration-runner.log"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; tools/native-r9700-hardware-run env APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock build/native-r9700-runtime/native_r9700_runner --native-prefill-proof --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --token-ids-json "[128000,128001]" --out logs/p3-scalar-migration.npz --log logs/p3-scalar-migration-runner.log; status=$?; printf "wrapper_exit_status: %d\\n" "$status"; exit "$status"; } >"$log" 2>&1'
```

- The request-bound hardware log identifies R9700 `1002:7551`, `gfx1201`, the selected pack/image/pack-preimage digests, entry/dispatch identity, and resolved `EvidenceRef` IDs and digests: `conformance` is `target_conformance/conformance`, `native_run` is `native_run/native_run`, and `resource_review`/`isa_review` are `offline_review/resource_review` and `offline_review/isa_review`. A promoted performance control carries its `benchmark/benchmark` record; a correctness-control pack carries a nonempty `benchmark_not_applicable_reason`. The hardware record is never relabeled `cpu_reference` and never invokes a CPU fallback after cache acceptance.
Expected real-hardware observations:

- The selected records are the exact scalar pack name/version and entry symbols requested by the migrated graph; no older/newer/other compatibility record is chosen.
- The runner log has `record_kind: native_run`, `evidence_slot: native_run`, `producer_kind: r9700_native`, `native_prefill_acceptance: pass`, `native_prefill_full_layer_loop_status: pass`, `failure_stage: none`, and `exit_status: 0`; the wrapper log has `wrapper_exit_status: 0`.
- The request-bound hardware log identifies R9700 `1002:7551`, `gfx1201`, the selected pack/image/pack-preimage digests, entry/dispatch identity, and the resolved `EvidenceRef` IDs/digests. It is `record_kind: native_run`, `evidence_slot: native_run`, `producer_kind: r9700_native`; it must not be relabeled `cpu_reference` and must not invoke a CPU fallback after cache acceptance.
- B0 scalar image bytes, descriptors, S-1/final-token behavior, producer-owned KV truth, and accepted-cache/fallback semantics remain unchanged; the migrated scalar record keeps its exact reviewed kernarg tail-padding value and zeroes that suffix before submission.

### P3 G0 migration

The F2 G0 publication command is copied verbatim below. P3 adds no CLI defaults, substitutions, or flags; the only P3 additions are the pack-consumption observations after the command.

```sh
/bin/bash -o pipefail -c '
  set -u
  mkdir -p logs/f2
  log=logs/f2/g0-publication.log
  {
    printf "%s\n" "command: F2 G0 HSA/asset/catalog/numerical/evidence gates and hardware publication"
    date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"
    status=0
    if [ "$status" -eq 0 ]; then
      test -s .superpowers/swarm/reports/g0-wmma-conformance.md || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
        tests/native_r9700/test_wmma_lane_map_asset.py \
        tests/native_r9700/test_linear_wmma_f16_asset.py \
        tests/native_r9700/test_hsa_code_image_generator.py \
        tests/native_r9700/test_hsa_code_image_loader.py \
        tests/native_r9700/test_kernel_assets.py \
        tests/native_r9700/test_kernel_catalog.py -v || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      tools/native-r9700-hardware-run \
        build/f2-wmma/standalone_wmma_gfx1201 \
        --g0 \
        --asset-root native_r9700/kernels/linear-wmma-f16-hsa-assets \
        --source-layout-version f16-row-major-nk-source-v1 \
        --tail-policy masked/padded \
        --geometry-rule f2-wmma-64x64-m-tail-v1 \
        --packing-version f2-wmma-physical-tile-v1 \
        --layout-record logs/f2/wmma-physical-layout-proof.json \
        --numpy-oracle-record logs/f2/numpy-oracle.json \
        --native-projection-record logs/f2/native-projection.json \
        --g0-report .superpowers/swarm/reports/g0-wmma-conformance.md \
        --log logs/f2/g0-wmma-dispatch.log || status=$?
    fi
    printf "wrapper_exit_status: %d\n" "$status"
    exit "$status"
  } 2>&1 | tee "$log"
'
```

P3 pack-consumption observations (the only additions to the copied F2 command contract):

- Offline validation consumes the accepted task-set-2/3 generated record and the immutable `.superpowers/swarm/reports/g0-wmma-conformance.md` record. It records the exact F2 names `f2-linear-wmma-f16-v1`, `f2-linear-gate-up-f16-v1`, `linear_wmma_f16`, `linear_wmma_f16.image`, `linear_wmma_f16.json`, `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`, and `f2-wmma-64x64-m-tail-v1`, together with the exact image/source/digest, descriptor/resource, shape/tail, numerical, ISA, and hardware evidence; the physical layout record is `record_kind: offline_review`, `evidence_slot: layout_proof`; it does not regenerate the lane map/image or alter tolerance/results, and it does not emit/admit the reserved `f2-wmma-physical-tile-v1` without that resolved layout record.
- The P3 pack-consumption output identifies the selected pack name/version, canonical `pack_sha256` preimage digest, imported image SHA-256, exact G0 record ID, `target: gfx1201`, entry symbol, and the resolved `EvidenceRef` IDs/digests. It records `pack_validation: pass`, `pack_consumption: pass`, `load_status: pass`, `dispatch_status: pass`, finite output/tail comparison within the immutable G0 policy, and process `exit_status: 0`.
- The hardware run is request-bound native evidence (`record_kind: native_run`, `evidence_slot: native_run`, `producer_kind: r9700_native`) with the resolved scalar-native/NumPy input binding; the scalar projection is `record_kind: target_conformance`, `evidence_slot: scalar_native_projection`, `producer_kind: r9700_native`; the NumPy reference is `record_kind: offline_oracle`, `evidence_slot: numpy_oracle`, `producer_kind: cpu_reference`, with target/image/pack/tool fields exactly empty. Any pack/image/G0 mismatch, missing evidence, or nonzero load/dispatch status blocks migration and returns to F2; it is not repaired by selecting another version.

The copied F2 invocation retains the complete argument set: `--asset-root`, `--source-layout-version`, `--tail-policy`, `--geometry-rule`, `--packing-version`, `--layout-record`, `--numpy-oracle-record`, `--native-projection-record`, `--g0-report`, and `--log`. P3 adds no replacement defaults or alternate runner mode.

## Q1

Set these variables explicitly in every command block:

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
QWEN_MODEL=${HOME}/Development/ml/models/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
MANIFEST=docs/upstream-reference-manifest.yaml
```

### Q1 source-pin check

```sh
mkdir -p logs
"$PY" -m native_r9700.qwen_text_adapter \
  --check-source-pin \
  --model "$QWEN_MODEL" \
  --manifest "$MANIFEST" \
  --out logs/q1-source-pin.json
```

Expected `logs/q1-source-pin.json` observations: `status="pass"` for converted-snapshot identity, `fallback_used=false`, `model_revision="3e6447f082e89cc7f0bc6e5441afd38dfce760ff"`, `base_model_revision="unavailable_in_pinned_conversion_metadata"`, `promotion_gate="blocked_base_model_revision"`, `mlx_vlm_revision="2b31570bdee86e2cdeea049761885aeed524a98c"`, `mlx_lm_revision="e2f2fb2aef987f86878d17638446183cffe21fe4"`, `model_fingerprint="4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371"`, `local_shard_count=3`, and all four metadata plus three shard digests equal §2. A missing or mismatched file must exit nonzero and name the concrete blocker; it must not select another directory. `status="pass"` here does not clear the Q1 promotion gate.

### Q1 tensor inventory

```sh
"$PY" -m native_r9700.qwen_text_adapter \
  --inventory \
  --model "$QWEN_MODEL" \
  --manifest "$MANIFEST" \
  --source-pin-report logs/q1-source-pin.json \
  --out logs/q1-qwen-tensor-inventory.json
```

Expected output: `schema_version=2`, `header_only=true`, `tensor_count=2180`, `language_model_tensor_count=1847`, `vision_tensor_count=333`, `affine_stem_count=498`, `affine_entry_count=1494`, `tensor_payload_bytes=16054262240`, `inventory_sha256=508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`, and `model_fingerprint=4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`. Tensor records must contain exactly the six §4.2 fields and affine classification must be in its separate sorted table; the command must emit no decoded weight array and must fail closed on any index/header/shard mismatch.

### Q1 hybrid-state capture

The concrete text-only probe is the tokenizer-derived sequence for `The capital of France is`: `[760,6511,314,9338,369]`. The prefix is `[760,6511,314,9338]`, `S=5`, `N=S-1=4`, and final prompt token is `369`.

```sh
"$PY" -m native_r9700.qwen_hybrid_cache \
  --capture-hybrid-state \
  --model "$QWEN_MODEL" \
  --token-ids-json '[760,6511,314,9338,369]' \
  --out logs/q1-qwen-hybrid-state.qwenspill \
  --report logs/q1-qwen-hybrid-state.json
```

The task-set-3 restore command consumes that opaque record and exercises the executable MLX boundary separately:

```sh
"$PY" -m native_r9700.qwen_hybrid_cache \
  --restore-hybrid-state \
  --model "$QWEN_MODEL" \
  --spill logs/q1-qwen-hybrid-state.qwenspill \
  --token-ids-json '[760,6511,314,9338,369]' \
  --out logs/q1-qwen-hybrid-restore.json
```

Expected capture report observations: `producer_kind="cpu_reference"`, `text_only=true`, `model_fingerprint=4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`, `runtime_layers=64`, `arrays_cache_layers=48`, `kv_cache_layers=16`, `committed_position=4`, `final_token_id=369`, full-attention layers exactly `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`, and a deterministic state/whole-record digest. Full-attention leaves must be `(1,4,4,256)`/`bfloat16`; linear leaves must be `(1,3,10240)`/`bfloat16` and `(1,48,128,128)`/`float32`. The capture must contain prefix state only; the final token is not in the captured K/V offset. The restore report must additionally prove actual MLX arrays were assigned to the pinned `ArraysCache.state`/`KVCache.state` with exact little-endian dtype/shape/layout and must reject opaque-leaf assignment.

### Q1 oracle fixtures

```sh
"$PY" -m native_r9700.ref_fixtures \
  --generate-qwen \
  --model "$QWEN_MODEL" \
  --token-ids-json '[760,6511,314,9338,369]' \
  --fixtures-dir tests/native_r9700/fixtures \
  --inventory logs/q1-qwen-tensor-inventory.json \
  --report logs/q1-qwen-oracle-fixtures.json
```

Expected output files are exactly `qwen_prompts.json`, `qwen_affine_windows.npz`, `qwen_hybrid_state_samples.npz`, `qwen_oracle_trace.npz`, and `qwen_fixtures_schema.json` under `tests/native_r9700/fixtures/`. The schema must contain `model_fingerprint=4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`, `base_model_revision="unavailable_in_pinned_conversion_metadata"`, `inventory_schema_version=2`, `inventory_sha256=508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`, exact source revisions/shard digests, deterministic per-file SHA-256 values, text-only rejection IDs, `S=5`/`prefix_length=4`/`final_token_id=369`, explicit recurrent/full-attention component metadata, and `producer_kind="cpu_reference"`, `native_evidence=false`. Regeneration must be byte-for-byte deterministic and must not alter any existing Llama fixture.

### Q1 oracle parity

```sh
"$PY" -m native_r9700.qwen_parity \
  --compare-fixtures \
  --model "$QWEN_MODEL" \
  --inventory logs/q1-qwen-tensor-inventory.json \
  --token-ids-json '[760,6511,314,9338,369]' \
  --out logs/q1-qwen-parity.json
```

Expected `logs/q1-qwen-parity.json`: `status="pass"`, exact model fingerprint match, `inventory_sha256=508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`, `producer_kind="cpu_reference"`, `prefix_length=4`, final-token input exactly `[369]`, and token/output comparisons localized to the declared fixture boundaries. It must record `native_evidence=false`; semantic similarity or an artifact relabeled `r9700_native` is not a pass. After an accepted cache, any decode failure is an error, not fallback/recompute.

### Q1 package review

```sh
"$PY" -m pytest \
  tests/native_r9700/test_qwen_text_adapter.py \
  tests/native_r9700/test_qwen_affine4_source.py \
  tests/native_r9700/test_model_weight_binder_contract.py \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py \
  tests/native_r9700/test_qwen_parity.py \
  tests/native_r9700/test_ref_fixtures.py \
  tests/native_r9700/test_fixture_catalog.py -v
"$PY" -m native_r9700.qwen_text_adapter \
  --check-source-pin --model "$QWEN_MODEL" --manifest "$MANIFEST" \
  --out logs/q1-source-pin.json
"$PY" -m native_r9700.qwen_text_adapter \
  --inventory --model "$QWEN_MODEL" --manifest "$MANIFEST" \
  --source-pin-report logs/q1-source-pin.json \
  --out logs/q1-qwen-tensor-inventory.json
"$PY" -m native_r9700.qwen_parity \
  --compare-fixtures --model "$QWEN_MODEL" \
  --fixtures-dir tests/native_r9700/fixtures \
  --token-ids-json '[760,6511,314,9338,369]' \
  --inventory logs/q1-qwen-tensor-inventory.json \
  --out logs/q1-qwen-parity.json
"$PY" - <<'PY'
import json
from pathlib import Path

expected = {
    "model_fingerprint": "4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371",
    "inventory_sha256": "508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4",
}
paths = {
    "inventory": Path("logs/q1-qwen-tensor-inventory.json"),
    "fixtures": Path("tests/native_r9700/fixtures/qwen_fixtures_schema.json"),
    "package": Path("logs/q1-qwen-acceptance-package.json"),
}
records = {}
for label, path in paths.items():
    if not path.is_file():
        raise SystemExit(f"missing Q1 identity record: {path}")
    records[label] = json.loads(path.read_text(encoding="utf-8"))
for label, record in records.items():
    for key, value in expected.items():
        if record.get(key) != value:
            raise SystemExit(f"{label} {key} does not match frozen Q1 identity")
if records["fixtures"].get("base_model_revision") != "unavailable_in_pinned_conversion_metadata":
    raise SystemExit("fixture base-model provenance marker is not fail-closed")
if records["package"].get("base_model_revision") != "unavailable_in_pinned_conversion_metadata":
    raise SystemExit("package base-model provenance marker is not fail-closed")
if records["package"].get("producer_kind") != "cpu_reference" or records["package"].get("native_evidence") is not False:
    raise SystemExit("package identity is not oracle-only")
print("q1 model/inventory identity matches inventory, fixtures, and package")
PY
```

Expected review result: all listed focused contracts pass; source/model/inventory/fixture/parity digests agree; the inventory command is rerun in this block and the deterministic comparison proves its model fingerprint and inventory digest match both fixture schema and the task-set-6 package identity record; no report or fixture carries `r9700_native`; the base-model provenance gate remains explicitly blocked; no Critical or Important review finding remains after re-review. No hardware run is part of Q1. Supervisor may append the repository's normal `git diff --check` after the source/report review; agents do not run it.
