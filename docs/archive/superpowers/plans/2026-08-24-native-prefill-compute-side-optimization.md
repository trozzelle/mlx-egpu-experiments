# Native R9700 Prefill Compute-Side Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce accepted Llama 3.2 1B native R9700 prefill time by accurately attributing the remaining wall time, profiling all ten GPU stages, narrowing only proven-expensive barriers, executing prompt positions in token blocks, and optimizing the measured attention-score hotspot without changing the prompt-cache contract.

**Architecture:** Keep `512a58a` as the direct-ring control and `f18da4f` as the approved design baseline. First make host/RPC accounting exclusive and add GPU-clock boundaries in unused bytes of the proven compute-control queue page. Then promote terminal-only completion and one dependency-grounded K/V overlap only when hardware A/B evidence passes. Finally replace the token-serial executor with block-capable buffer/geometry wiring already supported by the kernels, run the block-size ladder, and apply the in-stage Q/RoPE precompute only when the stage profile confirms attention score is dominant.

**Tech Stack:** C++17, AMD gfx1201 PM4/HSA code images, TinyGPU.app `APLRemotePCIDevice`/`PCIIface`, Python 3.12.8, pytest, NumPy/MLX/mlx-lm parity and serving harnesses.

**Design spec:** `docs/archive/superpowers/specs/2026-08-24-native-prefill-compute-side-optimization-design.md`

## Global Constraints

- Use `${PY}` for every Python command.
- Hardware target is AMD Radeon AI PRO R9700, PCI ID `1002:7551`, architecture `gfx1201`.
- Serialize every hardware command through the existing hardware lock; never kill the live TinyGPU server.
- `producer_kind=r9700_native` remains hardware-backed and fail-closed. CPU/NumPy is oracle evidence only.
- Preserve S-1 cache semantics: the prompt cache contains the prefix and mlx-lm receives only the final prompt token.
- Preserve direct-ring batching. Do not promote indirect buffers; record only the corrected IB encoding fact (`PACKET3 count=2`, valid bit).
- Do not create a timestamp or kernarg mapping. GPU timestamps use unused bytes in the proven compute-control queue page at `kRptrVa`.
- Production PM4 is byte-identical to the 590-dword control until a barrier mode passes its explicit hardware A/B gate.
- Do not add a silent runtime fallback from a promoted block size to block size 1.
- Do not touch Qwen, request batching, network transport, or SDMA design in this plan.
- Do not sweep the pre-existing archive-reorganization changes under `.superpowers/swarm/` and `docs/archive/tasks/native-r9700-producer/` into task commits.
- Pre-existing unrelated failures in the frozen 59-dword test, raw HIP generator test, and missing `HardwareLock::*` test closures are not part of this plan.
- Use the full build command from `AGENTS.md`; shorter historical commands omit current runtime sources.

## Execution branch and evidence layout

At execution time, use `superpowers:using-git-worktrees` and create an isolated worktree from `feature/native-r9700-producer` at or after `f18da4f`:

```sh
git worktree add .worktrees/opt-compute-side-token-blocks -b opt/compute-side-token-blocks feature/native-r9700-producer
```

Keep hardware evidence ignored under:

```text
logs/compute-side-opt/
  baseline/
  attribution/
  gpu-profile/
  barriers/
  blocks/{1,2,4,8,16,32}/
  score-precompute/
  acceptance/
```

Use these shell variables in every hardware task:

```sh
PY="${PY:?set PY to the pinned Python 3.12.8 interpreter}"
SOCK=${TMPDIR}/tinygpu.sock
MODEL=<tinygrad-kv-worker-worktree>/mlx_models/meta-Llama-3.2-1B-Instruct
RUNNER=$PWD/build/native-r9700-runtime/native_r9700_runner
export APL_REMOTE_SOCK=$SOCK
export NATIVE_R9700_PREFILL_RUNNER=$RUNNER
```

---

## Task 0: Freeze the accepted control and benchmark artifacts

**Files:**
- Read: `docs/archive/superpowers/specs/2026-08-24-native-prefill-compute-side-optimization-design.md`
- Read: `docs/archive/superpowers/plans/2026-08-24-native-prefill-compute-batching-inpage-kernargs.md`
- Evidence only: `logs/compute-side-opt/baseline/`

**Interfaces:**
- Consumes: accepted direct-ring implementation at `23578fa` and structured timing at `512a58a`.
- Produces: immutable baseline logs, NPZ digest, three-run wall median, and exact C1R/C2R token sets used by every later gate.

- [ ] **Step 1: Build the exact control**

Run the full C++ command from `AGENTS.md`, writing `build/native-r9700-runtime/native_r9700_runner`.

Expected: exit 0 and no compiler warnings.

- [ ] **Step 2: Run the hardware health gate**

```sh
mkdir -p logs/compute-side-opt/baseline
APL_REMOTE_SOCK=$SOCK $RUNNER --kernel-proof \
  > logs/compute-side-opt/baseline/kernel-proof.log 2>&1
APL_REMOTE_SOCK=$SOCK $RUNNER --vram-smoke \
  > logs/compute-side-opt/baseline/vram-smoke.log 2>&1
```

Expected in both logs: `pci_id: 1002:7551`, `arch: gfx1201`, `exit_status: 0`.

- [ ] **Step 3: Capture three prompt-128 direct-run controls**

```sh
TOKENS128=$($PY -c "import json; d=json.load(open('tests/native_r9700/fixtures/prompts.json'))['prompt-128']['token_ids']; print(json.dumps(d[:128]))")
for run in 1 2 3; do
  APL_REMOTE_SOCK=$SOCK $RUNNER --native-prefill-proof \
    --model "$MODEL" --token-ids-json "$TOKENS128" \
    --out "logs/compute-side-opt/baseline/prompt128-$run.npz" \
    --log "logs/compute-side-opt/baseline/prompt128-$run.log" \
    > "logs/compute-side-opt/baseline/prompt128-$run.stdout" 2>&1 || exit 1
done
```

Expected for every run: `n_prefix: 128`, `kernel_count: 20480`, `compute_submit_count: 2048`, `native_prefill_acceptance: pass`, `exit_status: 0`. Record the three `wall_usec` values and median; the prior control is approximately 43.7 seconds.

- [ ] **Step 4: Record immutable output identities**

```sh
shasum -a 256 logs/compute-side-opt/baseline/prompt128-*.npz \
  > logs/compute-side-opt/baseline/npz-sha256.txt
```

Do not commit ignored logs. Later tasks compare against these files by path and digest.

---

## Task 1: Make host phase accounting exclusive and complete

**Files:**
- Modify: `native_r9700/amdev_session.h:13-29`
- Modify: `native_r9700/amdev_session.cpp` (`PhaseTimers` accumulation and close snapshot)
- Modify: `native_r9700/runtime_contract.cpp:719-905`
- Modify: `native_r9700/runtime.h:6243-6270`
- Modify: `native_r9700/runner.cpp:157-225`
- Create: `tests/native_r9700/test_prefill_phase_accounting.py`

**Interfaces:**
- Consumes: `PhaseTimers`, `NativePrefillResult::wall_usec`, and existing `ScopedUsec` leaf timers.
- Produces: `finalize_phase_accounting(uint64_t wall_usec, PhaseTimers*)`; top-level inclusive phase fields; `sdma_submit_inclusive_usec`, `sdma_submit_exclusive_usec`, `measured_exclusive_total_usec`, and `unattributed_usec`.

- [ ] **Step 1: Write the failing accounting test**

Create `tests/native_r9700/test_prefill_phase_accounting.py` using the compile-probe pattern from `test_pm4_batch_contract.py`. The probe must instantiate this exact public contract:

```cpp
native_r9700::PhaseTimers timers;
timers.sdma_submit_inclusive_usec = 100;
timers.sdma_fence_wait_usec = 80;
timers.model_bind_inclusive_usec = 10;
timers.dispatch_build_inclusive_usec = 20;
timers.device_prepare_inclusive_usec = 30;
timers.embedding_upload_inclusive_usec = 40;
timers.weight_upload_inclusive_usec = 50;
timers.compute_loop_inclusive_usec = 60;
timers.kv_readback_inclusive_usec = 70;
timers.session_close_inclusive_usec = 80;
timers.npz_serialization_inclusive_usec = 90;
native_r9700::finalize_phase_accounting(500, &timers);
```

Assert:

```cpp
if (timers.sdma_submit_exclusive_usec != 20) return 1;
if (timers.measured_exclusive_total_usec != 450) return 2;
if (timers.unattributed_usec != 50) return 3;
```

Add a second case where nested fence time exceeds inclusive submit time and assert the exclusive value saturates at zero rather than underflowing.

- [ ] **Step 2: Run the focused test to prove the interface is absent**

```sh
$PY -m pytest tests/native_r9700/test_prefill_phase_accounting.py -q
```

Expected: compile failure naming `sdma_submit_inclusive_usec` or `finalize_phase_accounting`.

- [ ] **Step 3: Replace ambiguous fields and add the finalizer**

In `amdev_session.h`, replace `sdma_submit_usec` with the following fields and add the top-level intervals:

```cpp
long sdma_submit_inclusive_usec = 0;
long sdma_fence_wait_usec = 0;
long sdma_submit_exclusive_usec = 0;
long model_bind_inclusive_usec = 0;
long dispatch_build_inclusive_usec = 0;
long device_prepare_inclusive_usec = 0;
long embedding_upload_inclusive_usec = 0;
long weight_upload_inclusive_usec = 0;
long compute_loop_inclusive_usec = 0;
long kv_readback_inclusive_usec = 0;
long session_close_inclusive_usec = 0;
long npz_serialization_inclusive_usec = 0;
uint64_t measured_exclusive_total_usec = 0;
uint64_t unattributed_usec = 0;
```

Declare:

```cpp
void finalize_phase_accounting(uint64_t wall_usec, PhaseTimers* timers);
```

Implement it in `amdev_session.cpp`:

```cpp
void finalize_phase_accounting(uint64_t wall_usec, PhaseTimers* timers) {
  if (timers == nullptr) return;
  timers->sdma_submit_exclusive_usec =
      std::max(0L, timers->sdma_submit_inclusive_usec - timers->sdma_fence_wait_usec);
  const uint64_t top_level =
      static_cast<uint64_t>(std::max(0L, timers->model_bind_inclusive_usec)) +
      static_cast<uint64_t>(std::max(0L, timers->dispatch_build_inclusive_usec)) +
      static_cast<uint64_t>(std::max(0L, timers->device_prepare_inclusive_usec)) +
      static_cast<uint64_t>(std::max(0L, timers->embedding_upload_inclusive_usec)) +
      static_cast<uint64_t>(std::max(0L, timers->weight_upload_inclusive_usec)) +
      static_cast<uint64_t>(std::max(0L, timers->compute_loop_inclusive_usec)) +
      static_cast<uint64_t>(std::max(0L, timers->kv_readback_inclusive_usec)) +
      static_cast<uint64_t>(std::max(0L, timers->session_close_inclusive_usec)) +
      static_cast<uint64_t>(std::max(0L, timers->npz_serialization_inclusive_usec));
  timers->measured_exclusive_total_usec = std::min(wall_usec, top_level);
  timers->unattributed_usec = wall_usec - timers->measured_exclusive_total_usec;
}
```

- [ ] **Step 4: Instrument top-level intervals without double counting**

Use `ScopedUsec` or the existing `gettimeofday` pattern around these exact regions in `run_native_prefill()`:

- `build_llama_layer_weight_table` → `model_bind_inclusive_usec`
- `build_llama_persistent_dispatch` → `dispatch_build_inclusive_usec`
- `resident.prepare` → `device_prepare_inclusive_usec`
- all embedding block/row uploads → `embedding_upload_inclusive_usec`
- the complete nine-span upload chain for every layer → `weight_upload_inclusive_usec`
- the layer/token dispatch loops → `compute_loop_inclusive_usec`
- `resident.readback` → `kv_readback_inclusive_usec`
- `resident.close` → `session_close_inclusive_usec`
- `write_native_prefill_npz_atomic` → `npz_serialization_inclusive_usec`

Copy these outer values into the close-time snapshot after `resident.close()`; do not let the session snapshot overwrite them with zeros.

In `runner.cpp`, call:

```cpp
finalize_phase_accounting(result.wall_usec, &result.phase_timers);
```

immediately after setting `wall_usec` and before rendering key/value or JSON output.

- [ ] **Step 5: Update every structured field name**

Remove `sdma_submit_usec` from C++ output and emit these exact keys in key/value and JSON forms:

```text
sdma_submit_inclusive_usec
sdma_fence_wait_usec
sdma_submit_exclusive_usec
model_bind_inclusive_usec
dispatch_build_inclusive_usec
device_prepare_inclusive_usec
embedding_upload_inclusive_usec
weight_upload_inclusive_usec
compute_loop_inclusive_usec
kv_readback_inclusive_usec
session_close_inclusive_usec
npz_serialization_inclusive_usec
measured_exclusive_total_usec
unattributed_usec
```

Update Python evidence normalization only where it names the removed `sdma_submit_usec`; unknown additive fields remain ignored.

- [ ] **Step 6: Run focused and protocol tests**

```sh
$PY -m pytest tests/native_r9700/test_prefill_phase_accounting.py \
  tests/native_r9700/test_runtime_protocol.py \
  tests/native_r9700/test_native_worker_evidence.py -q
```

Expected: all pass.

- [ ] **Step 7: Build and commit**

```sh
git add native_r9700/amdev_session.h native_r9700/amdev_session.cpp \
  native_r9700/runtime_contract.cpp native_r9700/runtime.h native_r9700/runner.cpp \
  tests/native_r9700/test_prefill_phase_accounting.py
git commit -m "perf(native): make prefill phase accounting exclusive"
```

---

## Task 2: Break socket RPC cost down by operation

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:59-73,2247-2426`
- Modify: `native_r9700/amdev_session.h` (`RpcOperationTiming`, `PhaseTimers`)
- Modify: `native_r9700/amdev_session.cpp` (`close` snapshot)
- Modify: `native_r9700/runner.cpp` (structured RPC output)
- Create: `tests/native_r9700/test_rpc_accounting_contract.py`

**Interfaces:**
- Consumes: `RemoteCmd`, `RemoteClient::rpc`, `rpc_sysmem_fd`, and `mmio_write_fire_and_forget`.
- Produces: `constexpr size_t kRpcOperationCount = 14`; `RemoteRpcCounters`; `PhaseTimers::rpc_operations`; aggregate `socket_rpc_count` as the sum of all operation counts.

- [ ] **Step 1: Write the failing counter test**

Compile a probe that includes `native_amdev_transfer_probe.cpp` with `main` renamed. Exercise the counter without a socket:

```cpp
RemoteRpcCounters counters;
counters.record(RemoteCmd::MMIO_READ, 7);
counters.record(RemoteCmd::MMIO_READ, 11);
counters.record(RemoteCmd::SYSMEM_WRITE, 13);
if (counters.count(RemoteCmd::MMIO_READ) != 2) return 1;
if (counters.usec(RemoteCmd::MMIO_READ) != 18) return 2;
if (counters.total_count() != 3) return 3;
```

Also assert `kRpcOperationNames` is exactly:

```cpp
{"probe", "map_bar", "map_sysmem_fd", "cfg_read", "cfg_write", "reset",
 "mmio_read", "mmio_write", "map_sysmem", "sysmem_read", "sysmem_write",
 "resize_bar", "ping", "unknown"}
```

Map every declared `RemoteCmd` to one non-`unknown` index; reserve `unknown` only for out-of-range defensive conversion.

- [ ] **Step 2: Run to verify RED**

```sh
$PY -m pytest tests/native_r9700/test_rpc_accounting_contract.py -q
```

Expected: compile failure naming `RemoteRpcCounters`.

- [ ] **Step 3: Add fixed-size, allocation-free counters**

In the probe, define:

```cpp
constexpr std::size_t kRpcOperationCount = 14;
struct RemoteRpcCounters {
  std::array<uint64_t, kRpcOperationCount> counts{};
  std::array<uint64_t, kRpcOperationCount> usecs{};
  void record(RemoteCmd cmd, uint64_t usec);
  uint64_t count(RemoteCmd cmd) const;
  uint64_t usec(RemoteCmd cmd) const;
  uint64_t total_count() const;
};
```

Use a stack-only scope timer in each RPC entry point:

```cpp
class ScopedRemoteRpcTimer {
 public:
  ScopedRemoteRpcTimer(RemoteRpcCounters* counters, RemoteCmd cmd);
  ~ScopedRemoteRpcTimer();
 private:
  RemoteRpcCounters* counters_;
  RemoteCmd cmd_;
  timeval start_{};
};
```

Create it before the first send in `rpc`, `rpc_sysmem_fd`, and `mmio_write_fire_and_forget`. Increment counts once per issued wire request, including failed requests. Do not alter any frame, payload, read size, return value, or error text.

- [ ] **Step 4: Copy the operation table into the public result**

In `amdev_session.h` add:

```cpp
struct RpcOperationTiming {
  uint64_t count = 0;
  uint64_t usec = 0;
};
std::array<RpcOperationTiming, 14> rpc_operations{};
```

to `PhaseTimers` and include `<array>`. During `ResidentHsaSession::close()`, copy every `RemoteRpcCounters` slot before `reset_after_close()`, then set `socket_rpc_count` to the sum of `count` fields.

In `runner.cpp`, iterate the ordered operation names and emit `rpc_count_` and
`rpc_usec_` keys for every entry. The exact key suffixes are:

```text
probe
map_bar
map_sysmem_fd
cfg_read
cfg_write
reset
mmio_read
mmio_write
map_sysmem
sysmem_read
sysmem_write
resize_bar
ping
unknown
```

Emit all twenty-eight fields in both key/value and JSON output.

- [ ] **Step 5: Run focused tests and the full build**

```sh
$PY -m pytest tests/native_r9700/test_rpc_accounting_contract.py \
  tests/native_r9700/test_runtime_protocol.py \
  tests/native_r9700/test_resident_kernel_dispatch_contract.py -q
```

Expected: focused tests pass. If the known missing-`HardwareLock` closure failure appears in the last file, record it as pre-existing and run the two new/changed contract files separately to prove the patch.

- [ ] **Step 6: Hardware benchmark and attribution decision**

Run three prompt-128 trials into `logs/compute-side-opt/attribution/`. Assert:

- the fourteen operation counts sum to `socket_rpc_count`;
- `measured_exclusive_total_usec <= wall_usec`;
- `unattributed_usec` is nonnegative;
- median wall time is no more than three percent slower than Task 0.

Do not add a successful-submit diagnostic toggle unless `rpc_count_mmio_read` callsite evidence proves the production batch path performs those reads. The current `dispatch_batch()` has no such call.

- [ ] **Step 7: Commit**

```sh
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp \
  native_r9700/amdev_session.h native_r9700/amdev_session.cpp native_r9700/runner.cpp \
  tests/native_r9700/test_rpc_accounting_contract.py
git commit -m "perf(native): attribute TinyGPU RPCs by operation"
```

---

## Task 3: Add pure PM4 GPU-clock and completion encoders

**Files:**
- Modify: `native_r9700/amdev_packets.h:15-40`
- Modify: `native_r9700/amdev_packets.cpp:94-182`
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:303-345` (source-ground constants only)
- Create: `tests/native_r9700/test_gpu_timestamp_pm4_contract.py`

**Interfaces:**
- Consumes: gfx12 `PACKET3_RELEASE_MEM`, `PACKET3_ACQUIRE_MEM`, `CS_PARTIAL_FLUSH`, and tinygrad’s GPU-clock selector value 3.
- Produces: `Pm4StageTail`; `build_pm4_dispatch_words(const Pm4DispatchConfig&, const Pm4StageTail&)`; `build_pm4_gpu_timestamp_words(uint64_t)`; `build_pm4_timeline_signal_words(uint64_t, uint32_t)`; default `build_pm4_dispatch_words(config)` remains exactly 59 dwords.

- [ ] **Step 1: Write the failing PM4 encoding tests**

The compile probe must assert:

```cpp
const native_r9700::Pm4DispatchConfig config{
    0x100000ULL, 0x200000ULL, 0x300000ULL,
    0xc0040000U, 0x84U, 0U, false,
    64U, 1U, 1U, 1U, 1U, 1U, 7U};
const auto frozen = native_r9700::build_pm4_dispatch_words(config);
if (frozen.size() != 59) return 1;

const auto stamp = native_r9700::build_pm4_gpu_timestamp_words(0x123456780ULL);
if (stamp.empty()) return 2;

const auto terminal =
    native_r9700::build_pm4_timeline_signal_words(0x300000ULL, 9U);
if (terminal.empty()) return 3;
```

Decode the timestamp stream and assert:

- one `RELEASE_MEM` uses `data_sel == 3`;
- its destination is `0x123456780`;
- it is followed by `ACQUIRE_MEM` as in local tinygrad `ops_amd.py:372-376`;
- no timestamp packet writes a host timeline value;
- the default 59-dword stream is byte-identical to the current frozen stream.

Add tail variants:

```cpp
struct Pm4StageTail {
  bool emit_cs_partial_flush = true;
  bool emit_cache_release = true;
  bool write_timeline = true;
};
```

Assert `{true,true,false}` retains `CS_PARTIAL_FLUSH` and cache `RELEASE_MEM` but emits `data_sel=none`, while `{false,true,false}` omits only `CS_PARTIAL_FLUSH`.

- [ ] **Step 2: Run to verify RED**

```sh
$PY -m pytest tests/native_r9700/test_gpu_timestamp_pm4_contract.py -q
```

Expected: compile failure naming `Pm4StageTail`.

- [ ] **Step 3: Refactor the packet builder without changing the default bytes**

Extract the existing packet construction through `DISPATCH_DIRECT` into a private helper, then append:

```cpp
if (tail.emit_cs_partial_flush) {
  append_pm4_packet3(&words, kPacket3EventWrite,
                     {encode_event_write_cs_partial_flush()});
}
if (tail.emit_cache_release) {
  append_pm4_packet3(&words, kPacket3ReleaseMem,
                     {encode_release_mem_event(),
                      tail.write_timeline ? encode_release_mem_data_sel()
                                          : encode_release_mem_data_sel_none(),
                      tail.write_timeline ? lo32_impl(config.timeline_va) : 0U,
                      tail.write_timeline ? hi32_impl(config.timeline_va) : 0U,
                      tail.write_timeline ? config.timeline_value : 0U,
                      0U, 0U});
}
```

Keep the existing one-argument overload as:

```cpp
return build_pm4_dispatch_words(config, Pm4StageTail{});
```

so all frozen callers remain unchanged.

Implement `build_pm4_gpu_timestamp_words()` with the tinygrad sequence: ordering release with no data, clock-counter release (`data_sel=3`, interrupt none), then acquire. Implement `build_pm4_timeline_signal_words()` as one cache-flushing `RELEASE_MEM` that writes the supplied 32-bit value to the supplied VA. Use the generated gfx12 selector value from local `pm4_soc15.py`; do not derive it from the PDF text alone.

- [ ] **Step 4: Run PM4 contracts**

```sh
$PY -m pytest tests/native_r9700/test_gpu_timestamp_pm4_contract.py \
  tests/native_r9700/test_pm4_batch_contract.py \
  tests/native_r9700/test_pm4_timeline_contract.py -q
```

Expected: all pass; existing 59- and 118-dword controls are unchanged.

- [ ] **Step 5: Commit**

```sh
git add native_r9700/amdev_packets.h native_r9700/amdev_packets.cpp \
  experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp \
  tests/native_r9700/test_gpu_timestamp_pm4_contract.py
git commit -m "feat(native): encode gfx12 GPU stage timestamps"
```

---

## Task 4: Integrate optional per-stage GPU profiling

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:315-342` (control-page offsets)
- Modify: `native_r9700/amdev_session.h:109-180`
- Modify: `native_r9700/amdev_session.cpp:2429-2574`
- Modify: `native_r9700/runtime.h:6243-6270`
- Modify: `native_r9700/runtime_contract.cpp:804-857`
- Modify: `native_r9700/runner.cpp:40-50,157-225,592-625`
- Create: `tests/native_r9700/test_gpu_stage_profile_contract.py`

**Interfaces:**
- Consumes: Task 3 packet helpers and the proven compute-control page at `kRptrVa`.
- Produces: `ResidentHsaBatchOptions`; `GpuStageTickSample`; `NativePrefillRequest::gpu_stage_profile`; `NativePrefillResult::gpu_stage_profile`; runner flag `--gpu-stage-profile`.

- [ ] **Step 1: Write the failing layout and summary tests**

Assert the reviewed fixed offsets:

```cpp
constexpr uint64_t kGpuTimestampCpuOffset = 0x100ULL;
constexpr uint64_t kGpuTimestampVa = kRptrVa + kGpuTimestampCpuOffset;
constexpr uint32_t kGpuTimestampBoundaryCount = 11U;
constexpr uint64_t kGpuTimestampByteCount = 11ULL * sizeof(uint64_t);
```

The test must prove:

- `[0x100, 0x158)` fits page 0 of `compute_control`;
- it does not overlap RPTR `[0,8)`, WPTR `[8,16)`, or timeline `[16,20)`;
- a sample `{10,20,35,50,70,95,125,160,200,245,295}` produces the ten differences `{10,15,15,20,25,30,35,40,45,50}`;
- equal, decreasing, or zero boundaries fail validation with `gpu timestamp boundaries are not strictly increasing`.

Use this public result shape:

```cpp
struct GpuStageTickSample {
  std::array<uint64_t, 11> boundaries{};
};
struct ResidentHsaBatchOptions {
  bool capture_gpu_timestamps = false;
};
```

Add this member to `ResidentHsaDispatchResult`; it remains empty when profiling
is disabled:

```cpp
std::vector<GpuStageTickSample> gpu_stage_tick_samples;
```

- [ ] **Step 2: Run to verify RED**

```sh
$PY -m pytest tests/native_r9700/test_gpu_stage_profile_contract.py -q
```

Expected: compile failure naming `GpuStageTickSample`.

- [ ] **Step 3: Build a profiled batch without changing the disabled path**

Extend `dispatch_batch` with a trailing defaulted options argument:

```cpp
bool dispatch_batch(const std::vector<ResidentHsaStage>& stages,
                    ResidentHsaDispatchResult* result,
                    std::string* error_text,
                    const ResidentHsaBatchOptions& options = {});
```

When profiling is disabled, execute the current code path byte-for-byte.

When enabled:

1. zero exactly 88 bytes at CPU offset `0x100`;
2. prepend T0 using `build_pm4_gpu_timestamp_words(kGpuTimestampVa)`;
3. build each stage with cache completion and no host timeline write;
4. append Ti after each stage at `kGpuTimestampVa + i*8`;
5. after T10, append one terminal timeline `RELEASE_MEM` with the batch’s expected monotonic value;
6. ring once and poll once;
7. copy eleven local mapped `uint64_t` values after the poll;
8. validate strict monotonicity and append the sample to `result->gpu_stage_tick_samples`.

In profiled/terminal-only mode, increment `state.next_timeline_value` exactly
once per emitted host timeline signal, not once per stage. Pass that one value
to both `build_pm4_timeline_signal_words()` and the terminal poll. The frozen
per-stage policy retains one increment per stage.

No host wait, socket RPC, PTE, or allocation occurs between timestamps.

- [ ] **Step 4: Add result summarization and structured output**

Add to `NativePrefillRequest`:

```cpp
bool gpu_stage_profile = false;
```

Add to `NativePrefillResult`:

```cpp
std::array<uint64_t, 10> gpu_stage_tick_total{};
std::array<uint64_t, 10> gpu_stage_tick_min{};
std::array<uint64_t, 10> gpu_stage_tick_max{};
uint64_t gpu_stage_profile_sample_count = 0;
```

Aggregate each sample after the compute loop. Emit stage names in this exact order:

```text
rmsnorm
k_projection
v_projection
rope_kv
attention_score
attention_softmax
attention_context
o_projection
gate_up_projection
mlp_down
```

Emit raw ticks only: total, min, mean, max, sample count, and share of summed stage ticks. Do not emit microseconds or GB/s until the GPU clock unit is source-grounded.

- [ ] **Step 5: Extend runner parsing without breaking the native worker command**

Accept the existing ten arguments plus an optional final `--gpu-stage-profile`. Update help text. Existing command lines from `native_worker.py` remain valid and default to profiling off.

Add protocol tests proving:

- the old ten-argument command still parses;
- `--gpu-stage-profile` sets the request field;
- token IDs remain redacted from stdout/logs;
- the disabled path emits `gpu_stage_profile_sample_count: 0`.

- [ ] **Step 6: Run hardware-free tests and build**

```sh
$PY -m pytest tests/native_r9700/test_gpu_stage_profile_contract.py \
  tests/native_r9700/test_gpu_timestamp_pm4_contract.py \
  tests/native_r9700/test_runtime_protocol.py -q
```

Expected: all pass; full build exits 0.

- [ ] **Step 7: Hardware profile ladder**

Run health gates, then profile prompt lengths 1, 64, and 128:

```sh
APL_REMOTE_SOCK=$SOCK $RUNNER --native-prefill-proof \
  --model "$MODEL" --token-ids-json "$TOKENS128" \
  --out logs/compute-side-opt/gpu-profile/prompt128.npz \
  --log logs/compute-side-opt/gpu-profile/prompt128.log \
  --gpu-stage-profile
```

Expected:

- `exit_status: 0` and token-exact downstream acceptance;
- exactly `2048` profile samples for prompt-128;
- eleven strictly increasing boundaries per sample;
- one ranked ten-stage report.

Record the top stage and its share. Do not assert `timeline_wait_usec` is pure arithmetic.

- [ ] **Step 8: Commit**

```sh
git add experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp \
  native_r9700/amdev_session.h native_r9700/amdev_session.cpp \
  native_r9700/runtime.h native_r9700/runtime_contract.cpp native_r9700/runner.cpp \
  tests/native_r9700/test_gpu_stage_profile_contract.py \
  tests/native_r9700/test_runtime_protocol.py
git commit -m "feat(native): profile per-stage GPU clock ticks"
```

---

## Task 5: Remove intermediate host signals and test K/V overlap

**Files:**
- Modify: `native_r9700/amdev_session.h` (`ComputeCompletionPolicy`, `ComputeBarrierPolicy`)
- Modify: `native_r9700/amdev_session.cpp:2429-2574`
- Modify: `native_r9700/runtime.h` (`NativePrefillRequest` diagnostic policies)
- Modify: `native_r9700/runner.cpp` (diagnostic flags)
- Create: `tests/native_r9700/test_compute_barrier_policy.py`

**Interfaces:**
- Consumes: Task 3 `Pm4StageTail` and Task 4 GPU-stage profile.
- Produces: `ComputeCompletionPolicy::{PerStageTimeline,TerminalTimeline}` and `ComputeBarrierPolicy::{Full,OverlapKvProjections}`; default is promoted only after hardware evidence.

- [ ] **Step 1: Write the failing policy test**

Build a ten-stage PM4 batch under each policy and assert:

```text
PerStageTimeline + Full:
  10 CS_PARTIAL_FLUSH events
  10 timeline data writes

TerminalTimeline + Full:
  10 CS_PARTIAL_FLUSH events
  1 timeline data write (stage 9 only)
  9 cache-completion RELEASE_MEM packets with data_sel=none

TerminalTimeline + OverlapKvProjections:
  9 CS_PARTIAL_FLUSH events
  1 timeline data write
  the only omitted partial flush is after stage index 1 (K projection)
  the flush before stage index 3 (RoPE join) remains
```

Also assert the stage order is unchanged.

- [ ] **Step 2: Run to verify RED**

```sh
$PY -m pytest tests/native_r9700/test_compute_barrier_policy.py -q
```

Expected: compile failure naming `ComputeCompletionPolicy`.

- [ ] **Step 3: Add explicit batch policies**

Define:

```cpp
enum class ComputeCompletionPolicy {
  PerStageTimeline,
  TerminalTimeline,
};
enum class ComputeBarrierPolicy {
  Full,
  OverlapKvProjections,
};
```

For `TerminalTimeline`, set `write_timeline=false` for stages 0–8 and true for stage 9. Preserve cache completion on every stage.

For `OverlapKvProjections`, set `emit_cs_partial_flush=false` only for stage index 1. Do not alter ACQUIRE_MEM, stage 2 completion, or the stage-3 RoPE join barrier.

Expose runner-only A/B flags:

```text
--completion-policy per-stage|terminal
--barrier-policy full|overlap-kv
```

The Python native worker does not set these flags during the experiment.

- [ ] **Step 4: Run contract tests and build**

```sh
$PY -m pytest tests/native_r9700/test_compute_barrier_policy.py \
  tests/native_r9700/test_gpu_timestamp_pm4_contract.py \
  tests/native_r9700/test_pm4_batch_contract.py -q
```

Expected: all pass.

- [ ] **Step 5: Hardware A/B terminal-only completion**

Run three prompt-128 trials for `per-stage/full` and three for `terminal/full`, with GPU profiling enabled. Require:

- exact NPZ acceptance and no new faults;
- identical `kernel_count=20480` and `compute_submit_count=2048`;
- terminal mode has one timeline data write per batch by contract;
- terminal median does not regress more than three percent.

If terminal mode passes, make `TerminalTimeline` the production default and retain `PerStageTimeline` only as an explicit diagnostic control. If it fails any correctness gate, revert terminal-mode code and stop Task 5 before testing overlap.

- [ ] **Step 6: Hardware A/B K/V overlap**

Run three terminal/full and three terminal/overlap-kv prompt-128 trials. Promote overlap only when all correctness gates pass **and** median wall time improves by at least three percent. If improvement is below three percent, remove the overlap production path; preserve the measured result in ignored evidence and keep `Full` as default.

- [ ] **Step 7: Commit the selected policy**

Stage only the policy that passed the rules above plus its contract tests:

```sh
git add native_r9700/amdev_session.h native_r9700/amdev_session.cpp \
  native_r9700/runtime.h native_r9700/runner.cpp \
  tests/native_r9700/test_compute_barrier_policy.py
git commit -m "perf(native): use measured terminal compute completion"
```

If overlap is promoted, use a second commit:

```sh
git commit -m "perf(native): overlap independent K and V projections"
```

---

## Task 6: Add the token-block model and cut the runtime over at block size 1

**Files:**
- Modify: `native_r9700/llama_layer_executor.h:87-120`
- Modify: `native_r9700/llama_layer_executor.cpp:317-362,433-592`
- Modify: `native_r9700/runtime_contract.cpp:741-857`
- Modify: `tests/native_r9700/test_layer0_executor_contract.py`
- Create: `tests/native_r9700/test_llama_token_block_contract.py`

**Interfaces:**
- Consumes: existing sequence-aware kernel ABIs and ten-stage `layer_stages`.
- Produces: `LlamaTokenBlock`; block-capacity `build_llama_persistent_dispatch`; `set_llama_block_stage_state`; a clean block-size-1 runtime cutover with no retained token-only API.

- [ ] **Step 1: Write failing block partition and extent tests**

Use this exact public struct:

```cpp
struct LlamaTokenBlock {
  uint32_t hidden_buffer_index = 0;
  uint32_t position = 0;
  uint32_t token_count = 0;
};
```

Change the builder signature to:

```cpp
bool build_llama_persistent_dispatch(const LlamaLayerWeightTable& weights,
                                     uint32_t token_count,
                                     uint32_t block_capacity,
                                     LlamaPersistentDispatch* dispatch,
                                     std::string* error_text);
```

The probe must assert:

- `(token_count=1, block_capacity=1)` → one block `{position=0,count=1}`;
- `(17,8)` → blocks `{0,8}`, `{8,8}`, `{16,1}`;
- block capacity 0 and 129 fail before allocation;
- every block range stays within `token_count` and cache capacity 128;
- hidden allocation bytes are `block_capacity * 2048 * sizeof(uint16_t)`;
- score/probability scratch bytes are `block_capacity * 32 * 128 * sizeof(float)`;
- gate/up scratch bytes are `block_capacity * 8192 * sizeof(uint16_t)`.

- [ ] **Step 2: Write failing geometry tests**

Replace the token mutator with:

```cpp
bool set_llama_block_stage_state(
    std::vector<ResidentHsaStage>* stages,
    const std::vector<std::pair<uint32_t, uint32_t>>& hidden_binding_slots,
    const LlamaTokenBlock& block,
    std::string* error_text);
```

For `{position=16,count=8}`, assert exact workgroup counts:

```text
stage 0 rmsnorm             8
stage 1 k projection       64
stage 2 v projection       64
stage 3 rope kv            64
stage 4 attention score   256
stage 5 softmax           256
stage 6 context           256
stage 7 o projection      256
stage 8 gate/up          1024
stage 9 down              256
```

Assert every sequence-length scalar is 8, every position scalar is 16, and cache capacity is 128. The tail block `{position=16,count=1}` must restore the current single-token geometry.

- [ ] **Step 3: Run to verify RED**

```sh
$PY -m pytest tests/native_r9700/test_llama_token_block_contract.py -q
```

Expected: compile failure naming `LlamaTokenBlock`.

- [ ] **Step 4: Implement block partitioning and block-sized scratch**

In `LlamaPersistentDispatch`, replace `hidden_buffers` with:

```cpp
uint32_t block_capacity = 1;
std::vector<LlamaTokenBlock> token_blocks;
```

For each block, allocate one hidden buffer named with
`"llama.hidden.block" + std::to_string(block_index)` and
`block_capacity * 4096` bytes. Allocate one shared scratch set sized by block
capacity using the exact formulas from Step 1. Keep one K and one V cache per
layer at the existing 128-token capacity.

Do not add buffer-offset semantics to `ResidentHsaKernargBinding`; each block owns its own contiguous allocation and binds at base VA.

- [ ] **Step 5: Implement scalar and geometry mutation**

`set_llama_block_stage_state()` must:

1. validate `token_count in [1,block_capacity]` and `position+token_count <= 128`;
2. retarget the three hidden binding slots to the block buffer;
3. write sequence length at these exact `(stage, byte offset)` pairs:
   `(1,24)`, `(2,24)`, `(3,32)`, `(4,32)`, `(5,16)`, `(6,24)`,
   `(7,32)`, `(8,48)`, `(9,40)`;
4. write `(position,cache_capacity)` at `(3,36,40)`, `(4,36,40)`,
   `(5,20,24)`, and `(6,28,32)`;
5. multiply each one-token `global_x` by `block.token_count` using the table
   from Step 2.

Stage 0 has no sequence scalar; its row count is carried only by `global_x`.

- [ ] **Step 6: Cut the runtime over at block size 1**

Update the builder call immediately so this task compiles independently:

```cpp
if (!build_llama_persistent_dispatch(
        weight_table, static_cast<uint32_t>(request.token_ids.size()), 1U,
        &persistent_dispatch, &detail)) {
  fail(result, "persistent_dispatch_build", detail, error_text);
  return 1;
}
```

Use these loops so the clean cutover is executable in the same commit:

```cpp
for (const LlamaTokenBlock& block : persistent_dispatch.token_blocks) {
  Fp16WeightSpan row;
  if (!select_llama_embedding_row(weight_table.embed_tokens,
                                  request.token_ids[block.position],
                                  &row, &detail) ||
      !upload_span("block_position=" + std::to_string(block.position) +
                       " embedding_row",
                   block.hidden_buffer_index, row)) {
    std::string close_error;
    resident.close(&close_error);
    fail(result, "resident_embedding_upload", detail, error_text);
    return 1;
  }
}
for (uint32_t layer = 0; layer < persistent_dispatch.layer_stages.size(); ++layer) {
  for (const LlamaTokenBlock& block : persistent_dispatch.token_blocks) {
    if (!set_llama_block_stage_state(&persistent_dispatch.layer_stages[layer],
                                     persistent_dispatch.hidden_binding_slots,
                                     block, &detail) ||
        !resident.dispatch_batch(persistent_dispatch.layer_stages[layer],
                                 &dispatch_result, &detail, batch_options)) {
      std::string close_error;
      resident.close(&close_error);
      fail(result, "resident_dispatch_batch", detail, error_text);
      return 1;
    }
  }
}

```
Leave the nine layer-weight uploads at the start of each layer iteration,
before the inner block loop.

With capacity 1, this must retain 128 blocks, 2,048 submissions, and 20,480
kernels for prompt-128.

- [ ] **Step 7: Run executor and block-size-1 runtime contracts**

```sh
$PY -m pytest tests/native_r9700/test_llama_token_block_contract.py \
  tests/native_r9700/test_layer0_executor_contract.py \
  tests/native_r9700/test_llama_stage_layout.py \
  tests/native_r9700/test_runtime_protocol.py -q
```

Expected: all pass and the full build exits 0.

- [ ] **Step 8: Commit**

```sh
git add native_r9700/llama_layer_executor.h native_r9700/llama_layer_executor.cpp \
  native_r9700/runtime_contract.cpp \
  tests/native_r9700/test_layer0_executor_contract.py \
  tests/native_r9700/test_llama_token_block_contract.py
git commit -m "feat(native): cut Llama runtime over to token blocks"
```

---

## Task 7: Make token-block capacity configurable and upload multi-row blocks

**Files:**
- Modify: `native_r9700/runtime.h:6243-6270`
- Modify: `native_r9700/runtime_contract.cpp:719-857`
- Modify: `native_r9700/runner.cpp:40-50,592-625`
- Modify: `native_r9700/native_worker.py:15-16,133-153`
- Modify: `tests/native_r9700/test_runtime_protocol.py`
- Modify: `tests/native_r9700/test_native_worker_evidence.py`
- Create: `tests/native_r9700/test_block_prefill_runtime_contract.py`

**Interfaces:**
- Consumes: Task 6 block-size-1 runtime cutover.
- Produces: `NativePrefillRequest::block_tokens`; runner `--block-tokens`; diagnostic environment `NATIVE_R9700_PREFILL_BLOCK_TOKENS`; contiguous multi-row block upload; structured `block_tokens`/`block_count` output.

- [ ] **Step 1: Write failing CLI and worker tests**

Assert the legacy command defaults to block size 1. Assert this complete command:

```sh
native_r9700_runner --native-prefill-proof \
  --model synthetic-model --token-ids-json '[1,2]' \
  --out /tmp/block-prefill.npz --log /tmp/block-prefill.log \
  --block-tokens 8
```

sets `request.block_tokens == 8`, while values `0`, `3`, `129`, negative text,
or missing values fail before hardware work. Allowed diagnostic values are
exactly `1,2,4,8,16,32`.

In `native_worker.py`, when:

```python
NATIVE_R9700_PREFILL_BLOCK_TOKENS=8
```

is present, `_build_runner_command()` appends `['--block-tokens', '8']`. Invalid environment values raise the existing fail-closed worker error before `subprocess.run`.

- [ ] **Step 2: Run to verify RED**

```sh
$PY -m pytest tests/native_r9700/test_block_prefill_runtime_contract.py \
  tests/native_r9700/test_native_worker_evidence.py -q
```

Expected: failures naming `block_tokens` or absent command arguments.

- [ ] **Step 3: Add request/result fields and flexible runner parsing**

Add:

```cpp
uint32_t block_tokens = 1;
```

to `NativePrefillRequest`, and:

```cpp
uint32_t block_tokens = 1;
uint32_t block_count = 0;
```

to `NativePrefillResult`.

Replace `argc == 10` parsing with a strict parser that consumes the existing required option pairs and permits only the optional flags introduced by Tasks 4, 5, and 7. Duplicate flags and unknown flags fail with exit status 2.

- [ ] **Step 4: Pass requested capacity and upload contiguous multi-row blocks**

Replace the Task 6 literal capacity with:

```cpp
if (!build_llama_persistent_dispatch(
        weight_table, static_cast<uint32_t>(request.token_ids.size()),
        request.block_tokens, &persistent_dispatch, &detail)) {
  fail(result, "persistent_dispatch_build", detail, error_text);
  return 1;
}
```

Replace the one-row block upload with this multi-row helper:

```cpp
auto upload_embedding_block = [&](const LlamaTokenBlock& block) -> bool {
  std::vector<uint8_t> embedding_bytes(
      static_cast<size_t>(block.token_count) * kLlamaEmbeddingRowBytes);
  for (uint32_t offset = 0; offset < block.token_count; ++offset) {
    Fp16WeightSpan row;
    if (!select_llama_embedding_row(weight_table.embed_tokens,
                                    request.token_ids[block.position + offset],
                                    &row, &detail)) {
      return false;
    }
    std::ifstream source(row.shard_path, std::ios::binary);
    source.seekg(static_cast<std::streamoff>(row.data_offset));
    source.read(
        reinterpret_cast<char*>(embedding_bytes.data() +
                                static_cast<size_t>(offset) *
                                    kLlamaEmbeddingRowBytes),
        static_cast<std::streamsize>(kLlamaEmbeddingRowBytes));
    if (source.gcount() != static_cast<std::streamsize>(kLlamaEmbeddingRowBytes)) {
      detail = "embedding block source_read_failed";
      return false;
    }
  }
  const std::string& block_name =
      persistent_dispatch.request.buffers[block.hidden_buffer_index].name;
  return resident.upload_named(block_name, embedding_bytes.data(),
                               embedding_bytes.size(), &dispatch_result, &detail);
};
```

On helper failure, execute the existing fail-closed session close and return 1
with `failure_stage=resident_embedding_upload`. The tail upload length is
`block.token_count * 4096`; unused rows in the capacity-sized allocation remain
zero. Keep the Task 6 causal block dispatch loop unchanged. Set:

```cpp
result->block_tokens = request.block_tokens;
result->block_count =
    static_cast<uint32_t>(persistent_dispatch.token_blocks.size());
```


- [ ] **Step 5: Run software contracts**

```sh
$PY -m pytest tests/native_r9700/test_block_prefill_runtime_contract.py \
  tests/native_r9700/test_llama_token_block_contract.py \
  tests/native_r9700/test_runtime_protocol.py \
  tests/native_r9700/test_native_worker_evidence.py -q
```

Expected: all pass.

- [ ] **Step 6: Hardware block-size-1 control**

Build, run health gates, and run prompt-128 with `--block-tokens 1`. Require:

- token-exact C1R/C2R outputs;
- `block_count=128`;
- `kernel_count=20480`;
- `compute_submit_count=2048`;
- same selected completion/barrier policy as Task 5;
- median within three percent of the post-Task-5 block-size-1 control.

If Task 5 did not promote a PM4 policy, the 590-dword per-batch stream must match Task 0 byte-for-byte.

- [ ] **Step 7: Commit**

```sh
git add native_r9700/runtime.h native_r9700/runtime_contract.cpp native_r9700/runner.cpp \
  native_r9700/native_worker.py tests/native_r9700/test_runtime_protocol.py \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_block_prefill_runtime_contract.py
git commit -m "perf(native): configure Llama token block capacity"
```

---

## Task 8: Run the block-size ladder and promote the fastest stable size

**Files:**
- Modify after measurement: `native_r9700/runtime.h` (default block constant)
- Modify after measurement: `native_r9700/runner.cpp` (help/default output)
- Modify: `tests/native_r9700/test_block_prefill_runtime_contract.py`
- Evidence only: `logs/compute-side-opt/blocks/`

**Interfaces:**
- Consumes: Task 7 diagnostic block override and all C1R/C2R gates.
- Produces: one production default selected by a deterministic benchmark rule; full ladder evidence.

- [ ] **Step 1: Run one-layer/block boundary proof at size 2**

Use the existing stage/layer diagnostic machinery to compare block-size-2 hidden and K/V boundaries against two block-size-1 positions. Require finite outputs and bounded fp16 error no worse than the existing serial-native-vs-CPU layer tolerance. Stop on the first failing layer or stage.

- [ ] **Step 2: Run full native proof for each block size**

For `B in 2 4 8 16 32`, set:

```sh
export NATIVE_R9700_PREFILL_BLOCK_TOKENS=$B
```

Run prompt lengths 16, 64, and 128. Expected structural counts for prompt-128:

| B | Blocks | Submissions | Kernels |
|---:|---:|---:|---:|
| 1 | 128 | 2048 | 20480 |
| 2 | 64 | 1024 | 10240 |
| 4 | 32 | 512 | 5120 |
| 8 | 16 | 256 | 2560 |
| 16 | 8 | 128 | 1280 |
| 32 | 4 | 64 | 640 |

A size advances only when the prior size passes.

- [ ] **Step 3: Run C1R for every advancing size**

```sh
$PY -m native_r9700.parity \
  --model "$MODEL" --fixtures-dir tests/native_r9700/fixtures \
  --r-source both --max-new-tokens 4 \
  --artifacts-dir "logs/compute-side-opt/blocks/$B/c1r" \
  --json "logs/compute-side-opt/blocks/$B/c1r.json" \
  --log "logs/compute-side-opt/blocks/$B/c1r.log" \
  --report /tmp/native-r9700-compute-opt-path-c.md \
  --producer-kind r9700_native
```

Require exact tokens:

```text
prompt-0   [12366, 13, 578, 469]
prompt-16  [11, 706, 28995, 12207]
prompt-64  [279, 4216, 62520, 9478]
prompt-128 [13, 578, 30791, 17604]
```

- [ ] **Step 4: Run C2R for prompt 16 and 128**

```sh
$PY -m native_r9700.serving \
  --model "$MODEL" --fixtures-dir tests/native_r9700/fixtures \
  --threshold-tokens 1 --max-new-tokens 4 \
  --artifacts-dir "logs/compute-side-opt/blocks/$B/c2r" \
  --json "logs/compute-side-opt/blocks/$B/c2r.json" \
  --log "logs/compute-side-opt/blocks/$B/c2r.log"
```

Filter/inspect prompt 16 and 128 results. Require `route=native_producer`, `accepted_cache=true`, and `fallback_reason=null`/`none`.

- [ ] **Step 5: Benchmark and select the production size**

For every size that passes Steps 2–4, run three prompt-128 direct trials
without GPU profiling and write the measured medians to:

```json
{
  "1": 0,
  "2": 0,
  "4": 0,
  "8": 0,
  "16": 0,
  "32": 0
}
```

in `logs/compute-side-opt/blocks/block-ladder-summary.json`, replacing each
zero only for a size that passed every gate. Select the nonzero entry with the
lowest median wall time. A candidate must beat block size 1 by at least three
percent; otherwise select 1. If medians are within one percent, select the
smaller block.

- [ ] **Step 6: Promote the measured default**

Print and validate the selected integer:

```sh
SELECTED=$($PY -c "import json; d=json.load(open('logs/compute-side-opt/blocks/block-ladder-summary.json')); ok={int(k):v for k,v in d.items() if v>0}; base=ok[1]; best=min(ok, key=lambda k:(ok[k],k)); best=1 if ok[best] > base*0.97 else best; tied=[k for k,v in ok.items() if abs(v-ok[best]) <= ok[best]*0.01]; print(min(tied))")
case "$SELECTED" in 1|2|4|8|16|32) ;; *) exit 1 ;; esac
printf '%s\n' "$SELECTED"
```

Set `kDefaultLlamaPrefillBlockTokens` in `runtime.h` to that printed decimal,
replace the literal request default and runner help default with the constant,
and assert the same decimal in the runtime contract test. Do not commit a
constant that differs from this command's output.

- [ ] **Step 7: Stability gate**

Run ten prompt-128 trials at the selected default without TinyGPU restart. Every run must report `exit_status: 0`, native acceptance, exact decode tokens, and no new GPU fault. Any failure rejects promotion and selects the next-fastest size that already passed Steps 2–5.

- [ ] **Step 8: Commit the production default**

```sh
git add native_r9700/runtime.h native_r9700/runner.cpp \
  tests/native_r9700/test_block_prefill_runtime_contract.py
git commit -m "perf(native): promote measured Llama prefill block size"
```

---

## Task 9: Precompute Q and RoPE once inside the attention-score stage

**Gate:** Execute the kernel change only when the post-block GPU profile ranks `attention_score` first and assigns it at least 20 percent of summed stage ticks. If it does not meet both conditions, record the profile and skip the kernel mutation; speculative tuning is prohibited.

**Files:**
- Modify when gate passes: `native_r9700/kernels/llama_causal_attention_score_f16.cpp`
- Regenerate when gate passes: `native_r9700/kernels/llama-attention-score-hsa-assets/llama_causal_attention_score_f16.image`
- Regenerate when gate passes: `native_r9700/kernels/llama-attention-score-hsa-assets/llama_causal_attention_score_f16.json`
- Modify when gate passes: `native_r9700/kernel_assets.cpp` (digest/resource metadata)
- Modify: `tests/native_r9700/test_llama_attention_hsa_assets.py`
- Create: `tests/native_r9700/test_llama_attention_score_precompute.py`

**Interfaces:**
- Consumes: unchanged 48-byte attention-score kernarg ABI and stage-4 geometry `32 * sequence_length` workgroups of 64 lanes.
- Produces: the same attention-score output layout with one Q projection and RoPE computation per `(query_head, query_token)` workgroup.

- [ ] **Step 1: Re-profile the selected production block size**

Run prompt-128 with GPU profiling at the selected default. Save the stage ranking under `logs/compute-side-opt/score-precompute/before.log`. Evaluate the gate exactly as written above.

If the gate is false, create no kernel/source/asset change and continue to Task 10.

- [ ] **Step 2: Write failing source and manifest contracts**

The new test must require:

```text
__attribute__((shared)) float rotated_q[64]
__builtin_amdgcn_s_barrier()
lane < 32U
```

and assert the 2048-column Q projection loop appears once in source, outside the key-block loop. It must also require unchanged function parameters and unchanged manifest kernarg schema name/size:

```text
llama-causal-attention-score-f16-v1
48 bytes
```

Run the existing manifest integrity test against the current asset; the new source contract must fail before implementation.

- [ ] **Step 3: Implement cooperative Q/RoPE precompute**

Restructure the kernel workgroup as follows:

```cpp
__attribute__((shared)) float rotated_q[64];
if (lane < 32U) {
  const unsigned int q_row0 = query_head * 64U + lane;
  const unsigned int q_row1 = q_row0 + 32U;
  float q0 = 0.0f;
  float q1 = 0.0f;
  for (unsigned int column = 0U; column < 2048U; ++column) {
    const float activation = (float)__builtin_bit_cast(
        _Float16, normalized[(unsigned long long)query_token * 2048ULL + column]);
    q0 += activation * (float)__builtin_bit_cast(
        _Float16, q_projection_weight[(unsigned long long)q_row0 * 2048ULL + column]);
    q1 += activation * (float)__builtin_bit_cast(
        _Float16, q_projection_weight[(unsigned long long)q_row1 * 2048ULL + column]);
  }
  float inv_frequency = __builtin_powf(
      kRopeTheta, -2.0f * (float)lane / (float)kHeadDimension);
  const float wavelength = 6.2831853071795864769f / inv_frequency;
  if (wavelength > kOriginalContext / kLowFrequencyFactor) {
    inv_frequency /= kRopeFactor;
  } else if (wavelength >= kOriginalContext / kHighFrequencyFactor) {
    const float smooth =
        (kOriginalContext / wavelength - kLowFrequencyFactor) /
        (kHighFrequencyFactor - kLowFrequencyFactor);
    inv_frequency = (1.0f - smooth) * inv_frequency / kRopeFactor +
                    smooth * inv_frequency;
  }
  const float angle = (float)absolute_query * inv_frequency;
  const float cosine = __builtin_cosf(angle);
  const float sine = __builtin_sinf(angle);
  rotated_q[lane] = q0 * cosine - q1 * sine;
  rotated_q[lane + 32U] = q1 * cosine + q0 * sine;
}
__builtin_amdgcn_s_barrier();
```

Then each lane loops key blocks and computes:

```cpp
float score = 0.0f;
for (unsigned int dimension = 0; dimension < 64U; ++dimension) {
  const float key = (float)__builtin_bit_cast(
      _Float16,
      k_cache[((unsigned long long)kv_head * cache_capacity_tokens + key_token) *
                  kHeadDimension +
              dimension]);
  score += rotated_q[dimension] * key;
}
```

All 64 lanes must reach the barrier. Keep masking, causal bounds, scale `0.125f`, cache layout, and output offsets unchanged. This remains one stage and uses no new kernarg slot.

- [ ] **Step 4: Regenerate the gfx1201 asset**

```sh
SCHEMA=$($PY -c "import json; p='native_r9700/kernels/llama-attention-score-hsa-assets/llama_causal_attention_score_f16.json'; print(json.dumps(json.load(open(p))['kernarg_schema'], separators=(',',':')))")
$PY experiments/native-r9700-runtime/generate_hsa_code_image.py \
  --source native_r9700/kernels/llama_causal_attention_score_f16.cpp \
  --target gfx1201 --schema "$SCHEMA" \
  --tinygrad-root <tinygrad-checkout> \
  --out-dir /tmp/llama-attention-score-precompute
```

Copy only the regenerated `.image` and `.json` after verifying target `gfx1201`, zero relocations, unchanged ABI, source digest match, and no private-segment spill. Update `kernel_assets.cpp` with the generated image digest and exact resource metadata. Do not hand-edit generated manifest values.

- [ ] **Step 5: Run asset and executor contracts**

```sh
$PY -m pytest tests/native_r9700/test_llama_attention_score_precompute.py \
  tests/native_r9700/test_llama_attention_hsa_assets.py \
  tests/native_r9700/test_hsa_code_image_loader.py \
  tests/native_r9700/test_layer0_executor_contract.py -q
```

Expected: all pass.

- [ ] **Step 6: Hardware numerical ladder**

Run:

1. layer-0 attention-score trace at position 0;
2. positions 1, 63, and 127;
3. full 16-layer prompt 16;
4. full prompt 64;
5. full prompt 128.

At every point compare against the pre-change native block-size-1 oracle. Require finite outputs, existing bounded fp16 error, and no new fault before advancing.

- [ ] **Step 7: Hardware performance gate**

Run three prompt-128 profiles before and after. Require:

- attention-score mean ticks improve by at least ten percent;
- total prompt median does not regress;
- C1R/C2R remains exact.

If the kernel misses either performance rule, revert source, generated asset, and manifest changes; do not retain a more complex neutral kernel.

- [ ] **Step 8: Commit when the gate passes**

```sh
git add native_r9700/kernels/llama_causal_attention_score_f16.cpp \
  native_r9700/kernels/llama-attention-score-hsa-assets/llama_causal_attention_score_f16.image \
  native_r9700/kernels/llama-attention-score-hsa-assets/llama_causal_attention_score_f16.json \
  native_r9700/kernel_assets.cpp \
  tests/native_r9700/test_llama_attention_hsa_assets.py \
  tests/native_r9700/test_llama_attention_score_precompute.py
git commit -m "perf(native): precompute query RoPE in attention score"
```

---

## Task 10: Final acceptance, regression sweep, and durable command ledger

**Files:**
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Modify: `docs/archive/tasks/native-r9700-producer/README.md` (current performance/status only)
- Modify: `.superpowers/swarm/progress.md` only if it is not part of the pre-existing archive reorganization in the execution worktree; otherwise leave it untouched and record status in the existing durable README.
- Evidence only: `logs/compute-side-opt/acceptance/`

**Interfaces:**
- Consumes: selected completion/barrier policies, selected block size, and optional accepted score-kernel tuning.
- Produces: final reproducible build/benchmark/parity/serving commands and evidence-backed current status.

- [ ] **Step 1: Run focused source contracts**

```sh
$PY -m pytest tests/native_r9700/test_prefill_phase_accounting.py \
  tests/native_r9700/test_rpc_accounting_contract.py \
  tests/native_r9700/test_gpu_timestamp_pm4_contract.py \
  tests/native_r9700/test_gpu_stage_profile_contract.py \
  tests/native_r9700/test_compute_barrier_policy.py \
  tests/native_r9700/test_llama_token_block_contract.py \
  tests/native_r9700/test_block_prefill_runtime_contract.py -q
```

Include the score-precompute test only when Task 9’s gate passed and the kernel was committed.

- [ ] **Step 2: Run the broader native suite**

```sh
$PY -m pytest tests/native_r9700 -v
```

Separate new failures from the documented pre-existing closure failures. Do not claim a clean suite when the command reports known failures; report exact pass/fail/skip counts and prove every changed-path focused test separately.

- [ ] **Step 3: Rebuild and run health gates**

Use the full `AGENTS.md` build. Run kernel proof and VRAM smoke. Require R9700 identity and `exit_status: 0`.

- [ ] **Step 4: Run final C1R/C2R**

Run the Task 8 parity and serving commands at the promoted production default. Require all four exact C1R token sets and C2R prompt 16/128 with accepted native cache and no fallback.

- [ ] **Step 5: Run ten-run prompt-128 stability**

Do not restart TinyGPU between runs. Every run must pass and preserve the prompt-128 decoded tokens `[13, 578, 30791, 17604]`.

- [ ] **Step 6: Capture final three-run performance and attribution**

Run three prompt-128 trials without profiling for production median and one with profiling for the final stage ranking. Record:

```text
wall_usec median
tokens_per_sec median
kernel_count
compute_submit_count
selected block_tokens
selected completion policy
selected barrier policy
sdma_submit_inclusive_usec
sdma_fence_wait_usec
sdma_submit_exclusive_usec
measured_exclusive_total_usec
unattributed_usec
all per-operation RPC counts/usec
all ten GPU stage tick shares
```

Compare to 104.6-second original, 65.9-second persistent-SDMA, and 43.7-second direct-ring controls without double-counting SDMA nested time.

- [ ] **Step 7: Update existing durable docs**

In `validation-commands.md`, replace the stale partial build command with the full build and add the exact block/profile/parity/serving commands that passed.

In the durable README, state:

- the selected production block size and measured counts;
- the final median and throughput;
- whether terminal-only and K/V overlap were promoted;
- whether score precompute passed its gate;
- honest remaining top stage/unattributed fraction;
- IB remains deferred and is not a blocker.

Do not commit logs or generated `/tmp` artifacts.

- [ ] **Step 8: Validate docs and commit**

```sh
git diff --check
git add docs/tasks/native-r9700-producer/validation-commands.md \
  docs/archive/tasks/native-r9700-producer/README.md
git commit -m "docs(native): record compute-side optimization acceptance"
```

- [ ] **Step 9: Request final code review**

Invoke `superpowers:requesting-code-review` against the complete branch. The reviewer must check:

- default-disabled profiling has no hot-path allocation/RPC;
- no inclusive/nested timing double-count;
- timestamp page bounds and terminal-signal ordering;
- exact producer/consumer barriers;
- tail-block geometry and scratch extents;
- cache position semantics across blocks;
- generated gfx1201 asset provenance when Task 9 ran;
- every C1R/C2R/stability claim points to a fresh log.

Address valid findings, rerun the smallest affected contract and the final acceptance gate, then commit fixes separately.

## Stop conditions

Stop immediately and preserve full unfiltered evidence when any of these occurs:

- hardware identity differs from `1002:7551` / `gfx1201`;
- queue timeout, non-monotonic GPU timestamps, or a new GCVM/TCP/CPF/MEC/SDMA fault;
- block size 1 no longer reproduces the native numerical control;
- any larger block fails finite/layer/KV/C1R/C2R gates;
- accepted-cache decode falls back or recomputes the prefix;
- a barrier or kernel patch regresses median wall time beyond its task threshold.

A stopped optimization does not relabel the last passing smaller block or prior direct-ring implementation as failed. The last fully accepted configuration remains the production candidate.
