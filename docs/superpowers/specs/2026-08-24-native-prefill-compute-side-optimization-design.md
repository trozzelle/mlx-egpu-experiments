# Native R9700 Prefill Compute-Side Optimization Design

**Date:** 2026-08-24  
**Status:** Approved design  
**Baseline:** `feature/native-r9700-producer` at `512a58a`  
**Reference analysis:** `${HOME}/Downloads/ChatGPT-Diagnose R9700 Mapping Issues-20260824-1908.pdf`

## Goal

Complete the remaining compute-side optimization of the accepted Llama 3.2 1B native R9700 prefill path. Preserve the mlx-lm prompt-cache boundary, hardware-backed `producer_kind=r9700_native`, S-1 cache semantics, token-exact C1R/C2R acceptance, and fail-closed behavior.

The work proceeds from accurate attribution to GPU stage profiling, measured barrier reduction, real token-block prefill, and profile-directed tuning of only the dominant kernels. It does not reopen the validated SDMA/direct-ring work without evidence that compute improvements have made transport dominant again.

## Accepted baseline

The implementation through `512a58a` is the frozen control:

- `23578fa`: ten per-token stages are submitted as one 590-dword direct-ring batch, one doorbell, and one terminal host timeline poll.
- `512a58a`: structured host phase timers and submission/RPC counters.
- Prompt-128 median wall time is approximately 43.7 seconds at 2.93 prefix tokens/second.
- A prompt-128 prefill executes 20,480 kernels in 2,048 compute submissions.
- C1R is token-exact at prompt 0/16/64/128.
- C2R prompt 16/128 accepts the native prompt cache and decodes without fallback.
- Ten prompt-128 stability runs pass.

The direct-ring path remains the production baseline. The prior indirect-buffer attempt is not a blocker. The PDF identifies a likely packet-encoding error (`PACKET3_INDIRECT_BUFFER` payload count 2 plus the valid bit), but the current 590-dword command list occupies only about seven percent of the 32 KiB ring and PM4 construction/publication is negligible.

## Corrections to the reference analysis

The PDF supplies the correct overall direction, but two current-repository facts change the first implementation steps.

### Successful-submit diagnostics are not on the production batch path

`ResidentHsaSession::dispatch_batch()` in `native_r9700/amdev_session.cpp` writes the ring, flushes HDP, publishes WPTR, rings the doorbell, and polls the mapped timeline directly. It does not call `submit_compute_dispatch_with_post_doorbell_diagnostics()` or `log_compute_queue_post_doorbell_diagnostics()`.

Therefore, the 81,708 aggregate socket RPCs cannot be attributed to successful post-doorbell HQD diagnostics without measurement. The correct first change is fixed-cost per-operation RPC accounting and timing. Fail-only queue diagnostics should remain the production behavior of `dispatch_batch()`; legacy/proof paths may retain explicit successful-submit diagnostics where they are part of the proof contract.

### The kernels already expose multi-token geometry

The Llama kernels and `llama_stage_layout.cpp` already describe `[N, ...]` inputs and accept `sequence_length`/`position` where required. `llama_rmsnorm_f16` indexes rows by workgroup, and the projection, attention, output, and MLP kernels index token work from `sequence_length`.

The current executor deliberately forces `sequence_length=1`, allocates one hidden buffer per token, sizes shared scratch for one token, and invokes one ten-stage batch for every token/layer pair. Real token batching is therefore primarily an executor/buffer/geometry change first. Kernel rewrites are deferred until the GPU profile proves where they are needed.

## Constraints

- No CPU/NumPy artifact may satisfy native acceptance.
- Block size 1 must remain a byte-for-byte control for PM4 shape and token-exact control for observable output.
- No new fixed-VA or standalone sysmem mapping is introduced for profiling. The earlier kernarg-arena and IB-buffer failures make unnecessary mapping experiments unacceptable.
- Hardware commands are serialized through the existing hardware lock. A failed or timed-out queue is not reused.
- Numerical changes require serial-native comparison at layer/KV boundaries before C1R/C2R promotion.
- Additive profiling data must not allocate, format strings, or perform socket RPCs in the production compute hot path when profiling is disabled.
- Qwen3.8-27B, request batching, network transport, and SDMA redesign are separate slices.

## Architecture

### Phase A: Complete host and RPC attribution

The current host timing fields mix inclusive and nested intervals. In particular, `sdma_submit_usec` contains `sdma_fence_wait_usec`; adding them double-counts the fence wait. The structured result will expose:

- `sdma_submit_inclusive_usec`
- `sdma_fence_wait_usec`
- `sdma_submit_exclusive_usec`
- `measured_exclusive_total_usec`
- `unattributed_usec`

Outer regions will be measured explicitly:

- model metadata/binding
- persistent dispatch construction
- device/session preparation
- VM/page-table setup
- resident allocation
- HSA image upload
- embedding upload
- layer-weight upload
- KV readback
- NPZ serialization
- session close
- host loop/miscellaneous work

`RemoteClient` will use fixed-size counters indexed by `RemoteCmd`, plus distinct counters for fire-and-forget MMIO write and `MAP_SYSMEM_FD`. Each bucket records call count and elapsed microseconds. No map, string key, or heap allocation is used per RPC. The aggregate `socket_rpc_count` remains as the sum for compatibility with existing logs; operation buckets become the authoritative diagnosis.

Acceptance for Phase A is accounting, not a speedup: measured exclusive time must not exceed wall time, unattributed time must be nonnegative, and the three-run prompt-128 median must remain within three percent of the baseline.

### Phase B: GPU-clock stage profiling

Profiling uses the unused portion of the already mapped compute-control queue page at `kRptrVa`. The page currently stores RPTR, WPTR, and the terminal timeline in its first 24 bytes. A reviewed aligned offset in the same page will hold eleven 64-bit GPU-clock boundaries for one ten-stage batch.

This avoids a new PTE, TLB flush, sysmem allocation, or GFXHUB visibility assumption. It reuses the same physical page proven writable by `RELEASE_MEM` for the timeline.

The PM4 encoding is grounded in the local tinygrad AMD queue:

- `ops_amd.py` writes a GPU clock counter with `RELEASE_MEM data_sel__send_gpu_clock_counter`.
- local `pm4_soc15.py` defines that selector as value 3.
- tinygrad orders the timestamp with release/acquire operations and retains `CS_PARTIAL_FLUSH` after `DISPATCH_DIRECT`.

Profiling is optional and leaves production PM4 byte-for-byte unchanged when disabled. It records T0 before stage 0 and T1–T10 after stages 0–9, then performs the existing terminal host timeline signal and poll. The timestamp page is read only after terminal completion; no host wait or socket read is inserted between stages.

Initial reports use raw GPU clock counts. Conversion to microseconds and effective GB/s is added only after the counter frequency/unit is source-grounded or calibrated; the implementation must not guess. Reports include stage name, invocation count, min/mean/p50/p95/max ticks, and percentage of measured batch ticks.

The profiling ladder is:

1. token 0, layer 0;
2. token 0, layers 0–15;
3. tokens 0, 63, and 127 across representative layers;
4. one full prompt only if the targeted samples disagree about the ranking.

### Phase C: Dependency-grounded barrier narrowing

The production stream initially remains the validated per-stage sequence:

`ACQUIRE_MEM → program registers → DISPATCH_DIRECT → CS_PARTIAL_FLUSH → RELEASE_MEM`

Barrier work is split into two independent changes.

First, stages 0–8 stop writing host timeline values. Their `RELEASE_MEM` packets retain completion/cache semantics but use no data write. Stage 9 alone writes the terminal timeline value that the host polls.

Second, broad ordering is narrowed one dependency edge at a time. The stage buffer graph is:

1. RMSNorm: hidden → normalized
2. K projection: normalized → fresh K
3. V projection: normalized → fresh V
4. RoPE/KV write: fresh K + fresh V → K/V cache
5. attention score: normalized + K cache → scores
6. softmax: scores → probabilities
7. attention context: probabilities + V cache → context
8. output projection/residual: context + hidden → post-attention hidden
9. gate/up projection: post-attention hidden → gate/up
10. down projection/residual: gate/up + post-attention hidden → hidden

The first overlap candidate is the edge between K and V projection. These stages read the same immutable normalized input and write disjoint outputs; RoPE is their join. A no-partial-flush K→V variant is tested only after timestamps show material barrier/gap cost. The barrier before RoPE remains.

All other edges remain serialized until their exact producer/consumer buffers and GPU timestamps justify a narrower operation. Cache writeback/invalidation is never removed by packet deletion alone.

### Phase D: Real token-block prefill

The executor will represent prompt positions as token blocks rather than one hidden allocation per token:

```text
LlamaTokenBlock
  buffer_index   contiguous fp16 hidden rows [token_count, 2048]
  position       absolute first cache position
  token_count    live rows; tail block may be smaller than capacity
```

`build_llama_persistent_dispatch()` receives a block capacity and partitions the prompt into `ceil(token_count / block_capacity)` blocks. For each block it allocates one contiguous hidden buffer and uploads the selected embedding rows as one contiguous byte span.

Shared scratch is sized for the block capacity:

- normalized/context/post-attention hidden: `B × 2048 × fp16`
- fresh K/V: `8 × B × 64 × fp16`
- attention scores/probabilities: `32 × B × 128 × fp32`
- gate/up: `B × 8192 × fp16`

A block-stage update replaces the token-only mutation API. For each live block it writes:

- `sequence_length = block.token_count`
- `position = block.position`
- `cache_capacity_tokens = 128`
- stage workgroup counts scaled by the live token count, including the smaller tail block
- the hidden-buffer binding for that block

Layer-major execution remains unchanged at the architectural level: upload the layer’s nine weight spans once, then process blocks in increasing causal position. K/V caches remain per-layer and persistent across blocks.

The hardware ladder is `B=1 → 2 → 4 → 8 → 16 → 32`. Each step must pass before the next starts. Block size 1 is the serial-native oracle. The promoted production size is the fastest stable median, not automatically the largest block.

For prompt-128, block size 16 has the structural target:

- 8 blocks × 16 layers = 128 compute submissions
- 8 × 16 × 10 = 1,280 kernel executions

versus 2,048 submissions and 20,480 executions at block size 1.

### Phase E: Profile-directed kernel tuning

Only stages proven dominant by Phase B are modified.

The first explicit candidate is `llama_causal_attention_score_f16`. It currently projects and RoPE-rotates Q inside each key-token lane, repeating the 2048-element projection and `powf/cosf/sinf` work for every key token. If the stage profile confirms it is dominant, the replacement stays within the same stage and kernarg ABI:

1. each 64-lane workgroup computes the 64-element query head once for one `(query_head, query_token)`;
2. it applies RoPE once per query dimension pair;
3. it retains the rotated Q in LDS/registers;
4. lanes reuse that Q across key blocks when computing Q·K.

Keeping this as one stage avoids an eleventh kernarg slot and preserves the validated ten-stage batch shape.

Other tuning is conditional on measured classification:

- lane-0 RMSNorm or softmax dominance: wave-level reduction and cooperative rows;
- bandwidth-bound projections: token-tiled weight reuse and coalesced packed loads;
- reduction-bound kernels: wave reductions before LDS designs;
- occupancy/instruction bound kernels: resource metadata and workgroup geometry A/B tests.

No multi-kernel fusion, matrix-core rewrite, or broad source rewrite is included without a measured dominant stage and a focused numerical oracle.

## Data flow

For block size `B`, one layer executes:

```text
hidden block [B,2048]
  → RMSNorm
normalized [B,2048]
  → K projection ─┐
  → V projection ─┴→ RoPE + contiguous KV cache write at block.position
  → Q/attention score against cache positions <= each absolute query
  → causal softmax
  → attention context
  → output projection + residual
  → gate/up projection
  → down projection + residual
hidden block [B,2048]
```

Blocks execute in increasing position, so each block’s causal attention observes all earlier blocks plus the causal prefix within itself. Layers execute outermost so each layer’s weights are transferred once.

## Failure handling

- Invalid block size, tail geometry, scratch extent, or cache range fails before device submission.
- GPU timestamp counters must be monotonic within a profiled batch; zero, decreasing, or missing boundaries fail the profiling command but do not alter production acceptance.
- Any timeout captures the existing full queue/consumption diagnostics. No automatic replay or fallback occurs after cache acceptance.
- A block-size parity failure stops the ladder at that size; larger sizes do not run.
- A barrier experiment is reverted if it changes any acceptance output, introduces non-finite data, creates a new GPU fault, or regresses the three-run median by more than three percent.
- The serial block-size-1 path remains available as the native oracle during development; production promotion is a clean selection of one validated block size, not a silent runtime fallback.

## Verification and acceptance

### Hardware-free contracts

- host timing arithmetic distinguishes inclusive, nested, and exclusive intervals;
- per-operation RPC counts sum to the aggregate count;
- timestamp slots fit the proven control page and do not overlap RPTR/WPTR/timeline;
- profiling-disabled PM4 matches the frozen production stream;
- the profiled stream has eleven ordered clock destinations and one terminal host signal;
- block partitioning handles exact and tail blocks;
- every scratch allocation covers the maximum live block geometry;
- every stage scalar and workgroup count matches its kernel indexing model.

### Hardware gates

For every promoted barrier mode, block size, and tuned kernel:

1. kernel proof and VRAM smoke remain healthy;
2. no new GCVM/TCP/CPF/MEC/SDMA faults;
3. finite outputs;
4. bounded per-layer hidden and KV error versus block size 1;
5. C1R prompt 0/16/64/128 token-exact;
6. C2R prompt 16/128: `route=native_producer`, `accepted_cache=true`, no fallback;
7. ten prompt-128 runs without TinyGPU restart;
8. three prompt-128 timing runs, reporting median wall time, tokens/sec, host/RPC accounting, GPU stage ranking, submission count, and kernel count.

The plan may promote a larger block only when it preserves all acceptance gates and improves the three-run median. The final report must separate host/transport, GPU completion, attributed GPU stage ticks, and unattributed time.

## Non-goals

- Qwen3.8-27B support.
- Multiple concurrent requests.
- Network transport.
- A new kernarg arena or timestamp mapping.
- Indirect-buffer production promotion.
- Further SDMA work before compute improvements make the current inclusive SDMA interval a dominant fraction.
- Byte-identical K/V when token-exact decode and bounded native-vs-native numerical evidence pass; byte identity is not the existing acceptance contract.
