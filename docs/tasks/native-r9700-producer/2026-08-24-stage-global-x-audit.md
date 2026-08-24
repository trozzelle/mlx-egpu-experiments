# Stage global_x / launch-geometry audit — 2026-08-24

Follows the `1013f19` fix (DISPATCH_DIRECT dimensions are **workgroup counts, not
work-items**; the "global divisible by workgroup" preflight was removed). This
audit is the "deferred, not blocking" item from `HANDOFF.md` and grounds the
correct workgroup count for each of the nine Llama stage kernels.

## Premise

Both Llama dispatch builders launch every stage with `workgroup_x = 64` and
`workgroup_y = workgroup_z = 1`:

- `build_llama_stage_dispatch` (single-token layer-0 trace)
- `build_llama_persistent_dispatch` (full 16-layer prefill)

The persistent path dispatches **one token at a time**: `set_llama_token_stage_scalars`
writes `sequence_length = 1` and `position = token` before each stage run, so the
correct `global_x` is always evaluated at `sequence_length = 1`.

## Kernel-by-kernel indexing model and correct global_x (sequence_length = 1)

| Stage | Kernel (`native_r9700/kernels/`) | Workgroup index = | Work-item index = | Correct `global_x` | Current |
|---|---|---|---|---|---|
| 0 | `llama_rmsnorm_f16.cpp` | `row` (lane-0 does the whole row serially) | `lane` (only `lane == 0` runs) | `1` | trace `1`; **persistent was `64`** (fixed here) |
| 1 | `llama_k_projection_f16.cpp` | `token*8 + kv_head` | `head_dimension` (0..63) | `8` | `512` (over-dispatch) |
| 2 | `llama_v_projection_f16.cpp` | `token*8 + kv_head` | `lane` (0..63) | `8` | `512` (over-dispatch) |
| 3 | `llama_rope_kv_f16.cpp` | `token*8 + kv_head` | `dimension` (0..63) | `8` | `512` (over-dispatch) |
| 4 | `llama_causal_attention_score_f16.cpp` | `query_head*seq_len + query_token` | `key_token` (0..63) | `32` | `2048` (over-dispatch) |
| 5 | `llama_causal_attention_softmax_f32.cpp` | `query_head*seq_len + query_token` (lane-0 only) | lane (only `lane == 0`) | `32` | `2048` (over-dispatch) |
| 6 | `llama_causal_attention_context_f16.cpp` | `query_head*seq_len + query_token` | `dimension` (0..63) | `32` | `2048` (over-dispatch) |
| 7 | `llama_o_projection_f16.cpp` | `token` | `output_column` (0..63) | see finding | `2048` |
| 8 | `llama_gated_mlp_f16.cpp` | `token` | `output_column` (0..63) | see finding | `2048` |

## Findings

1. **Persistent RMSNorm corruption (fixed).** `build_llama_persistent_dispatch`
   launched stage 0 with `global_x = 64`; the RMSNorm kernel treats
   `workgroup_id_x()` as the output row and lane-0 writes a full 2048-fp16 row, so
   `global_x = 64` wrote 64 rows past the single-row `normalized` buffer (4 KiB),
   corrupting the adjacent fresh-K/V buffers. The single-token trace dispatch was
   already corrected to `1` in `1013f19`; this session applied the same correction
   to the persistent dispatch (line 548).

2. **Stages 1–6 over-dispatch was harmless but wrong — corrected.** `512` and
   `2048` exceeded the correct `8` / `32` workgroups; every excess workgroup
   returned immediately via the kernel's own `token >= sequence_length` /
   `query_head >= 32` guard, so output was correct but wasted empty workgroups.
   Both dispatch builders now use `global_x = 8` (stages 1–3) and `global_x = 32`
   (stages 4–6) at `sequence_length = 1`. Kernel `.cpp` source `sha256` matched the
   committed HSA asset manifests for all nine kernels, so the sources are
   authoritative for the indexing model.

3. **Stage 7/8 output width was 64 columns, not 2048 (blocking) — fixed.**
   `llama_o_projection_f16.cpp` and `llama_gated_mlp_f16.cpp` derived the output
   column from `__builtin_amdgcn_workitem_id_x()` only, so with `workgroup_x = 64`
   each kernel wrote only the first 64 of 2048 hidden columns. Both kernels are now
   column-block indexed (`workgroup = token*32 + block`,
   `output_column = block*64 + workitem_id`) and launched with `global_x = 32`;
   their HSA assets were regenerated via
   `experiments/native-r9700-runtime/generate_hsa_code_image.py` (COMGR) and the
   `kLlamaKernelManifest` digests/`rsrc3` updated. `descriptor_offset` and
   `entry_offset` were unchanged (o-proj 1664/5888, gated-mlp 1792/6144).

4. **Attention key-token span caps recurrence at length 64.** The score/softmax/
   context kernels index `key_token` (or the softmax row walk) through
   `__builtin_amdgcn_workitem_id_x()` (`[0, 64)`) while the resident cache holds
   128 tokens. Positions 0–63 are fully covered, but a query at position ≥ 64 has
   no work-item for `key_token` 64..position, so its scores are unwritten. Lengths
   2/6/16/64 are therefore within range; length 128 needs a wider work-item span
   (or a loop) in the attention kernels before that recurrence step.

## Changes made this session

- `native_r9700/llama_layer_executor.cpp` — persistent dispatch stage 0 RMSNorm
  `global_x` corrected `64 -> 1` (mirrors `1013f19` single-token fix); stages 1–3
  `512 -> 8` and stages 4–8 `2048 -> 32` in both dispatch builders.
- `native_r9700/kernels/llama_o_projection_f16.cpp` and
  `llama_gated_mlp_f16.cpp` — column-block indexing so the full 2048-wide hidden
  row is written.
- `native_r9700/kernels/llama-o-projection-hsa-assets/` and
  `llama-gated-mlp-hsa-assets/` — regenerated `.image` + `.json`.
- `native_r9700/kernel_assets.cpp` — `kLlamaKernelManifest` digests updated and
  o-projection `rsrc3` `48 -> 64`.

## Hardware verification this session

- Health gate passes: `--kernel-proof` and `--vram-smoke` both `exit_status: 0`.
- `--stage normalized` (token 128000) still retires token-exact with
  `sha256 a0ab94d1…` after the rebuild — the geometry/asset changes did not move
  stage 0.

### Stage-1 root cause: single-dispatch compute ring (fixed)

- `--stage fresh_k` first failed closed with `trace_nonfinite`. Retaining the raw
  readback (new `.nonfinite.bin` diagnostic) showed the output is **uniform
  `0x7c00` (+Inf) across all 512 fp16 values** — an unwritten buffer, not an
  arithmetic result.
- Root cause: the compute ring was single-dispatch. `submit_compute_dispatch`
  wrote every PM4 batch at `ring[0]` and set `wptr = words.size()` (59) each
  time, so after the first dispatch the CP's `rptr` was already 59 and a second
  dispatch (`wptr` reset to 59) was invisible (`rptr == wptr`). Its `RELEASE_MEM`
  never ran; the readback saw the unwritten buffer. Two fixes: zero the
  completion timeline before each submit (so the poll actually waits), and make
  the ring circular (read prior wptr, write at `wptr % ring_dwords` with wrap,
  advance the cumulative wptr in the control mapping and the MEC doorbell).
- Result: all ten layer-0/token-0 stages now retire finite. `hidden`, `normalized`,
  `fresh_k` are **bit-exact** vs the CPU oracle; `fresh_v` is 1 ULP off on 1/512
  elements; `context`/`post_attention_hidden` are ULP-level (fp16 accumulation
  order vs NumPy matmul).

### Stage-8 root cause: fused gated-MLP kernel is numerically wrong (open)

- `--stage final_hidden` (new trace boundary, `layer0.hidden`) retires finite but
  **all 2048 columns differ** from a validated NumPy recomputation (the NumPy
  forward is bit-exact for `normalized` and `layer0_K`, so it is trustworthy).
- The gated-MLP `.cpp` source is correct (weight shapes `(8192,2048)` gate/up and
  `(2048,8192)` down, uint64 indexing, no scratch/spill: `private_seg_fixed=0`).
  The wrong-but-finite output points at the COMGR-compiled 8192×2048 fused kernel
  itself (loop-miscompilation or a `__builtin_expf`/SiLU transcendental defect —
  `expf` is only trivially exercised at `expf(0)` by the position-0 softmax).
- Consequence: the full 16-layer prefill now runs (`kernel_count=144`,
  `native_prefill_acceptance=pass`, `producer_kind=r9700_native`) and layer 0 K/V
  is correct, but layers 1–15 are wrong (they consume layer 0's gated-MLP hidden),
  so C1R prompt-0 is `P=[264,3224,7559,304]` vs `R=[12366,13,578,469]`.

### Round-3 narrowing (still open)

- The wrong output survives restructure: splitting the fused kernel into a
  `gate_up` (RMSNorm + gate/up) + `mlp_down` (SiLU + down + residual) pair, and
  separately removing the SiLU (`silu_gate = gate`), each still leave the result
  wrong (max diff ~9.2 with SiLU, ~2.8 without). Rounding the RMSNorm output to
  fp16 in `gate_up` does not change it. So this is **not** the fused-loop shape,
  and **not** a simple SiLU/`expf` defect.
- The working kernels (K/V/o projection) read weights up to 8 MiB; the failing
  MLP kernels read 32 MiB `gate/up/down` weights. The full prefill transfers
  ~2.07 GB (weights fully uploaded), so the remaining suspect is the resident
  **GPU-VA/PTB mapping of the 32 MiB weight buffers** (or their SDMA chunking),
  not the kernel arithmetic. Next: read back a 32 MiB weight span on-device and
  compare it byte-for-byte with the safetensors shard.

## Remaining

- Verify the 32 MiB gate/up/down weight buffers are byte-identical on-device
  (add a weight readback or a bounded gate/up trace), then repair the mapping if
  truncated; otherwise continue kernel-level isolation.
- Re-verify `final_hidden` and layer 1 K/V, then re-run C1R parity (prompt-0).
- Then layer-0 recurrence at lengths 2/6/16/64, the attention key-token span for
  length 128, and C2R.
