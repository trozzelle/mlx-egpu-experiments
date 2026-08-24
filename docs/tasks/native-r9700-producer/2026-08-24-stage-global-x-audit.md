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

### Stage-4 root cause: missing query RoPE (fixed)

- The earlier "gated-MLP is wrong" and "32 MiB weight" hypotheses were a
  **reference bug**, not a kernel bug: my NumPy recomputation repeated the V
  vector element-wise (`np.repeat(v, 4, axis=1)`) instead of repeating each KV
  head, so the GQA context and everything downstream was wrong. With the correct
  head-wise repeat, the gated-MLP output and the CPU reference agree to fp16 ULP
  (max ~1 ULP).
- The real defect was that the attention **score kernel projected Q on the fly
  but never RoPE-rotated it**, while the K cache was already rotated by the
  rope-kv stage. Position 0 is RoPE identity, so the single-token trace looked
  correct; every position > 0 attended with an unrotated query, corrupting
  multi-token prefixes and every layer ≥ 1. Fixed by rotating the query in the
  score kernel (pair-wise projection + the same llama3-scaled cos/sin split-half
  rotation as the rope-kv kernel, using `absolute_query`).
- Result: n=2 prefill matches the CPU reference to fp16 ULP across all 16 layers
  (max ~4 ULP at layer 15), and **C1R prompt-0 is token-exact**
  `P == R == [12366, 13, 578, 469]`. This is the first native R9700 C1R
  acceptance.

### Performance split + 16-token acceptance (done)

- Split the fused gated-MLP into `llama_gate_up_projection_f16` (RMSNorm +
  gate/up projected once, fp16 out) + `llama_mlp_down_f16` (SiLU + down +
  residual; the 32 KiB gate/up is cache-resident). This removes the 137 GB
  redundant per-dispatch read and drops the 16-token prefill from >300 s (and a
  timeline timeout) to ~59 s, still ULP-correct.
- Added `prompt-16` (16 prefix + 1 final token). **C1R is token-exact**:
  `P == R == [11, 706, 28995, 12207]`, and **C2R imported-cache serving passes**
  (`route=native_producer`, `accepted_cache=true`, `fallback_reason=none`,
  `decoded_tokens == [11, 706, 28995, 12207]`) — the native producer now reaches
  token-for-token C1R/C2R parity at the meaningful 16-token length.

## Remaining

- The score kernel still recomputes `powf`/`cosf`/`sinf` per (head, key, pair);
  precompute the RoPE cos/sin once (performance only).
- Widen the attention key-token span past 64 for the full 128-token cache, then
  run the 64/128-token progression (the 222/661 fixtures also exceed the current
  128-token cache). Qwen can then resume.
