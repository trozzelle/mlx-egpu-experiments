# Llama native numerical-debug plan — 2026-08-23

## Goal

Make the R9700 Llama 3.2 1B producer emit finite, numerically correct prefix K/V for increasingly long prompts, then prove exact mlx-lm final-token decode parity. The minimum meaningful acceptance length is 16 prefix tokens; the target progression continues through 64 and 128 tokens, the current native cache capacity.

## Facts and constraints

- The hardware runtime is now healthy: C0 kernel proof and resident VRAM smoke pass.
- The dedicated staging PTB preserves legacy C0 fixed mappings while retaining 1 MiB transfer chunks. Do not revisit that topology while debugging model math.
- A two-token full native prefill completes all 16 layers, emits a schema-valid `r9700_native` NPZ, and converts to an mlx-lm prompt cache.
- Native C1R prompt-0 fails: P tokens are `[0,0,0,0]`, R tokens are `[12366,13,578,469]`; K/V comparisons are NaN.
- CPU/NumPy is an oracle only. It may compute and serialize per-stage reference tensors, never populate an accepted native artifact.
- Preserve S-1 semantics: producer caches prefix tokens only; mlx-lm receives the final prompt token.
- Stop at the first non-finite or out-of-tolerance stage. Do not run later layers, longer prompts, C2R, or Qwen while an earlier gate fails.

## Diagnostic artifact contract

Add a fail-closed, request-scoped stage trace mode to the native runner and a matching oracle generator.

For each traced stage record:

- `token_index`, `layer_index`, stage name, HSA image digest, kernarg bytes/digest, scalar fields, buffer names, GPU VAs, byte count.
- Device readback digest and finite-value count for every declared output buffer.
- Optional raw fp16/fp32 output artifact under a run-local directory, never promoted into the accepted NPZ.
- Oracle shape/dtype, digest, finite-value count, max absolute error, mean absolute error, and first mismatch coordinate/value.
- `failure_stage=llama_layer<L>_<stage>_numeric` on the first failed comparison.

The trace must be bounded: one token, one layer, one named stage or stage prefix per invocation. It must clean temporary files and never publish the final NPZ on a numeric failure.

## Phase A — establish a trustworthy oracle

1. Add a Python oracle command for Llama layer-0, token 0.
   - Reuse the existing strict model loader and NumPy reference primitives.
   - Produce the exact tensors at the nine stage boundaries:
     `hidden`, `normalized`, `fresh_k`, `fresh_v`, `k_cache`, `v_cache`, `attention_scores`, `attention_probabilities`, `context`, and `post_attention_hidden`.
   - Record actual input token IDs, position, RoPE constants, weight names, shapes, and dtypes.
2. Add focused tests proving the oracle rejects wrong model geometry, token position, shape, and dtype.
3. Capture and commit only small deterministic fixture slices/digests needed by no-hardware tests; keep full oracle tensors in generated run directories.

Gate: CPU oracle tensors are finite and deterministic across two runs for prompt-0’s first prefix token.

## Phase B — isolate the first bad stage

1. Add native trace-mode readback for a single named resident buffer after each selected stage.
2. Begin with layer 0, token 0 and compare only the embedding/hidden input before `rmsnorm`.
   - Verify selected embedding row bytes against the safetensors source.
   - Verify hidden-buffer bytes after the embedding transfer.
3. Execute stages strictly in order:
   1. `rmsnorm`
   2. `k_projection`
   3. `v_projection`
   4. `rope_kv`
   5. `attention_score`
   6. `attention_softmax`
   7. `attention_context`
   8. `o_projection`
   9. `gated_mlp`
4. For each stage, first assert output has no NaN/Inf, then compare shape, dtype, digest, and tolerance against its oracle tensor.
5. Stop at the first failing stage; write a single compact diagnostic report with its precise kernargs, input/output buffer slices, and first mismatch.

Gate: identify one earliest divergent stage and one concrete violated invariant. No later stage debugging before that result exists.

## Phase C — repair one stage at a time

For the first failed stage:

1. Write a focused failing hardware-free contract around the identified invariant: asset schema, grid geometry, buffer offsets/strides, scalar position, fp16/fp32 accumulation, RoPE layout, or causal-mask extent.
2. Compare the native kernel source, generated gfx1201 asset manifest, and `LlamaStageAssetConfig`/kernarg bindings against the oracle’s tensor contract.
3. Make the smallest source/asset/binding change that addresses that invariant. Do not modify later stages or cache serialization.
4. Run the stage trace on hardware for layer 0/token 0. Require finite output and numeric agreement before proceeding.
5. Repeat for the next stage only after the previous one passes.

Per-stage gate: all inputs and output finite; exact integer metadata; fp16/fp32 tolerance stated in the test; first mismatch absent. A stage that cannot meet its declared numerical contract remains fail-closed.

## Phase D — complete layer and token recurrence

1. Run all nine passing stages for layer 0/token 0.
2. Validate layer-0 `fresh_k`, `fresh_v`, and cache placement against the oracle.
3. Run token 1 at layer 0. Validate cache position 1, causal-score extent 2, softmax normalization, and output hidden state.
4. Expand layer 0 through prefix lengths 6, 16, 64, and 128 using deterministic fixture token sequences.
5. At every length, validate only the newly written K/V slot plus bounded attention output slices; full cache comparison is required at lengths 6, 16, 64, and 128.

Gate: layer 0 remains finite and numerically within tolerance at every target length; no overwrite, cache-position, or causal-mask error.

## Phase E — layer recurrence through all 16 layers

1. Promote the validated layer-0 contract to layer 1 with layer-specific weight provenance and outputs.
2. For each layer 1–15, compare its first-token stage boundaries; stop at the first divergent layer/stage.
3. After all layers pass one token, repeat two tokens, then the 6/16/64/128-token progression.
4. Produce the final native NPZ only after all layers pass the selected target length.

Gate: all 16 K/V pairs are finite, correct shape `(1, 8, N, 64)`, fp16, and numerically match oracle K/V within the declared tolerance.

## Phase F — consumer acceptance

1. Convert the hardware-generated NPZ with unchanged `native_r9700.kv_cache`.
2. Run native C1R parity first at prompt-0, then every committed fixture prompt.
3. Require exact P/R generated-token equality, not similarity.
4. Run C2R imported-cache serving using only the accepted cache and the final prompt token. Confirm no decode fallback recomputes the prefix.
5. Only after C1R and C2R pass at the meaningful 16-token length may Qwen native work resume.

## Verification commands

Use the pinned interpreter:

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/native_r9700/test_layer0_executor_contract.py tests/native_r9700/test_runtime_vram_contract.py -q
$PY -m pytest tests/test_native_amdev_transfer_contract.py -q
$PY -m pytest tests/native_r9700 -q
```

Hardware sequence for every changed numerical premise:

```sh
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --kernel-proof

APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --vram-smoke
```

Do not launch full prefill until the relevant layer/stage trace passes. Do not claim native acceptance until C1R is token-exact and C2R consumes the accepted imported cache.
