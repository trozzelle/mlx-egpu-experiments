# C1R-6z attention context weighted-sum proof

Status: complete; initial review findings and rereview blocker fixed.

## Decision
Implement the next C1 attention slice as a proof-only weighted-sum context matmul: fixture probabilities cast to fp16 as A, layer-0 head0 V projection cols0:64 as B, and the proven split-AB 8x16x8 kernel dispatched once per 8-column output tile.

Reason: it advances from score tiles into `probs @ V` without inventing a native softmax kernel. The contract explicitly records `probs_source: fixture_attention_probs_fp32_cast_to_fp16` and `softmax_status: not_implemented_fixture_probs`; `native_prefill_acceptance` stays `open`.

## Implementation notes
- Chain: `layer0_attention_context_head0_tokens0_5_cols0_64_weighted_sum_chain`.
- Inputs: `layer0_attention_context_head0_tokens0_5_cols0_64_probs_fp16`, `layer0_attention_context_head0_tokens0_5_cols0_64_v_as_b_fp16`.
- Output: fp32 `(8,64)` stitched from eight 8x8 output tiles; valid context shape `5x64`.
- Runtime/bridge use the split-AB resident VA pattern: activation `0x0000200000001000`, model/V `0x0000200000011000`, output `0x0000200000012000`.
- Root cause fixed during hardware verification: the initial context bridge cloned the combined-input K-tile `PrimitiveSpec` and loaded the wrong kernel text, which produced zero output. The final bridge explicitly selects `c1r6k-layer0-k-tile-split-ab-gemm-v1` and passes the V tile stream via the model-weight kernarg.
- Review fixes applied: real bridge now emits `output_tile1_cols: 8:16`; runtime validates `output_tile1_cols`, `activation_upload_status`, and `model_weight_upload_status`; tests cover missing upload markers; the accidentally broadened cols0:16 `output_tile_count` assertion was restored to `2`; rereview found and the supervisor removed the temporary `NATIVE_R9700_CONTEXT_READBACK_DUMP` hook.

## Verification
- Focused context host tests before review fixes: `4 passed in 4.27s`.
- Focused review-fix tests: `3 passed in 6.10s`.
- Full native regression: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` -> `203 passed, 2 warnings in 128.88s`; `git diff --check ...` exited 0 with no output.
- Real wrapper hardware proof: exit `0`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_context_head0_tokens0_5_cols0_64_weighted_sum_chain-2026-08-19T23:01:20Z.log`.
- Hardware markers: `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 2.9802322387695312e-08`, `max_ulp_diff: 2`, `byte_mismatch_count: 23`, `wrapper_exit_status: 0`, `exit_status: 0`.
