# C1R-6y scaled/masked attention-score proof

Status: complete; reviewer passed.

## Decision
Implement the scaled/masked score proof as a seeded fp32 accumulator chain, not as a post-raw kernel. The Q operand is pre-scaled by exact power-of-two `0.125` and the output tile is seeded with the causal/padding mask (`0` for valid cells, `-inf` elsewhere) before accumulation.

Reason: this reuses the proven 8x16x8 accumulator kernel and keeps the proof honest: it proves hardware accumulation of scaled/masked score tiles while leaving softmax/context/native prefill open.

## Scope
- Chain: `layer0_attention_scores_head0_tokens0_5_scaled_masked_chain`.
- Inputs: `layer0_attention_scores_head0_tokens0_5_scaled_masked_q_scaled_fp16`, `layer0_attention_scores_head0_tokens0_5_scaled_masked_k_as_b_fp16`, `layer0_attention_scores_head0_tokens0_5_scaled_masked_seed_fp32`.
- Output: fp32 `(8,8)`, valid score shape `5x5`, finite causal score count `15`.
- Acceptance remains `hardware_primitive_chain_only`; `native_prefill_acceptance` remains `open`.

## Verification
- Focused host tests: `4 passed in 4.23s`.
- Real wrapper hardware proof: exit `0`, log `logs/c1-runner-primitive-chain-proof-layer0_attention_scores_head0_tokens0_5_scaled_masked_chain-2026-08-19T22:26:51Z.log`.
- Hardware markers: `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `max_abs_diff: 0`, `max_ulp_diff: 0`, `wrapper_exit_status: 0`, `exit_status: 0`.
- Reviewer: `C1ScaledMaskedReview`, no findings.
