# C1R-6aa native attention softmax proof

Status: complete; hardware wrapper proof passed.

## Decision

Implement `layer0_attention_probs_head0_tokens0_5_softmax_from_scaled_masked_chain` as a native gfx1201 probability tile proof over the fixture scaled/masked fp32 score tile. This closes the prior `softmax_status: not_implemented_fixture_probs` gap for probabilities themselves, but does not yet make the context chain consume the native softmax output and does not claim full native prefill acceptance.

## Root-cause notes from hardware debugging

- `global_store_b32` must use the `vsrc=<data>` form for this tinygrad RDNA4 DSL when hardware stores the value; `vdst=<data>` decoded plausibly but wrote the incoming `v0` value (`0x00000007`) on hardware.
- The working address register pattern follows existing SiLU kernels: preserve incoming `v0`, use `v[1:2]` as the zero address pair, and use `s[4:5]` / `s[6:7]` as output/input base pointers from the 24-byte kernarg layout.
- Vector global loads require `s_wait_loadcnt(0)`; `s_wait_kmcnt(0)` is only for scalar kernarg loads.
- The softmax kernel uses v0..v14, so `kC1SoftmaxKernelRsrc1 = kKernelReferenceRsrc1 | 0x7U` remains required.
- Runtime wrapper marker constants were aligned to the actual bridge-resident generic pages: score `0x0000200000001000` PTB 1 and output `0x0000200000004000` PTB 4.

## Verification

- Focused runtime contracts: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_probs_head0_tokens0_5_softmax_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_attention_probs_softmax_row_sum_marker -q` -> `3 passed in 6.26s`.
- Bridge/runner compile: `xcrun --sdk macosx clang++ ... c1_primitive_bridge.cpp` and `runtime.cpp runner.cpp` -> exit 0 with no output.
- Hardware wrapper proof: `NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_attention_probs_head0_tokens0_5_softmax_from_scaled_masked_chain` -> exit 0, log `logs/c1-runner-primitive-chain-proof-layer0_attention_probs_head0_tokens0_5_softmax_from_scaled_masked_chain-2026-08-19T23:36:36Z.log`, `primitive_chain_proof_wrapper_status: pass`, `mismatch_count: 0`, `max_abs_diff: 5.9604644775390625e-08`, `max_ulp_diff: 3`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `wrapper_exit_status: 0`, `exit_status: 0`.

## Remaining C1 gap

Next slice: integrate scaled/masked score -> native softmax -> context weighted-sum so the context proof consumes native probabilities rather than `fixture_attention_probs_fp32_cast_to_fp16`.
