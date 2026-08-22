# C1R pivot: product prefill worker acceptance

## Decision
Stop expanding O-proj width proofs as the default C1R path. Treat existing primitive-chain proofs as confidence anchors, not the product acceptance gate.

## New acceptance target
C1R is accepted when the native prefill worker can run the intended Llama prompt path end-to-end far enough to emit a usable prefill artifact or KV cache through the intended interface.

## Allowed gaps
CPU fallback or oracle-backed regions are allowed only when explicitly logged. They are known limitations, not hidden success paths.

## Immediate validation
Run one real prefill smoke scenario against the intended interface:

1. Produce a prefill artifact for the local Llama model and prompt path.
2. Convert or emit a KV cache if the artifact contains usable layer K/V data.
3. Inspect logs/artifact metadata for R9700-backed primitive use and explicit fallback/oracle regions.
4. If usable, move to C2 worker/serving integration.

## Smoke result
- Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.serving --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --producer-model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --prompt-name prompt-0 --threshold-tokens 2 --max-new-tokens 4 --artifacts-dir artifacts/c1r-prefill-smoke --json .superpowers/swarm/reports/c1r-prefill-smoke-result.json --log logs/c1r-prefill-smoke.log --report .superpowers/swarm/reports/c1r-prefill-smoke-report.md`
- Result: exit `0`, `gate_result=pass`, `route=native_producer`, `accepted_cache=true`.
- Decode check: prompt `prompt-0`, `S=6`, `n_prefix=5`, decoded tokens `[12366, 13, 578, 469]` exactly matched `baseline_r_tokens.json`.
- Prefill artifact: `artifacts/c1r-prefill-smoke/prompt-0.prefill.npz`, `producer_kind=cpu_reference`, `num_layers=16`, fp16 K/V shape `(1, 8, 5, 64)`.
- Prompt-cache artifact: `artifacts/c1r-prefill-smoke/prompt-0.prompt-cache.safetensors`, 32 tensors, metadata `offset=5`, `num_layers=16`, `n_kv_heads=8`, `head_dim=64`; accepted by the serving wrapper's `load_prompt_cache` validation.
- Limitation: this is an accepted imported-cache product seam with the CPU reference producer. It is not a native R9700 prefill-speed claim.

## Non-goals after pivot
- No more exhaustive O-proj cols384:2048 expansion unless the smoke exposes that exact gap as a hard blocker.
- No claim of proof-complete full hardware implementation.
- No Qwen support in C1R; Qwen remains a separate target-expansion phase.

## Rationale
Full per-component hardware proof improves confidence and future regression stability, but does not directly buy a usable prefill worker faster. Product utility now comes from end-to-end prefill artifact usability with honest fallback/oracle logging.
