# LN-1C first stage evidence — 2026-08-23

## Environment

- Runtime: TinyGPU.app/APLRemotePCIDevice/PCIIface, `1002:7551`, `gfx1201`.
- Model: local Llama 3.2 1B MLX safetensors model.
- Request: layer 0, token ID 128000, position 0.

## Precondition

Fresh runner `--kernel-proof` and `--vram-smoke` passed before tracing.

## Boundary results

1. `hidden`
   - Oracle and native each reported `layer0.embedding_row`, fp16 `[1,2048]`, 4096 bytes, 2048 finite values.
   - SHA-256 matches exactly: `4d2c5ceaca8ace6263af0d595b6d47040dc4a91b6abf1b72edbe89418129b808`.
   - Artifacts: `logs/ln-1-oracle/hidden/`, `logs/ln-1-native/layer0-token0-hidden/`.
2. `normalized`
   - Oracle reported finite fp16 `[1,2048]` output.
   - Native trace fails closed before publication: `failure_stage: trace_nonfinite`, `failure_text: trace output contains NaN or infinity`, `exit_status: 1`.

## First failed invariant

`llama_layer0_rmsnorm_numeric`: layer-0 RMSNorm native output must contain exactly 2048 finite fp16 values and match the oracle boundary representation. The input embedding transfer is proven byte-exact; do not inspect K/V/RoPE/attention/MLP until RMSNorm passes.

## Next task

LN-2 is unblocked only for the RMSNorm stage asset, kernarg schema/binding, scalar epsilon, workgroup geometry, and output buffer contract. It must add a focused failing contract, make one minimal fix, and rerun this exact normalized trace.