# LQ-W3 stage asset integration

## Integrated assets

The source-to-HSA generator now has closed reviewed identities and exact schemas for Llama RMSNorm, K/V projection, RoPE/KV, fused causal score, softmax, context, O projection, gated MLP, and Qwen affine4, DeltaNet, and full-attention source kernels. Fresh `gfx1201` images and manifests were generated under their corresponding `native_r9700/kernels/*-hsa-assets/` directories.

`native_r9700/kernel_assets.cpp` contains static reviewed manifest entries for all nine Llama stage images. It verifies full HSA image hashes and permits the generator's bounded 4 MiB image limit rather than the obsolete 4 KiB C0 raw-code limit.

## Executor wiring
`native_r9700/kernel_assets.cpp` contains static reviewed manifest entries for all nine Llama stage images. It verifies full HSA image hashes and permits the generator's bounded 4 MiB image limit rather than the obsolete 4 KiB C0 raw-code limit.
## Executor boundary

Qwen now has a text-only stage planner that preserves the actual interleaved
64-layer cache order and selects affine4+DeltaNet for `ArraysCache` versus
affine4+full attention for `KVCache`. Llama's existing executor now recognizes
the nine reviewed asset identities, but physical AMDev HSA resident dispatch
still needs an image-entry-aware dynamic-buffer backend before it can execute a
layer or publish K/V.
```sh
${PY} -m pytest \
  tests/native_r9700/test_hsa_code_image_generator.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_llama_stage_layout.py \
  tests/native_r9700/test_layer0_executor_contract.py \
  tests/native_r9700/test_llama_attention_hsa_assets.py \
  tests/native_r9700/test_qwen_hsa_kernel_assets.py -q
```

The physical AMDev resident HSA stage-dispatch loop remains the next blocking implementation task; no native prefill/parity claim is made by this integration.
