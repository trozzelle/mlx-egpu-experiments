# LN-2 RMSNorm Zero-Store Asset

## Diagnostic boundary

`llama_rmsnorm_zero_store_f16` is a digest-bound gfx1201 HSA image generated from `native_r9700/kernels/llama_rmsnorm_zero_store_f16.cpp`. It keeps the RMSNorm 32-byte kernarg ABI exactly:

- `hidden_input` at byte 0 (`uint64`)
- `scale` at byte 8 (`uint64`)
- `hidden_output` at byte 16 (`uint64`)
- `epsilon` at byte 24 (`float32`)

The kernel uses the existing stage-0 launch shape (one 64-lane workgroup for this one 2048-element trace row), ignores its input/scale/epsilon payloads, and stores 2048 fp16 zero bit patterns through `hidden_output`.

The image and manifest are in `native_r9700/kernels/llama-rmsnorm-zero-store-hsa-assets/`. The reviewed image digest is `8be1b744e76cab295943e9a78b7cabdfd20d6e22c16f92862baf140f27b1de47`.

## Isolation

`--rmsnorm-zero-store` is accepted only for the layer-0, position-0 `normalized` trace with all three existing diagnostic preconditions:

- `--rmsnorm-unit-scale`
- `--rmsnorm-zero-input`
- `--rmsnorm-output-sentinel`

It swaps only stage 0 of `build_llama_layer0_stage_trace_dispatch`; `build_llama_persistent_dispatch` and the production `llama_rmsnorm_f16` stage table remain unchanged. The trace output includes `rmsnorm_kernel`, and a nonfinite failure artifact retains the existing PM4/buffer evidence while recording that kernel identity.

## Hardware command

```sh
build/native-r9700-runtime/native_r9700_runner --llama-stage-trace --model <model-dir> --token-id 0 --layer 0 --position 0 --stage normalized --trace-dir <empty-trace-dir> --rmsnorm-unit-scale --rmsnorm-zero-input --rmsnorm-output-sentinel --rmsnorm-zero-store
```

A passing diagnostic reports `rmsnorm_kernel:"llama_rmsnorm_zero_store_f16"`, the image digest above, `finite_count:2048`, and a 4096-byte trace payload of fp16 zeroes. That proves the trace PM4/output mapping and directs repair to RMSNorm arithmetic/code. A continued `trace_nonfinite` failure publishes the existing failure artifact with the zero-store kernel identity and leaves the asset/PM4 convention suspect.

## Contracts added

- `tests/native_r9700/test_llama_rmsnorm_zero_store_asset.py` pins source behavior, exact manifest schema, digests, trace-only selection, CLI admission, and excludes persistent-prefill substitution.
- `tests/native_r9700/test_hsa_code_image_generator.py` pins the reviewed generator registration and ABI admission.

No validation was run, per assignment.
