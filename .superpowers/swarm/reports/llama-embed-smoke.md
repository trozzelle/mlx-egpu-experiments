# Llama embedding-row smoke

The native `--llama-embed-smoke --model <dir> --token-id <uint32>` route is a selected-row hardware slice only. It canonicalizes and binds the supported Llama model before any TinyGPU connection, rejects out-of-vocabulary tokens, range-checks the selected safetensors span, and reads exactly one 4096-byte F16 embedding row with `pread`.

The route loads the attested generated HSA image from `native_r9700/kernels/llama-hsa-assets` without source compilation. The physical path reserves the lower-BAR dynamic page-table boundary, maps resident image/row/hidden/selected-row buffers, uploads the real row and zero selected-row scalar through SDMA, dispatches the HSA entry with three 64-bit kernargs, downloads hidden output, and compares the full 4096-byte result exactly with the source row.

Every invocation emits the model/span, HSA, resident-VA/physical-offset, dynamic-PTB, PM4, SDMA, byte-comparison, and first-failure evidence to stdout and a timestamped `logs/llama-embed-smoke-*.log`. This slice does not claim prefill acceptance: `native_prefill_acceptance: open` is emitted on both success and failure.

No hardware command was run by this implementation task.
