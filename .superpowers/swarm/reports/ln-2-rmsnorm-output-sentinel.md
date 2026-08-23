# LN-2 RMSNorm output-sentinel probe

## Scope

`--rmsnorm-output-sentinel` is a trace-only `--llama-stage-trace` discriminator. It is accepted only for the `normalized` boundary when both `--rmsnorm-zero-input` and `--rmsnorm-unit-scale` are present. `LlamaStageTraceRequest` remains separate from native prefill and does not carry an NPZ destination.

## Sentinel setup

Before resident preparation, the probe verifies buffer index 11 is the 4096-byte `layer0.normalized` scratch allocation with no existing upload. It uploads exactly 2048 little-endian fp16 `1.0` words (`0x3c00`) to that buffer. No input, scale, or other resident buffer is initialized by this helper.

## Trace interpretation

Successful trace JSON and CLI result output include `output_initialization: "sentinel_f16_one"`. They continue to publish the actual post-dispatch raw bytes, SHA-256, and `finite_count`:

- all zero fp16 words: the RMSNorm kernel overwrote every output element;
- retained fp16 `1.0` words or a mixture: the output store was absent, addressed incorrectly, or incomplete;
- non-finite output: the existing fail-closed nonfinite diagnostic remains in effect and now records the same output-initialization provenance.

## Validation

Not run, per the assignment's no-validation constraint (including no executor, git, or hardware invocation).
