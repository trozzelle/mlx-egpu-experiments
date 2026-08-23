# LN-2 RMSNorm zero-input probe

## Scope

`--llama-stage-trace` accepts `--rmsnorm-zero-input` only when paired with
`--rmsnorm-unit-scale`. The existing trace gate still requires layer 0 and
position 0; both diagnostic overrides additionally require the `normalized`
boundary. The request type is separate from native prefill, has no NPZ/cache
path, and cannot produce an accepted prefill artifact.

Before resident preparation, the probe replaces only buffer 0,
`layer0.embedding_row`, with exactly 2048 F16 zero values (4096 zero bytes).
The unit-scale probe independently retains its buffer-1 replacement with 2048
F16 `1.0` values. RMSNorm epsilon kernargs, output buffer, HSA image,
resource descriptors, and dispatch geometry are unchanged.

## One-command discriminator

Run the same model token used for the failed normalized trace with a fresh
trace directory:

```sh
APL_REMOTE_SOCK=<tinygpu-socket> build/native-r9700-runtime/native_r9700_runner \
  --llama-stage-trace --model <mlx-model-dir> --token-id <failed-token-id> \
  --layer 0 --position 0 --stage normalized --trace-dir <fresh-zero-input-trace-dir> \
  --rmsnorm-unit-scale --rmsnorm-zero-input
```

A finite result records both `"scale_source":"unit_f16_one"` and
`"input_source":"zero_f16"`. Under a valid positive epsilon, zero input
and unit scale must produce finite fp16 zero output. A non-finite result still
publishes metadata-only `layer0-token0-normalized.failure.json`, now carrying
both source discriminators with the existing kernarg, buffer, image, PM4, and
failure information.

If this probe remains non-finite, model-hidden delivery and scale contents are
ruled out; investigate RMSNorm execution/asset argument handling or output
storage. A finite zero result instead isolates the failure to model-backed
input delivery.

## Focused contracts

- CLI rejects zero input without unit scale before any device setup.
- Request/result metadata requires `rmsnorm_zero_input` and `input_source`.
- The trace fault harness proves buffer 0 becomes exactly 4096 zero bytes while
  buffer 1 remains unit F16, and verifies non-finite diagnostics preserve both
  source fields.
- Trace dispatch remains bounded and contains no prefill serialization path.

Validation was intentionally not run, per assignment.
