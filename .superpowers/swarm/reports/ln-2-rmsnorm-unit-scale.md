# LN-2 RMSNorm unit-scale probe

## Scope

`--llama-stage-trace` now accepts the explicit diagnostic-only suffix
`--rmsnorm-unit-scale`. It is rejected unless the existing bounded trace is
layer 0, position 0, and the `normalized` boundary. The trace request keeps
this option separate from native prefill; it has no NPZ/cache path and cannot
change accepted artifacts.

Before resident preparation, the trace worker replaces only buffer 1, the
binder-validated `model.layers.0.input_layernorm.weight` upload, with exactly
2048 little-endian F16 `1.0` values (`00 3c`). The hidden input, output buffer,
epsilon kernarg, HSA image, resource descriptors, and dispatch geometry remain
unchanged.

## One-command discriminator

Run the same layer-0 token-id used for the failed normalized trace, with a new
trace directory:

```sh
APL_REMOTE_SOCK=<tinygpu-socket> build/native-r9700-runtime/native_r9700_runner \
  --llama-stage-trace --model <mlx-model-dir> --token-id <failed-token-id> \
  --layer 0 --position 0 --stage normalized --trace-dir <fresh-unit-scale-trace-dir> \
  --rmsnorm-unit-scale
```

A finite normalized result records `"scale_source":"unit_f16_one"` in the
success JSON. A non-finite result publishes the same field in the metadata-only
`layer0-token0-normalized.failure.json`. If the unit-scale output becomes
finite, the live uploaded scale payload is implicated. If it remains non-finite,
the failure remains downstream of that payload (epsilon, RMSNorm output path,
or asset/dispatch state).

## Focused contracts

- CLI rejects `--rmsnorm-unit-scale` for any non-`normalized` boundary before
device setup.
- The trace fault harness proves the helper overwrites all 4096 scale bytes as
F16 one while preserving the embedding upload byte-for-byte.
- The non-finite diagnostic harness requires
`"scale_source":"unit_f16_one"`, preserving the existing
`complete_nonfinite_trace` publication and cleanup coverage.

Validation was intentionally not run, per assignment.
