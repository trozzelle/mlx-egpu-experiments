# F1 persistent serving GREEN implementation

**Task set:** F1 task set 4 — persistent serving consumer route
**Owner:** `F1ServingGreen`
**Status:** Implemented; supervisor validation pending

## Scope

Updated only `native_r9700/serving.py` and this report. The existing one-shot subprocess/control path remains selected when `service_dispatch` is absent. The persistent path is selected only for above-threshold prompts when a live dispatcher callable is supplied.

## Implemented behavior

- Issues public `LoadModel`, `Prefill`, and `UnloadModel` envelopes through the supplied dispatcher, with distinct lifecycle request IDs and the complete public prompt token list in `Prefill`.
- Sends the pinned file cache specification and bounded request timeout; does not invoke `subprocess.run` on the persistent route.
- Consumes the public cache projection and evidence, including service-owned NPZ/cache/log paths and producer fingerprint.
- Validates the loaded model fingerprint plus cache schema, producer/model/request identity, exact 16-layer geometry, batch and `S-1` offset/sequence/absolute positions, canonical RoPE, `float16`/`B,H,S,D` layout, `KVCache` class/variant, and exactly sixteen empty per-layer metadata values (`meta_state` or flattened `0.0`–`0.15`). Both the service descriptor and loaded prompt-cache metadata are checked before acceptance.
- Uses the existing full-prompt fallback only for pre-acceptance producer/service/cache identity rejection. Public child/device failures are terminal; decode failures after cache acceptance raise `NativePrefillError` without a second Prefill or full-prompt retry.
- Accepted caches reach `generate_step` with only the final prompt token and the imported cache. Result evidence and typed identity projections are retained in the serving result.

## Focused supervisor command

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_serving.py -v
```

Per assignment, this worker did not run tests, builds, linters, formatters, package managers, hardware commands, or git operations. The supervisor owns focused validation and the combined F1 gate.
