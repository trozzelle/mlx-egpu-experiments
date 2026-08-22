# LQ-W0-2 Qwen binder reviewer fixes

## Changed files

- `native_r9700/qwen_weight_binder.h`
- `native_r9700/qwen_weight_binder.cpp`
- `tests/native_r9700/test_qwen_text_adapter.py`
- `.superpowers/swarm/reports/lq-w0-qwen-fix.md`

## RED contract

- The focused C++ probe validates a complete, same-file affine raw triplet using `QwenWeightBinder::validate` without supplying a metadata output object.
- The probe changes only the scales span to a second nonempty safetensors source file and requires loud rejection mentioning the source file; it then restores the shared source file, aliases the scales byte range with the weight range, and requires the existing overlap rejection.

## Decisions

- `validate` is a validate-only, caller-owned API: it neither copies nor retains string-bearing raw metadata. Its successful path directly compares the literal Qwen layer namespace, decimal layer digits, suffixes, stems, source-file identities, and byte ranges.
- Validation preserves the prior affine mode/bits/group size, 64-layer boundary, bounded-window overflow, exact layer identity, nonempty-span, identity-stem, and overlap checks. It adds the required one-source-file invariant before a bounded device window can be formed.
- No safetensors payload is accessed, no device memory is allocated, and no numerical computation is performed.

## Supervisor validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_qwen_text_adapter.py -q
```
