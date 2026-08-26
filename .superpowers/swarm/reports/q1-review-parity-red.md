# Q1 parity identity RED review

**Status:** RED contract added; supervisor validation pending  
**Owner:** `Q1ParityReviewRed`  
**Scope:** Qwen parity admission of the model directory against the pinned source/inventory identity.  
**Non-goals:** production implementation, fixture changes, source-pin generation, numerical comparison, model loading, and hardware execution.

## Changed test surface

Only `tests/native_r9700/test_qwen_parity.py` was extended. No production module or fixture artifact was changed. The new test invokes the public `compare_qwen_fixtures` production function with the committed Qwen fixture package and canonical inventory report.

## New RED contract

`test_qwen_parity_rejects_expected_basename_with_mismatched_source_identity` creates a real temporary model directory whose basename is exactly the frozen Qwen revision (`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`) but whose present `config.json` identifies a different model. It supplies otherwise valid committed fixture/inventory inputs and requires `QwenParityError` from `compare_qwen_fixtures`.

This is intentionally not a source-text assertion and does not patch or mock the validator. A revision-looking directory must be bound to the actual verified source-pin/inventory identity; matching the basename alone is insufficient. The production change that makes this pass is to validate the model directory's frozen metadata/source identity before accepting the parity report and to fail closed on sidecar drift.

## Expected RED cause

The current `compare_qwen_fixtures` implementation validates the fixture package and the inventory's top-level frozen fields, then admits `model_dir` solely when `Path(model_dir).name` equals the pinned revision. It never reads or verifies the model directory's metadata. Therefore this test reaches the current successful return despite the wrong `config.json` and fails with `DID NOT RAISE`, specifically exposing basename-only model validation.

## Supervisor focused command

Run exactly:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_parity.py -v
```

This lane did not run tests, builds, linters, formatters, package managers, model loads, hardware commands, or git commands.
