# C0B SDMA Task 1 — Contract tests

## Status

Needs review. RED contract added; implementation is absent by design.

## Changed files

- `tests/test_native_amdev_transfer_contract.py`

## Expected RED

Supervisor should run:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected failure: the three new SDMA self-test modes and extended VM control-page output are absent until Task 2.

## Source grounding

- SDMA ring setup comes from tinygrad `runtime/support/am/ip.py` lines 497-556.
- SDMA submit/write-pointer/doorbell flow comes from `runtime/ops_amd.py` lines 524-560 and queue doorbell lines 679-688.
- SDMA fence packet comes from `runtime/autogen/am/sdma_6_0_0.py` lines 232-273 and field helpers around 2991-3042.
- SDMA HWID and doorbell constants come from `runtime/autogen/am/am.py` `SDMA0_HWID = 42` and `AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 = 256`.

## Guardrails

No C++ implementation, hardware command, validation command, package manager, formatter, linter, git command, or broad test suite was run by this task agent.
