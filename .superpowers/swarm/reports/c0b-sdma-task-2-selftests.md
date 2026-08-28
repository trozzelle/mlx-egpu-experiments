# C0B SDMA Task 2 — Deterministic self-tests

## Status

Needs review. C++ deterministic SDMA helpers/self-tests added.

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`

## Interfaces produced

- `am_sdma` fixed queue/control-page constants.
- `build_sdma_fence_packet(...)`.
- `build_sdma_submit_words(...)`.
- `--self-test sdma-ring-setup`.
- `--self-test sdma-fence-packet-encoding`.
- `--self-test sdma-submit-sequence`.

## Supervisor command

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected GREEN: all no-hardware tests pass.

## Guardrails

No hardware command, package manager, formatter, linter, git command, project-wide suite, tinygrad runtime import/call, libusb path, generic queue scheduler, or production runtime API was added by this task agent.
