# C0A compute Task 1 contracts report

## Changed files
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-1-contracts.md`

## Supervisor validation command to run later

```sh
cd <former-native-r9700-worktree> && ${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Expected RED reason
The focused pytest run is expected to fail because production C++ does not yet implement the new no-hardware self-tests: `compute-vm-layout`, `gfx-ring-registers`, `compute-mqd-encoding`, and `pm4-dispatch-sequence`.

## Non-goals respected
- No production C++ changes.
- No hardware command run.
- No validation command edits.
- No tinygrad runtime dependency added.
- No scheduler, allocator, or runtime framework work.
- No C1/C2/C3 work.

## Supervisor RED result
`${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` exited `1`.

Observed expected failures: 5 failed, 11 passed. The four new self-tests returned `unknown self-test`, and help output did not yet list `compute-vm-layout`, `gfx-ring-registers`, `compute-mqd-encoding`, or `pm4-dispatch-sequence`.
