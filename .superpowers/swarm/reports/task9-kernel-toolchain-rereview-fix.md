# Task9 kernel-toolchain rereview fix

## Documentation corrections

- `task9-kernel-toolchain-review-red.md` now records the `.rodata` and
  provenance failures as historical RED evidence observed before the
  implementation fix, and names the later focused-suite supervisor GREEN
  evidence.
- `task9-kernel-toolchain-discovery.md` now selects the Tinygrad checkout with
  `NATIVE_R9700_TINYGRAD_ROOT`, defaulting to the workspace-relative
  `../tinygrad` checkout rather than presenting a developer-specific path as
  universal.

No generator or test behavior changed.

## Supervisor verification command (not run in this task)

```sh
git diff --check
```
