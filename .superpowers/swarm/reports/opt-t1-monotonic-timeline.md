# T1 — Monotonic timeline value in the PM4 dispatch

## Status
Done. Two plan corrections were required (documented below); the delivered
change preserves the frozen 3-arg 59-dword stream byte-for-byte.

## Changed files and symbols

1. `native_r9700/amdev_packets.h`
   - `Pm4DispatchConfig` gains `uint32_t timeline_value = 1;` (default `1`).
   - Placed as the **last** member (after `global_z`), see "Plan corrections".

2. `native_r9700/amdev_packets.cpp`
   - `build_pm4_dispatch_words(const Pm4DispatchConfig&)` now emits
     `config.timeline_value` in the RELEASE_MEM packet3 value slot (the 5th
     payload dword), replacing the hard-coded `kReleaseMemTimelineValue`.
   - The 3-arg overload `build_pm4_dispatch_words(uint64_t,uint64_t,uint64_t)`
     is **unchanged**; its 13-field positional aggregate initializer still binds
     identically, and `timeline_value` takes its default `1`.

3. `tests/native_r9700/test_pm4_timeline_contract.py` (new)
   - `test_timeline_value_is_written_into_release_mem_payload`: config with
     `timeline_value = 0x2A` emits `0x2A` in the RELEASE_MEM value slot.
   - `test_frozen_three_arg_overload_still_emits_value_one`: the 3-arg overload
     still emits value `1`.

## Plan corrections (required for correctness)

1. **Field placement: end of struct, not after `timeline_va`.**
   The plan asserts the 3-arg overload "already relies on the default
   `timeline_value = 1`". It does not — it passes all 13 fields positionally:
   `{code_va, kernargs_va, timeline_va, kKernelReferenceRsrc1, ...}`. Four
   additional sites in `native_r9700/amdev_session.cpp` (lines 1206, 1551,
   1793, 2296) do the same. Inserting `timeline_value` between `timeline_va`
   and `rsrc1` would silently shift every subsequent field, making the 3-arg
   overload emit `0xc00c0040` as the timeline value and corrupting every
   `amdev_session.cpp` dispatch config. Placing the field last keeps all 13
   positional initializers binding unchanged and lets `timeline_value` take its
   default `1`, satisfying both "do not change the 3-arg overload" and
   "preserve the frozen stream byte-for-byte". Task 3 consumes the field by
   name (`pm4.timeline_value = ...`), so position does not affect it.

2. **Test: opcode detection.** The plan wrote `(w >> 30) == 0x46`. `w >> 30`
   is the packet *type* (always `3` for packet3), and `0x46` is EVENT_WRITE.
   The RELEASE_MEM opcode is `0x49`, located at `(w >> 8) & 0xFF`. The plan's
   expression matches nothing, so the test would fail with
   "RELEASE_MEM packet not found". Corrected to
   `(w >> 8) & 0xFF == 0x49`.

3. **Test: value slot index.** The plan asserted `words[idx + 4]`. RELEASE_MEM
   payload order is `event, data_sel, lo32(timeline_va), hi32(timeline_va),
   value, 0, 0`, so the value is `header + 5` (payload dword 4). `idx + 4`
   reads `hi32(timeline_va)` (always `0` for the test VAs), so the assertion
   would fail. Corrected to `header + 5`.

## Note

`kReleaseMemTimelineValue` (constexpr, amdev_packets.cpp:50) is now unused
inside `amdev_packets.cpp` but retained per the plan (the new field's comment
references it as the historical value). It compiles cleanly under the test's
`-std=c++17 -O2` flags. Flagging in case the supervisor prefers it removed.

## Supervisor verification command (do NOT run)

```
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3; $PY -m pytest tests/native_r9700/test_pm4_timeline_contract.py -q
```
