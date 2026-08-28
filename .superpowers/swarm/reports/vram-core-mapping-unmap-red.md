# VRAM resident unmap-failure RED contract

## Selector

- `tests/native_r9700/test_resident_memory_contract.py`

## Fail-closed cleanup contract

The in-memory page-mapper callback can now independently fail a selected `kUnmap` attempt while preserving the stale GPU-VA-to-physical-page mapping. This exercises the real `ResidentMemory` planner outcome rather than callback call counts.

A two-page allocation whose second `kMap` fails and whose rollback `kUnmap` also fails must leave the entire attempted physical allocation and GPU-VA reservation quarantined. After removing only the map-failure injection, a later independent two-page allocation must succeed in nonoverlapping physical and VA ranges, while the mapper still contains the stale first-page mapping. Retrying unmap is not part of this contract.

Likewise, a one-page allocation whose `release_all()` `kUnmap` fails must retain that physical allocation and VA reservation. Once the injection is removed, a later one-page allocation must succeed at nonoverlapping physical and VA ranges while coexisting with the stale mapping.

The existing successful rollback and successful `release_all()` probes remain unchanged: successful cleanup still deterministically reuses the earliest released physical and VA range.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_resident_memory_contract.py -q
```

## Intended RED state

The current planner reclaims physical and VA ownership even after the injected `kUnmap` callback failure. Its next allocation therefore attempts to reuse the stale range; the mapper rejects that stale GPU VA, so the new allocation cannot establish the required independent range. The supervisor command above was recorded and intentionally not run in this task.
