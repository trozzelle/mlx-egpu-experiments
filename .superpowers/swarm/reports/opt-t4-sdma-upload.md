# T4 — Hoist SDMA queue setup out of the weight-upload chunk loop

File: `native_r9700/amdev_session.cpp`

## Change

`setup_sdma_queue0` was previously re-programmed inside the per-1 MiB-chunk
`while (offset < byte_count)` loop in `ResidentHsaSession::upload_named`.
It is now executed at most once per prepared session, while the per-chunk
fence clear + submit + poll is preserved unchanged (required for correctness:
the SDMA fence value is fixed at 1, so a single fence cannot distinguish the
last chunk from the first).

### 1. `Impl` struct — new flag

After `bool prepared = false;`:

```cpp
  // SDMA queue-0 setup is programmed once per prepared session, not once per
  // 1 MiB upload chunk. Reset on close so a reused session re-programs once.
  bool sdma_queue_configured = false;
```

### 2. `upload_named` — one-time guard before the loop

Inserted after `std::string detail;` and before `while (offset < byte_count)`:

```cpp
  if (!state.sdma_queue_configured) {
    if (!setup_sdma_queue0(*state.client, &state.log, &detail)) return fail(detail);
    state.sdma_queue_configured = true;
  }
```

### 3. `upload_named` — loop body unchanged except the setup call removed

Before:

```cpp
    if (!setup_sdma_queue0(*state.client, &state.log, &detail) ||
        !submit_sdma_copy(*state.client, &state.log, &state.sdma_control_mapping,
                          state.staging.gpu_va, state.buffers[buffer_index].gpu_va + offset, chunk,
                          am_sdma::kFenceValue, 0, &detail) ||
        !poll_sdma_fence(state.sdma_control_mapping, &detail)) {
```

After:

```cpp
    if (!submit_sdma_copy(*state.client, &state.log, &state.sdma_control_mapping,
                          state.staging.gpu_va, state.buffers[buffer_index].gpu_va + offset, chunk,
                          am_sdma::kFenceValue, 0, &detail) ||
        !poll_sdma_fence(state.sdma_control_mapping, &detail)) {
```

The memcpy staging, `std::atomic_thread_fence(seq_cst)`,
`std::memset(..., am_sdma::kFenceOffset, 0, sizeof(uint32_t))` fence clear,
`submit_sdma_copy`, and `poll_sdma_fence` remain verbatim inside the loop.

### 4. `reset_after_close` — reset on teardown

After `prepared = false;`:

```cpp
    sdma_queue_configured = false;
```

`reset_after_close` is reached from both `close()` branches (early return when
`resident == nullptr`, and the normal path), and from the destructor via
`close()`, so a reused session re-programs SDMA once on its next upload.

## Acceptance

- `setup_sdma_queue0` is now called at most once per prepared session in the
  upload path, not once per 1 MiB chunk.
- Per-chunk fence clear + submit + poll remain intact.

## Supervisor verification (DO NOT run in this task)

Full runner build, then:

```
PY="${PY:?set PY to the pinned Python 3.12.8 interpreter}"; $PY -m pytest tests/test_native_amdev_transfer_contract.py -q
```
