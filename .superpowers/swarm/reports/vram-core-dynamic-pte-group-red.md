# Dynamic PTE group RED contracts

## Selector

- `tests/native_r9700/test_dynamic_page_table_contract.py`

## Added contracts

- A map covering the entire resident VA window succeeds when its exclusive end is exactly `resident_gpu_va_limit`. It reserves only the PTB1 dynamic table page.
- A two-page map that crosses from the final fixed PTB0 leaf into PTB1 fails when PTB1 zeroing fails. The failure leaves the already mapped PTB0 leaf owned for explicit cleanup. Once the injected failure clears, unmapping the original range succeeds without any PTE operation against the unlinked PTB1 page, and the PTB0 leaf can be mapped again.

## Supervisor RED command (do not run in this task)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_dynamic_page_table_contract.py -q
```

## Intended RED state

The current mapper rejects a valid full-window range whose end equals the exclusive resident limit, and it does not retain enough cross-group map ownership for explicit cleanup after a later PTB zero failure. The supervisor command above is recorded and intentionally not run in this task.
