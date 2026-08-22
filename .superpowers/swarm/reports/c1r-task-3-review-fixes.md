# C1R-3 review fixes

Status: Done

Findings addressed:
- Runtime marker: changed C1 wrapper/tests back to the frozen C0 marker `TinyGPU.app/APLRemotePCIDevice/PCIIface`; source evidence is the C0 logs and `native_r9700/runtime.h::kRuntimeSubstrate`.
- Trace fixture artifact: `tests/native_r9700/fixtures/layer_trace_fixtures.npz` exists in the worktree and is referenced by schema/tests; it must remain part of the patch with the JSON/test edits.
- Clean checkout kernel proof: `RuntimeSession::kernel_proof` now creates `logs/` before compiling the frozen C0 probe to `logs/native-r9700-c0a25-probe`.

Verification:
- `python3 -m pytest tests/native_r9700/test_runtime_contract.py -q` -> 11 passed.
- `python3 -m pytest tests/test_native_amdev_transfer_contract.py -q` -> 23 passed.
- `python3 -m pytest tests/native_r9700 -q` -> 128 passed, 2 warnings.
