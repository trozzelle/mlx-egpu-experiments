# Native worker spine report

## Files changed
- `native_r9700/native_worker.py` — added fail-closed native prefill runner orchestration shell.
- `native_r9700/prefill.py` — preserved `cpu_reference` NumPy route; routed `r9700_native` CLI requests through `native_worker.run_native_prefill(...)`; prevented `prefill_prompt_prefix(..., producer_kind="r9700_native")` from relabeling CPU tensors.
- `tests/native_r9700/test_prefill.py` — added/updated fail-closed CLI contracts for `r9700_native`, CPU-reference masquerade rejection, and CPU route relabel prevention.
- `tests/native_r9700/test_runtime_contract.py` — added native-worker-only contracts for JSON parsing, key/value log parsing, hardware evidence validation, CPU masquerade rejection, and open acceptance on missing runner output.

## Contracts added
- Public worker seam: `native_worker.run_native_prefill(model_dir, token_ids, out_npz, log_path) -> dict[str, object]`.
- Required result fields: `producer_kind`, `native_prefill_acceptance`, `runtime_substrate`, `hardware_log_path`, `prefill_npz_path`, `kernel_count`, `transfer_bytes`, `failure_stage`, `exit_status`.
- Acceptance requires all of:
  - `producer_kind == "r9700_native"`
  - `native_prefill_acceptance == "pass"`
  - `exit_status == 0`
  - `prefill_npz_path` equals the requested output path and exists
  - `kernel_count > 0`
  - `transfer_bytes > 0`
- Any missing or mismatched acceptance evidence is rewritten/reported as `native_prefill_acceptance: open` and unaccepted output NPZ is removed.
- `cpu_reference` remains the only NumPy producer route and cannot satisfy native acceptance.

## No-validation policy
No tests, linters, formatters, package-manager commands, hardware commands, project-wide suites, or git commands were run, per Wave 32 executor constraints.

## Exact supervisor command to run
`${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_runtime_contract.py -q`

## Risks / follow-ups
- Default runner path is `native_r9700/runner`; until a real `--native-prefill-proof` runner binary/mode exists, `r9700_native` remains fail-closed with `native_prefill_acceptance: open`.
- The accepted path intentionally only trusts runner output with nonzero kernel and transfer evidence; Task 3+ must preserve these fields when writing real native NPZ artifacts.
- The worker removes the requested NPZ on failed acceptance to prevent stale accepted artifacts from masquerading as native output.

## Coordination messages
- Sent to `Layer0DataflowProof`: Task 1 would edit `native_r9700/native_worker.py`, `native_r9700/prefill.py`, `tests/native_r9700/test_prefill.py`, and only native-worker/fail-closed helper/tests in `tests/native_r9700/test_runtime_contract.py`; avoided layer0 proof regions.
- Received from `Layer0DataflowProof`: Task 2 would keep edits focused on native layer0 proof/runtime files and avoid the native-worker-only top-region tests.
- Sent to `Task3NPZScout`: shared the Task 1 public interface and exact fail-closed acceptance fields/conditions for Task 3 consumption.
