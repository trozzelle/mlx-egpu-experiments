# C1R-W0-1 Retire Bridge Fallback

## Minimal cutover

- Removed the default source-build fallback and every runtime/runner entry point for primitive-chain and layer-0 proof execution.
- Replaced the former primitive proof wrapper with `RuntimeSession::legacy_primitive_diagnostic`, exposed only as `--legacy-primitive-diagnostic <name>`. It executes only `NATIVE_R9700_C1_PRIMITIVE_BRIDGE` when explicitly injected.
- No injection returns nonzero with `failure_stage: legacy_proof_unavailable`.
- `--native-prefill-proof` no longer invokes primitive diagnostics, reports the same failure stage, never reports `native_prefill_acceptance: pass`, and removes the requested NPZ.
- Preserved C0 lifecycle, kernarg, kernel-proof, transfer-proof, and Python CPU-reference paths.

## Changed files and symbols

- `native_r9700/runtime.h`: replaced primitive/chain/layer proof declarations with `legacy_primitive_diagnostic`; retained the fail-closed native-prefill API.
- `native_r9700/runtime.cpp`: implemented `RuntimeSession::legacy_primitive_diagnostic`; removed primitive-chain and native-layer0 bridge execution; simplified `RuntimeSession::native_prefill_proof` to the fail-closed cutover result.
- `native_r9700/runner.cpp`: removed product-facing primitive, chain, and layer0 modes/help; added `--legacy-primitive-diagnostic`.
- `tests/native_r9700/test_runtime_contract.py`: removed archive-derived primitive/chain/layer0 assertions; retained lifecycle/kernarg/C0/transfer tests; added injected legacy-diagnostic protocol coverage and the no-injection native-prefill contract.
- `docs/archive/tasks/native-r9700-producer/README.md` and `validation-commands.md`: mark the retired source archive forensic-only and document the explicit legacy diagnostic limitation.

## RED gate observed before this change

`test_native_prefill_proof_reports_legacy_proof_unavailable_without_primitive_bridge` was observed RED because no-injection native prefill reported an archive fallback failure rather than `failure_stage: legacy_proof_unavailable`.

## Supervisor commands

Supervisor GREEN command:

```sh
${PY} -m pytest \
  tests/native_r9700/test_runtime_contract.py -q
```

Supervisor C++ compile command:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
```

Per executor policy, these commands were not run by this worker.
