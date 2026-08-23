# LN-2 RMSNorm unit-scale review

## Verdict: PASS

No severity findings.

## Review evidence

- **Scope guard:** `runner.cpp:315-350` accepts the suffix only for
  `--llama-stage-trace`, and rejects every non-`normalized` stage before it
  constructs a runtime request. `run_llama_stage_trace` repeats the same guard
  at `runtime_contract.cpp:842-846`, so direct runtime callers cannot dispatch
  the override at another boundary.
- **Binding and bytes:** the trace builder orders the selected embedding row at
  buffer 0 and `input_layernorm` at buffer 1 (`llama_layer_executor.cpp:188-197,
  247-250`); RMSNorm stage 0 binds buffer 1 at kernarg offset 8. The replacement
  accepts only that exact named, 4096-byte upload and overwrites all 2048 values
  as explicit little-endian F16 one bytes `00 3c`
  (`runtime_contract.cpp:276-301`). It changes no other dispatch buffer.
- **Truthful metadata:** successful trace JSON and the non-finite diagnostic
  both derive `scale_source` from the selected request value. The unit probe
  records `unit_f16_one`; the unmodified path records `model_f16`
  (`runtime_contract.cpp:304-305, 831, 938-940, 1023-1033`).
- **No accepted/full-prefill effects:** the override is applied only to the
  local stage-trace dispatch after trace construction (`runtime_contract.cpp:884-896`).
  `run_native_prefill` remains separate and is not called from the trace path.
  A non-finite readback keeps the result failed and publishes only the distinct
  `.failure.json` diagnostic, never raw/accepted trace output or NPZ
  (`runtime_contract.cpp:969-984`). Collision checks also reserve both
  diagnostic paths (`864-881`).
- **Diagnostic fault behavior:** the failure-publication seam preserves staged
  write, file fsync, rename, parent fsync, cleanup, and failed-cleanup reporting
  (`runtime_contract.cpp:462-492, 547-572`). The expanded harness covers each
  injected failure and asserts no accepted artifact fields/output.

No tests, build, or hardware validation were run, per assignment.
