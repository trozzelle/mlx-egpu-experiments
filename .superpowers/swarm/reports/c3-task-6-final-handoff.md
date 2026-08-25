# C3 task set 6 - Final decision and handoff

Status: **Reclassified by ADR 0005**.

## Corrected final state

This handoff is reclassified by ADR 0005. The selected product path remains the C1/C2 imported
prompt-cache producer path, but the implementation available at the time of this report is
CPU/NumPy reference work, not a completed native R9700/eGPU producer.

Reference evidence that remains valid:

1. `native_r9700.prefill` exports the validated `S-1` prompt prefix K/V as a CPU/NumPy reference.
2. `native_r9700.kv_cache` emits mlx-lm-compatible prompt-cache artifacts.
3. `native_r9700.serving` validates the complete prompt-cache HBI before acceptance, supports
   fallback/security behavior, and lets mlx-lm consume imported cache artifacts.

Original objective state: C1R and C2R remain open until model-forward prefill tensor work executes on
the R9700/eGPU and C2 large-prompt serving uses that accepted producer route.

## Evidence

- C2 integration: `logs/c2-serving/result.json`, `logs/c2-serving/run.log`, `docs/path-a-validation-results.md` Path C2 section.
- C2 fallback smoke: `logs/c2-serving-unavailable/result.json`, `logs/c2-serving-unavailable/run.log`.
- C2 security review: `.superpowers/swarm/reports/c2-task-7-security-review.md`; re-review approved with 0 findings.
- C3 measurement: `logs/c3-evidence/c3-evidence-result.json`, `logs/c3-evidence/c3-evidence-run.log`.
- C3 evidence report: `.superpowers/swarm/reports/c3-task-1-evidence.md`.
- C3 seam decision: `.superpowers/swarm/reports/c3-task-2-seam-decision.md`.
- C3 seam scouts: `agent://C3MlxSeamScout`, `agent://C3OMLXSeamScout`.
- C3 final review: `agent://C3FinalReview` initially returned CHANGES_REQUIRED because this handoff was missing and the phase handoff note was stale; both findings were addressed by this closure pass.

## Closure rationale

C3 timing shows the current per-request producer prefill subprocess/model path is the bottleneck, not the prompt-cache interchange:

| Prompt | S | Native mlx-lm full decode | Producer prefill subprocess | KV emit subprocess | Cache import+validate | Imported final-token decode |
|---|---:|---:|---:|---:|---:|---:|
| prompt-1 | 222 | 58.502 ms | 1,487.333 ms | 76.677 ms | 0.532 ms | 41.389 ms |
| prompt-2 | 661 | 56.508 ms | 2,696.293 ms | 74.776 ms | 0.603 ms | 41.547 ms |

Direct `mlx-lm` backend first was rejected because local `mlx-lm` exposes a stable prompt-cache injection seam but no narrow R9700 backend hook. A true backend would replace model prefill/batch/model-forward paths and would require broad validation.

oMLX direct backend first was rejected because local oMLX wraps `mlx-lm`; its feasible seam is an imported-cache/scheduler insertion shape, not an AMD/R9700/TinyGPU backend hook.

A shared backend layer was rejected because it expands validation across both `mlx-lm` and oMLX cache/batch semantics before direct backend evidence is justified.

## Task set outcomes

- C3-1: Done. Evidence intake and timing report complete.
- C3-2: Done. Backend seam decision recorded: defer direct backend.
- C3-3: Not required. No prompt-cache boundary change.
- C3-4: Dropped/Deferred. No direct backend prototype.
- C3-5: Dropped/Deferred. No prototype comparison; task set 1 timing evidence is the comparison basis.
- C3-6: Done. Final handoff and ledgers updated.

## Next action

Do not start C3 backend implementation from this evidence. Execute
`docs/archive/tasks/native-r9700-producer/phase-c1-c2-r9700-recovery-plan.md` first.

Candidate follow-on questions only after C2R passes with real R9700/eGPU prefill:

1. Can a resident R9700/eGPU producer avoid per-request model/config load and subprocess startup
   while still writing the same prompt-cache artifacts?
2. Can the NPZ -> safetensors conversion be folded into a single resident producer service without
   losing reviewable artifacts?
3. Does any resident path preserve C1R parity, C2R fallback/security semantics, and redacted logs?

Do not bypass, retire, or demote serialized prompt-cache artifacts without reopening the boundary
decision and writing the required design/ADR update.

## Verification

Final verification command:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v && git diff --check docs/archive/tasks/native-r9700-producer/phase-c3-native-backend-decision.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c3-task-1-evidence.md .superpowers/swarm/reports/c3-task-2-seam-decision.md .superpowers/swarm/reports/c3-task-6-final-handoff.md
```

Result: `pytest: 119 passed, 2 warnings in 10.93s`; command exited 0, so `git diff --check` also passed with no output.
