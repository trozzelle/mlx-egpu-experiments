# C3 task set 1 — C2 evidence intake and backend justification

Status: **Done**.

## Source evidence

- C2 integration result: `logs/c2-serving/result.json` — `gate_result=pass`, `exit_status=0`, `prompt_count=3`.
- C2 run log: `logs/c2-serving/run.log`.
- C2 report section: `docs/path-a-validation-results.md` §Path C2.
- C2 security gate: `.superpowers/swarm/reports/c2-task-7-security-review.md`; final reviewer `agent://C2SecurityReReview` approved with 0 findings.
- C3 measurement pass: `logs/c3-evidence/c3-evidence-result.json` and `logs/c3-evidence/c3-evidence-run.log`.

## C2 serving evidence

C2 passes correctness and fallback gates:

| Prompt | S | Route | Result |
|---|---:|---|---|
| prompt-0 | 6 | `native_mlx_fallback` | below threshold; baseline exact |
| prompt-1 | 222 | `native_producer` | accepted cache; baseline exact |
| prompt-2 | 661 | `native_producer` | accepted cache; baseline exact |

Producer-unavailable smoke passes with `route=native_mlx_fallback`, `fallback_reason=producer_failed`, `accepted_cache=false`, `prompt_cache_path=null`, baseline exact, and `exit_status=0`.

## C3 timing evidence

Measured locally against `../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct` with the model loaded once in-process for mlx-lm baseline/imported decode, and C2-style subprocess/file handoff for producer work. Commands and C2-created prefill logs redact raw token IDs.

| Prompt | S | Cache bytes | Native mlx-lm full prompt decode | Producer prefill subprocess | KV cache emit subprocess | Cache import+validate | Imported final-token decode | Imported path total, excluding model load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prompt-1 | 222 | 7,244,672 | 58.502 ms | 1,487.333 ms | 76.677 ms | 0.532 ms | 41.389 ms | 1,605.931 ms |
| prompt-2 | 661 | 21,629,864 | 56.508 ms | 2,696.293 ms | 74.776 ms | 0.603 ms | 41.547 ms | 2,813.218 ms |

Model load for the measurement process: 2,293.979 ms.

Correctness during measurement:

- prompt-1: native full-prompt tokens and imported-cache tokens both matched baseline R tokens exactly.
- prompt-2: native full-prompt tokens and imported-cache tokens both matched baseline R tokens exactly.

## Bottleneck assessment

Observed bottleneck is not prompt-cache import/validation: it is ~0.5–0.6 ms. Final-token decode after import is ~41.5 ms. KV cache emit subprocess is ~75 ms. The current serialized producer path is dominated by per-request producer prefill subprocess time: ~1.49 s at S=222 and ~2.70 s at S=661.

Native mlx-lm full-prompt decode for the same prompts is ~56–59 ms in this measurement, so the current C2 path is correctness-valid but not performance-competitive. A direct native consumer backend could plausibly remove serialization and consumer import overhead, but the measured overhead to attack first is process/model invocation and producer implementation cost, not the KV interchange import itself.

## Recommendation for task set 2

Do **not** start a broad direct mlx-lm/oMLX backend prototype from this evidence alone. The safer C3 decision is to defer direct backend implementation and keep the imported-cache producer path as the stable correctness boundary while measuring/considering a narrower resident-producer optimization path first:

1. preserve prompt-cache artifacts as fallback/review output;
2. avoid demoting ADR 0001 KV interchange boundary in C3;
3. record that direct backend remains plausible only after a resident native producer or more granular timing proves serialization/import, not producer execution, is the limiting bottleneck;
4. skip C3 prototype task set 4 unless task set 2 intentionally accepts the larger backend-risk tradeoff.

This gives task set 2 enough evidence to choose **no direct backend yet / defer C3 implementation** without blocking future backend work.
