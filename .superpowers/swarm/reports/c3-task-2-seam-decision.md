# C3 task set 2 - Backend seam decision

Status: **Done**.

## Decision

Choose **no direct backend yet / defer C3 implementation**.

Keep the C1/C2 imported prompt-cache producer path as the stable product path and review boundary. Do not start a direct `mlx-lm` backend, direct oMLX backend, or shared backend layer in C3.

## Grounds

- C2 correctness/fallback/security gates are green.
- C3 measurement artifact: `logs/c3-evidence/c3-evidence-result.json`.
- Prompt-cache import/validation is not the bottleneck: prompt-1 `0.532 ms`, prompt-2 `0.603 ms`.
- KV cache emit subprocess is visible but secondary: prompt-1 `76.677 ms`, prompt-2 `74.776 ms`.
- Final-token decode after import is small relative to producer work: prompt-1 `41.389 ms`, prompt-2 `41.547 ms`.
- Current producer prefill subprocess/model path dominates: prompt-1 `1,487.333 ms` at `S=222`; prompt-2 `2,696.293 ms` at `S=661`.
- Native `mlx-lm` full-prompt baseline for the same prompts is already ~`56-59 ms`; a cache-import optimization cannot make the current producer path competitive.

## Candidate evaluation

### mlx-lm direct backend first

Rejected.

Local `mlx-lm` 0.31.3 exposes a stable prompt-cache injection seam (`make_prompt_cache`, `load_prompt_cache`, `generate_step` with S-1 cache plus final token), but not a narrow R9700 backend hook. A true direct backend would have to replace or bypass model prefill paths in `generate_step`, `PromptProcessingBatch`, and Llama attention/model forward code. That is a backend rewrite, not a seam.

A smaller in-memory cache construction seam would remove only safetensors emit/import overhead (~75 ms + ~0.6 ms) and would leave the measured 1.49-2.70 s producer prefill path dominant. If it stopped writing prompt-cache artifacts, it would demote ADR 0001 and require task set 3 before implementation.

Scout: `agent://C3MlxSeamScout`.

### oMLX direct backend first

Rejected.

Local oMLX wraps `mlx-lm` and has an imported-cache/scheduler insertion shape around `BatchGenerator`, `PromptProcessingBatch`, and `make_prompt_cache`; it does not expose an AMD/R9700/TinyGPU backend hook. Local native kernels are MLX/Apple Metal-oriented. oMLX integration would be a larger consumer surface before a performance-relevant producer fix.

An oMLX imported-cache seam may be valid future work, but it preserves the same prompt-cache boundary and does not attack the measured producer subprocess/model bottleneck unless paired with a resident/in-process native producer.

Scout: `agent://C3OMLXSeamScout`.

### Shared backend layer behind mlx-lm and oMLX

Rejected for C3.

A shared layer would need to abstract batch-generation, cache merge/filter/extract semantics, and model-forward ownership across `mlx-lm` and oMLX. That increases validation scope before evidence shows direct backend work is justified.

### No direct backend yet

Selected.

This keeps the only proven boundary green: serialized `mlx-lm` prompt-cache artifacts with C1 parity and C2 serving behavior. It also keeps review/security artifacts local and inspectable.

## Task set 3 requirement

No ADR/design update is required for this decision because no durable boundary changes. ADR 0001 remains active: serialized KV prompt-cache is the Path A / first Path C boundary. Task set 3 becomes required only if a future decision bypasses, retires, or demotes prompt-cache artifacts on a fast path.

## Next action

Close C3 as deferred/no-prototype. Recommended follow-on work, if desired, is not a direct consumer backend; it is a separately-scoped resident producer or in-process producer measurement that preserves prompt-cache artifacts until it proves correctness and performance.
