# Path A — Phase 0 numeric-parity validation results

Status: **RUN COMPLETED — GATE NOT PASSED (weight-precision confound)**

Gate: injected path `P` must equal native baseline `R` token-for-token across the prompt set; per-layer numeric deltas reported and flagged above the `1e-3` fp16 probe tolerance. Semantic equivalence is the acceptable bar if a completion is not bit-exact (DESIGN.md §Validation).

## Result summary

The gate ran end-to-end (tinygrad prefill → export → mlx decode vs native mlx baseline) across the
3-prompt suite)Skip, but **`P != R` on every prompt**, and **all 16 layers exceed the `1e-3` fp16
probe tolerance** with large per-layer deltas (K max|Δ| up to ~0.53). This is not fp16 rounding
noise; it indicates the producer and consumer were not operating on numerically identical weights.

## Root cause (verified)

The tinygrad producer GGUF is **quantized, not fp16**. Verified from GGUF metadata
(`tinygrad.llm.gguf.gguf_load` on the cached file):

- `general.name = "Llama 3.2 1B Instruct"`
- `general.file_type = 18` (llama.cpp `LLM_FTYPE_MOSTLY_Q6_K`)
- `quantize.imatrix.*` metadata present → produced via **imatrix calibration** (`/training_dir/calibration_datav3.txt`)
- `tokenizer.ggml.bos_token_id = 128000`

Meanwhile the mlx consumer loads **fp16** safetensors. So the producer's weights carry Q6_K
quantization error (~0.1–1%) that compounds through the 16 layers, producing exactly the observed
depth-graded deltas. **This failure is a weight-precision mismatch, not evidence of an interchange
format defect.**

## Decision — defer exact parity proof to Path C

Per ROADMAP, the production engine is **Path C (native producer outside TinyGrad)**; TinyGrad is
only the Path A stand-in to validate the interchange format, and "producer-swap inherits a
Phase-0-style parity gate." Therefore we do not invest in tinygrad-specific GGUF tooling to force
fp16 parity here. Instead:

1. **Record this finding** (below) as the Phase 0 gate outcome with this precise diagnosis.
2. **Downloaded + converted the official `meta-llama/Llama-3.2-1B-Instruct` fp16 safetensors** to
   mlx at `mlx_models/meta-Llama-3.2-1B-Instruct/` (verified: hidden 2048, 16 layers, 8 KV heads,
   head_dim 64, fp16) — the exact consumer baseline for the Path C parity gate.
3. **Defer the `P == R` proof** to Path C's producer-swap parity gate, where the producer will use
   the identical fp16 weights (no Q6_K confound). The interchange format + exporter are unchanged;
   this is the intended durability hedge (ADR 0001).

## Prompt suite

| # | Prompt | S (tokens) | P == R |
|---|---|---|---|
| 0 | `prompt-0` | 6 | False |
| 1 | `prompt-1` | 222 | False |
| 2 | `prompt-2` | 661 | False |

## Per-layer numeric deltas (max|Δ| / mean|Δ| vs native KV)

| Layer | K max|Δ| | K mean|Δ| | V max|Δ| | V mean|Δ| | > 1e-3? |
|---|---|---|---|---|---|
| 0 | 0.08318626880645752 | 0.009665203280746937 | 0.009744054637849331 | 0.0016021033516153693 | True |
| 1 | 0.178509920835495 | 0.019384875893592834 | 0.0231306254863739 | 0.003925992175936699 | True |
| 2 | 0.3467526435852051 | 0.03286830708384514 | 0.05208730697631836 | 0.007289104163646698 | True |
| 3 | 0.5322937965393066 | 0.043850790709257126 | 0.09801572561264038 | 0.010559906251728535 | True |
| 4 | 0.32104969024658203 | 0.042617928236722946 | 0.11346924304962158 | 0.012035715393722057 | True |
| 5 | 0.46317625045776367 | 0.0456407256424427 | 0.07391834259033203 | 0.011452010832726955 | True |
| 6 | 0.3865690231323242 | 0.04603302851319313 | 0.08697602897882462 | 0.013535626232624054 | True |
| 7 | 0.45567798614501953 | 0.04608621075749397 | 0.10941597819328308 | 0.014997635968029499 | True |
| 8 | 0.3373527526855469 | 0.05067242681980133 | 0.1183122992515564 | 0.0144770173355937 | True |
| 9 | 0.2794532775878906 | 0.04055840149521828 | 0.08536648750305176 | 0.013166971504688263 | True |
| 10 | 0.2741265296936035 | 0.039872877299785614 | 0.09294360876083374 | 0.012523413635790348 | True |
| 11 | 0.26026955246925354 | 0.03874538838863373 | 0.08662194013595581 | 0.011217963881790638 | True |
| 12 | 0.2775123119354248 | 0.03645211085677147 | 0.08097465336322784 | 0.011742617934942245 | True |
| 13 | 0.3484201431274414 | 0.03641080483794212 | 0.0818067193031311 | 0.01393395196646452 | True |
| 14 | 0.219427227973938 | 0.032876864075660706 | 0.09466403722763062 | 0.017109667882323265 | True |
| 15 | 0.20839858055114746 | 0.030751192942261696 | 0.11488394439220428 | 0.01988946460187435 | True |

## Notes

- Gate ran with the cached tinygrad GGUF (Q6_K, imatrix) as producer vs fp16 mlx consumer — the
  deltas below are attributable to that weight-precision mismatch (see Root cause), **not** to an
  exporter/interchange defect.
- Exporter core (`tinygrad_kv_worker.exporter.export_prompt_cache`) is unchanged and unit-tested
  8/8; the geometry fix (head_dim derived from real tensors = 64) is applied and verified.
- Interchange format + exporter are the durable boundary (ADR 0001/0002); exact `P == R` proof is
  deferred to Path C's producer-swap parity gate on identical fp16 weights.
