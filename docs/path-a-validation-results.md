# Path A — Phase 0 numeric-parity validation results

Status: **RUN COMPLETED — GATE PASSED**

Gate: injected path `P` must equal native baseline `R` token-for-token across the prompt set; per-layer numeric deltas are reported and flagged above the `1e-3` fp16 probe tolerance for diagnosis.

## Result summary

- Gate result: **PASS** (3/3 prompts token-exact).
- Run log: `${HOME}/Development/ml/tools/egpu/.worktrees/tinygrad-kv-worker-phase0/logs/runs/20260816-191810-659350000_meta-f16-final.log`.
- Producer weights: `mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf`.
- Consumer weights: `mlx_models/meta-Llama-3.2-1B-Instruct`.
- Source provenance: official fp16 `meta-llama/Llama-3.2-1B-Instruct` weights on both sides (F16 GGUF producer + mlx safetensors consumer).
- MLX prompt-cache contract: export the `S-1` prefix cache and pass the final prompt token to `generate_step`; passing full `S` plus the full prompt duplicates the prompt.
- Llama-3 RoPE scaling loaded from the MLX `config.json` sidecar and applied to tinygrad's RoPE precompute; the generated GGUF metadata records `rope.freq_base` but not `rope_scaling`.

## Prompt suite

| # | Prompt | S (tokens) | P == R |
|---|---|---|---|
| 0 | `prompt-0` | 6 | True |
| 1 | `prompt-1` | 222 | True |
| 2 | `prompt-2` | 661 | True |

## Per-layer numeric deltas (max|Δ| / mean|Δ| vs native KV)

| Layer | K max|Δ| | K mean|Δ| | V max|Δ| | V mean|Δ| | > 1e-3? |
|---|---|---|---|---|---|
| 0 | 0.0076389312744140625 | 0.00034630033769644797 | 0.00037539005279541016 | 2.0771845811395906e-05 | True |
| 1 | 0.012783050537109375 | 0.0007303535821847618 | 0.0011725425720214844 | 9.534387208987027e-05 | True |
| 2 | 0.020800083875656128 | 0.0009972760453820229 | 0.003493070602416992 | 0.00019622135732788593 | True |
| 3 | 0.03212451934814453 | 0.0011345782550051808 | 0.0035077929496765137 | 0.00024412901257164776 | True |
| 4 | 0.019255638122558594 | 0.0009987273951992393 | 0.003443121910095215 | 0.00026612283545546234 | True |
| 5 | 0.023622244596481323 | 0.0011769216507673264 | 0.004047870635986328 | 0.00027579572633840144 | True |
| 6 | 0.01784515380859375 | 0.0012270527658984065 | 0.003530248999595642 | 0.0003416658437345177 | True |
| 7 | 0.019090652465820312 | 0.0011853931937366724 | 0.00464707612991333 | 0.00039114351966418326 | True |
| 8 | 0.015564918518066406 | 0.0012953929835930467 | 0.0034224987030029297 | 0.0003630796854849905 | True |
| 9 | 0.015559196472167969 | 0.0010691970819607377 | 0.005664646625518799 | 0.0003188189584761858 | True |
| 10 | 0.014158546924591064 | 0.0010957547929137945 | 0.0030842944979667664 | 0.0002942346327472478 | True |
| 11 | 0.01657867431640625 | 0.0010965528199449182 | 0.002843424677848816 | 0.00026680887094698846 | True |
| 12 | 0.01378631591796875 | 0.001003173179924488 | 0.003891170024871826 | 0.00027916315593756735 | True |
| 13 | 0.012420654296875 | 0.001003806246444583 | 0.0032302141189575195 | 0.00034421152668073773 | True |
| 14 | 0.011698722839355469 | 0.001017598551698029 | 0.006541907787322998 | 0.0004919123603031039 | True |
| 15 | 0.01645660400390625 | 0.0010221578413620591 | 0.0058591365814208984 | 0.0005629300139844418 | True |

## Notes

- Flagged layers > 1e-3 fp16 probe tolerance: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]. These are diagnostic tinygrad-vs-MLX implementation deltas; the token gate passed.
