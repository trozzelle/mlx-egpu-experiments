# C2 task set 4 — mlx-lm integration run and report append

Status: **Done**.

## Commands run

Producer-unavailable fallback smoke:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.serving \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --producer-model /tmp/native-r9700-missing-producer-model \
  --fixtures-dir tests/native_r9700/fixtures \
  --prompt-name prompt-1 \
  --max-new-tokens 4 \
  --threshold-tokens 128 \
  --producer-timeout-s 5 \
  --artifacts-dir logs/c2-serving-unavailable \
  --json logs/c2-serving-unavailable/result.json \
  --log logs/c2-serving-unavailable/run.log
```

Output: `C2 serving status=pass prompts=1`.

Full fixture-suite integration/report:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.serving \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --max-new-tokens 4 \
  --threshold-tokens 128 \
  --producer-timeout-s 300 \
  --artifacts-dir logs/c2-serving \
  --json logs/c2-serving/result.json \
  --log logs/c2-serving/run.log \
  --report docs/path-a-validation-results.md
```

Output: `C2 serving status=pass prompts=3`.

## Results

- `logs/c2-serving/result.json`: `gate_result: pass`, `exit_status: 0`, `prompt_count: 3`.
- `prompt-0`: `S=6`, `route=native_mlx_fallback`, `fallback_reason=below_threshold`, `accepted_cache=false`, decoded tokens exactly match baseline `[12366, 13, 578, 469]`.
- `prompt-1`: `S=222`, `n_prefix=221`, `route=native_producer`, `accepted_cache=true`, metadata `{offset: 221, num_layers: 16, n_kv_heads: 8, head_dim: 64}`, decoded tokens exactly match baseline `[128009, 128006, 78191, 271]`.
- `prompt-2`: `S=661`, `n_prefix=660`, `route=native_producer`, `accepted_cache=true`, metadata `{offset: 660, num_layers: 16, n_kv_heads: 8, head_dim: 64}`, decoded tokens exactly match baseline `[128009, 128006, 128006, 128006]`.
- `logs/c2-serving-unavailable/result.json`: producer command fails before acceptance, `route=native_mlx_fallback`, `fallback_reason=producer_failed`, `accepted_cache=false`, `prompt_cache_path=null`, `gate_result=pass`, `exit_status=0`, decoded tokens exactly match baseline `[128009, 128006, 78191, 271]`.
- `docs/path-a-validation-results.md` has the Path C2 section at lines 96-117 and preserves Path A and C1 sections.

## Review

`agent://C2WrapperFinalReview` approved the wrapper/integration artifacts with no Critical, Important, or Minor findings and stated the supervisor can mark C2 task sets 2-5 done/dropped as scoped and proceed to C2 security review.
