# C1R-8f attention head1 CPU oracle fixtures

## Scope

Add the next attention-width CPU oracle fixture slice: layer0 head1, tokens0:5, context cols64:128. This is fixture/reference coverage only; no bridge or native prefill acceptance claim is added.

## Work boundary

- Path: `<former-native-r9700-worktree>`
- Branch: `feature/native-r9700-producer`

## Implemented by agent `AttentionHead1Fixture`

Changed files:

- `native_r9700/ref_fixtures.py`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`

Changed symbols reported:

- `_LAYER0_Q_ROPE_HEAD1_INDEX`
- `_layer0_q_rope_tokens0_5_head1_full_head`
- `_layer0_attention_scores_head1_tokens0_5_scaled_masked`
- `_layer0_attention_probs_head1_tokens0_5_softmax_from_scaled_masked`
- `_layer0_attention_context_head1_tokens0_5_cols64_128`
- `generate_layer_trace`
- head1 Q-RoPE/scaled-mask/softmax/context specs and oracle tests in `tests/native_r9700/test_ref_fixtures.py`

## Decision

Keep head1 as CPU oracle/fixture contract. Hardware bridge coverage still needs a separate primitive-chain implementation. This preserves the current C1 acceptance boundary: C1 native prefill remains blocked until full attention width and full layer hidden state are hardware-resident and verified.

## Verification

Agent RED evidence:

- `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -q -k 'head1 or schema_shape_dtype'`
- Initial result: failed with 5 expected missing-key/schema failures for the new head1 fixture keys.

Agent generation:

- Default generation failed because this worktree has no local `mlx_models/`; default path was treated as a Hugging Face repo.
- Successful regeneration used:
  - `${PY} -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures`
  - Result: wrote 21 fixture files.

Agent focused tests:

- `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -q -k 'head1 or schema_shape_dtype'` -> `15 passed, 37 deselected`
- `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -q` -> `52 passed`

Supervisor verification:

- Command: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -q`
- Result: `52 passed in 0.10s`.

## Remaining blocker

Only head0 and head1 CPU oracles are covered. Full attention width requires the remaining heads and corresponding bridge/runtime contracts before native prefill can honestly pass.
