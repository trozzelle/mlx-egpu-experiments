# Llama attention HSA assets RED contract

## Selector

- `tests/native_r9700/test_llama_attention_hsa_assets.py`

## Contract

Three fresh, checked-in, regular HIP sources must provide the direct causal
attention stages:

- `native_r9700/kernels/llama_causal_attention_score_f16.cpp`
- `native_r9700/kernels/llama_causal_softmax_f32.cpp`
- `native_r9700/kernels/llama_attention_context_f16.cpp`

Each source must expose its named C-linkage GPU kernel for `gfx1201`, operate
within a 64-token bounded prefix tile, use GPU workgroup/workitem indexing, and
have no host, fixture, archive, C0, CPU-model, or LDS machinery. Each generated
HSA image must have zero group and private segment bytes, exactly one admitted
symbol, and a manifest that binds the reviewed source and image hashes.

The score stage accepts fp16 Q and the resident `k_cache` window, writes fp32
scores, and has this exact 32-byte ABI:

```text
q            uint64  offset 0
k_cache      uint64  offset 8
scores       uint64  offset 16
token_count  uint64  offset 24
```

It maps 32 query heads onto 8 resident K heads by GQA group, accumulates the
64-element fp16 dot product in fp32, scales it by `1/sqrt(64)`, and writes
negative infinity whenever `key_token > query_token`.

The fp32 softmax stage has this exact 24-byte ABI:

```text
scores         uint64  offset 0
probabilities  uint64  offset 8
token_count    uint64  offset 16
```

It keeps future causal positions at zero and uses fp32 row maximum,
shifted-exponential, and normalization accumulation.

The context stage reads fp32 probabilities and the resident `v_cache` window,
accumulates V contributions in fp32, and writes fp16 context through this exact
32-byte ABI:

```text
probabilities  uint64  offset 0
v_cache        uint64  offset 8
context        uint64  offset 16
token_count    uint64  offset 24
```

It uses the same 32-to-8 GQA mapping as score. K/V pointers are the future
resident lower-BAR-window bindings; the contract admits neither projection,
staging, nor host-copy substitutes.

## Supervisor RED command (do not run in this task)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_llama_attention_hsa_assets.py -q
```

## Intended current RED

The supervisor command is recorded but deliberately not run in this task. The
focused test currently fails at the missing fresh score source before generator,
compiler, driver, device, fixture, archive, C0, or CPU-model paths can run. No
source implementation or generated asset is introduced by this contract.
