# Llama embedding-row smoke RED contract

## Selector

- `tests/native_r9700/test_runtime_llama_embed_contract.py`

## Contract

The no-hardware test compiles the complete native runner dependency closure and
executes only `--help`. The help surface must advertise:

```text
--llama-embed-smoke --model <dir> --token-id <uint32>
```

The future hardware success schema requires a model identity; redacted explicit
`uint32` token id with model token count and provenance; one binder-validated
4 KiB safetensors span; exactly one host staging read and one uploaded-row
window; selected-row GPU scalar zero; a loaded, attested HSA code image with
SHA-256 and entry offset; resident embedding-row and hidden-output buffers;
one PM4 dispatch; passing SDMA H2D/D2H; byte-exact fp16 row-to-hidden equality;
and `native_prefill_acceptance: open`.

It explicitly rejects CPU model math, fixture rows, archive sources, and C0
assets as producer inputs.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_runtime_llama_embed_contract.py -q
```

## Intended current RED

The supervisor command is deliberately not run in this task. The current
runner help does not expose `--llama-embed-smoke`, so the no-hardware help
contract is RED before any model, device, or hardware command is selected.
