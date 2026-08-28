# Llama streamed HSA prefill RED contract

## Selector

- `tests/native_r9700/test_native_hsa_prefill_contract.py`

## Contract

The no-hardware gate compiles the complete native runner closure and invokes
only `--help`. The existing `--native-prefill-proof` entry point must explicitly
advertise a **16-layer streamed HSA Llama prefill**; the gate does not open
TinyGPU, a driver, or a model.

A future successful hardware result is accepted only when it attests all of the
following:

- an actual TinyGPU / APLRemotePCIDevice / PCIIface hardware execution and
  readable hardware log;
- exactly 16 layers in ordered stream order `0` through `15`;
- HSA code-image loading with a valid SHA-256 and entry offset, and every
  dispatched kernel accounted as both an HSA-image dispatch and a dispatch from
  a resident lower-BAR window;
- an atomically renamed NPZ at the requested path containing exactly the 32
  fp16 K/V arrays (`layer0_K` through `layer15_V`) plus strict scalar metadata;
- prompt semantics of `n_prefix = S - 1`, with one final decode token;
- accepted-cache decode with a recompute count of zero; and
- no CPU model math, fixture row, archive, or C0 asset contribution.

The accepted NPZ's K/V arrays must each be shaped `(1, 8, S - 1, 64)`, and its
model identity must match the hardware result. A syntactically valid artifact
or an otherwise complete-looking log cannot substitute for the full hardware
attestation.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_native_hsa_prefill_contract.py -q
```

## Intended current RED

The supervisor command is deliberately recorded but not run in this task. The
current help text describes the native-prefill seam generically and does not
advertise the required 16-layer streamed HSA Llama prefill capability. The
contract therefore fails at the no-hardware CLI boundary before it can contact
a driver, open TinyGPU, or read a model.
