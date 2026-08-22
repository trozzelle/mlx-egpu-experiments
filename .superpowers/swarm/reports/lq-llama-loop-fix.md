# Llama persistent-loop fixes

- Added file-backed, range-checked selection of the 4 KiB F16 embedding row for each token; the runtime uploads that raw row into the resident hidden-input buffer before its layer group. No embedding values are decoded or computed on the CPU.
- Bound the model-wide embedding span into the persistent weight table, limited requests to the resident 128-token cache capacity, and refreshed the RoPE/attention sequence (`1`), position (token index), and cache-stride (`128`) kernarg fields before every layer's stage group.
- Marked every persistent per-layer K/V cache for a final full-size (131072-byte) readback.
- Added focused no-hardware contracts for distinct selected row ranges, out-of-range row rejection, token-specific dynamic kernargs, and explicit full K/V cache readback configuration.

Verification was not run because the assignment explicitly prohibits commands.
