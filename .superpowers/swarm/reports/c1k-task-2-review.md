# C1 Task Set 2 Review — Weight/Config Container Decision + Narrow Loader (Lane B)

**Reviewer:** C1LoaderReviewer
**Date:** 2026-08-18
**Scope:** `native_r9700/__init__.py`, `native_r9700/config.py`, `native_r9700/loader.py`,
`tests/native_r9700/test_loader.py`, appended C1 loader section in
`docs/tasks/native-r9700-producer/validation-commands.md`.
**Method:** Read-only source inspection. No commands run (per validation policy).

---

## Decision: APPROVE

No Critical or Important findings. The lane is correct, maintainable,
architecturally aligned with the frozen C1 contract, and appropriately narrow.
Three Minor (non-blocking) notes below.

---

## Summary

- **Container decision (MLX safetensors dir) is sound.** The MLX safetensors
  directory carries both the fp16 weights (`F16` header dtype) and the complete
  `config.json` sidecar with the Llama-3 `rope_scaling` block. The loader reads
  **the same on-disk config the Phase-0 MLX consumer reads**, so geometry and
  RoPE parity are guaranteed by construction — satisfying the "exact config
  parity" claim. The recorded rationale (F16 GGUF records `rope.freq_base` but
  not `rope_scaling`, so it cannot give parity alone) matches the C1 contract
  and the Phase-0 patching history.
- **Loader is narrow and correct.** Pure stdlib (`json`, `os`, `argparse`,
  `dataclass`); **no tinygrad dependency** in the producer path (contract
  requirement met). Reads only `config.json` + safetensors header records,
  never weight payloads.
- **Geometry/provenance validation is loud and precise.** Every locked geometry
  field (`num_layers=16`, `n_kv_heads=8`, `head_dim=64`, `hidden=2048`,
  `rope_theta=500000`, `rope_scaling` llama3 block) is checked via
  `_checked_int` / explicit comparison and raises a typed
  `GeometryMismatchError`/`UnsupportedModelError` on deviation.
- **Dtype validation is correct.** On-disk safetensors headers are validated
  F16-only (any non-F16 tensor in any shard → `UnsupportedDtypeError`).
  Config `torch_dtype` is correctly treated as advisory (HF-original = bf16)
  and rejected only for non-half families (e.g. `int8` → `UnsupportedDtypeError`).
- **Tests defend real contracts.** Geometry parse from a real fixture config,
  dtype rejection, all error paths (missing dir/config, geometry mismatch,
  unsupported model_type/architectures, rope_scaling-required, CLI exit codes).
  Deterministic/isolated (tmp_path + subprocess with explicit cwd). No
  plumbing/source-text assertions.

---

## Quality-Bar Verdict per Criterion

### Correctness — PASS
All specified acceptance criteria are met: exact geometry parse
(16/8/64/2048, rope_theta 500000, llama3 scaling), loud failure on missing
config/dir, geometry mismatch, unsupported model/dtype, and unsupported
architectures. Provenance reads the MLX consumer's own file, so parity is by
construction. The dtype contract (F16 on-disk weights) is enforced at the
authoritative header level. No contract deviation found.

### Maintainability — PASS
Clean naming (typed exception hierarchy, frozen dataclasses `Llama32Config`/
`ModelData`), clear separation of config parsing (`config.py`) from
loading/report (`loader.py`) from CLI (`main`), and explicit advisory-vs-
authoritative provenance comments. Docstrings carry the container decision and
contract rationale inline, which is where future readers will look.

### Architectural Fit — PASS
No tinygrad dependency. fp16-first parity matches the frozen contract. The
RoPE sidecar is consumed from the exact MLX consumer config, satisfying the
"exact config parity" goal that the GGUF path cannot. `__init__.py`
intentionally avoids re-exports so the package imports without side effects —
appropriate for the C++-first package.

### Simplicity / Anti-over-engineering — PASS with Minor notes
The loader is close to the narrowest adequate solution for the first parity
modelgang. The only generality beyond the first model's needs is the shard/
index machinery (Minor 1). Everything else is single-model-locked by design
and the non-goals (no generic GGUF runner, no extra models, no quantization)
are honored.

---

## Findings

### Minor 1 — Speculative generality: shard/index machinery for a single-file model
- **File:** `native_r9700/loader.py:54-72` (`_find_weight_index`,
  `_shards_for_index`), `:48-51` (`weight_index_path`, `weight_shards`),
  `:146-165` (multi-shard loop)
- **Body:** The loader implements safetensors index/shard support
  (`.index.json` expansion, iterating multiple shards, `weight_shards` list,
  `weight_index_source`/`weight_shard_count` report lines). The first parity
  model, Llama-3.2-1B, ships as a single `model.safetensors` file — no index
  and no sharding. This machinery is correctly implemented but currently dead
  generality beyond the narrowest adequate solution the task set requires.
  Not a defect; flagged against the task's explicit anti-over-engineering
  criterion. If single-file coverage is the contract, dropping the index path
  (or deferring it to when a sharded second model exists) would trim it.
- **Priority:** 2 (non-blocking) — **Confidence:** 0.7

### Minor 2 — `max_position_embeddings` is not validated (asymmetry with locked geometry)
- **File:** `native_r9700/config.py:218-221`
- **Body:** Every locked geometry field is validated through `_checked_int`/
  explicit comparison, but `max_position_embeddings` is parsed with a fallback
  to `original_max_position_embeddings` and **never checked against any
  SUPPORTED_* value**. A config with a wrong `max_position_embeddings` silently
  passes and is echoed in the report, diluting the documented "reject geometry
  mismatch" guarantee. This field is not in the task set's locked geometry
  list (16/8/64/2048 + rope), so it is out-of-contract — noted only as an
  internal consistency improvement, not a defect.
- **Priority:** 3 (note only) — **Confidence:** 0.6

### Minor 3 — `format_report` hardcodes validated geometry while using `cfg` elsewhere
- **File:** `native_r9700/loader.py:176-179`
- **Body:** `format_report` hardcodes `"num_layers: 16"`, `"n_kv_heads: 8"`,
  `"head_dim: 64"`, `"hidden_size: 2048"` while rendering the rest from `cfg`
  (`intermediate_size`, `vocab_size`, `max_position_embeddings`, `rope_theta`).
  Functionally safe (the values are locked and validated equal), but the
  mixed style invites drift if a SUPPORTED_* constant ever changes and the
  literals are missed. Could read from `cfg`/`SUPPORTED_*` for consistency.
- **Priority:** 3 (note only) — **Confidence:** 0.8

---

## Notes (non-code)

- Supervisor brief cites "10 loader tests pass"; `tests/native_r9700/test_loader.py`
  actually contains **19** tests. This is a counting discrepancy in the brief,
  not a code defect. The on-disk suite is the authoritative count.
- The C1 loader section appended to `validation-commands.md` correctly records
  the container decision, the loader command, the expected report lines
  (16/8/64/2048, rope_theta 500000, rope_scaling llama3, weight_dtype F16),
  and the focused-test invocation. Consistent with the code and the task set 2
  target/change/acceptance.

---

## Conclusion

The lane meets the frozen C1 contract with a correct, maintainable, narrow
loader and tests that defend the real contracts. **APPROVE.** The three Minor
notes are non-blocking and can be folded into a later cleanup wave if the
maintainer chooses.
