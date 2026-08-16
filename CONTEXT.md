# CONTEXT

Project language for the eGPU-assisted MLX-LM / oMLX inference work. Glossary only — no
implementation detail, spec, or decisions. Architecture lives in `docs/ARCHITECTURE.md`,
implementation contracts in `docs/DESIGN.md`, sequencing in `docs/ROADMAP.md`, decisions in
`docs/adr/`.

---

**KV tensor**:
A single per-layer attention key or value tensor for one sequence. Standard inference vocabulary.
_Avoid_: referring to a whole sequence's cache as a single "KV tensor" (that is a KV cache).

**KV cache**:
The collection of all per-layer KV tensors for one sequence/request. The complete attention state
a decoder needs to continue generation. Standard inference vocabulary (SGLang, vLLM, llama.cpp).
_Avoid_: "KV cache" and "prompt cache" are **not** synonyms in this project (see Prompt cache).

**Prompt cache**:
A *portable, serialized image* of a KV cache that crosses the device boundary from a prefill
producer (Path A: tinygrad; Path C: a native engine) to a consumer (mlx-lm / oMLX). Distinct from a
KV cache: a prompt cache is the interchange artifact (e.g. mlx-lm's `save_prompt_cache` /
`load_prompt_cache` `.safetensors`), not the in-memory state. For Path A the interchange format is
mlx-lm's cache schema; for the Path C endgame the format **may evolve** and is a candidate contract,
not binding.
_Avoid_: treating "prompt cache" and "KV cache" as the same thing; the former is serialized, the
latter is in-memory.

**Prefill producer**:
The component that runs the prompt forward pass and emits a prompt cache. Producer owns the KV
truth for the prefilled portion; the consumer treats the imported prompt cache as fixed
compatibility state. Not a single implementation — it is tinygrad in Path A and a native engine in
Path C.
_Avoid_: "prefill daemon" (a Path A Phase-1 *implementation term* — see `docs/DESIGN.md`, not
architecture language).

**Prefill consumer** (the decode host):
The component that decodes from an imported prompt cache — mlx-lm and oMLX on Apple Silicon Metal.
Consumers never recompute the prefilled portion; they hold it as compatibility state.

**KV interchange format**:
The durable contract for the prompt cache: layout, dtype, per-layer schema, and position/RoPE
semantics. This is the durable product boundary for Path A, defined in `docs/DESIGN.md`. Path C may
redesign it.
_Avoid_: "KV ABI" — implementation-flavored and implies a fixed binary ABI; the format is a
versioned interchange schema.

---

**Deprecated terms**:
- "Radeon R9700" — superseded by the correct model: AMD Radeon AI PRO R9700 (RDNA4 / gfx12-class,
  32 GB workstation GPU, ASUS TURBO variant).
