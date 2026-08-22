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
producer to a consumer. Distinct from a KV cache: a prompt cache is the interchange artifact
(e.g. mlx-lm's `save_prompt_cache` / `load_prompt_cache` `.safetensors`), not the in-memory state.
_Avoid_: treating "prompt cache" and "KV cache" as the same thing; the former is serialized, the
latter is in-memory.

**Prefill producer**:
The component that runs the prompt forward pass and emits a prompt cache. Producer owns the KV
truth for the prefilled portion; the consumer treats the imported prompt cache as fixed
compatibility state. Not a single implementation — it is tinygrad in Path A and a native R9700
producer in Path C.
_Avoid_: "prefill daemon" (a Path A Phase-1 *implementation term* — see `docs/DESIGN.md`, not
architecture language).

**Prefill consumer** (the decode host):
The component that decodes from an imported prompt cache — mlx-lm and oMLX on Apple Silicon Metal.
Consumers never recompute the prefilled portion.

**KV interchange format**:
The durable contract for the prompt cache: layout, dtype, per-layer schema, and position/RoPE
semantics. This is the product boundary for Path A and the first Path C native-producer stage.
Later native-consumer-backend work may evolve or retire it, but only after an explicit gate.
_Avoid_: "KV ABI" — implementation-flavored and implies a fixed binary ABI; the format is a
versioned interchange schema.

**Path C**:
The tinygrad-free track. It starts with a native R9700 prefill producer behind the KV interchange
format, after a short dual-track runtime spike, and only later may become a native mlx-lm/oMLX
consumer backend.
_Avoid_: treating Path C as "rewrite mlx-lm first", "fork DwarfStar", or "full inference engine" as
the initial boundary.

**Native R9700 producer**:
A Path C prefill producer that runs model-forward kernels on the AMD Radeon AI PRO R9700 without
tinygrad and emits a prompt cache through the KV interchange format.
_Avoid_: generic ROCm backend, DwarfStar fork, full server, or decode owner.

**Native consumer backend**:
A later-stage integration where mlx-lm or oMLX schedules R9700 work directly instead of receiving a
serialized prompt cache from a producer.
_Avoid_: treating this as the first Path C acceptance gate.

**DwarfStar reference**:
Antirez' `ds4` / DwarfStar codebase used as prior art for narrow native inference engines and
Metal/ROCm kernel structure.
_Avoid_: `fs4`; treating DwarfStar as a dependency, target architecture, or general GGUF runner.

---

**Deprecated terms**:
- "Radeon R9700" — superseded by the correct model: AMD Radeon AI PRO R9700 (RDNA4 / gfx12-class,
  32 GB workstation GPU, ASUS TURBO variant).
