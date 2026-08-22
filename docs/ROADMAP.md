# Roadmap

This roadmap sequences capabilities for `docs/ARCHITECTURE.md` and `docs/DESIGN.md`. It is not an
implementation backlog; implementation plans should be created separately when a phase is ready
(e.g. via `plan-to-agent-task-docs`).

## Roadmap principles

- Durable vocabulary comes from `CONTEXT.md`; contracts from `DESIGN.md`.
- Phases are capability gates, ordered by dependency and risk.
- Producer changes must pass the Phase-0-style token-exact parity gate before serving integration.
- Path C is staged: native producer first, native mlx-lm/oMLX backend later.
- DwarfStar is a reference corpus, not a dependency or fork target.

## Current baseline

- AMD eGPU (AI PRO R9700, RDNA4/gfx12-class) works via tinygrad/TinyGPU with `DEV=AMD JITBEAM=2`.
- Phase 0 passed: official Meta Llama 3.2 1B fp16 tinygrad/R9700 producer KV matches the mlx-lm
  fp16 consumer baseline token-for-token. Report: `docs/path-a-validation-results.md`.
- Logging exists for GPU/harness runs; local logs are review artifacts under `logs/` and stay
  uncommitted.
- No persistent R9700/eGPU model-forward producer daemon or native consumer backend exists yet.
- Current `native_r9700.prefill` CPU/NumPy work is a reference producer / prompt-cache ABI oracle,
  not R9700 compute. Current `native_r9700.serving` work is an imported-cache wrapper around that
  reference path, not completion evidence for native C2.

## Superseded direction

The earlier roadmap treated Path C as a deferred endgame after Path A daemon and consumer-wrapper
phases. Phase 0 has now validated the central theory, so Path C becomes the primary next design
track. Path A daemon/consumer work remains a bridge option, not a prerequisite for Path C.

The earlier draft also allowed semantic equivalence as a fallback success bar for Phase 0. That is
superseded: the producer-swap gate is token-exact `P == R`; semantic equivalence may be recorded but
does not pass the gate.

---

## Phase 0: Validated KV interchange parity (complete)

**Outcome:** A tinygrad-produced prompt cache (KV interchange format v1) lets mlx-lm decode from
the imported cache and exactly match native mlx-lm decode for the gate prompts.

### Capabilities

- Exporter: tinygrad `cache_kv` tensor → mlx-lm-format prompt cache `.safetensors`.
- Injection harness: tinygrad prefill → export → `load_prompt_cache` → `generate_step` decode on
  Metal, compared against a native mlx baseline.
- Suite-level per-layer numeric delta report and run log path.

### Dependencies

- Official Meta Llama 3.2 1B fp16 weights converted to both MLX and F16 GGUF.
- Working `DEV=AMD JITBEAM=2` tinygrad setup.
- Llama-3 RoPE sidecar from MLX config; `S-1` prompt-cache contract for `generate_step`.

### Promotion gate

- Passed: `P == R` token-for-token for all gate prompts.

### Validation and review expectation

- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v` → `17 passed`.
- `py_compile tinygrad_kv_worker/harness.py` → exit 0.
- GPU parity run wrote `docs/path-a-validation-results.md` and a local run log.

---

## Bridge Phase A1/A2: Optional tinygrad producer service and consumer wrapper

**Outcome:** If needed before Path C lands, package the validated tinygrad producer behind a local
service and consume it from real mlx-lm/oMLX serving.

### Capabilities

- Persistent local producer process with model resident on `DEV=AMD`.
- Local Unix-socket/stdio request-response contract: token ids in, prompt-cache bytes out.
- Thin mlx-lm wrapper with prompt-length threshold and native-prefill fallback.
- Optional oMLX imported-cache seam.

### Dependencies

- Phase 0 exporter/harness.
- `DESIGN.md` producer daemon and consumer integration contracts.

### Promotion gate

- Multiple requests served without weight reload; returned caches load in mlx-lm and decode through
  the same parity gate.

### Validation and review expectation

- Transport review before any TCP/network exposure.
- Integration run appends results to `docs/path-a-validation-results.md`.

---

## Phase C0: Native runtime discovery

**Outcome:** Decide the first tinygrad-free runtime substrate using evidence, not assumption.

### Capabilities

- Local macOS eGPU path: prove a minimal R9700 kernel launch and host↔device buffer movement outside
  tinygrad, or record why it is blocked.
- Linux ROCm/HIP reference path: build and run a minimal reference kernel path, using DwarfStar's
  ROCm structure as prior art where useful.
- Shared diagnostic shape: local logs, timing/error reporting, CPU reference comparison.

### Dependencies

- ADR 0003 hybrid staged boundary.
- DwarfStar source review for Metal/ROCm layout and quality gates.
- Current TinyGPU/tinygrad AMD facts from `docs/pinned-upstream-interfaces.md`.

### Promotion gate

- One substrate is selected for the first native R9700 producer, or the phase explicitly records a
  split plan with one production substrate and one reference substrate.

### Validation and review expectation

- Minimal kernel output compared against a CPU reference.
- Run logs retained locally.
- Runtime choice recorded in `docs/DESIGN.md` before model kernels start.

---

## Phase C1: Native R9700 producer parity (reopened; CPU reference reclassified)

**Outcome:** A tinygrad-free producer runs Llama 3.2 1B fp16 prefill model-forward tensor work on the
R9700/eGPU, emits a prompt cache, and passes the Phase-0-style token-exact gate with mlx-lm as the
consumer.

### Current state

- C0 selected the macOS TinyGPU.app/APLRemotePCIDevice/PCIIface native AMDev substrate (C0A25
  minimal-kernel PASS; ADR 0004).
- The completed CPU/NumPy implementation proves the KV interchange ABI, RoPE/position semantics,
  `S-1` final-token injection, and parity harness shape.
- Per ADR 0005, that CPU/NumPy path is not C1 acceptance evidence because model-forward compute does
  not run on the R9700/eGPU.

### Capabilities

- Native R9700/eGPU implementation of the model-forward pieces required for Llama 3.2 1B fp16
  prefill.
- KV emission in the existing KV interchange format for the `S-1` prefix.
- Numeric delta reporting against CPU reference and native mlx-lm baseline.

### Dependencies

- Phase C0 selected runtime substrate — complete.
- CPU/NumPy reference producer and fixture oracle — complete, usable only as a diagnostic/reference.
- Phase 0 prompt set, weights, RoPE sidecar contract, and report format.
- `DESIGN.md` Native R9700 producer contract and ADR 0005 acceptance correction.

### Promotion gate

- `P == R` token-for-token for all Phase 0 gate prompts using an R9700/eGPU producer route.
- Hardware logs prove model-forward prefill kernels ran on the selected R9700/eGPU substrate.
- Deltas reported and diagnosable; every native GPU run has a reviewable local log.

### Validation and review expectation

- Minimal-kernel tests remain green.
- Producer parity run writes a corrected Path C section in `docs/path-a-validation-results.md`.
- Code review focuses on model geometry, RoPE/position semantics, K/V layout, transfer boundaries,
  kernel execution evidence, and failure handling.

---

## Phase C2: R9700 producer serving integration (reopened; CPU wrapper reclassified)

**Outcome:** The actual R9700/eGPU producer from C1 is usable by real mlx-lm serving through the
imported-cache seam, with prompt-length threshold and fallback behavior defined.

### Current state

- The existing `native_r9700.serving` wrapper proves mlx-lm can consume imported prompt-cache
  artifacts through the final-token seam.
- Per ADR 0005, it is reference C2 evidence until its large-prompt producer route executes
  model-forward prefill on the R9700/eGPU.

### Capabilities

- Local producer invocation from an mlx-lm wrapper, backed by the R9700/eGPU producer route.
- Prompt-length threshold and native mlx-lm fallback for small prompts or producer failure before
  cache acceptance.
- Optional oMLX imported-cache integration through the same consumer seam only after the R9700 route
  is proven.

### Dependencies

- Phase C1 R9700/eGPU producer parity.
- Consumer integration seam in `DESIGN.md`.
- ADR 0005 producer identity requirements in logs/reports.

### Promotion gate

- Large prompts use the R9700/eGPU producer; small prompts and unavailable-producer cases fall back
  to native mlx-lm prefill without corrupting decode state.
- Results match the producer-swap parity/quality gate for the Phase 0 prompt set.

### Validation and review expectation

- Integration run against mlx-lm native baseline.
- Security review before any non-local transport.
- oMLX scope decision recorded if oMLX is included.

---

## Phase C3: Native consumer backend decision and prototype

**Outcome:** Decide whether to retire the serialized prompt-cache fast path for a direct mlx-lm/oMLX
R9700 backend, and prototype only if the prior phases justify it.

### Capabilities

- Backend seam selection: mlx-lm first, oMLX first, or shared layer.
- Direct scheduling prototype for a narrow kernel/model slice.
- Compatibility story for prompt-cache artifacts as fallback/review outputs.

### Dependencies

- Phase C2 serving evidence.
- Measured transfer overhead and prefill performance from C1/C2.
- New design update or ADR if the KV interchange boundary is superseded on the fast path.

### Promotion gate

- Native backend prototype improves the measured bottleneck without losing the producer-swap
  correctness gate.

### Validation and review expectation

- Prototype compared against imported-cache path and native mlx-lm baseline.
- ADR required if the prompt-cache boundary is retired or demoted from the fast path.

---

## Deferred or rejected directions

- **Direct native backend first:** rejected as initial Path C sequencing; too many risks coupled
  before a tinygrad-free producer passes parity.
- **DwarfStar fork:** rejected. Use `antirez/ds4` as a reference for narrow engine/kernels/testing,
  not as this project's architecture.
- **Generic ROCm platform:** rejected until a concrete model/runtime path needs it.
- **Semantic-equivalence producer gate:** rejected; token-exact `P == R` is required.
- **Multi-node / distributed prefill:** out of scope for this roadmap.

## Handoff to task docs

When a phase is ready for execution, use `plan-to-agent-task-docs` to turn the phase's capability
gate into task documents, referencing: this ROADMAP (phase), `docs/DESIGN.md` (contracts),
`docs/pinned-upstream-interfaces.md` (external API pins), `docs/egpu-prefill-offload-reference.md`
(research), and the ADRs. Do not encode executable subagent assignments in this roadmap.
