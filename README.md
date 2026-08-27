# R9700 eGPU Inference

This repository develops two co-equal products for the AMD Radeon AI PRO R9700 on Apple Silicon:

- **R9700 Prefill Service** — a persistent native prefill producer for mlx-lm/oMLX consumers, using the serialized mlx-lm prompt cache as the compatibility artifact.
- **Portable Inference Device Platform** — the in-repository TinyGPU Device Owner, inference-shaped HAL, and admitted Kernel Packs.

The accepted baseline includes native R9700 kernel/transfer/resident-VRAM proof, token-exact 16-layer Llama 3.2 1B prefill through prompt-128, and imported-cache serving without prefix recomputation or fallback.

## Source layout

- `native_r9700/` — native runtime, service, cache, benchmark, Kernel Pack, and model-contract source.
- `tinygpu/` — TinyGPU DriverKit extension, Xcode project, conformance client/tests, app source, entitlements, and installer scripts.
- `tinygrad_kv_worker/` — historical tinygrad Path A correctness control.
- `tests/` — Python/native contract and regression tests.
- `docs/` — architecture, design, roadmap, implementation plan, ADRs, task packets, and evidence ledgers.

See `CONTEXT.md`, `docs/ARCHITECTURE.md`, `docs/DESIGN.md`, and `docs/ROADMAP.md` for the current domain language and capability gates.

## TinyGPU license and provenance

The code under `tinygpu/` is a vendored and locally modified derivative of the TinyGPU installer subtree from [`tinygrad/tinygrad`](https://github.com/tinygrad/tinygrad):

- Upstream revision: `12addee14f1d728793648ceca307a5fde2b24cea`
- Upstream path: `extra/usbgpu/tbgpu/installer`
- Local import checkpoint: `f18261437`
- Upstream license: **MIT**
- Required copyright and permission notice: [`tinygpu/LICENSE`](tinygpu/LICENSE)

Local changes include the structured TGPU v1.0 ABI, separate inference/recovery/diagnostic roles, R9700-only package scope, legacy proxy quarantine, fail-closed cold lifecycle and evidence, bounded resource ownership, host-visible allocation, client-death cleanup, response-payload preservation, and conformance contracts.

The machine-readable provenance, source-tree digest, file-level license review, target scope, modification record, image status, and linked conformance evidence are recorded as `tinygpu-device-owner-vendor` in `docs/upstream-reference-manifest.yaml`. Tinygrad AMDev source used as a differential oracle is read-only reference material and is not vendored into this repository.

**The TinyGPU MIT license does not implicitly license the rest of this repository.** This repository currently does not declare one project-wide license; do not assume that files outside `tinygpu/` are MIT-licensed unless a file or component explicitly says so.

## Development checks

Use the pinned interpreter:

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests -q
```

TinyGPU host contracts and unsigned Xcode builds run from `tinygpu/`. Exact commands and current blockers are maintained in:

- `docs/tasks/native-r9700-producer/validation-commands.md`
- `docs/tasks/r9700-products/phase-p1-tinygpu-device-owner.md`
- `.superpowers/swarm/progress.md`

Native hardware acceptance requires fresh R9700-bound evidence. CPU/NumPy results and pre-warmed state are never native acceptance substitutes.
