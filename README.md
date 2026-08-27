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

## TinyGPU license

`tinygpu/` is a locally modified derivative of [`tinygrad/tinygrad`](https://github.com/tinygrad/tinygrad), whose upstream code is MIT-licensed. The required notice is in [`tinygpu/LICENSE`](tinygpu/LICENSE).

Detailed source provenance and the local modification record are in [`docs/upstream-reference-manifest.yaml`](docs/upstream-reference-manifest.yaml).

The MIT notice applies to the TinyGPU-derived code under `tinygpu/`; this repository does not currently declare a project-wide license.

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
