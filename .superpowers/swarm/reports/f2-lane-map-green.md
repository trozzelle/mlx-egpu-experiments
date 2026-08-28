# F2 task set 2 — WMMA lane-map GREEN implementation

## Status and scope

- **Task:** F2 task set 2, independent gfx1201 wave32 WMMA lane-map proof.
- **Owner:** `F2LaneReview` implementation handoff; supervisor owns final hardware evidence and publication.
- **Changed production files:** `native_r9700/wmma_lane_map.py` and `native_r9700/wmma_lane_map_runner.cpp`.
- **Verification policy:** no compiler, pytest, formatter, package-manager, hardware, or git command was run in this implementation lane. The exact verification commands below are recorded for supervisor execution.

## Implemented review findings

- The comparator now computes `pack_sha256` from the immutable diagnostic preimage under domain `r9700-wmma-lane-map-diagnostic-pack-v1`. The preimage is the canonical JSON of the normalized schema/target/source/image/manifest/ABI/geometry/instruction/raw-order/numerical-policy record and its manifest byte digest. Observed top-level or nested pack claims are ignored; identity no longer falls back to request-bound JSON.
- Observed conformance input is fail-closed to the exact admitted runtime substrate `TinyGPU.app/APLRemotePCIDevice/PCIIface`, in addition to the existing `1002:7551`/`gfx1201`/wave32/instruction checks.
- The host runner accepts only `--asset-root <path> --log <path>` (or hardware-free `--help`), rejects malformed arguments before asset/device/log work, and publishes no partial log. It validates the direct-child manifest/image and source path, bounded file sizes, source/image SHA-256 values, descriptor/ABI/geometry/readback metadata, diagnostic-only admission, and wave32 image properties.
- The runner builds finite deterministic FP16 A/B and FP32 C tags from the frozen element formulas, allocates named A/B/C/observations buffers through the existing `AMDevSession::dispatch_resident_hsa` path, performs exactly three tagged dispatches and 2048-byte observation readbacks, decodes all 32x16 raw words, checks the returned AMDev hardware identity, and atomically writes the request-bound observed JSON schema. No CPU/tinygrad/HIP/mock result is emitted as hardware evidence.

## Exact supervisor verification commands

Compile the runner with the current AMDev/runtime source closure:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp \
  native_r9700/runtime_contract.cpp \
  native_r9700/prefill_npz.cpp \
  native_r9700/vram_layout.cpp \
  native_r9700/vram_allocator.cpp \
  native_r9700/dynamic_page_table.cpp \
  native_r9700/resident_memory.cpp \
  native_r9700/vram_smoke_asset.cpp \
  native_r9700/hsa_code_image_asset.cpp \
  native_r9700/model_weight_binder.cpp \
  native_r9700/llama_stage_layout.cpp \
  native_r9700/llama_layer_executor.cpp \
  native_r9700/kernel_assets.cpp \
  native_r9700/amdev_session.cpp \
  native_r9700/kernel_catalog.cpp \
  native_r9700/device_memory.cpp \
  native_r9700/hardware_lock.cpp \
  native_r9700/runtime.cpp \
  native_r9700/native_resource_worker.cpp \
  native_r9700/wmma_lane_map_runner.cpp \
  -I native_r9700 -o build/f2-wmma/wmma_lane_map_gfx1201
```

Run the focused hardware-free contracts after compilation:

```sh
${PY} -m pytest \
  tests/native_r9700/test_wmma_lane_map_asset.py \
  tests/native_r9700/test_wmma_lane_map_runner.py -v
```

Run the supervisor-owned hardware proof only after the focused contracts pass:

```sh
tools/native-r9700-hardware-run \
  build/f2-wmma/wmma_lane_map_gfx1201 \
  --asset-root native_r9700/kernels/wmma-lane-map-gfx1201-hsa-assets \
  --log logs/f2/wmma-lane-map-proof.json
```

Then compare the observed log using the pinned calculator outputs and the comparator CLI frozen in `f2-contract-freeze.md` §3.2. Hardware acceptance remains supervisor-owned; a missing, mismatched, or non-admitted runtime identity is a failed proof.
