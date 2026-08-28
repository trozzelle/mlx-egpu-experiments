# Native R9700 Product Worker Rearchitecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the retired C1 source-as-data proof lane with a small, maintainable native R9700 prefill worker that preserves the existing KV interchange format and can advance C1R from real model-forward execution to token-exact parity.

**Architecture:** `native_r9700` keeps Python as the fail-closed orchestration and consumer-facing seam. The C++ worker owns selected-substrate AMDev execution, GPU-resident model-forward dataflow, K/V NPZ emission, and evidence. It is split by responsibility into AMDev session, packet/dispatch, kernel catalog, layer executor, and worker modules. The CPU/NumPy producer remains the oracle; `kv_cache.py`, `parity.py`, and `serving.py` retain the durable prompt-cache seam.

**Tech Stack:** Python 3.12.8, pytest, NumPy, MLX/`mlx_lm`, safetensors, C++17, macOS `xcrun --sdk macosx clang++`, TinyGPU.app / `APLRemotePCIDevice` / `PCIIface`, AMD Radeon AI PRO R9700 (`1002:7551`, `gfx1201`).

## Global Constraints

- Use `${PY}`; never use `python3` from `PATH`.
- The active first target is Llama 3.2 1B fp16 with 16 layers and per-layer fp16 K/V `(1, 8, N, 64)`.
- Preserve the KV interchange format and mlx-lm `S-1` injection rule. The final prompt token goes to `generate_step`; the producer owns all preceding KV truth.
- `r9700_native` accepts only real R9700 model-forward work. CPU may load, orchestrate, compare, and serialize; it must not supply accepted model math.
- `r9700_native` fails closed until it emits a valid full 16-layer NPZ plus required hardware evidence.
- No tinygrad import/call in the native producer path. Tinygrad commands are comparison controls only.
- Keep the selected C1 substrate: TinyGPU.app / `APLRemotePCIDevice` / `PCIIface`, PCI `1002:7551`, RDNA4 `gfx1201`.
- Do not restore, compile, link, parse, or depend on `artifacts/native-r9700-c1-proof-archive/20260821T202312Z/native_r9700/c1_primitive_bridge.cpp` in the product path. It is forensic evidence only.
- Do not change `kv_cache.py` K/V tensor schema or start C3/native consumer work.
- Do not create a generic ROCm abstraction, a network transport, or a new build system. Use explicit documented `xcrun` source lists.
- Logs, build outputs, models, and forensic archives remain local/uncommitted. This planning task does not commit changes.

---

## Scope and Supersession

`docs/archive/superpowers/plans/2026-08-21-native-r9700-prefill-worker-benchmark.md` correctly identifies the product objective, but its Tasks 2–3 name the now-retired `native_r9700/c1_primitive_bridge.cpp`. This plan supersedes those implementation mechanics only. The native worker contract, C1R acceptance, C2R dependency, benchmark-after-C2 sequence, and all ADR constraints remain unchanged.

The archived bridge is **not a large module to split**. Its manifest identifies it as a 236,577,976-byte source-as-data proof artifact. Restoring it would recreate the failure mode: fixture bytes, expected outputs, experimental variants, AMDev setup, dispatch, comparison, logging, and CLI dispatch coupled in one translation unit.

## Locked Product Modules

| Module | Files | Responsibility | Explicit non-responsibility |
|---|---|---|---|
| Runtime contract | `native_r9700/runtime.h`, `native_r9700/runtime_contract.cpp` | Stable log types, lifecycle error vocabulary, 24-byte kernarg codec, narrow runner-facing request/result types. | Primitive-chain registry and fixture-derived constants. |
| AMDev session | `native_r9700/amdev_session.h/.cpp` | Source-grounded TinyGPU connection, device identity, BAR/VM lifecycle, GPU allocation ownership. | Llama geometry and model-stage ordering. |
| AMDev packets | `native_r9700/amdev_packets.h/.cpp` | Pure SDMA and PM4 word encoders, validation, named dispatch inputs. | Socket, buffer, model, or log ownership. |
| Device memory | `native_r9700/device_memory.h/.cpp` | Named buffer allocation, bounded upload/download, streaming chunk accounting, cleanup. | Kernel selection and model math. |
| Kernel catalog | `native_r9700/kernel_catalog.h/.cpp` | Compact named gfx1201 kernel descriptors: identity, digest, accepted geometry, kernarg layout, launch shape. | Embedded fixture/expected-output arrays and chain-specific CLI modes. |
| Llama layer executor | `native_r9700/llama_layer_executor.h/.cpp` | GPU-resident Llama layer stage ordering, model-weight binding, K/V materialization, typed stage evidence. | Cache serialization and consumer decode. |
| Native prefill worker | `native_r9700/native_prefill_worker.h/.cpp`, `runner.cpp` | Request validation, 16-layer loop, NPZ write, evidence result and fail-loud exit. | CPU-reference tensor calculation and prompt-cache conversion. |
| Python orchestration | existing `native_worker.py`, `prefill.py` | Invokes the worker, validates `r9700_native` evidence and NPZ, deletes rejected output. | Direct AMDev calls. |
| Fixture catalog | `native_r9700/fixture_catalog.py`, existing fixture files | Declarative oracle geometry, tensor names, tolerances, and digests. | GPU source generation or a head/band-specific API per fixture. |

The C++ request/result seam is intentionally small:

```cpp
struct NativePrefillRequest {
  std::string model_dir;
  std::vector<uint32_t> token_ids;
  std::string out_npz_path;
  std::string log_path;
};

struct NativePrefillResult {
  std::string producer_kind;              // exactly "r9700_native"
  std::string native_prefill_acceptance;  // "open" or "pass"
  std::string prefill_npz_path;
  std::string hardware_log_path;
  uint64_t kernel_count;
  uint64_t transfer_bytes;
  std::string failure_stage;
  std::string failure_text;
  int exit_status;
};

int run_native_prefill(const NativePrefillRequest&, NativePrefillResult*, std::string* error_text);
```

`run_native_prefill` returns `0` only when it emits an accepted 16-layer NPZ. It returns nonzero on any missing hardware evidence, device error, malformed model geometry, partial output, or failed numerical gate. `native_prefill_acceptance` remains `open` for layer-only diagnostic modes.

---

## Wave 0: Remove the Retired Proof Dependency

### Task 1: Establish the product/legacy cutover

**Files:**
- Modify: `native_r9700/runtime.h`, `native_r9700/runtime.cpp`, `native_r9700/runner.cpp`
- Modify: `tests/native_r9700/test_runtime_contract.py`
- Modify: `docs/archive/tasks/native-r9700-producer/README.md`, `docs/tasks/native-r9700-producer/validation-commands.md`

**Interfaces:**
- Consumes: current `RuntimeSession` runner contract and `NATIVE_R9700_C1_PRIMITIVE_BRIDGE` test injection.
- Produces: no default compile command or runtime source reference to `c1_primitive_bridge.cpp`; explicit legacy diagnostics fail unavailable unless a caller supplies a diagnostic executable.

- [ ] **Step 1: Write the RED regression**

Add a no-hardware test that invokes every active runner mode without `NATIVE_R9700_C1_PRIMITIVE_BRIDGE` and asserts no output contains `c1_primitive_bridge.cpp`, no archive path is opened, and the legacy diagnostic exits nonzero with `failure_stage: legacy_proof_unavailable`.

- [ ] **Step 2: Run the RED test**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -q -k legacy_proof_unavailable
```

Expected: failure because the current fallback still builds `native_r9700/c1_primitive_bridge.cpp`.

- [ ] **Step 3: Remove the fallback and obsolete product modes**

Delete the default source-build branch that names the retired file. Keep environment-injected diagnostic execution only behind a clearly named legacy diagnostic path. Remove product-facing help entries and source references for primitive-chain modes that cannot execute without the archive. Retain `--lifecycle-dry-run`, C0 kernel/transfer proof wrappers, and native-prefill runner mode.

- [ ] **Step 4: Decouple active tests from archived source**

Delete tests that parse archived bridge arrays or require its byte layout. Keep behavior tests that use injected fake executables and assert protocol fields, failure-loud behavior, kernarg serialization, and no premature `native_prefill_acceptance: pass`.

- [ ] **Step 5: Verify and record the cutover**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp native_r9700/device_memory.cpp native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Expected: tests pass; compile succeeds without the archived bridge; documentation names the archive as forensic-only.

### Task 2: Bound generated evidence outside source operations

**Files:**
- Modify: `.gitignore`, `docs/tasks/native-r9700-producer/validation-commands.md`
- Create: `native_r9700/run_paths.py`
- Test: `tests/native_r9700/test_run_paths.py`

**Interfaces:**
- Produces: `run_root() -> pathlib.Path` and `new_run_dir(label: str) -> pathlib.Path`.
- Contract: uses `NATIVE_R9700_RUN_ROOT` when set; otherwise `logs/native-r9700-runs`; rejects labels containing path separators; never writes generated logs under `native_r9700/`.

- [ ] **Step 1: Write RED path tests**

```python
def test_new_run_dir_honors_configured_ignored_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVE_R9700_RUN_ROOT", str(tmp_path / "runs"))
    path = new_run_dir("layer0")
    assert path.parent == tmp_path / "runs"
    assert path.name.startswith("layer0-")


def test_new_run_dir_rejects_path_traversal():
    with pytest.raises(ValueError, match="label must not contain a path separator"):
        new_run_dir("../escape")
```

- [ ] **Step 2: Implement `run_paths.py`**

Use `datetime.now(timezone.utc)` and an alphanumeric UTC suffix. Create only the configured directory. Preserve existing historical `logs/` and archive files; do not delete evidence during this task.

- [ ] **Step 3: Verify**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_run_paths.py -q
```

Expected: pass. Document `NATIVE_R9700_RUN_ROOT` as the sole product-run output setting.

---

## Wave 1: Parallel Runtime and Oracle Preparation

These tasks share only the Wave-0 cutover contract. Each creates disjoint files. They may run concurrently; no worker edits `runtime.h`, `runtime.cpp`, `runner.cpp`, `ref_fixtures.py`, or `test_runtime_contract.py` outside its assigned packet.

### Task 3: Extract pure AMDev packet encoding

**Files:**
- Create: `native_r9700/amdev_packets.h`, `native_r9700/amdev_packets.cpp`
- Modify: `native_r9700/runtime.cpp`
- Test: `tests/native_r9700/test_amdev_packets.py`

**Interfaces:**

```cpp
std::vector<uint32_t> build_sdma_copy_words(uint64_t src_va, uint64_t dst_va,
                                             uint32_t byte_count, uint32_t fence_value);
std::vector<uint32_t> build_pm4_dispatch_words(uint64_t code_va, uint64_t kernargs_va,
                                                uint64_t timeline_va);
```

- [ ] **Step 1: Write RED encoder tests**

Port the existing runtime assertions for SDMA length/opcode/fence and the 59-dword PM4 dispatch packet into `test_amdev_packets.py`. Tests import no TinyGPU and run only pure encoders.

- [ ] **Step 2: Extract the exact proven encoders**

Move, without changing emitted words, the current source-grounded encoder bodies from `runtime.cpp` into `amdev_packets.cpp`. `runtime.cpp` includes `amdev_packets.h`; it keeps no duplicate implementation.

- [ ] **Step 3: Verify**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_amdev_packets.py -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -c native_r9700/amdev_packets.cpp -I native_r9700 -o build/native-r9700-runtime/amdev_packets.o
```

Expected: byte-for-byte packet tests pass; C++ object compiles.

### Task 4: Extract device session and bounded memory ownership

**Files:**
- Create: `native_r9700/amdev_session.h`, `native_r9700/amdev_session.cpp`, `native_r9700/device_memory.h`, `native_r9700/device_memory.cpp`
- Modify: `native_r9700/c1_transfer_bridge.cpp`
- Test: `tests/native_r9700/test_device_memory_contract.py`

**Interfaces:**

```cpp
struct DeviceBuffer { uint64_t gpu_va; uint64_t size_bytes; std::string name; };
class DeviceMemory {
 public:
  bool allocate(std::string name, uint64_t size_bytes, DeviceBuffer*, std::string*);
  bool upload(const DeviceBuffer&, const uint8_t* data, uint64_t size_bytes, std::string*);
  bool download(const DeviceBuffer&, uint8_t* data, uint64_t size_bytes, std::string*);
  void release_all();
};
```

- [ ] **Step 1: Write RED validation tests**

Compile a fake/no-hardware harness that verifies zero-size allocation, duplicate names, unknown buffers, oversize upload/download, and use-after-release fail with a nonempty error. Assert transfer accounting increments exactly by requested byte count only after success.

- [ ] **Step 2: Extract C0-backed mechanics once**

Move TinyGPU connection, BAR/VM setup, and C0-proven transfer helper use behind `amdev_session`. Move named allocation bookkeeping and bounded transfer validation into `DeviceMemory`. `c1_transfer_bridge.cpp` becomes a narrow adapter using these modules or is deleted after the equivalent runtime transfer proof passes; do not duplicate AMDev mechanics.

- [ ] **Step 3: Verify no-hardware contract and hardware transfer proof**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_device_memory_contract.py -q
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Then run the existing C0B native AMDev/SDMA transfer command recorded in `validation-commands.md`. Expected: no-hardware contracts pass; hardware log reports the selected substrate, `1002:7551`, `gfx1201`, exact CPU comparison, and `exit_status: 0`.

### Task 5: Replace fixture function proliferation with a catalog

**Files:**
- Create: `native_r9700/fixture_catalog.py`
- Modify: `native_r9700/ref_fixtures.py`, `tests/native_r9700/test_ref_fixtures.py`
- Create: `tests/native_r9700/test_fixture_catalog.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class FixtureSpec:
    name: str
    archive_name: str
    arrays: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    tolerance: str
    sha256: str

_SPECS: tuple[FixtureSpec, ...]
_BY_NAME: dict[str, FixtureSpec]

def fixture_specs() -> tuple[FixtureSpec, ...]:
    return _SPECS

def fixture_spec(name: str) -> FixtureSpec:
    return _BY_NAME[name]
```

- [ ] **Step 1: Write RED catalog tests**

Assert every committed NPZ referenced by the catalog exists, has exactly the declared array names/dtypes/shapes, and has the declared SHA-256. Assert all output bands and attention heads use entries in the same catalog rather than head/band-specific Python lookup functions.

- [ ] **Step 2: Implement catalog and migrate generation**

Move fixed geometry/fixture metadata into `FixtureSpec` entries. Replace repeated per-band wrapper functions with one parameterized generator that consumes `FixtureSpec`. Preserve byte-identical committed fixture data unless an independently justified oracle correction is required.

- [ ] **Step 3: Verify**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_fixture_catalog.py tests/native_r9700/test_ref_fixtures.py -q
```

Expected: catalog validates all committed fixtures and existing oracle tests pass.

### Task 6: Split runtime tests around public seams

**Files:**
- Create: `tests/native_r9700/test_runtime_lifecycle.py`, `tests/native_r9700/test_runtime_protocol.py`, `tests/native_r9700/test_native_worker_evidence.py`
- Modify: `tests/native_r9700/test_runtime_contract.py`

**Interfaces:**
- Consumes: `RuntimeSession` lifecycle/log protocol and Python `run_native_prefill` result schema.
- Produces: test modules partitioned by lifecycle, external command protocol, and accepted native-evidence/NPZ behavior.

- [ ] **Step 1: Move tests without changing assertions**

Move lifecycle/kernarg/packet tests to `test_runtime_lifecycle.py`; injected executable and exact marker tests to `test_runtime_protocol.py`; native NPZ/evidence acceptance tests to `test_native_worker_evidence.py`. Preserve test names or add stable aliases only when pytest collection requires a unique name.

- [ ] **Step 2: Delete archive-dependent test helpers**

Remove `_archived_bridge_source_text_or_skip` and every test that parses archived C++ arrays. The replacement test surface is fixture catalog integrity plus runner protocol/evidence behavior.

- [ ] **Step 3: Verify**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_runtime_lifecycle.py tests/native_r9700/test_runtime_protocol.py tests/native_r9700/test_native_worker_evidence.py -q
```

Expected: all active runtime tests run without opening the forensic archive.

---

## Wave 2: Serialized Integration into a Real Native Worker

Wave 2 begins only after Wave 1 is merged, reviewed, and the integration owner verifies there is one AMDev implementation of each mechanism.

### Task 7: Integrate the small runtime contract

**Files:**
- Modify: `native_r9700/runtime.h`, `native_r9700/runtime.cpp`, `native_r9700/runner.cpp`
- Create: `native_r9700/runtime_contract.cpp`
- Modify: focused runtime test modules

**Interfaces:**
- Consumes: Tasks 3–6 modules.
- Produces: `NativePrefillRequest`, `NativePrefillResult`, `run_native_prefill`, and a thin runner command `--native-prefill-proof`.

- [ ] **Step 1: Write RED runner-contract tests**

Use a no-hardware invocation of `--native-prefill-proof --model missing --token-ids-json '[1]' --out <path> --log <path>`. Assert exit nonzero, `producer_kind: r9700_native`, `native_prefill_acceptance: open`, a specific `failure_stage`, and no output NPZ.

- [ ] **Step 2: Reduce `runtime.h` to stable types**

Keep lifecycle/status/log/Kernargs declarations and the request/result types. Move packet functions into `amdev_packets.h`; remove primitive-chain metadata, fixture hashes, and obsolete proof-chain public entry points.

- [ ] **Step 3: Wire the runner to the worker seam**

Parse only `--model`, `--token-ids-json`, `--out`, and `--log` for native prefill. Validate JSON token IDs are nonempty unsigned integers before device setup. Write one JSON result plus a `key: value` log on every path.

- [ ] **Step 4: Verify**

Run:

```sh
${PY} -m pytest tests/native_r9700/test_runtime_lifecycle.py tests/native_r9700/test_runtime_protocol.py -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp native_r9700/device_memory.cpp native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Expected: focused tests and the explicit multi-source build pass.

### Task 8: Add compact kernel catalog and resident layer-0 executor

**Files:**
- Create: `native_r9700/kernel_catalog.h`, `native_r9700/kernel_catalog.cpp`, `native_r9700/llama_layer_executor.h`, `native_r9700/llama_layer_executor.cpp`
- Modify: `native_r9700/runtime.cpp`
- Test: `tests/native_r9700/test_kernel_catalog.py`, `tests/native_r9700/test_layer0_executor_contract.py`

**Interfaces:**

```cpp
struct KernelDescriptor {
  std::string name;
  std::string sha256;
  uint32_t workgroup_x;
  uint32_t workgroup_y;
  uint32_t workgroup_z;
  uint32_t kernarg_bytes;
};
const KernelDescriptor* find_kernel(std::string_view name);

struct LayerExecutionEvidence {
  uint32_t layer_index;
  uint64_t kernel_count;
  uint64_t transfer_bytes;
  std::string k_shape;
  std::string v_shape;
  std::string hidden_shape;
};
bool execute_llama_layer0(const NativePrefillRequest& request,
                          DeviceMemory* device_memory,
                          LayerExecutionEvidence* evidence,
                          std::string* error_text);
```

- [ ] **Step 1: Write RED descriptor and fail-closed tests**

Assert duplicate kernel names, invalid zero workgroup dimensions, non-64-character SHA values, unknown kernel lookup, missing model weights, and fixture-sourced intermediate input claims fail loudly. Assert layer-0 evidence always reports `native_prefill_acceptance: open`.

- [ ] **Step 2: Implement catalog without operands**

Store compact descriptors only. Kernel executable data must be sourced from a named, reviewed kernel asset or compiler output and identified by digest. Do not embed model weights, fixture inputs, expected outputs, or generated C++ byte arrays.

- [ ] **Step 3: Implement one resident layer-0 vertical slice**

For prompt-0, load real model weights and token-derived inputs; allocate device buffers once; execute layer-0 stages in model order; preserve GPU-produced intermediates between stages; read back only K/V and final hidden output for CPU-oracle comparison. Record every kernel stage and transfer boundary.

- [ ] **Step 4: Verify**

Run focused tests, then compile the explicit Wave-2 source list. Run the hardware layer-0 command discovered by this task and append its exact command only after a successful observed run to `validation-commands.md`.

Expected: layer-0 log proves real inputs/model weights, selected hardware identity, resident dataflow, K/V and hidden comparison, `native_prefill_acceptance: open`, and `exit_status: 0`.

---

## Wave 3: Full Prefill, Parity, and C2R Handoff

### Task 9: Extend the worker to all layers and emit NPZ

**Files:**
- Modify: `native_r9700/llama_layer_executor.cpp`, `native_r9700/native_prefill_worker.h`, `native_r9700/native_prefill_worker.cpp`, `native_r9700/runtime.cpp`
- Modify: `native_r9700/native_worker.py`, `native_r9700/prefill.py`
- Test: `tests/native_r9700/test_native_worker_evidence.py`, `tests/native_r9700/test_prefill.py`

**Interfaces:**
- Consumes: Task 8 resident layer execution.
- Produces: `run_native_prefill` full 16-layer NPZ with scalar metadata `model`, `n_prefix`, `num_layers`, `producer_kind` and exactly `layer{i}_K` / `layer{i}_V` fp16 arrays.

- [ ] **Step 1: Write RED full-NPZ tests**

Assert a result with 15 layers, malformed scalar metadata, wrong `(1, 8, N, 64)` geometry, non-fp16 K/V, CPU producer identity, nonzero worker exit, or missing hardware log is rejected and the NPZ is removed.

- [ ] **Step 2: Implement 16-layer execution**

Loop layers 0–15 with GPU-generated hidden state flowing directly into the next layer. Stream weights by an explicit bounded `DeviceMemory` buffer when full residency is unavailable. Record per-layer kernel counts, transfer bytes, and failure stage. Write NPZ atomically only after all arrays and evidence validate in memory.

- [ ] **Step 3: Wire Python acceptance**

`native_worker.run_native_prefill` and `prefill.py --producer-kind r9700_native` accept only the full result. Retain CPU-reference behavior unchanged. A hardware/shape failure removes the native output and exits nonzero.

- [ ] **Step 4: Verify prompt-0 native artifact**

Run the worker on prompt-0 and convert its NPZ using the documented `native_r9700.kv_cache` command. Expected: 16 layers, 32 fp16 K/V tensors, `offset=S-1`, `producer_kind=r9700_native`, and hardware log with `exit_status: 0`.

### Task 10: C1R parity gate and review package

**Files:**
- Modify: `native_r9700/parity.py`, `tests/native_r9700/test_parity.py`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`, `docs/path-a-validation-results.md`, `.superpowers/swarm/progress.md`

**Interfaces:**
- Consumes: Task 9 accepted NPZ route.
- Produces: one report per prompt and aggregate `P == R` result containing exact tokens, cache path, hardware log path, producer identity, and K/V delta diagnostics.

- [ ] **Step 1: Write RED parity evidence tests**

Reject `r9700_native` reports with missing hardware evidence, missing cache paths, a CPU-reference producer label, or any token mismatch. Verify the final token is passed to the consumer while the cache covers `S-1` tokens.

- [ ] **Step 2: Run prompt-0 first**

Execute native prefill, cache conversion/import, and mlx-lm decode. Diagnose only observed K/V geometry, RoPE/absolute-position, layer-order, or precision deltas. Do not relax token exactness.

- [ ] **Step 3: Run all Phase-0 prompts**

Run prompt-0, prompt-1, and prompt-2 separately so failures retain bounded logs and artifacts. Produce one aggregate report only after all three runs are observed.

- [ ] **Step 4: Update durable evidence**

Only after all prompts pass, add observed commands to `validation-commands.md`, update the Path C section of `docs/path-a-validation-results.md`, and append C1R evidence/status to the swarm ledger.

- [ ] **Step 5: Review gate**

Require a reviewer to inspect model geometry, RoPE, K/V layout, producer identity, hardware evidence, atomic output handling, and `P == R`. Critical or Important findings block C2R.

### Task 11: C2R native-serving handoff

**Files:**
- Modify: `native_r9700/serving.py`, `tests/native_r9700/test_serving.py`
- Modify only if delegation requires it: `tinygrad_kv_worker/harness.py`, `tests/test_harness_c2_serving.py`
- Modify after observed runs: `docs/tasks/native-r9700-producer/validation-commands.md`, `docs/path-a-validation-results.md`, `.superpowers/swarm/progress.md`

**Interfaces:**
- Consumes: C1R-accepted `r9700_native` result.
- Produces: serving JSON with `requested_producer_kind`, `actual_producer_kind`, `accepted_cache`, `hardware_log_path`, `fallback_reason`, and timing fields for prefill, cache emission/import, final-token decode, and total latency.

- [ ] **Step 1: Write RED routing tests**

Assert large-prompt native requests reject or fallback before cache acceptance when the worker is unavailable, reports CPU reference, emits malformed output, or lacks hardware evidence. Assert decode failure after cache acceptance is returned as an error and never recomputes the prefilled prefix.

- [ ] **Step 2: Route accepted large prompts**

Invoke only the accepted native `prefill.py` route, then existing `kv_cache.py`, cache validation/import, and mlx-lm final-token decode. Do not duplicate cache conversion logic in the harness.

- [ ] **Step 3: Run C2R evidence**

Run prompt-0 and one larger Phase-0 prompt through direct serving. Run below-threshold, unavailable-producer, and malformed-output fallbacks. Record exact commands only after observed success.

- [ ] **Step 4: Final review handoff**

Require focused transport/security and code review. C3 stays blocked unless the native C2R route, exactness, and timing evidence pass.

---

## Parallelism and Integration Rules

1. **Wave 0 is serialized.** Task 1 owns the cutover because it changes current runtime/test behavior. Task 2 touches only run-path policy and may begin once Task 1 chooses the legacy diagnostic names.
2. **Wave 1 fans out to four workers:** Tasks 3, 4, 5, and 6. Each owns disjoint files. A supervisor must reject edits to another packet’s files rather than resolve overlapping changes by hand.
3. **Wave 2 is serialized under one integration owner.** The C++ types and source list are shared; parallel edits here would create incompatible AMDev interfaces.
4. **Wave 3 is dependency-ordered:** Task 9 → Task 10 → Task 11. C1R parity must not race C++ changes; C2R must not consume an unaccepted producer.
5. **Review gates serialize after every wave.** Workers do not run project-wide suites or hardware commands concurrently. The supervisor runs focused checks after each merge and owns hardware evidence, ledger updates, and all status promotion.
6. **No task may reintroduce archive coupling.** The archive path is allowed only in manual forensic documentation, never runtime, test, build, or product command code.

## Verification Matrix

| Gate | Required evidence |
|---|---|
| Cutover | Active build/tests contain no `c1_primitive_bridge.cpp` reference; legacy diagnostic fails explicitly when not injected. |
| AMDev extraction | Pure packet tests plus selected-substrate transfer proof maintain expected bytes and exact round-trip. |
| Fixture catalog | Every declared artifact has matching arrays, geometry, dtype, tolerance, and digest. |
| Layer-0 | Real model/token inputs, resident hardware stages, K/V/hidden oracle comparison, open acceptance. |
| Full prefill | Atomic 16-layer fp16 NPZ, `r9700_native`, selected-hardware evidence. |
| C1R | Every Phase-0 prompt reports token-exact `P == R`; logs prove native model-forward kernels. |
| C2R | Large prompts use accepted native producer; pre-acceptance fallback and post-acceptance no-recompute behavior pass. |

## Plan Self-Review

- **Coverage:** Wave 0 removes the blocker; Waves 1–2 establish reusable native execution; Wave 3 reaches C1R then C2R. KV format, identity, hardware, no-tinygrad, and fail-closed requirements are preserved.
- **No false acceptance:** Layer diagnostics explicitly remain `open`; only full native prefill with hardware evidence can be `pass`; parity remains token-exact.
- **Scope:** Qwen, network transport, generic ROCm abstraction, direct consumer work, source-archive restoration, and benchmark work before C2R are excluded.
- **Validation:** All current focused Python/C++ commands are explicit. New hardware worker commands are intentionally discovered only by their implementing task and recorded after observed success, per the validation ledger policy.
