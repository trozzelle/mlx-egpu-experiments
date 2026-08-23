# Pinned Upstream Interfaces — Path A (Phases 0–2)

Reference capture of the external interfaces Path A builds on. These are upstream, version-churny
surfaces; pin the exact contracts here so the exporter / daemon don't silently break. Re-verify
against upstream before each phase if the pinned commit changes.

Capured: 2026-08-16 (tinygrad, mlx-lm `main`; oMLX `main`).

---

## 1. tinygrad LLM module — API surface the exporter calls

Package moved from `tinygrad/llm.py` → `tinygrad/llm/` (`__init__.py`, `__main__.py`, `cli.py`,
`gguf.py`, `model.py`, `serve.py`).

- **Loader:** `Transformer.from_gguf(path_or_url, max_context)` (`cli.py:~210`). GGUF only.
- **Tokenizer:** `SimpleTokenizer.from_gguf_kv(kv)` (`cli.py:~212`).
- **KV storage (per block)** — `TransformerBlock._init_state` (`model.py:~168`):

  ```python
  self.cache_kv = Tensor.empty(
      2, x.shape[0],           # axis 0 = stacked [K, V]
      self.config.n_kv_heads,
      self.config.max_context,
      self.config.head_dim,
      dtype=dtypes.default_float,  # default fp32
      device=x.device)
  ```
  Shape `[2, B, n_kv_heads, max_context, head_dim]`; K in slot 0, V in slot 1. On the AMD device.
- **Forward / prefill** — `Transformer.forward(tokens, start_pos, temperature)` (`model.py:~290`)
  returns a **sampled token id** (Gumbel-argmax), *not* logits and *not* KV.
  `Transformer.__call__` dispatches to `prefill_jit` (batch) or `rollout_jit` (single token).
- **Prompt loop:** `generate(tokens, chunk_size=32, temperature=0.0)` (`model.py:~430`) — chunked
  prefill, then 1-token decode, updating `start_pos`. `get_start_pos` (`model.py:~410`) computes
  prefix reuse from `self._cached_tokens` (multi-turn KV reuse in the HTTP server).
- **KV update path** (`TransformerBlock._attention`, `model.py:~150`): K/V written in place into
  `cache_kv` via a `store` UOp; valid prefix re-read per step:
  ```python
  k = assigned_kv[0, :, :, 0:start_pos+T, :]
  v = assigned_kv[1, :, :, 0:start_pos+T, :]
  ```
- **No KV export / IPC / logits-out API** exists. Only `.to('CPU').numpy()` inside-process.

Export work: slice valid prefix `[..., :S, :]`, split axis 0 (K/V), cast fp16, write mlx cache.

---

## 2. mlx-lm KV cache ABI — the interchange contract

Sources: `mlx_lm/models/cache.py` (`KVCache`, `save_prompt_cache`, `load_prompt_cache`,
`make_prompt_cache`); `mlx_lm/generate.py` (`generate_step`, `_model_call`, `_step`, prefill loop).

- **Cache construct:** `make_prompt_cache(model, max_kv_size=None)` → `[KVCache() per layer]`.
- **`KVCache`** (standard; GQA/RoPE LLMs):
  - `keys`, `values`: `mx.array`, shape `(B, n_kv_heads, S, head_dim)`. For Llama 3.2 1B:
    `(1, 8, S, 64)` each, **fp16**. (head_dim = hidden 2048 / 32 heads = 64 for
    Llama 3.2 1B; the earlier `128` was a research error.)
  - `update_and_fetch(keys, values)`: preserves input dtype on preallocated `mx.zeros`; grows by
    `step=256`.
  - `state` property → `(keys[..., :offset], values[..., :offset])`.
  - **`meta_state` (REV snapshot 2026-08-16, mlx-lm 0.31.3):** the standard `KVCache` has **no**
    `meta_state` override — it inherits `_BaseCache.meta_state`, whose **setter raises `ValueError`
    for any truthy value** (`"This cache has no meta_state but a meta_state was set."`). So
    `KVCache.from_state(state, str(S))` **raises for `S > 0`**, and per-layer `meta_state` must stay
    `""`. Only `QuantizedKVCache` / `RotatingKVCache` / `ChunkedKVCache` define accepting
    `meta_state` overrides. `offset` is reconstructed from `state.keys.shape[2]`, so the exported
    offset survives the round-trip via `keys.shape[2]`, not `meta_state`.
  - `from_state(state, meta_state)`, `trim(n)`, `is_trimmable()`.
- **Serialize / deserialize (the Phase 0–2 bridge):**
  - `save_prompt_cache(file, cache, metadata={})` — writes each layer's `state` arrays + class name +
    empty `meta_state`, plus a **global `metadata` dict**, to `.safetensors`. The exporter records
    the exported offset in global metadata:
    `{'offset': str(N), 'num_layers':…, 'n_kv_heads':…, 'head_dim':…}`.
  - `load_prompt_cache(file, return_metadata=False)` — rebuilds `[KVCache.from_state(...)]`
    (dispatch on class name); with `return_metadata=True` returns `(cache, metadata)`.
  - **Downstream note:** if upstream restores a `meta_state` override on the standard `KVCache`,
    per-layer metadata can switch to `str(S)` — one-line change; the global copy makes this safe.
- **Prefill seam:** `generate_step(prompt, model, prompt_cache=None, prefill_step_size=2048, …)`:
  - `generate_step` always processes the supplied `prompt`: it prefilles `prompt[:-1]` into the
    provided cache, then runs `_step(prompt[-1])` to produce the first decoded token.
  - Therefore injected full-prompt decode uses an imported cache for the `S-1` prefix and supplies
    only the final prompt token. Supplying full `S` cache plus the full prompt duplicates the prompt.
  - Without a pre-supplied cache, the same loop builds a native cache from `prompt[:-1]`;
    `mx.eval([c.state ...])`; `mx.clear_cache()` (`generate.py:430-453`).

---

## 3. oMLX seam (Phase 2, optional)

Sources: `omlx/scheduler.py`, `omlx/custom_kernels/`, `omlx/cluster/worker.py`,
`docs/distributed-cluster.md`.

- **oMLX is Python, wraps mlx-lm** (`scheduler.py:33-48`): imports `BatchGenerator`,
  `PromptProcessingBatch`, `KVCache as _MLXKVCache`, `make_prompt_cache`.
- **Insertion seam:** `scheduler.py` monkey-patches mlx-lm caches with batch-aware
  `filter/extract/extend` (`:901-912`); `make_prompt_cache` builds the prompt cache.
- **External-process precedent (transport to mirror):** `cluster/worker.py` = stdio, newline-delimited
  JSON worker; coordinator runs a rank-0 mlx-lm HTTP endpoint, spawns isolated rank processes
  (Ring/Thunderbolt RDMA/JACCL for MLX groups). See `docs/distributed-cluster.md`.

---

## 4. tinygrad AMD runtime / TinyGPU (hardware transport)

Sources: `docs.tinygrad.org/tinygpu/`, `docs/runtime.md`, `tinygrad/runtime/ops_amd.py`,
`docs.tinygrad.org/developer/am/`.

- **macOS AMD transport = USB/DMA (TinyGPU)**, not Vulkan/Metal:
  `AMDDevice` selects `USBIface` on macOS (`ops_amd.py:940`), enumerating
  `USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)`.
- **Compiler:** HIP/COMGR (`DEV=AMD:HIP`) or LLVM (`DEV=AMD:LLVM`). `JITBEAM=2` = kernel autotuning.
- **Supported arch assert (`ops_amd.py:951`):** targets `(9,4,2)`, `(9,5,0)`, or gfx 11/12.
  AI PRO R9700 = **RDNA4 / gfx12-class** → supported.
- **AM driver (`PCI` interface):** userspace RDNA3/RDNA4 driver; single compute queue bound at
  `pipe=0 queue=0`; SDMA at `engine=0 queue=0`; `AM_RESET`, `AM_DEBUG` env vars. Linux-side; the
  macOS path uses the USB interface instead.
- **Process-local, no tensor IPC.** `AMDAllocator` maps `va_addr` into the opening process; `Device`
  singletons cached per-PID. → Daemon must ferry serialized bytes (token ids in / safetensors out).

---

## 5. Version-pinning note

All line numbers above were captured from the referenced upstream `main` on 2026-08-16. Before each
phase's implementation, re-read the pinned upstream files and update this capture if APIs drifted.
The single most load-bearing contract for Phases 0–2 and Path C is **§2 (mlx-lm KV cache ABI)** —
keep it green.
---

## 6. mac-amdgpu — native DriverKit reference driver (behavioral control)

Reference clone: `${HOME}/Development/ml/tools/mac-amdgpu` (cloned 2026-08-23,
`https://github.com/lemonade-sdk/mac-amdgpu`, v0.1.48 @ `3bdeed2`).

A third-party native PCIDriverKit driver extension — a **dext**, not a kext, not TinyGPU — that
ports Linux `amdgpu` kernel-driver slices into a DriverKit system extension and talks to the GPU
directly over PCIe, bypassing Metal. Target: AMD Radeon AI PRO R9700 (`1002:7551`, gfx1201, rev C0)
over Thunderbolt 5.

- **Bringup state:** all 15 IP-block stages green — `IPDiscovery → IHInit → GMCInit → PSPInit →
  PSPLoadSOS → PSPRingCreate → TMRSetup → PSPFwLoad → SMUInit → IMUInit → RLCInit → CPInit →
  MESInit → GFXInit → SDMAInit`; SDMA VRAM→VRAM copy running (MMIO WPTR mode); KIQ PM4 NOP+fence
  smoke test; command-stream ABI (`CSCreate/CSWriteDwords/CSDestroy`).
- **Not yet done:** GFX compute-kernel execution — no MES GFX queue with a real compute kernel, no
  register-allocated kernel ISA run. (Our C0A25 `--kernel-proof` is ahead of this on the compute
  path.)
- **Requirements:** macOS Tahoe 26.2+, **SIP disabled** (dev entitlements), Xcode 26 + DriverKit SDK
  25.4+, paid Apple Developer membership with granted `…driverkit.transport.pci` and
  `allow-any-userclient-access` entitlements. These are that project's dev/signing setup, not
  TinyGPU requirements.

Facts worth reusing for native init/GART/PTE work:

- BAR0 maps the **bottom** of VRAM (low 256 MiB) on this hardware; their VRAM bump allocator is
  pinned to `[vram_start + 24MB, vram_start + 256MB)` to stay inside the visible aperture.
- GART page table in VRAM at `vram_start + 0x700000` (upstream `amdgpu_gart_table_vram_alloc`),
  GFX12 PTE format with the `IS_PTE` bit, NBIO HDP `remap_hdp_registers`.
- SMU mailbox handshake (`SetDriverDramAddr → RunDcBtc → EnableAllSmuFeatures`) unblocks the IMU
  autoload; without it `BOOTLOAD_STATUS` stays 0 even after PSP signs off.
- Firmware: `psp_14_0_3_sos.bin`, `smu_14_0_3.bin`, `sdma_7_0_1.bin`,
  `gc_12_0_1_{rlc,imu,pfp,me,mec,uni_mes}.bin`.

Role here: **behavioral control, not a substrate.** It independently confirms 256 MiB BAR0 +
indirect VRAM IP discovery is correct, and its init/GART/PTE/PSP/SMU sequences are a source
reference for any future native-init work. It is not a drop-in for the compute path (no GFX kernel
execution yet).

---

## 7. PCIe BAR / I/O-memory diagnostic analysis (three-failure-class model)

`docs/Diagnosing and Resolving PCIe BAR and I_O-Memory Mapping Failures for an AMD Radeon AI PRO
R9700 eGP.pdf` (2026-08-23).

Core correction: the observed session failure is **not** one generic "PCIe BAR mapping bug". Three
classes:

1. **Class A — host/DriverKit PCI resource failure:** BAR missing/zero-size, `GetBARInfo` fails,
   Memory Space Enable clear, all-ones from every register.
2. **Class B — GPU indirect-VRAM/init failure after BAR mapping succeeds:** BAR0/2/5 valid and BAR5
   register reads work, but the GPU-internal indirect VRAM (RSMU/MM_INDEX) path returns garbage.
   This is the post-power-cycle state the session hit; a cold cable disconnect repaired it.
3. **Class C — GPU VM/queue-programming failure after BAR + firmware work:** BARs/discovery/VRAM/
   SDMA/queue setup succeed, but compute fails on GPU-VA or MQD/HQD address encoding. This is where
   the session ended (the live blocker).

Load-bearing facts:

- **256 MiB BAR0 is normal, not a bug.** `large_bar=False` is expected; indirect VRAM via BAR5/RSMU
  is the correct path. mac-amdgpu and Linux both treat BAR0 as the visible aperture, not all VRAM.
- **TinyGPU is a DriverKit dext, not a kext.** SIP / AuxKC rebuild / Reduced Security / kext signing
  are not normal runtime prerequisites for the released TinyGPU dext.
- **`GetBARInfo(barIndex, &memoryIndex, …)` never assumes `barIndex == memoryIndex`.** `MemoryRead*`
  take the returned memoryIndex; a read error sets the value to `-1` (`0xffffffff`), so an isolated
  all-ones read needs supporting metadata before it means "unassigned BAR".
- Endpoint driver must re-enable **Memory Space Enable + Bus Master Enable** on every configure.

Recommendation (adopted): do not change BAR sizing / SIP / ReBAR / TinyGPU install while fixing the
compute fault; treat the MQD/HQD address problem as a separate Class C bug.

---

## 8. ChatGPT R9700 diagnosis — launch reliability + transcendental re-frame (2026-08-23)

`docs/ChatGPT-Diagnose R9700 Mapping Issues-20260823-1619.pdf`.

Separates three failure domains the RMSNorm debugging had conflated: (1) cold-start/queue-lifecycle
state, (2) program-image launch reliability (the original RMSNorm image executes; newly generated
images stall), (3) RMSNorm numerical correctness (NaN). Load-bearing corrections:

- **Instruction-cache coherency (top suspect for the new-image stalls).** Uploading a replacement
  code image at a reused GPU VA and dispatching it does not prove the GPU instruction fetch sees the
  new bytes; `acquire_mem(gli=0, gl2=0)` is not sufficient for overwrite-then-execute. Needs a
  code-install barrier (GLI_INV + GLK_INV/WB + GL1_INV + GL2_INV/WB + GLM_INV/WB + GLV_INV) and/or a
  fresh-VA/fresh-page code slot. This re-explains the rsqrt/epsilon stalls as stale-cache execution,
  not a rsqrt-ISA or transcendental fault.
- **Queue poisoning.** A faulting/timing-out compute queue must be fully torn down (dequeue request,
  SPI queue reset, `CP_HQD_ACTIVE==0`, 64-bit RPTR/WPTR carrier reset) before the next test; otherwise
  later results are not independent ("sentinel sandwich" around each candidate image).
- **`CP_MEC_RS64_EXCEPTION_STATUS` is MEC CP-firmware state, not the shader PC.** Low bits: bit 0
  illegal-instruction, bit 1 misaligned-address, bit 2 unaligned-instruction, bit 3 page-fault, bits
  26:4 = RS64 exception instruction address. `0xc67a` ⇒ RS64 addr `0x0c67` + misaligned + page-fault;
  it is not a `.text` offset. Correlate with GCVM L2 protection-fault state and SQ wave/trap status.
- **`COMPUTE_PGM_RSRC3` does not encode VGPR/SGPR on GFX12.** The changing field across the 32/64/
  128/160 values is `INST_PREF_SIZE` (bits 11:4), not register allocation (see §9).
- **The `V_S_SQRT_F32` + `V_DIV_SCALE` + `V_RCP` + Newton FMAs + `V_DIV_FMAS` + `V_DIV_FIXUP` sequence
  is the normal LLVM precise-division lowering**, not proof of a compiler bug; splitting a C expression
  into locals does not stop LLVM recombining it. Isolate the NaN only after launch and code visibility
  are deterministic.

Recommended order before any more RMSNorm math: (1) deterministic queue teardown + fresh-start
asserts; (2) code-upload GLI/GLK/GL1/GL2/GLM/GLV invalidation; (3) fresh-VA code-image allocation;
(4) `--override-rsrc3` (0x00 vs generated vs old); (5) PM4 word-by-word packet annotation + A/B/C/D
markers; (6) salvage output + fault registers after every timeout; (7) verify code via a GPUVM-mediated
copy, not just BAR0; (8) compare complete old/new descriptors (wave32, scratch, user-SGPR, dispatch
pointer, entry alignment); (9) then one-operation sqrt/reciprocal/rsqrt microkernels.

---

## 9. LLVM AMDGPU backend reference (gfx1201 target + RSRC3)

`https://llvm.org/docs/AMDGPUUsage.html`.

- gfx1201 = target triple `amdgpu12.01` (major subarch `amdgpu12`, generic processor `gfx12-generic`).
- **`COMPUTE_PGM_RSRC3` for GFX12** (authoritative field layout):
  - bits 3:0 — RESERVED (must be 0);
  - **bits 11:4 — `INST_PREF_SIZE`** (8 bits): instruction bytes to prefetch from the kernel entry
    before wavefront start, 0..255 × 128-byte granularity — `0x40`⇒512 B, `0x80`⇒1024 B, `0xa0`⇒1280 B;
  - bit 13 — `GLG_EN` (group launch guarantee);
  - bits 16:14, 17, 20:18 — RESERVED on GFX120*.
- The transcendental intrinsics (`__builtin_amdgcn_rsqf`/`rcpf`/`sqrtf`) are not enumerated in this
  document (they live in the LLVM AMDGPU intrinsic reference), but it pins the RSRC3 prefetch field
  that changed between the old image (0xa0) and the new images (0x80 rsqrt, 0x40 epsilon).
