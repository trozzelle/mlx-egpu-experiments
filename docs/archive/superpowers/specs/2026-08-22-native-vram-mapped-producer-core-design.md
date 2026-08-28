# Native VRAM-Mapped Producer Core Design

## Decision

Build the native producer around a shared R9700 VRAM-mapped multi-buffer execution core before adding any real Llama or Qwen stage asset. The existing C0 direct-PM4 path is retained for device discovery, VMID0 setup, SDMA, compute queue, fences, BAR0 code load, and PM4 submission. Its fixed 4 KiB input/output/code/kernarg mapping and vector-backed `DeviceMemory` seam are not a model-execution path.

Qwen target: repository-selected `mlx-community/Qwen3.8-27B-4bit` snapshot, text-only first. Image/video processing is excluded from the first producer. The target remains Qwen3.5 VLM architecture with 64 hybrid layers; text-only reduces input processing, not the required hybrid state/cache semantics.

## Goal

Provide a hardware-backed resident-buffer and dispatch layer able to address VRAM-resident model weights, activations, K/V/state buffers, code, and kernargs. On that core, implement a native Llama 3.2 1B producer and a separate native Qwen3.8-27B text producer without CPU tensor math, archived bridge assets, or tinygrad runtime calls in the product path.

## Non-negotiable constraints

- Target hardware/substrate: TinyGPU.app / `APLRemotePCIDevice` / `PCIIface`, PCI `1002:7551`, `gfx1201`.
- `producer_kind=r9700_native` is accepted only for actual hardware model-forward work and a request-bound successful hardware log.
- The Llama S-1 NPZ and mlx-lm prompt-cache ABI remain unchanged.
- No arbitrary BAR0 physical range or VRAM allocation scheme is guessed. A source-grounded TinyGPU/AMDev VRAM ownership/allocation mechanism is a prerequisite.
- C0/archived code, fixture tensors, expected outputs, CPU/NumPy model computations, and tinygrad product execution cannot supply any Llama/Qwen runtime operand or asset.
- Fresh kernel assets remain file-backed, SHA-256-bound, exact `gfx1201`, descriptor-backed, and reviewed. The current compiler probe is not a product asset.
- Qwen cannot reuse Llama config, weight binding, RoPE, stage sequencing, or K/V-only cache assumptions.

## Architecture

### Shared native core

`AMDevSession` gains a narrow hardware-resident buffer registry, not a new runtime:

```text
reviewed allocation contract
  -> VRAM reservation/allocation record
  -> retained physical mappings + dynamic 4-level VM tables
  -> named ResidentBuffer { VA, size, physical pages, ownership }
  -> SDMA upload/readback by resident VA
  -> descriptor/code-object load + entry address
  -> PM4 dispatch with opaque stage kernargs containing resident VAs
```

The core retains every mapping and page-table allocation for the session lifetime, uses source-backed PTE flags, flushes MMHUB and GC TLBs after table changes, and releases all resources deterministically. It has no model names, tokenizer, cache policy, or CPU model math.

The first implementation must establish one real VRAM allocation/ownership method from TinyGPU/AMDev source or observed service behavior. If no such method exists, the core remains blocked; substituting host/system memory or a guessed BAR0 range is explicitly rejected by the user decision.

The selected TinyGPU service has a 256 MiB BAR0 aperture, not a resizable
full-VRAM BAR. The implemented core therefore owns only source-derived
lower-aperture pools: 64 MiB of dynamic page-table space after C0 exclusions
and 159.9375 MiB of real payload VRAM. Llama must stream bounded weight
windows; it must not stage its 501 MiB embedding table. Qwen's full hybrid
state exceeds this aperture, so its later producer requires an explicit
state-spill/residency design rather than an unsupported full-cache allocation.

### Kernel assets

The asset generator must accept fresh HIP/GCN source and emit an ELF/HSACO plus a reviewable manifest. Generated HIP assets require complete code-object handling: select the kernel descriptor by symbol, apply required relocations, honor `kernel_code_entry_byte_offset`, and map every required code/rodata range. Raw `.text` extraction alone remains valid only for independently proven position-independent single-kernel assets.

Every stage dispatch consumes a materialized `KernelDescriptor` and explicit kernarg schema. The VRAM core maps code/kernarg pages at descriptor-selected VAs; it does not infer resource or tensor layouts.

### Llama producer branch

Llama stays the first acceptance branch:

- Llama 3.2 1B, 16 layers, 8 KV heads, head dimension 64, fp16 K/V.
- Standard embedding, RMSNorm, Q/K/V/O/MLP GEMMs, Llama RoPE, causal attention, residual, SiLU-gated MLP, and K/V materialization.
- GPU resident hidden state flows layer-to-layer. Only final K/V and required evidence read back.
- Full result is atomically written as the existing 16-layer NPZ, converted by unchanged `kv_cache.py`, then evaluated by C1R and C2R.

### Qwen producer branch

Qwen is a separate model/cache adapter over the shared core:

- Model snapshot: `<model-hub>/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
- Text-only input uses Qwen tokenizer semantics and explicitly rejects image/video tokens for this phase.
- Nested `text_config`; 64 layers; hidden size 5120; intermediate size 17408; 24 query heads; 4 KV heads; head dimension 256; affine 4-bit group size 64.
- The stage scheduler preserves the 3-linear/1-full hybrid cadence: 48 linear-attention layers and 16 full-attention layers.
- The Qwen adapter owns affine-4-bit dequantization, mRoPE/partial-RoPE handling, full-attention K/V, and linear-attention recurrent/convolution state.
- Before cache export/import work, capture the selected MLX-VLM runtime cache object/state and offset behavior for one text-only prompt. No K/V-only serialization is presumed for hybrid layers.

## Error behavior

- Any unavailable/ambiguous VRAM allocation source blocks before device mapping.
- A resident buffer is rejected on unowned physical range, page-table collision, non-page-aligned VA/size, mapping failure, stale handle, transfer range violation, or TLB flush failure.
- A dispatch is rejected before PM4 submission on non-materialized code, unresolved relocation, invalid entry offset, unregistered resident VA, kernarg schema mismatch, descriptor/asset digest mismatch, or incorrect model adapter geometry.
- Llama and Qwen failures retain `native_prefill_acceptance=open` until each has completed its own end-to-end hardware/parity gate.

## Verification

1. No-hardware C++ contracts cover resident allocation ownership, page-table allocation/collision, mapping lifetimes, transfer bounds, code-object entry validation, and dispatch preflight.
2. One supervisor-owned VRAM allocation smoke proves a non-overlapping allocation, CPU-visible upload/readback, selected hardware identity, page-table/TLB evidence, and `exit_status: 0`.
3. A fresh GPU vector/elementwise kernel proves a resident input/output pair through the shared core. This is the only primitive smoke required before model assets.
4. Llama layer-0 then 16-layer artifacts must be hardware-backed and satisfy unchanged NPZ/cache contracts and token-exact C1R/C2R.
5. Qwen first captures its MLX-VLM text cache ABI, then proves full 64-layer text prefill and its own reference decode/cache acceptance. Qwen does not inherit Llama acceptance artifacts.

## Scope decomposition

This design produces three implementation plans in dependency order:

1. VRAM-mapped resident execution core.
2. Native Llama 3.2 1B producer on that core.
3. Native Qwen3.8-27B affine-4-bit text producer and hybrid cache adapter on that core.

The shared core is the immediate plan. Llama and Qwen plans begin only after its VRAM allocation and resident-dispatch smoke pass.
