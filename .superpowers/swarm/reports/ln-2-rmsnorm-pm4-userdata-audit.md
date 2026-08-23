# RMSNorm PM4 kernarg user-data audit

## Scope and result

This is a source/asset audit only; it does not execute a GPU dispatch.

**Result:** the resident HSA path puts `kKernargsVa` in `COMPUTE_USER_DATA_0:1`, and the RMSNorm code image is configured to receive a kernarg-segment pointer.  The `s[0:1]` convention is therefore justified by both the RMSNorm descriptor and the frozen, hardware-proven C0 kernel convention.  This evidence rules out neither a bad kernarg-page mapping nor a bad individual pointer stored in that page, but it does rule out a missing PM4 user-data write / incompatible no-kernarg-SGPR descriptor as the explanation.

## Exact delivery chain

| Step | Evidence | Consequence |
| --- | --- | --- |
| Fixed GPU virtual address | `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:205, 313` sets `am_vm::kVaBase = 0x0000200000000000` and `kKernargsVa = kVaBase + 6 * 0x1000`, i.e. `0x0000200000006000`. `native_r9700/runtime_contract.cpp:273-274` independently records the resident HSA kernarg VA as that same value. | The address used by the HSA session is a fixed GPU VA, not a CPU pointer. |
| CPU-side contents | `native_r9700/amdev_session.cpp:581-600` computes `compute_control_mapping.data + kComputeControlKernargsCpuOffset`, clears one 4 KiB page, copies the supplied byte layout, fences, and compares the copied bytes. The frozen C0 definition makes that offset page 1 (`native_amdev_transfer_probe.cpp:335-342`). | The resident session writes the dispatch arguments to CPU-mapped compute-control page 1 and detects a CPU-side copy mismatch. |
| GPU-side mapping | `native_amdev_transfer_probe.cpp:3443-3475`, especially line 3468, maps `compute_control->sys_pages[1]` at `am_compute::kKernargsVa` with system-memory PTE flags. The C0 proof log records the ten-page compute-control allocation and page 1 physical address (`logs/c1-runner-kernel-proof-2026-08-23T02:57:12Z.log:96-111`). | The CPU page holding the copied args is the GPU page addressed by `kKernargsVa`; it is not merely adjacent host memory. |
| PM4 user data | `native_r9700/amdev_packets.cpp:141-176` emits a `PACKET3_SET_SH_REG` at the C0-derived `kComputeUserData0SetShOffset = 0x240` with payload `lo32(config.kernargs_va), hi32(config.kernargs_va)` (lines 59 and 160-162). Every resident HSA stage passes `am_compute::kKernargsVa` to this builder (`amdev_session.cpp:2284-2290`). | The packet writes the 64-bit kernarg GPU VA into consecutive compute user-data words 0 and 1 for the dispatched shader. |
| Queue baseline | The C0 MQD initialization also seeds `kMqdComputeUserData0` with the low 32 bits of `kKernargsVa` (`native_amdev_transfer_probe.cpp:677-700`), while the per-dispatch PM4 sequence writes both low and high words (`:638-660`). | The dispatch packet, not an implicit host ABI, supplies the complete 64-bit value at launch. |

The product PM4 builder is a direct parameterization of the frozen C0 sequence: its user-data packet is identical to `build_compute_dispatch_words` in `native_amdev_transfer_probe.cpp:638-674` (the frozen sequence's write is at lines 657-659).  There is no alternate resident-HSA PM4 path that skips this assignment.

## RMSNorm asset ABI and SGPR requirement

The source-level ABI is four fields, all in the first 32 bytes:

| Offset | Field | Asset evidence |
| ---: | --- | --- |
| 0 | `hidden_input` pointer | `native_r9700/kernels/llama-rmsnorm-hsa-assets/llama_rmsnorm_f16.json:87-110` |
| 8 | `scale` pointer | same manifest |
| 16 | `hidden_output` pointer | same manifest |
| 24 | `epsilon` (`float32`) | same manifest |

The stage table uses the same field offsets and 32-byte schema (`native_r9700/llama_stage_layout.cpp:15, 28-43, 215-218`). `ResidentHsaSession::dispatch` starts with the stage's literal bytes, then overwrites only declared `ResidentHsaKernargBinding` 64-bit slots with the live buffer GPU VAs before calling the shared page binder (`native_r9700/amdev_session.cpp:2261-2283`). Thus PM4 supplies one pointer to this 32-byte struct; the three resource pointers are loaded indirectly by the shader from it.

The RMSNorm image contains its AMDHSA descriptor at image offset 1536. Its manifest reports:

- entry offset `5888`, which is 256-byte aligned;
- `rsrc1 = 3222208513` (`0xc00f0001`), `rsrc2 = 132` (`0x84`), and `rsrc3 = 160` (`0xa0`);
- no relocations; and
- the reviewed `llama-rmsnorm-f16-v1` 32-byte kernarg schema.

These values are from `llama_rmsnorm_f16.json:1-119`. The checked-in generator extracts the descriptor layout as `kernarg` at byte 8, `rsrc3` at byte 44, `rsrc1` at byte 48, `rsrc2` at byte 52, and `kernel_code_properties` / `kernarg_preload` at bytes 56 / 58 (`experiments/native-r9700-runtime/generate_hsa_code_image.py:907-937`). It rejects an image with a nonzero preload value and requires code properties `0x408` (`:220-221, 925-930`). The RMSNorm descriptor's 64 bytes at offset 1536 encode:

```
kernarg size:             0x20
entry delta:              0x1100  (1536 + 0x1100 = 5888)
rsrc3 / rsrc1 / rsrc2:    0x000000a0 / 0xc00f0001 / 0x00000084
code properties / preload: 0x0408 / 0x0000
```

`0x408` is the project-accepted descriptor encoding for an enabled kernarg-segment pointer with wavefront-size-32; the exact same value is used as the generator's `KERNEL_CODE_PROPERTIES` admission constant. Critically, it is not a preload configuration: the descriptor's `kernarg_preload` is zero and the generator rejects nonzero preload.

## C0 comparable proof

The project has a stronger local convention than a label alone. The frozen C0 assembly kernel `native_r9700/kernels/vram_smoke_add_gfx1201.s` states both sides explicitly:

- `s_load_b64 s[2:3], s[0:1], 0x0`, `s_load_b64 s[4:5], s[0:1], 0x8`, and `s_load_b64 s[6:7], s[0:1], 0x10` load its three kernarg pointers (lines 6-8);
- `.amdhsa_user_sgpr_kernarg_segment_ptr 1` enables that incoming pointer (line 23);
- `.amdhsa_kernarg_size 24` declares the indirect record (line 25).

The C0 manifest has `rsrc2 = 0x84` and the three-pointer layout at offsets 0/8/16 (`native_r9700/kernels/vram-smoke-assets/vram_smoke_add_gfx1201.json:5-39`). Its recorded hardware proof reports descriptor code properties `0x00000408`, successful kernarg write, successful launch, byte-exact expected output, CPU comparison pass, and `exit_status: 0` (`logs/c1-runner-kernel-proof-2026-08-23T02:57:12Z.log:38-60, 147-153`).

The checked-in HSA embed-row asset supplies a like-for-like code-image comparison, though this audit does **not** treat it as an independent hardware-success claim. Its manifest declares `rsrc2 = 132` (`0x84`) (`native_r9700/kernels/llama-hsa-assets/llama_embed_row_f16.json:84-114`). Its descriptor at image offset 1536 has the same `code properties / preload: 0x0408 / 0x0000` encoding as RMSNorm; its only relevant differences are the 24-byte ABI and its compiled `rsrc1`/`rsrc3`. That confirms the HSA asset-generation convention is consistent across the two product images, while the C0 log above supplies the hardware execution proof for the convention.

RMSNorm differs in resource counts (`rsrc1` and `rsrc3`) and ABI size (32 rather than 24 bytes), which is expected for its compiled code, but it retains the two facts relevant here: `rsrc2 = 0x84` and descriptor properties `0x408` / preload zero. Therefore its assumption that the enabled kernarg pointer arrives in the initial user-SGPR pair, as C0 assembly names `s[0:1]`, matches the PM4 user-data programming convention used by the live resident HSA path.

## What the zero-input/unit-scale sentinel can and cannot classify

`native_r9700/kernels/llama_rmsnorm_f16.cpp:6-25` makes only lane 0 execute the row: it returns other lanes at line 8, reads all 2048 input and scale elements, and stores every `hidden_output[row_offset + column]` at line 24. For row 0 with zero fp16 input, unit fp16 scale, and a finite positive epsilon, its specified expression is finite zero for every output element.

Consequently, initializing **only** the normalized-output allocation to fp16 `1.0` before the dispatch gives a useful two-part observation:

1. **Any retained `1.0` sentinel** proves the corresponding output location was not overwritten by this source's required store. That is compatible with a wrong/missing output binding, output-page/PTE visibility fault, launch/code non-execution, or a premature/faulted execution. It is not evidence that RMS arithmetic produced a nonfinite value.
2. **All 2048 locations overwritten, but nonfinite** eliminates the missing-store explanation. The remaining fault can still be address/contents related (for example a wrong input or scale pointer loaded from an otherwise correctly delivered kernarg record), or can be in shader execution/arithmetic. The PM4/descriptor evidence above makes a missing `COMPUTE_USER_DATA_0:1` programming step specifically unlikely, but a trace cannot prove the individual three pointer values or GPU page translations are correct.

This classification preserves the distinction requested: the PM4 user-data route and descriptor are compatible; output-sentinel retention diagnoses delivery/store failure, whereas an overwritten nonfinite result requires investigation of the executed kernel path and the data it actually read.
