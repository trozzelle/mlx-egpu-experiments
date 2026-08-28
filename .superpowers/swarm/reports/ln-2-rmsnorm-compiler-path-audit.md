# LN-2 RMSNorm compiler-path audit

## Scope

Static audit only: the checked-in RMSNorm source, direct-COMGR generator, current
`gfx1201` image/manifest, and the resident-VRAM finite smoke asset. No generator,
test, executor, or hardware command was run.

## Source-level result

`native_r9700/kernels/llama_rmsnorm_f16.cpp:1-26` is one C-linkage GPU kernel.
Only lane zero executes: it accumulates 2,048 fp16 input values in fp32, computes
`1 / sqrt(sum_of_squares / 2048 + epsilon)`, then writes 2,048 fp16 products to
`hidden_output`. There is no inline assembly, barrier, LDS/shared memory, scratch
allocation, or cross-lane dependence after lanes 1--63 return.

For all-zero input, unit fp16 scale, and the trace route's required strictly
positive finite epsilon, the source computation is finite and writes fp16 zeroes.
The route rejects a non-finite or `<= 0.0F` epsilon before materializing kernargs
(`native_r9700/llama_stage_layout.cpp:426-430`). This qualification matters:
source-level `epsilon == 0` would divide by zero, and a negative epsilon would
sqrt a negative value; merely saying "finite epsilon" is not independently
sufficient in C/IEEE arithmetic.

The current source digest is
`67d2d8f4e4acf13c9380530fbbbcf5fa96b953509457d514dea2e191405e961a`, equal to
`llama_rmsnorm_f16.json:117-119`.

## Exact source-to-image path

`generate_hsa_code_image.py` has no compiler wrapper or alternate target path:

1. `_reviewed_asset` accepts only a source path listed in `REVIEWED_ASSETS`
   (`:940-949`); the RMSNorm tuple fixes kernel name
   `llama_rmsnorm_f16`, the three pointer names in ABI order, scalar
   `epsilon: float`, and a **28-byte compiler ABI** (`:272-280`).
2. `_read_reviewed_source` rejects symlinks, requires the resolved checked-in
   canonical source and a stable regular-file inode while reading (`:951-990`).
   `validate_source_profile` then requires exactly one matching C-linkage global
   kernel and precisely the expected pointer/scalar argument count and names
   (`:478-534`). It is an admission check, not a semantic proof of generated
   arithmetic.
3. `generate` calls Tinygrad's `compile_hip(source_text, "gfx1201", asm=False)`
   (`:1189-1219`). The active Tinygrad implementation is
   `<tinygrad-checkout>/tinygrad/runtime/support/compiler_amd.py`.
   It sets ISA `amdgcn-amd-amdhsa--gfx1201` and HIP language (`:48-52`), compiles
   source to bitcode with `-O3 -mcumode --hip-version=6.0.32830`,
   `-D__HIPCC_RTC__`, C++14, `-nogpuinc`, `--offload-arch=gfx1201`, and
   `-Xclang -disable-llvm-passes -Xclang -aux-triple -Xclang
   x86_64-unknown-linux-gnu` (`:69-81`). It then codegens bitcode with
   `-O3 -mllvm -amdgpu-internalize-symbols` and links it (`:82-87`).
4. The generator admits a constrained allocated-ELF layout, applies ELF
   relocations, obtains the entry from the unique kernel ELF symbol, requires a
   256-byte aligned entry, and derives `rsrc1/2/3` from the copied 64-byte
   `.rodata` descriptor (`generate_hsa_code_image.py:1188-1275`, `:907-937`).
   It rejects compiler `kernarg_size` other than the 28-byte logical ABI, then
   changes that field in the emitted image to the declared 32-byte allocation
   size. Thus the final descriptor's 32 bytes are intentional tail padding, not
   four additional source arguments.

No listed option enables fast/unsafe floating-point transformations. A conforming
compiler therefore has no source-language permission to turn the qualified zero
case into nonfinite results. The generator nevertheless cannot prove arithmetic
semantics: it neither freezes the Tinygrad/COMGR toolchain revision nor compares
disassembly/IR to source. A compiler/codegen defect can produce an ELF image that
passes every structural, resource, and ABI check above. The zero-store diagnostic
is consequently useful precisely because it removes arithmetic from that
unverified compilation/codegen surface while retaining the same downstream image,
resource, and dispatch path.

## Current emitted RMSNorm asset

`native_r9700/kernels/llama-rmsnorm-hsa-assets/llama_rmsnorm_f16.json` records:

| property | value |
| --- | --- |
| target | `gfx1201` |
| image digest / bytes | `0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0` / 15,857 |
| ABI | `llama-rmsnorm-f16-v1`; 32 bytes: pointers at 0, 8, 16; `float32 epsilon` at 24 (`:87-111`) |
| descriptor / entry | 1536 / 5888 (`:2,21,53-63`) |
| resources | `rsrc1=0xc00f0001`, `rsrc2=0x00000084`, `rsrc3=0x000000a0` (`:113-116`) |
| allocation/relocation shape | ten admitted sections, zero relocations, one kernel symbol target (`:3-19`) |

A direct decode of the checked-in image at offset 1536 agrees with the manifest:
`group=0`, `private=0`, `kernarg=32`, `entry_delta=0x1100`,
`rsrc3=0x000000a0`, `rsrc1=0xc00f0001`, `rsrc2=0x00000084`,
`properties=0x0408`, and `preload=0`. In particular,
`1536 + 0x1100 = 5888`. The current image digest also equals the manifest.
There is no presently observed descriptor/manifest resource mismatch.

## Known finite HSA convention

The resident VRAM smoke asset is an independently observed finite hardware
execution: `logs/c1-runner-vram-smoke-2026-08-23T02:57:14Z.log:1-48` records one
`gfx1201` native dispatch with `cpu_comparison_status: pass`, `failure_stage:
none`, and `exit_status: 0`. Its asset metadata uses descriptor-derived
`rsrc1=0xe0100000`, `rsrc2=0x00000084`, `rsrc3=0x00000010`, target `gfx1201`, a
24-byte three-pointer ABI, 64-wide workgroup, and zero LDS
(`native_r9700/kernels/vram-smoke-assets/vram_smoke_add_gfx1201.json:1-39`).
The assembly explicitly specifies its AMDHSA descriptor and matching target/ABI
metadata (`native_r9700/kernels/vram_smoke_add_gfx1201.s:18-63`); its generator
reads `compute_pgm_rsrc1/2/3` from that descriptor and labels the metadata
`source_amdgpu_metadata` (`generate_vram_smoke_add_gfx1201_asset.py:305-325,
:363-380`).

This proves that the native PM4/resource convention accepts `gfx1201` code with
literal descriptor-derived resource words and a 64-wide dispatch. It does **not**
prove that a direct-COMGR C++ image's instruction stream is correct, and its
resource words must not be copied to another kernel: register allocation changes
`rsrc1`/`rsrc3` per emitted code.

## Constraints for the zero-store asset

1. It must be a separately registered `REVIEWED_ASSETS` source. An arbitrary
   temporary/copied source will be rejected; source, kernel name, canonical path,
   expected schema, pointer names/order, scalar name/type, and 28-byte compiler
   ABI must be added together.
2. Retain the RMSNorm ABI exactly: three 64-bit arguments in this order
   `hidden_input` @0, `scale` @8, `hidden_output` @16, then `float epsilon` @24.
   The generated runtime descriptor must remain 32 bytes after the generator's
   tail-padding patch. The zero-store source may deliberately leave the first two
   pointers and epsilon unread, but it must retain them as ABI parameters.
3. Retain the trace geometry: workgroup X 64, one output-producing lane, and the
   same `row * 2048 + column` fp16 output extent. Store bit-pattern `0x0000` to
   every output element; do not load hidden input/scale, perform normalization,
   use epsilon, use LDS/barriers, or add a second dispatch. This makes all output
   finite zero independently of input/scale/epsilon arithmetic.
4. Generate through this same direct-COMGR `gfx1201`, `asm=False` path; do not
   substitute the known-good hand-written smoke assembly. The resulting image
   must supply its own descriptor-derived `rsrc1/2/3`, entry offset, image digest,
   and ELF layout. Do not reuse RMSNorm or smoke resource constants.
5. Integrate only as the trace diagnostic mode: bind the existing 32-byte kernarg
   payload and existing output mapping, preserve the finite sentinel prefill, and
   keep accepted prefill/KV/parity/later-stage routes untouched. Existing
   nonfinite diagnostics must remain the failure-reporting path.

**Decision:** Current evidence does not identify a legal source-level arithmetic
route from zero input, unit scale, and the validated positive finite epsilon to
nonfinite output. It does establish that the compiler/generator's structural
checks cannot exclude a direct-COMGR code-generation defect. A same-ABI
zero-store asset is therefore the narrow discriminator: finite zero output
isolates the fault to RMSNorm arithmetic/codegen; nonfinite output leaves image
mapping, resource programming, or output mapping suspect.
