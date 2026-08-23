# LN-2 RMSNorm arithmetic decode

## Scope and decode limit

Static analysis only; no trace, validation, compilation, hardware, or source edits were performed.

The examined executable is `native_r9700/kernels/llama-rmsnorm-hsa-assets/llama_rmsnorm_f16.image`, SHA-256 `0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0`.  Its manifest puts `.text` at image offset `0x1700` (5888), size 1664 bytes (`llama_rmsnorm_f16.json:59-63`).

Instruction names, opcodes, field layouts, and the stated instruction semantics below were decoded from AMD's public machine-readable RDNA4 ISA specification (August 2026).  The flattened code image contains neither local symbols nor debug line information, and there is no installed AMDGPU disassembler.  Thus, the source-to-ISA correlation is based on the exact executable fields and control/data dependencies, not compiler debug locations.  Where a full scalar micro-sequence would require symbolic control-flow reconstruction, this report gives the exact decoded instructions and does not claim more.

## Source operation being lowered

`native_r9700/kernels/llama_rmsnorm_f16.cpp` establishes the complete intended arithmetic:

- `:11-16` initializes `sum_of_squares` to `0.0f`, converts each fp16 input to f32, and accumulates `value * value`.
- `:18-19` computes `inverse_rms = 1.0f / sqrtf(sum_of_squares * (1.0f / 2048.0f) + epsilon)`.
- `:20-24` converts input and scale from fp16, multiplies them with `inverse_rms`, converts the result to fp16, and stores it.

For a zero input and positive finite epsilon this is exactly `0 * 1 * (1 / sqrt(epsilon))`, which is finite zero.  The executable also loads epsilon from kernarg byte 24 at `0x1820` (`f4000000 f8000018`, `s_load_b32 s0, s[0:1], 0x18`), so the arithmetic block below receives the actual ABI scalar.

## Executable arithmetic sequence

### Square accumulation and RMS argument

The reduction contains mixed-precision fused multiply-add operations.  Its first pair is:

| `.text` offset | Little-endian words | Decoded instruction and operands | Correlation |
|---|---|---|---|
| `0x1774` | `cc200009 18360301` | `V_FMA_MIX_F32 v9, v1, v1, s13` | square/add in f32 from mixed fp16/f32 sources |
| `0x1780` | `cc201801 1c260301` | `V_FMA_MIX_F32 v1, v1, v1, v9` | paired square/add |

The same `V_FMA_MIX_F32` reduction shape occurs at `0x1788`, `0x1794`, `0x179c`, `0x17a8`, `0x17b0`, `0x17bc`, `0x17c8`, `0x17d4`, `0x17dc`, `0x17e8`, `0x17f0`, `0x17fc`, `0x1804`, and `0x1810`.  `V_FMA_MIX_F32` is the RDNA4 operation that fuses the multiply and add while accepting mixed half/single inputs.  The accumulator is recovered to scalar state by:

| Offset | Words | Decoded instruction |
|---|---|---|
| `0x1818` | `7e1a0501` | `V_READFIRSTLANE_B32 s13, v1` |

The exact mean-plus-epsilon operation is then present directly after the epsilon load:

| Offset | Words | Decoded instruction | Meaning |
|---|---|---|---|
| `0x1820` | `f4000000 f8000018` | `S_LOAD_B32 s0, s[0:1], 0x18` | epsilon -> `s0` |
| `0x182c` | `a300000d 3a000000` | `S_FMAMK_F32 s0, s13, 0x3a000000, s0` | `s0 = s13 * 0.00048828125f + s0`; `0x3a000000` is exactly `1/2048` |

`S_FMAMK_F32` is scalar fused multiply-add with a literal multiplier.  This is a direct lowering of source line 19, not merely a plausible interpretation of the code shape.

### Square-root / reciprocal lowering

The compiler does not emit a single `V_RSQ_F32`.  It emits an expanded square-root/division sequence.  The exact, relevant decoded instructions are:

| Offset | Words | Decoded operation |
|---|---|---|
| `0x1838` | `a201ff00 4f800000` | `S_MUL_F32 s1, s0, 0x4f800000` |
| `0x1850` | `d6880004 00000001` | `V_S_SQRT_F32 s4, s1` |
| `0x18e4` | `d6fc7c00 03c80000` | `V_DIV_SCALE_F32` |
| `0x18ec` | `d6fc6a03 03c800f2` | `V_DIV_SCALE_F32` |
| `0x18f8` | `7e025500` | `V_RCP_F32 v1, s0` |
| `0x1908` | `d6130002 03ca0300` | `V_FMA_F32 v2, v0, v1, 1.0f` |
| `0x1914` | `56020302` | `V_FMAC_F32 v1, v2, v1` |
| `0x1918` | `10040303` | `V_MUL_F32 v2, v3, v1` |
| `0x1920` | `d6130004 040e0500` | `V_FMA_F32 v4, v0, v2, v3` |
| `0x1928` | `56040304` | `V_FMAC_F32 v2, v4, v1` |
| `0x1930` | `56060500` | `V_FMAC_F32 v3, v0, v2` |
| `0x1934` | `d6370000 040a0303` | `V_DIV_FMAS_F32` |
| `0x1944` | `d6270000 03c80100` | `V_DIV_FIXUP_F32` |

`0x4f800000` is f32 `4294967296.0f`.  The surrounding scalar condition/select and scalar FMA instructions at `0x1840-0x18dc` are the associated special-case/refinement setup.  The exact dependency chain through all of those scalar temporaries is the decode limit stated above; it is sufficient to establish that this is a compiler-generated sqrt-plus-reciprocal/division lowering rather than a source-level absence of epsilon or an ABI issue.

### Final product, conversion, and store

The first unrolled output group is:

| Offset | Words | Decoded operation |
|---|---|---|
| `0x195c` | `ee048008 00000002 00000001` | `GLOBAL_LOAD_U16 v2, [s8 + v1 + 0]` (input) |
| `0x1968` | `ee04800a 00000003 00000001` | `GLOBAL_LOAD_U16 v3, [s10 + v1 + 0]` (scale) |
| `0x1990` | `7e041702` | `V_CVT_F32_F16 v2, v2` |
| `0x1998` | `7e061703` | `V_CVT_F32_F16 v3, v3` |
| `0x199c` | `10040702` | `V_MUL_F32 v2, v2, v3` |
| `0x19a4` | `cc210002 02020500` | `V_FMA_MIXLO_F16 v2, v0, v2, 0` |
| `0x19ac` | `ee06400c 01000000 00000001` | `GLOBAL_STORE_B16 [s12 + v1 + 0], v2` |

`V_FMA_MIXLO_F16` accepts mixed f16/f32 inputs and converts the fused result to fp16 low bits; here it implements the final inverse-RMS multiplication/conversion with a zero addend before the B16 store.  The input/scale/load, convert/product, mixed-fma conversion, and B16-store structure repeats at offsets +2 through +14 of the eight-element unroll.

## Zero-store comparison

`llama_rmsnorm_zero_store_f16.cpp` retains the same four-argument signature and row/lane guard, but explicitly discards input, scale, and epsilon (`:10-12`) and writes `0U` in its output loop (`:13-16`).  Its manifest keeps the `.text` entry at `0x1700` but reduces code from 1664 to 640 bytes (`llama_rmsnorm_zero_store_f16.json:59-63`).

The resource fields differ only as expected for the removed arithmetic/data paths:

| Field | RMSNorm | Zero store |
|---|---:|---:|
| `rsrc1` | `0xc00f0001` | `0xc00f0000` |
| `rsrc2` | `0x84` | `0x84` |
| `rsrc3` | `0xa0` | `0x20` |
| `.text` bytes | 1664 | 640 |

The zero-store code contains only address/control and stores for this operation; its sole vector operation in the prefix is `V_MOV_B32 v2, s6` at `0x1740`, not floating-point norm arithmetic.  Its first output stores begin at `0x1764` as `GLOBAL_STORE_B16` operations.  Therefore the known-good zero-store execution excludes the shared dispatch/output mapping enough to focus the remaining failure on arithmetic/code-generation/execution of the source kernel, while not proving a particular instruction is faulty.

## Highest-confidence arithmetic fault candidate

**Candidate: the expanded `sqrt` / reciprocal-division lowering spanning `0x1838-0x194c`, especially the `V_S_SQRT_F32` followed by `V_DIV_SCALE_F32`, `V_RCP_F32`, Newton-style FMAs, `V_DIV_FMAS_F32`, and `V_DIV_FIXUP_F32`.**

This is the highest-confidence candidate because it is the only arithmetic unique to computing the nonzero normalization factor.  With zero input, every mixed FMA reduction term and final input/scale product is zero; neither can by itself make the specified inputs nonfinite.  The zero-store image omits this complete block and produces finite values under the same trace dispatch, output VA, kernarg ABI, and PM4 path.  The candidate is not a claim that `V_S_SQRT_F32` itself has been proven defective: the current static evidence cannot distinguish a code-generation defect in this lowering from an execution-state defect affecting it.  A scalar epsilon/sqrt/reciprocal probe is the smallest justified next discrimination.
