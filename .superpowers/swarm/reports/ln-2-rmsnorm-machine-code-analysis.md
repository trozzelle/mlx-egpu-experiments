# LN-2 RMSNorm machine-code analysis

## Scope and method

Static analysis only. The examined artifact is
`native_r9700/kernels/llama-rmsnorm-hsa-assets/llama_rmsnorm_f16.image`, digest
`0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0`.
The manifest places its `.text` at image offset `0x1700` (5888), with size
1664 bytes, and identifies that as the entry offset
(`llama_rmsnorm_f16.json:60-63,85-86`).

I decoded the actual little-endian instruction words from that `.text` using
the gfx1201/RDNA4 SMEM and VGLOBAL field layout used by the generated-asset
compiler path. In particular, SMEM is an 8-byte instruction with opcode
bits `[18:13]`, `sdata[12:6]`, `sbase[5:0]`, and a 24-bit immediate in the
second word; VGLOBAL is a 12-byte instruction with opcode `[21:14]`, scalar
address `saddr[6:0]`, and its vector operands/offset in the following two
words. This is code-image decode, not an inference from the descriptor.

## Actual kernarg loads

The first three SMEM loads in the machine code are exactly the 32-byte
kernarg ABI:

| Image offset | Little-endian instruction words | Decoded instruction | Karg bytes read | ABI field(s) |
|---|---|---|---|---|
| `0x1714` | `f4004100 f8000000` | `s_load_b128 s[4:7], s[0:1], 0x0` | `0..15` | `hidden_input` (u64 @ 0) and `scale` (u64 @ 8) |
| `0x171c` | `f4002200 f8000010` | `s_load_b64 s[8:9], s[0:1], 0x10` | `16..23` | `hidden_output` (u64 @ 16) |
| `0x1820` | `f4000000 f8000018` | `s_load_b32 s0, s[0:1], 0x18` | `24..27` | `epsilon` (f32 @ 24) |

Details that make this unambiguous:

- The three opcodes are respectively RDNA4 `SMEMOp.S_LOAD_B128 = 2`,
  `S_LOAD_B64 = 1`, and `S_LOAD_B32 = 0`.
- Each uses `sbase=0`, therefore the same kernarg base `s[0:1]`; each has
  `soffset=NULL` and the listed immediate byte offset.
- The first 16-byte transfer is contiguous, so it cannot exchange the first
  two pointers without the code itself loading different bytes.

The generated asset tooling freezes the same schema at
`generate_hsa_code_image.py:37-47`, requires the compiler-reported kernarg
allocation to equal that schema at `:919-936`, and publishes it in this
asset's manifest at `llama_rmsnorm_f16.json:87-111`.

## Actual data loads and output stores

The output loop's memory triplet is present in the machine code. The first
iteration is:

| Image offset | Words | Decoded operation | Operand facts |
|---|---|---|---|
| `0x195c` | `ee048008 00000002 00000001` | `global_load_u16` | op 18, `vdst=v2`, `saddr=s8`, `vaddr=v1`, `ioffset=0` |
| `0x1968` | `ee04800a 00000003 00000001` | `global_load_u16` | op 18, `vdst=v3`, `saddr=s10`, `vaddr=v1`, `ioffset=0` |
| `0x19ac` | `ee06400c 01000000 00000001` | `global_store_b16` | op 25, `saddr=s12`, `vaddr=v1`, **`vsrc=v2`**, `ioffset=0` |

The same three-instruction shape repeats with `ioffset` 2, 4, 6, 8, 10, 12,
and 14 at the following offsets:

- input-like reads (`saddr=s8`, result `v2`): `0x19bc`, `0x1a04`, `0x1a4c`,
  `0x1a94`, `0x1adc`, `0x1b24`, `0x1b6c`;
- scale-like reads (`saddr=s10`, result `v3`): `0x19c8`, `0x1a10`, `0x1a58`,
  `0x1aa0`, `0x1ae8`, `0x1b30`, `0x1b78`;
- output stores (`saddr=s12`, source `v2`): `0x19f4`, `0x1a3c`, `0x1a84`,
  `0x1acc`, `0x1b14`, `0x1b5c`, `0x1ba4`.

`GLOBAL_LOAD_U16` is RDNA4 VGLOBAL opcode 18 and `GLOBAL_STORE_B16` is
opcode 25. The store is therefore a real 16-bit global write from `v2`, not
a descriptor-only assertion and not a store through the scale read's `v3`.
The two reads followed by arithmetic and the `v2` store match the HIP output
loop's `input`, then `weight`, then `hidden_output` sequence
(`llama_rmsnorm_f16.cpp:20-24`). Its preceding reduction loop reads only
`hidden_input` (`:10-19`).

The complete 1664-byte text has no local symbol/debug information in the
flattened image. A target AMDGPU disassembler is not installed in this
worktree, so this report does **not** claim a full symbolic reconstruction of
the scalar/VGPR address-building instructions between the three kernarg loads
and the VGLOBAL operations. The exact `s8`, `s10`, and `s12` names above are
instruction operands, and the direct kernarg byte loads are fully decoded.
This limitation does not affect the demonstrated kernarg offsets or the fact
that the executable image performs B16 output stores.

## Source ABI and host bind order

All four independently represented contracts agree:

| Role | HIP signature | generated schema / manifest | stage bind order |
|---|---|---|---|
| input | `hidden_input` arg 0 | offset 0, u64 | `hidden` @ 0 |
| scale | `scale` arg 1 | offset 8, u64 | `input_layernorm_weight` @ 8 |
| output | `hidden_output` arg 2 | offset 16, u64 | `normalized` @ 16 |
| epsilon | `epsilon` arg 3 | offset 24, f32 | `epsilon` @ 24 |

The HIP signature is at `llama_rmsnorm_f16.cpp:1-5`. The host stage layout is
at `llama_stage_layout.cpp:28-43`; it declares a 32-byte RMSNorm kernarg
record at `:215-218` and rejects a non-finite or non-positive epsilon at
`:426-430`. Trace kernarg materialization writes every buffer VA at its
registered byte offset (`runtime_contract.cpp:173-190`) and identifies
RMSNorm's scalar as epsilon at byte 24 (`:200-221`).

## Finding

**No code-image/source ABI mismatch was found.** The actual executable loads
`{input@0, scale@8, output@16, epsilon@24}` in that order and issues concrete
16-bit global output stores after two 16-bit reads. In particular, there is
no observed swap of scale/output, no epsilon read at a pointer slot, and no
missing output store that could turn a valid zero-input/unit-scale invocation
into NaN.

For zero input and finite positive epsilon, the HIP expression is
`0 * 1 * (1 / sqrt(epsilon))`, hence finite zero
(`llama_rmsnorm_f16.cpp:18-24`). Since the machine-code kernarg byte offsets
agree with the source and bind order, this evidence leaves input delivery,
execution-state correctness, or a code-generation/arithmetic issue beyond
these decoded ABI and memory instructions as the remaining classes; it does
not support changing the RMSNorm ABI or its scale contents.
