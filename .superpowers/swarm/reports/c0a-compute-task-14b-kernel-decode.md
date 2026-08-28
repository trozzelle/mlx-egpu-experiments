# C0A23 Task 14b — Embedded Kernel Store-Format Decode

**Date:** 2026-08-18
**Wave:** C0A Compute 23
**Assignee:** C0A23KernelDecode
**Files touched:**
- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`

---

## 1. Bottom line (derived values)

The 21-word program embedded in `kKernelText` is **RDNA4 / gfx1201** machine code
(`kKernelArch = "gfx1201"`, `kKernelBlobTarget = "gfx1201"`, probe lines 105/110 —
AMD Radeon AI PRO R9700). It contains **exactly one store instruction**:

| Field | Derived value |
|---|---|
| `store_instruction_count` | `1` |
| `store_class` | `global` (VGLOBAL) |
| `store_primary_op` | `GLOBAL_STORE_B128` (`VGLOBALOp` = 29) |
| `store_addressing` | `base+offset` (register base `v[4:5]`/`s[4:5]` + `ioffset=0`; **not** ADDTID) |
| `store_element_bounds` | `0..3` (128-bit store = 4 u32 elements, `ioffset=0`) |

The plan's original expectation (`rdna3`, 4 flat/global stores, `store_class: flat_or_global`)
is **superseded** by the authoritative decode: this kernel is RDNA4 and has a *single*
global 128-bit store.

---

## 2. Source grounding

Decode was performed (read-only) against tinygrad (RDNA4 tables):

- tinygrad `tinygrad/runtime/autogen/amd/rdna4/enum.py:631` → `VGLOBALOp.GLOBAL_STORE_B128 = 29`
- tinygrad `tinygrad/runtime/autogen/amd/rdna4/ins.py:107` → `class VGLOBAL` with
  `encoding = FixedBitField(31, 24, 0b11101110)` and `op = EnumBitField(21, 14, VGLOBALOp)`
- `Inst._base_size = (max(field.hi) + 8) / 8` (tinygrad `renderer/amd/dsl.py`, class `Inst`) gives
  RDNA4 instruction sizes used by the walker.

Decoding was verified with tinygrad's own disassembler
(`tinygrad.renderer.amd.decode_inst(..., arch="rdna4")`), not just by hand:

```
word12 @0x44 0xee074004 -> VGLOBAL global_store_b128(...)  size=12
```

rdna3 tables cannot decode this program (every word after `0x10` fails the rdna3
encoding table); rdna4 decodes all 21 words cleanly — confirming the gfx1201/RDNA4 target.

---

## 3. 21-word bitfield decode

Program = bytes `0x00..0x53` of `kKernelText` (defined at probe line 143; the array is
512 bytes total, `kKernelReferenceTextByteCount`). Each row lists the little-endian u32
word, its top-bit encoding fields, the decoded RDNA4 instruction class/op, and the
instruction size (words consumed).

| # | off | word | enc bit fields | class | op | size |
|---|---|---|---|---|---|---|
| 0 | 0x00 | `0xf4004100` | enc[31:26]=`111101` | SMEM | `S_LOAD_B128` | 8 |
| 1 | 0x08 | `0x30080084` | bit31=`0` | VOP2 | `V_LSHLREV_B32` | 4 |
| 2 | 0x0c | `0xf4002000` | enc[31:26]=`111101` | SMEM | `S_LOAD_B64` | 8 |
| 3 | 0x14 | `0xbfc70000` | enc[31:23]=`101111111` | SOPP | `S_WAIT_KMCNT` | 4 |
| 4 | 0x18 | `0xee05c006` | enc[31:24]=`11101110` op=23 | VGLOBAL | `GLOBAL_LOAD_B128` | 12 |
| 5 | 0x24 | `0xf4000000` | enc[31:26]=`111101` | SMEM | `S_LOAD_B32` | 8 |
| 6 | 0x2c | `0xbfc00000` | enc[31:23]=`101111111` | SOPP | `S_WAIT_LOADCNT` | 4 |
| 7 | 0x30 | `0xbfc70000` | enc[31:23]=`101111111` | SOPP | `S_WAIT_KMCNT` | 4 |
| 8 | 0x34 | `0x4a020200` | bit31=`0` op[30:25]=`010010` | VOP2 | `V_ADD_NC_U32` (v[1]+=s[0]) | 4 |
| 9 | 0x38 | `0x4a040400` | bit31=`0` op[30:25]=`010010` | VOP2 | `V_ADD_NC_U32` (v[2]+=s[0]) | 4 |
| 10 | 0x3c | `0x4a060600` | bit31=`0` op[30:25]=`010010` | VOP2 | `V_ADD_NC_U32` (v[3]+=s[0]) | 4 |
| 11 | 0x40 | `0x4a000000` | bit31=`0` op[30:25]=`010010` | VOP2 | `V_ADD_NC_U32` (v[0]+=s[0]) | 4 |
| 12 | 0x44 | `0xee074004` | enc[31:24]=`11101110` op[21:14]=`00011101`=29 | VGLOBAL | **`GLOBAL_STORE_B128`** | 12 |
| 13 | 0x50 | `0xbfb00000` | enc[31:23]=`101111111` | SOPP | `S_ENDPGM` | 4 |

The words in the plan's list correspond to the *instruction-stream* words; the walker
advances by `Inst._base_size` (SMEM=8, VOP2=4, SOPP=4, VGLOBAL/VFLAT/VSCRATCH=12 bytes),
so the 21 source words pack into 14 decoded instructions above. Rows 8-11 are the four
`v_add_nc_u32` adds (these are the `0x4a` words the plan mis-read as stores; per rdna3
they are `V_MAX_I32` VOP2, per rdna4 `V_ADD_NC_U32` — **not** store ops). Row 12 is the
single store.

---

## 4. Exact enum names/values cited (rdna4)

- `VGLOBALOp.GLOBAL_STORE_B128 = 29` (`enum.py:631`)
- class `VGLOBAL`: `encoding = FixedBitField(31,24,0b11101110)`, `op = EnumBitField(21,14,…)`
  (`ins.py:107-109`)
- `VGLOBALOp.GLOBAL_LOAD_B128 = 23`
- `GLOBAL_STORE_ADDTID_B32 = 41` — *not* used; the actual store op is `29`, so addressing is
  base+offset, not work-item-indexed.
- `VOP2Op.V_ADD_NC_U32 = 37` (rdna4 `enum.py`, `VOP2` class `ins.py:158-159`).
- Store-class op values shared across VGLOBAL/VFLAT/VSCRATCH: `*_STORE_B8=24 … B128=29`,
  `*_STORE_D16_HI_B8=36`, `*_STORE_D16_HI_B16=37`, `*_STORE_ADDTID_B32=41`.

---

## 5. Cause hypothesis

### (a) 16-bit halfword swap on elements 0..3

**Confirmed (decoded operands):** the only store is `global_store_b128` with
`vsrc = v[0:3]` (a 128-bit packed vector = 4 software-u32 lanes) and the four preceding
`v_add_nc_u32` ops are 32-bit-per-lane adds. The observed readback signature
(`kKernelObservedOutputBytesHex`, probe line 120) is
`00000200000003000000040000000500 …` i.e. elements `0x00020000, 0x00030000, 0x00040000, 0x00050000`
— each expected u32 `2,3,4,5` with its two 16-bit halves rotated.

**Inferred (mechanism, best-supported hypothesis):** the 128-bit packed store/load path
commits the four 32-bit ALU lanes as eight 16-bit memory lanes; the vector unit's packed-lane
layout interleaves the two halves of adjacent u32 so each element's high/low 16-bit halves
land in swapped positions. This is a **32-bit-compute vs 16-bit-packed-128 memory** lane-width
mismatch in the embedded assembly (the compiler chose B128 rather than
`GLOBAL_STORE_D16_HI_B16`, yet the layout behaves D16-swizzled), producing the observed
halfword rotation on the 4 written elements. The previous Task-5 report's `c0l` signature is
exactly this `swap_and_partial` class.

### (b) Only 4 of 8 elements written

**Confirmed:** the single store covers 128 bits = 4 u32 (elements 0..3) and is addressed
`base+offset` — address from `v[4:5]`/`s[4:5]` register base with `ioffset=0`, **not**
`GLOBAL_STORE_ADDTID_B32` (op 41). With `kDispatchGlobalSizeX = 2` (probe line 469) two
work-items launch, but both use the same work-item-invariant base + offset, so both write
the *same* 128-bit slice (elements 0..3).

**Inferred (mechanism):** a work-item-indexed version of the kernel (each work-item writing
a distinct 128-bit slice, elements 4..7 for work-item 1 via ADDTID) would have produced 8 of 8.
Because the embedded program dropped the ADDTID addressing and relies on a fixed
base+`ioffset=0`, the second work-item aliases the first's slice. Result: exactly elements
0..3 are nonzero, elements 4..7 remain zero — **4 of 8**.

**Confirmed / inferred summary:**
- CONFIRMED: one store, `GLOBAL_STORE_B128`, base+offset addressing, `ioffset=0`,
  `vsrc=v[0:3]` (4 u32), global_size_x=2.
- CONFIRMED: the observed outcome is `swap_and_partial` (4 written, all halfword-swapped).
- INFERRED: the precise packed-lane/ALV lane-width interleave that mechanically produces the
  halfword rotation, and the exact reason the compiler dropped ADDTID. Both are the
  best-supported mechanisms consistent with the decoded operands.

---

## 6. Deliverables (code)

### `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Added `int run_kernel_text_decode_self_test()` (line 4619): walks the 21-word program with a
  small hard-coded local decode table (no tinygrad runtime dependency), counts store-class
  instructions, and emits the required lines. Guards against drift with `self_test_failure`.
- Registered `--self-test kernel-text-decode` in `main()` (after `compute-readback-classifier`)
  and in `print_help()` (line 6611).

### `tests/test_native_amdev_transfer_contract.py`
- `EXPECTED_KERNEL_TEXT_DECODE_LINES` (line 402) with the derived values.
- `test_kernel_text_decode_self_test_reports_store_ops` (line 623) mirroring
  `test_mec_rs64_pipe_activation_self_test_reports_steady_state_encoding`.
- Added `--self-test kernel-text-decode` to `test_help_lists_hardware_modes` (line 662).

---

## 7. Supervisor validation commands (to run later)

```bash
# (a) Focused pytest for the new self-test
${PY} -m pytest \
  'tests/test_native_amdev_transfer_contract.py::test_kernel_text_decode_self_test_reports_store_ops' -v

# (b) Build (compile) the probe standalone
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o /tmp/native_amdev_transfer_probe

# (c) Full contract suite
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -q
```

Executors did not run any of these (per wave constraint); the supervisor validates.
