# C0 MQD Address-Field Byte-Swap — Re-derivation Result

Date: 2026-08-23
Status: root cause localized; fix not yet applied

## Symptom

`--kernel-proof` reaches the doorbell (`doorbell_hit=1`) but MEC raises
`rs64_exception_status=0xc67a` = page-fault(bit3) + misaligned-addr(bit1).
The HQD ring-base register reads back byte-swapped:

```
hqd_pq_base=0x00000070, hqd_pq_base_hi=0x00000020
```

Correct encode for ring `0x2000007000` is `PQ_BASE=0x20000070, HI=0x00`.

## Re-derivation findings (all verified against source)

1. **Encoding is correct.** `build_compute_mqd()` encodes the address fields
   byte-for-byte identically to tinygrad's reference (`ip.py:323-334`):
   `pq_base = lo32(addr>>8)/hi32(addr>>8)`, `rptr_report = lo32(addr)/hi32(addr)`,
   `wptr_poll = lo32(addr)/hi32(addr)`, `eop = lo32(addr>>8)/hi32(addr>>8)`.

2. **Register-copy mapping is correct.** `kMqdHqdRegisterCopyStart = 0x80`,
   `kMqdCpMqdBaseAddrLo = 0x80`, `kCpMqdBaseAddr = 8105`, `kCpHqdPqBase = 8113` —
   identical to tinygrad's `mqd_st_mv[0x80+i] -> wreg(CP_MQD_BASE_ADDR+i)`.

3. **RPC write/read path is byte-faithful.** `append_u32_le`, `mmio_write_bar0`,
   and `RemoteClient::rpc` all send/receive raw little-endian bytes with no swap.
   Confirmed empirically: tinygrad BAR0 write+read of `0x20000070` round-trips
   byte-exactly.

4. **The corruption is introduced by the MEC, not the CPU.** The native runner's
   `write_and_verify_compute_mqd` verify passes (VRAM MQD correct at write time).
   After the MEC processes the queue, both the HQD register and the VRAM MQD read
   back byte-swapped. The MEC's MQD load (or context-save write-back) byte-swaps
   the address fields.

## Strong lead: gfx1201 removed `endian_swap`

`regCP_HQD_PQ_CONTROL` in older gfx blocks (e.g. regs offset 4694) has an
`endian_swap` field at bits (17,18). The gfx1201 block (offset 8122) **removes**
it (replaced by `slot_based_wptr` at (18,19)). This correlates with the observed
MEC endianness behavior on gfx1201.

## Conclusion

The original "byte-swap the encoding" fix is **not** correct — the encoding is
already right. The bug is gfx1201 MEC MQD-load endianness. Fix options:

1. **Empirically calibrate the byte-swap** and compensate in the VRAM MQD write
   (only the address fields; control fields are unaffected).
2. **Investigate the gfx1201 VRAM MQD layout** (the fields may be byte-packed in
   the on-VRAM MQD format vs the 32-bit register copy format).
3. **Match tinygrad's queue setup exactly** (write HQD registers via MMIO only;
   drop the VRAM MQD write for the active-queue path).
## Sourcing verification (complete)

- **Layout correct**: `struct_v12_compute_mqd` (tinygrad `am.py:1821`) field offsets
  match the native runner's indices exactly (`cp_hqd_pq_base_lo` = byte 0x220/dword
  0x88, `_hi` = 0x224/0x89, rptr/wptr report/poll = 0x22c/0x230/0x234/0x238).
- **Encoding correct**: amdgpu gfx12 KFD MQD manager
  (`kfd_mqd_manager_v12.c`) sets `cp_hqd_pq_base_lo = lower_32_bits(queue_address >> 8)`,
  no byte swap. Identical to the native runner and tinygrad.
- **RPC frame correct**: `build_remote_cmd_frame` emits `<BIIQQQ` little-endian,
  33 bytes — identical to tinygrad `system.py:367`.
- **Large write byte-faithful**: live probe wrote a 2048-byte buffer with a
  sentinel u32 at offset 0x220 via the same bridge; round-trip exact, no swap.

Conclusion: every CPU-side layer is verified correct. The corruption is the
**gfx1201 MEC MQD-load endianness** — the MEC reads the address fields
byte-swapped (non-trivial permutation, not a clean 32-bit swap). This requires
empirical calibration to determine the exact permutation, then compensation in the
VRAM MQD write.
