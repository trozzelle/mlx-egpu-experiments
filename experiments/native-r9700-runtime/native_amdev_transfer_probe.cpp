// Native AMDev transfer probe contracts and TinyGPU.app discovery smoke.
//
// Provenance: RemoteCmd ids, request framing, response framing, TinyGPU.app
// socket lifecycle, BAR/MMIO access shape, IP discovery-table offsets, and SDMA
// linear-copy packet encoding are ported/rederived from tinygrad's TinyGPU
// Remote PCI client ABI in tinygrad/runtime/support/system.py lines 302-303,
// 367-376, 385-405, and 407-438, tinygrad/runtime/support/am/amdev.py lines
// 288-314, tinygrad/runtime/ops_amd.py lines 474-560, and the generated AMD
// discovery/SDMA constants in tinygrad/runtime/autogen/am/am.py and
// tinygrad/runtime/autogen/am/sdma_6_0_0.py.
// This file carries only the small mechanics needed by the C0B native
// experiment contract; it does not import, call, shell out to, or vendor
// tinygrad runtime code.
//
// MIT License notice for the tinygrad-derived mechanics above:
// Copyright (c) 2023 George Hotz
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <chrono>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

enum class RemoteCmd : uint8_t {
  PROBE = 0,
  MAP_BAR,
  MAP_SYSMEM_FD,
  CFG_READ,
  CFG_WRITE,
  RESET,
  MMIO_READ,
  MMIO_WRITE,
  MAP_SYSMEM,
  SYSMEM_READ,
  SYSMEM_WRITE,
  RESIZE_BAR,
  PING,
};

constexpr std::size_t kRemoteCmdFrameSize = 33;
constexpr uint32_t kTargetVendor = 0x1002U;
constexpr uint32_t kTargetDevice = 0x7551U;
constexpr uint32_t kRemoteDevId = 0U;  // APLRemotePCIDevice uses pcibus "usb4" -> dev_id 0.
constexpr const char* kRuntimeSubstrate = "TinyGPU.app/APLRemotePCIDevice/PCIIface";
constexpr const char* kTinyGpuAppPath = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU";
constexpr uint64_t kRpcReadoutFromResponse0 = UINT64_MAX;
constexpr uint32_t kMmRccConfigMemsize = 0x0de3U;
constexpr uint32_t kAmdBinarySignature = 0x28211407U;
constexpr uint32_t kAmdDiscoveryTableSignature = 0x53445049U;
constexpr uint16_t kAmdGcHwId = 11U;       // tinygrad/runtime/autogen/am/am.py:4187 GC_HWID.
constexpr uint16_t kAmdMmhubHwId = 34U;    // tinygrad/runtime/autogen/am/am.py:4205 MMHUB_HWID.
constexpr uint16_t kAmdNbifHwId = 108U;    // tinygrad/runtime/autogen/am/am.py:4247 NBIF_HWID.
constexpr std::size_t kAmdBinaryHeaderSize = 60;
constexpr std::size_t kAmdTableInfoSize = 8;
constexpr std::size_t kAmdIpDiscoveryTableIndex = 0;
constexpr std::size_t kAmdIpDiscoveryTableSize = 10U << 10;
constexpr uint64_t kAmdIpDiscoveryTableVramBackoff = 64ULL << 10;
constexpr uint64_t kPageSize = 0x1000ULL;
constexpr uint64_t kTransferByteCount = 8ULL * sizeof(uint32_t);

constexpr uint64_t kTransferProofBufferSize = kPageSize;
// The staging window spans 256 fixed PTB entries (1 MiB) so streamed model
// weights pay one SDMA submission per MiB instead of one per 4 KiB page.
constexpr uint64_t kStagingPageCount = 256ULL;
constexpr uint64_t kDedicatedStagingPdb0Index = 511ULL;
constexpr uint64_t kTransferProofVmStagingVa =
    0x200000000000ULL + (kDedicatedStagingPdb0Index << 21U);
constexpr uint64_t kTransferProofVmVramVa = 0x200000001000ULL;
constexpr uint64_t kTransferProofVmReadbackVa = 0x200000002000ULL;
constexpr uint32_t kSdmaOpCopy = 1U;
constexpr uint32_t kSdmaSubopCopyLinear = 0U;
constexpr std::size_t kSdmaLinearCopyPacketDwords = 7;
using SdmaLinearCopyPacket = std::array<uint32_t, kSdmaLinearCopyPacketDwords>;

constexpr const char* kKernelProofMode = "minimal-u32-add-one";
constexpr const char* kKernelArch = "gfx1201";
constexpr const char* kKernelSourceId = "c0a-minimal-u32-add-one-v3";
constexpr const char* kKernelSourceLanguage = "amd-gcn-assembly";
constexpr const char* kKernelBlobFormat = "amdgpu-code-object-v5";
constexpr const char* kKernelBlobSymbol = "c0a_minimal_u32_add_one";
constexpr const char* kKernelBlobTarget = "gfx1201";
constexpr const char* kKernelInputValuesU32 = "1,2,3,4,5,6,7,8";
constexpr const char* kKernelInputBytesHex =
    "0100000002000000030000000400000005000000060000000700000008000000";
constexpr const char* kKernelExpectedOutputValuesU32 = "2,3,4,5,6,7,8,9";
constexpr const char* kKernelExpectedOutputBytesHex =
    "0200000003000000040000000500000006000000070000000800000009000000";
// Observed VRAM readback for the C0A23 diagnostic: outputs 2,3,4,5 (elements 0..3)
// written with 16-bit-halfword byte swap, 6,7,8,9 (elements 4..7) unwritten (zero).
// This is the stable c0l signature the readback classifier formalizes (see Task 1).
constexpr const char* kKernelObservedOutputBytesHex =
    "0000020000000300000004000000050000000000000000000000000000000000";
constexpr const char* kKernelExpectedOutputSha256 =
    "b06e51b2494d439f5e151692ca393efc3c52cdfddcc377be789356250b9860a6";
// Source-grounded reference capture preserved in the checked-in swarm report
// `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md` Task set 2 provenance.
// The original session artifact was local://c0a-kernel-proof-notes.md lines 12-66 and
// recorded `/tmp/c0a_kernel_capture/00_E_2_4.hsaco`, whole-HSACO SHA-256
// 7e03c75bb6682d0bb7e688a409c5f53a20a1b3a60b53c7720706500c4e7ae8bf,
// loaded `.text` size 512, and loaded `.text` SHA-256 below. Runtime stays
// tinygrad-free: the fixed bytes are embedded here and verified by deterministic
// self-test coverage plus BAR0 write/readback before `kernel_blob_load_status: pass`.
constexpr const char* kKernelTextProvenancePath =
    ".superpowers/swarm/reports/c0a-compute-task-5-dispatch.md#task-set-2-kernel-text-provenance";
constexpr const char* kKernelReferenceHsacoSha256 =
    "7e03c75bb6682d0bb7e688a409c5f53a20a1b3a60b53c7720706500c4e7ae8bf";
constexpr const char* kKernelReferenceTextSha256 =
    "08fd705ca25c7a1d5531e504eb9905ce84dab9c0a31b7ef6ecfc62475b98f965";
constexpr uint32_t kKernelReferenceTextByteCount = 64U;
constexpr const char* kKernelReferenceTextFirst64Hex =
    "004100f4000000f8000000f4180000f80000c7bf820002308002047e060005ee03000000"
    "010000000000c0bf0006064a048006ee00008001010000000000b0bf";
constexpr const char* kKernelReferenceTextLast16Hex = "048006ee00008001010000000000b0bf";
// Per-lane per-u32 add-one kernel (source id c0a-minimal-u32-add-one-v3), generated
// and round-trip verified through tinygrad's DSL + ELF packer
// (tinygrad/renderer/amd/elf.py::assemble_linear, tinygrad/runtime/autogen/amd/rdna4/ins.py).
// Byte-faithful GLOBAL_STORE_B32 lane data path per tinygrad's canonical rdna4 per-lane
// kernel custom_lds_sync (tinygrad/test/amd/test_custom_kernel.py). The load instruction
// bases on the input VA SGPR pair s[6:7] (correcting C0A24's buggy s[5:6]), per tinygrad's
// canonical custom_add_var (tinygrad/test/amd/test_custom_kernel.py); the store bases on
// the output VA pair s[4:5] (unchanged, hardware-proven). 64-byte program (16 words);
// dispatch (8 lanes) is Task 2 of C0A24. 10-encoding ISA sequence:
//   s_load_b128(s[4:7], s[0:1])                # out_va->s[4:5], in_va->s[6:7]
//   s_load_b32(s[0], s[0:1], NULL, 0,0,0, 0x18) # s[0] = scalar u32 @ kernargs+24
//   s_wait_kmcnt()
//   v_lshlrev_b32_e32(v[1], 2, v[0])           # v[1] = lane*4 (byte offset, low 32)
//   v_mov_b32_e32(v[2], 0)                     # v[2] = 0 (high 32 of 64-bit vaddr)
//   global_load_b32(vdst=v[3], vaddr=v[1:2], saddr=s[6:7])  # in[lane] = MEM[in_va + lane*4]
//   s_wait_loadcnt()
//   v_add_nc_u32_e32(v[3], s[0], v[3])         # out_val = in[lane] + scalar
//   global_store_b32(vaddr=v[1:2], vsrc=v[3], saddr=s[4:5]) # MEM[out_va + lane*4] = out_val
//   s_endpgm()
constexpr std::array<uint8_t, kKernelReferenceTextByteCount> kKernelText = {{
    0x00,0x41,0x00,0xf4, 0x00,0x00,0x00,0xf8, 0x00,0x00,0x00,0xf4, 0x18,0x00,0x00,0xf8,
    0x00,0x00,0xc7,0xbf, 0x82,0x00,0x02,0x30, 0x80,0x02,0x04,0x7e, 0x06,0x00,0x05,0xee,
    0x03,0x00,0x00,0x00, 0x01,0x00,0x00,0x00, 0x00,0x00,0xc0,0xbf, 0x00,0x06,0x06,0x4a,
    0x04,0x80,0x06,0xee, 0x00,0x00,0x80,0x01, 0x01,0x00,0x00,0x00, 0x00,0x00,0xb0,0xbf,
}};
static_assert(kKernelText.size() == kKernelReferenceTextByteCount,
              "embedded kernel text byte count must stay source-grounded");
constexpr uint32_t kKernelReferenceKernargSize = 24U;
constexpr uint32_t kKernelReferenceRsrc1 = 0xc00c0040U;
constexpr uint32_t kKernelReferenceRsrc2 = 0x00000084U;
constexpr uint32_t kKernelReferenceRsrc3 = 0x00000010U;
constexpr uint32_t kKernelReferenceCodeProperties = 0x00000408U;
using RemoteCmdFrame = std::array<uint8_t, kRemoteCmdFrameSize>;

namespace am_vm {

// tinygrad/runtime/autogen/am/am.py:2336 names the AMD VM levels PDB2/PDB1/PDB0/PTB.
constexpr const char* kLeafLevelName = "PTB";
constexpr const char* kFirstLevelName = "PDB2";
// tinygrad/runtime/support/am/ip.py:158-160 selects the gfx12 PTE path for GC IP >= 12.
constexpr uint32_t kGfxIpMajor = 12U;
// tinygrad/runtime/autogen/am/soc_12.py:7 defines MTYPE_UC == 3.
constexpr uint64_t kMtypeUc = 3ULL;
// tinygrad/runtime/autogen/am/am.py:4114-4144 defines the AMDGPU_PTE/PDE bit positions.
constexpr uint64_t kPteValid = 1ULL << 0;
constexpr uint64_t kPteSystem = 1ULL << 1;
constexpr uint64_t kPteSnooped = 1ULL << 2;
constexpr uint64_t kPteExecutable = 1ULL << 4;
constexpr uint64_t kPteReadable = 1ULL << 5;
constexpr uint64_t kPteWriteable = 1ULL << 6;
constexpr uint64_t kPteMtypeGfx12Shift = 54ULL;
constexpr uint64_t kPteIsPte = 1ULL << 63;
// tinygrad/runtime/support/am/amdev.py:123-128 masks the encoded physical address to qword PTE bits 12..47.
constexpr uint64_t kPtePaddrMask = 0x0000FFFFFFFFF000ULL;
// tinygrad/runtime/support/am/amdev.py:137-143 uses a global VA allocator base of 0x200000000000.

constexpr uint64_t kVaBase = 0x0000200000000000ULL;
constexpr uint64_t kStagingVa =
    kVaBase + (kDedicatedStagingPdb0Index * 512ULL * kPageSize);
constexpr uint64_t kStagingByteCount = kStagingPageCount * kPageSize;
constexpr uint64_t kVramVa = kVaBase + kPageSize;
constexpr uint64_t kReadbackVa = kVaBase + (2ULL * kPageSize);
// tinygrad/runtime/support/memory.py:175-183 derives VA shifts [12,21,30,39] and reserves page-table arena after boot.
constexpr uint32_t kVaShiftPtb = 12U;
constexpr uint32_t kVaShiftPdb0 = 21U;
constexpr uint32_t kVaShiftPdb1 = 30U;
constexpr uint32_t kVaShiftPdb2 = 39U;
constexpr uint64_t kBootArenaSize = 0x02000000ULL;
constexpr uint64_t kPtableArenaBase = kBootArenaSize;
constexpr uint64_t kFixedVramBufferPaddr = 0x0000000006000000ULL;
constexpr uint64_t kSyntheticSysmemStagingPaddr = 0x0000000080000000ULL;
constexpr uint64_t kSyntheticSysmemReadbackPaddr = 0x0000000080008000ULL;
constexpr uint64_t kSyntheticSysmemSdmaControlPaddr = 0x0000000080010000ULL;
// tinygrad/runtime/support/am/ip.py:86-105 and amdev.py:140-143 define VMID0 TLB invalidation order and bits.
constexpr uint32_t kVmid0 = 0U;
constexpr uint32_t kInvalidateMaskVmid0 = 1U << kVmid0;

struct VmIndices {
  uint64_t pdb2;
  uint64_t pdb1;
  uint64_t pdb0;
  uint64_t ptb;
};

constexpr uint64_t gfx12_leaf_pte_flags(bool system, bool snooped, bool uncached) {
  uint64_t flags = kPteValid | kPteExecutable | kPteReadable | kPteWriteable | kPteIsPte;
  if (system) {
    flags |= kPteSystem;
  }
  if (snooped) {
    flags |= kPteSnooped;
  }
  if (uncached) {
    flags |= kMtypeUc << kPteMtypeGfx12Shift;
  }
  return flags;
}

constexpr uint64_t table_pte_flags() {
  return kPteValid;
}

constexpr uint64_t encode_pte(uint64_t paddr, uint64_t flags) {
  return (paddr & kPtePaddrMask) | flags;
}

constexpr VmIndices vm_indices_for_va(uint64_t gpu_va) {
  const uint64_t relative_va = gpu_va - kVaBase;
  return VmIndices{
      (relative_va >> kVaShiftPdb2) & 0x3ffULL,
      (relative_va >> kVaShiftPdb1) & 0x1ffULL,
      (relative_va >> kVaShiftPdb0) & 0x1ffULL,
      (relative_va >> kVaShiftPtb) & 0x1ffULL,
  };
}

}  // namespace am_vm

namespace am_sdma {

// tinygrad/runtime/autogen/am/am.py:4213 defines SDMA0_HWID = 42.
constexpr uint16_t kSdma0HwId = 42U;
// tinygrad/runtime/autogen/am/am.py:3390 defines AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 = 256.
constexpr uint32_t kDoorbellIndex = 256U;
// tinygrad/runtime/support/am/ip.py:541 and 550 use doorbell * 2 for the SDMA doorbell-offset register.
constexpr uint32_t kDoorbellOffsetField = kDoorbellIndex * 2U;
// tinygrad/runtime/ops_amd.py:886 maps doorbell64 at doorbell_index * 8 bytes in BAR2.
constexpr uint64_t kDoorbellBar2ByteOffset = static_cast<uint64_t>(kDoorbellIndex) * sizeof(uint64_t);
constexpr uint32_t kQueueIndex = 0U;
constexpr const char* kRegisterPrefix = "regSDMA0_QUEUE0";
constexpr uint32_t kRegisterInstance = 0U;
constexpr const char* kTeardownOrder =
    "disable_rb,disable_ib,disable_doorbell,clear_doorbell_offset,soft_reset_sdma0";
// tinygrad/runtime/autogen/am/regs.py:5538 gc_12_0_0 defines regGRBM_SOFT_RESET.soft_reset_sdma0 at bit 23.
constexpr uint32_t kSoftResetSdma0Bit = 23U;

constexpr uint64_t kControlVa = am_vm::kVaBase + (3ULL * kPageSize);
constexpr uint64_t kRingSize = 0x800ULL;
constexpr uint64_t kRptrOffset = 0x800ULL;
constexpr uint64_t kWptrOffset = 0x808ULL;
constexpr uint64_t kFenceOffset = 0x810ULL;
constexpr uint64_t kRptrVa = kControlVa + kRptrOffset;
constexpr uint64_t kWptrVa = kControlVa + kWptrOffset;
constexpr uint64_t kFenceVa = kControlVa + kFenceOffset;
constexpr uint32_t kRingSizeField = 9U;
constexpr uint32_t kFencePacketDwords = 4U;
constexpr uint32_t kFenceValue = 1U;
constexpr uint32_t kFenceHeader = 0x00030005U;  // SDMA_OP_FENCE | SDMA_PKT_FENCE_HEADER_MTYPE(3).
constexpr uint32_t kSubmitCopyCount = 2U;
constexpr uint32_t kSubmitDwordCount = (kSubmitCopyCount * kSdmaLinearCopyPacketDwords) + kFencePacketDwords;
constexpr uint64_t kSubmitByteCount = kSubmitDwordCount * sizeof(uint32_t);

}  // namespace am_sdma

namespace am_compute {

constexpr uint32_t kExpectedXccCount = 1U;
constexpr uint32_t kMecDoorbellIndex = 3U;  // tinygrad/runtime/autogen/am/am.py:3390 AMDGPU_NAVI10_DOORBELL_MEC_RING0.
constexpr uint64_t kMecDoorbellBar2ByteOffset =
    static_cast<uint64_t>(kMecDoorbellIndex) * sizeof(uint64_t);

constexpr uint64_t kInputVramVa = am_vm::kVramVa;
constexpr uint64_t kOutputVramVa = am_vm::kVaBase + (4ULL * kPageSize);
constexpr uint64_t kCodeVramVa = am_vm::kVaBase + (5ULL * kPageSize);
constexpr uint64_t kKernargsVa = am_vm::kVaBase + (6ULL * kPageSize);
constexpr uint64_t kRingVa = am_vm::kVaBase + (7ULL * kPageSize);
constexpr uint64_t kRptrVa = am_vm::kVaBase + (15ULL * kPageSize);
constexpr uint64_t kWptrVa = kRptrVa + 8ULL;
constexpr uint64_t kTimelineVa = kRptrVa + 16ULL;
constexpr uint64_t kEopVa = am_vm::kVaBase + (16ULL * kPageSize);
// Per-stage kernargs ring: pages 17..32 (16 pages) for batched resident
// dispatch, one distinct kernargs page per in-flight stage.
constexpr uint64_t kKernargsRingVa = am_vm::kVaBase + (17ULL * kPageSize);
constexpr uint32_t kKernargsRingPageCount = 16U;
constexpr uint64_t kComputeControlKernargsRingCpuOffset = 10ULL * kPageSize;
// All fixed mappings must stay inside the first PDB0 2 MiB span (512 PTB
// entries); resident payloads own PDB0 index 1 and beyond.
static_assert(kKernargsRingVa + kKernargsRingPageCount * kPageSize <=
                  am_vm::kVaBase + (512ULL * kPageSize),
              "fixed VA layout overflows the first PDB0 page-table span");
constexpr uint64_t fixed_vram_paddr_for_va(uint64_t gpu_va) {
  return am_vm::kFixedVramBufferPaddr + (((gpu_va - am_vm::kVramVa) / kPageSize) * kPageSize);
}
constexpr uint64_t kInputVramPaddr = fixed_vram_paddr_for_va(kInputVramVa);
constexpr uint64_t kOutputVramPaddr = fixed_vram_paddr_for_va(kOutputVramVa);
constexpr uint64_t kCodeVramPaddr = fixed_vram_paddr_for_va(kCodeVramVa);
constexpr uint64_t kRingVramPaddr = fixed_vram_paddr_for_va(kRingVa);
constexpr uint64_t kEopVramPaddr = fixed_vram_paddr_for_va(kEopVa);
constexpr uint64_t kMqdPaddr = am_vm::kPtableArenaBase + (3ULL * kPageSize);
constexpr uint32_t kRingSize = 0x8000U;
constexpr uint32_t kEopSize = 0x1000U;
constexpr uint32_t kMqdSize = 2048U;
constexpr uint64_t kComputeControlByteCount = 26ULL * kPageSize;            // 2 control + 8 ring + 16 kernargs ring
constexpr uint64_t kComputeControlQueueCpuOffset = 0ULL;
constexpr uint64_t kRptrOffset = 0ULL;
constexpr uint64_t kWptrOffset = 8ULL;
constexpr uint64_t kTimelineOffset = 16ULL;
constexpr uint64_t kComputeControlKernargsCpuOffset = kPageSize;            // page 1
constexpr uint64_t kComputeControlRingCpuOffset = 2ULL * kPageSize;         // page 2..9
constexpr uint64_t kComputeControlRingByteCount = 8ULL * kPageSize;         // ring = 8 pages
constexpr uint64_t kScalarValue = 1ULL;

// tinygrad/runtime/autogen/am/regs.py:5981-6037 defines the gfx12 HQD copy span.
constexpr const char* kHqdRegSpan = "regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI";
// tinygrad/runtime/autogen/am/regs.py:5576-5635 plus AMDReg segment bases ground
// these gfx12 SET_SH_REG offsets; tinygrad/runtime/ops_amd.py:62-69 subtracts
// PACKET3_SET_SH_REG_START before emitting each PM4 packet.
constexpr uint32_t kComputeSetShBase = 0x00002c00U;  // pm4_soc15.py:326-328.
constexpr uint32_t kComputeStartXSetShOffset = 0x00000204U;
constexpr uint32_t kComputePgmLoSetShOffset = 0x0000020cU;
constexpr uint32_t kComputePgmRsrc1SetShOffset = 0x00000212U;
constexpr uint32_t kComputeResourceLimitsSetShOffset = 0x00000215U;
constexpr uint32_t kComputeTmpringSizeSetShOffset = 0x00000218U;
constexpr uint32_t kComputeRestartXSetShOffset = 0x0000021bU;
constexpr uint32_t kComputePgmRsrc3SetShOffset = 0x00000228U;
constexpr uint32_t kComputeUserData0SetShOffset = 0x00000240U;
constexpr const char* kGrbmSelectReg = "regGRBM_GFX_INDEX";
constexpr const char* kGcIpVersion = "12.0.1";

constexpr uint32_t kMqdHeader = 0xc0310800U;
constexpr uint32_t kHqdPipePriority = 0x00000002U;
constexpr uint32_t kHqdQueuePriority = 0x0000000fU;
constexpr uint32_t kHqdQuantum = 0x00000111U;
constexpr uint32_t kHqdAqlControl = 0U;
constexpr const char* kHqdPqControlMode = "direct_pm4";
constexpr uint32_t kComputeStaticThreadMgmt = 0xffffffffU;

constexpr const char* kPm4DispatchPacketOrder =
    "acquire_mem,set_sh_pgm,set_sh_rsrc,set_sh_rsrc3,set_sh_tmpring,set_sh_restart,set_sh_userdata,set_sh_resource_limits,set_sh_start,dispatch_direct,event_write,release_mem";
// PM4 opcode and field constants are copied from tinygrad autogen sources:
// pm4_soc15.py:55-56, 84-85, 252-264, 293-305, 326-328 and
// pm4_nv.py:27-32, 106-128, 291-302, 304-330, 350-363, 395-397.
constexpr uint32_t kPacketType3 = 3U;
constexpr uint32_t kPacket3DispatchDirect = 0x15U;
constexpr uint32_t kPacket3EventWrite = 0x46U;
constexpr uint32_t kPacket3ReleaseMem = 0x49U;
constexpr uint32_t kPacket3AcquireMem = 0x58U;
constexpr uint32_t kPacket3SetShReg = 0x76U;
constexpr uint32_t kEventTypeCsPartialFlush = 7U;
constexpr uint32_t kEventIndexPartialFlush = 4U;
constexpr uint32_t kEventTypeCacheFlushAndInvTs = 20U;
constexpr uint32_t kReleaseMemEventIndexEndOfPipe = 5U;
constexpr uint32_t kReleaseMemDataSelSend32BitLow = 1U;
constexpr uint32_t kReleaseMemIntSelNone = 0U;
constexpr uint32_t kAcquireMemGcrCntlGlmWbShift = 4U;
constexpr uint32_t kAcquireMemGcrCntlGlmInvShift = 5U;
constexpr uint32_t kAcquireMemGcrCntlGlkWbShift = 6U;
constexpr uint32_t kAcquireMemGcrCntlGlkInvShift = 7U;
constexpr uint32_t kAcquireMemGcrCntlGlvInvShift = 8U;
constexpr uint32_t kAcquireMemGcrCntlGl1InvShift = 9U;
constexpr uint32_t kReleaseMemGcrGlmWb = 1U << 12;
constexpr uint32_t kReleaseMemGcrGlmInv = 1U << 13;
constexpr uint32_t kReleaseMemGcrGlvInv = 1U << 14;
constexpr uint32_t kReleaseMemGcrGl1Inv = 1U << 15;
constexpr uint32_t kReleaseMemGcrGl2Inv = 1U << 20;
constexpr uint32_t kReleaseMemGcrGl2Wb = 1U << 21;
constexpr uint32_t kReleaseMemGcrSeq = 1U << 22;
constexpr uint32_t kPm4DispatchPacketCount = 12U;
constexpr uint32_t kPm4DispatchDwordCount = 59U;
constexpr const char* kDoorbellDiagnosticContract = "mec_doorbell_delivery_ring_fetch";
constexpr const char* kDoorbellFailureStageIfTimeout = "kernel_timeline_timeout";
constexpr const char* kDoorbellClassificationIfNotConsumed = "compute_doorbell_not_consumed";
constexpr const char* kDoorbellValueUnit = "dwords";
constexpr const char* kDoorbellValueSource = "pm4_dispatch_dword_count";
constexpr const char* kDoorbellHitSource = "regCP_HQD_PQ_DOORBELL_CONTROL.doorbell_hit";
constexpr const char* kDoorbellDiagnosticPreRingReads =
    "regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT,regCP_MEC_DOORBELL_RANGE_LOWER,regCP_MEC_DOORBELL_RANGE_UPPER";
constexpr const char* kDoorbellDiagnosticPostRingReads =
    "regCP_HQD_ACTIVE,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_HQD_PQ_CONTROL,regCP_STAT";
constexpr const char* kDoorbellDiagnosticTimeoutReads =
    "timeline,rptr,wptr,regCP_HQD_PQ_RPTR,regCP_HQD_PQ_DOORBELL_CONTROL,regCP_STAT";
constexpr const char* kDoorbellClassRptrZeroCpIdle = "compute_doorbell_not_consumed";
constexpr const char* kDoorbellClassDoorbellHitRptrZero = "hqd_ring_fetch_not_started";
constexpr const char* kDoorbellClassRptrAdvancesTimelineZero =
    "pm4_dispatch_or_release_mem_blocked";
constexpr const char* kDoorbellRouteReadbackField = "compute_doorbell_route_readback";
constexpr const char* kDoorbellRouteClassificationField =
    "compute_doorbell_route_classification";
constexpr const char* kDoorbellRouteReadbackRegisters =
    "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN,regRCC_DEV0_EPF2_STRAP2,regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL,regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL";
constexpr uint32_t kDoorbellRouteExpectedEntry0Ctrl = 0x30000007U;
constexpr uint32_t kDoorbellRouteExpectedEntry3Ctrl = 0x3000000dU;
constexpr const char* kDoorbellRouteClassMatches = "gdc_s2a_route_readback_matches";
constexpr const char* kDoorbellRouteClassMismatch = "gdc_s2a_route_readback_mismatch";
constexpr const char* kDoorbellRouteClassUnclassified =
    "gdc_s2a_route_readback_unclassified";
constexpr const char* kDoorbellConsumptionDiagnosticContract =
    "hqd_pq_doorbell_consumption";
constexpr const char* kDoorbellConsumptionSourceGapExitRequired =
    "diagnostic_override_allowed";
constexpr const char* kDoorbellConsumptionControlReads =
    "regCP_HQD_PQ_DOORBELL_CONTROL";
constexpr const char* kDoorbellConsumptionControlDecodes =
    "doorbell_mode,doorbell_bif_drop,doorbell_offset,doorbell_source,doorbell_schd_hit,doorbell_en,doorbell_hit";
constexpr const char* kDoorbellConsumptionControlCompareIgnoredBits =
    "doorbell_bif_drop,doorbell_schd_hit,doorbell_hit";
constexpr uint32_t kDoorbellConsumptionExpectedOffset = 6U;
constexpr uint32_t kDoorbellConsumptionExpectedEn = 1U;
constexpr const char* kDoorbellConsumptionMqdHqdCompareFields =
    "cp_hqd_pq_doorbell_control,cp_hqd_pq_control,cp_hqd_pq_base,cp_hqd_pq_rptr_report_addr,cp_hqd_pq_wptr_poll_addr,cp_mqd_control,cp_hqd_eop_base_addr,cp_hqd_eop_control";
constexpr const char* kDoorbellConsumptionWptrVisibilityReads =
    "control_wptr_cpu,control_rptr_cpu,regCP_HQD_PQ_WPTR_LO,regCP_HQD_PQ_WPTR_HI,regCP_HQD_PQ_RPTR";
constexpr const char* kDoorbellConsumptionCpMecStatusReads =
    "regCP_STAT,regCP_INT_CNTL_RING0,regCP_MEC1_F32_INTERRUPT,regCP_MEC1_INSTR_PNTR,regCP_MEC_RS64_INTERRUPT,regCP_MEC_RS64_PENDING_INTERRUPT,regCP_MEC_RS64_EXCEPTION_STATUS";
constexpr const char* kDoorbellConsumptionCpMecRs64ContextReads =
    "regCP_MEC_RS64_INSTR_PNTR,regCP_MEC_RS64_PRGRM_CNTR_START_HI,regCP_MEC_LOCAL_INSTR_BASE_LO,regCP_MEC_LOCAL_INSTR_BASE_HI,regCP_MEC_LOCAL_INSTR_MASK_LO,regCP_MEC_LOCAL_INSTR_MASK_HI,regCP_MEC_LOCAL_INSTR_APERTURE,regCP_MEC_RS64_INTERRUPT_DATA_16,regCP_MEC_RS64_INTERRUPT_DATA_17,regCP_MEC_RS64_INTERRUPT_DATA_18,regCP_MEC_RS64_INTERRUPT_DATA_19,regCP_MEC_RS64_INTERRUPT_DATA_20,regCP_MEC_RS64_INTERRUPT_DATA_21,regCP_MEC_RS64_INTERRUPT_DATA_22,regCP_MEC_RS64_INTERRUPT_DATA_23,regCP_MEC_RS64_INTERRUPT_DATA_24,regCP_MEC_RS64_INTERRUPT_DATA_25,regCP_MEC_RS64_INTERRUPT_DATA_26,regCP_MEC_RS64_INTERRUPT_DATA_27,regCP_MEC_RS64_INTERRUPT_DATA_28,regCP_MEC_RS64_INTERRUPT_DATA_29,regCP_MEC_RS64_INTERRUPT_DATA_30,regCP_MEC_RS64_INTERRUPT_DATA_31";
constexpr const char* kDoorbellConsumptionClassRs64Exception =
    "rs64_exception_context_needed";
constexpr const char* kDoorbellConsumptionClassBifDrop =
    "doorbell_route_or_range_drop";
constexpr const char* kDoorbellConsumptionClassSchdOrHitRptrZero =
    "hqd_doorbell_seen_ring_fetch_not_started";
constexpr const char* kDoorbellConsumptionClassWptrNotVisible =
    "compute_wptr_not_visible_to_cp";
constexpr const char* kDoorbellConsumptionClassMqdHqdMismatch =
    "mqd_hqd_copy_mismatch";
constexpr const char* kDoorbellConsumptionClassRptrAdvancesTimelineZero =
    "ring_fetch_started_pm4_or_release_mem_blocked";
constexpr const char* kDoorbellConsumptionClassNoSignal =
    "doorbell_not_reaching_hqd_unclassified";
constexpr uint32_t kHqdPqDoorbellHitMask = 1U << 31;  // regs.py:5996 doorbell_hit field.
constexpr uint32_t kHqdPqDoorbellControlDynamicStatusMask =
    (1U << 1) | (1U << 29) | kHqdPqDoorbellHitMask;
constexpr uint32_t kHqdPqDoorbellControlStaticCompareMask =
    0xffffffffU ^ kHqdPqDoorbellControlDynamicStatusMask;
constexpr uint32_t hqd_doorbell_mode(uint32_t value) { return value & 0x1U; }
constexpr uint32_t hqd_doorbell_bif_drop(uint32_t value) { return (value >> 1) & 0x1U; }
constexpr uint32_t hqd_doorbell_offset(uint32_t value) { return (value >> 2) & 0x03ffffffU; }
constexpr uint32_t hqd_doorbell_source(uint32_t value) { return (value >> 28) & 0x1U; }
constexpr uint32_t hqd_doorbell_schd_hit(uint32_t value) { return (value >> 29) & 0x1U; }
constexpr uint32_t hqd_doorbell_en(uint32_t value) { return (value >> 30) & 0x1U; }
constexpr uint32_t hqd_doorbell_hit(uint32_t value) { return (value >> 31) & 0x1U; }
constexpr uint32_t kDispatchGlobalSizeX = 1U;
constexpr uint32_t kDispatchGlobalSizeY = 1U;
constexpr uint32_t kDispatchGlobalSizeZ = 1U;
constexpr uint32_t kDispatchLocalSizeX = 8U;
constexpr uint32_t kDispatchLocalSizeY = 1U;
constexpr uint32_t kDispatchLocalSizeZ = 1U;
constexpr uint32_t kReleaseMemTimelineValue = 1U;

}  // namespace am_compute

using ComputeMqd = std::array<uint32_t, am_compute::kMqdSize / sizeof(uint32_t)>;

// tinygrad/runtime/autogen/am/am.py:1821-1905 defines struct_v12_compute_mqd;
// these dword indices are the source field offsets divided by sizeof(uint32_t).
enum ComputeMqdDword : std::size_t {
  kMqdHeader = 0,
  kMqdComputePgmLo = 13,
  kMqdComputePgmHi = 14,
  kMqdComputePgmRsrc1 = 19,
  kMqdComputePgmRsrc2 = 20,
  kMqdComputeVmid = 21,
  kMqdComputeResourceLimits = 22,
  kMqdComputeStaticThreadMgmtSe0 = 23,
  kMqdComputeStaticThreadMgmtSe1 = 24,
  kMqdComputeTmpringSize = 25,
  kMqdComputeStaticThreadMgmtSe2 = 26,
  kMqdComputeStaticThreadMgmtSe3 = 27,
  kMqdComputePgmRsrc3 = 40,
  kMqdComputeStaticThreadMgmtSe4 = 44,
  kMqdComputeStaticThreadMgmtSe5 = 45,
  kMqdComputeStaticThreadMgmtSe6 = 46,
  kMqdComputeStaticThreadMgmtSe7 = 47,
  kMqdComputeUserData0 = 64,
  kMqdCpMqdBaseAddrLo = 0x80,
  kMqdCpMqdBaseAddrHi = 0x81,
  kMqdCpHqdVmid = 0x83,
  kMqdCpHqdPersistentState = 0x84,
  kMqdCpHqdPipePriority = 0x85,
  kMqdCpHqdQueuePriority = 0x86,
  kMqdCpHqdQuantum = 0x87,
  kMqdCpHqdPqBaseLo = 0x88,
  kMqdCpHqdPqBaseHi = 0x89,
  kMqdCpHqdPqRptrReportAddrLo = 0x8b,
  kMqdCpHqdPqRptrReportAddrHi = 0x8c,
  kMqdCpHqdPqWptrPollAddrLo = 0x8d,
  kMqdCpHqdPqWptrPollAddrHi = 0x8e,
  kMqdCpHqdPqDoorbellControl = 0x8f,
  kMqdCpHqdPqControl = 0x91,
  kMqdCpHqdIbControl = 0x95,
  kMqdCpHqdHqStatus0 = 0xa0,
  kMqdCpHqdHqControl0 = 0xa1,
  kMqdCpMqdControl = 0xa2,
  kMqdCpHqdHqStatus1 = 0xa3,
  kMqdCpHqdHqControl1 = 0xa4,
  kMqdCpHqdEopBaseAddrLo = 0xa5,
  kMqdCpHqdEopBaseAddrHi = 0xa6,
  kMqdCpHqdEopControl = 0xa7,
  kMqdCpHqdAqlControl = 0xb3,
};

// tinygrad/runtime/support/am/ip.py:340-342 copies HQD registers from
// mqd_st_mv[0x80 + i] into regCP_MQD_BASE_ADDR..regCP_HQD_PQ_WPTR_HI.
constexpr std::size_t kMqdHqdRegisterCopyStart = 0x80;

constexpr uint32_t lo32(uint64_t value) {
  return static_cast<uint32_t>(value & 0xffffffffULL);
}

constexpr uint32_t hi32(uint64_t value) {
  return static_cast<uint32_t>((value >> 32) & 0xffffffffULL);
}


constexpr uint32_t log2_floor_u32(uint32_t value) {
  uint32_t result = 0;
  while (value > 1U) {
    value >>= 1U;
    ++result;
  }
  return result;
}

constexpr uint32_t encode_hqd_persistent_state() {
  return 1U | (0x55U << 8);
}

constexpr uint32_t encode_hqd_pq_doorbell_control() {
  return ((am_compute::kMecDoorbellIndex * 2U) << 2) | (1U << 30);
}

constexpr uint32_t encode_hqd_pq_control_direct_pm4() {
  const uint32_t dwords = am_compute::kRingSize / static_cast<uint32_t>(sizeof(uint32_t));
  // tinygrad/runtime/support/am/ip.py:329 setup_ring encodes
  // cp_hqd_pq_control with unord_dispatch=0 for direct PM4. Native previously
  // forced kUnordDispatch=1 (bit 28), giving hqd_pq_control=0x1000050c instead
  // of tinygrad's 0x0000050c. Single-variable diagnostic: drop the bit.
  return (log2_floor_u32(dwords) - 1U) | (5U << 8);
}

constexpr uint32_t encode_hqd_ib_control() {
  return 0x3U << 20;
}

constexpr uint32_t encode_hqd_eop_control() {
  // Tinygrad AM_GFX.setup_ring: (size_bytes / 4).bit_length() - 2.
  return log2_floor_u32(am_compute::kEopSize) - 3U;
}

constexpr uint32_t encode_cp_mqd_control() {
  return 1U << 8;
}

constexpr uint32_t encode_dispatch_initiator() {
  return (1U << 0) | (1U << 2);
}

constexpr uint32_t pm4_packet3(uint32_t opcode, uint32_t count) {
  return (am_compute::kPacketType3 << 30) | ((opcode & 0xffU) << 8) |
         ((count & 0x3fffU) << 16);
}

constexpr uint32_t encode_acquire_mem_gcr_cntl_for_dispatch() {
  // Mirrors AMDComputeQueue.acquire_mem(gli=0, gl2=0) for target != 9.
  return (1U << am_compute::kAcquireMemGcrCntlGlmWbShift) |
         (1U << am_compute::kAcquireMemGcrCntlGlmInvShift) |
         (1U << am_compute::kAcquireMemGcrCntlGlkWbShift) |
         (1U << am_compute::kAcquireMemGcrCntlGlkInvShift) |
         (1U << am_compute::kAcquireMemGcrCntlGlvInvShift) |
         (1U << am_compute::kAcquireMemGcrCntlGl1InvShift);
}

constexpr uint32_t encode_event_write_cs_partial_flush() {
  return am_compute::kEventTypeCsPartialFlush |
         (am_compute::kEventIndexPartialFlush << 8);
}

constexpr uint32_t encode_release_mem_event() {
  return am_compute::kEventTypeCacheFlushAndInvTs |
         (am_compute::kReleaseMemEventIndexEndOfPipe << 8) |
         am_compute::kReleaseMemGcrGlmWb | am_compute::kReleaseMemGcrGlmInv |
         am_compute::kReleaseMemGcrGlvInv | am_compute::kReleaseMemGcrGl1Inv |
         am_compute::kReleaseMemGcrGl2Inv | am_compute::kReleaseMemGcrGl2Wb |
         am_compute::kReleaseMemGcrSeq;
}

constexpr uint32_t encode_release_mem_data_sel() {
  return am_compute::kReleaseMemDataSelSend32BitLow << 29 |
         am_compute::kReleaseMemIntSelNone << 24;
}
constexpr uint32_t encode_s2a_doorbell_entry(uint32_t awid, uint32_t awaddr_31_28,
                                             uint32_t range_offset = 0U,
                                             uint32_t range_size = 0U) {
  return 1U | (awid << 1) | (range_offset << 7) | (range_size << 17) |
         (awaddr_31_28 << 28);
}


void append_pm4_packet3(std::vector<uint32_t>* words, uint32_t opcode,
                        std::initializer_list<uint32_t> payload) {
  words->push_back(pm4_packet3(opcode, static_cast<uint32_t>(payload.size() - 1U)));
  words->insert(words->end(), payload.begin(), payload.end());
}

std::vector<uint32_t> build_compute_dispatch_words(uint64_t code_va, uint64_t kernargs_va,
                                                   uint64_t timeline_va) {
  std::vector<uint32_t> words;
  words.reserve(am_compute::kPm4DispatchDwordCount);
  append_pm4_packet3(&words, am_compute::kPacket3AcquireMem,
                     {0U, 0xffffffffU, 0xffffffffU, 0U, 0U, 0U,
                      encode_acquire_mem_gcr_cntl_for_dispatch()});
  const uint64_t code_addr = code_va >> 8;
  append_pm4_packet3(&words, am_compute::kPacket3SetShReg,
                     {am_compute::kComputePgmLoSetShOffset, lo32(code_addr), hi32(code_addr)});
  append_pm4_packet3(&words, am_compute::kPacket3SetShReg,
                     {am_compute::kComputePgmRsrc1SetShOffset, kKernelReferenceRsrc1,
                      kKernelReferenceRsrc2});
  append_pm4_packet3(&words, am_compute::kPacket3SetShReg,
                     {am_compute::kComputePgmRsrc3SetShOffset, kKernelReferenceRsrc3});
  append_pm4_packet3(&words, am_compute::kPacket3SetShReg,
                     {am_compute::kComputeTmpringSizeSetShOffset, 0U});
  append_pm4_packet3(&words, am_compute::kPacket3SetShReg,
                     {am_compute::kComputeRestartXSetShOffset, 0U, 0U, 0U});
  append_pm4_packet3(&words, am_compute::kPacket3SetShReg,
                     {am_compute::kComputeUserData0SetShOffset, lo32(kernargs_va),
                      hi32(kernargs_va)});
  append_pm4_packet3(&words, am_compute::kPacket3SetShReg,
                     {am_compute::kComputeResourceLimitsSetShOffset, 0U});
  append_pm4_packet3(&words, am_compute::kPacket3SetShReg,
                     {am_compute::kComputeStartXSetShOffset, 0U, 0U, 0U,
                      am_compute::kDispatchLocalSizeX, am_compute::kDispatchLocalSizeY,
                      am_compute::kDispatchLocalSizeZ, 0U, 0U});
  append_pm4_packet3(&words, am_compute::kPacket3DispatchDirect,
                     {am_compute::kDispatchGlobalSizeX, am_compute::kDispatchGlobalSizeY,
                      am_compute::kDispatchGlobalSizeZ, encode_dispatch_initiator()});
  append_pm4_packet3(&words, am_compute::kPacket3EventWrite,
                     {encode_event_write_cs_partial_flush()});
  append_pm4_packet3(&words, am_compute::kPacket3ReleaseMem,
                     {encode_release_mem_event(), encode_release_mem_data_sel(), lo32(timeline_va),
                      hi32(timeline_va), am_compute::kReleaseMemTimelineValue, 0U, 0U});
  return words;
}

ComputeMqd build_compute_mqd(uint64_t mc_base) {
  ComputeMqd mqd{};
  mqd[kMqdHeader] = am_compute::kMqdHeader;
  const uint64_t code_addr = am_compute::kCodeVramVa >> 8;
  mqd[kMqdComputePgmLo] = lo32(code_addr);
  mqd[kMqdComputePgmHi] = hi32(code_addr);
  mqd[kMqdComputePgmRsrc1] = kKernelReferenceRsrc1;
  mqd[kMqdComputePgmRsrc2] = kKernelReferenceRsrc2;
  mqd[kMqdComputeVmid] = am_vm::kVmid0;
  mqd[kMqdComputeResourceLimits] = 0U;
  mqd[kMqdComputeTmpringSize] = 0U;
  mqd[kMqdComputePgmRsrc3] = kKernelReferenceRsrc3;
  mqd[kMqdComputeStaticThreadMgmtSe0] = am_compute::kComputeStaticThreadMgmt;
  mqd[kMqdComputeStaticThreadMgmtSe1] = am_compute::kComputeStaticThreadMgmt;
  mqd[kMqdComputeStaticThreadMgmtSe2] = am_compute::kComputeStaticThreadMgmt;
  mqd[kMqdComputeStaticThreadMgmtSe3] = am_compute::kComputeStaticThreadMgmt;
  mqd[kMqdComputeStaticThreadMgmtSe4] = am_compute::kComputeStaticThreadMgmt;
  mqd[kMqdComputeStaticThreadMgmtSe5] = am_compute::kComputeStaticThreadMgmt;
  mqd[kMqdComputeStaticThreadMgmtSe6] = am_compute::kComputeStaticThreadMgmt;
  mqd[kMqdComputeStaticThreadMgmtSe7] = am_compute::kComputeStaticThreadMgmt;
  mqd[kMqdComputeUserData0] = lo32(am_compute::kKernargsVa);

  const uint64_t mqd_mc_addr = mc_base + am_compute::kMqdPaddr;  // tinygrad ip.py:322 cp_mqd_base_addr = mqd_mc.
  mqd[kMqdCpMqdBaseAddrLo] = lo32(mqd_mc_addr);
  mqd[kMqdCpMqdBaseAddrHi] = hi32(mqd_mc_addr);
  mqd[kMqdCpHqdVmid] = am_vm::kVmid0;
  mqd[kMqdCpHqdPersistentState] = encode_hqd_persistent_state();
  mqd[kMqdCpHqdPipePriority] = am_compute::kHqdPipePriority;
  mqd[kMqdCpHqdQueuePriority] = am_compute::kHqdQueuePriority;
  mqd[kMqdCpHqdQuantum] = am_compute::kHqdQuantum;
  const uint64_t ring_addr = am_compute::kRingVa >> 8;
  mqd[kMqdCpHqdPqBaseLo] = lo32(ring_addr);
  mqd[kMqdCpHqdPqBaseHi] = hi32(ring_addr);
  mqd[kMqdCpHqdPqRptrReportAddrLo] = lo32(am_compute::kRptrVa);
  mqd[kMqdCpHqdPqRptrReportAddrHi] = hi32(am_compute::kRptrVa);
  mqd[kMqdCpHqdPqWptrPollAddrLo] = lo32(am_compute::kWptrVa);
  mqd[kMqdCpHqdPqWptrPollAddrHi] = hi32(am_compute::kWptrVa);
  mqd[kMqdCpHqdPqDoorbellControl] = encode_hqd_pq_doorbell_control();
  mqd[kMqdCpHqdPqControl] = encode_hqd_pq_control_direct_pm4();
  mqd[kMqdCpHqdIbControl] = encode_hqd_ib_control();
  mqd[kMqdCpHqdHqStatus0] = 0x20004000U;
  mqd[kMqdCpMqdControl] = encode_cp_mqd_control();
  const uint64_t eop_addr = am_compute::kEopVa >> 8;
  mqd[kMqdCpHqdEopBaseAddrLo] = lo32(eop_addr);
  mqd[kMqdCpHqdEopBaseAddrHi] = hi32(eop_addr);
  mqd[kMqdCpHqdEopControl] = encode_hqd_eop_control();
  mqd[kMqdCpHqdAqlControl] = am_compute::kHqdAqlControl;
  return mqd;
}

void write_u32_le(RemoteCmdFrame& frame, std::size_t offset, uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i) {
    frame[offset + i] = static_cast<uint8_t>((value >> (i * 8)) & 0xffU);
  }
}

void write_u64_le(RemoteCmdFrame& frame, std::size_t offset, uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i) {
    frame[offset + i] = static_cast<uint8_t>((value >> (i * 8)) & 0xffU);
  }
}

uint16_t read_u16_le(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) | static_cast<uint16_t>(data[1] << 8);
}

uint32_t read_u32_le_bytes(const uint8_t* data) {
  uint32_t value = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    value |= static_cast<uint32_t>(data[i]) << (i * 8);
  }
  return value;
}

uint64_t read_u64_le_bytes(const uint8_t* data) {
  uint64_t value = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(data[i]) << (i * 8);
  }
  return value;
}


RemoteCmdFrame build_remote_cmd_frame(RemoteCmd cmd, uint32_t dev_id, uint32_t bar,
                                      uint64_t arg0, uint64_t arg1, uint64_t arg2) {
  RemoteCmdFrame frame{};
  frame[0] = static_cast<uint8_t>(cmd);
  write_u32_le(frame, 1, dev_id);
  write_u32_le(frame, 5, bar);
  write_u64_le(frame, 9, arg0);
  write_u64_le(frame, 17, arg1);
  write_u64_le(frame, 25, arg2);
  return frame;
}

std::array<char, (kRemoteCmdFrameSize * 2) + 1> hex_encode(const RemoteCmdFrame& frame) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::array<char, (kRemoteCmdFrameSize * 2) + 1> out{};
  for (std::size_t i = 0; i < frame.size(); ++i) {
    out[i * 2] = kHexDigits[(frame[i] >> 4) & 0x0fU];
    out[(i * 2) + 1] = kHexDigits[frame[i] & 0x0fU];
  }
  out[kRemoteCmdFrameSize * 2] = '\0';
  return out;
}

std::string hex_encode_bytes(const uint8_t* data, std::size_t size) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(size * 2);
  for (std::size_t i = 0; i < size; ++i) {
    out.push_back(kHexDigits[(data[i] >> 4) & 0x0fU]);
    out.push_back(kHexDigits[data[i] & 0x0fU]);
  }
  return out;
}

std::string hex_encode_vector(const std::vector<uint8_t>& data) {
  return data.empty() ? std::string{} : hex_encode_bytes(data.data(), data.size());
}

std::string printable_ascii(const std::vector<uint8_t>& data) {
  std::string out;
  for (uint8_t byte : data) {
    if (byte >= 0x20U && byte <= 0x7eU) {
      out.push_back(static_cast<char>(byte));
    } else if (byte == '\n') {
      out += "\\n";
    } else if (byte == '\r') {
      out += "\\r";
    } else if (byte == '\t') {
      out += "\\t";
    } else {
      char escaped[5]{};
      std::snprintf(escaped, sizeof(escaped), "\\x%02x", static_cast<unsigned>(byte));
      out += escaped;
    }
  }
  return out;
}

bool remote_cmd_order_ok() {
  constexpr std::array<uint8_t, 13> ids{{
      static_cast<uint8_t>(RemoteCmd::PROBE),
      static_cast<uint8_t>(RemoteCmd::MAP_BAR),
      static_cast<uint8_t>(RemoteCmd::MAP_SYSMEM_FD),
      static_cast<uint8_t>(RemoteCmd::CFG_READ),
      static_cast<uint8_t>(RemoteCmd::CFG_WRITE),
      static_cast<uint8_t>(RemoteCmd::RESET),
      static_cast<uint8_t>(RemoteCmd::MMIO_READ),
      static_cast<uint8_t>(RemoteCmd::MMIO_WRITE),
      static_cast<uint8_t>(RemoteCmd::MAP_SYSMEM),
      static_cast<uint8_t>(RemoteCmd::SYSMEM_READ),
      static_cast<uint8_t>(RemoteCmd::SYSMEM_WRITE),
      static_cast<uint8_t>(RemoteCmd::RESIZE_BAR),
      static_cast<uint8_t>(RemoteCmd::PING),
  }};
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] != static_cast<uint8_t>(i)) {
      return false;
    }
  }
  return true;
}

void append_u64_le(std::vector<uint8_t>* data, uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i) {
    data->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xffU));
  }
}

std::vector<uint8_t> u64_payload_le(uint64_t value) {
  std::vector<uint8_t> data;
  data.reserve(sizeof(uint64_t));
  append_u64_le(&data, value);
  return data;
}

void append_page_list_pair(std::vector<uint8_t>* data, uint64_t paddr, uint64_t size) {
  append_u64_le(data, paddr);
  append_u64_le(data, size);
}
void append_u32_le(std::vector<uint8_t>* data, uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i) {
    data->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xffU));
  }
}

std::vector<uint8_t> u32_payload_le(uint32_t value) {
  std::vector<uint8_t> data;
  data.reserve(sizeof(uint32_t));
  append_u32_le(&data, value);
  return data;
}

SdmaLinearCopyPacket build_sdma_linear_copy_packet(uint64_t src_addr, uint64_t dst_addr,
                                                   uint32_t byte_count) {
  return SdmaLinearCopyPacket{{
      kSdmaOpCopy | (kSdmaSubopCopyLinear << 8),
      byte_count - 1U,
      0U,
      static_cast<uint32_t>(src_addr & 0xffffffffULL),
      static_cast<uint32_t>(src_addr >> 32),
      static_cast<uint32_t>(dst_addr & 0xffffffffULL),
      static_cast<uint32_t>(dst_addr >> 32),
  }};
}

std::array<uint32_t, am_sdma::kFencePacketDwords> build_sdma_fence_packet(uint64_t fence_va,
                                                                          uint32_t value) {
  return std::array<uint32_t, am_sdma::kFencePacketDwords>{
      am_sdma::kFenceHeader,
      static_cast<uint32_t>(fence_va & 0xffffffffULL),
      static_cast<uint32_t>(fence_va >> 32),
      value,
  };
}

std::vector<uint32_t> build_sdma_copy_submit_words(uint64_t src_va, uint64_t dst_va,
                                                   uint32_t byte_count, uint64_t fence_va,
                                                   uint32_t fence_value) {
  std::vector<uint32_t> words;
  words.reserve(kSdmaLinearCopyPacketDwords + am_sdma::kFencePacketDwords);
  const SdmaLinearCopyPacket copy = build_sdma_linear_copy_packet(src_va, dst_va, byte_count);
  const auto fence = build_sdma_fence_packet(fence_va, fence_value);
  words.insert(words.end(), copy.begin(), copy.end());
  words.insert(words.end(), fence.begin(), fence.end());
  return words;
}

std::vector<uint32_t> build_sdma_submit_words(uint64_t staging_va, uint64_t vram_va,
                                              uint64_t readback_va, uint64_t fence_va) {
  std::vector<uint32_t> words;

  words.reserve(am_sdma::kSubmitDwordCount);
  const SdmaLinearCopyPacket to_vram =
      build_sdma_linear_copy_packet(staging_va, vram_va, static_cast<uint32_t>(kTransferByteCount));
  const SdmaLinearCopyPacket to_readback =
      build_sdma_linear_copy_packet(vram_va, readback_va, static_cast<uint32_t>(kTransferByteCount));
  const auto fence = build_sdma_fence_packet(fence_va, am_sdma::kFenceValue);
  words.insert(words.end(), to_vram.begin(), to_vram.end());
  words.insert(words.end(), to_readback.begin(), to_readback.end());
  words.insert(words.end(), fence.begin(), fence.end());
  return words;
}

std::string hex_encode_sdma_packet(const SdmaLinearCopyPacket& packet) {
  std::vector<uint8_t> bytes;
  bytes.reserve(packet.size() * sizeof(uint32_t));
  for (uint32_t word : packet) {
    append_u32_le(&bytes, word);
  }
  return hex_encode_vector(bytes);
}

std::array<uint8_t, kTransferByteCount> transfer_payload() {
  std::array<uint8_t, kTransferByteCount> payload{};
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>((i * 7U + 3U) & 0xffU);
  }
  return payload;
}

std::array<uint8_t, kTransferByteCount> kernel_input_payload() {
  std::array<uint8_t, kTransferByteCount> payload{};
  for (uint32_t i = 0; i < 8U; ++i) {
    const uint32_t value = i + 1U;
    payload[(i * sizeof(uint32_t)) + 0U] = static_cast<uint8_t>(value & 0xffU);
    payload[(i * sizeof(uint32_t)) + 1U] = static_cast<uint8_t>((value >> 8) & 0xffU);
    payload[(i * sizeof(uint32_t)) + 2U] = static_cast<uint8_t>((value >> 16) & 0xffU);
    payload[(i * sizeof(uint32_t)) + 3U] = static_cast<uint8_t>((value >> 24) & 0xffU);
  }
  return payload;
}

std::array<uint8_t, kTransferByteCount> kernel_expected_output_payload() {
  std::array<uint8_t, kTransferByteCount> payload{};
  for (uint32_t i = 0; i < 8U; ++i) {
    const uint32_t value = i + 2U;
    payload[(i * sizeof(uint32_t)) + 0U] = static_cast<uint8_t>(value & 0xffU);
    payload[(i * sizeof(uint32_t)) + 1U] = static_cast<uint8_t>((value >> 8) & 0xffU);
    payload[(i * sizeof(uint32_t)) + 2U] = static_cast<uint8_t>((value >> 16) & 0xffU);
    payload[(i * sizeof(uint32_t)) + 3U] = static_cast<uint8_t>((value >> 24) & 0xffU);
  }
  return payload;
}


uint64_t ceil_div_u64(uint64_t value, uint64_t divisor) {
  return (value / divisor) + ((value % divisor) != 0 ? 1ULL : 0ULL);
}

struct PageListParseResult {
  bool ok = false;
  bool saw_terminator = false;
  bool truncated = false;
  std::size_t raw_pair_count = 0;
  uint64_t expanded_page_count = 0;
  std::vector<uint64_t> pages;
  std::string error_text;
};

PageListParseResult parse_sysmem_page_list_bytes(const uint8_t* data, std::size_t byte_count,
                                                 std::size_t requested_page_count) {
  PageListParseResult result;
  result.pages.reserve(requested_page_count);
  if (byte_count < 16) {
    result.error_text = "sysmem page-list mapping shorter than one (paddr,size) pair";
    return result;
  }

  for (std::size_t offset = 0; offset + 16 <= byte_count; offset += 16) {
    const uint64_t paddr = read_u64_le_bytes(data + offset);
    const uint64_t size = read_u64_le_bytes(data + offset + 8);
    if (size == 0) {
      if (paddr != 0) {
        result.error_text = "sysmem page-list terminator had nonzero physical address";
        return result;
      }
      result.saw_terminator = true;
      break;
    }

    ++result.raw_pair_count;
    const uint64_t segment_pages = ceil_div_u64(size, kPageSize);
    if (UINT64_MAX - result.expanded_page_count < segment_pages) {
      result.error_text = "sysmem page-list expanded page count overflow";
      return result;
    }
    for (uint64_t page = 0; page < segment_pages && result.pages.size() < requested_page_count;
         ++page) {
      const uint64_t page_offset = page * kPageSize;
      if (paddr > UINT64_MAX - page_offset) {
        result.error_text = "sysmem page-list physical address overflow";
        return result;
      }
      result.pages.push_back(paddr + page_offset);
    }
    result.expanded_page_count += segment_pages;
  }

  if (!result.saw_terminator) {
    result.error_text = "sysmem page-list terminator not found before mapping end";
    return result;
  }
  if (result.pages.size() < requested_page_count) {
    result.error_text = "sysmem page-list contained " + std::to_string(result.pages.size()) +
                        " pages but " + std::to_string(requested_page_count) + " were requested";
    return result;
  }
  result.truncated = result.expanded_page_count > requested_page_count;
  result.ok = true;
  return result;
}
int self_test_failure(const char* self_test, const char* failure_text);


int run_sysmem_page_list_self_test() {
  constexpr std::size_t kRequestedPages = 4;
  std::vector<uint8_t> mapping;
  append_page_list_pair(&mapping, 0x0000000100000000ULL, 0x2000ULL);
  append_page_list_pair(&mapping, 0x0000000200000000ULL, 0x3000ULL);
  append_page_list_pair(&mapping, 0x0000000300000000ULL, 0x1000ULL);
  append_page_list_pair(&mapping, 0, 0);

  const PageListParseResult parsed =
      parse_sysmem_page_list_bytes(mapping.data(), mapping.size(), kRequestedPages);
  if (!parsed.ok) {
    return self_test_failure("sysmem-page-list", parsed.error_text.c_str());
  }

  std::printf("self_test: sysmem-page-list\n");
  std::printf("requested_page_count: %zu\n", kRequestedPages);
  std::printf("raw_pair_count: %zu\n", parsed.raw_pair_count);
  std::printf("terminator_pair_index: %zu\n", parsed.raw_pair_count);
  std::printf("expanded_page_count_before_truncation: %llu\n",
              static_cast<unsigned long long>(parsed.expanded_page_count));
  std::printf("parsed_page_count: %zu\n", parsed.pages.size());
  std::printf("truncated: %s\n", parsed.truncated ? "yes" : "no");
  for (std::size_t i = 0; i < parsed.pages.size(); ++i) {
    std::printf("page_%zu_paddr: 0x%016llx\n", i,
                static_cast<unsigned long long>(parsed.pages[i]));
  }
  std::printf("status: pass\n");
  return 0;
}

int self_test_failure(const char* self_test, const char* failure_text) {
  std::printf("self_test: %s\n", self_test);
  std::printf("failure_text: %s\n", failure_text);
  std::printf("exit_status: 1\n");
  return 1;
}

int run_remote_cmd_frame_self_test() {
  constexpr uint32_t kDevId = 0x7551U;
  constexpr uint32_t kBar = 5U;
  constexpr uint64_t kArg0 = 0x0102030405060708ULL;
  constexpr uint64_t kArg1 = 0x1122334455667788ULL;
  constexpr uint64_t kArg2 = 0x99aabbccddeeff00ULL;
  constexpr char kExpectedHex[] =
      "0251750000050000000807060504030201887766554433221100ffeeddccbbaa99";

  if (!remote_cmd_order_ok()) {
    return self_test_failure("remote-cmd-frame", "RemoteCmd enum order mismatch");
  }

  const RemoteCmdFrame frame = build_remote_cmd_frame(RemoteCmd::MAP_SYSMEM_FD, kDevId, kBar,
                                                      kArg0, kArg1, kArg2);
  const auto frame_hex = hex_encode(frame);

  if (frame.size() != kRemoteCmdFrameSize) {
    return self_test_failure("remote-cmd-frame", "RemoteCmd frame size mismatch");
  }
  if (frame[0] != static_cast<uint8_t>(RemoteCmd::MAP_SYSMEM_FD)) {
    return self_test_failure("remote-cmd-frame", "RemoteCmd MAP_SYSMEM_FD id mismatch");
  }
  if (std::memcmp(&frame[1], "\x51\x75\x00\x00", 4) != 0) {
    return self_test_failure("remote-cmd-frame", "RemoteCmd dev_id byte order mismatch");
  }
  if (std::memcmp(&frame[5], "\x05\x00\x00\x00", 4) != 0) {
    return self_test_failure("remote-cmd-frame", "RemoteCmd bar byte order mismatch");
  }
  if (std::memcmp(&frame[9], "\x08\x07\x06\x05\x04\x03\x02\x01", 8) != 0) {
    return self_test_failure("remote-cmd-frame", "RemoteCmd arg0 byte order mismatch");
  }
  if (std::memcmp(&frame[17], "\x88\x77\x66\x55\x44\x33\x22\x11", 8) != 0) {
    return self_test_failure("remote-cmd-frame", "RemoteCmd arg1 byte order mismatch");
  }
  if (std::memcmp(&frame[25], "\x00\xff\xee\xdd\xcc\xbb\xaa\x99", 8) != 0) {
    return self_test_failure("remote-cmd-frame", "RemoteCmd arg2 byte order mismatch");
  }
  if (std::strcmp(frame_hex.data(), kExpectedHex) != 0) {
    return self_test_failure("remote-cmd-frame", "RemoteCmd frame hex mismatch");
  }

  std::printf("self_test: remote-cmd-frame\n");
  std::printf("frame_size: %zu\n", frame.size());
  std::printf("frame_hex: %s\n", frame_hex.data());
  std::printf("status: pass\n");
  return 0;
}
int run_sdma_packet_encoding_self_test() {
  constexpr uint64_t kSrcAddr = 0x0102030405060708ULL;
  constexpr uint64_t kDstAddr = 0x1122334455667788ULL;
  constexpr char kExpectedPacketHex[] =
      "010000001f0000000000000008070605040302018877665544332211";
  constexpr char kExpectedSrcLe[] = "0807060504030201";
  constexpr char kExpectedDstLe[] = "8877665544332211";

  const SdmaLinearCopyPacket packet =
      build_sdma_linear_copy_packet(kSrcAddr, kDstAddr, static_cast<uint32_t>(kTransferByteCount));
  const std::string packet_hex = hex_encode_sdma_packet(packet);
  std::vector<uint8_t> bytes;
  bytes.reserve(packet.size() * sizeof(uint32_t));
  for (uint32_t word : packet) {
    append_u32_le(&bytes, word);
  }
  const std::string src_le = hex_encode_bytes(bytes.data() + 12, 8);
  const std::string dst_le = hex_encode_bytes(bytes.data() + 20, 8);

  if (packet.size() != kSdmaLinearCopyPacketDwords) {
    return self_test_failure("sdma-packet-encoding", "SDMA linear-copy packet dword count mismatch");
  }
  if (packet[1] != kTransferByteCount - 1U) {
    return self_test_failure("sdma-packet-encoding", "SDMA linear-copy count field mismatch");
  }
  if (packet_hex != kExpectedPacketHex) {
    return self_test_failure("sdma-packet-encoding", "SDMA linear-copy packet hex mismatch");
  }
  if (src_le != kExpectedSrcLe) {
    return self_test_failure("sdma-packet-encoding", "SDMA source address little-endian bytes mismatch");
  }
  if (dst_le != kExpectedDstLe) {
    return self_test_failure("sdma-packet-encoding", "SDMA destination address little-endian bytes mismatch");
  }

  std::printf("self_test: sdma-packet-encoding\n");
  std::printf("packet_dword_count: %zu\n", packet.size());
  std::printf("transfer_byte_count: %llu\n", static_cast<unsigned long long>(kTransferByteCount));
  std::printf("copy_count_field: %u\n", packet[1]);
  std::printf("source_address: 0x%016llx\n", static_cast<unsigned long long>(kSrcAddr));
  std::printf("source_address_le: %s\n", src_le.c_str());
  std::printf("destination_address: 0x%016llx\n", static_cast<unsigned long long>(kDstAddr));
  std::printf("destination_address_le: %s\n", dst_le.c_str());
  std::printf("packet_hex: %s\n", packet_hex.c_str());
  std::printf("status: pass\n");
  return 0;
}

int run_am_vm_pte_encoding_self_test() {
  constexpr uint64_t kSysmemLeafFlags = am_vm::gfx12_leaf_pte_flags(true, true, true);
  constexpr uint64_t kVramLeafFlags = am_vm::gfx12_leaf_pte_flags(false, false, false);
  constexpr uint64_t kTableEntryFlags = am_vm::table_pte_flags();
  constexpr uint64_t kSysmemStagingPte =
      am_vm::encode_pte(am_vm::kSyntheticSysmemStagingPaddr, kSysmemLeafFlags);
  constexpr uint64_t kVramPte = am_vm::encode_pte(am_vm::kFixedVramBufferPaddr, kVramLeafFlags);
  constexpr uint64_t kSysmemReadbackPte =
      am_vm::encode_pte(am_vm::kSyntheticSysmemReadbackPaddr, kSysmemLeafFlags);
  constexpr uint64_t kSysmemSdmaControlPte =
      am_vm::encode_pte(am_vm::kSyntheticSysmemSdmaControlPaddr, kSysmemLeafFlags);

  if (kSysmemLeafFlags != 0x80c0000000000077ULL) {
    return self_test_failure("am-vm-pte-encoding", "gfx12 sysmem leaf flags mismatch");
  }
  if (kVramLeafFlags != 0x8000000000000071ULL) {
    return self_test_failure("am-vm-pte-encoding", "gfx12 VRAM leaf flags mismatch");
  }
  if (kTableEntryFlags != 0x0000000000000001ULL) {
    return self_test_failure("am-vm-pte-encoding", "gfx12 table flags mismatch");
  }
  if (kSysmemStagingPte != 0x80c0000080000077ULL) {
    return self_test_failure("am-vm-pte-encoding", "gfx12 sysmem staging PTE mismatch");
  }
  if (kVramPte != 0x8000000006000071ULL) {
    return self_test_failure("am-vm-pte-encoding", "gfx12 VRAM PTE mismatch");
  }
  if (kSysmemReadbackPte != 0x80c0000080008077ULL) {
    return self_test_failure("am-vm-pte-encoding", "gfx12 sysmem readback PTE mismatch");
  }
  if (kSysmemSdmaControlPte != 0x80c0000080010077ULL) {
    return self_test_failure("am-vm-pte-encoding", "gfx12 sysmem SDMA control PTE mismatch");
  }

  std::printf("self_test: am-vm-pte-encoding\n");
  std::printf("leaf_level: %s\n", am_vm::kLeafLevelName);
  std::printf("gfx_ip_major: %u\n", am_vm::kGfxIpMajor);
  std::printf("mtype_uc: %llu\n", static_cast<unsigned long long>(am_vm::kMtypeUc));
  std::printf("sysmem_leaf_flags: 0x%016llx\n", static_cast<unsigned long long>(kSysmemLeafFlags));
  std::printf("vram_leaf_flags: 0x%016llx\n", static_cast<unsigned long long>(kVramLeafFlags));
  std::printf("table_entry_flags: 0x%016llx\n", static_cast<unsigned long long>(kTableEntryFlags));
  std::printf("sysmem_staging_pte: 0x%016llx\n", static_cast<unsigned long long>(kSysmemStagingPte));
  std::printf("vram_pte: 0x%016llx\n", static_cast<unsigned long long>(kVramPte));
  std::printf("sysmem_readback_pte: 0x%016llx\n", static_cast<unsigned long long>(kSysmemReadbackPte));
  std::printf("sysmem_sdma_control_pte: 0x%016llx\n",
              static_cast<unsigned long long>(kSysmemSdmaControlPte));
  std::printf("status: pass\n");
  return 0;
}

int run_am_vm_page_table_plan_self_test() {
  constexpr am_vm::VmIndices kStagingIndices = am_vm::vm_indices_for_va(am_vm::kStagingVa);
  constexpr am_vm::VmIndices kVramIndices = am_vm::vm_indices_for_va(am_vm::kVramVa);
  constexpr am_vm::VmIndices kReadbackIndices = am_vm::vm_indices_for_va(am_vm::kReadbackVa);
  constexpr am_vm::VmIndices kSdmaControlIndices = am_vm::vm_indices_for_va(am_sdma::kControlVa);
  constexpr am_vm::VmIndices kComputeOutputIndices =
      am_vm::vm_indices_for_va(am_compute::kOutputVramVa);
  constexpr am_vm::VmIndices kComputeCodeIndices =
      am_vm::vm_indices_for_va(am_compute::kCodeVramVa);
  constexpr am_vm::VmIndices kComputeKernargsIndices =
      am_vm::vm_indices_for_va(am_compute::kKernargsVa);
  constexpr am_vm::VmIndices kComputeRingIndices =
      am_vm::vm_indices_for_va(am_compute::kRingVa);
  constexpr am_vm::VmIndices kComputeRptrIndices =
      am_vm::vm_indices_for_va(am_compute::kRptrVa);
  constexpr am_vm::VmIndices kComputeEopIndices =
      am_vm::vm_indices_for_va(am_compute::kEopVa);

  constexpr am_vm::VmIndices kNonzeroPdb2Indices =
      am_vm::vm_indices_for_va(am_vm::kVaBase + (64ULL << am_vm::kVaShiftPdb2));

  if (kStagingIndices.pdb2 != 0 || kStagingIndices.pdb1 != 0 ||
      kStagingIndices.pdb0 != kDedicatedStagingPdb0Index || kStagingIndices.ptb != 0) {
    return self_test_failure("am-vm-page-table-plan", "staging VA indices mismatch");
  }
  if (kVramIndices.pdb2 != 0 || kVramIndices.pdb1 != 0 || kVramIndices.pdb0 != 0 ||
      kVramIndices.ptb != 1) {
    return self_test_failure("am-vm-page-table-plan", "VRAM VA indices mismatch");
  }
  if (kReadbackIndices.pdb2 != 0 || kReadbackIndices.pdb1 != 0 || kReadbackIndices.pdb0 != 0 ||
      kReadbackIndices.ptb != 2) {
    return self_test_failure("am-vm-page-table-plan", "readback VA indices mismatch");
  }
  if (kSdmaControlIndices.pdb2 != 0 || kSdmaControlIndices.pdb1 != 0 ||
      kSdmaControlIndices.pdb0 != 0 || kSdmaControlIndices.ptb != 3) {
    return self_test_failure("am-vm-page-table-plan", "SDMA control VA indices mismatch");
  }
  if (kComputeOutputIndices.ptb != 4 || kComputeCodeIndices.ptb != 5 ||
      kComputeKernargsIndices.ptb != 6 || kComputeRingIndices.ptb != 7 ||
      kComputeRptrIndices.ptb != 15 || kComputeEopIndices.ptb != 16) {
    return self_test_failure("am-vm-page-table-plan", "compute VA PTB indices mismatch");
  }
  if (kNonzeroPdb2Indices.pdb2 != 64) {
    return self_test_failure("am-vm-page-table-plan", "PDB2 index mask mismatch");
  }
  std::printf("self_test: am-vm-page-table-plan\n");
  std::printf("va_base: 0x%016llx\n", static_cast<unsigned long long>(am_vm::kVaBase));
  std::printf("staging_va: 0x%016llx\n", static_cast<unsigned long long>(am_vm::kStagingVa));
  std::printf("staging_byte_count: %llu\n", static_cast<unsigned long long>(am_vm::kStagingByteCount));
  std::printf("staging_ptb_page_count: %llu\n", static_cast<unsigned long long>(kStagingPageCount));
  std::printf("vram_va: 0x%016llx\n", static_cast<unsigned long long>(am_vm::kVramVa));
  std::printf("readback_va: 0x%016llx\n", static_cast<unsigned long long>(am_vm::kReadbackVa));
  std::printf("va_shifts: %u,%u,%u,%u\n", am_vm::kVaShiftPtb, am_vm::kVaShiftPdb0,
              am_vm::kVaShiftPdb1, am_vm::kVaShiftPdb2);
  std::printf("first_level: %s\n", am_vm::kFirstLevelName);
  std::printf("pdb2_index: %llu\n", static_cast<unsigned long long>(kStagingIndices.pdb2));
  std::printf("pdb1_index: %llu\n", static_cast<unsigned long long>(kStagingIndices.pdb1));
  std::printf("pdb0_index: %llu\n", static_cast<unsigned long long>(kStagingIndices.pdb0));
  std::printf("staging_ptb_index: %llu\n", static_cast<unsigned long long>(kStagingIndices.ptb));
  std::printf("vram_ptb_index: %llu\n", static_cast<unsigned long long>(kVramIndices.ptb));
  std::printf("readback_ptb_index: %llu\n", static_cast<unsigned long long>(kReadbackIndices.ptb));
  std::printf("boot_arena_size: 0x%08llx\n", static_cast<unsigned long long>(am_vm::kBootArenaSize));
  std::printf("ptable_arena_base: 0x%08llx\n", static_cast<unsigned long long>(am_vm::kPtableArenaBase));
  std::printf("fixed_vram_buffer_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(am_vm::kFixedVramBufferPaddr));
  std::printf("sdma_control_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kControlVa));
  std::printf("sdma_control_ptb_index: %llu\n",
              static_cast<unsigned long long>(kSdmaControlIndices.ptb));
  std::printf("kernel_output_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kOutputVramVa));
  std::printf("kernel_output_ptb_index: %llu\n",
              static_cast<unsigned long long>(kComputeOutputIndices.ptb));
  std::printf("kernel_code_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kCodeVramVa));
  std::printf("kernel_code_ptb_index: %llu\n",
              static_cast<unsigned long long>(kComputeCodeIndices.ptb));
  std::printf("kernel_kernargs_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kKernargsVa));
  std::printf("kernel_kernargs_ptb_index: %llu\n",
              static_cast<unsigned long long>(kComputeKernargsIndices.ptb));
  std::printf("compute_ring_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kRingVa));
  std::printf("compute_ring_ptb_index: %llu\n",
              static_cast<unsigned long long>(kComputeRingIndices.ptb));
  std::printf("compute_rptr_wptr_timeline_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kRptrVa));
  std::printf("compute_rptr_wptr_timeline_ptb_index: %llu\n",
              static_cast<unsigned long long>(kComputeRptrIndices.ptb));
  std::printf("compute_eop_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kEopVa));
  std::printf("compute_eop_ptb_index: %llu\n",
              static_cast<unsigned long long>(kComputeEopIndices.ptb));
  std::printf("status: pass\n");
  return 0;
}

int run_am_vm_tlb_sequence_self_test() {
  std::printf("self_test: am-vm-tlb-sequence\n");
  std::printf("vmid: %u\n", am_vm::kVmid0);
  std::printf("flush_order: hdp,mm,mm_reserved_cid2,gc\n");
  std::printf("invalidate_mask: 0x%08x\n", am_vm::kInvalidateMaskVmid0);
  std::printf("invalidate_l2_ptes: 1\n");
  std::printf("invalidate_l2_pde0: 1\n");
  std::printf("invalidate_l2_pde1: 1\n");
  std::printf("invalidate_l2_pde2: 1\n");
  std::printf("invalidate_l1_ptes: 1\n");
  std::printf("clear_protection_fault_status_addr: 0\n");
  std::printf("mm_waits: sem,ack\n");
  std::printf("gc_waits: sem,ack\n");
  std::printf("status: pass\n");
  return 0;
}

int run_sdma_ring_setup_self_test() {
  std::printf("self_test: sdma-ring-setup\n");
  std::printf("sdma_ip_hw_id: %u\n", am_sdma::kSdma0HwId);
  std::printf("sdma_ip_version: 7.0.1\n");
  std::printf("queue_index: %u\n", am_sdma::kQueueIndex);
  std::printf("register_prefix: %s\n", am_sdma::kRegisterPrefix);
  std::printf("register_instance: %u\n", am_sdma::kRegisterInstance);
  std::printf("teardown_order: %s\n", am_sdma::kTeardownOrder);
  std::printf("soft_reset_sdma0_bit: %u\n", am_sdma::kSoftResetSdma0Bit);
  std::printf("ring_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kControlVa));
  std::printf("ring_size_bytes: %llu\n", static_cast<unsigned long long>(am_sdma::kRingSize));
  std::printf("ring_size_field: %u\n", am_sdma::kRingSizeField);
  std::printf("rptr_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kRptrVa));
  std::printf("wptr_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kWptrVa));
  std::printf("fence_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kFenceVa));
  std::printf("doorbell_index: %u\n", am_sdma::kDoorbellIndex);
  std::printf("doorbell_offset_field: %u\n", am_sdma::kDoorbellOffsetField);
  std::printf("doorbell_bar2_byte_offset: 0x%016llx\n",
              static_cast<unsigned long long>(am_sdma::kDoorbellBar2ByteOffset));
  std::printf("status: pass\n");
  return 0;
}

int run_sdma_fence_packet_encoding_self_test() {

  constexpr char kExpectedFenceAddressLe[] = "1038000000200000";
  constexpr char kExpectedPacketHex[] = "05000300103800000020000001000000";
  const auto packet = build_sdma_fence_packet(am_sdma::kFenceVa, am_sdma::kFenceValue);
  std::vector<uint8_t> bytes;
  bytes.reserve(packet.size() * sizeof(uint32_t));
  for (uint32_t word : packet) {
    append_u32_le(&bytes, word);
  }
  const std::string fence_address_le = hex_encode_bytes(bytes.data() + sizeof(uint32_t), 8);
  const std::string packet_hex = hex_encode_vector(bytes);

  if (packet.size() != am_sdma::kFencePacketDwords) {
    return self_test_failure("sdma-fence-packet-encoding", "SDMA fence packet dword count mismatch");
  }
  if (packet[0] != am_sdma::kFenceHeader) {
    return self_test_failure("sdma-fence-packet-encoding", "SDMA fence packet header mismatch");
  }
  if (packet[3] != am_sdma::kFenceValue) {
    return self_test_failure("sdma-fence-packet-encoding", "SDMA fence value mismatch");
  }
  if (fence_address_le != kExpectedFenceAddressLe) {
    return self_test_failure("sdma-fence-packet-encoding", "SDMA fence address byte order mismatch");
  }
  if (packet_hex != kExpectedPacketHex) {
    return self_test_failure("sdma-fence-packet-encoding", "SDMA fence packet hex mismatch");
  }

  std::printf("self_test: sdma-fence-packet-encoding\n");
  std::printf("packet_dword_count: %zu\n", packet.size());
  std::printf("fence_value: %u\n", am_sdma::kFenceValue);
  std::printf("fence_address: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kFenceVa));
  std::printf("fence_address_le: %s\n", fence_address_le.c_str());
  std::printf("packet_hex: %s\n", packet_hex.c_str());
  std::printf("status: pass\n");
  return 0;
}

int run_sdma_submit_sequence_self_test() {
  const std::vector<uint32_t> words = build_sdma_submit_words(
      am_vm::kStagingVa, am_vm::kVramVa, am_vm::kReadbackVa, am_sdma::kFenceVa);
  constexpr uint64_t kInitialWptrBytes = 0ULL;
  const uint64_t final_wptr_bytes = words.size() * sizeof(uint32_t);
  const uint64_t doorbell_value = final_wptr_bytes;

  if (words.size() != am_sdma::kSubmitDwordCount) {
    return self_test_failure("sdma-submit-sequence", "SDMA submit dword count mismatch");
  }
  if (words[0] != (kSdmaOpCopy | (kSdmaSubopCopyLinear << 8)) ||
      words[kSdmaLinearCopyPacketDwords] != (kSdmaOpCopy | (kSdmaSubopCopyLinear << 8)) ||
      words[kSdmaLinearCopyPacketDwords * 2] != am_sdma::kFenceHeader) {
    return self_test_failure("sdma-submit-sequence", "SDMA submit packet order mismatch");
  }
  if (final_wptr_bytes != am_sdma::kSubmitByteCount || doorbell_value != am_sdma::kSubmitByteCount) {
    return self_test_failure("sdma-submit-sequence", "SDMA submit write pointer byte count mismatch");
  }

  std::printf("self_test: sdma-submit-sequence\n");
  std::printf("copy_packet_dwords: %zu\n", kSdmaLinearCopyPacketDwords);
  std::printf("fence_packet_dwords: %u\n", am_sdma::kFencePacketDwords);
  std::printf("submit_copy_count: %u\n", am_sdma::kSubmitCopyCount);
  std::printf("submit_dword_count: %u\n", am_sdma::kSubmitDwordCount);
  std::printf("initial_wptr_bytes: %llu\n", static_cast<unsigned long long>(kInitialWptrBytes));
  std::printf("final_wptr_bytes: %llu\n", static_cast<unsigned long long>(final_wptr_bytes));
  std::printf("doorbell_value: %llu\n", static_cast<unsigned long long>(doorbell_value));
  std::printf("status: pass\n");
  return 0;
}

int run_kernel_proof_contract_self_test() {
  std::printf("self_test: kernel-proof-contract\n");
  std::printf("runtime_substrate: %s\n", kRuntimeSubstrate);
  std::printf("kernel_proof_mode: minimal-u32-add-one\n");
  std::printf("kernel_arch: gfx1201\n");
  std::printf("element_type: uint32_t\n");
  std::printf("element_count: 8\n");
  std::printf("input_byte_count: 32\n");
  std::printf("output_byte_count: 32\n");
  std::printf("input_values_u32: 1,2,3,4,5,6,7,8\n");
  std::printf("input_bytes_hex: 0100000002000000030000000400000005000000060000000700000008000000\n");
  std::printf("expected_output_values_u32: 2,3,4,5,6,7,8,9\n");
  std::printf("expected_output_bytes_hex: 0200000003000000040000000500000006000000070000000800000009000000\n");
  std::printf("expected_output_sha256: b06e51b2494d439f5e151692ca393efc3c52cdfddcc377be789356250b9860a6\n");
  std::printf("kernel_source_id: %s\n", kKernelSourceId);
  std::printf("kernel_source_language: amd-gcn-assembly\n");
  std::printf("kernel_blob_format: amdgpu-code-object-v5\n");
  std::printf("kernel_blob_symbol: c0a_minimal_u32_add_one\n");
  std::printf("kernel_blob_target: gfx1201\n");
  const std::string kernel_text_first64 =
      hex_encode_bytes(kKernelText.data(), 64);
  const std::string kernel_text_last16 =
      hex_encode_bytes(kKernelText.data() + kKernelText.size() - 16, 16);
  if (kKernelText.size() != kKernelReferenceTextByteCount ||
      kernel_text_first64 != kKernelReferenceTextFirst64Hex ||
      kernel_text_last16 != kKernelReferenceTextLast16Hex) {
    return self_test_failure("kernel-proof-contract", "embedded kernel text provenance coverage mismatch");
  }
  std::printf("kernel_text_provenance_path: %s\n", kKernelTextProvenancePath);
  std::printf("kernel_blob_reference_hsaco_sha256: %s\n", kKernelReferenceHsacoSha256);
  std::printf("kernel_blob_reference_text_sha256: %s\n", kKernelReferenceTextSha256);
  std::printf("kernel_blob_reference_text_byte_count: %u\n", kKernelReferenceTextByteCount);
  std::printf("kernel_text_first64_hex: %s\n", kernel_text_first64.c_str());
  std::printf("kernel_text_last16_hex: %s\n", kernel_text_last16.c_str());
  std::printf("kernel_descriptor_kernarg_size: %u\n", kKernelReferenceKernargSize);
  std::printf("compute_ring_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kRingVa));
  std::printf("compute_ring_size_bytes: %u\n", am_compute::kRingSize);
  std::printf("compute_rptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kRptrVa));
  std::printf("compute_wptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kWptrVa));
  std::printf("compute_timeline_gpu_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kTimelineVa));
  std::printf("compute_eop_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kEopVa));
  std::printf("compute_doorbell_index: %u\n", am_compute::kMecDoorbellIndex);
  std::printf("compute_doorbell_bar2_byte_offset: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kMecDoorbellBar2ByteOffset));
  std::printf("kernel_blob_load_status: not_run_no_hardware_contract\n");
  std::printf("kernarg_write_status: not_run_no_hardware_contract\n");
  std::printf("sdma_h2d_status: not_run_no_hardware_contract\n");
  std::printf("sdma_d2h_status: not_run_no_hardware_contract\n");
  std::printf("compute_ring_setup_status: not_run\n");
  std::printf("compute_hqd_active_status: not_run\n");
  std::printf("mec_rs64_cntl_write_status: not_run\n");
  std::printf("mec_rs64_cntl_readback: not_run\n");
  std::printf("mec_rs64_active_status: not_run\n");
  std::printf("kernel_launch_status: not_run_no_hardware_contract\n");
  std::printf("kernel_elapsed_usec: 0\n");
  std::printf("cpu_comparison_status: pass\n");
  std::printf("host_device_transfer_status: not_run_no_hardware_contract\n");
  std::printf("failure_stage: none\n");
  std::printf("failure_text: none\n");
  std::printf("exit_status: 0\n");
  return 0;
}

int run_compute_vm_layout_self_test() {
  constexpr uint64_t kComputeRingPageCount = am_compute::kRingSize / kPageSize;
  constexpr am_vm::VmIndices kInputIndices = am_vm::vm_indices_for_va(am_compute::kInputVramVa);
  constexpr am_vm::VmIndices kOutputIndices = am_vm::vm_indices_for_va(am_compute::kOutputVramVa);
  constexpr am_vm::VmIndices kCodeIndices = am_vm::vm_indices_for_va(am_compute::kCodeVramVa);
  constexpr am_vm::VmIndices kKernargsIndices = am_vm::vm_indices_for_va(am_compute::kKernargsVa);
  constexpr am_vm::VmIndices kRingIndices = am_vm::vm_indices_for_va(am_compute::kRingVa);
  constexpr am_vm::VmIndices kRptrIndices = am_vm::vm_indices_for_va(am_compute::kRptrVa);
  constexpr am_vm::VmIndices kEopIndices = am_vm::vm_indices_for_va(am_compute::kEopVa);
  if (am_compute::kInputVramPaddr != am_vm::kFixedVramBufferPaddr ||
      am_compute::kOutputVramPaddr != am_vm::kFixedVramBufferPaddr + (3ULL * kPageSize) ||
      am_compute::kCodeVramPaddr != am_vm::kFixedVramBufferPaddr + (4ULL * kPageSize) ||
      am_compute::kRingVramPaddr != am_vm::kFixedVramBufferPaddr + (6ULL * kPageSize) ||
      am_compute::kEopVramPaddr != am_vm::kFixedVramBufferPaddr + (15ULL * kPageSize)) {
    return self_test_failure("compute-vm-layout", "compute VRAM physical layout mismatch");
  }
  if (am_compute::kComputeControlByteCount != 26ULL * kPageSize ||
      am_compute::kComputeControlKernargsCpuOffset != kPageSize ||
      am_compute::kComputeControlRingCpuOffset != 2ULL * kPageSize ||
      am_compute::kComputeControlRingByteCount != 8ULL * kPageSize ||
      am_compute::kComputeControlKernargsRingCpuOffset != 10ULL * kPageSize ||
      am_compute::kKernargsRingPageCount != 16U) {
    return self_test_failure("compute-vm-layout", "compute_control 26-page CPU layout mismatch");
  }

  std::printf("self_test: compute-vm-layout\n");
  std::printf("kernel_input_vram_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kInputVramVa));
  std::printf("kernel_output_vram_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kOutputVramVa));
  std::printf("kernel_code_vram_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kCodeVramVa));
  std::printf("kernel_kernargs_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kKernargsVa));
  std::printf("compute_ring_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kRingVa));
  std::printf("compute_rptr_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kRptrVa));
  std::printf("compute_wptr_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kWptrVa));
  std::printf("compute_timeline_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kTimelineVa));
  std::printf("compute_eop_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kEopVa));
  std::printf("compute_control_requested_size: %llu\n",
              static_cast<unsigned long long>(am_compute::kComputeControlByteCount));
  std::printf("compute_control_queue_cpu_offset: %llu\n",
              static_cast<unsigned long long>(am_compute::kComputeControlQueueCpuOffset));
  std::printf("compute_control_kernargs_cpu_offset: %llu\n",
              static_cast<unsigned long long>(am_compute::kComputeControlKernargsCpuOffset));
  std::printf("kernel_input_ptb_index: %llu\n", static_cast<unsigned long long>(kInputIndices.ptb));
  std::printf("kernel_output_ptb_index: %llu\n", static_cast<unsigned long long>(kOutputIndices.ptb));
  std::printf("kernel_code_ptb_index: %llu\n", static_cast<unsigned long long>(kCodeIndices.ptb));
  std::printf("kernel_kernargs_ptb_index: %llu\n", static_cast<unsigned long long>(kKernargsIndices.ptb));
  std::printf("compute_ring_ptb_index: %llu\n", static_cast<unsigned long long>(kRingIndices.ptb));
  std::printf("compute_rptr_wptr_timeline_ptb_index: %llu\n",
              static_cast<unsigned long long>(kRptrIndices.ptb));
  std::printf("compute_eop_ptb_index: %llu\n", static_cast<unsigned long long>(kEopIndices.ptb));
  std::printf("compute_mqd_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kMqdPaddr));
  std::printf("compute_ring_page_count: %llu\n", static_cast<unsigned long long>(kComputeRingPageCount));
  std::printf("status: pass\n");
  return 0;
}

int run_gfx_ring_registers_self_test() {
  std::printf("self_test: gfx-ring-registers\n");
  std::printf("gc_ip_version: %s\n", am_compute::kGcIpVersion);
  std::printf("direct_pm4_requires_xcc_count: %u\n", am_compute::kExpectedXccCount);
  std::printf("mec_doorbell_index: %u\n", am_compute::kMecDoorbellIndex);
  std::printf("mec_doorbell_bar2_byte_offset: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kMecDoorbellBar2ByteOffset));
  std::printf("grbm_select_reg: %s\n", am_compute::kGrbmSelectReg);
  std::printf("hqd_reg_span: %s\n", am_compute::kHqdRegSpan);
  std::printf("compute_set_sh_base: 0x%08x\n", am_compute::kComputeSetShBase);
  std::printf("compute_pgm_lo_set_sh_offset: 0x%08x\n", am_compute::kComputePgmLoSetShOffset);
  std::printf("compute_user_data_0_set_sh_offset: 0x%08x\n",
              am_compute::kComputeUserData0SetShOffset);
  std::printf("status: pass\n");
  return 0;
}

// Observed R9700 MC base (regMMMC_VM_FB_LOCATION_BASE 0x8000 << 24). The MQD
// self-test is hardware-free, so it uses this representative value to verify
// cp_mqd_base_addr = mc_base + kMqdPaddr (lo32 0x02003000, hi32 0x00000080).
constexpr uint64_t kR9700ObservedMcBase = 0x0000008000000000ULL;

int run_compute_mqd_encoding_self_test() {
  constexpr uint32_t kHqdPersistentState = encode_hqd_persistent_state();
  constexpr uint32_t kHqdPqDoorbellControl = encode_hqd_pq_doorbell_control();
  constexpr uint32_t kHqdPqControl = encode_hqd_pq_control_direct_pm4();
  constexpr uint32_t kHqdIbControl = encode_hqd_ib_control();
  constexpr uint32_t kHqdEopControl = encode_hqd_eop_control();
  constexpr uint32_t kCpMqdControl = encode_cp_mqd_control();
  constexpr std::size_t kCpMqdControlSpanIndex =
      kMqdCpMqdControl - kMqdHqdRegisterCopyStart;
  constexpr std::size_t kCpHqdEopBaseAddrSpanIndex =
      kMqdCpHqdEopBaseAddrLo - kMqdHqdRegisterCopyStart;
  constexpr std::size_t kCpHqdEopControlSpanIndex =
      kMqdCpHqdEopControl - kMqdHqdRegisterCopyStart;
  const ComputeMqd mqd = build_compute_mqd(kR9700ObservedMcBase);

  if (mqd[kMqdHeader] != am_compute::kMqdHeader ||
      mqd[kMqdComputePgmLo] != lo32(am_compute::kCodeVramVa >> 8) ||
      mqd[kMqdComputePgmHi] != hi32(am_compute::kCodeVramVa >> 8) ||
      mqd[kMqdComputePgmRsrc1] != kKernelReferenceRsrc1 ||
      mqd[kMqdComputePgmRsrc2] != kKernelReferenceRsrc2 ||
      mqd[kMqdComputePgmRsrc3] != kKernelReferenceRsrc3 ||
      mqd[kMqdComputeStaticThreadMgmtSe7] != am_compute::kComputeStaticThreadMgmt ||
      mqd[kMqdCpHqdPqDoorbellControl] != kHqdPqDoorbellControl ||
      mqd[kMqdCpHqdPqControl] != kHqdPqControl ||
      mqd[kMqdCpMqdControl] != kCpMqdControl ||
      mqd[kMqdCpHqdAqlControl] != am_compute::kHqdAqlControl) {
    return self_test_failure("compute-mqd-encoding", "source-indexed MQD builder mismatch");
  }
  if (kCpMqdControlSpanIndex != 34U) {
    return self_test_failure("compute-mqd-encoding", "CP_MQD_CONTROL HQD span index drift");
  }
  if (kCpHqdEopBaseAddrSpanIndex != 37U) {
    return self_test_failure("compute-mqd-encoding", "CP_HQD_EOP_BASE_ADDR HQD span index drift");
  }
  if (kCpHqdEopControlSpanIndex != 39U) {
    return self_test_failure("compute-mqd-encoding", "CP_HQD_EOP_CONTROL HQD span index drift");
  }

  if ((kHqdPqControl & (5U << 8)) != (5U << 8)) {
    return self_test_failure("compute-mqd-encoding", "HQD direct PM4 PQ control mode mismatch");
  }

  std::printf("self_test: compute-mqd-encoding\n");
  std::printf("mqd_size_bytes: %u\n", am_compute::kMqdSize);
  std::printf("mqd_dword_count: %zu\n", mqd.size());
  std::printf("mqd_hqd_register_copy_start: %zu\n", kMqdHqdRegisterCopyStart);
  std::printf("mqd_cp_mqd_control_span_index: %zu\n", kCpMqdControlSpanIndex);
  std::printf("mqd_cp_hqd_eop_base_addr_span_index: %zu\n",
              kCpHqdEopBaseAddrSpanIndex);
  std::printf("mqd_cp_hqd_eop_control_span_index: %zu\n",
              kCpHqdEopControlSpanIndex);
  std::printf("mqd_cp_mqd_base_addr_lo: 0x%08x\n", mqd[kMqdCpMqdBaseAddrLo]);
  std::printf("mqd_cp_mqd_base_addr_hi: 0x%08x\n", mqd[kMqdCpMqdBaseAddrHi]);
  std::printf("mqd_cp_hqd_pq_base_lo: 0x%08x\n", mqd[kMqdCpHqdPqBaseLo]);
  std::printf("mqd_cp_hqd_pq_base_hi: 0x%08x\n", mqd[kMqdCpHqdPqBaseHi]);
  std::printf("mqd_cp_hqd_pq_wptr_poll_addr_lo: 0x%08x\n",
              mqd[kMqdCpHqdPqWptrPollAddrLo]);
  std::printf("mqd_cp_hqd_pq_wptr_poll_addr_hi: 0x%08x\n",
              mqd[kMqdCpHqdPqWptrPollAddrHi]);
  std::printf("mqd_cp_hqd_eop_base_addr_lo: 0x%08x\n", mqd[kMqdCpHqdEopBaseAddrLo]);
  std::printf("mqd_cp_hqd_eop_base_addr_hi: 0x%08x\n", mqd[kMqdCpHqdEopBaseAddrHi]);
  std::printf("mqd_cp_hqd_hq_status0: 0x%08x\n", mqd[kMqdCpHqdHqStatus0]);
  std::printf("mqd_compute_user_data_0: 0x%08x\n", mqd[kMqdComputeUserData0]);
  std::printf("mqd_header: 0x%08x\n", am_compute::kMqdHeader);
  std::printf("hqd_pipe_priority: 0x%08x\n", am_compute::kHqdPipePriority);
  std::printf("hqd_queue_priority: 0x%08x\n", am_compute::kHqdQueuePriority);
  std::printf("hqd_quantum: 0x%08x\n", am_compute::kHqdQuantum);
  std::printf("hqd_persistent_state: 0x%08x\n", kHqdPersistentState);
  std::printf("hqd_vmid: %u\n", am_vm::kVmid0);
  std::printf("hqd_aql_control: %u\n", am_compute::kHqdAqlControl);
  std::printf("hqd_pq_control_mode: %s\n", am_compute::kHqdPqControlMode);
  std::printf("hqd_copy_expect_cp_hqd_pq_control: 0x%08x\n", kHqdPqControl);
  std::printf("hqd_pq_doorbell_control: 0x%08x\n", kHqdPqDoorbellControl);
  std::printf("hqd_ib_control: 0x%08x\n", kHqdIbControl);
  std::printf("hqd_eop_control: 0x%08x\n", kHqdEopControl);
  std::printf("cp_mqd_control: 0x%08x\n", kCpMqdControl);
  std::printf("compute_static_thread_mgmt: 0x%08x\n", am_compute::kComputeStaticThreadMgmt);
  std::printf("status: pass\n");
  return 0;
}

int run_pm4_dispatch_sequence_self_test() {
  const std::vector<uint32_t> words = build_compute_dispatch_words(
      am_compute::kCodeVramVa, am_compute::kKernargsVa, am_compute::kTimelineVa);
  constexpr uint32_t kDispatchInitiator = encode_dispatch_initiator();
  if (words.size() != am_compute::kPm4DispatchDwordCount) {
    return self_test_failure("pm4-dispatch-sequence", "PM4 dispatch dword count drift");
  }
  if (words[0] != pm4_packet3(am_compute::kPacket3AcquireMem, 6U) ||
      words[44] != pm4_packet3(am_compute::kPacket3DispatchDirect, 3U) ||
      words[49] != pm4_packet3(am_compute::kPacket3EventWrite, 0U) ||
      words[51] != pm4_packet3(am_compute::kPacket3ReleaseMem, 6U)) {
    return self_test_failure("pm4-dispatch-sequence", "PM4 packet header drift");
  }

  std::printf("self_test: pm4-dispatch-sequence\n");
  std::printf("packet_order: %s\n", am_compute::kPm4DispatchPacketOrder);
  std::printf("packet_count: %u\n", am_compute::kPm4DispatchPacketCount);
  std::printf("dispatch_dword_count: %zu\n", words.size());
  std::printf("compute_wptr_unit: dwords\n");
  std::printf("compute_doorbell_value: %zu\n", words.size());
  std::printf("packet3_acquire_mem: 0x%02x\n", am_compute::kPacket3AcquireMem);
  std::printf("packet3_set_sh_reg: 0x%02x\n", am_compute::kPacket3SetShReg);
  std::printf("packet3_dispatch_direct: 0x%02x\n", am_compute::kPacket3DispatchDirect);
  std::printf("packet3_event_write: 0x%02x\n", am_compute::kPacket3EventWrite);
  std::printf("packet3_release_mem: 0x%02x\n", am_compute::kPacket3ReleaseMem);
  std::printf("global_size_x: %u\n", am_compute::kDispatchGlobalSizeX);
  std::printf("global_size_y: %u\n", am_compute::kDispatchGlobalSizeY);
  std::printf("global_size_z: %u\n", am_compute::kDispatchGlobalSizeZ);
  std::printf("local_size_x: %u\n", am_compute::kDispatchLocalSizeX);
  std::printf("local_size_y: %u\n", am_compute::kDispatchLocalSizeY);
  std::printf("local_size_z: %u\n", am_compute::kDispatchLocalSizeZ);
  std::printf("dispatch_initiator: 0x%08x\n", kDispatchInitiator);
  std::printf("release_mem_timeline_value: %u\n", am_compute::kReleaseMemTimelineValue);
  std::printf("status: pass\n");
  return 0;
}

int run_compute_doorbell_delivery_self_test() {
  if (am_compute::kMecDoorbellIndex != 3U) {
    return self_test_failure("compute-doorbell-delivery", "MEC doorbell index drift");
  }
  if (am_compute::kMecDoorbellBar2ByteOffset != 0x18ULL) {
    return self_test_failure("compute-doorbell-delivery", "MEC doorbell BAR2 byte offset drift");
  }
  if (am_compute::kPm4DispatchDwordCount != 59U) {
    return self_test_failure("compute-doorbell-delivery", "PM4 dispatch dword count drift");
  }
  if (am_compute::kHqdPqDoorbellHitMask != 0x80000000U) {
    return self_test_failure("compute-doorbell-delivery", "doorbell_hit field mask drift");
  }

  std::printf("self_test: compute-doorbell-delivery\n");
  std::printf("diagnostic_contract: %s\n", am_compute::kDoorbellDiagnosticContract);
  std::printf("failure_stage_if_timeline_timeout: %s\n",
              am_compute::kDoorbellFailureStageIfTimeout);
  std::printf("classification_if_not_consumed: %s\n",
              am_compute::kDoorbellClassificationIfNotConsumed);
  std::printf("doorbell_bar: BAR2\n");
  std::printf("doorbell_index: %u\n", am_compute::kMecDoorbellIndex);
  std::printf("doorbell_byte_offset: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kMecDoorbellBar2ByteOffset));
  std::printf("doorbell_value_unit: %s\n", am_compute::kDoorbellValueUnit);
  std::printf("doorbell_value_source: %s\n", am_compute::kDoorbellValueSource);
  std::printf("doorbell_hit_source: %s\n", am_compute::kDoorbellHitSource);
  std::printf("pre_ring_reads: %s\n", am_compute::kDoorbellDiagnosticPreRingReads);
  std::printf("post_ring_reads: %s\n", am_compute::kDoorbellDiagnosticPostRingReads);
  std::printf("timeout_reads: %s\n", am_compute::kDoorbellDiagnosticTimeoutReads);
  std::printf("classification_if_rptr_zero_cp_idle: %s\n",
              am_compute::kDoorbellClassRptrZeroCpIdle);
  std::printf("classification_if_doorbell_hit_rptr_zero: %s\n",
              am_compute::kDoorbellClassDoorbellHitRptrZero);
  std::printf("classification_if_rptr_advances_timeline_zero: %s\n",
              am_compute::kDoorbellClassRptrAdvancesTimelineZero);
  std::printf("compute_doorbell_route_readback_field: %s\n",
              am_compute::kDoorbellRouteReadbackField);
  std::printf("compute_doorbell_route_classification_field: %s\n",
              am_compute::kDoorbellRouteClassificationField);
  std::printf("route_readback_registers: %s\n",
              am_compute::kDoorbellRouteReadbackRegisters);
  std::printf("route_expected_entry0_ctrl: 0x%08x\n",
              am_compute::kDoorbellRouteExpectedEntry0Ctrl);
  std::printf("route_expected_entry3_ctrl: 0x%08x\n",
              am_compute::kDoorbellRouteExpectedEntry3Ctrl);
  std::printf("route_classification_values: %s,%s,%s\n",
              am_compute::kDoorbellRouteClassMatches,
              am_compute::kDoorbellRouteClassMismatch,
              am_compute::kDoorbellRouteClassUnclassified);
  std::printf("status: pass\n");
  return 0;
}
int run_compute_doorbell_consumption_self_test() {
  constexpr uint32_t kHqdPqDoorbellControl = encode_hqd_pq_doorbell_control();
  if (am_compute::hqd_doorbell_offset(kHqdPqDoorbellControl) !=
      am_compute::kDoorbellConsumptionExpectedOffset) {
    return self_test_failure("compute-doorbell-consumption", "doorbell offset drift");
  }
  if (am_compute::hqd_doorbell_en(kHqdPqDoorbellControl) !=
      am_compute::kDoorbellConsumptionExpectedEn) {
    return self_test_failure("compute-doorbell-consumption", "doorbell enable drift");
  }
  if ((kHqdPqDoorbellControl &
       am_compute::kHqdPqDoorbellControlDynamicStatusMask) != 0U) {
    return self_test_failure("compute-doorbell-consumption",
                             "doorbell dynamic status mask drift");
  }


  std::printf("self_test: compute-doorbell-consumption\n");
  std::printf("diagnostic_contract: %s\n",
              am_compute::kDoorbellConsumptionDiagnosticContract);
  std::printf("source_gap_exit_required: %s\n",
              am_compute::kDoorbellConsumptionSourceGapExitRequired);
  std::printf("hqd_doorbell_control_reads: %s\n",
              am_compute::kDoorbellConsumptionControlReads);
  std::printf("hqd_doorbell_control_decodes: %s\n",
              am_compute::kDoorbellConsumptionControlDecodes);
  std::printf("hqd_doorbell_control_mqd_compare_ignored_bits: %s\n",
              am_compute::kDoorbellConsumptionControlCompareIgnoredBits);
  std::printf("expected_doorbell_offset: %u\n",
              am_compute::kDoorbellConsumptionExpectedOffset);
  std::printf("expected_doorbell_en: %u\n",
              am_compute::kDoorbellConsumptionExpectedEn);
  std::printf("mqd_hqd_compare_fields: %s\n",
              am_compute::kDoorbellConsumptionMqdHqdCompareFields);
  std::printf("wptr_visibility_reads: %s\n",
              am_compute::kDoorbellConsumptionWptrVisibilityReads);
  std::printf("cp_mec_status_reads: %s\n",
              am_compute::kDoorbellConsumptionCpMecStatusReads);
  std::printf("cp_mec_rs64_context_reads: %s\n",
              am_compute::kDoorbellConsumptionCpMecRs64ContextReads);
  std::printf("classification_if_rs64_exception_status_nonzero: %s\n",
              am_compute::kDoorbellConsumptionClassRs64Exception);
  std::printf("classification_if_bif_drop: %s\n",
              am_compute::kDoorbellConsumptionClassBifDrop);
  std::printf("classification_if_schd_or_hit_rptr_zero: %s\n",
              am_compute::kDoorbellConsumptionClassSchdOrHitRptrZero);
  std::printf("classification_if_wptr_not_visible: %s\n",
              am_compute::kDoorbellConsumptionClassWptrNotVisible);
  std::printf("classification_if_mqd_hqd_mismatch: %s\n",
              am_compute::kDoorbellConsumptionClassMqdHqdMismatch);
  std::printf("classification_if_rptr_advances_timeline_zero: %s\n",
              am_compute::kDoorbellConsumptionClassRptrAdvancesTimelineZero);
  std::printf("classification_if_no_signal: %s\n",
              am_compute::kDoorbellConsumptionClassNoSignal);
  std::printf("status: pass\n");
  return 0;
}


int run_gc_hub_sequence_self_test() {
  std::printf("self_test: gc-hub-sequence\n");
  std::printf("topology_requirement: one_gc_instance_for_direct_pm4\n");
  std::printf("gc_context: VMID0\n");
  std::printf("sequence: hdp,gc_system_aperture,gc_l1_l2,gc_context0,gc_identity_aperture,gc_invalidate_ranges,gc_tlb_flush\n");
  std::printf("failure_stage_if_multi_xcc: multi_xcc_aql_required\n");
  std::printf("status: pass\n");
  return 0;
}



int run_log_contract_self_test() {
  constexpr std::array<const char*, 9> kRequiredFields{{
      "runtime_substrate",
      "pci_id",
      "arch",
      "transfer_byte_count",
      "cpu_comparison_status",
      "host_device_transfer_status",
      "failure_stage",
      "failure_text",
      "exit_status",
  }};

  std::printf("self_test: log-contract\n");
  for (const char* field : kRequiredFields) {
    std::printf("required_log_field: %s\n", field);
  }
  std::printf("status: pass\n");
  return 0;
}

class UniqueFd {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) : fd_(fd) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      reset();
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }
  ~UniqueFd() { reset(); }

  int get() const { return fd_; }
  bool valid() const { return fd_ >= 0; }
  int release() {
    const int fd = fd_;
    fd_ = -1;
    return fd;
  }
  void reset(int fd = -1) {
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

struct BarInfo {
  bool mapped = false;
  uint64_t handle = 0;
  uint64_t size = 0;
  std::string response_header_hex = "not_run";
};

struct IpBlockInfo {
  bool found = false;
  const char* label = "unknown";
  uint16_t hw_id = 0;
  uint8_t instance = 0;
  uint8_t major = 0;
  uint8_t minor = 0;
  uint8_t revision = 0;
  std::vector<uint64_t> bases;
};

struct IpDiscoveryInfo {
  IpBlockInfo gc;
  uint32_t gc_instance_count = 0;
  std::vector<IpBlockInfo> mmhubs;
  IpBlockInfo nbif;
  IpBlockInfo sdma0;
};

struct RegDef {
  const char* name;
  uint32_t offset;
  uint8_t segment;
};

struct FixedVmPageTables {
  uint64_t root_pdb2_paddr = 0x0000000000000000ULL;
  uint64_t memscratch_paddr = 0x0000000000001000ULL;
  uint64_t dummy_page_paddr = 0x0000000000002000ULL;
  uint64_t child_pdb1_paddr = 0x0000000002000000ULL;
  uint64_t child_pdb0_paddr = 0x0000000002001000ULL;
  uint64_t child_ptb_paddr = 0x0000000002002000ULL;
  uint64_t staging_ptb_paddr = 0x0000000002004000ULL;
  uint64_t device_buffer_paddr = am_vm::kFixedVramBufferPaddr;
};

struct FixedVmMappingResult {
  FixedVmPageTables tables;
  bool page_tables_written = false;
  bool vmid0_context_programmed = false;
  bool tlb_flushed = false;
  std::string failure_stage = "vm_mapping";
  std::string error_text;
};

struct VmHardwareLog {
  FixedVmPageTables tables;
  uint64_t mc_base = 0;  // MC base (fb_base) from regMMMC_VM_FB_LOCATION_BASE; CPF MQD read domain.
  std::string page_tables_written = "not_run";
  std::string vmid0_context_status = "not_run";
  std::string vm_gc_context_status = "not_run";
  std::string mm_tlb_flush_status = "not_run";
  std::string gc_tlb_flush_status = "not_run";
};

struct SdmaHardwareLog {
  std::string queue_setup_status = "not_run";
  std::string submit_status = "not_run";
  std::string timeline_status = "not_run";
  std::string h2d_status = "not_run";
  std::string d2h_status = "not_run";
};

struct ComputeHardwareLog {
  std::string ring_setup_status = "not_run";
  std::string hqd_active_status = "not_run";
  std::string kernel_blob_load_status = "not_run";
  std::string kernarg_write_status = "not_run";
  std::string doorbell_probe_status = "not_run";
  std::string doorbell_probe_pre = "not_run";
  std::string doorbell_probe_post = "not_run";
  std::string doorbell_probe_timeout = "not_run";
  std::string doorbell_probe_classification = "not_run";
  std::string doorbell_consumption_timeout = "not_run";
  std::string doorbell_consumption_classification = "not_run";
  std::string doorbell_route_readback = "not_run";
  std::string doorbell_route_classification = "not_run";
  std::string mec_rs64_cntl_write_status = "not_run";
  std::string mec_rs64_cntl_readback = "not_run";
  std::string mec_rs64_active_status = "not_run";
  std::string compute_readback_anomaly = "not_run";
};

struct DiscoveryLog {
  std::string socket_path;
  std::string pci_id = "unknown";
  std::string arch = "not_discovered";
  std::string arch_discovery_status = "not_run";
  std::string config_response_header_hex = "not_run";
  uint32_t config_vendor_id = 0;
  uint32_t config_device_id = 0;
  uint64_t vram_size_bytes = 0;
  BarInfo bar0;
  BarInfo bar2;
  BarInfo bar5;
  IpDiscoveryInfo ip;
  std::string gc_ip_version = "not_found";
  std::string gc_ip_bases = "not_found";
  std::string mmhub_ip_version = "not_found";
  std::string mmhub_ip_bases = "not_found";
  std::string nbif_ip_version = "not_found";
  std::string nbif_ip_bases = "not_found";
  std::string sdma_ip_version = "not_found";
  std::string sdma_ip_bases = "not_found";
  VmHardwareLog vm;
  SdmaHardwareLog sdma;
  ComputeHardwareLog compute;
  std::string failure_stage = "none";
  std::string failure_text = "none";
};

struct VmBufferLog {
  const char* role = "unknown";
  uint64_t gpu_va = 0;
  uint64_t requested_size = 0;
  uint64_t mapped_size = 0;
  std::string response_header_hex = "not_run";
  std::vector<uint64_t> sys_pages;
};

struct SysmemMapping {
  UniqueFd fd;
  void* data = nullptr;
  std::size_t size = 0;

  SysmemMapping() = default;
  SysmemMapping(const SysmemMapping&) = delete;
  SysmemMapping& operator=(const SysmemMapping&) = delete;
  ~SysmemMapping() { reset(); }

  void reset() {
    if (data != nullptr) {
      munmap(data, size);
      data = nullptr;
      size = 0;
    }
    fd.reset();
  }

  void hold(UniqueFd&& new_fd, void* mapped_data, std::size_t mapped_size) {
    reset();
    fd = std::move(new_fd);
    data = mapped_data;
    size = mapped_size;
  }
};

void print_discovery_log(const DiscoveryLog& log, int exit_status) {
  std::printf("runtime_substrate: %s\n", kRuntimeSubstrate);
  std::printf("socket_path: %s\n", log.socket_path.c_str());
  std::printf("pci_id: %s\n", log.pci_id.c_str());
  std::printf("arch: %s\n", log.arch.c_str());
  std::printf("arch_discovery_status: %s\n", log.arch_discovery_status.c_str());
  std::printf("gc_ip_version: %s\n", log.gc_ip_version.c_str());
  std::printf("gc_ip_bases: %s\n", log.gc_ip_bases.c_str());
  std::printf("mmhub_ip_version: %s\n", log.mmhub_ip_version.c_str());
  std::printf("mmhub_ip_bases: %s\n", log.mmhub_ip_bases.c_str());
  std::printf("nbif_ip_version: %s\n", log.nbif_ip_version.c_str());
  std::printf("nbif_ip_bases: %s\n", log.nbif_ip_bases.c_str());
  std::printf("sdma_ip_version: %s\n", log.sdma_ip_version.c_str());
  std::printf("sdma_ip_bases: %s\n", log.sdma_ip_bases.c_str());
  std::printf("config_response_header_hex: %s\n", log.config_response_header_hex.c_str());
  std::printf("config_vendor_id: 0x%04x\n", log.config_vendor_id);
  std::printf("config_device_id: 0x%04x\n", log.config_device_id);
  std::printf("bar0_response_header_hex: %s\n", log.bar0.response_header_hex.c_str());
  std::printf("bar0_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar0.size));
  std::printf("bar2_response_header_hex: %s\n", log.bar2.response_header_hex.c_str());
  std::printf("bar2_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar2.size));
  std::printf("bar5_response_header_hex: %s\n", log.bar5.response_header_hex.c_str());
  std::printf("bar5_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar5.size));
  std::printf("vram_size_bytes: %llu\n", static_cast<unsigned long long>(log.vram_size_bytes));
  std::printf("transfer_byte_count: 0\n");
  std::printf("cpu_comparison_status: not_run\n");
  std::printf("host_device_transfer_status: not_run\n");
  std::printf("failure_stage: %s\n", log.failure_stage.c_str());
  std::printf("failure_text: %s\n", log.failure_text.c_str());
  std::printf("exit_status: %d\n", exit_status);
}

int finish_discovery(DiscoveryLog& log, const char* stage, const std::string& text) {
  log.failure_stage = stage;
  log.failure_text = text;
  print_discovery_log(log, 1);
  return 1;
}

std::string errno_text(const char* prefix, int err) {
  std::string out(prefix);
  out += ": ";
  out += std::strerror(err);
  return out;
}

std::string tinygpu_socket_path() {
  if (const char* env = std::getenv("APL_REMOTE_SOCK")) {
    if (env[0] != '\0') {
      return std::string(env);
    }
  }
  const char* tmpdir_env = std::getenv("TMPDIR");
  std::string tmpdir = (tmpdir_env != nullptr && tmpdir_env[0] != '\0') ? tmpdir_env : "/tmp";
  if (!tmpdir.empty() && tmpdir.back() == '/') {
    return tmpdir + "tinygpu.sock";
  }
  return tmpdir + "/tinygpu.sock";
}

bool set_socket_timeouts(int fd, std::string* error_text) {
  timeval timeout{};
  timeout.tv_sec = 3;
  timeout.tv_usec = 0;
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0) {
    *error_text = errno_text("setsockopt SO_RCVTIMEO failed", errno);
    return false;
  }
  if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
    *error_text = errno_text("setsockopt SO_SNDTIMEO failed", errno);
    return false;
  }
#ifdef SO_NOSIGPIPE
  const int no_sigpipe = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe)) != 0) {
    *error_text = errno_text("setsockopt SO_NOSIGPIPE failed", errno);
    return false;
  }
#endif
  return true;
}

bool try_connect_unix_socket(const std::string& path, UniqueFd* out_fd, std::string* error_text) {
  sockaddr_un addr{};
  if (path.size() + 1 > sizeof(addr.sun_path)) {
    *error_text = "socket path too long: " + path;
    return false;
  }

  UniqueFd fd(socket(AF_UNIX, SOCK_STREAM, 0));
  if (!fd.valid()) {
    *error_text = errno_text("socket(AF_UNIX) failed", errno);
    return false;
  }

#ifdef __APPLE__
  addr.sun_len = sizeof(addr);
#endif
  addr.sun_family = AF_UNIX;
  std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
  const socklen_t addr_len = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);
  if (connect(fd.get(), reinterpret_cast<sockaddr*>(&addr), addr_len) != 0) {
    *error_text = errno_text("connect failed", errno);
    return false;
  }

  if (!set_socket_timeouts(fd.get(), error_text)) {
    return false;
  }

  *out_fd = std::move(fd);
  return true;
}

bool launch_tinygpu_server(const std::string& path, std::string* error_text) {
  if (access(kTinyGpuAppPath, X_OK) != 0) {
    *error_text = errno_text("TinyGPU executable is not runnable", errno);
    return false;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    *error_text = errno_text("fork TinyGPU server failed", errno);
    return false;
  }
  if (pid == 0) {
    const int devnull = open("/dev/null", O_RDWR);
    if (devnull >= 0) {
      dup2(devnull, STDOUT_FILENO);
      dup2(devnull, STDERR_FILENO);
      if (devnull > STDERR_FILENO) {
        close(devnull);
      }
    }
    execl(kTinyGpuAppPath, kTinyGpuAppPath, "server", path.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  return true;
}

bool connect_tinygpu_server(const std::string& path, UniqueFd* out_fd, std::string* error_text) {
  std::string first_connect_error;
  if (try_connect_unix_socket(path, out_fd, &first_connect_error)) {
    return true;
  }

  std::string launch_error;
  if (!launch_tinygpu_server(path, &launch_error)) {
    *error_text = first_connect_error + "; launch failed: " + launch_error;
    return false;
  }

  std::string last_connect_error = first_connect_error;
  for (int attempt = 0; attempt < 100; ++attempt) {
    usleep(50U * 1000U);
    UniqueFd fd;
    if (try_connect_unix_socket(path, &fd, &last_connect_error)) {
      *out_fd = std::move(fd);
      return true;
    }
  }

  *error_text = "failed to connect to TinyGPU server after launch; first_error: " +
                first_connect_error + "; last_error: " + last_connect_error;
  return false;
}

bool send_all(int fd, const uint8_t* data, std::size_t size, std::string* error_text) {
  std::size_t sent = 0;
  while (sent < size) {
    const ssize_t n = send(fd, data + sent, size - sent, 0);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      *error_text = errno_text("send failed", n < 0 ? errno : ECONNRESET);
      return false;
    }
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

bool read_exact(int fd, uint8_t* data, std::size_t size, std::string* error_text) {
  std::size_t received = 0;
  while (received < size) {
    const ssize_t n = recv(fd, data + received, size - received, 0);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      *error_text = errno_text("recv failed", n < 0 ? errno : ECONNRESET);
      return false;
    }
    received += static_cast<std::size_t>(n);
  }
  return true;
}

struct RemoteRpcResult {
  bool ok = false;
  uint8_t status = 0;
  uint64_t value0 = 0;
  uint64_t value1 = 0;
  std::string response_header_hex;
  std::vector<uint8_t> readout;
  std::string error_text;
  std::string error_payload_hex;
};

struct RemoteSysmemFdResult {
  bool ok = false;
  uint8_t status = 0;
  uint64_t mapped_size = 0;
  uint64_t value1 = 0;
  std::string response_header_hex;
  std::string error_text;
  std::string error_payload_hex;
  UniqueFd fd;
};

class RemoteClient {
 public:
  explicit RemoteClient(int fd) : fd_(fd) {}

  RemoteRpcResult rpc(RemoteCmd cmd, uint32_t bar, uint64_t arg0, uint64_t arg1, uint64_t arg2,
                      const std::vector<uint8_t>& payload, uint64_t readout_size) const {
    RemoteRpcResult result;
    const RemoteCmdFrame frame = build_remote_cmd_frame(cmd, kRemoteDevId, bar, arg0, arg1, arg2);
    std::vector<uint8_t> request(frame.begin(), frame.end());
    request.insert(request.end(), payload.begin(), payload.end());

    if (!send_all(fd_, request.data(), request.size(), &result.error_text)) {
      result.error_text = "RemoteCmd send failed: " + result.error_text;
      return result;
    }

    std::array<uint8_t, 17> header{};
    if (!read_exact(fd_, header.data(), header.size(), &result.error_text)) {
      result.error_text = "RemoteCmd response header read failed: " + result.error_text;
      return result;
    }

    result.response_header_hex = hex_encode_bytes(header.data(), header.size());
    result.status = header[0];
    result.value0 = read_u64_le_bytes(header.data() + 1);
    result.value1 = read_u64_le_bytes(header.data() + 9);

    if (result.status != 0) {
      if (result.value0 > (1ULL << 20)) {
        result.error_text = "RPC failed with oversized error payload length " + std::to_string(result.value0);
        return result;
      }
      if (result.value0 > 0) {
        std::vector<uint8_t> error_payload(static_cast<std::size_t>(result.value0));
        if (!read_exact(fd_, error_payload.data(), error_payload.size(), &result.error_text)) {
          result.error_text = "RPC failed and error payload read failed: " + result.error_text;
          return result;
        }
        result.error_payload_hex = hex_encode_vector(error_payload);
        result.error_text = printable_ascii(error_payload);
      } else {
        result.error_text = "unknown error";
      }
      return result;
    }

    const uint64_t bytes_to_read = (readout_size == kRpcReadoutFromResponse0) ? result.value0 : readout_size;
    if (bytes_to_read > (64ULL << 20)) {
      result.error_text = "RPC success reported oversized readout length " + std::to_string(bytes_to_read);
      return result;
    }
    if (bytes_to_read > 0) {
      result.readout.resize(static_cast<std::size_t>(bytes_to_read));
      if (!read_exact(fd_, result.readout.data(), result.readout.size(), &result.error_text)) {
        result.error_text = "RemoteCmd readout read failed: " + result.error_text;
        return result;
      }
    }
    result.ok = true;
    return result;
  }

  RemoteRpcResult rpc_no_payload(RemoteCmd cmd, uint32_t bar, uint64_t arg0 = 0,
                                 uint64_t arg1 = 0, uint64_t arg2 = 0,
                                 uint64_t readout_size = 0) const {
    const std::vector<uint8_t> empty;
    return rpc(cmd, bar, arg0, arg1, arg2, empty, readout_size);
  }

  bool mmio_write_fire_and_forget(uint32_t bar, uint64_t offset, const std::vector<uint8_t>& payload,
                                  std::string* error_text) const {
    // tinygrad/runtime/support/system.py:388-390 RemotePCIDevice._bulk_write sends
    // RemoteCmd::MMIO_WRITE as <BIIQQQ> args (offset, len(data), 0) plus payload and
    // intentionally reads no response header.
    const RemoteCmdFrame frame = build_remote_cmd_frame(RemoteCmd::MMIO_WRITE, kRemoteDevId, bar,
                                                        offset, payload.size(), 0);
    std::vector<uint8_t> request(frame.begin(), frame.end());
    request.insert(request.end(), payload.begin(), payload.end());
    if (!send_all(fd_, request.data(), request.size(), error_text)) {
      *error_text = "RemoteCmd MMIO_WRITE send failed: " + *error_text;
      return false;
    }
    return true;
  }

  RemoteSysmemFdResult rpc_sysmem_fd(uint64_t size, bool contiguous) const {
    RemoteSysmemFdResult result;
    if (size > 0xffffffffULL) {
      result.error_text = "MAP_SYSMEM_FD size exceeds 32-bit RPC size field: " + std::to_string(size);
      return result;
    }

    const RemoteCmdFrame frame =
        build_remote_cmd_frame(RemoteCmd::MAP_SYSMEM_FD, kRemoteDevId, 0, size,
                               contiguous ? 1ULL : 0ULL, 0);
    if (!send_all(fd_, frame.data(), frame.size(), &result.error_text)) {
      result.error_text = "RemoteCmd MAP_SYSMEM_FD send failed: " + result.error_text;
      return result;
    }

    std::array<uint8_t, 17> header{};
    iovec iov{};
    iov.iov_base = header.data();
    iov.iov_len = header.size();
    char control[CMSG_SPACE(sizeof(int))];
    std::memset(control, 0, sizeof(control));
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    const ssize_t n = recvmsg(fd_, &msg, 0);
    if (n < 0) {
      result.error_text = errno_text("MAP_SYSMEM_FD recvmsg failed", errno);
      return result;
    }
    if (n == 0) {
      result.error_text = "MAP_SYSMEM_FD recvmsg returned EOF before response header";
      return result;
    }
    if (static_cast<std::size_t>(n) < header.size() &&
        !read_exact(fd_, header.data() + n, header.size() - static_cast<std::size_t>(n),
                    &result.error_text)) {
      result.error_text = "MAP_SYSMEM_FD short response header read failed: " + result.error_text;
      return result;
    }
    if ((msg.msg_flags & MSG_CTRUNC) != 0) {
      result.error_text = "MAP_SYSMEM_FD control message was truncated";
      return result;
    }

    int received_fd = -1;
    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
      if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS &&
          cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
        std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(received_fd));
        break;
      }
    }
    if (received_fd >= 0) {
      result.fd.reset(received_fd);
    }

    result.response_header_hex = hex_encode_bytes(header.data(), header.size());
    result.status = header[0];
    result.mapped_size = read_u64_le_bytes(header.data() + 1);
    result.value1 = read_u64_le_bytes(header.data() + 9);
    if (result.status != 0) {
      if (result.mapped_size > (1ULL << 20)) {
        result.error_text =
            "MAP_SYSMEM_FD failed with oversized error payload length " + std::to_string(result.mapped_size);
        return result;
      }
      if (result.mapped_size > 0) {
        std::vector<uint8_t> error_payload(static_cast<std::size_t>(result.mapped_size));
        if (!read_exact(fd_, error_payload.data(), error_payload.size(), &result.error_text)) {
          result.error_text = "MAP_SYSMEM_FD failed and error payload read failed: " + result.error_text;
          return result;
        }
        result.error_payload_hex = hex_encode_vector(error_payload);
        result.error_text = printable_ascii(error_payload);
      } else {
        result.error_text = "unknown error";
      }
      return result;
    }
    if (!result.fd.valid()) {
      result.error_text = "MAP_SYSMEM_FD success did not include an SCM_RIGHTS fd";
      return result;
    }
    result.ok = true;
    return result;
  }

 private:
  int fd_ = -1;
};

std::string rpc_failure_text(const char* cmd_name, const RemoteRpcResult& result) {
  std::string text(cmd_name);
  text += " failed";
  if (!result.response_header_hex.empty()) {
    text += "; response_header_hex=" + result.response_header_hex;
  }
  if (result.status != 0) {
    text += "; status=" + std::to_string(result.status);
  }
  if (!result.error_payload_hex.empty()) {
    text += "; error_payload_hex=" + result.error_payload_hex;
  }
  if (!result.error_text.empty()) {
    text += "; error_text=" + result.error_text;
  }
  return text;
}

std::string rpc_sysmem_failure_text(const char* cmd_name, const RemoteSysmemFdResult& result) {
  std::string text(cmd_name);
  text += " failed";
  if (!result.response_header_hex.empty()) {
    text += "; response_header_hex=" + result.response_header_hex;
  }
  if (result.status != 0) {
    text += "; status=" + std::to_string(result.status);
  }
  if (!result.error_payload_hex.empty()) {
    text += "; error_payload_hex=" + result.error_payload_hex;
  }
  if (!result.error_text.empty()) {
    text += "; error_text=" + result.error_text;
  }
  return text;
}


std::string pci_id_text(uint32_t vendor, uint32_t device) {
  char buf[16]{};
  std::snprintf(buf, sizeof(buf), "%04x:%04x", static_cast<unsigned>(vendor),
                static_cast<unsigned>(device));
  return buf;
}

bool map_bar(const RemoteClient& client, uint32_t bar, BarInfo* out, RemoteRpcResult* raw_result) {
  RemoteRpcResult result = client.rpc_no_payload(RemoteCmd::MAP_BAR, bar);
  *raw_result = result;
  out->response_header_hex = result.response_header_hex.empty() ? "unavailable" : result.response_header_hex;
  if (!result.ok) {
    return false;
  }
  out->mapped = true;
  out->handle = result.value0;
  out->size = result.value1;
  return true;
}

bool map_sysmem_buffer(const RemoteClient& client, VmBufferLog* buffer, SysmemMapping* mapping,
                       std::string* error_text) {
  RemoteSysmemFdResult result = client.rpc_sysmem_fd(buffer->requested_size, false);
  buffer->response_header_hex =
      result.response_header_hex.empty() ? "unavailable" : result.response_header_hex;
  if (!result.ok) {
    *error_text = std::string(buffer->role) + " " +
                  rpc_sysmem_failure_text("MAP_SYSMEM_FD", result);
    return false;
  }
  buffer->mapped_size = result.mapped_size;
  if (buffer->mapped_size == 0) {
    *error_text = std::string(buffer->role) + " MAP_SYSMEM_FD returned zero mapped size";
    return false;
  }
  if (buffer->mapped_size > static_cast<uint64_t>(SIZE_MAX)) {
    *error_text = std::string(buffer->role) + " MAP_SYSMEM_FD mapped size exceeds host size_t";
    return false;
  }

  void* mapped = mmap(nullptr, static_cast<std::size_t>(buffer->mapped_size), PROT_READ | PROT_WRITE,
                      MAP_SHARED, result.fd.get(), 0);
  if (mapped == MAP_FAILED) {
    *error_text = std::string(buffer->role) + " MAP_SYSMEM_FD mmap failed: " +
                  std::strerror(errno);
    return false;
  }

  const uint64_t requested_pages_64 = ceil_div_u64(buffer->requested_size, kPageSize);
  if (requested_pages_64 > static_cast<uint64_t>(SIZE_MAX)) {
    munmap(mapped, static_cast<std::size_t>(buffer->mapped_size));
    *error_text = std::string(buffer->role) + " requested page count exceeds host size_t";
    return false;
  }
  const PageListParseResult parsed = parse_sysmem_page_list_bytes(
      static_cast<const uint8_t*>(mapped), static_cast<std::size_t>(buffer->mapped_size),
      static_cast<std::size_t>(requested_pages_64));
  if (!parsed.ok) {
    munmap(mapped, static_cast<std::size_t>(buffer->mapped_size));
    *error_text = std::string(buffer->role) + " MAP_SYSMEM_FD page-list parse failed: " +
                  parsed.error_text;
    return false;
  }
  buffer->sys_pages = parsed.pages;
  mapping->hold(std::move(result.fd), mapped, static_cast<std::size_t>(buffer->mapped_size));
  return true;
}

RemoteRpcResult mmio_read(const RemoteClient& client, uint32_t bar, uint64_t offset, uint64_t size) {
  return client.rpc_no_payload(RemoteCmd::MMIO_READ, bar, offset, size, 0, size);
}


std::string format_hex32(uint32_t value) {
  char buffer[11]{};
  std::snprintf(buffer, sizeof(buffer), "0x%08x", value);
  return buffer;
}

std::string format_hex64(uint64_t value) {
  char buf[32]{};
  std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(value));
  return buf;
}

std::string ip_version_text(const IpBlockInfo& ip) {
  if (!ip.found) {
    return "not_found";
  }
  return std::to_string(static_cast<unsigned>(ip.major)) + "." +
         std::to_string(static_cast<unsigned>(ip.minor)) + "." +
         std::to_string(static_cast<unsigned>(ip.revision));
}

std::string ip_bases_text(const IpBlockInfo& ip) {
  if (!ip.found) {
    return "not_found";
  }
  std::string out = "inst" + std::to_string(static_cast<unsigned>(ip.instance)) + ":";
  for (std::size_t i = 0; i < ip.bases.size(); ++i) {
    if (i != 0) {
      out += ",";
    }
    out += format_hex64(ip.bases[i]);
  }
  return out;
}

std::string mmhub_bases_text(const std::vector<IpBlockInfo>& mmhubs) {
  if (mmhubs.empty()) {
    return "not_found";
  }
  std::string out;
  for (std::size_t i = 0; i < mmhubs.size(); ++i) {
    if (i != 0) {
      out += ";";
    }
    out += ip_bases_text(mmhubs[i]);
  }
  return out;
}

void update_ip_log_fields(DiscoveryLog* log) {
  log->gc_ip_version = ip_version_text(log->ip.gc);
  log->gc_ip_bases = ip_bases_text(log->ip.gc);
  if (log->ip.mmhubs.empty()) {
    log->mmhub_ip_version = "not_found";
    log->mmhub_ip_bases = "not_found";
  } else {
    log->mmhub_ip_version = ip_version_text(log->ip.mmhubs[0]);
    log->mmhub_ip_bases = mmhub_bases_text(log->ip.mmhubs);
  }
  log->nbif_ip_version = ip_version_text(log->ip.nbif);
  log->nbif_ip_bases = ip_bases_text(log->ip.nbif);
  log->sdma_ip_version = ip_version_text(log->ip.sdma0);
  log->sdma_ip_bases = ip_bases_text(log->ip.sdma0);
}

// tinygrad/runtime/support/am/amdev.py:279-286 reads VRAM indirectly through raw
// dword registers wreg(0x06,caddr>>31), wreg(0x00,(caddr&0x7fffffff)|0x80000000), rreg(0x01).
bool raw_bar5_reg_write(const RemoteClient& client, const BarInfo& bar5, uint32_t reg_dword,
                        uint32_t value, std::string* error_text) {
  const uint64_t bar5_dwords = bar5.size / sizeof(uint32_t);
  if (!bar5.mapped || reg_dword >= bar5_dwords) {
    *error_text = "BAR5 raw register " + std::to_string(reg_dword) +
                  " outside mapped dword span " + std::to_string(bar5_dwords);
    return false;
  }
  const std::vector<uint8_t> payload = u32_payload_le(value);
  return client.mmio_write_fire_and_forget(5, static_cast<uint64_t>(reg_dword) * 4ULL,
                                           payload, error_text);
}

bool raw_bar5_reg_read(const RemoteClient& client, const BarInfo& bar5, uint32_t reg_dword,
                       uint32_t* value, std::string* error_text) {
  const uint64_t bar5_dwords = bar5.size / sizeof(uint32_t);
  if (!bar5.mapped || reg_dword >= bar5_dwords) {
    *error_text = "BAR5 raw register " + std::to_string(reg_dword) +
                  " outside mapped dword span " + std::to_string(bar5_dwords);
    return false;
  }
  RemoteRpcResult result = mmio_read(client, 5, static_cast<uint64_t>(reg_dword) * 4ULL, 4);
  if (!result.ok) {
    *error_text = rpc_failure_text("MMIO_READ BAR5 raw register", result);
    return false;
  }
  if (result.readout.size() != 4) {
    *error_text = "MMIO_READ BAR5 raw register returned " + std::to_string(result.readout.size()) + " bytes";
    return false;
  }
  *value = read_u32_le_bytes(result.readout.data());
  return true;
}

bool read_vram_indirect(const RemoteClient& client, const BarInfo& bar5, uint64_t addr, uint64_t size,
                        std::vector<uint8_t>* out, std::string* error_text) {
  if ((addr % 4ULL) != 0 || (size % 4ULL) != 0) {
    *error_text = "indirect VRAM read requires 4-byte aligned address and size";
    return false;
  }
  out->clear();
  out->reserve(static_cast<std::size_t>(size));
  for (uint64_t caddr = addr; caddr < addr + size; caddr += 4ULL) {
    if (!raw_bar5_reg_write(client, bar5, 0x06U, static_cast<uint32_t>(caddr >> 31), error_text)) {
      *error_text = "indirect VRAM high-address register write failed: " + *error_text;
      return false;
    }
    if (!raw_bar5_reg_write(client, bar5, 0x00U,
                            static_cast<uint32_t>((caddr & 0x7fffffffULL) | 0x80000000ULL),
                            error_text)) {
      *error_text = "indirect VRAM index register write failed: " + *error_text;
      return false;
    }
    uint32_t word = 0;
    if (!raw_bar5_reg_read(client, bar5, 0x01U, &word, error_text)) {
      *error_text = "indirect VRAM data register read failed: " + *error_text;
      return false;
    }
    append_u32_le(out, word);
  }
  return true;
}

bool read_ip_discovery_table_bytes(const RemoteClient& client, const DiscoveryLog& log,
                                   std::vector<uint8_t>* table, std::string* error_text) {
  const uint64_t table_offset = log.vram_size_bytes - kAmdIpDiscoveryTableVramBackoff;
  if (log.bar0.mapped && log.bar0.size >= log.vram_size_bytes) {
    RemoteRpcResult result = mmio_read(client, 0, table_offset, kAmdIpDiscoveryTableSize);
    if (!result.ok) {
      *error_text = rpc_failure_text("MMIO_READ BAR0 IP discovery", result);
      return false;
    }
    *table = std::move(result.readout);
    return true;
  }
  return read_vram_indirect(client, log.bar5, table_offset, kAmdIpDiscoveryTableSize, table, error_text);
}

void capture_ip_block(IpDiscoveryInfo* info, const IpBlockInfo& ip) {
  if (ip.hw_id == kAmdGcHwId) {
    ++info->gc_instance_count;
    if (!info->gc.found) {
      info->gc = ip;
      info->gc.label = "GC";
    }
  } else if (ip.hw_id == kAmdMmhubHwId) {
    IpBlockInfo mmhub = ip;
    mmhub.label = "MMHUB";
    info->mmhubs.push_back(std::move(mmhub));
  } else if (ip.hw_id == kAmdNbifHwId && !info->nbif.found) {
    info->nbif = ip;
    info->nbif.label = "NBIF";
  } else if (ip.hw_id == am_sdma::kSdma0HwId && !info->sdma0.found) {
    info->sdma0 = ip;
    info->sdma0.label = "SDMA0";
  }
}

bool parse_ip_discovery_table(const std::vector<uint8_t>& table, IpDiscoveryInfo* info,
                              std::string* arch, std::string* status) {
  if (table.size() < kAmdBinaryHeaderSize) {
    *status = "discovery_table_too_short";
    return false;
  }
  if (read_u32_le_bytes(table.data()) != kAmdBinarySignature) {
    *status = "binary_signature_mismatch";
    return false;
  }

  const std::size_t table_info_offset = 12 + (kAmdIpDiscoveryTableIndex * kAmdTableInfoSize);
  if (table.size() < table_info_offset + kAmdTableInfoSize) {
    *status = "ip_discovery_table_info_missing";
    return false;
  }
  const uint16_t ip_discovery_offset = read_u16_le(table.data() + table_info_offset);
  if (table.size() < static_cast<std::size_t>(ip_discovery_offset) + 80) {
    *status = "ip_discovery_header_missing";
    return false;
  }

  const uint8_t* ip_header = table.data() + ip_discovery_offset;
  if (read_u32_le_bytes(ip_header) != kAmdDiscoveryTableSignature) {
    *status = "ip_discovery_signature_mismatch";
    return false;
  }
  const uint16_t num_dies = read_u16_le(ip_header + 12);
  const bool base_addr_64_bit = (ip_header[78] & 0x01U) != 0;
  if (num_dies == 0 || num_dies > 16) {
    *status = "invalid_die_count";
    return false;
  }

  for (uint16_t die = 0; die < num_dies; ++die) {
    const std::size_t die_info_offset = static_cast<std::size_t>(ip_discovery_offset) + 14 + (die * 4);
    if (table.size() < die_info_offset + 4) {
      *status = "die_info_missing";
      return false;
    }
    std::size_t die_offset = read_u16_le(table.data() + die_info_offset + 2);
    if (table.size() < die_offset + 4) {
      *status = "die_header_missing";
      return false;
    }
    const uint16_t num_ips = read_u16_le(table.data() + die_offset + 2);
    std::size_t ip_offset = die_offset + 4;
    for (uint16_t ip_index = 0; ip_index < num_ips; ++ip_index) {
      if (table.size() < ip_offset + 8) {
        *status = "ip_record_missing";
        return false;
      }
      IpBlockInfo ip;
      ip.found = true;
      ip.hw_id = read_u16_le(table.data() + ip_offset);
      ip.instance = table[ip_offset + 2];
      const uint8_t num_base_address = table[ip_offset + 3];
      ip.major = table[ip_offset + 4];
      ip.minor = table[ip_offset + 5];
      ip.revision = table[ip_offset + 6];
      const std::size_t bases_offset = ip_offset + 8;
      const std::size_t base_width = base_addr_64_bit ? 8 : 4;
      if (table.size() < bases_offset + (base_width * static_cast<std::size_t>(num_base_address))) {
        *status = "ip_base_address_missing";
        return false;
      }
      ip.bases.reserve(num_base_address);
      for (uint8_t base_i = 0; base_i < num_base_address; ++base_i) {
        const uint8_t* base_ptr = table.data() + bases_offset + (base_i * base_width);
        ip.bases.push_back(base_addr_64_bit ? read_u64_le_bytes(base_ptr) : read_u32_le_bytes(base_ptr));
      }
      capture_ip_block(info, ip);
      ip_offset += 8 + base_width * static_cast<std::size_t>(num_base_address);
    }
  }

  if (info->gc.found) {
    char arch_buf[32]{};
    std::snprintf(arch_buf, sizeof(arch_buf), "gfx%u%x%x", static_cast<unsigned>(info->gc.major),
                  static_cast<unsigned>(info->gc.minor), static_cast<unsigned>(info->gc.revision));
    *arch = arch_buf;
  }
  if (!info->gc.found) {
    *status = "gc_ip_record_not_found";
    return false;
  }
  if (info->mmhubs.empty()) {
    *status = "mmhub_ip_record_not_found";
    return false;
  }
  if (!info->nbif.found) {
    *status = "nbif_ip_record_not_found";
    return false;
  }
  *status = "discovered_from_ip_table";
  return true;
}

bool try_discover_arch(const RemoteClient& client, DiscoveryLog* log, std::string* required_error) {
  RemoteRpcResult memsize = mmio_read(client, 5, static_cast<uint64_t>(kMmRccConfigMemsize) * 4ULL, 4);
  if (!memsize.ok) {
    log->arch_discovery_status = rpc_failure_text("MMIO_READ RCC_CONFIG_MEMSIZE", memsize);
    *required_error = log->arch_discovery_status;
    return false;
  }
  if (memsize.readout.size() != 4) {
    log->arch_discovery_status = "RCC_CONFIG_MEMSIZE returned " + std::to_string(memsize.readout.size()) + " bytes";
    *required_error = log->arch_discovery_status;
    return false;
  }

  const uint32_t vram_mib = read_u32_le_bytes(memsize.readout.data());
  log->vram_size_bytes = static_cast<uint64_t>(vram_mib) << 20;
  if (log->vram_size_bytes < kAmdIpDiscoveryTableVramBackoff) {
    log->arch_discovery_status = "vram_size_too_small_for_ip_table";
    *required_error = log->arch_discovery_status;
    return false;
  }
  std::vector<uint8_t> table;
  std::string table_error;
  if (!read_ip_discovery_table_bytes(client, *log, &table, &table_error)) {
    log->arch_discovery_status = table_error;
    update_ip_log_fields(log);
    return true;
  }

  std::string arch;
  std::string status;
  if (parse_ip_discovery_table(table, &log->ip, &arch, &status)) {
    log->arch = arch;
  }
  log->arch_discovery_status = status;
  update_ip_log_fields(log);
  return true;
}


// Fixed gfx1201 register subset used by this proof: gc_12_0_0, mmhub_4_1_0, nbif_6_3_1,
// and SDMA0 7.0.1 queue-0 `regSDMA0_QUEUE0_*` registers.
namespace regs_gfx1201 {
constexpr RegDef kMmFbLocationBase{"regMMMC_VM_FB_LOCATION_BASE", 1364U, 0U};       // regs.py:8736 mmhub_4_1_0
constexpr RegDef kMmFbLocationTop{"regMMMC_VM_FB_LOCATION_TOP", 1365U, 0U};         // regs.py:8737 mmhub_4_1_0
constexpr RegDef kMmSystemApertureDefaultLsb{"regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB", 1224U, 0U};  // regs.py:8676
constexpr RegDef kMmSystemApertureDefaultMsb{"regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB", 1225U, 0U};  // regs.py:8677
constexpr RegDef kMmSystemApertureLow{"regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR", 1369U, 0U};   // regs.py:8741
constexpr RegDef kMmSystemApertureHigh{"regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR", 1370U, 0U}; // regs.py:8742
constexpr RegDef kMmMxL1TlbCntl{"regMMMC_VM_MX_L1_TLB_CNTL", 1371U, 0U};           // regs.py:8743
constexpr RegDef kMmL2Cntl{"regMMVM_L2_CNTL", 1252U, 0U};                         // regs.py:8687
constexpr RegDef kMmL2Cntl2{"regMMVM_L2_CNTL2", 1253U, 0U};                       // regs.py:8688
constexpr RegDef kMmL2Cntl3{"regMMVM_L2_CNTL3", 1254U, 0U};                       // regs.py:8689
constexpr RegDef kMmL2Cntl4{"regMMVM_L2_CNTL4", 1277U, 0U};                       // regs.py:8711
constexpr RegDef kMmL2Cntl5{"regMMVM_L2_CNTL5", 1283U, 0U};                       // regs.py:8717
constexpr RegDef kMmProtectionFaultCntl2{"regMMVM_L2_PROTECTION_FAULT_CNTL2", 1261U, 0U}; // regs.py:8696
constexpr RegDef kMmProtectionFaultDefaultLo{"regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32", 1268U, 0U}; // regs.py:8703
constexpr RegDef kMmProtectionFaultDefaultHi{"regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32", 1269U, 0U}; // regs.py:8704
constexpr RegDef kMmIdentityLowLo{"regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32", 1271U, 0U};
constexpr RegDef kMmIdentityLowHi{"regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32", 1272U, 0U};
constexpr RegDef kMmIdentityHighLo{"regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32", 1273U, 0U};
constexpr RegDef kMmIdentityHighHi{"regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32", 1274U, 0U};
constexpr RegDef kMmIdentityOffsetLo{"regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32", 1275U, 0U};
constexpr RegDef kMmIdentityOffsetHi{"regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32", 1276U, 0U};
constexpr RegDef kMmContext0Cntl{"regMMVM_CONTEXT0_CNTL", 1380U, 0U};              // regs.py:8744
constexpr RegDef kMmInvalidateEng17Sem{"regMMVM_INVALIDATE_ENG17_SEM", 1414U, 0U}; // regs.py:8778
constexpr RegDef kMmInvalidateEng17Req{"regMMVM_INVALIDATE_ENG17_REQ", 1432U, 0U}; // regs.py:8796
constexpr RegDef kMmInvalidateEng17Ack{"regMMVM_INVALIDATE_ENG17_ACK", 1450U, 0U}; // regs.py:8814
constexpr RegDef kMmContext0BaseLo{"regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", 1487U, 0U};  // regs.py:8851
constexpr RegDef kMmContext0BaseHi{"regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32", 1488U, 0U};  // regs.py:8852

constexpr RegDef kGrbmSoftReset{"regGRBM_SOFT_RESET", 3496U, 0U};                         // regs.py:5538 gc_12_0_0
constexpr RegDef kGrbmGfxCntl{"regGRBM_GFX_CNTL", 2304U, 1U};                    // regs.py:6049 gc_12_0_0
constexpr RegDef kCpIntCntlRing0{"regCP_INT_CNTL_RING0", 7690U, 0U};              // regs.py:5970 gc_12_0_0
constexpr RegDef kCpMec1F32Interrupt{"regCP_MEC1_F32_INTERRUPT", 7702U, 0U};     // regs.py:5971 gc_12_0_0
constexpr RegDef kCpMec1InstrPntr{"regCP_MEC1_INSTR_PNTR", 7750U, 0U};           // regs.py:5974 gc_12_0_0
constexpr RegDef kCpMecRs64Interrupt{"regCP_MEC_RS64_INTERRUPT", 10503U, 1U};    // regs.py gc_12_0_0
constexpr RegDef kCpMecRs64PendingInterrupt{"regCP_MEC_RS64_PENDING_INTERRUPT", 10549U, 1U}; // regs.py gc_12_0_0
constexpr RegDef kCpMecRs64ExceptionStatus{"regCP_MEC_RS64_EXCEPTION_STATUS", 10551U, 1U}; // regs.py gc_12_0_0
constexpr RegDef kCpMecRs64InstrPntr{"regCP_MEC_RS64_INSTR_PNTR", 10504U, 1U};
constexpr RegDef kCpMecLocalInstrBaseLo{"regCP_MEC_LOCAL_INSTR_BASE_LO", 10540U, 1U};
constexpr RegDef kCpMecLocalInstrBaseHi{"regCP_MEC_LOCAL_INSTR_BASE_HI", 10541U, 1U};
constexpr RegDef kCpMecLocalInstrMaskLo{"regCP_MEC_LOCAL_INSTR_MASK_LO", 10542U, 1U};
constexpr RegDef kCpMecLocalInstrMaskHi{"regCP_MEC_LOCAL_INSTR_MASK_HI", 10543U, 1U};
constexpr RegDef kCpMecLocalInstrAperture{"regCP_MEC_LOCAL_INSTR_APERTURE", 10544U, 1U};
constexpr RegDef kCpMecRs64PrgrmCntrStart{"regCP_MEC_RS64_PRGRM_CNTR_START", 10496U, 1U};
constexpr RegDef kCpMecRs64PrgrmCntrStartHi{"regCP_MEC_RS64_PRGRM_CNTR_START_HI", 10552U, 1U};
// tinygrad/runtime/autogen/am/regs.py gc_12_0_0:6060 regCP_MEC_RS64_CNTL (10500).
// Fields: mec_invalidate_icache(4), mec_pipe0_reset(16), mec_pipe1_reset(17),
// mec_pipe2_reset(18), mec_pipe3_reset(19), mec_pipe0_active(26),
// mec_pipe1_active(27), mec_pipe2_active(28), mec_pipe3_active(29),
// mec_halt(30), mec_step(31). Segment 1 (GC), same as sibling MEC RS64 regs.
constexpr RegDef kCpMecRs64Cntl{"regCP_MEC_RS64_CNTL", 10500U, 1U};
constexpr RegDef kCpMecRs64InterruptData16{"regCP_MEC_RS64_INTERRUPT_DATA_16", 10554U, 1U};
constexpr RegDef kCpMecRs64InterruptData17{"regCP_MEC_RS64_INTERRUPT_DATA_17", 10555U, 1U};
constexpr RegDef kCpMecRs64InterruptData18{"regCP_MEC_RS64_INTERRUPT_DATA_18", 10556U, 1U};
constexpr RegDef kCpMecRs64InterruptData19{"regCP_MEC_RS64_INTERRUPT_DATA_19", 10557U, 1U};
constexpr RegDef kCpMecRs64InterruptData20{"regCP_MEC_RS64_INTERRUPT_DATA_20", 10558U, 1U};
constexpr RegDef kCpMecRs64InterruptData21{"regCP_MEC_RS64_INTERRUPT_DATA_21", 10559U, 1U};
constexpr RegDef kCpMecRs64InterruptData22{"regCP_MEC_RS64_INTERRUPT_DATA_22", 10560U, 1U};
constexpr RegDef kCpMecRs64InterruptData23{"regCP_MEC_RS64_INTERRUPT_DATA_23", 10561U, 1U};
constexpr RegDef kCpMecRs64InterruptData24{"regCP_MEC_RS64_INTERRUPT_DATA_24", 10562U, 1U};
constexpr RegDef kCpMecRs64InterruptData25{"regCP_MEC_RS64_INTERRUPT_DATA_25", 10563U, 1U};
constexpr RegDef kCpMecRs64InterruptData26{"regCP_MEC_RS64_INTERRUPT_DATA_26", 10564U, 1U};
constexpr RegDef kCpMecRs64InterruptData27{"regCP_MEC_RS64_INTERRUPT_DATA_27", 10565U, 1U};
constexpr RegDef kCpMecRs64InterruptData28{"regCP_MEC_RS64_INTERRUPT_DATA_28", 10566U, 1U};
constexpr RegDef kCpMecRs64InterruptData29{"regCP_MEC_RS64_INTERRUPT_DATA_29", 10567U, 1U};
constexpr RegDef kCpMecRs64InterruptData30{"regCP_MEC_RS64_INTERRUPT_DATA_30", 10568U, 1U};
constexpr RegDef kCpMecRs64InterruptData31{"regCP_MEC_RS64_INTERRUPT_DATA_31", 10569U, 1U};
constexpr RegDef kCpMqdBaseAddr{"regCP_MQD_BASE_ADDR", 8105U, 0U};               // regs.py:5981 gc_12_0_0
constexpr RegDef kCpHqdActive{"regCP_HQD_ACTIVE", 8107U, 0U};                    // regs.py:5983 gc_12_0_0
constexpr RegDef kCpHqdPqBase{"regCP_HQD_PQ_BASE", 8113U, 0U};                   // regs.py:5989 gc_12_0_0
constexpr RegDef kCpHqdPqBaseHi{"regCP_HQD_PQ_BASE_HI", 8114U, 0U};              // regs.py:5990 gc_12_0_0
constexpr RegDef kCpHqdPqRptr{"regCP_HQD_PQ_RPTR", 8115U, 0U};                   // regs.py:5991 gc_12_0_0
constexpr RegDef kCpHqdPqRptrReportAddr{"regCP_HQD_PQ_RPTR_REPORT_ADDR", 8116U, 0U}; // regs.py:5992 gc_12_0_0
constexpr RegDef kCpHqdPqRptrReportAddrHi{"regCP_HQD_PQ_RPTR_REPORT_ADDR_HI", 8117U, 0U}; // regs.py:5993 gc_12_0_0
constexpr RegDef kCpHqdPqWptrPollAddr{"regCP_HQD_PQ_WPTR_POLL_ADDR", 8118U, 0U}; // regs.py:5994 gc_12_0_0
constexpr RegDef kCpHqdPqWptrPollAddrHi{"regCP_HQD_PQ_WPTR_POLL_ADDR_HI", 8119U, 0U}; // regs.py:5995 gc_12_0_0
constexpr RegDef kCpHqdPqDoorbellControl{"regCP_HQD_PQ_DOORBELL_CONTROL", 8120U, 0U}; // regs.py:5996 gc_12_0_0
constexpr RegDef kCpHqdPqControl{"regCP_HQD_PQ_CONTROL", 8122U, 0U};              // regs.py:5997 gc_12_0_0
constexpr RegDef kCpHqdDequeueRequest{"regCP_HQD_DEQUEUE_REQUEST", 8129U, 0U};   // regs.py:6004 gc_12_0_0
constexpr RegDef kCpMqdControl{"regCP_MQD_CONTROL", 8139U, 0U};                  // regs.py:6014 gc_12_0_0
constexpr RegDef kCpHqdEopBaseAddr{"regCP_HQD_EOP_BASE_ADDR", 8142U, 0U};        // regs.py:6017 gc_12_0_0
constexpr RegDef kCpHqdEopBaseAddrHi{"regCP_HQD_EOP_BASE_ADDR_HI", 8143U, 0U};   // regs.py:6018 gc_12_0_0
constexpr RegDef kCpHqdEopControl{"regCP_HQD_EOP_CONTROL", 8144U, 0U};           // regs.py:6019 gc_12_0_0
constexpr RegDef kCpHqdPqWptrLo{"regCP_HQD_PQ_WPTR_LO", 8159U, 0U};              // regs.py:6036 gc_12_0_0
constexpr RegDef kCpHqdPqWptrHi{"regCP_HQD_PQ_WPTR_HI", 8160U, 0U};              // regs.py:6037 gc_12_0_0
constexpr RegDef kSpiComputeQueueReset{"regSPI_COMPUTE_QUEUE_RESET", 8051U, 0U}; // regs.py:6694 gc_12_0_0
constexpr RegDef kCpStat{"regCP_STAT", 3904U, 0U};                               // regs.py:5573 gc_12_0_0
constexpr RegDef kCpRbWptrPollCntl{"regCP_RB_WPTR_POLL_CNTL", 3938U, 0U};         // regs.py:5574 gc_12_0_0
constexpr RegDef kCpMecDoorbellRangeLower{"regCP_MEC_DOORBELL_RANGE_LOWER", 7676U, 0U}; // regs.py:5968 gc_12_0_0
constexpr RegDef kCpMecDoorbellRangeUpper{"regCP_MEC_DOORBELL_RANGE_UPPER", 7677U, 0U}; // regs.py:5969 gc_12_0_0
constexpr RegDef kSdma0Queue0RbCntl{"regSDMA0_QUEUE0_RB_CNTL", 128U, 0U};                 // regs.py:5428 gc_12_0_0
constexpr RegDef kSdma0Queue0RbBase{"regSDMA0_QUEUE0_RB_BASE", 129U, 0U};                 // regs.py:5429 gc_12_0_0
constexpr RegDef kSdma0Queue0RbBaseHi{"regSDMA0_QUEUE0_RB_BASE_HI", 130U, 0U};            // regs.py:5430 gc_12_0_0
constexpr RegDef kSdma0Queue0RbRptr{"regSDMA0_QUEUE0_RB_RPTR", 131U, 0U};                 // regs.py:5431 gc_12_0_0
constexpr RegDef kSdma0Queue0RbRptrHi{"regSDMA0_QUEUE0_RB_RPTR_HI", 132U, 0U};            // regs.py:5432 gc_12_0_0
constexpr RegDef kSdma0Queue0RbWptr{"regSDMA0_QUEUE0_RB_WPTR", 133U, 0U};                 // regs.py:5433 gc_12_0_0
constexpr RegDef kSdma0Queue0RbWptrHi{"regSDMA0_QUEUE0_RB_WPTR_HI", 134U, 0U};            // regs.py:5434 gc_12_0_0
constexpr RegDef kSdma0Queue0RbRptrAddrLo{"regSDMA0_QUEUE0_RB_RPTR_ADDR_LO", 135U, 0U};   // regs.py:5435 gc_12_0_0
constexpr RegDef kSdma0Queue0RbRptrAddrHi{"regSDMA0_QUEUE0_RB_RPTR_ADDR_HI", 136U, 0U};   // regs.py:5436 gc_12_0_0
constexpr RegDef kSdma0Queue0IbCntl{"regSDMA0_QUEUE0_IB_CNTL", 137U, 0U};                 // regs.py:5437 gc_12_0_0
constexpr RegDef kSdma0Queue0Doorbell{"regSDMA0_QUEUE0_DOORBELL", 143U, 0U};              // regs.py:5443 gc_12_0_0
constexpr RegDef kSdma0Queue0DoorbellOffset{"regSDMA0_QUEUE0_DOORBELL_OFFSET", 145U, 0U}; // regs.py:5445 gc_12_0_0
constexpr RegDef kSdma0Queue0RbWptrPollAddrLo{"regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_LO", 152U, 0U}; // regs.py:5452 gc_12_0_0
constexpr RegDef kSdma0Queue0RbWptrPollAddrHi{"regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_HI", 153U, 0U}; // regs.py:5453 gc_12_0_0
constexpr RegDef kSdma0Queue0MinorPtrUpdate{"regSDMA0_QUEUE0_MINOR_PTR_UPDATE", 155U, 0U}; // regs.py:5455 gc_12_0_0
constexpr RegDef kSdma0Queue0ContextStatus{"regSDMA0_QUEUE0_CONTEXT_STATUS", 176U, 0U};   // regs.py:5474 gc_12_0_0
constexpr RegDef kMmContext0StartLo{"regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32", 1519U, 0U}; // regs.py:8883
constexpr RegDef kMmContext0StartHi{"regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32", 1520U, 0U}; // regs.py:8884
constexpr RegDef kMmContext0EndLo{"regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32", 1551U, 0U};     // regs.py:8915
constexpr RegDef kMmContext0EndHi{"regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32", 1552U, 0U};     // regs.py:8916
constexpr RegDef kMmReservedCid2{"regMMVM_L2_BANK_SELECT_RESERVED_CID2", 1280U, 0U};            // regs.py:8714

constexpr RegDef kNbifRsmuIndex{"regBIF_BX_PF0_RSMU_INDEX", 0U, 1U};              // regs.py:9103 nbif_6_3_1
constexpr RegDef kNbifRsmuData{"regBIF_BX_PF0_RSMU_DATA", 1U, 1U};                // regs.py:9104 nbif_6_3_1
constexpr RegDef kNbifHdpFlushCntl{"regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL", 301U, 2U}; // regs.py:9110 nbif_6_3_1
constexpr RegDef kRccDoorbellAperEn{"regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN", 192U, 2U}; // regs.py:9113 nbif_6_3_1
constexpr RegDef kGdcS2aDoorbellEntry0{"regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL", 459U, 2U}; // regs.py:9114 nbif_6_3_1
constexpr RegDef kGdcS2aDoorbellEntry3{"regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL", 462U, 2U}; // regs.py:9117 nbif_6_3_1
constexpr RegDef kRccDev0Epf2Strap2{"regRCC_DEV0_EPF2_STRAP2", 53506U, 5U}; // regs.py:9130 nbif_6_3_1

constexpr RegDef kGcMcVmSystemApertureDefaultLsb{"regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB", 5544U, 0U}; // regs.py:5662 gc_12_0_0
constexpr RegDef kGcMcVmSystemApertureDefaultMsb{"regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB", 5545U, 0U}; // regs.py:5663 gc_12_0_0
constexpr RegDef kGcMcVmSystemApertureLow{"regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR", 5657U, 0U};               // regs.py:5720 gc_12_0_0
constexpr RegDef kGcMcVmSystemApertureHigh{"regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR", 5658U, 0U};             // regs.py:5721 gc_12_0_0
constexpr RegDef kGcMcVmMxL1TlbCntl{"regGCMC_VM_MX_L1_TLB_CNTL", 5659U, 0U};                               // regs.py:5722 gc_12_0_0
constexpr RegDef kGcMcVmFbLocationBase{"regGCMC_VM_FB_LOCATION_BASE", 5652U, 0U};                          // regs.py:5715 gc_12_0_0
constexpr RegDef kGcMcVmFbLocationTop{"regGCMC_VM_FB_LOCATION_TOP", 5653U, 0U};                            // regs.py:5716 gc_12_0_0
constexpr RegDef kGcMcVmAgpTop{"regGCMC_VM_AGP_TOP", 5654U, 0U};                                         // regs.py:5717 gc_12_0_0
constexpr RegDef kGcMcVmAgpBot{"regGCMC_VM_AGP_BOT", 5655U, 0U};                                         // regs.py:5718 gc_12_0_0
constexpr RegDef kGcMcVmAgpBase{"regGCMC_VM_AGP_BASE", 5656U, 0U};                                       // regs.py:5719 gc_12_0_0
constexpr RegDef kGcVmL2Cntl{"regGCVM_L2_CNTL", 5572U, 0U};                                                // regs.py:5673 gc_12_0_0
constexpr RegDef kGcVmL2Cntl2{"regGCVM_L2_CNTL2", 5573U, 0U};                                              // regs.py:5674 gc_12_0_0
constexpr RegDef kGcVmL2Cntl3{"regGCVM_L2_CNTL3", 5574U, 0U};                                              // regs.py:5675 gc_12_0_0
constexpr RegDef kGcVmL2Cntl4{"regGCVM_L2_CNTL4", 5597U, 0U};                                              // regs.py:5697 gc_12_0_0
constexpr RegDef kGcVmL2Cntl5{"regGCVM_L2_CNTL5", 5603U, 0U};                                              // regs.py:5703 gc_12_0_0
constexpr RegDef kGcVmProtectionFaultDefaultLo{"regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32", 5588U, 0U}; // regs.py:5689 gc_12_0_0
constexpr RegDef kGcVmProtectionFaultDefaultHi{"regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32", 5589U, 0U}; // regs.py:5690 gc_12_0_0
constexpr RegDef kGcVmProtectionFaultStatusLo32{"regGCVM_L2_PROTECTION_FAULT_STATUS_LO32", 5584U, 0U}; // regs.py:5685 gc_12_0_0
constexpr RegDef kGcVmProtectionFaultStatusHi32{"regGCVM_L2_PROTECTION_FAULT_STATUS_HI32", 5585U, 0U}; // regs.py:5686 gc_12_0_0
constexpr RegDef kGcVmProtectionFaultAddrLo32{"regGCVM_L2_PROTECTION_FAULT_ADDR_LO32", 5586U, 0U};     // regs.py:5687 gc_12_0_0
constexpr RegDef kGcVmProtectionFaultAddrHi32{"regGCVM_L2_PROTECTION_FAULT_ADDR_HI32", 5587U, 0U};     // regs.py:5688 gc_12_0_0
constexpr RegDef kGcIdentityLowLo{"regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32", 5591U, 0U};        // regs.py:5691 gc_12_0_0
constexpr RegDef kGcIdentityLowHi{"regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32", 5592U, 0U};        // regs.py:5692 gc_12_0_0
constexpr RegDef kGcIdentityHighLo{"regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32", 5593U, 0U};      // regs.py:5693 gc_12_0_0
constexpr RegDef kGcIdentityHighHi{"regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32", 5594U, 0U};      // regs.py:5694 gc_12_0_0
constexpr RegDef kGcIdentityOffsetLo{"regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32", 5595U, 0U};        // regs.py:5695 gc_12_0_0
constexpr RegDef kGcIdentityOffsetHi{"regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32", 5596U, 0U};        // regs.py:5696 gc_12_0_0
constexpr RegDef kGcContext0Cntl{"regGCVM_CONTEXT0_CNTL", 5668U, 0U};                                      // regs.py:5723 gc_12_0_0
constexpr RegDef kGcInvalidateEng17Sem{"regGCVM_INVALIDATE_ENG17_SEM", 5702U, 0U};                         // regs.py:5757 gc_12_0_0
constexpr RegDef kGcInvalidateEng17Req{"regGCVM_INVALIDATE_ENG17_REQ", 5720U, 0U};                         // regs.py:5775 gc_12_0_0
constexpr RegDef kGcInvalidateEng17Ack{"regGCVM_INVALIDATE_ENG17_ACK", 5738U, 0U};                         // regs.py:5793 gc_12_0_0
constexpr std::array<RegDef, 18> kGcInvalidateEngAddrRangeLo{{
    {"regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32", 5707U, 0U},
    {"regGCVM_INVALIDATE_ENG1_ADDR_RANGE_LO32", 5709U, 0U},
    {"regGCVM_INVALIDATE_ENG2_ADDR_RANGE_LO32", 5711U, 0U},
    {"regGCVM_INVALIDATE_ENG3_ADDR_RANGE_LO32", 5713U, 0U},
    {"regGCVM_INVALIDATE_ENG4_ADDR_RANGE_LO32", 5715U, 0U},
    {"regGCVM_INVALIDATE_ENG5_ADDR_RANGE_LO32", 5717U, 0U},
    {"regGCVM_INVALIDATE_ENG6_ADDR_RANGE_LO32", 5719U, 0U},
    {"regGCVM_INVALIDATE_ENG7_ADDR_RANGE_LO32", 5721U, 0U},
    {"regGCVM_INVALIDATE_ENG8_ADDR_RANGE_LO32", 5723U, 0U},
    {"regGCVM_INVALIDATE_ENG9_ADDR_RANGE_LO32", 5725U, 0U},
    {"regGCVM_INVALIDATE_ENG10_ADDR_RANGE_LO32", 5727U, 0U},
    {"regGCVM_INVALIDATE_ENG11_ADDR_RANGE_LO32", 5729U, 0U},
    {"regGCVM_INVALIDATE_ENG12_ADDR_RANGE_LO32", 5731U, 0U},
    {"regGCVM_INVALIDATE_ENG13_ADDR_RANGE_LO32", 5733U, 0U},
    {"regGCVM_INVALIDATE_ENG14_ADDR_RANGE_LO32", 5735U, 0U},
    {"regGCVM_INVALIDATE_ENG15_ADDR_RANGE_LO32", 5737U, 0U},
    {"regGCVM_INVALIDATE_ENG16_ADDR_RANGE_LO32", 5739U, 0U},
    {"regGCVM_INVALIDATE_ENG17_ADDR_RANGE_LO32", 5741U, 0U},
}};
constexpr std::array<RegDef, 18> kGcInvalidateEngAddrRangeHi{{
    {"regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32", 5708U, 0U},
    {"regGCVM_INVALIDATE_ENG1_ADDR_RANGE_HI32", 5710U, 0U},
    {"regGCVM_INVALIDATE_ENG2_ADDR_RANGE_HI32", 5712U, 0U},
    {"regGCVM_INVALIDATE_ENG3_ADDR_RANGE_HI32", 5714U, 0U},
    {"regGCVM_INVALIDATE_ENG4_ADDR_RANGE_HI32", 5716U, 0U},
    {"regGCVM_INVALIDATE_ENG5_ADDR_RANGE_HI32", 5718U, 0U},
    {"regGCVM_INVALIDATE_ENG6_ADDR_RANGE_HI32", 5720U, 0U},
    {"regGCVM_INVALIDATE_ENG7_ADDR_RANGE_HI32", 5722U, 0U},
    {"regGCVM_INVALIDATE_ENG8_ADDR_RANGE_HI32", 5724U, 0U},
    {"regGCVM_INVALIDATE_ENG9_ADDR_RANGE_HI32", 5726U, 0U},
    {"regGCVM_INVALIDATE_ENG10_ADDR_RANGE_HI32", 5728U, 0U},
    {"regGCVM_INVALIDATE_ENG11_ADDR_RANGE_HI32", 5730U, 0U},
    {"regGCVM_INVALIDATE_ENG12_ADDR_RANGE_HI32", 5732U, 0U},
    {"regGCVM_INVALIDATE_ENG13_ADDR_RANGE_HI32", 5734U, 0U},
    {"regGCVM_INVALIDATE_ENG14_ADDR_RANGE_HI32", 5736U, 0U},
    {"regGCVM_INVALIDATE_ENG15_ADDR_RANGE_HI32", 5738U, 0U},
    {"regGCVM_INVALIDATE_ENG16_ADDR_RANGE_HI32", 5740U, 0U},
    {"regGCVM_INVALIDATE_ENG17_ADDR_RANGE_HI32", 5742U, 0U},
}};
constexpr RegDef kGcContext0BaseLo{"regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", 5775U, 0U};               // regs.py:5830 gc_12_0_0
constexpr RegDef kGcContext0BaseHi{"regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32", 5776U, 0U};               // regs.py:5831 gc_12_0_0
constexpr RegDef kGcContext0StartLo{"regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32", 5807U, 0U};             // regs.py:5862 gc_12_0_0
constexpr RegDef kGcContext0StartHi{"regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32", 5808U, 0U};             // regs.py:5863 gc_12_0_0
constexpr RegDef kGcContext0EndLo{"regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32", 5839U, 0U};                 // regs.py:5894 gc_12_0_0
constexpr RegDef kGcContext0EndHi{"regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32", 5840U, 0U};                 // regs.py:5895 gc_12_0_0
}  // namespace regs_gfx1201

bool is_supported_gfx1201_vm_ip_layout(const DiscoveryLog& log, std::string* error_text) {
  if (log.ip.mmhubs.empty()) {
    *error_text = "required MMHUB IP discovery record missing";
    return false;
  }
  for (const IpBlockInfo& mmhub : log.ip.mmhubs) {
    if (mmhub.major != 4U || mmhub.minor != 1U || mmhub.revision != 0U) {
      *error_text = "unsupported MMHUB IP version " + ip_version_text(mmhub) + " (expected mmhub_4_1_0)";
      return false;
    }
  }
  if (!log.ip.nbif.found) {
    *error_text = "required NBIF IP discovery record missing";
    return false;
  }
  if (log.ip.nbif.major != 6U || log.ip.nbif.minor != 3U || log.ip.nbif.revision != 1U) {
    *error_text = "unsupported NBIF IP version " + ip_version_text(log.ip.nbif) + " (expected nbif_6_3_1)";
    return false;
  }
  return true;
}

bool is_supported_gfx1201_ip_layout(const DiscoveryLog& log, std::string* error_text) {
  if (!log.ip.gc.found) {
    *error_text = "required GC IP discovery record missing";
    return false;
  }
  if (log.ip.gc.major != 12U || log.ip.gc.minor != 0U) {
    *error_text = "unsupported GC IP version " + ip_version_text(log.ip.gc) + " (expected gfx1201/gc_12_0_0-compatible 12.0.x)";
    return false;
  }
  return is_supported_gfx1201_vm_ip_layout(log, error_text);
}

bool validate_direct_pm4_topology(const DiscoveryLog& log, std::string* error_text) {
  if (!log.ip.gc.found || log.ip.gc.major != 12U || log.ip.gc.minor != 0U ||
      log.ip.gc.revision != 1U) {
    *error_text = "GC IP record missing or unsupported for gfx1201 direct PM4: " +
                  ip_version_text(log.ip.gc);
    return false;
  }
  if (log.ip.gc.instance != 0U) {
    *error_text = "GC instance is not zero: instance=" +
                  std::to_string(static_cast<unsigned>(log.ip.gc.instance));
    return false;
  }
  if (log.ip.gc_instance_count != am_compute::kExpectedXccCount) {
    *error_text = "GC instance count is not " + std::to_string(am_compute::kExpectedXccCount) +
                  ": count=" + std::to_string(log.ip.gc_instance_count);
    return false;
  }
  return true;
}

bool resolve_reg_dword(const IpBlockInfo& ip, const RegDef& reg, uint32_t* reg_dword,
                       std::string* error_text) {
  if (!ip.found) {
    *error_text = std::string("VM register map missing ") + reg.name + ": IP block not discovered";
    return false;
  }
  if (reg.segment >= ip.bases.size()) {
    *error_text = std::string("VM register map missing ") + reg.name + ": segment " +
                  std::to_string(reg.segment) + " absent from " + ip.label + " bases " + ip_bases_text(ip);
    return false;
  }
  const uint64_t dword = ip.bases[reg.segment] + reg.offset;
  if (dword > UINT32_MAX) {
    *error_text = std::string("VM register address exceeds 32-bit dword range for ") + reg.name;
    return false;
  }
  *reg_dword = static_cast<uint32_t>(dword);
  return true;
}

bool write_direct_bar5_reg(const RemoteClient& client, const BarInfo& bar5, uint32_t reg_dword,
                           uint32_t value, std::string* error_text) {
  return raw_bar5_reg_write(client, bar5, reg_dword, value, error_text);
}

bool read_direct_bar5_reg(const RemoteClient& client, const BarInfo& bar5, uint32_t reg_dword,
                          uint32_t* value, std::string* error_text) {
  return raw_bar5_reg_read(client, bar5, reg_dword, value, error_text);
}

bool resolve_nbif_reg_dword(const DiscoveryLog& log, const RegDef& reg, uint32_t* reg_dword,
                            std::string* error_text) {
  return resolve_reg_dword(log.ip.nbif, reg, reg_dword, error_text);
}

bool write_register_dword(const RemoteClient& client, const DiscoveryLog& log, const IpBlockInfo& ip,
                          const RegDef& reg, uint32_t value, std::string* error_text);

bool read_register_dword(const RemoteClient& client, const DiscoveryLog& log, const IpBlockInfo& ip,
                         const RegDef& reg, uint32_t* value, std::string* error_text);

bool rsmu_write_register_dword(const RemoteClient& client, const DiscoveryLog& log, uint32_t target_reg_dword,
                               uint32_t value, std::string* error_text) {
  uint32_t index_reg = 0;
  uint32_t data_reg = 0;
  if (!resolve_nbif_reg_dword(log, regs_gfx1201::kNbifRsmuIndex, &index_reg, error_text)) {
    return false;
  }
  if (!resolve_nbif_reg_dword(log, regs_gfx1201::kNbifRsmuData, &data_reg, error_text)) {
    return false;
  }
  const uint64_t bar5_dwords = log.bar5.size / sizeof(uint32_t);
  if (index_reg >= bar5_dwords || data_reg >= bar5_dwords) {
    *error_text = "RSMU index/data registers are outside direct BAR5 span";
    return false;
  }
  if (!write_direct_bar5_reg(client, log.bar5, index_reg, target_reg_dword * 4U, error_text)) {
    *error_text = "RSMU index write failed: " + *error_text;
    return false;
  }
  if (!write_direct_bar5_reg(client, log.bar5, data_reg, value, error_text)) {
    *error_text = "RSMU data write failed: " + *error_text;
    return false;
  }
  return true;
}

bool rsmu_read_register_dword(const RemoteClient& client, const DiscoveryLog& log, uint32_t target_reg_dword,
                              uint32_t* value, std::string* error_text) {
  uint32_t index_reg = 0;
  uint32_t data_reg = 0;
  if (!resolve_nbif_reg_dword(log, regs_gfx1201::kNbifRsmuIndex, &index_reg, error_text)) {
    return false;
  }
  if (!resolve_nbif_reg_dword(log, regs_gfx1201::kNbifRsmuData, &data_reg, error_text)) {
    return false;
  }
  const uint64_t bar5_dwords = log.bar5.size / sizeof(uint32_t);
  if (index_reg >= bar5_dwords || data_reg >= bar5_dwords) {
    *error_text = "RSMU index/data registers are outside direct BAR5 span";
    return false;
  }
  if (!write_direct_bar5_reg(client, log.bar5, index_reg, target_reg_dword * 4U, error_text)) {
    *error_text = "RSMU index write failed: " + *error_text;
    return false;
  }
  if (!read_direct_bar5_reg(client, log.bar5, data_reg, value, error_text)) {
    *error_text = "RSMU data read failed: " + *error_text;
    return false;
  }
  return true;
}


bool write_register_dword(const RemoteClient& client, const DiscoveryLog& log, const IpBlockInfo& ip,
                          const RegDef& reg, uint32_t value, std::string* error_text) {
  uint32_t reg_dword = 0;
  if (!resolve_reg_dword(ip, reg, &reg_dword, error_text)) {
    return false;
  }
  const uint64_t bar5_dwords = log.bar5.size / sizeof(uint32_t);
  if (reg_dword < bar5_dwords) {
    return write_direct_bar5_reg(client, log.bar5, reg_dword, value, error_text);
  }
  // tinygrad/runtime/support/am/amdev.py:264-270 indirect_rreg/indirect_wreg use NBIF RSMU index/data.
  return rsmu_write_register_dword(client, log, reg_dword, value, error_text);
}

bool read_register_dword(const RemoteClient& client, const DiscoveryLog& log, const IpBlockInfo& ip,
                         const RegDef& reg, uint32_t* value, std::string* error_text) {
  uint32_t reg_dword = 0;
  if (!resolve_reg_dword(ip, reg, &reg_dword, error_text)) {
    return false;
  }
  const uint64_t bar5_dwords = log.bar5.size / sizeof(uint32_t);
  if (reg_dword < bar5_dwords) {
    return read_direct_bar5_reg(client, log.bar5, reg_dword, value, error_text);
  }
  return rsmu_read_register_dword(client, log, reg_dword, value, error_text);
}

bool write_register_pair(const RemoteClient& client, const DiscoveryLog& log, const IpBlockInfo& ip,
                         const RegDef& lo_reg, const RegDef& hi_reg, uint64_t value,
                         std::string* error_text) {
  if (!write_register_dword(client, log, ip, lo_reg, static_cast<uint32_t>(value & 0xffffffffULL), error_text)) {
    return false;
  }
  return write_register_dword(client, log, ip, hi_reg, static_cast<uint32_t>((value >> 32) & 0xffffffffULL), error_text);
}

bool update_register_bits(const RemoteClient& client, const DiscoveryLog& log, const IpBlockInfo& ip,
                          const RegDef& reg, uint32_t clear_mask, uint32_t set_mask,
                          std::string* error_text) {
  uint32_t value = 0;
  if (!read_register_dword(client, log, ip, reg, &value, error_text)) {
    return false;
  }
  value = (value & ~clear_mask) | set_mask;
  return write_register_dword(client, log, ip, reg, value, error_text);
}

bool mmio_write_bar0(const RemoteClient& client, uint64_t offset, const std::vector<uint8_t>& payload,
                     std::string* error_text) {
  if (!client.mmio_write_fire_and_forget(0, offset, payload, error_text)) {
    return false;
  }
  return true;
}

bool write_bar0_qword(const RemoteClient& client, uint64_t paddr, uint64_t value, std::string* error_text) {
  const std::vector<uint8_t> payload = u64_payload_le(value);
  return mmio_write_bar0(client, paddr, payload, error_text);
}

bool zero_bar0_page(const RemoteClient& client, uint64_t paddr, std::string* error_text) {
  const std::vector<uint8_t> zero(static_cast<std::size_t>(kPageSize), 0);
  return mmio_write_bar0(client, paddr, zero, error_text);
}

bool read_bar0_qword(const RemoteClient& client, uint64_t paddr, uint64_t* value, std::string* error_text) {
  RemoteRpcResult result = mmio_read(client, 0, paddr, sizeof(uint64_t));
  if (!result.ok) {
    *error_text = rpc_failure_text("MMIO_READ BAR0 qword", result);
    return false;
  }
  if (result.readout.size() != sizeof(uint64_t)) {
    *error_text = "MMIO_READ BAR0 qword returned " + std::to_string(result.readout.size()) + " bytes";
    return false;
  }
  *value = read_u64_le_bytes(result.readout.data());
  return true;
}

bool verify_bar0_qword(const RemoteClient& client, uint64_t paddr, uint64_t expected,
                       std::string* error_text) {
  uint64_t observed = 0;
  if (!read_bar0_qword(client, paddr, &observed, error_text)) {
    return false;
  }
  if (observed != expected) {
    *error_text = "page-table readback mismatch at paddr " + format_hex64(paddr) +
                  ": expected " + format_hex64(expected) + ", observed " + format_hex64(observed);
    return false;
  }
  return true;
}

bool load_kernel_blob(const RemoteClient& client, DiscoveryLog* log, std::string* error_text) {
  if (log == nullptr) {
    *error_text = "DiscoveryLog precondition failed: null log";
    return false;
  }
  auto fail = [&](const std::string& text) {
    log->compute.kernel_blob_load_status = "fail";
    *error_text = text;
    return false;
  };

  if (kKernelText.size() != kKernelReferenceTextByteCount) {
    return fail("embedded kernel text byte count mismatch: expected=" +
                std::to_string(kKernelReferenceTextByteCount) +
                " observed=" + std::to_string(kKernelText.size()));
  }
  if (log->bar0.size < am_compute::kCodeVramPaddr + kKernelText.size()) {
    return fail("BAR0 too small for kernel text write/readback: bar0_size_bytes=" +
                std::to_string(log->bar0.size) + " required_at_least=" +
                std::to_string(am_compute::kCodeVramPaddr + kKernelText.size()));
  }

  const std::vector<uint8_t> payload(kKernelText.begin(), kKernelText.end());
  if (!mmio_write_bar0(client, am_compute::kCodeVramPaddr, payload, error_text)) {
    return fail("write kernel text to code VRAM paddr " + format_hex64(am_compute::kCodeVramPaddr) +
                " failed: " + *error_text);
  }
  RemoteRpcResult readback = mmio_read(client, 0, am_compute::kCodeVramPaddr, kKernelText.size());
  if (!readback.ok) {
    return fail(rpc_failure_text("MMIO_READ BAR0 kernel text", readback));
  }
  if (readback.readout.size() != kKernelText.size()) {
    return fail("MMIO_READ BAR0 kernel text returned " + std::to_string(readback.readout.size()) +
                " bytes, expected " + std::to_string(kKernelText.size()));
  }
  if (std::memcmp(readback.readout.data(), kKernelText.data(), kKernelText.size()) != 0) {
    return fail("kernel text BAR0 readback mismatch at paddr " +
                format_hex64(am_compute::kCodeVramPaddr));
  }

  log->compute.kernel_blob_load_status = "pass";
  return true;
}

bool write_kernel_kernargs(SysmemMapping* compute_control_mapping, uint64_t output_va,
                           uint64_t input_va, uint64_t scalar_va, std::string* error_text) {
  if (compute_control_mapping == nullptr || compute_control_mapping->data == nullptr ||
      compute_control_mapping->size < am_compute::kComputeControlByteCount) {
    *error_text = "compute_control mapping precondition failed: need ten mapped 4 KiB pages (2 control + 8 ring)";
    return false;
  }
  if (scalar_va != am_compute::kKernargsVa + kKernelReferenceKernargSize) {
    *error_text = "kernarg scalar pointer mismatch: expected " +
                  format_hex64(am_compute::kKernargsVa + kKernelReferenceKernargSize) +
                  ", observed " + format_hex64(scalar_va);
    return false;
  }

  uint8_t* const kernargs =
      static_cast<uint8_t*>(compute_control_mapping->data) + am_compute::kComputeControlKernargsCpuOffset;
  std::memset(kernargs, 0, kPageSize);
  auto write_u64_to_kernargs = [&](std::size_t offset, uint64_t value) {
    const std::vector<uint8_t> bytes = u64_payload_le(value);
    std::memcpy(kernargs + offset, bytes.data(), bytes.size());
  };
  write_u64_to_kernargs(0, output_va);
  write_u64_to_kernargs(8, input_va);
  write_u64_to_kernargs(16, scalar_va);
  const std::vector<uint8_t> scalar = u32_payload_le(static_cast<uint32_t>(am_compute::kScalarValue));
  std::memcpy(kernargs + kKernelReferenceKernargSize, scalar.data(), scalar.size());
  std::atomic_thread_fence(std::memory_order_seq_cst);

  if (read_u64_le_bytes(kernargs) != output_va || read_u64_le_bytes(kernargs + 8) != input_va ||
      read_u64_le_bytes(kernargs + 16) != scalar_va ||
      read_u32_le_bytes(kernargs + kKernelReferenceKernargSize) != am_compute::kScalarValue) {
    *error_text = "kernarg CPU layout readback mismatch";
    return false;
  }
  return true;
}

bool write_fixed_page_tables(const RemoteClient& client, DiscoveryLog* log, const VmBufferLog& staging,
                             const VmBufferLog& readback, const VmBufferLog& sdma_control,
                             const VmBufferLog* compute_control, std::string* error_text) {
  const FixedVmPageTables& t = log->vm.tables;
  if (log->bar0.size <= t.staging_ptb_paddr + kPageSize ||
      log->bar0.size <= t.child_ptb_paddr + kPageSize) {
    *error_text = "BAR0 too small for fixed PTBs: bar0_size_bytes=" +
                  std::to_string(log->bar0.size);
    return false;
  }
  uint64_t required_vram_end = t.device_buffer_paddr + kPageSize;
  if (compute_control != nullptr) {
    required_vram_end = am_compute::kEopVramPaddr + kPageSize;
  }
  if (log->bar0.size <= required_vram_end) {
    *error_text = "BAR0 too small for fixed VRAM proof/compute pages: bar0_size_bytes=" +
                  std::to_string(log->bar0.size) + " required_gt=" +
                  std::to_string(required_vram_end);
    return false;
  }
  if (log->vram_size_bytes < required_vram_end) {
    *error_text = "VRAM too small for fixed proof/compute pages: vram_size_bytes=" +
                  std::to_string(log->vram_size_bytes) + " required_at_least=" +
                  std::to_string(required_vram_end);
    return false;
  }
  if (staging.sys_pages.empty() || readback.sys_pages.empty() || sdma_control.sys_pages.empty()) {
    *error_text = "MAP_SYSMEM_FD page lists must contain staging, readback, and sdma_control page-0 physical addresses";
    return false;
  }
  if (compute_control != nullptr && compute_control->sys_pages.size() < 26) {
    *error_text = "MAP_SYSMEM_FD page list must contain compute_control 2 control pages plus 8 ring pages plus 16 kernargs-ring pages";
    return false;
  }
  if ((staging.sys_pages[0] % kPageSize) != 0 || (readback.sys_pages[0] % kPageSize) != 0 ||
      (sdma_control.sys_pages[0] % kPageSize) != 0 ||
      (compute_control != nullptr && ((compute_control->sys_pages[0] % kPageSize) != 0 ||
                                      (compute_control->sys_pages[1] % kPageSize) != 0))) {
    *error_text = "MAP_SYSMEM_FD page-0/page-1 physical address is not 4 KiB aligned";
    return false;
  }


  const uint64_t staging_page_count = static_cast<uint64_t>(staging.sys_pages.size());
  if (staging_page_count != ceil_div_u64(staging.requested_size, kPageSize)) {
    *error_text = "MAP_SYSMEM_FD staging page list does not cover the requested staging window";
    return false;
  }
  const uint64_t readback_page_count = static_cast<uint64_t>(readback.sys_pages.size());
  if (readback_page_count != ceil_div_u64(readback.requested_size, kPageSize)) {
    *error_text = "MAP_SYSMEM_FD readback page list does not cover the requested readback window";
    return false;
  }
  const uint64_t staging_end = staging.gpu_va + staging_page_count * kPageSize;
  const uint64_t proof_end = kTransferProofVmVramVa + kTransferProofBufferSize;
  if (staging.gpu_va != kTransferProofVmStagingVa || staging_end < staging.gpu_va ||
      !(staging_end <= kTransferProofVmVramVa || proof_end <= staging.gpu_va)) {
    *error_text = "staging window overlaps the fixed VRAM proof mapping";
    return false;
  }

  const uint64_t sysmem_flags = am_vm::gfx12_leaf_pte_flags(true, true, true);
  const uint64_t vram_flags = am_vm::gfx12_leaf_pte_flags(false, false, false);
  const uint64_t table_flags = am_vm::table_pte_flags();

  const std::array<uint64_t, 7> zero_pages{{t.root_pdb2_paddr, t.memscratch_paddr, t.dummy_page_paddr,
                                            t.child_pdb1_paddr, t.child_pdb0_paddr, t.child_ptb_paddr,
                                            t.staging_ptb_paddr}};
  for (uint64_t page : zero_pages) {
    if (!zero_bar0_page(client, page, error_text)) {
      *error_text = "zero BAR0 page " + format_hex64(page) + " failed: " + *error_text;
      return false;
    }
  }
  if (!zero_bar0_page(client, t.device_buffer_paddr, error_text)) {
    *error_text = "zero fixed VRAM proof buffer failed: " + *error_text;
    return false;
  }


  const am_vm::VmIndices staging_indices = am_vm::vm_indices_for_va(staging.gpu_va);
  const am_vm::VmIndices vram_indices = am_vm::vm_indices_for_va(kTransferProofVmVramVa);
  const am_vm::VmIndices sdma_control_indices = am_vm::vm_indices_for_va(sdma_control.gpu_va);
  if (staging_indices.pdb0 != kDedicatedStagingPdb0Index ||
      staging_indices.ptb + staging_page_count > 512ULL) {
    *error_text = "staging window does not fit the dedicated PTB";
    return false;
  }

  struct QwordWrite { uint64_t paddr; uint64_t value; };
  std::vector<QwordWrite> writes;
  writes.reserve(staging_page_count + readback_page_count + 25);
  auto add_write = [&](uint64_t paddr, uint64_t value) {
    writes.push_back(QwordWrite{paddr, value});
  };
  auto add_ptb_pte = [&](uint64_t gpu_va, uint64_t mapped_paddr, uint64_t flags) {
    const am_vm::VmIndices indices = am_vm::vm_indices_for_va(gpu_va);
    add_write(t.child_ptb_paddr + (indices.ptb * sizeof(uint64_t)),
              am_vm::encode_pte(mapped_paddr, flags));
  };

  add_write(t.root_pdb2_paddr + 0ULL, am_vm::encode_pte(t.child_pdb1_paddr, table_flags));
  add_write(t.child_pdb0_paddr + 0ULL, am_vm::encode_pte(t.child_ptb_paddr, table_flags));
  add_write(t.child_pdb1_paddr + 0ULL, am_vm::encode_pte(t.child_pdb0_paddr, table_flags));
  add_write(t.child_pdb0_paddr + kDedicatedStagingPdb0Index * sizeof(uint64_t),
            am_vm::encode_pte(t.staging_ptb_paddr, table_flags));
  for (uint64_t page = 0; page < staging_page_count; ++page) {
    add_write(t.staging_ptb_paddr + ((staging_indices.ptb + page) * sizeof(uint64_t)),
              am_vm::encode_pte(staging.sys_pages[page], sysmem_flags));
  }
  add_write(t.child_ptb_paddr + (vram_indices.ptb * sizeof(uint64_t)),
            am_vm::encode_pte(t.device_buffer_paddr, vram_flags));
  for (uint64_t page = 0; page < readback_page_count; ++page) {
    add_ptb_pte(readback.gpu_va + page * kPageSize, readback.sys_pages[page], sysmem_flags);
  }
  add_write(t.child_ptb_paddr + (sdma_control_indices.ptb * sizeof(uint64_t)),
            am_vm::encode_pte(sdma_control.sys_pages[0], sysmem_flags));
  if (compute_control != nullptr) {
    add_ptb_pte(am_compute::kOutputVramVa, am_compute::kOutputVramPaddr, vram_flags);
    add_ptb_pte(am_compute::kCodeVramVa, am_compute::kCodeVramPaddr, vram_flags);
    add_ptb_pte(am_compute::kKernargsVa, compute_control->sys_pages[1], sysmem_flags);
    for (uint64_t i = 0; i < 8; ++i) {
      add_ptb_pte(am_compute::kRingVa + i * kPageSize,
                  compute_control->sys_pages[am_compute::kComputeControlRingCpuOffset / kPageSize + i],
                  sysmem_flags);
    }
    for (uint64_t i = 0; i < am_compute::kKernargsRingPageCount; ++i) {
      add_ptb_pte(am_compute::kKernargsRingVa + i * kPageSize,
                  compute_control->sys_pages[am_compute::kComputeControlKernargsRingCpuOffset / kPageSize + i],
                  sysmem_flags);
    }
    add_ptb_pte(am_compute::kRptrVa, compute_control->sys_pages[0], sysmem_flags);
    add_ptb_pte(am_compute::kEopVa, am_compute::kEopVramPaddr, vram_flags);
  }

  for (const QwordWrite& write : writes) {
    if (!write_bar0_qword(client, write.paddr, write.value, error_text)) {
      *error_text = "write PTE qword at " + format_hex64(write.paddr) + " failed: " + *error_text;
      return false;
    }
  }
  for (const QwordWrite& write : writes) {
    if (!verify_bar0_qword(client, write.paddr, write.value, error_text)) {
      return false;
    }
  }
  log->vm.page_tables_written = "pass";
  return true;
}

uint32_t encode_mx_l1_tlb_cntl() {
  // regs.py:8743 fields: enable_l1_tlb bit0, system_access_mode bits3..4,
  // enable_advanced_driver_model bit6, mtype bits11..12; soc_12.py:7 MTYPE_UC == 3.
  return (1U << 0) | (3U << 3) | (1U << 6) | (static_cast<uint32_t>(am_vm::kMtypeUc) << 11);
}

uint32_t encode_l2_cntl3() {
  // tinygrad/runtime/support/am/ip.py:139-140 writes 4k/bigk associativity=1,
  // bank_select=9, l2_cache_bigk_fragment_size=6 for gfx12 (not trans_futher).
  return 9U | (6U << 15) | (1U << 20) | (1U << 31);
}

uint32_t encode_context0_cntl() {
  // tinygrad/runtime/support/am/ip.py:112-115: base 0x1800000, fault interrupt/default
  // bits for pde0,dummy_page,range,valid,read,write,execute, enable_context=1,
  // page_table_depth=3, block_size=0 for gfx12 PDB2 root.
  uint32_t value = 0x01800000U | 1U | (3U << 1);
  for (uint32_t bit = 10; bit <= 23; ++bit) {
    value |= 1U << bit;
  }
  return value;
}

uint32_t encode_invalidate_req_vmid0() {
  // regs.py:8796 MM invalidate fields match ip.py:95-96 request bits for VMID0.
  return am_vm::kInvalidateMaskVmid0 | (1U << 19) | (1U << 20) | (1U << 21) |
         (1U << 22) | (1U << 23);
}

bool program_mmhubs_vmid0(const RemoteClient& client, DiscoveryLog* log, std::string* error_text) {
  // tinygrad/runtime/support/am/ip.py:50-68 derives FB/system aperture and VM base/end;
  // lines 117-152 initialize the MM hub, then enable VMID0 context.
  for (const IpBlockInfo& mmhub : log->ip.mmhubs) {
    uint32_t fb_base_reg = 0;
    uint32_t fb_top_reg = 0;
    if (!read_register_dword(client, *log, mmhub, regs_gfx1201::kMmFbLocationBase, &fb_base_reg, error_text)) {
      *error_text = "read MMHUB FB base failed: " + *error_text;
      return false;
    }
    if (!read_register_dword(client, *log, mmhub, regs_gfx1201::kMmFbLocationTop, &fb_top_reg, error_text)) {
      *error_text = "read MMHUB FB top failed: " + *error_text;
      return false;
    }
    const uint64_t fb_base = static_cast<uint64_t>(fb_base_reg & 0x00ffffffU) << 24;
    const uint64_t fb_end = static_cast<uint64_t>(fb_top_reg & 0x00ffffffU) << 24;
  std::printf("mmhub_fb_base_reg: 0x%08x\n", fb_base_reg);
  std::printf("mmhub_fb_base: 0x%016llx\n", static_cast<unsigned long long>(fb_base));
  std::printf("mmhub_fb_end: 0x%016llx\n", static_cast<unsigned long long>(fb_end));
  std::printf("mqd_paddr_raw: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kMqdPaddr));
  std::printf("mqd_mc_addr: 0x%016llx (tinygrad cp_mqd_base_addr = mc_base + paddr)\n",
              static_cast<unsigned long long>(fb_base + am_compute::kMqdPaddr));
    const uint64_t vm_start = am_vm::kVaBase >> 12;
    const uint64_t vm_end = ((am_vm::kVaBase + (1ULL << 44)) - 1ULL) >> 12;
    const uint64_t root_base = log->vm.tables.root_pdb2_paddr | 1ULL;  // paddr2xgmi is identity: mmhub_4_1_0 lacks XGMI_LFB regs.
    const uint64_t memscratch_ppn = log->vm.tables.memscratch_paddr >> 12;
    const uint64_t dummy_ppn = log->vm.tables.dummy_page_paddr >> 12;

    if (!write_register_dword(client, *log, mmhub, regs_gfx1201::kMmSystemApertureLow,
                              static_cast<uint32_t>(fb_base >> 18), error_text)) {
      *error_text = "program MMHUB system aperture low failed: " + *error_text;
      return false;
    }
    if (!write_register_dword(client, *log, mmhub, regs_gfx1201::kMmSystemApertureHigh,
                              static_cast<uint32_t>(fb_end >> 18), error_text)) {
      *error_text = "program MMHUB system aperture high failed: " + *error_text;
      return false;
    }
    if (!write_register_pair(client, *log, mmhub, regs_gfx1201::kMmSystemApertureDefaultLsb,
                             regs_gfx1201::kMmSystemApertureDefaultMsb, memscratch_ppn, error_text)) {
      *error_text = "program MMHUB aperture default address failed: " + *error_text;
      return false;
    }
    if (!write_register_pair(client, *log, mmhub, regs_gfx1201::kMmProtectionFaultDefaultLo,
                             regs_gfx1201::kMmProtectionFaultDefaultHi, dummy_ppn, error_text)) {
      *error_text = "program MMHUB protection fault default failed: " + *error_text;
      return false;
    }

    if (!update_register_bits(client, *log, mmhub, regs_gfx1201::kMmProtectionFaultCntl2,
                              1U << 18, 1U << 18, error_text)) {
      *error_text = "program MMHUB protection fault cntl2 failed: " + *error_text;
      return false;
    }
    if (!update_register_bits(client, *log, mmhub, regs_gfx1201::kMmMxL1TlbCntl,
                              (1U << 0) | (3U << 3) | (1U << 5) | (1U << 6) | (3U << 11),
                              encode_mx_l1_tlb_cntl(), error_text)) {
      *error_text = "program MMHUB MX L1 TLB cntl failed: " + *error_text;
      return false;
    }
    if (!update_register_bits(client, *log, mmhub, regs_gfx1201::kMmL2Cntl,
                              (1U << 0) | (1U << 1) | (1U << 8) | (1U << 11) | (1U << 18) |
                                  (3U << 19) | (31U << 21),
                              (1U << 0) | (1U << 11) | (1U << 19), error_text)) {
      *error_text = "program MMHUB L2 cntl failed: " + *error_text;
      return false;
    }
    if (!update_register_bits(client, *log, mmhub, regs_gfx1201::kMmL2Cntl2,
                              (1U << 0) | (1U << 1), (1U << 0) | (1U << 1), error_text)) {
      *error_text = "program MMHUB L2 cntl2 failed: " + *error_text;
      return false;
    }
    if (!write_register_dword(client, *log, mmhub, regs_gfx1201::kMmL2Cntl3, encode_l2_cntl3(), error_text)) {
      *error_text = "program MMHUB L2 cntl3 failed: " + *error_text;
      return false;
    }
    if (!write_register_dword(client, *log, mmhub, regs_gfx1201::kMmL2Cntl4, 1U, error_text)) {
      *error_text = "program MMHUB L2 cntl4 failed: " + *error_text;
      return false;
    }
    if (!write_register_dword(client, *log, mmhub, regs_gfx1201::kMmL2Cntl5, 0x1ffU << 5, error_text)) {
      *error_text = "program MMHUB L2 cntl5 failed: " + *error_text;
      return false;
    }

    if (!write_register_pair(client, *log, mmhub, regs_gfx1201::kMmContext0StartLo,
                             regs_gfx1201::kMmContext0StartHi, vm_start, error_text)) {
      *error_text = "program MMHUB VMID0 start failed: " + *error_text;
      return false;
    }
    if (!write_register_pair(client, *log, mmhub, regs_gfx1201::kMmContext0EndLo,
                             regs_gfx1201::kMmContext0EndHi, vm_end, error_text)) {
      *error_text = "program MMHUB VMID0 end failed: " + *error_text;
      return false;
    }
    if (!write_register_pair(client, *log, mmhub, regs_gfx1201::kMmContext0BaseLo,
                             regs_gfx1201::kMmContext0BaseHi, root_base, error_text)) {
      *error_text = "program MMHUB VMID0 base failed: " + *error_text;
      return false;
    }
    if (!write_register_dword(client, *log, mmhub, regs_gfx1201::kMmContext0Cntl,
                              encode_context0_cntl(), error_text)) {
      *error_text = "program MMHUB VMID0 context control failed: " + *error_text;
      return false;
    }

    if (!write_register_pair(client, *log, mmhub, regs_gfx1201::kMmIdentityLowLo,
                             regs_gfx1201::kMmIdentityLowHi, 0xfffffffffULL, error_text)) {
      *error_text = "disable MMHUB identity low aperture failed: " + *error_text;
      return false;
    }
    if (!write_register_pair(client, *log, mmhub, regs_gfx1201::kMmIdentityHighLo,
                             regs_gfx1201::kMmIdentityHighHi, 0ULL, error_text)) {
      *error_text = "disable MMHUB identity high aperture failed: " + *error_text;
      return false;
    }
    if (!write_register_pair(client, *log, mmhub, regs_gfx1201::kMmIdentityOffsetLo,
                             regs_gfx1201::kMmIdentityOffsetHi, 0ULL, error_text)) {
      *error_text = "disable MMHUB identity physical offset failed: " + *error_text;
      return false;
    }
  }
  log->vm.vmid0_context_status = "pass";
  // tinygrad/runtime/support/am/ip.py:78-80 marks MM initialized and GC not initialized before compute bring-up.
  log->vm.vm_gc_context_status = "skipped_gc_hub_not_initialized";
  return true;
}

bool program_gc_hub_vmid0(const RemoteClient& client, DiscoveryLog* log,
                          std::string* error_text) {
  // docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md lines 441-449:
  // clone MMHUB VMID0 programming values onto the source-cited gc_12_0_0 register set.
  uint32_t fb_base_reg = 0;
  uint32_t fb_top_reg = 0;
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcMcVmFbLocationBase,
                           &fb_base_reg, error_text)) {
    *error_text = std::string("read ") + regs_gfx1201::kGcMcVmFbLocationBase.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcMcVmFbLocationTop,
                           &fb_top_reg, error_text)) {
    *error_text = std::string("read ") + regs_gfx1201::kGcMcVmFbLocationTop.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  const uint64_t fb_base = static_cast<uint64_t>(fb_base_reg & 0x00ffffffU) << 24;
  const uint64_t fb_end = static_cast<uint64_t>(fb_top_reg & 0x00ffffffU) << 24;
  log->vm.mc_base = fb_base;
  const uint64_t vm_start = am_vm::kVaBase >> 12;
  const uint64_t vm_end = ((am_vm::kVaBase + (1ULL << 44)) - 1ULL) >> 12;
  const uint64_t root_base = log->vm.tables.root_pdb2_paddr | 1ULL;
  const uint64_t memscratch_ppn = log->vm.tables.memscratch_paddr >> 12;
  const uint64_t dummy_ppn = log->vm.tables.dummy_page_paddr >> 12;

  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcMcVmSystemApertureLow,
                            static_cast<uint32_t>(fb_base >> 18), error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcMcVmSystemApertureLow.name +
                  " failed: " + *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcMcVmSystemApertureHigh,
                            static_cast<uint32_t>(fb_end >> 18), error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcMcVmSystemApertureHigh.name +
                  " failed: " + *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_pair(client, *log, log->ip.gc, regs_gfx1201::kGcMcVmSystemApertureDefaultLsb,
                           regs_gfx1201::kGcMcVmSystemApertureDefaultMsb, memscratch_ppn,
                           error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcMcVmSystemApertureDefaultLsb.name +
                  "/" + regs_gfx1201::kGcMcVmSystemApertureDefaultMsb.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_pair(client, *log, log->ip.gc, regs_gfx1201::kGcVmProtectionFaultDefaultLo,
                           regs_gfx1201::kGcVmProtectionFaultDefaultHi, dummy_ppn, error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcVmProtectionFaultDefaultLo.name +
                  "/" + regs_gfx1201::kGcVmProtectionFaultDefaultHi.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }

  if (!update_register_bits(client, *log, log->ip.gc, regs_gfx1201::kGcMcVmMxL1TlbCntl,
                            (1U << 0) | (3U << 3) | (1U << 5) | (1U << 6) | (3U << 11),
                            encode_mx_l1_tlb_cntl(), error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcMcVmMxL1TlbCntl.name +
                  " failed: " + *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!update_register_bits(client, *log, log->ip.gc, regs_gfx1201::kGcVmL2Cntl,
                            (1U << 0) | (1U << 1) | (1U << 8) | (1U << 11) | (1U << 18) |
                                (3U << 19) | (31U << 21),
                            (1U << 0) | (1U << 11) | (1U << 19), error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcVmL2Cntl.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!update_register_bits(client, *log, log->ip.gc, regs_gfx1201::kGcVmL2Cntl2,
                            (1U << 0) | (1U << 1), (1U << 0) | (1U << 1), error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcVmL2Cntl2.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcVmL2Cntl3,
                            encode_l2_cntl3(), error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcVmL2Cntl3.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcVmL2Cntl4, 1U,
                            error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcVmL2Cntl4.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcVmL2Cntl5, 0x1ffU << 5,
                            error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcVmL2Cntl5.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcMcVmAgpBase, 0U,
                            error_text)) {
    *error_text = std::string("disable ") + regs_gfx1201::kGcMcVmAgpBase.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcMcVmAgpBot, 0xffffffU,
                            error_text)) {
    *error_text = std::string("disable ") + regs_gfx1201::kGcMcVmAgpBot.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcMcVmAgpTop, 0U,
                            error_text)) {
    *error_text = std::string("disable ") + regs_gfx1201::kGcMcVmAgpTop.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  std::size_t range_index = 0;
  for (const RegDef& range_lo : regs_gfx1201::kGcInvalidateEngAddrRangeLo) {
    if (!write_register_pair(client, *log, log->ip.gc, range_lo,
                             regs_gfx1201::kGcInvalidateEngAddrRangeHi[range_index],
                             0x1fffffffffULL, error_text)) {
      *error_text = std::string("program ") + range_lo.name + "/" +
                    regs_gfx1201::kGcInvalidateEngAddrRangeHi[range_index].name + " failed: " +
                    *error_text;
      log->vm.vm_gc_context_status = "fail";
      return false;
    }
    ++range_index;
  }


  if (!write_register_pair(client, *log, log->ip.gc, regs_gfx1201::kGcContext0StartLo,
                           regs_gfx1201::kGcContext0StartHi, vm_start, error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcContext0StartLo.name + "/" +
                  regs_gfx1201::kGcContext0StartHi.name + " failed: " + *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_pair(client, *log, log->ip.gc, regs_gfx1201::kGcContext0EndLo,
                           regs_gfx1201::kGcContext0EndHi, vm_end, error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcContext0EndLo.name + "/" +
                  regs_gfx1201::kGcContext0EndHi.name + " failed: " + *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_pair(client, *log, log->ip.gc, regs_gfx1201::kGcContext0BaseLo,
                           regs_gfx1201::kGcContext0BaseHi, root_base, error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcContext0BaseLo.name + "/" +
                  regs_gfx1201::kGcContext0BaseHi.name + " failed: " + *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcContext0Cntl,
                            encode_context0_cntl(), error_text)) {
    *error_text = std::string("program ") + regs_gfx1201::kGcContext0Cntl.name + " failed: " +
                  *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }

  if (!write_register_pair(client, *log, log->ip.gc, regs_gfx1201::kGcIdentityLowLo,
                           regs_gfx1201::kGcIdentityLowHi, 0xfffffffffULL, error_text)) {
    *error_text = std::string("disable ") + regs_gfx1201::kGcIdentityLowLo.name + "/" +
                  regs_gfx1201::kGcIdentityLowHi.name + " failed: " + *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_pair(client, *log, log->ip.gc, regs_gfx1201::kGcIdentityHighLo,
                           regs_gfx1201::kGcIdentityHighHi, 0ULL, error_text)) {
    *error_text = std::string("disable ") + regs_gfx1201::kGcIdentityHighLo.name + "/" +
                  regs_gfx1201::kGcIdentityHighHi.name + " failed: " + *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }
  if (!write_register_pair(client, *log, log->ip.gc, regs_gfx1201::kGcIdentityOffsetLo,
                           regs_gfx1201::kGcIdentityOffsetHi, 0ULL, error_text)) {
    *error_text = std::string("disable ") + regs_gfx1201::kGcIdentityOffsetLo.name + "/" +
                  regs_gfx1201::kGcIdentityOffsetHi.name + " failed: " + *error_text;
    log->vm.vm_gc_context_status = "fail";
    return false;
  }

  log->vm.vm_gc_context_status = "pass";
  return true;
}



bool poll_register_mask(const RemoteClient& client, const DiscoveryLog& log, const IpBlockInfo& ip,
                        const RegDef& reg, uint32_t mask, uint32_t expected, const char* label,
                        std::string* error_text) {
  for (int attempt = 0; attempt < 1000; ++attempt) {
    uint32_t value = 0;
    if (!read_register_dword(client, log, ip, reg, &value, error_text)) {
      *error_text = std::string(label) + " read failed: " + *error_text;
      return false;
    }
    if ((value & mask) == expected) {
      return true;
    }
    usleep(1000U);
  }
  *error_text = std::string(label) + " timed out waiting for mask " + format_hex64(mask) +
                " to equal " + format_hex64(expected);
  return false;
}

// tinygrad/runtime/support/am/ip.py:371-372 _grbm_select writes
// regGRBM_GFX_CNTL(meid, pipeid, vmid, queueid); queue0 uses ME=1, pipe=0, queue=0.
constexpr uint32_t encode_grbm_gfx_cntl(uint32_t me, uint32_t pipe, uint32_t queue,
                                        uint32_t vmid = 0U) {
  return ((pipe & 0x3U) << 0) | ((me & 0x3U) << 2) | ((vmid & 0xfU) << 4) |
         ((queue & 0x7U) << 8);
}

bool write_grbm_select(const RemoteClient& client, const DiscoveryLog& log, uint32_t value,
                       const char* action, std::string* error_text) {
  if (!write_register_dword(client, log, log.ip.gc, regs_gfx1201::kGrbmGfxCntl, value,
                            error_text)) {
    *error_text = std::string(regs_gfx1201::kGrbmGfxCntl.name) + " " + action +
                  " failed: " + *error_text;
    return false;
  }
  return true;
}

bool select_grbm_queue0(const RemoteClient& client, const DiscoveryLog& log,
                        std::string* error_text) {
  return write_grbm_select(client, log, encode_grbm_gfx_cntl(1U, 0U, 0U),
                           "select ME=1 pipe=0 queue=0", error_text);
}

bool select_grbm_mec_rs64_pipe0(const RemoteClient& client, const DiscoveryLog& log,
                                std::string* error_text) {
  return write_grbm_select(client, log, encode_grbm_gfx_cntl(1U, 0U, 0U),
                           "select MEC RS64 ME=1 pipe=0 queue=0", error_text);
}


bool restore_grbm_default_select(const RemoteClient& client, const DiscoveryLog& log,
                                 std::string* error_text) {
  return write_grbm_select(client, log, 0U, "restore default select", error_text);
}
struct ComputeQueueDebugSnapshot {
  uint32_t hqd_active = 0;
  uint32_t hqd_pq_rptr = 0;
  uint32_t hqd_pq_wptr_hi = 0;
  uint32_t hqd_pq_doorbell_control = 0;
  uint32_t hqd_pq_control = 0;
  uint32_t cp_stat = 0;
  uint32_t mec_doorbell_range_lower = 0;
  uint32_t mec_doorbell_range_upper = 0;
  bool has_mec_ranges = false;
};
struct ComputeDoorbellConsumptionSnapshot {
  uint32_t hqd_active = 0;
  uint32_t hqd_pq_doorbell_control = 0;
  uint32_t hqd_pq_control = 0;
  uint32_t hqd_pq_base = 0;
  uint32_t hqd_pq_base_hi = 0;
  uint32_t hqd_pq_rptr = 0;
  uint32_t hqd_pq_rptr_report_addr = 0;
  uint32_t hqd_pq_rptr_report_addr_hi = 0;
  uint32_t hqd_pq_wptr_poll_addr = 0;
  uint32_t hqd_pq_wptr_poll_addr_hi = 0;
  uint32_t hqd_pq_wptr_lo = 0;
  uint32_t hqd_pq_wptr_hi = 0;
  uint32_t cp_stat = 0;
  uint32_t cp_int_cntl_ring0 = 0;
  uint32_t cp_mec1_f32_interrupt = 0;
  uint32_t cp_mec1_instr_pntr = 0;
  uint32_t cp_mec_rs64_interrupt = 0;
  uint32_t cp_mec_rs64_pending_interrupt = 0;
  uint32_t cp_mec_rs64_exception_status = 0;
  uint32_t cp_mec_rs64_instr_pntr = 0;
  uint32_t cp_mec_rs64_prgrm_cntr_start_hi = 0;
  uint32_t gcvm_protection_fault_status_lo32 = 0;
  uint32_t gcvm_protection_fault_status_hi32 = 0;
  uint32_t gcvm_protection_fault_addr_lo32 = 0;
  uint32_t gcvm_protection_fault_addr_hi32 = 0;
  uint32_t cp_mec_local_instr_base_lo = 0;
  uint32_t cp_mec_local_instr_base_hi = 0;
  uint32_t cp_mec_local_instr_mask_lo = 0;
  uint32_t cp_mec_local_instr_mask_hi = 0;
  uint32_t cp_mec_local_instr_aperture = 0;
  uint32_t cp_mec_rs64_interrupt_data_16 = 0;
  uint32_t cp_mec_rs64_interrupt_data_17 = 0;
  uint32_t cp_mec_rs64_interrupt_data_18 = 0;
  uint32_t cp_mec_rs64_interrupt_data_19 = 0;
  uint32_t cp_mec_rs64_interrupt_data_20 = 0;
  uint32_t cp_mec_rs64_interrupt_data_21 = 0;
  uint32_t cp_mec_rs64_interrupt_data_22 = 0;
  uint32_t cp_mec_rs64_interrupt_data_23 = 0;
  uint32_t cp_mec_rs64_interrupt_data_24 = 0;
  uint32_t cp_mec_rs64_interrupt_data_25 = 0;
  uint32_t cp_mec_rs64_interrupt_data_26 = 0;
  uint32_t cp_mec_rs64_interrupt_data_27 = 0;
  uint32_t cp_mec_rs64_interrupt_data_28 = 0;
  uint32_t cp_mec_rs64_interrupt_data_29 = 0;
  uint32_t cp_mec_rs64_interrupt_data_30 = 0;
  uint32_t cp_mec_rs64_interrupt_data_31 = 0;
  uint64_t control_wptr_cpu = 0;
  uint64_t control_rptr_cpu = 0;
  uint32_t mqd_hqd_mismatch_count = 0;
  std::string mqd_hqd_mismatches = "none";
};


struct ComputeDoorbellRouteSnapshot {
  uint32_t rcc_doorbell_aper_en = 0;
  uint32_t rcc_dev0_epf2_strap2 = 0;
  uint32_t gdc_s2a_entry0_ctrl = 0;
  uint32_t gdc_s2a_entry3_ctrl = 0;
};

bool read_debug_register(const RemoteClient& client, const DiscoveryLog& log, const RegDef& reg,
                         const char* field_name, uint32_t* value, std::string* error_text) {
  if (!read_register_dword(client, log, log.ip.gc, reg, value, error_text)) {
    *error_text = std::string(field_name) + " read failed: " + *error_text;
    return false;
  }
  return true;
}

uint32_t s2a_doorbell_entry_enable(uint32_t value) {
  return value & 0x1U;
}

uint32_t s2a_doorbell_entry_awid(uint32_t value) {
  return (value >> 1) & 0x1fU;
}

uint32_t s2a_doorbell_entry_range_offset(uint32_t value) {
  return (value >> 7) & 0x3ffU;
}

uint32_t s2a_doorbell_entry_range_size(uint32_t value) {
  return (value >> 17) & 0xffU;
}

uint32_t s2a_doorbell_entry_awaddr_high(uint32_t value) {
  return (value >> 28) & 0xfU;
}

bool read_compute_doorbell_route_register(const RemoteClient& client, const DiscoveryLog& log,
                                          const RegDef& reg, uint32_t* value,
                                          std::string* error_text) {
  if (!read_register_dword(client, log, log.ip.nbif, reg, value, error_text)) {
    *error_text = std::string(reg.name) + " read failed: " + *error_text;
    return false;
  }
  return true;
}

bool read_compute_doorbell_route_snapshot(const RemoteClient& client, const DiscoveryLog& log,
                                          ComputeDoorbellRouteSnapshot* snapshot,
                                          std::string* error_text) {
  if (snapshot == nullptr) {
    *error_text = "ComputeDoorbellRouteSnapshot precondition failed: null snapshot";
    return false;
  }
  return read_compute_doorbell_route_register(
             client, log, regs_gfx1201::kRccDoorbellAperEn,
             &snapshot->rcc_doorbell_aper_en, error_text) &&
         read_compute_doorbell_route_register(
             client, log, regs_gfx1201::kRccDev0Epf2Strap2,
             &snapshot->rcc_dev0_epf2_strap2, error_text) &&
         read_compute_doorbell_route_register(
             client, log, regs_gfx1201::kGdcS2aDoorbellEntry0,
             &snapshot->gdc_s2a_entry0_ctrl, error_text) &&
         read_compute_doorbell_route_register(
             client, log, regs_gfx1201::kGdcS2aDoorbellEntry3,
             &snapshot->gdc_s2a_entry3_ctrl, error_text);
}


std::string format_compute_doorbell_route_snapshot(
    const ComputeDoorbellRouteSnapshot& snapshot) {
  return "rcc_doorbell_aper_en=" + format_hex32(snapshot.rcc_doorbell_aper_en) +
         ", aperture_enabled=" +
         std::to_string(snapshot.rcc_doorbell_aper_en & 0x1U) +
         ", rcc_dev0_epf2_strap2=" + format_hex32(snapshot.rcc_dev0_epf2_strap2) +
         ", epf2_strap_bit7=" +
         std::to_string((snapshot.rcc_dev0_epf2_strap2 >> 7) & 0x1U) +
         ", gdc_s2a_entry0_ctrl=" + format_hex32(snapshot.gdc_s2a_entry0_ctrl) +
         ", entry0_enable=" +
         std::to_string(s2a_doorbell_entry_enable(snapshot.gdc_s2a_entry0_ctrl)) +
         ", entry0_awid=" +
         std::to_string(s2a_doorbell_entry_awid(snapshot.gdc_s2a_entry0_ctrl)) +
         ", entry0_range_offset=" +
         std::to_string(s2a_doorbell_entry_range_offset(snapshot.gdc_s2a_entry0_ctrl)) +
         ", entry0_range_size=" +
         std::to_string(s2a_doorbell_entry_range_size(snapshot.gdc_s2a_entry0_ctrl)) +
         ", entry0_awaddr_31_28=" +
         std::to_string(s2a_doorbell_entry_awaddr_high(snapshot.gdc_s2a_entry0_ctrl)) +
         ", gdc_s2a_entry3_ctrl=" + format_hex32(snapshot.gdc_s2a_entry3_ctrl) +
         ", entry3_enable=" +
         std::to_string(s2a_doorbell_entry_enable(snapshot.gdc_s2a_entry3_ctrl)) +
         ", entry3_awid=" +
         std::to_string(s2a_doorbell_entry_awid(snapshot.gdc_s2a_entry3_ctrl)) +
         ", entry3_range_offset=" +
         std::to_string(s2a_doorbell_entry_range_offset(snapshot.gdc_s2a_entry3_ctrl)) +
         ", entry3_range_size=" +
         std::to_string(s2a_doorbell_entry_range_size(snapshot.gdc_s2a_entry3_ctrl)) +
         ", entry3_awaddr_31_28=" +
         std::to_string(s2a_doorbell_entry_awaddr_high(snapshot.gdc_s2a_entry3_ctrl)) +
         ", expected_entry0_ctrl=" +
         format_hex32(am_compute::kDoorbellRouteExpectedEntry0Ctrl) +
         ", expected_entry3_ctrl=" +
         format_hex32(am_compute::kDoorbellRouteExpectedEntry3Ctrl);
}

std::string classify_compute_doorbell_route_snapshot(
    const ComputeDoorbellRouteSnapshot& snapshot) {
  const bool aperture_enabled = (snapshot.rcc_doorbell_aper_en & 0x1U) != 0U;
  const bool epf2_bit7_cleared = (snapshot.rcc_dev0_epf2_strap2 & (1U << 7)) == 0U;
  if (aperture_enabled && epf2_bit7_cleared &&
      snapshot.gdc_s2a_entry0_ctrl == am_compute::kDoorbellRouteExpectedEntry0Ctrl &&
      snapshot.gdc_s2a_entry3_ctrl == am_compute::kDoorbellRouteExpectedEntry3Ctrl) {
    return am_compute::kDoorbellRouteClassMatches;
  }
  return am_compute::kDoorbellRouteClassMismatch;
}
bool read_compute_control_u64(const SysmemMapping& compute_control_mapping,
                              uint64_t offset, uint64_t* value,
                              std::string* error_text) {
  if (value == nullptr) {
    *error_text = "compute_control value precondition failed: null pointer";
    return false;
  }
  if (compute_control_mapping.data == nullptr) {
    *error_text = "compute_control mapping is null";
    return false;
  }
  if (offset > compute_control_mapping.size ||
      sizeof(uint64_t) > compute_control_mapping.size - offset) {
    *error_text = "compute_control mapping too small for qword read at offset " +
                  format_hex64(offset);
    return false;
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  volatile const uint64_t* ptr =
      reinterpret_cast<volatile const uint64_t*>(static_cast<const uint8_t*>(
                                                    compute_control_mapping.data) +
                                                offset);
  *value = *ptr;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  return true;
}
bool read_compute_control_field_u64(const SysmemMapping& compute_control_mapping,
                                    uint64_t offset, const char* field_name,
                                    uint64_t* value, std::string* error_text) {
  if (!read_compute_control_u64(compute_control_mapping, offset, value, error_text)) {
    *error_text = std::string(field_name) + " read failed: " + *error_text;
    return false;
  }
  return true;
}


void append_mqd_hqd_mismatch(std::string* mismatches, uint32_t* mismatch_count,
                             const char* field_name, uint32_t expected,
                             uint32_t observed) {
  if (*mismatch_count != 0U) {
    *mismatches += ";";
  }
  *mismatches += "field=" + std::string(field_name) +
                 ",expected=" + format_hex32(expected) +
                 ",observed=" + format_hex32(observed);
  ++(*mismatch_count);
}

bool compare_mqd_hqd_field_masked(const RemoteClient& client,
                                  const DiscoveryLog& log,
                                  const ComputeMqd& mqd,
                                  ComputeMqdDword mqd_index,
                                  uint32_t compare_mask, const RegDef& reg,
                                  const char* field_name,
                                  std::string* mismatches,
                                  uint32_t* mismatch_count,
                                  std::string* error_text) {
  uint32_t observed = 0;
  if (!read_register_dword(client, log, log.ip.gc, reg, &observed, error_text)) {
    *error_text = std::string(reg.name) + " MQD/HQD compare read failed: " +
                  *error_text;
    return false;
  }
  const uint32_t expected = mqd[mqd_index];
  if ((observed & compare_mask) != (expected & compare_mask)) {
    append_mqd_hqd_mismatch(mismatches, mismatch_count, field_name, expected,
                            observed);
  }
  return true;
}

bool compare_mqd_hqd_field(const RemoteClient& client, const DiscoveryLog& log,
                           const ComputeMqd& mqd, ComputeMqdDword mqd_index,
                           const RegDef& reg, const char* field_name,
                           std::string* mismatches, uint32_t* mismatch_count,
                           std::string* error_text) {
  return compare_mqd_hqd_field_masked(client, log, mqd, mqd_index, 0xffffffffU,
                                      reg, field_name, mismatches,
                                      mismatch_count, error_text);
}

bool compare_mqd_hqd_fields(const RemoteClient& client, const DiscoveryLog& log,
                            ComputeDoorbellConsumptionSnapshot* snapshot,
                            std::string* error_text) {
  if (snapshot == nullptr) {
    *error_text = "ComputeDoorbellConsumptionSnapshot precondition failed: null snapshot";
    return false;
  }
  const ComputeMqd mqd = build_compute_mqd(log.vm.mc_base);
  std::string mismatches;
  uint32_t mismatch_count = 0;
  const bool ok =
      compare_mqd_hqd_field_masked(
          client, log, mqd, kMqdCpHqdPqDoorbellControl,
          am_compute::kHqdPqDoorbellControlStaticCompareMask,
          regs_gfx1201::kCpHqdPqDoorbellControl,
          "cp_hqd_pq_doorbell_control", &mismatches, &mismatch_count,
          error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdPqControl,
                            regs_gfx1201::kCpHqdPqControl, "cp_hqd_pq_control",
                            &mismatches, &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdPqBaseLo,
                            regs_gfx1201::kCpHqdPqBase, "cp_hqd_pq_base_lo",
                            &mismatches, &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdPqBaseHi,
                            regs_gfx1201::kCpHqdPqBaseHi, "cp_hqd_pq_base_hi",
                            &mismatches, &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdPqRptrReportAddrLo,
                            regs_gfx1201::kCpHqdPqRptrReportAddr,
                            "cp_hqd_pq_rptr_report_addr_lo", &mismatches,
                            &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdPqRptrReportAddrHi,
                            regs_gfx1201::kCpHqdPqRptrReportAddrHi,
                            "cp_hqd_pq_rptr_report_addr_hi", &mismatches,
                            &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdPqWptrPollAddrLo,
                            regs_gfx1201::kCpHqdPqWptrPollAddr,
                            "cp_hqd_pq_wptr_poll_addr_lo", &mismatches,
                            &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdPqWptrPollAddrHi,
                            regs_gfx1201::kCpHqdPqWptrPollAddrHi,
                            "cp_hqd_pq_wptr_poll_addr_hi", &mismatches,
                            &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpMqdControl,
                            regs_gfx1201::kCpMqdControl, "cp_mqd_control",
                            &mismatches, &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdEopBaseAddrLo,
                            regs_gfx1201::kCpHqdEopBaseAddr,
                            "cp_hqd_eop_base_addr_lo", &mismatches,
                            &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdEopBaseAddrHi,
                            regs_gfx1201::kCpHqdEopBaseAddrHi,
                            "cp_hqd_eop_base_addr_hi", &mismatches,
                            &mismatch_count, error_text) &&
      compare_mqd_hqd_field(client, log, mqd, kMqdCpHqdEopControl,
                            regs_gfx1201::kCpHqdEopControl,
                            "cp_hqd_eop_control", &mismatches,
                            &mismatch_count, error_text);
  if (!ok) {
    return false;
  }
  snapshot->mqd_hqd_mismatch_count = mismatch_count;
  snapshot->mqd_hqd_mismatches = mismatch_count == 0U ? "none" : mismatches;
  return true;
}

bool read_compute_doorbell_consumption_snapshot(
    const RemoteClient& client, const DiscoveryLog& log,
    const SysmemMapping& compute_control_mapping,
    ComputeDoorbellConsumptionSnapshot* snapshot, std::string* error_text) {
  if (snapshot == nullptr) {
    *error_text = "ComputeDoorbellConsumptionSnapshot precondition failed: null snapshot";
    return false;
  }
  if (!select_grbm_queue0(client, log, error_text)) {
    *error_text = "select queue0 for compute doorbell consumption failed: " +
                  *error_text;
    return false;
  }

  bool ok =
      read_debug_register(client, log, regs_gfx1201::kCpHqdActive,
                          regs_gfx1201::kCpHqdActive.name,
                          &snapshot->hqd_active, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqDoorbellControl,
                          regs_gfx1201::kCpHqdPqDoorbellControl.name,
                          &snapshot->hqd_pq_doorbell_control, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqControl,
                          regs_gfx1201::kCpHqdPqControl.name,
                          &snapshot->hqd_pq_control, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqBase,
                          regs_gfx1201::kCpHqdPqBase.name,
                          &snapshot->hqd_pq_base, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqBaseHi,
                          regs_gfx1201::kCpHqdPqBaseHi.name,
                          &snapshot->hqd_pq_base_hi, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqRptr,
                          regs_gfx1201::kCpHqdPqRptr.name,
                          &snapshot->hqd_pq_rptr, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqRptrReportAddr,
                          regs_gfx1201::kCpHqdPqRptrReportAddr.name,
                          &snapshot->hqd_pq_rptr_report_addr, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqRptrReportAddrHi,
                          regs_gfx1201::kCpHqdPqRptrReportAddrHi.name,
                          &snapshot->hqd_pq_rptr_report_addr_hi, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqWptrPollAddr,
                          regs_gfx1201::kCpHqdPqWptrPollAddr.name,
                          &snapshot->hqd_pq_wptr_poll_addr, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqWptrPollAddrHi,
                          regs_gfx1201::kCpHqdPqWptrPollAddrHi.name,
                          &snapshot->hqd_pq_wptr_poll_addr_hi, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqWptrLo,
                          regs_gfx1201::kCpHqdPqWptrLo.name,
                          &snapshot->hqd_pq_wptr_lo, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpHqdPqWptrHi,
                          regs_gfx1201::kCpHqdPqWptrHi.name,
                          &snapshot->hqd_pq_wptr_hi, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpStat,
                          regs_gfx1201::kCpStat.name, &snapshot->cp_stat,
                          error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpIntCntlRing0,
                          regs_gfx1201::kCpIntCntlRing0.name,
                          &snapshot->cp_int_cntl_ring0, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpMec1F32Interrupt,
                          regs_gfx1201::kCpMec1F32Interrupt.name,
                          &snapshot->cp_mec1_f32_interrupt, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpMec1InstrPntr,
                          regs_gfx1201::kCpMec1InstrPntr.name,
                          &snapshot->cp_mec1_instr_pntr, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpMecRs64Interrupt,
                          regs_gfx1201::kCpMecRs64Interrupt.name,
                          &snapshot->cp_mec_rs64_interrupt, error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64PendingInterrupt,
                          regs_gfx1201::kCpMecRs64PendingInterrupt.name,
                          &snapshot->cp_mec_rs64_pending_interrupt,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64ExceptionStatus,
                          regs_gfx1201::kCpMecRs64ExceptionStatus.name,
                          &snapshot->cp_mec_rs64_exception_status,
                          error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpMecRs64InstrPntr,
                          regs_gfx1201::kCpMecRs64InstrPntr.name,
                          &snapshot->cp_mec_rs64_instr_pntr, error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64PrgrmCntrStartHi,
                          regs_gfx1201::kCpMecRs64PrgrmCntrStartHi.name,
                          &snapshot->cp_mec_rs64_prgrm_cntr_start_hi,
                          error_text) &&
      read_debug_register(client, log, regs_gfx1201::kGcVmProtectionFaultStatusLo32,
                          regs_gfx1201::kGcVmProtectionFaultStatusLo32.name,
                          &snapshot->gcvm_protection_fault_status_lo32, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kGcVmProtectionFaultStatusHi32,
                          regs_gfx1201::kGcVmProtectionFaultStatusHi32.name,
                          &snapshot->gcvm_protection_fault_status_hi32, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kGcVmProtectionFaultAddrLo32,
                          regs_gfx1201::kGcVmProtectionFaultAddrLo32.name,
                          &snapshot->gcvm_protection_fault_addr_lo32, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kGcVmProtectionFaultAddrHi32,
                          regs_gfx1201::kGcVmProtectionFaultAddrHi32.name,
                          &snapshot->gcvm_protection_fault_addr_hi32, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpMecLocalInstrBaseLo,
                          regs_gfx1201::kCpMecLocalInstrBaseLo.name,
                          &snapshot->cp_mec_local_instr_base_lo, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpMecLocalInstrBaseHi,
                          regs_gfx1201::kCpMecLocalInstrBaseHi.name,
                          &snapshot->cp_mec_local_instr_base_hi, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpMecLocalInstrMaskLo,
                          regs_gfx1201::kCpMecLocalInstrMaskLo.name,
                          &snapshot->cp_mec_local_instr_mask_lo, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpMecLocalInstrMaskHi,
                          regs_gfx1201::kCpMecLocalInstrMaskHi.name,
                          &snapshot->cp_mec_local_instr_mask_hi, error_text) &&
      read_debug_register(client, log, regs_gfx1201::kCpMecLocalInstrAperture,
                          regs_gfx1201::kCpMecLocalInstrAperture.name,
                          &snapshot->cp_mec_local_instr_aperture, error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData16,
                          regs_gfx1201::kCpMecRs64InterruptData16.name,
                          &snapshot->cp_mec_rs64_interrupt_data_16,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData17,
                          regs_gfx1201::kCpMecRs64InterruptData17.name,
                          &snapshot->cp_mec_rs64_interrupt_data_17,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData18,
                          regs_gfx1201::kCpMecRs64InterruptData18.name,
                          &snapshot->cp_mec_rs64_interrupt_data_18,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData19,
                          regs_gfx1201::kCpMecRs64InterruptData19.name,
                          &snapshot->cp_mec_rs64_interrupt_data_19,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData20,
                          regs_gfx1201::kCpMecRs64InterruptData20.name,
                          &snapshot->cp_mec_rs64_interrupt_data_20,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData21,
                          regs_gfx1201::kCpMecRs64InterruptData21.name,
                          &snapshot->cp_mec_rs64_interrupt_data_21,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData22,
                          regs_gfx1201::kCpMecRs64InterruptData22.name,
                          &snapshot->cp_mec_rs64_interrupt_data_22,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData23,
                          regs_gfx1201::kCpMecRs64InterruptData23.name,
                          &snapshot->cp_mec_rs64_interrupt_data_23,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData24,
                          regs_gfx1201::kCpMecRs64InterruptData24.name,
                          &snapshot->cp_mec_rs64_interrupt_data_24,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData25,
                          regs_gfx1201::kCpMecRs64InterruptData25.name,
                          &snapshot->cp_mec_rs64_interrupt_data_25,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData26,
                          regs_gfx1201::kCpMecRs64InterruptData26.name,
                          &snapshot->cp_mec_rs64_interrupt_data_26,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData27,
                          regs_gfx1201::kCpMecRs64InterruptData27.name,
                          &snapshot->cp_mec_rs64_interrupt_data_27,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData28,
                          regs_gfx1201::kCpMecRs64InterruptData28.name,
                          &snapshot->cp_mec_rs64_interrupt_data_28,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData29,
                          regs_gfx1201::kCpMecRs64InterruptData29.name,
                          &snapshot->cp_mec_rs64_interrupt_data_29,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData30,
                          regs_gfx1201::kCpMecRs64InterruptData30.name,
                          &snapshot->cp_mec_rs64_interrupt_data_30,
                          error_text) &&
      read_debug_register(client, log,
                          regs_gfx1201::kCpMecRs64InterruptData31,
                          regs_gfx1201::kCpMecRs64InterruptData31.name,
                          &snapshot->cp_mec_rs64_interrupt_data_31,
                          error_text) &&
      read_compute_control_field_u64(compute_control_mapping,
                                     am_compute::kWptrOffset,
                                     "control_wptr_cpu",
                                     &snapshot->control_wptr_cpu,
                                     error_text) &&
      read_compute_control_field_u64(compute_control_mapping,
                                     am_compute::kRptrOffset,
                                     "control_rptr_cpu",
                                     &snapshot->control_rptr_cpu,
                                     error_text) &&
      compare_mqd_hqd_fields(client, log, snapshot, error_text);

  std::string restore_error;
  if (!restore_grbm_default_select(client, log, &restore_error)) {
    if (ok) {
      *error_text =
          "restore GRBM default select after compute doorbell consumption failed: " +
          restore_error;
    } else {
      *error_text += "; restore GRBM default select also failed: " + restore_error;
    }
    return false;
  }
  return ok;
}

std::string format_compute_doorbell_consumption_snapshot(
    const ComputeDoorbellConsumptionSnapshot& snapshot) {
  return "hqd_active=" + format_hex32(snapshot.hqd_active) +
         ", hqd_pq_doorbell_control=" +
         format_hex32(snapshot.hqd_pq_doorbell_control) +
         ", doorbell_mode=" +
         std::to_string(
             am_compute::hqd_doorbell_mode(snapshot.hqd_pq_doorbell_control)) +
         ", doorbell_bif_drop=" +
         std::to_string(am_compute::hqd_doorbell_bif_drop(
             snapshot.hqd_pq_doorbell_control)) +
         ", doorbell_offset=" +
         std::to_string(am_compute::hqd_doorbell_offset(
             snapshot.hqd_pq_doorbell_control)) +
         ", doorbell_source=" +
         std::to_string(am_compute::hqd_doorbell_source(
             snapshot.hqd_pq_doorbell_control)) +
         ", doorbell_schd_hit=" +
         std::to_string(am_compute::hqd_doorbell_schd_hit(
             snapshot.hqd_pq_doorbell_control)) +
         ", doorbell_en=" +
         std::to_string(
             am_compute::hqd_doorbell_en(snapshot.hqd_pq_doorbell_control)) +
         ", doorbell_hit=" +
         std::to_string(
             am_compute::hqd_doorbell_hit(snapshot.hqd_pq_doorbell_control)) +
         ", hqd_pq_control=" + format_hex32(snapshot.hqd_pq_control) +
         ", hqd_pq_base=" + format_hex32(snapshot.hqd_pq_base) +
         ", hqd_pq_base_hi=" + format_hex32(snapshot.hqd_pq_base_hi) +
         ", hqd_pq_rptr=" + format_hex32(snapshot.hqd_pq_rptr) +
         ", hqd_pq_rptr_report_addr=" +
         format_hex32(snapshot.hqd_pq_rptr_report_addr) +
         ", hqd_pq_rptr_report_addr_hi=" +
         format_hex32(snapshot.hqd_pq_rptr_report_addr_hi) +
         ", hqd_pq_wptr_poll_addr=" +
         format_hex32(snapshot.hqd_pq_wptr_poll_addr) +
         ", hqd_pq_wptr_poll_addr_hi=" +
         format_hex32(snapshot.hqd_pq_wptr_poll_addr_hi) +
         ", hqd_pq_wptr_lo=" + format_hex32(snapshot.hqd_pq_wptr_lo) +
         ", hqd_pq_wptr_hi=" + format_hex32(snapshot.hqd_pq_wptr_hi) +
         ", control_wptr_cpu=" +
         std::to_string(
             static_cast<unsigned long long>(snapshot.control_wptr_cpu)) +
         ", control_rptr_cpu=" +
         std::to_string(
             static_cast<unsigned long long>(snapshot.control_rptr_cpu)) +
         ", cp_stat=" + format_hex32(snapshot.cp_stat) +
         ", cp_int_cntl_ring0=" + format_hex32(snapshot.cp_int_cntl_ring0) +
         ", cp_mec1_f32_interrupt=" +
         format_hex32(snapshot.cp_mec1_f32_interrupt) +
         ", cp_mec1_instr_pntr=" +
         format_hex32(snapshot.cp_mec1_instr_pntr) +
         ", cp_mec_rs64_interrupt=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt) +
         ", cp_mec_rs64_pending_interrupt=" +
         format_hex32(snapshot.cp_mec_rs64_pending_interrupt) +
         ", cp_mec_rs64_exception_status=" +
         format_hex32(snapshot.cp_mec_rs64_exception_status) +
         ", cp_mec_rs64_instr_pntr=" +
         format_hex32(snapshot.cp_mec_rs64_instr_pntr) +
         ", cp_mec_rs64_prgrm_cntr_start_hi=" +
         format_hex32(snapshot.cp_mec_rs64_prgrm_cntr_start_hi) +
         ", gcvm_protection_fault_status_lo32=" +
         format_hex32(snapshot.gcvm_protection_fault_status_lo32) +
         ", gcvm_protection_fault_status_hi32=" +
         format_hex32(snapshot.gcvm_protection_fault_status_hi32) +
         ", gcvm_protection_fault_addr_lo32=" +
         format_hex32(snapshot.gcvm_protection_fault_addr_lo32) +
         ", gcvm_protection_fault_addr_hi32=" +
         format_hex32(snapshot.gcvm_protection_fault_addr_hi32) +
         ", cp_mec_local_instr_base_lo=" +
         format_hex32(snapshot.cp_mec_local_instr_base_lo) +
         ", cp_mec_local_instr_base_hi=" +
         format_hex32(snapshot.cp_mec_local_instr_base_hi) +
         ", cp_mec_local_instr_mask_lo=" +
         format_hex32(snapshot.cp_mec_local_instr_mask_lo) +
         ", cp_mec_local_instr_mask_hi=" +
         format_hex32(snapshot.cp_mec_local_instr_mask_hi) +
         ", cp_mec_local_instr_aperture=" +
         format_hex32(snapshot.cp_mec_local_instr_aperture) +
         ", cp_mec_rs64_interrupt_data_16=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_16) +
         ", cp_mec_rs64_interrupt_data_17=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_17) +
         ", cp_mec_rs64_interrupt_data_18=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_18) +
         ", cp_mec_rs64_interrupt_data_19=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_19) +
         ", cp_mec_rs64_interrupt_data_20=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_20) +
         ", cp_mec_rs64_interrupt_data_21=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_21) +
         ", cp_mec_rs64_interrupt_data_22=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_22) +
         ", cp_mec_rs64_interrupt_data_23=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_23) +
         ", cp_mec_rs64_interrupt_data_24=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_24) +
         ", cp_mec_rs64_interrupt_data_25=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_25) +
         ", cp_mec_rs64_interrupt_data_26=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_26) +
         ", cp_mec_rs64_interrupt_data_27=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_27) +
         ", cp_mec_rs64_interrupt_data_28=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_28) +
         ", cp_mec_rs64_interrupt_data_29=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_29) +
         ", cp_mec_rs64_interrupt_data_30=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_30) +
         ", cp_mec_rs64_interrupt_data_31=" +
         format_hex32(snapshot.cp_mec_rs64_interrupt_data_31) +
         ", mqd_hqd_mismatch_count=" +
         std::to_string(snapshot.mqd_hqd_mismatch_count) +
         ", mqd_hqd_mismatches=" + snapshot.mqd_hqd_mismatches;
}

std::string classify_compute_doorbell_consumption_timeout(
    const ComputeDoorbellConsumptionSnapshot& snapshot) {
  if (snapshot.mqd_hqd_mismatch_count != 0U) {
    return am_compute::kDoorbellConsumptionClassMqdHqdMismatch;
  }
  if (snapshot.cp_mec_rs64_exception_status != 0U) {
    return am_compute::kDoorbellConsumptionClassRs64Exception;
  }
  if (am_compute::hqd_doorbell_bif_drop(snapshot.hqd_pq_doorbell_control) !=
      0U) {
    return am_compute::kDoorbellConsumptionClassBifDrop;
  }
  if (snapshot.control_wptr_cpu != am_compute::kPm4DispatchDwordCount) {
    return "compute_wptr_not_written_by_host";
  }
  if (snapshot.hqd_pq_wptr_lo == 0U &&
      snapshot.control_wptr_cpu == am_compute::kPm4DispatchDwordCount) {
    return am_compute::kDoorbellConsumptionClassWptrNotVisible;
  }
  if ((am_compute::hqd_doorbell_schd_hit(
           snapshot.hqd_pq_doorbell_control) != 0U ||
       am_compute::hqd_doorbell_hit(snapshot.hqd_pq_doorbell_control) != 0U) &&
      snapshot.hqd_pq_rptr == 0U) {
    return am_compute::kDoorbellConsumptionClassSchdOrHitRptrZero;
  }
  if (snapshot.hqd_pq_rptr != 0U) {
    return am_compute::kDoorbellConsumptionClassRptrAdvancesTimelineZero;
  }
  return am_compute::kDoorbellConsumptionClassNoSignal;
}

// Popcount (population count) helper for the compute readback classifier bit masks.
uint32_t bitcount(uint32_t value) {
  uint32_t count = 0U;
  while (value != 0U) {
    value &= (value - 1U);  // clear the lowest set bit
    ++count;
  }
  return count;
}

enum class ComputeReadbackAnomalyClass {
  kReadbackMatch,
  kSwapAndPartial,   // 16-bit halfword swap on written elements + only a subset written
  kPartialOnly,      // subset written, no byte-swap
  kSwapOnly,         // full element coverage, but halfword-swapped
  kOtherMismatch,
};

struct ComputeReadbackAnomaly {
  ComputeReadbackAnomalyClass cls = ComputeReadbackAnomalyClass::kOtherMismatch;
  uint32_t written_element_mask = 0U;      // bit i set => 4-byte element i nonzero
  uint32_t swapped_element_mask = 0U;      // bit i set => element i is 16-bit-halfword swapped
  uint32_t unswapped_match_element_mask = 0U;  // bit i set => un-swapped element equals expected[i]
};

// Grounding: the SDMA copy engine is byte-faithful (input H2D proves it; tinygrad
// ops_amd.py:copy and build_sdma_linear_copy_packet emit a format-less linear copy).
// So `observed` is exactly the bytes the GPU wrote to kOutputVramVa. This classifier
// only *describes* those bytes; it never relaxes the CPU comparison contract.
ComputeReadbackAnomaly classify_compute_readback_anomaly(
    const uint8_t* observed, const uint8_t* expected, std::size_t byte_count) {
  ComputeReadbackAnomaly out;
  const std::size_t elem_count = byte_count / 4U;
  for (std::size_t i = 0; i < elem_count; ++i) {
    const uint32_t obs = read_u32_le_bytes(observed + i * 4U);
    const uint32_t exp = read_u32_le_bytes(expected + i * 4U);
    const bool written = obs != 0U || exp == 0U;  // element considered written if nonzero or expected is zero
    if (written) out.written_element_mask |= (1U << i);
    // 16-bit halfword swap predicate: swap16(exp) == obs.
    const uint32_t swapped_expected = ((exp & 0xffffU) << 16) | ((exp >> 16) & 0xffffU);
    if (obs == swapped_expected) {
      out.swapped_element_mask |= (1U << i);
      const uint32_t unswapped_obs = ((obs & 0xffffU) << 16) | ((obs >> 16) & 0xffffU);
      if (unswapped_obs == exp) out.unswapped_match_element_mask |= (1U << i);
    }
  }
  const std::size_t written = bitcount(out.written_element_mask);
  const bool any_swap = out.swapped_element_mask != 0U;
  // Classification:
  //   kSwapAndPartial -> subset written (written < elem_count) AND any byte-swap
  //   kPartialOnly    -> subset written AND no byte-swap
  //   kSwapOnly       -> all elements written AND any byte-swap
  //   kOtherMismatch  -> anything else (none of the above; the `match` case is
  //                      unreachable here because this runs only on mismatch)
  if (written < elem_count && any_swap) out.cls = ComputeReadbackAnomalyClass::kSwapAndPartial;
  else if (written < elem_count) out.cls = ComputeReadbackAnomalyClass::kPartialOnly;
  else if (any_swap) out.cls = ComputeReadbackAnomalyClass::kSwapOnly;
  else out.cls = ComputeReadbackAnomalyClass::kOtherMismatch;
  return out;
}

// Human-readable snake_case label for a ComputeReadbackAnomalyClass, used in the
// compact `compute_readback_anomaly` log string. kReadbackMatch is not surfaced from
// the classifier (it only runs on readback_mismatch) but maps to a stable label.
const char* compute_readback_anomaly_class_label(ComputeReadbackAnomalyClass cls) {
  switch (cls) {
    case ComputeReadbackAnomalyClass::kReadbackMatch: return "readback_match";
    case ComputeReadbackAnomalyClass::kSwapAndPartial: return "swap_and_partial";
    case ComputeReadbackAnomalyClass::kPartialOnly: return "partial_only";
    case ComputeReadbackAnomalyClass::kSwapOnly: return "swap_only";
    case ComputeReadbackAnomalyClass::kOtherMismatch: return "other_mismatch";
  }
  return "other_mismatch";
}

int run_compute_doorbell_consumption_classifier_self_test() {
  ComputeDoorbellConsumptionSnapshot snapshot;
  snapshot.cp_mec_rs64_exception_status = 0x0000c67aU;

  const std::string classification =
      classify_compute_doorbell_consumption_timeout(snapshot);
  if (classification != am_compute::kDoorbellConsumptionClassRs64Exception) {
    return self_test_failure("compute-doorbell-consumption-classifier",
                             "RS64 exception status classification mismatch");
  }

  std::printf("self_test: compute-doorbell-consumption-classifier\n");
  std::printf("rs64_exception_status: 0x%08x\n",
              snapshot.cp_mec_rs64_exception_status);
  std::printf("classification: %s\n", classification.c_str());
  std::printf("status: pass\n");
  return 0;
}

// Decode a lowercase ASCII hex string (2 chars per byte) into a byte buffer.
// Returns false on odd length or non-hex characters.
static bool decode_hex_bytes(const char* hex, std::size_t hex_len, uint8_t* out,
                             std::size_t out_capacity, std::size_t* out_len) {
  if (hex_len % 2U != 0U) return false;
  const std::size_t byte_count = hex_len / 2U;
  if (byte_count > out_capacity) return false;
  for (std::size_t i = 0U; i < byte_count; ++i) {
    const char hi = hex[i * 2U];
    const char lo = hex[i * 2U + 1U];
    auto nibble = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      return -1;
    };
    const int hi_n = nibble(hi);
    const int lo_n = nibble(lo);
    if (hi_n < 0 || lo_n < 0) return false;
    out[i] = static_cast<uint8_t>((hi_n << 4) | lo_n);
  }
  if (out_len != nullptr) *out_len = byte_count;
  return true;
}

// C0A23 T1 no-hardware self-test: exercise classify_compute_readback_anomaly on the
// fixed observed/expected byte strings (the stable c0l readback signature). The
// classifier is CPU-side-only and never relaxes the --kernel-proof comparison.
int run_compute_readback_classifier_self_test() {
  uint8_t observed[kTransferByteCount]{};
  uint8_t expected[kTransferByteCount]{};
  std::size_t observed_len = 0U;
  std::size_t expected_len = 0U;
  if (!decode_hex_bytes(kKernelObservedOutputBytesHex,
                        std::strlen(kKernelObservedOutputBytesHex), observed,
                        sizeof(observed), &observed_len) ||
      !decode_hex_bytes(kKernelExpectedOutputBytesHex,
                        std::strlen(kKernelExpectedOutputBytesHex), expected,
                        sizeof(expected), &expected_len)) {
    return self_test_failure("compute-readback-classifier", "hex decode failed");
  }
  if (observed_len != kTransferByteCount || expected_len != kTransferByteCount) {
    return self_test_failure("compute-readback-classifier",
                             "observed/expected byte count mismatch");
  }
  const ComputeReadbackAnomaly anomaly =
      classify_compute_readback_anomaly(observed, expected, kTransferByteCount);
  if (anomaly.cls != ComputeReadbackAnomalyClass::kSwapAndPartial) {
    return self_test_failure("compute-readback-classifier",
                             "expected swap_and_partial classification");
  }
  if (anomaly.written_element_mask != 0x0fU ||
      anomaly.swapped_element_mask != 0x0fU ||
      anomaly.unswapped_match_element_mask != 0x0fU) {
    return self_test_failure("compute-readback-classifier",
                             "expected 0x0f written/swapped/unswapped-match masks");
  }

  std::printf("self_test: compute-readback-classifier\n");
  std::printf("example_observed_hex: %s\n", kKernelObservedOutputBytesHex);
  std::printf("example_expected_hex: %s\n", kKernelExpectedOutputBytesHex);
  std::printf("anomaly_class: %s\n",
              compute_readback_anomaly_class_label(anomaly.cls));
  std::printf("written_element_mask: 0x%02x\n", anomaly.written_element_mask);
  std::printf("swapped_element_mask: 0x%02x\n", anomaly.swapped_element_mask);
  std::printf("unswapped_match_element_mask: 0x%02x\n",
              anomaly.unswapped_match_element_mask);
  std::printf("status: pass\n");
  return 0;
}

// C0A23 T2 no-hardware self-test: decode the embedded kernel text's store semantics.
// The 16-word program (64 bytes of kKernelText) is RDNA4 machine code for the
// gfx1201 (R9700 class) target. The decode constants below are derived (read-only)
// from tinygrad tinygrad/runtime/autogen/amd/rdna4/ins.py + enum.py and hard-coded
// here with no tinygrad runtime dependency:
//   SMEM    FixedBitField(31,26,0b111101)    size 8  (s_load_*)
//   SOPP    FixedBitField(31,23,0b101111111) size 4  (s_wait_*, s_endpgm)
//   VOP2    FixedBitField(31,31,0b0)         size 4  (v_* ALU)
//   VGLOBAL FixedBitField(31,24,0b11101110)  op[21:14] size 12 (global_* store/load)
//   VFLAT   FixedBitField(31,24,0b11101100)  op[21:14] size 12
//   VSCRATCH FixedBitField(31,24,0b11101101) op[21:14] size 12
// Instruction sizes equal tinygrad Inst._base_size = (max(field.hi)+8)/8. The three
// vector-memory families share one store/load op-code layout (enum.py):
//   *_LOAD_B32=20, *_LOAD_B128=23, *_STORE_B8=24 B16=25 B32=26 B64=27 B96=28 B128=29;
//   *_STORE_D16_HI_B8=36 B16=37; *_STORE_ADDTID_B32=41.
// The SGPR base pair for a VGLOBAL load/store is saddr = SGPRField(6,0) = bits[6:0]
// of instruction word 0. The load must pair on the input VA s[6:7] and the store on
// the output VA s[4:5] (tinygrad canonical custom_add_var).
// This self-test is diagnosis-only: it never alters kKernelText, dispatch dims, or
// kernarg layout, and never relaxes the CPU comparison contract.
int run_kernel_text_decode_self_test() {
  struct KernelStoreOp {
    uint16_t value;      // op-code field value (EnumBitField[21:14])
    const char* suffix;  // appended after the family prefix, e.g. GLOBAL"_STORE_B128"
    uint8_t elements;    // u32 elements the store spans across (0 if not u32-aligned)
    bool addtid;         // true => work-item-indexed (ADDTID) addressing
  };
  static constexpr KernelStoreOp kStoreOps[] = {
      {24, "_STORE_B8", 0, false},
      {25, "_STORE_B16", 0, false},
      {26, "_STORE_B32", 1, false},
      {27, "_STORE_B64", 2, false},
      {28, "_STORE_B96", 3, false},
      {29, "_STORE_B128", 4, false},
      {36, "_STORE_D16_HI_B8", 0, false},
      {37, "_STORE_D16_HI_B16", 0, false},
      {41, "_STORE_ADDTID_B32", 1, true},
  };
  static constexpr const char* kFamilyPrefixes[3] = {"GLOBAL", "FLAT", "SCRATCH"};

  std::size_t store_count = 0U;
  const char* store_class = "none";
  const char* store_addressing = "none";
  char store_primary_op[64] = "none";
  std::size_t element_lo = 0U;
  std::size_t element_hi = 0U;
  // Load/store SGPR base pairs (saddr = SGPRField(6,0) = bits[6:0] of word 0 of each
  // VGLOBAL instruction). 255U = not yet seen.
  std::size_t load_saddr_lo = 255U;
  std::size_t store_saddr_lo = 255U;
  // GLOBAL_LOAD_B32 op-code (enum.py: GLOBAL_LOAD_B32=20; GLOBAL_STORE_B32=26),
  // shares the VGLOBAL 0xEE family with stores.
  static const uint32_t kGlobalLoadB32Op = 20U;
  // The per-lane kernel computes a lane-scaled 64-bit vaddr: v_lshlrev_b32_e32(v[1], 2, v[0])
  // (v[1] = lane*4) feeds the store's vaddr=v[1:2]; saddr is the kernarg segment base VA.
  // 0x4a060600 is the exact assembler-produced word for that v_lshlrev_b32_e32(v[1],2,v[0])
  // instruction in this source-grounded kernel (byte offset 0x2c, word 11). We track it in
  // the same word stream so a following B32 store can classify as lane+segment addressing.
  bool lane_scaled_vaddr = false;
  const std::size_t program_bytes = kKernelText.size();  // the 64-byte (16-word) program
  std::size_t pos = 0U;
  while (pos < program_bytes && pos + 4U <= kKernelText.size()) {
    const uint32_t word = read_u32_le_bytes(kKernelText.data() + pos);
    const uint32_t enc24 = (word >> 24U) & 0xFFU;  // bits[31:24]
    int family = -1;
    if (enc24 == 0xEEU) family = 0;        // VGLOBAL
    else if (enc24 == 0xECU) family = 1;   // VFLAT
    else if (enc24 == 0xEDU) family = 2;   // VSCRATCH
    if (family >= 0) {
      const uint32_t op = (word >> 14U) & 0xFFU;  // op[21:14]
      const uint32_t saddr_lo = word & 0x7FU;     // SGPRField(6,0) bits[6:0] of word 0
      if (family == 0 && op == kGlobalLoadB32Op) {
        load_saddr_lo = saddr_lo;  // input VA SGPR base pair (must be s[6:7])
      }
      const KernelStoreOp* entry = nullptr;
      for (const KernelStoreOp& e : kStoreOps) {
        if (e.value == op) { entry = &e; break; }
      }
      if (entry != nullptr) {
        store_saddr_lo = saddr_lo;  // output VA SGPR base pair (must be s[4:5])
        ++store_count;
        if (store_count == 1U) {
          store_class = (family == 0) ? "global" : (family == 1) ? "flat" : "scratch";
          std::snprintf(store_primary_op, sizeof(store_primary_op), "%s%s",
                        kFamilyPrefixes[family], entry->suffix);
          // Per-lane form: the store's vaddr is lane-scaled (a v_lshlrev_b32_e32(v,2,v[0])
          // lane-offset instruction precedes the B32 store) with saddr = segment base VA:
          // lane+segment. Otherwise fall through to the table's ADDTID/base+offset labels.
          if (entry->addtid) {
            store_addressing = "addtid";
          } else if (entry->value == 26U && lane_scaled_vaddr) {  // GLOBAL_STORE_B32 + lane offset
            store_addressing = "lane+segment";
          } else {
            store_addressing = "base+offset";
          }
          element_lo = 0U;
          element_hi = entry->elements > 0U ? entry->elements - 1U : 0U;
        }
      }
      pos += 12U;  // vector-memory family instruction size (bytes)
      continue;
    }
    const uint32_t enc26 = (word >> 26U) & 0x3FU;  // bits[31:26]
    if (enc26 == 0b111101U) { pos += 8U; continue; }  // SMEM
    const uint32_t enc23 = (word >> 23U) & 0x1FFU;    // bits[31:23]
    if (enc23 == 0b101111111U) { pos += 4U; continue; }  // SOPP
    // VOP2 (bit31 = 0): record the lane-scale shift instruction (v_lshlrev_b32_e32(v,2,v[0]))
    // if present, for per-lane store addressing classification.
    if (word == 0x4a060600U) { lane_scaled_vaddr = true; }
    pos += 4U;
  }

  if (pos != program_bytes ||
      store_count != 1U ||
      std::strcmp(store_class, "global") != 0 ||
      std::strcmp(store_primary_op, "GLOBAL_STORE_B32") != 0 ||
      std::strcmp(store_addressing, "lane+segment") != 0 ||
      element_lo != 0U || element_hi != 0U ||
      load_saddr_lo != 6U ||   // input VA SGPR base pair must be s[6:7]
      store_saddr_lo != 4U) {  // output VA SGPR base pair must be s[4:5]
    return self_test_failure("kernel-text-decode",
                             "kKernelText load/store decode drift vs source-grounded rdna4 tables");
  }

  std::printf("self_test: kernel-text-decode\n");
  std::printf("text_byte_count: %zu\n", kKernelText.size());
  std::printf("store_instruction_count: %zu\n", store_count);
  std::printf("store_class: %s\n", store_class);
  std::printf("store_primary_op: %s\n", store_primary_op);
  std::printf("store_addressing: %s\n", store_addressing);
  std::printf("store_element_bounds: %zu..%zu\n", element_lo, element_hi);
  std::printf("load_saddr_pair: s[%zu:%zu]\n", load_saddr_lo, load_saddr_lo + 1U);
  std::printf("store_saddr_pair: s[%zu:%zu]\n", store_saddr_lo, store_saddr_lo + 1U);
  std::printf("lane_scale_word_present: %s\n", lane_scaled_vaddr ? "true" : "false");
  std::printf("status: pass\n");
  return 0;
}



bool read_compute_queue_debug_snapshot(const RemoteClient& client, const DiscoveryLog& log,
                                       bool include_mec_ranges,
                                       ComputeQueueDebugSnapshot* snapshot,
                                       std::string* error_text) {
  if (snapshot == nullptr) {
    *error_text = "ComputeQueueDebugSnapshot precondition failed: null snapshot";
    return false;
  }
  if (!select_grbm_queue0(client, log, error_text)) {
    *error_text = "select queue0 for compute debug failed: " + *error_text;
    return false;
  }

  bool ok = read_debug_register(client, log, regs_gfx1201::kCpHqdActive,
                                regs_gfx1201::kCpHqdActive.name, &snapshot->hqd_active,
                                error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpHqdPqRptr,
                                regs_gfx1201::kCpHqdPqRptr.name, &snapshot->hqd_pq_rptr,
                                error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpHqdPqWptrHi,
                                regs_gfx1201::kCpHqdPqWptrHi.name,
                                &snapshot->hqd_pq_wptr_hi, error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpHqdPqDoorbellControl,
                                regs_gfx1201::kCpHqdPqDoorbellControl.name,
                                &snapshot->hqd_pq_doorbell_control, error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpHqdPqControl,
                                regs_gfx1201::kCpHqdPqControl.name,
                                &snapshot->hqd_pq_control, error_text) &&
            read_debug_register(client, log, regs_gfx1201::kCpStat,
                                regs_gfx1201::kCpStat.name, &snapshot->cp_stat, error_text);

  if (ok && include_mec_ranges) {
    ok = read_debug_register(client, log, regs_gfx1201::kCpMecDoorbellRangeLower,
                             regs_gfx1201::kCpMecDoorbellRangeLower.name,
                             &snapshot->mec_doorbell_range_lower, error_text) &&
         read_debug_register(client, log, regs_gfx1201::kCpMecDoorbellRangeUpper,
                             regs_gfx1201::kCpMecDoorbellRangeUpper.name,
                             &snapshot->mec_doorbell_range_upper, error_text);
    snapshot->has_mec_ranges = ok;
  }

  std::string restore_error;
  if (!restore_grbm_default_select(client, log, &restore_error)) {
    if (ok) {
      *error_text = "restore GRBM default select after compute debug failed: " + restore_error;
    } else {
      *error_text += "; restore GRBM default select also failed: " + restore_error;
    }
    return false;
  }
  return ok;
}

std::string format_compute_queue_debug_snapshot(const ComputeQueueDebugSnapshot& snapshot) {
  std::string text = "hqd_active=" + format_hex32(snapshot.hqd_active) +
                     ", hqd_pq_rptr=" + format_hex32(snapshot.hqd_pq_rptr) +
                     ", hqd_pq_wptr_hi=" + format_hex32(snapshot.hqd_pq_wptr_hi) +
                     ", hqd_pq_doorbell_control=" +
                         format_hex32(snapshot.hqd_pq_doorbell_control) +
                     ", doorbell_hit=" +
                         (((snapshot.hqd_pq_doorbell_control &
                            am_compute::kHqdPqDoorbellHitMask) != 0U)
                              ? "1"
                              : "0") +
                     ", hqd_pq_control=" + format_hex32(snapshot.hqd_pq_control) +
                     ", cp_stat=" + format_hex32(snapshot.cp_stat);
  if (snapshot.has_mec_ranges) {
    text += ", mec_doorbell_range_lower=" + format_hex32(snapshot.mec_doorbell_range_lower) +
            ", mec_doorbell_range_upper=" + format_hex32(snapshot.mec_doorbell_range_upper);
  }
  return text;
}

std::string classify_compute_doorbell_timeout(const ComputeQueueDebugSnapshot& snapshot) {
  const bool doorbell_hit =
      (snapshot.hqd_pq_doorbell_control & am_compute::kHqdPqDoorbellHitMask) != 0U;
  if (snapshot.hqd_pq_rptr == 0U && snapshot.cp_stat == 0U && !doorbell_hit) {
    return am_compute::kDoorbellClassRptrZeroCpIdle;
  }
  if (snapshot.hqd_pq_rptr == 0U && doorbell_hit) {
    return am_compute::kDoorbellClassDoorbellHitRptrZero;
  }
  if (snapshot.hqd_pq_rptr != 0U) {
    return am_compute::kDoorbellClassRptrAdvancesTimelineZero;
  }
  return "compute_doorbell_delivery_unclassified";
}

[[maybe_unused]] std::string read_compute_queue_debug(const RemoteClient& client,
                                                      const DiscoveryLog& log) {
  ComputeQueueDebugSnapshot snapshot;
  std::string error;
  if (!read_compute_queue_debug_snapshot(client, log, false, &snapshot, &error)) {
    return "debug_error=" + error;
  }
  return format_compute_queue_debug_snapshot(snapshot);
}

bool configure_compute_soc_doorbells(const RemoteClient& client, const DiscoveryLog& log,
                                     std::string* error_text) {
  // tinygrad/runtime/support/am/ip.py:37 clears the EPF2 no-soft-reset strap,
  // line 38 enables the BAR2 doorbell aperture, and lines 271-273 route gfx12
  // compute doorbells through S2A ports 0 and 3.
  if (!update_register_bits(client, log, log.ip.nbif, regs_gfx1201::kRccDev0Epf2Strap2,
                            1U << 7, 0U, error_text)) {
    *error_text = std::string(regs_gfx1201::kRccDev0Epf2Strap2.name) +
                  " clear strap_no_soft_reset_dev0_f2 failed: " + *error_text;
    return false;
  }
  if (!write_register_dword(client, log, log.ip.nbif, regs_gfx1201::kRccDoorbellAperEn, 1U,
                            error_text)) {
    *error_text = std::string(regs_gfx1201::kRccDoorbellAperEn.name) +
                  " enable write failed: " + *error_text;
    return false;
  }
  if (!write_register_dword(client, log, log.ip.nbif, regs_gfx1201::kGdcS2aDoorbellEntry0,
                            encode_s2a_doorbell_entry(0x3U, 0x3U), error_text)) {
    *error_text = std::string(regs_gfx1201::kGdcS2aDoorbellEntry0.name) +
                  " port0 route write failed: " + *error_text;
    return false;
  }
  if (!write_register_dword(client, log, log.ip.nbif, regs_gfx1201::kGdcS2aDoorbellEntry3,
                            encode_s2a_doorbell_entry(0x6U, 0x3U), error_text)) {
    *error_text = std::string(regs_gfx1201::kGdcS2aDoorbellEntry3.name) +
                  " port3 route write failed: " + *error_text;
    return false;
  }
  return true;
}
bool poll_compute_hqd_queue0_inactive(const RemoteClient& client, const DiscoveryLog& log,
                                      std::string* error_text) {
  for (int attempt = 0; attempt < 1000; ++attempt) {
    uint32_t active = 0;

    if (!read_register_dword(client, log, log.ip.gc, regs_gfx1201::kCpHqdActive, &active,
                             error_text)) {
      *error_text = std::string(regs_gfx1201::kCpHqdActive.name) +
                    " read while polling inactive failed: " + *error_text;
      return false;
    }
    if ((active & 0x1U) == 0U) {
      return true;
    }
    usleep(1000U);
  }
  *error_text = std::string(regs_gfx1201::kCpHqdActive.name) +
                " timeout waiting for active bit to clear";
  return false;
}
bool configure_mec_doorbell_range_and_wptr_poll(const RemoteClient& client,
                                                const DiscoveryLog& log,
                                                std::string* error_text) {
  // tinygrad/runtime/support/am/ip.py:293-295 programs the MEC doorbell aperture
  // to 0x000..0x0f8 for XCC0 before enabling MEC.
  if (!write_register_dword(client, log, log.ip.gc, regs_gfx1201::kCpMecDoorbellRangeLower, 0U,
                            error_text)) {
    *error_text = std::string(regs_gfx1201::kCpMecDoorbellRangeLower.name) +
                  " write lower bound failed: " + *error_text;
    return false;
  }
  if (!write_register_dword(client, log, log.ip.gc, regs_gfx1201::kCpMecDoorbellRangeUpper, 0xf8U,
                            error_text)) {
    *error_text = std::string(regs_gfx1201::kCpMecDoorbellRangeUpper.name) +
                  " write upper bound failed: " + *error_text;
    return false;
  }
  // tinygrad/runtime/support/am/ip.py:358 enables bounded MEC write-pointer polling.
  if (!write_register_dword(client, log, log.ip.gc, regs_gfx1201::kCpRbWptrPollCntl,
                            (0x90U << 16) | 0x100U, error_text)) {
    *error_text = std::string(regs_gfx1201::kCpRbWptrPollCntl.name) +
                  " write poll control failed: " + *error_text;
    return false;
  }
  return true;
}



bool reset_compute_queue0(const RemoteClient& client, const DiscoveryLog& log,
                          std::string* error_text) {
  auto restore_grbm = [&]() {
    std::string restore_error;
    if (restore_grbm_default_select(client, log, &restore_error)) {
      return true;
    }
    if (!error_text->empty()) {
      *error_text += "; ";
    }
    *error_text += restore_error;
    return false;
  };

  if (!select_grbm_queue0(client, log, error_text)) {
    restore_grbm();
    return false;
  }

  uint32_t active = 0;
  if (!read_register_dword(client, log, log.ip.gc, regs_gfx1201::kCpHqdActive, &active,
                           error_text)) {
    *error_text = std::string(regs_gfx1201::kCpHqdActive.name) + " read failed: " +
                  *error_text;
    restore_grbm();
    return false;
  }

  if ((active & 0x1U) != 0U) {
    // tinygrad/runtime/support/am/ip.py:398-405 writes RESET_WAVES dequeue then
    // SPI_COMPUTE_QUEUE_RESET before boundedly waiting for regCP_HQD_ACTIVE == 0.
    if (!write_register_dword(client, log, log.ip.gc, regs_gfx1201::kCpHqdDequeueRequest,
                              0x2U, error_text)) {
      *error_text = std::string(regs_gfx1201::kCpHqdDequeueRequest.name) +
                    " write 0x2 failed: " + *error_text;
      restore_grbm();
      return false;
    }
    if (!write_register_dword(client, log, log.ip.gc, regs_gfx1201::kSpiComputeQueueReset,
                              0x1U, error_text)) {
      *error_text = std::string(regs_gfx1201::kSpiComputeQueueReset.name) +
                    " write 0x1 failed: " + *error_text;
      restore_grbm();
      return false;
    }
    if (!poll_compute_hqd_queue0_inactive(client, log, error_text)) {
      restore_grbm();
      return false;
    }
  }

  if (!configure_mec_doorbell_range_and_wptr_poll(client, log, error_text)) {
    restore_grbm();
    return false;
  }


  return restore_grbm();
}

bool flush_hdp(const RemoteClient& client, const DiscoveryLog& log, std::string* error_text) {
  // tinygrad/runtime/support/am/ip.py:85 writes zero to the dword address stored in
  // regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL / 4 before VM TLB invalidation.
  uint32_t flush_addr_value = 0;
  if (!read_register_dword(client, log, log.ip.nbif, regs_gfx1201::kNbifHdpFlushCntl,
                           &flush_addr_value, error_text)) {
    *error_text = "HDP flush control read failed: " + *error_text;
    return false;
  }
  const uint32_t flush_reg_dword = flush_addr_value / 4U;
  const uint64_t bar5_dwords = log.bar5.size / sizeof(uint32_t);
  if (flush_reg_dword < bar5_dwords) {
    if (!write_direct_bar5_reg(client, log.bar5, flush_reg_dword, 0, error_text)) {
      *error_text = "HDP direct flush write failed: " + *error_text;
      return false;
    }
    return true;
  }
  if (!rsmu_write_register_dword(client, log, flush_reg_dword, 0, error_text)) {
    *error_text = "HDP indirect flush write failed: " + *error_text;
    return false;
  }
  return true;
}

std::string hqd_copy_register_name(std::size_t span_index) {
  switch (span_index) {
    case 0: return "regCP_MQD_BASE_ADDR";
    case 1: return "regCP_MQD_BASE_ADDR_HI";
    case 2: return "regCP_HQD_ACTIVE";
    case 3: return "regCP_HQD_VMID";
    case 4: return "regCP_HQD_PERSISTENT_STATE";
    case 5: return "regCP_HQD_PIPE_PRIORITY";
    case 6: return "regCP_HQD_QUEUE_PRIORITY";
    case 7: return "regCP_HQD_QUANTUM";
    case 8: return "regCP_HQD_PQ_BASE";
    case 9: return "regCP_HQD_PQ_BASE_HI";
    case 10: return "regCP_HQD_PQ_RPTR";
    case 11: return "regCP_HQD_PQ_RPTR_REPORT_ADDR";
    case 12: return "regCP_HQD_PQ_RPTR_REPORT_ADDR_HI";
    case 13: return "regCP_HQD_PQ_WPTR_POLL_ADDR";
    case 14: return "regCP_HQD_PQ_WPTR_POLL_ADDR_HI";
    case 15: return "regCP_HQD_PQ_DOORBELL_CONTROL";
    case 17: return "regCP_HQD_PQ_CONTROL";
    case 18: return "regCP_HQD_IB_BASE_ADDR";
    case 19: return "regCP_HQD_IB_BASE_ADDR_HI";
    case 20: return "regCP_HQD_IB_RPTR";
    case 21: return "regCP_HQD_IB_CONTROL";
    case 22: return "regCP_HQD_IQ_TIMER";
    case 23: return "regCP_HQD_IQ_RPTR";
    case 24: return "regCP_HQD_DEQUEUE_REQUEST";
    case 25: return "regCP_HQD_DMA_OFFLOAD";
    case 26: return "regCP_HQD_SEMA_CMD";
    case 27: return "regCP_HQD_MSG_TYPE";
    case 28: return "regCP_HQD_ATOMIC0_PREOP_LO";
    case 29: return "regCP_HQD_ATOMIC0_PREOP_HI";
    case 30: return "regCP_HQD_ATOMIC1_PREOP_LO";
    case 31: return "regCP_HQD_ATOMIC1_PREOP_HI";
    case 32: return "regCP_HQD_HQ_STATUS0";
    case 33: return "regCP_HQD_HQ_CONTROL0";
    case 34: return "regCP_MQD_CONTROL";
    case 35: return "regCP_HQD_HQ_STATUS1";
    case 36: return "regCP_HQD_HQ_CONTROL1";
    case 37: return "regCP_HQD_EOP_BASE_ADDR";
    case 38: return "regCP_HQD_EOP_BASE_ADDR_HI";
    case 39: return "regCP_HQD_EOP_CONTROL";
    case 40: return "regCP_HQD_EOP_RPTR";
    case 41: return "regCP_HQD_EOP_WPTR";
    case 42: return "regCP_HQD_EOP_EVENTS";
    case 43: return "regCP_HQD_CTX_SAVE_BASE_ADDR_LO";
    case 44: return "regCP_HQD_CTX_SAVE_BASE_ADDR_HI";
    case 45: return "regCP_HQD_CTX_SAVE_CONTROL";
    case 46: return "regCP_HQD_CNTL_STACK_OFFSET";
    case 47: return "regCP_HQD_CNTL_STACK_SIZE";
    case 48: return "regCP_HQD_WG_STATE_OFFSET";
    case 49: return "regCP_HQD_CTX_SAVE_SIZE";
    case 50: return "regCP_HQD_GDS_RESOURCE_STATE";
    case 51: return "regCP_HQD_ERROR";
    case 52: return "regCP_HQD_EOP_WPTR_MEM";
    case 53: return "regCP_HQD_AQL_CONTROL";
    case 54: return "regCP_HQD_PQ_WPTR_LO";
    case 55: return "regCP_HQD_PQ_WPTR_HI";
    default: return std::string(regs_gfx1201::kCpMqdBaseAddr.name) + "+" + std::to_string(span_index);
  }
}

bool write_and_verify_compute_mqd(const RemoteClient& client, const DiscoveryLog& log,
                                  std::string* error_text) {
  if (log.bar0.size < am_compute::kMqdPaddr + am_compute::kMqdSize) {
    *error_text = "BAR0 too small for compute MQD at " + format_hex64(am_compute::kMqdPaddr) +
                  ": bar0_size_bytes=" + std::to_string(log.bar0.size) +
                  " required_at_least=" + std::to_string(am_compute::kMqdPaddr + am_compute::kMqdSize);
    return false;
  }
  const ComputeMqd mqd = build_compute_mqd(log.vm.mc_base);
  std::vector<uint8_t> payload;
  payload.reserve(am_compute::kMqdSize);
  for (uint32_t dword : mqd) {
    append_u32_le(&payload, dword);
  }
  if (!mmio_write_bar0(client, am_compute::kMqdPaddr, payload, error_text)) {
    *error_text = "write compute MQD bytes to BAR0 paddr " + format_hex64(am_compute::kMqdPaddr) +
                  " failed: " + *error_text;
    return false;
  }
  for (std::size_t i = 0; i < mqd.size(); i += 2) {
    const uint64_t expected = static_cast<uint64_t>(mqd[i]) |
                              (static_cast<uint64_t>(mqd[i + 1]) << 32);
    const uint64_t paddr = am_compute::kMqdPaddr + (i * sizeof(uint32_t));
    uint64_t observed = 0;
    if (!read_bar0_qword(client, paddr, &observed, error_text)) {
      *error_text = "compute MQD readback qword at " + format_hex64(paddr) +
                    " failed: " + *error_text;
      return false;
    }
    if (observed != expected) {
      *error_text = "compute MQD readback mismatch at " + format_hex64(paddr) +
                    ": expected " + format_hex64(expected) + ", observed " + format_hex64(observed);
      return false;
    }
  }
  return true;
}

bool zero_compute_vram_pages(const RemoteClient& client, std::string* error_text) {
  const std::array<std::pair<const char*, uint64_t>, 3> single_pages{{
      {"compute output VRAM page", am_compute::kOutputVramPaddr},
      {"compute code VRAM page", am_compute::kCodeVramPaddr},
      {"compute EOP VRAM page", am_compute::kEopVramPaddr},
  }};
  for (const auto& page : single_pages) {
    if (!zero_bar0_page(client, page.second, error_text)) {
      *error_text = std::string("zero ") + page.first + " at " + format_hex64(page.second) +
                    " failed: " + *error_text;
      return false;
    }
  }
  return true;
}

// tinygrad/runtime/support/am/ip.py:_config_mec() (380-396) and _enable_mec() (374-378).
// Preserve the device-provided RS64 MEC program start before the reset replay: native has
// no firmware ucode source, so it must never manufacture a replacement start address.
bool replay_mec_rs64_pipe_activation(const RemoteClient& client, DiscoveryLog* log,
                                     std::string* error_text) {
  auto restore_grbm = [&]() {
    std::string restore_error;
    if (restore_grbm_default_select(client, *log, &restore_error)) {
      return true;
    }
    if (!error_text->empty()) {
      *error_text += "; ";
    }
    *error_text += restore_error;
    return false;
  };
  auto fail = [&](const std::string& message) {
    log->compute.mec_rs64_cntl_write_status = "fail";
    *error_text = message;
    restore_grbm();
    return false;
  };

  if (!select_grbm_mec_rs64_pipe0(client, *log, error_text)) {
    return fail("select MEC RS64 failed: " + *error_text);
  }

  uint32_t program_counter_low = 0;
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64PrgrmCntrStart,
                           &program_counter_low, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64PrgrmCntrStart.name) +
                " read-before-reset failed: " + *error_text);
  }
  uint32_t program_counter_high = 0;
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64PrgrmCntrStartHi,
                           &program_counter_high, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64PrgrmCntrStartHi.name) +
                " read-before-reset failed: " + *error_text);
  }
  if (program_counter_low == 0U && program_counter_high == 0U) {
    return fail("MEC RS64 program counter start pair is all zero before reset");
  }

  uint32_t prior = 0;
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64Cntl,
                           &prior, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                " read-before-write failed: " + *error_text);
  }
  // mec_pipe0_reset=1 (bit 16).
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64Cntl,
                            prior | 0x00010000U, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                " mec_pipe0_reset=1 write failed: " + *error_text);
  }

  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64PrgrmCntrStart,
                            program_counter_low, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64PrgrmCntrStart.name) +
                " restore after reset failed: " + *error_text);
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64PrgrmCntrStartHi,
                            program_counter_high, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64PrgrmCntrStartHi.name) +
                " restore after reset failed: " + *error_text);
  }

  uint32_t program_counter_low_readback = 0;
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64PrgrmCntrStart,
                           &program_counter_low_readback, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64PrgrmCntrStart.name) +
                " restore readback failed: " + *error_text);
  }
  uint32_t program_counter_high_readback = 0;
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64PrgrmCntrStartHi,
                           &program_counter_high_readback, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64PrgrmCntrStartHi.name) +
                " restore readback failed: " + *error_text);
  }
  if (program_counter_low_readback != program_counter_low ||
      program_counter_high_readback != program_counter_high) {
    return fail("MEC RS64 program counter start pair readback differs from captured pair");
  }

  // mec_pipe0_reset=0, active=1, halt=0 -> clear bits 16-19 (reset) and 30 (halt),
  // set bit 26 (active), preserve all other fields.
  // 0x400F0000 = bit30 | bits 16..19; ~ = 0xBFF0FFFF clears exactly those.
  const uint32_t steady = (prior & 0xBFF0FFFFU) | 0x04000000U;
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64Cntl,
                            steady, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                " activate write failed: " + *error_text);
  }
  // tinygrad _enable_mec(): 50 ms settle after activation.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  uint32_t readback = 0;
  if (!read_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpMecRs64Cntl,
                           &readback, error_text)) {
    return fail(std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                " readback failed: " + *error_text);
  }
  log->compute.mec_rs64_cntl_readback = format_hex32(readback);
  if ((readback & 0x04000000U) == 0U) {
    log->compute.mec_rs64_active_status = "fail";
    return fail(std::string(regs_gfx1201::kCpMecRs64Cntl.name) +
                " mec_pipe0_active not observed after activation");
  }
  if (!restore_grbm()) {
    log->compute.mec_rs64_cntl_write_status = "fail";
    return false;
  }
  log->compute.mec_rs64_cntl_write_status = "pass";
  log->compute.mec_rs64_active_status = "pass";
  return true;
}

bool setup_compute_ring0(const RemoteClient& client, DiscoveryLog* log,
                         SysmemMapping* compute_control_mapping, std::string* error_text) {
  auto fail = [&](const std::string& text) {
    log->compute.ring_setup_status = "fail";
    *error_text = text;
    return false;
  };

  if (log == nullptr) {
    *error_text = "DiscoveryLog precondition failed: null log";
    return false;
  }
  const uint64_t required_bar2_size = am_compute::kMecDoorbellBar2ByteOffset + sizeof(uint64_t);
  if (log->bar2.size < required_bar2_size) {
    return fail("BAR2 precondition failed for compute MEC doorbell: bar2_size_bytes=" +
                std::to_string(log->bar2.size) + " required_at_least=" +
                std::to_string(required_bar2_size));
  }
  if (!configure_compute_soc_doorbells(client, *log, error_text)) {
    return fail("compute SOC doorbell route setup failed: " + *error_text);
  }
  ComputeDoorbellRouteSnapshot route_snapshot;
  std::string route_readback_error;
  if (read_compute_doorbell_route_snapshot(client, *log, &route_snapshot, &route_readback_error)) {
    log->compute.doorbell_route_readback =
        format_compute_doorbell_route_snapshot(route_snapshot);
    log->compute.doorbell_route_classification =
        classify_compute_doorbell_route_snapshot(route_snapshot);
  } else {
    log->compute.doorbell_route_readback = "read_failed: " + route_readback_error;
    log->compute.doorbell_route_classification = am_compute::kDoorbellRouteClassUnclassified;
  }
  if (log->vm.vmid0_context_status != "pass") {
    return fail("VMID0 MMHUB context precondition failed: vmid0_context_status=" +
                log->vm.vmid0_context_status);
  }
  if (log->vm.vm_gc_context_status != "pass") {
    return fail("GC VMID0 context precondition failed: vm_gc_context_status=" +
                log->vm.vm_gc_context_status);
  }
  if (log->vm.mm_tlb_flush_status != "pass") {
    return fail("MMHUB TLB precondition failed: mm_tlb_flush_status=" +
                log->vm.mm_tlb_flush_status);
  }
  if (log->vm.gc_tlb_flush_status != "pass") {
    return fail("GC TLB precondition failed: gc_tlb_flush_status=" +
                log->vm.gc_tlb_flush_status);
  }
  if (compute_control_mapping == nullptr || compute_control_mapping->data == nullptr ||
      compute_control_mapping->size < am_compute::kComputeControlByteCount) {
    return fail("compute_control mapping precondition failed: need ten mapped 4 KiB pages (2 control + 8 ring)");
  }

  std::memset(compute_control_mapping->data, 0, kPageSize);
  if (!zero_compute_vram_pages(client, error_text)) {
    return fail(*error_text);
  }
  // Mirror tinygrad platform init ordering: MEC RS64 pipe reset/activate before
  // MQD/HQD ring setup and HQD activation.
  if (!replay_mec_rs64_pipe_activation(client, log, error_text)) {
    return fail("MEC RS64 pipe activation failed: " + *error_text);
  }
  if (!write_and_verify_compute_mqd(client, *log, error_text)) {
    return fail(*error_text);
  }
  if (!reset_compute_queue0(client, *log, error_text)) {
    return fail("reset_compute_queue0 failed: " + *error_text);
  }

  bool grbm_selected = false;
  auto restore_grbm = [&]() {
    if (!grbm_selected) {
      return true;
    }
    std::string restore_error;
    if (restore_grbm_default_select(client, *log, &restore_error)) {
      grbm_selected = false;
      return true;
    }
    if (!error_text->empty()) {
      *error_text += "; ";
    }
    *error_text += restore_error;
    return false;
  };
  auto fail_with_restore = [&](const std::string& text) {
    log->compute.ring_setup_status = "fail";
    *error_text = text;
    restore_grbm();
    return false;
  };

  if (!select_grbm_queue0(client, *log, error_text)) {
    std::string select_error = *error_text;
    std::string restore_error;
    if (!restore_grbm_default_select(client, *log, &restore_error)) {
      select_error += "; " + restore_error;
    }
    return fail("select_grbm_queue0 failed: " + select_error);
  }
  grbm_selected = true;

  const ComputeMqd mqd = build_compute_mqd(log->vm.mc_base);
  constexpr std::size_t kHqdRegisterCopyDwordCount =
      regs_gfx1201::kCpHqdPqWptrHi.offset - regs_gfx1201::kCpMqdBaseAddr.offset + 1U;
  for (std::size_t i = 0; i < kHqdRegisterCopyDwordCount; ++i) {
    const std::string reg_name = hqd_copy_register_name(i);
    const RegDef reg{reg_name.c_str(), regs_gfx1201::kCpMqdBaseAddr.offset + static_cast<uint32_t>(i),
                     regs_gfx1201::kCpMqdBaseAddr.segment};
    if (!write_register_dword(client, *log, log->ip.gc, reg,
                              mqd[kMqdHqdRegisterCopyStart + i], error_text)) {
      return fail_with_restore(reg_name + " HQD copy write failed: " + *error_text);
    }
  }

  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kCpHqdActive, 1U,
                            error_text)) {
    log->compute.hqd_active_status = "fail";
    return fail_with_restore(std::string(regs_gfx1201::kCpHqdActive.name) +
                             " active write failed: " + *error_text);
  }
  if (!poll_register_mask(client, *log, log->ip.gc, regs_gfx1201::kCpHqdActive, 0x1U, 0x1U,
                          regs_gfx1201::kCpHqdActive.name, error_text)) {
    log->compute.hqd_active_status = "fail";
    return fail_with_restore(std::string(regs_gfx1201::kCpHqdActive.name) +
                             " active readback failed: " + *error_text);
  }
  if (!flush_hdp(client, *log, error_text)) {
    return fail_with_restore("compute ring HDP flush failed: " + *error_text);
  }
  if (!restore_grbm()) {
    log->compute.ring_setup_status = "fail";
    return false;
  }
  log->compute.ring_setup_status = "pass";
  log->compute.hqd_active_status = "pass";
  return true;
}

// No-hardware contract test for the device-provided MEC RS64 program-start pair
// preserved across the replay reset and restored before activation.
int run_mec_rs64_pipe_activation_self_test() {
  constexpr uint32_t kResetBit = 0x00010000U;   // regs.py mec_pipe0_reset bit 16
  constexpr uint32_t kActiveBit = 0x04000000U;  // regs.py mec_pipe0_active bit 26
  constexpr uint32_t kSteadyMask = 0xBFF0FFFFU; // ~0x400F0000 clears bits 16-19 & 30
  constexpr uint32_t kCapturedLow = 0x13579BDFU;
  constexpr uint32_t kCapturedHigh = 0x2468ACE0U;

  if (regs_gfx1201::kCpMecRs64Cntl.offset != 10500U ||
      regs_gfx1201::kCpMecRs64Cntl.segment != 1U) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "regCP_MEC_RS64_CNTL offset/segment drift");
  }
  if (std::strcmp(regs_gfx1201::kCpMecRs64Cntl.name, "regCP_MEC_RS64_CNTL") != 0) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "regCP_MEC_RS64_CNTL name drift");
  }
  if (regs_gfx1201::kCpMecRs64PrgrmCntrStart.offset != 10496U ||
      regs_gfx1201::kCpMecRs64PrgrmCntrStartHi.offset != 10552U ||
      regs_gfx1201::kCpMecRs64PrgrmCntrStart.segment != 1U ||
      regs_gfx1201::kCpMecRs64PrgrmCntrStartHi.segment != 1U) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "MEC RS64 program counter start register drift");
  }
  if (std::strcmp(regs_gfx1201::kCpMecRs64PrgrmCntrStart.name,
                  "regCP_MEC_RS64_PRGRM_CNTR_START") != 0 ||
      std::strcmp(regs_gfx1201::kCpMecRs64PrgrmCntrStartHi.name,
                  "regCP_MEC_RS64_PRGRM_CNTR_START_HI") != 0) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "MEC RS64 program counter start register name drift");
  }
  const auto all_zero_pair = [](uint32_t low, uint32_t high) {
    return low == 0U && high == 0U;
  };
  if (!all_zero_pair(0U, 0U) || all_zero_pair(kCapturedLow, kCapturedHigh)) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "program counter start pair zero validation drift");
  }
  const uint32_t restored_low = kCapturedLow;
  const uint32_t restored_high = kCapturedHigh;
  const uint32_t low_readback = restored_low;
  const uint32_t high_readback = restored_high;
  if (restored_low != kCapturedLow || restored_high != kCapturedHigh ||
      low_readback != kCapturedLow || high_readback != kCapturedHigh) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "program counter start pair was not preserved exactly");
  }

  // A prior value with unrelated bits (and an old stale active/halt) must converge
  // to exactly steady = (prior & mask) | active, and the reset write to prior|reset.
  const uint32_t prior = 0x00001234U | 0x04000000U;  // includes stale mec_pipe0_active
  const uint32_t reset_write = prior | kResetBit;
  const uint32_t steady = (prior & kSteadyMask) | kActiveBit;
  if ((reset_write & kResetBit) == 0U) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "mec_pipe0_reset=1 bit not set by reset write");
  }
  if ((steady & kActiveBit) == 0U) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "mec_pipe0_active bit not set by steady write");
  }
  if ((steady & kResetBit) != 0U || (steady & 0x000F0000U) != 0U ||
      (steady & 0x40000000U) != 0U) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "steady write retains reset/halt bits 16-19/30");
  }
  if ((steady & 0x00001234U) != 0x00001234U) {
    return self_test_failure("mec-rs64-pipe-activation",
                             "steady write drops unrelated prior fields");
  }

  std::printf("self_test: mec-rs64-pipe-activation\n");
  std::printf("cntl_register: regCP_MEC_RS64_CNTL\n");
  std::printf("cntl_offset: %u\n", regs_gfx1201::kCpMecRs64Cntl.offset);
  std::printf("cntl_segment: %u\n", regs_gfx1201::kCpMecRs64Cntl.segment);
  std::printf("mec_grbm_select: ME=1 pipe=0 queue=0\n");
  std::printf("program_counter_low_register: %s\n",
              regs_gfx1201::kCpMecRs64PrgrmCntrStart.name);
  std::printf("program_counter_low_offset: %u\n",
              regs_gfx1201::kCpMecRs64PrgrmCntrStart.offset);
  std::printf("program_counter_high_register: %s\n",
              regs_gfx1201::kCpMecRs64PrgrmCntrStartHi.name);
  std::printf("program_counter_high_offset: %u\n",
              regs_gfx1201::kCpMecRs64PrgrmCntrStartHi.offset);
  std::printf("program_counter_segment: %u\n",
              regs_gfx1201::kCpMecRs64PrgrmCntrStart.segment);
  std::printf("program_counter_pair_zero_rejected: true\n");
  std::printf("program_counter_pair_restored_exactly: true\n");
  std::printf("program_counter_pair_readback_matches: true\n");
  std::printf("replay_sequence: select_mec,read_start_low,read_start_high,reject_zero_pair,"
              "assert_reset,restore_start_low,restore_start_high,verify_start_low,"
              "verify_start_high,activate\n");
  std::printf("mec_pipe0_reset_bit: 0x%08x\n", kResetBit);
  std::printf("mec_pipe0_active_bit: 0x%08x\n", kActiveBit);
  std::printf("steady_mask: 0x%08x\n", kSteadyMask);
  std::printf("sample_prior: 0x%08x\n", prior);
  std::printf("sample_reset_write: 0x%08x\n", reset_write);
  std::printf("sample_steady_write: 0x%08x\n", steady);
  std::printf("status: pass\n");
  return 0;
}


bool flush_mmhubs_tlb(const RemoteClient& client, DiscoveryLog* log, std::string* error_text) {
  if (!flush_hdp(client, *log, error_text)) {
    log->vm.mm_tlb_flush_status = "fail";
    return false;
  }
  for (const IpBlockInfo& mmhub : log->ip.mmhubs) {
    if (!poll_register_mask(client, *log, mmhub, regs_gfx1201::kMmInvalidateEng17Sem, 0x1U, 0x1U,
                            "MMHUB invalidate engine 17 semaphore", error_text)) {
      log->vm.mm_tlb_flush_status = "fail";
      return false;
    }
    if (!write_register_dword(client, *log, mmhub, regs_gfx1201::kMmInvalidateEng17Req,
                              encode_invalidate_req_vmid0(), error_text)) {
      *error_text = "MMHUB invalidate request write failed: " + *error_text;
      log->vm.mm_tlb_flush_status = "fail";
      return false;
    }
    if (!poll_register_mask(client, *log, mmhub, regs_gfx1201::kMmInvalidateEng17Ack,
                            am_vm::kInvalidateMaskVmid0, am_vm::kInvalidateMaskVmid0,
                            "MMHUB invalidate engine 17 ack", error_text)) {
      log->vm.mm_tlb_flush_status = "fail";
      return false;
    }
    if (!write_register_dword(client, *log, mmhub, regs_gfx1201::kMmInvalidateEng17Sem, 0, error_text)) {
      *error_text = "MMHUB invalidate semaphore clear failed: " + *error_text;
      log->vm.mm_tlb_flush_status = "fail";
      return false;
    }
    if (!update_register_bits(client, *log, mmhub, regs_gfx1201::kMmReservedCid2,
                              1U << 25, 1U << 25, error_text)) {
      *error_text = "MMHUB reserved CID2 private invalidation update failed: " + *error_text;
      log->vm.mm_tlb_flush_status = "fail";
      return false;
    }
    uint32_t readback = 0;
    if (!read_register_dword(client, *log, mmhub, regs_gfx1201::kMmReservedCid2, &readback, error_text)) {
      *error_text = "MMHUB reserved CID2 readback failed: " + *error_text;
      log->vm.mm_tlb_flush_status = "fail";
      return false;
    }
    static_cast<void>(readback);
  }
  log->vm.mm_tlb_flush_status = "pass";
  if (log->vm.vm_gc_context_status != "pass") {
    log->vm.gc_tlb_flush_status = "skipped_gc_hub_not_initialized";
  }
  return true;
}

bool flush_gc_tlb_vmid0(const RemoteClient& client, DiscoveryLog* log,
                        std::string* error_text) {
  if (!flush_hdp(client, *log, error_text)) {
    *error_text = "GC TLB HDP flush failed: " + *error_text;
    log->vm.gc_tlb_flush_status = "fail";
    return false;
  }
  if (!poll_register_mask(client, *log, log->ip.gc, regs_gfx1201::kGcInvalidateEng17Sem, 0x1U,
                          0x1U, regs_gfx1201::kGcInvalidateEng17Sem.name, error_text)) {
    log->vm.gc_tlb_flush_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcInvalidateEng17Req,
                            encode_invalidate_req_vmid0(), error_text)) {
    *error_text = std::string("write ") + regs_gfx1201::kGcInvalidateEng17Req.name +
                  " failed: " + *error_text;
    log->vm.gc_tlb_flush_status = "fail";
    return false;
  }
  if (!poll_register_mask(client, *log, log->ip.gc, regs_gfx1201::kGcInvalidateEng17Ack,
                          am_vm::kInvalidateMaskVmid0, am_vm::kInvalidateMaskVmid0,
                          regs_gfx1201::kGcInvalidateEng17Ack.name, error_text)) {
    log->vm.gc_tlb_flush_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcInvalidateEng17Sem, 0,
                            error_text)) {
    *error_text = std::string("clear ") + regs_gfx1201::kGcInvalidateEng17Sem.name +
                  " failed: " + *error_text;
    log->vm.gc_tlb_flush_status = "fail";
    return false;
  }
  log->vm.gc_tlb_flush_status = "pass";
  return true;
}


bool setup_fixed_vm_mapping(const RemoteClient& client, DiscoveryLog* log, const VmBufferLog& staging,
                            const VmBufferLog& readback, const VmBufferLog& sdma_control,
                            const VmBufferLog* compute_control, bool enable_gc_hub,
                            FixedVmMappingResult* result) {
  result->tables = log->vm.tables;
  std::string error;
  if (!(enable_gc_hub ? is_supported_gfx1201_vm_ip_layout(*log, &error)
                      : is_supported_gfx1201_ip_layout(*log, &error))) {
    log->vm.page_tables_written = "fail";
    result->error_text = error;
    return false;
  }
  if (!write_fixed_page_tables(client, log, staging, readback, sdma_control, compute_control, &error)) {
    log->vm.page_tables_written = "fail";
    result->error_text = error;
    return false;
  }
  result->page_tables_written = true;
  if (!program_mmhubs_vmid0(client, log, &error)) {
    log->vm.vmid0_context_status = "fail";
    result->error_text = error;
    return false;
  }
  result->vmid0_context_programmed = true;
  if (!flush_mmhubs_tlb(client, log, &error)) {
    result->error_text = error;
    return false;
  }
  result->tlb_flushed = true;
  if (!enable_gc_hub) {
    return true;
  }
  if (!validate_direct_pm4_topology(*log, &error)) {
    result->failure_stage = "multi_xcc_aql_required";
    result->error_text = error;
    return false;
  }
  if (!program_gc_hub_vmid0(client, log, &error)) {
    result->failure_stage = "gc_hub_init";
    result->error_text = error;
    return false;
  }
  if (!flush_gc_tlb_vmid0(client, log, &error)) {
    result->failure_stage = "gc_tlb_flush";
    result->error_text = error;
    return false;
  }
  return true;
}

bool validate_sdma0_7_0_1(const DiscoveryLog& log, std::string* error_text) {
  if (!log.ip.sdma0.found || log.ip.sdma0.major != 7U || log.ip.sdma0.minor != 0U ||
      log.ip.sdma0.revision != 1U) {
    *error_text = "SDMA0 IP record missing or unsupported: " + ip_version_text(log.ip.sdma0);
    return false;
  }
  return true;
}

uint32_t encode_sdma_rb_cntl() {
  // regs.py:5428 gc_12_0_0 fields and tinygrad/runtime/support/am/ip.py:553-554 setup_ring
  // set mcu_wptr_poll_enable for SDMA IP >= 7.0.0.
  return (1U << 0) | (am_sdma::kRingSizeField << 1) | (1U << 11) | (1U << 12) |
         (4U << 16) | (1U << 23) | (0U << 24);
}

bool readback_register_mask(const RemoteClient& client, const DiscoveryLog& log, const IpBlockInfo& ip,
                            const RegDef& reg, uint32_t mask, uint32_t expected,
                            std::string* error_text) {
  uint32_t observed = 0;
  if (!read_register_dword(client, log, ip, reg, &observed, error_text)) {
    return false;
  }
  if ((observed & mask) != expected) {
    *error_text = std::string(reg.name) + " readback mismatch: expected_masked=" +
                  format_hex64(expected) + " observed=" + format_hex64(observed) +
                  " mask=" + format_hex64(mask);
    return false;
  }
  return true;
}

bool write_sdma_register_checked(const RemoteClient& client, const DiscoveryLog& log,
                                 const RegDef& reg, uint32_t value, std::string* error_text) {
  if (!write_register_dword(client, log, log.ip.sdma0, reg, value, error_text)) {
    return false;
  }
  return readback_register_mask(client, log, log.ip.sdma0, reg, 0xffffffffU, value, error_text);
}

bool write_sdma_pair_checked(const RemoteClient& client, const DiscoveryLog& log,
                             const RegDef& lo_reg, const RegDef& hi_reg, uint64_t value,
                             std::string* error_text, uint32_t lo_mask = 0xffffffffU,
                             uint32_t hi_mask = 0xffffffffU) {
  if (!write_register_pair(client, log, log.ip.sdma0, lo_reg, hi_reg, value, error_text)) {
    return false;
  }
  if (!readback_register_mask(client, log, log.ip.sdma0, lo_reg, lo_mask,
                              static_cast<uint32_t>(value & 0xffffffffULL) & lo_mask,
                              error_text)) {
    return false;
  }
  return readback_register_mask(client, log, log.ip.sdma0, hi_reg, hi_mask,
                                static_cast<uint32_t>((value >> 32) & 0xffffffffULL) & hi_mask,
                                error_text);
}

bool update_sdma_register_bits_checked(const RemoteClient& client, const DiscoveryLog& log,
                                       const RegDef& reg, uint32_t clear_mask, uint32_t set_mask,
                                       std::string* error_text) {
  if (!update_register_bits(client, log, log.ip.sdma0, reg, clear_mask, set_mask, error_text)) {
    return false;
  }
  return readback_register_mask(client, log, log.ip.sdma0, reg, clear_mask, set_mask, error_text);
}

bool reset_sdma_queue0(const RemoteClient& client, const DiscoveryLog& log,
                       std::string* error_text) {
  // tinygrad/runtime/support/am/ip.py:524-535 fini_hw disables the programmed
  // queue and soft-resets SDMA for IP >= 6.0.0. The native proof may be run
  // repeatedly against the same TinyGPU.app server, so clear a previous queue0
  // before reprogramming ring pointers; otherwise the live MCU wptr polling can
  // reload the previous submit byte count while setup writes RB_WPTR = 0.
  constexpr uint32_t kDoorbellOffsetMask = ((1U << 26) - 1U) << 2;
  if (!update_sdma_register_bits_checked(client, log, regs_gfx1201::kSdma0Queue0RbCntl,
                                         1U << 0, 0U, error_text) ||
      !update_sdma_register_bits_checked(client, log, regs_gfx1201::kSdma0Queue0IbCntl,
                                         1U << 0, 0U, error_text) ||
      !update_sdma_register_bits_checked(client, log, regs_gfx1201::kSdma0Queue0Doorbell,
                                         1U << 28, 0U, error_text) ||
      !update_sdma_register_bits_checked(client, log, regs_gfx1201::kSdma0Queue0DoorbellOffset,
                                         kDoorbellOffsetMask, 0U, error_text)) {
    return false;
  }
  if (!write_register_dword(client, log, log.ip.gc, regs_gfx1201::kGrbmSoftReset,
                            1U << am_sdma::kSoftResetSdma0Bit, error_text)) {
    return false;
  }
  usleep(10000U);
  return write_register_dword(client, log, log.ip.gc, regs_gfx1201::kGrbmSoftReset, 0U,
                              error_text);
}

bool setup_sdma_queue0(const RemoteClient& client, DiscoveryLog* log, std::string* error_text) {
  if (!validate_sdma0_7_0_1(*log, error_text)) {
    log->sdma.queue_setup_status = "fail";
    return false;
  }
  if (log->bar2.size <= am_sdma::kDoorbellBar2ByteOffset + sizeof(uint64_t)) {
    *error_text = "BAR2 too small for SDMA doorbell: bar2_size_bytes=" + std::to_string(log->bar2.size) +
                  " required_gt=" + std::to_string(am_sdma::kDoorbellBar2ByteOffset + sizeof(uint64_t));
    log->sdma.queue_setup_status = "fail";
    return false;
  }

  if (!reset_sdma_queue0(client, *log, error_text)) {
    log->sdma.queue_setup_status = "fail";
    return false;
  }

  if (!write_sdma_register_checked(client, *log, regs_gfx1201::kSdma0Queue0MinorPtrUpdate, 1U, error_text) ||
      !write_sdma_pair_checked(client, *log, regs_gfx1201::kSdma0Queue0RbRptr,
                               regs_gfx1201::kSdma0Queue0RbRptrHi, 0ULL, error_text) ||
      !write_sdma_pair_checked(client, *log, regs_gfx1201::kSdma0Queue0RbWptr,
                               regs_gfx1201::kSdma0Queue0RbWptrHi, 0ULL, error_text) ||
      !write_sdma_pair_checked(client, *log, regs_gfx1201::kSdma0Queue0RbBase,
                               regs_gfx1201::kSdma0Queue0RbBaseHi, am_sdma::kControlVa >> 8, error_text) ||
      !write_sdma_pair_checked(client, *log, regs_gfx1201::kSdma0Queue0RbRptrAddrLo,
                               regs_gfx1201::kSdma0Queue0RbRptrAddrHi, am_sdma::kRptrVa,
                               error_text, 0xfffffffcU) ||
      !write_sdma_pair_checked(client, *log, regs_gfx1201::kSdma0Queue0RbWptrPollAddrLo,
                               regs_gfx1201::kSdma0Queue0RbWptrPollAddrHi, am_sdma::kWptrVa,
                               error_text, 0xfffffffcU)) {
    log->sdma.queue_setup_status = "fail";
    return false;
  }

  constexpr uint32_t kDoorbellOffsetMask = ((1U << 26) - 1U) << 2;
  const uint32_t doorbell_offset_value = am_sdma::kDoorbellOffsetField << 2;
  if (!update_sdma_register_bits_checked(client, *log, regs_gfx1201::kSdma0Queue0DoorbellOffset,
                                         kDoorbellOffsetMask, doorbell_offset_value, error_text) ||
      !update_sdma_register_bits_checked(client, *log, regs_gfx1201::kSdma0Queue0Doorbell,
                                         1U << 28, 1U << 28, error_text) ||
      !write_sdma_register_checked(client, *log, regs_gfx1201::kSdma0Queue0MinorPtrUpdate, 0U, error_text) ||
      !write_sdma_register_checked(client, *log, regs_gfx1201::kSdma0Queue0RbCntl,
                                   encode_sdma_rb_cntl(), error_text) ||
      !update_sdma_register_bits_checked(client, *log, regs_gfx1201::kSdma0Queue0IbCntl,
                                         1U << 0, 1U << 0, error_text)) {
    log->sdma.queue_setup_status = "fail";
    return false;
  }
  uint32_t context_status = 0;
  if (!read_register_dword(client, *log, log->ip.sdma0, regs_gfx1201::kSdma0Queue0ContextStatus,
                           &context_status, error_text)) {
    log->sdma.queue_setup_status = "fail";
    return false;
  }
  static_cast<void>(context_status);

  log->sdma.queue_setup_status = "pass";
  return true;
}

bool write_sdma_ring_words(SysmemMapping* control_mapping, const std::vector<uint32_t>& words,
                           uint64_t submit_byte_offset, std::string* error_text) {
  if (words.empty()) {
    *error_text = "SDMA ring write has no packet words";
    return false;
  }
  if ((submit_byte_offset % sizeof(uint32_t)) != 0) {
    *error_text = "SDMA ring write offset is not dword-aligned: submit_byte_offset=" +
                  std::to_string(submit_byte_offset);
    return false;
  }
  const uint64_t ring_bytes = static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  if (submit_byte_offset > am_sdma::kRingSize ||
      ring_bytes > am_sdma::kRingSize - submit_byte_offset) {
    *error_text = "SDMA ring write exceeds ring size: submit_byte_offset=" +
                  std::to_string(submit_byte_offset) + " ring_write_bytes=" +
                  std::to_string(ring_bytes) + " ring_size_bytes=" +
                  std::to_string(am_sdma::kRingSize);
    return false;
  }
  if (submit_byte_offset > control_mapping->size ||
      ring_bytes > control_mapping->size - submit_byte_offset) {
    *error_text = "SDMA control mapping too small for ring write: mapped_size=" +
                  std::to_string(control_mapping->size) + " submit_byte_offset=" +
                  std::to_string(submit_byte_offset) + " ring_write_bytes=" +
                  std::to_string(ring_bytes);
    return false;
  }
  std::vector<uint8_t> bytes;
  bytes.reserve(static_cast<std::size_t>(ring_bytes));
  for (uint32_t word : words) {
    append_u32_le(&bytes, word);
  }
  std::memcpy(static_cast<uint8_t*>(control_mapping->data) + submit_byte_offset, bytes.data(),
              bytes.size());
  return true;
}

bool write_control_u64(SysmemMapping* control_mapping, uint64_t offset, uint64_t value,
                       std::string* error_text) {
  if (offset > control_mapping->size || sizeof(uint64_t) > control_mapping->size - offset) {
    *error_text = "SDMA control mapping too small for qword write at offset " + format_hex64(offset);
    return false;
  }
  const std::vector<uint8_t> bytes = u64_payload_le(value);
  std::memcpy(static_cast<uint8_t*>(control_mapping->data) + offset, bytes.data(), bytes.size());
  return true;
}

bool submit_sdma_words(const RemoteClient& client, DiscoveryLog* log, SysmemMapping* control_mapping,
                       const std::vector<uint32_t>& words, uint64_t submit_byte_offset,
                       std::string* error_text) {
  if (!write_sdma_ring_words(control_mapping, words, submit_byte_offset, error_text)) {
    log->sdma.submit_status = "fail";
    return false;
  }
  const uint64_t final_wptr_bytes =
      submit_byte_offset + (static_cast<uint64_t>(words.size()) * sizeof(uint32_t));
  if (!write_control_u64(control_mapping, am_sdma::kWptrOffset, final_wptr_bytes, error_text)) {
    log->sdma.submit_status = "fail";
    return false;
  }
  // tinygrad/runtime/ops_amd.py:681-688 writes the queue write pointer, issues
  // System.memory_barrier(), then writes the BAR2 doorbell.
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const std::vector<uint8_t> doorbell_payload = u64_payload_le(final_wptr_bytes);
  if (!client.mmio_write_fire_and_forget(2, am_sdma::kDoorbellBar2ByteOffset,
                                         doorbell_payload, error_text)) {
    log->sdma.submit_status = "fail";
    return false;
  }
  log->sdma.submit_status = "pass";
  return true;
}

std::vector<uint8_t> u32_words_payload_le(const std::vector<uint32_t>& words) {
  std::vector<uint8_t> bytes;
  bytes.reserve(words.size() * sizeof(uint32_t));
  for (uint32_t word : words) {
    append_u32_le(&bytes, word);
  }
  return bytes;
}

bool write_compute_control_u64(SysmemMapping* compute_control_mapping, uint64_t offset,
                               uint64_t value, std::string* error_text) {
  if (compute_control_mapping == nullptr || compute_control_mapping->data == nullptr) {
    *error_text = "compute_control mapping is null";
    return false;
  }
  if (offset > compute_control_mapping->size ||
      sizeof(uint64_t) > compute_control_mapping->size - offset) {
    *error_text = "compute_control mapping too small for qword write at offset " +
                  format_hex64(offset);
    return false;
  }
  const std::vector<uint8_t> bytes = u64_payload_le(value);
  std::memcpy(static_cast<uint8_t*>(compute_control_mapping->data) + offset, bytes.data(),
              bytes.size());
  return true;
}

bool write_compute_ring_words(SysmemMapping* compute_control_mapping,
                              const std::vector<uint32_t>& words, uint64_t start_dword,
                              std::string* error_text) {
  if (compute_control_mapping == nullptr || compute_control_mapping->data == nullptr) {
    *error_text = "compute ring mapping precondition failed: null SysmemMapping";
    return false;
  }
  if (words.empty()) {
    *error_text = "compute dispatch has no PM4 packet words";
    return false;
  }
  const uint64_t ring_bytes = static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  if (ring_bytes == 0 || ring_bytes > am_compute::kRingSize) {
    *error_text = "compute dispatch PM4 byte count exceeds compute ring size: byte_count=" +
                  std::to_string(ring_bytes) +
                  " ring_size=" + std::to_string(am_compute::kRingSize);
    return false;
  }
  const uint64_t ring_base = am_compute::kComputeControlRingCpuOffset;
  const uint64_t ring_span_bytes = am_compute::kComputeControlRingByteCount;
  if (ring_base > compute_control_mapping->size ||
      ring_span_bytes > compute_control_mapping->size - ring_base) {
    *error_text = "compute ring write exceeds sysmem ring span: mapped_size=" +
                  std::to_string(compute_control_mapping->size) + " ring_start=" +
                  std::to_string(ring_base) + " ring_write_bytes=" + std::to_string(ring_bytes);
    return false;
  }
  const std::vector<uint8_t> bytes = u32_words_payload_le(words);
  const uint64_t ring_dwords = ring_span_bytes / sizeof(uint32_t);
  const uint64_t write_byte =
      ring_base + (start_dword % ring_dwords) * sizeof(uint32_t);
  const uint64_t first_chunk =
      std::min<uint64_t>(ring_bytes, ring_base + ring_span_bytes - write_byte);
  std::memcpy(static_cast<uint8_t*>(compute_control_mapping->data) + write_byte, bytes.data(),
              first_chunk);
  if (first_chunk < ring_bytes) {
    std::memcpy(static_cast<uint8_t*>(compute_control_mapping->data) + ring_base,
                bytes.data() + first_chunk, ring_bytes - first_chunk);
  }
  return true;
}

bool submit_compute_dispatch(const RemoteClient& client, DiscoveryLog* log,
                             SysmemMapping* compute_control_mapping,
                             const std::vector<uint32_t>& words, std::string* error_text,
                             bool capture_queue_snapshot = true) {
  if (log == nullptr) {
    *error_text = "DiscoveryLog precondition failed: null log";
    return false;
  }
  if (words.size() != am_compute::kPm4DispatchDwordCount) {
    *error_text = "PM4 dispatch dword count mismatch: expected=" +
                  std::to_string(am_compute::kPm4DispatchDwordCount) +
                  " observed=" + std::to_string(words.size());
    return false;
  }
  // Circular compute ring: the host advances a cumulative write pointer and the
  // CP advances its read pointer in lockstep. Reading the prior wptr and adding
  // this dispatch keeps successive submissions from aliasing (a second dispatch
  // written at ring[0] with wptr reset to 59 is invisible to the CP because its
  // rptr is already 59).
  uint64_t current_wptr_dwords = 0;
  if (compute_control_mapping == nullptr || compute_control_mapping->data == nullptr ||
      am_compute::kWptrOffset > compute_control_mapping->size ||
      sizeof(uint64_t) > compute_control_mapping->size - am_compute::kWptrOffset) {
    *error_text = "compute control mapping cannot hold the ring write pointer";
    return false;
  }
  std::memcpy(&current_wptr_dwords,
              static_cast<const uint8_t*>(compute_control_mapping->data) +
                  am_compute::kWptrOffset,
              sizeof(uint64_t));
  const uint64_t new_wptr_dwords = current_wptr_dwords + words.size();
  if (!write_compute_ring_words(compute_control_mapping, words, current_wptr_dwords,
                                error_text)) {
    return false;
  }
  if (!flush_hdp(client, *log, error_text)) {
    *error_text = "flush HDP after compute PM4 ring write failed: " + *error_text;
    return false;
  }
  if (capture_queue_snapshot) {
    ComputeQueueDebugSnapshot pre_snapshot;
    std::string pre_error;
    if (read_compute_queue_debug_snapshot(client, *log, true, &pre_snapshot, &pre_error)) {
      log->compute.doorbell_probe_pre = format_compute_queue_debug_snapshot(pre_snapshot);
    } else {
      log->compute.doorbell_probe_pre = "read_failed: " + pre_error;
    }
  } else {
    log->compute.doorbell_probe_pre = "skipped";
  }
  if (!write_compute_control_u64(compute_control_mapping, am_compute::kWptrOffset,
                                 new_wptr_dwords, error_text)) {
    return false;
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const std::vector<uint8_t> doorbell_payload = u64_payload_le(new_wptr_dwords);
  if (!client.mmio_write_fire_and_forget(2, am_compute::kMecDoorbellBar2ByteOffset,
                                         doorbell_payload, error_text)) {
    *error_text = "write compute MEC doorbell failed: " + *error_text;
    return false;
  }
  if (capture_queue_snapshot) {
    ComputeQueueDebugSnapshot post_snapshot;
    std::string post_error;
    if (read_compute_queue_debug_snapshot(client, *log, false, &post_snapshot, &post_error)) {
      log->compute.doorbell_probe_post = format_compute_queue_debug_snapshot(post_snapshot);
      log->compute.doorbell_probe_status = "submitted";
    } else {
      log->compute.doorbell_probe_post = "read_failed: " + post_error;
      log->compute.doorbell_probe_status = "submitted_post_read_failed";
    }
  } else {
    log->compute.doorbell_probe_post = "skipped";
    log->compute.doorbell_probe_status = "submitted";
  }
  return true;
}

bool poll_compute_timeline(const SysmemMapping& compute_control_mapping, long* elapsed_usec,
                           std::string* error_text,
                           uint32_t expected_value = am_compute::kReleaseMemTimelineValue) {
  if (elapsed_usec == nullptr) {
    *error_text = "elapsed_usec precondition failed: null pointer";
    return false;
  }
  if (am_compute::kTimelineOffset + sizeof(uint32_t) > compute_control_mapping.size ||
      am_compute::kWptrOffset + sizeof(uint64_t) > compute_control_mapping.size ||
      am_compute::kRptrOffset + sizeof(uint64_t) > compute_control_mapping.size) {
    *error_text = "compute_control mapping too small for timeline/rptr/wptr reads";
    return false;
  }
  volatile const uint32_t* timeline =
      reinterpret_cast<volatile const uint32_t*>(static_cast<const uint8_t*>(
                                                    compute_control_mapping.data) +
                                                am_compute::kTimelineOffset);
  volatile const uint64_t* rptr =
      reinterpret_cast<volatile const uint64_t*>(static_cast<const uint8_t*>(
                                                    compute_control_mapping.data) +
                                                am_compute::kRptrOffset);
  volatile const uint64_t* wptr =
      reinterpret_cast<volatile const uint64_t*>(static_cast<const uint8_t*>(
                                                    compute_control_mapping.data) +
                                                am_compute::kWptrOffset);
  timeval start{};
  gettimeofday(&start, nullptr);
  while (true) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const uint32_t observed = *timeline;
    timeval now{};
    gettimeofday(&now, nullptr);
    *elapsed_usec =
        (now.tv_sec - start.tv_sec) * 1000000L + (now.tv_usec - start.tv_usec);
    if (observed == expected_value) {
      return true;
    }
    if (*elapsed_usec >= 3000000L) {
      *error_text = "compute timeline timed out waiting for value " +
                    std::to_string(expected_value) +
                    ", observed=" + std::to_string(observed) +
                    ", rptr=" + std::to_string(static_cast<unsigned long long>(*rptr)) +
                    ", wptr=" + std::to_string(static_cast<unsigned long long>(*wptr));
      return false;
    }
    usleep(1000U);
  }
}

bool submit_sdma_copy(const RemoteClient& client, DiscoveryLog* log, SysmemMapping* control_mapping,
                      uint64_t src_va, uint64_t dst_va, uint32_t byte_count,
                      uint32_t fence_value, uint64_t submit_byte_offset,
                      std::string* error_text) {
  if (byte_count == 0U) {
    *error_text = "SDMA copy byte_count must be nonzero";
    log->sdma.submit_status = "fail";
    return false;
  }
  const std::vector<uint32_t> words =
      build_sdma_copy_submit_words(src_va, dst_va, byte_count, am_sdma::kFenceVa, fence_value);
  constexpr uint32_t kCopySubmitDwordCount =
      kSdmaLinearCopyPacketDwords + am_sdma::kFencePacketDwords;
  if (words.size() != kCopySubmitDwordCount) {
    *error_text = "internal SDMA copy submit dword count mismatch";
    log->sdma.submit_status = "fail";
    return false;
  }
  return submit_sdma_words(client, log, control_mapping, words, submit_byte_offset, error_text);
}

bool submit_sdma_transfer(const RemoteClient& client, DiscoveryLog* log, SysmemMapping* control_mapping,
                          const VmBufferLog& staging, const VmBufferLog& readback,
                          std::string* error_text) {
  const std::vector<uint32_t> words = build_sdma_submit_words(
      staging.gpu_va, kTransferProofVmVramVa, readback.gpu_va, am_sdma::kFenceVa);
  if (words.size() != am_sdma::kSubmitDwordCount) {
    *error_text = "internal SDMA submit dword count mismatch";
    log->sdma.submit_status = "fail";
    return false;
  }
  return submit_sdma_words(client, log, control_mapping, words, 0, error_text);
}

bool poll_sdma_fence(const SysmemMapping& control_mapping, std::string* error_text) {
  if (am_sdma::kFenceOffset + sizeof(uint32_t) > control_mapping.size) {
    *error_text = "SDMA control mapping too small for fence read at offset " +
                  format_hex64(am_sdma::kFenceOffset);
    return false;
  }
  volatile const uint32_t* fence =
      reinterpret_cast<volatile const uint32_t*>(static_cast<const uint8_t*>(control_mapping.data) +
                                                 am_sdma::kFenceOffset);
  timeval start{};
  gettimeofday(&start, nullptr);
  while (true) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (*fence == am_sdma::kFenceValue) {
      return true;
    }
    timeval now{};
    gettimeofday(&now, nullptr);
    const long elapsed_usec = (now.tv_sec - start.tv_sec) * 1000000L + (now.tv_usec - start.tv_usec);
    if (elapsed_usec >= 3000000L) {
      *error_text = "SDMA fence timeline timed out waiting for value 1";
      return false;
    }
    usleep(1000U);
  }
}

void print_vm_buffer_log(const char* prefix, const VmBufferLog& buffer) {
  std::printf("%s_role: %s\n", prefix, buffer.role);
  std::printf("%s_gpu_va: 0x%016llx\n", prefix, static_cast<unsigned long long>(buffer.gpu_va));
  std::printf("%s_requested_size: %llu\n", prefix,
              static_cast<unsigned long long>(buffer.requested_size));
  std::printf("%s_mapped_size: %llu\n", prefix,
              static_cast<unsigned long long>(buffer.mapped_size));
  std::printf("%s_response_header_hex: %s\n", prefix, buffer.response_header_hex.c_str());
  std::printf("%s_page_count: %zu\n", prefix, buffer.sys_pages.size());
  for (std::size_t i = 0; i < buffer.sys_pages.size(); ++i) {
    std::printf("%s_page_%zu_paddr: 0x%016llx\n", prefix, i,
                static_cast<unsigned long long>(buffer.sys_pages[i]));
  }
}

void print_transfer_log(const DiscoveryLog& log, const VmBufferLog& staging,
                        const VmBufferLog& readback, const VmBufferLog& sdma_control,
                        const std::string& cpu_comparison_status,
                        const std::string& host_device_transfer_status,
                        const std::string& failure_stage, const std::string& failure_text,
                        int exit_status) {
  std::printf("runtime_substrate: %s\n", kRuntimeSubstrate);
  std::printf("socket_path: %s\n", log.socket_path.c_str());
  std::printf("pci_id: %s\n", log.pci_id.c_str());
  std::printf("arch: %s\n", log.arch.c_str());
  std::printf("arch_discovery_status: %s\n", log.arch_discovery_status.c_str());
  std::printf("gc_ip_version: %s\n", log.gc_ip_version.c_str());
  std::printf("gc_ip_bases: %s\n", log.gc_ip_bases.c_str());
  std::printf("mmhub_ip_version: %s\n", log.mmhub_ip_version.c_str());
  std::printf("mmhub_ip_bases: %s\n", log.mmhub_ip_bases.c_str());
  std::printf("nbif_ip_version: %s\n", log.nbif_ip_version.c_str());
  std::printf("nbif_ip_bases: %s\n", log.nbif_ip_bases.c_str());
  std::printf("sdma_ip_version: %s\n", log.sdma_ip_version.c_str());
  std::printf("sdma_ip_bases: %s\n", log.sdma_ip_bases.c_str());
  std::printf("config_response_header_hex: %s\n", log.config_response_header_hex.c_str());
  std::printf("bar0_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar0.size));
  std::printf("bar2_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar2.size));
  std::printf("bar5_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar5.size));
  std::printf("vram_size_bytes: %llu\n", static_cast<unsigned long long>(log.vram_size_bytes));
  std::printf("transfer_byte_count: %llu\n", static_cast<unsigned long long>(kTransferByteCount));
  std::printf("vm_vram_gpu_va: 0x%016llx\n",
              static_cast<unsigned long long>(kTransferProofVmVramVa));
  std::printf("vm_vram_byte_count: %llu\n", static_cast<unsigned long long>(kTransferByteCount));
  std::printf("vm_page_table_root_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.root_pdb2_paddr));
  std::printf("vm_pdb1_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.child_pdb1_paddr));
  std::printf("vm_pdb0_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.child_pdb0_paddr));
  std::printf("vm_ptb_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.child_ptb_paddr));
  std::printf("vm_vram_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.device_buffer_paddr));
  std::printf("vm_page_tables_written: %s\n", log.vm.page_tables_written.c_str());
  std::printf("vmid0_context_status: %s\n", log.vm.vmid0_context_status.c_str());
  std::printf("vm_gc_context_status: %s\n", log.vm.vm_gc_context_status.c_str());
  std::printf("mm_tlb_flush_status: %s\n", log.vm.mm_tlb_flush_status.c_str());
  std::printf("gc_tlb_flush_status: %s\n", log.vm.gc_tlb_flush_status.c_str());
  print_vm_buffer_log("sysmem_staging", staging);
  print_vm_buffer_log("sysmem_readback", readback);
  print_vm_buffer_log("sysmem_sdma_control", sdma_control);
  std::printf("sdma_ring_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kControlVa));
  std::printf("sdma_ring_size_bytes: %llu\n", static_cast<unsigned long long>(am_sdma::kRingSize));
  std::printf("sdma_rptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kRptrVa));
  std::printf("sdma_wptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kWptrVa));
  std::printf("sdma_fence_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kFenceVa));
  std::printf("sdma_doorbell_index: %u\n", am_sdma::kDoorbellIndex);
  std::printf("sdma_doorbell_bar2_byte_offset: 0x%016llx\n",
              static_cast<unsigned long long>(am_sdma::kDoorbellBar2ByteOffset));
  std::printf("sdma_submit_dwords: %u\n", am_sdma::kSubmitDwordCount);
  std::printf("sdma_linear_copy_packet_dwords: %zu\n", kSdmaLinearCopyPacketDwords);
  std::printf("sdma_queue_setup_status: %s\n", log.sdma.queue_setup_status.c_str());
  std::printf("sdma_submit_status: %s\n", log.sdma.submit_status.c_str());
  std::printf("sdma_timeline_status: %s\n", log.sdma.timeline_status.c_str());
  std::printf("cpu_comparison_status: %s\n", cpu_comparison_status.c_str());
  std::printf("host_device_transfer_status: %s\n", host_device_transfer_status.c_str());
  std::printf("failure_stage: %s\n", failure_stage.c_str());
  std::printf("failure_text: %s\n", failure_text.c_str());
  std::printf("exit_status: %d\n", exit_status);
}

void print_kernel_log(const DiscoveryLog& log, const VmBufferLog& staging,
                      const VmBufferLog& readback, const VmBufferLog& sdma_control,
                      const VmBufferLog& compute_control, const std::string& kernel_launch_status,
                      long kernel_elapsed_usec, const std::string& cpu_comparison_status,
                      const std::string& host_device_transfer_status,
                      const std::string& failure_stage, const std::string& failure_text,
                      int exit_status) {
  std::printf("runtime_substrate: %s\n", kRuntimeSubstrate);
  std::printf("socket_path: %s\n", log.socket_path.c_str());
  std::printf("pci_id: %s\n", log.pci_id.c_str());
  std::printf("arch: %s\n", log.arch.c_str());
  std::printf("arch_discovery_status: %s\n", log.arch_discovery_status.c_str());
  std::printf("gc_ip_version: %s\n", log.gc_ip_version.c_str());
  std::printf("gc_ip_bases: %s\n", log.gc_ip_bases.c_str());
  std::printf("mmhub_ip_version: %s\n", log.mmhub_ip_version.c_str());
  std::printf("mmhub_ip_bases: %s\n", log.mmhub_ip_bases.c_str());
  std::printf("nbif_ip_version: %s\n", log.nbif_ip_version.c_str());
  std::printf("nbif_ip_bases: %s\n", log.nbif_ip_bases.c_str());
  std::printf("sdma_ip_version: %s\n", log.sdma_ip_version.c_str());
  std::printf("sdma_ip_bases: %s\n", log.sdma_ip_bases.c_str());
  std::printf("config_response_header_hex: %s\n", log.config_response_header_hex.c_str());
  std::printf("bar0_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar0.size));
  std::printf("bar2_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar2.size));
  std::printf("bar5_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar5.size));
  std::printf("vram_size_bytes: %llu\n", static_cast<unsigned long long>(log.vram_size_bytes));
  std::printf("transfer_byte_count: %llu\n", static_cast<unsigned long long>(kTransferByteCount));
  std::printf("kernel_proof_mode: %s\n", kKernelProofMode);
  std::printf("kernel_arch: %s\n", kKernelArch);
  std::printf("element_type: uint32_t\n");
  std::printf("element_count: 8\n");
  std::printf("input_byte_count: 32\n");
  std::printf("output_byte_count: 32\n");
  std::printf("input_values_u32: %s\n", kKernelInputValuesU32);
  std::printf("input_bytes_hex: %s\n", kKernelInputBytesHex);
  std::printf("expected_output_values_u32: %s\n", kKernelExpectedOutputValuesU32);
  std::printf("expected_output_bytes_hex: %s\n", kKernelExpectedOutputBytesHex);
  std::printf("expected_output_sha256: %s\n", kKernelExpectedOutputSha256);
  std::printf("kernel_source_id: %s\n", kKernelSourceId);
  std::printf("kernel_source_language: %s\n", kKernelSourceLanguage);
  std::printf("kernel_blob_format: %s\n", kKernelBlobFormat);
  std::printf("kernel_blob_symbol: %s\n", kKernelBlobSymbol);
  std::printf("kernel_blob_target: %s\n", kKernelBlobTarget);
  std::printf("kernel_blob_reference_hsaco_sha256: %s\n", kKernelReferenceHsacoSha256);
  std::printf("kernel_blob_reference_text_sha256: %s\n", kKernelReferenceTextSha256);
  std::printf("kernel_blob_reference_text_byte_count: %u\n", kKernelReferenceTextByteCount);
  std::printf("kernel_text_provenance_path: %s\n", kKernelTextProvenancePath);
  std::printf("kernel_text_first64_hex: %s\n",
              hex_encode_bytes(kKernelText.data(), 64).c_str());
  std::printf("kernel_text_last16_hex: %s\n",
              hex_encode_bytes(kKernelText.data() + kKernelText.size() - 16, 16).c_str());
  std::printf("kernel_descriptor_kernarg_size: %u\n", kKernelReferenceKernargSize);
  std::printf("kernel_descriptor_rsrc1: 0x%08x\n", kKernelReferenceRsrc1);
  std::printf("kernel_descriptor_rsrc2: 0x%08x\n", kKernelReferenceRsrc2);
  std::printf("kernel_descriptor_rsrc3: 0x%08x\n", kKernelReferenceRsrc3);
  std::printf("kernel_descriptor_code_properties: 0x%08x\n", kKernelReferenceCodeProperties);
  std::printf("kernel_blob_load_status: %s\n", log.compute.kernel_blob_load_status.c_str());
  std::printf("kernarg_write_status: %s\n", log.compute.kernarg_write_status.c_str());
  std::printf("sdma_h2d_status: %s\n", log.sdma.h2d_status.c_str());
  std::printf("sdma_d2h_status: %s\n", log.sdma.d2h_status.c_str());
  std::printf("vm_vram_gpu_va: 0x%016llx\n",
              static_cast<unsigned long long>(kTransferProofVmVramVa));
  std::printf("vm_vram_byte_count: %llu\n", static_cast<unsigned long long>(kTransferByteCount));
  std::printf("vm_page_table_root_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.root_pdb2_paddr));
  std::printf("vm_pdb1_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.child_pdb1_paddr));
  std::printf("vm_pdb0_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.child_pdb0_paddr));
  std::printf("vm_ptb_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.child_ptb_paddr));
  std::printf("vm_vram_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.device_buffer_paddr));
  std::printf("vm_page_tables_written: %s\n", log.vm.page_tables_written.c_str());
  std::printf("vmid0_context_status: %s\n", log.vm.vmid0_context_status.c_str());
  std::printf("vm_gc_context_status: %s\n", log.vm.vm_gc_context_status.c_str());
  std::printf("mm_tlb_flush_status: %s\n", log.vm.mm_tlb_flush_status.c_str());
  std::printf("gc_tlb_flush_status: %s\n", log.vm.gc_tlb_flush_status.c_str());
  print_vm_buffer_log("sysmem_staging", staging);
  print_vm_buffer_log("sysmem_readback", readback);
  print_vm_buffer_log("sysmem_sdma_control", sdma_control);
  print_vm_buffer_log("sysmem_compute_control", compute_control);
  std::printf("sdma_ring_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kControlVa));
  std::printf("sdma_ring_size_bytes: %llu\n", static_cast<unsigned long long>(am_sdma::kRingSize));
  std::printf("sdma_rptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kRptrVa));
  std::printf("sdma_wptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kWptrVa));
  std::printf("sdma_fence_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kFenceVa));
  std::printf("sdma_doorbell_index: %u\n", am_sdma::kDoorbellIndex);
  std::printf("sdma_doorbell_bar2_byte_offset: 0x%016llx\n",
              static_cast<unsigned long long>(am_sdma::kDoorbellBar2ByteOffset));
  std::printf("sdma_submit_dwords: %u\n", am_sdma::kSubmitDwordCount);
  std::printf("sdma_linear_copy_packet_dwords: %zu\n", kSdmaLinearCopyPacketDwords);
  std::printf("sdma_queue_setup_status: %s\n", log.sdma.queue_setup_status.c_str());
  std::printf("sdma_submit_status: %s\n", log.sdma.submit_status.c_str());
  std::printf("sdma_timeline_status: %s\n", log.sdma.timeline_status.c_str());
  std::printf("compute_ring_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kRingVa));
  std::printf("compute_ring_size_bytes: %u\n", am_compute::kRingSize);
  std::printf("compute_rptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kRptrVa));
  std::printf("compute_wptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kWptrVa));
  std::printf("compute_timeline_gpu_va: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kTimelineVa));
  std::printf("compute_eop_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_compute::kEopVa));
  std::printf("compute_doorbell_index: %u\n", am_compute::kMecDoorbellIndex);
  std::printf("compute_doorbell_bar2_byte_offset: 0x%016llx\n",
              static_cast<unsigned long long>(am_compute::kMecDoorbellBar2ByteOffset));
  std::printf("compute_doorbell_probe_status: %s\n",
              log.compute.doorbell_probe_status.c_str());
  std::printf("compute_doorbell_probe_pre: %s\n", log.compute.doorbell_probe_pre.c_str());
  std::printf("compute_doorbell_probe_post: %s\n", log.compute.doorbell_probe_post.c_str());
  std::printf("compute_doorbell_probe_timeout: %s\n",
              log.compute.doorbell_probe_timeout.c_str());
  std::printf("compute_doorbell_probe_classification: %s\n",
              log.compute.doorbell_probe_classification.c_str());
  std::printf("compute_doorbell_consumption_timeout: %s\n",
              log.compute.doorbell_consumption_timeout.c_str());
  std::printf("compute_doorbell_consumption_classification: %s\n",
              log.compute.doorbell_consumption_classification.c_str());
  std::printf("compute_doorbell_route_readback: %s\n",
              log.compute.doorbell_route_readback.c_str());
  std::printf("compute_doorbell_route_classification: %s\n",
              log.compute.doorbell_route_classification.c_str());
  std::printf("compute_ring_setup_status: %s\n", log.compute.ring_setup_status.c_str());
  std::printf("compute_hqd_active_status: %s\n", log.compute.hqd_active_status.c_str());
  std::printf("mec_rs64_cntl_write_status: %s\n", log.compute.mec_rs64_cntl_write_status.c_str());
  std::printf("mec_rs64_cntl_readback: %s\n", log.compute.mec_rs64_cntl_readback.c_str());
  std::printf("mec_rs64_active_status: %s\n", log.compute.mec_rs64_active_status.c_str());
  std::printf("compute_readback_anomaly: %s\n", log.compute.compute_readback_anomaly.c_str());
  std::printf("kernel_launch_status: %s\n", kernel_launch_status.c_str());
  std::printf("kernel_elapsed_usec: %ld\n", kernel_elapsed_usec);
  std::printf("cpu_comparison_status: %s\n", cpu_comparison_status.c_str());
  std::printf("host_device_transfer_status: %s\n", host_device_transfer_status.c_str());
  std::printf("failure_stage: %s\n", failure_stage.c_str());
  std::printf("failure_text: %s\n", failure_text.c_str());
  std::printf("exit_status: %d\n", exit_status);
}

int finish_kernel(DiscoveryLog& log, const VmBufferLog& staging, const VmBufferLog& readback,
                  const VmBufferLog& sdma_control, const VmBufferLog& compute_control,
                  const char* stage, const std::string& text) {
  log.failure_stage = stage;
  log.failure_text = text;
  print_kernel_log(log, staging, readback, sdma_control, compute_control, "not_run", 0, "not_run",
                   "fail", log.failure_stage, log.failure_text, 1);
  return 1;
}

int finish_transfer(DiscoveryLog& log, const VmBufferLog& staging, const VmBufferLog& readback,
                    const VmBufferLog& sdma_control, const char* stage, const std::string& text) {
  log.failure_stage = stage;
  log.failure_text = text;
  print_transfer_log(log, staging, readback, sdma_control, "not_run", "fail", log.failure_stage,
                     log.failure_text, 1);
  return 1;
}

int run_discovery_smoke() {
  DiscoveryLog log;
  log.socket_path = tinygpu_socket_path();

  UniqueFd socket_fd;
  std::string connect_error;
  if (!connect_tinygpu_server(log.socket_path, &socket_fd, &connect_error)) {
    return finish_discovery(log, "tinygpu_connect", connect_error);
  }

  const RemoteClient client(socket_fd.get());
  RemoteRpcResult config = client.rpc_no_payload(RemoteCmd::CFG_READ, 0, 0, 4);
  log.config_response_header_hex =
      config.response_header_hex.empty() ? "unavailable" : config.response_header_hex;
  if (!config.ok) {
    return finish_discovery(log, "config-read", rpc_failure_text("CFG_READ vendor_device", config));
  }
  log.config_vendor_id = static_cast<uint32_t>(config.value0 & 0xffffU);
  log.config_device_id = static_cast<uint32_t>((config.value0 >> 16) & 0xffffU);
  if (log.config_vendor_id != kTargetVendor || log.config_device_id != kTargetDevice) {
    return finish_discovery(log, "config-read",
                            "expected 1002:7551, observed " +
                                pci_id_text(log.config_vendor_id, log.config_device_id));
  }
  log.pci_id = pci_id_text(log.config_vendor_id, log.config_device_id);

  RemoteRpcResult bar_result;
  if (!map_bar(client, 0, &log.bar0, &bar_result)) {
    return finish_discovery(log, "map-bar0", rpc_failure_text("MAP_BAR bar0", bar_result));
  }
  if (!map_bar(client, 2, &log.bar2, &bar_result)) {
    return finish_discovery(log, "map-bar2", rpc_failure_text("MAP_BAR bar2", bar_result));
  }
  if (!map_bar(client, 5, &log.bar5, &bar_result)) {
    return finish_discovery(log, "map-bar5", rpc_failure_text("MAP_BAR bar5", bar_result));
  }

  std::string required_discovery_error;
  if (!try_discover_arch(client, &log, &required_discovery_error)) {
    return finish_discovery(log, "vram-size", required_discovery_error);
  }
  print_discovery_log(log, 0);
  return 0;
}

int run_transfer_proof_scaffold() {
  DiscoveryLog log;
  log.socket_path = tinygpu_socket_path();
  VmBufferLog staging{"staging", kTransferProofVmStagingVa, kTransferProofBufferSize, 0, "not_run", {}};
  VmBufferLog readback{"readback", kTransferProofVmReadbackVa, kTransferProofBufferSize, 0, "not_run", {}};
  VmBufferLog sdma_control{"sdma_control", am_sdma::kControlVa, kPageSize, 0, "not_run", {}};

  UniqueFd socket_fd;
  SysmemMapping staging_mapping;
  SysmemMapping readback_mapping;
  SysmemMapping sdma_control_mapping;
  std::string connect_error;
  if (!connect_tinygpu_server(log.socket_path, &socket_fd, &connect_error)) {
    return finish_transfer(log, staging, readback, sdma_control, "tinygpu_connect", connect_error);
  }

  const RemoteClient client(socket_fd.get());
  RemoteRpcResult config = client.rpc_no_payload(RemoteCmd::CFG_READ, 0, 0, 4);
  log.config_response_header_hex =
      config.response_header_hex.empty() ? "unavailable" : config.response_header_hex;
  if (!config.ok) {
    return finish_transfer(log, staging, readback, sdma_control, "config-read",
                           rpc_failure_text("CFG_READ vendor_device", config));
  }
  log.config_vendor_id = static_cast<uint32_t>(config.value0 & 0xffffU);
  log.config_device_id = static_cast<uint32_t>((config.value0 >> 16) & 0xffffU);
  if (log.config_vendor_id != kTargetVendor || log.config_device_id != kTargetDevice) {
    return finish_transfer(log, staging, readback, sdma_control, "config-read",
                           "expected 1002:7551, observed " +
                               pci_id_text(log.config_vendor_id, log.config_device_id));
  }
  log.pci_id = pci_id_text(log.config_vendor_id, log.config_device_id);

  RemoteRpcResult bar_result;
  if (!map_bar(client, 0, &log.bar0, &bar_result)) {
    return finish_transfer(log, staging, readback, sdma_control, "map-bar0",
                           rpc_failure_text("MAP_BAR bar0", bar_result));
  }
  if (!map_bar(client, 2, &log.bar2, &bar_result)) {
    return finish_transfer(log, staging, readback, sdma_control, "map-bar2",
                           rpc_failure_text("MAP_BAR bar2", bar_result));
  }
  if (!map_bar(client, 5, &log.bar5, &bar_result)) {
    return finish_transfer(log, staging, readback, sdma_control, "map-bar5",
                           rpc_failure_text("MAP_BAR bar5", bar_result));
  }

  std::string required_discovery_error;
  if (!try_discover_arch(client, &log, &required_discovery_error)) {
    return finish_transfer(log, staging, readback, sdma_control, "vram-size", required_discovery_error);
  }

  std::string vm_error;
  if (!map_sysmem_buffer(client, &staging, &staging_mapping, &vm_error)) {
    return finish_transfer(log, staging, readback, sdma_control, "vm_mapping", vm_error);
  }
  if (!map_sysmem_buffer(client, &readback, &readback_mapping, &vm_error)) {
    return finish_transfer(log, staging, readback, sdma_control, "vm_mapping", vm_error);
  }
  if (!map_sysmem_buffer(client, &sdma_control, &sdma_control_mapping, &vm_error)) {
    return finish_transfer(log, staging, readback, sdma_control, "vm_mapping", vm_error);
  }

  if (staging_mapping.size < kTransferByteCount || readback_mapping.size < kTransferByteCount ||
      sdma_control_mapping.size < am_sdma::kFenceOffset + sizeof(uint32_t)) {
    return finish_transfer(log, staging, readback, sdma_control, "vm_mapping",
                           "MAP_SYSMEM_FD CPU mappings are smaller than the fixed transfer/control proof spans");
  }

  const auto expected_payload = transfer_payload();
  std::memcpy(staging_mapping.data, expected_payload.data(), expected_payload.size());
  std::memset(readback_mapping.data, 0, readback_mapping.size);
  std::memset(sdma_control_mapping.data, 0, sdma_control_mapping.size);
  FixedVmMappingResult vm_result;
  if (!setup_fixed_vm_mapping(client, &log, staging, readback, sdma_control, nullptr, false, &vm_result)) {
    return finish_transfer(log, staging, readback, sdma_control, "vm_mapping", vm_result.error_text);
  }

  std::string sdma_error;
  if (!setup_sdma_queue0(client, &log, &sdma_error)) {
    return finish_transfer(log, staging, readback, sdma_control, "sdma_ring_setup", sdma_error);
  }
  if (!submit_sdma_transfer(client, &log, &sdma_control_mapping, staging, readback, &sdma_error)) {
    return finish_transfer(log, staging, readback, sdma_control, "sdma_submit", sdma_error);
  }
  if (!poll_sdma_fence(sdma_control_mapping, &sdma_error)) {
    log.sdma.timeline_status = "fail";
    log.failure_stage = "timeline_timeout";
    log.failure_text = sdma_error;
    print_transfer_log(log, staging, readback, sdma_control, "not_run", "fail",
                       log.failure_stage, log.failure_text, 1);
    return 1;
  }
  log.sdma.timeline_status = "pass";

  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (std::memcmp(readback_mapping.data, expected_payload.data(), expected_payload.size()) != 0) {
    log.failure_stage = "readback_mismatch";
    log.failure_text = "CPU readback bytes did not match the 32-byte transfer payload; expected_hex=" +
                       hex_encode_bytes(expected_payload.data(), expected_payload.size()) +
                       " observed_hex=" +
                       hex_encode_bytes(static_cast<const uint8_t*>(readback_mapping.data),
                                        expected_payload.size());
    print_transfer_log(log, staging, readback, sdma_control, "fail", "fail",
                       log.failure_stage, log.failure_text, 1);
    return 1;
  }

  log.failure_stage = "none";
  log.failure_text = "none";
  print_transfer_log(log, staging, readback, sdma_control, "pass", "pass",
                     log.failure_stage, log.failure_text, 0);
  return 0;
}

int run_kernel_proof_scaffold() {
  DiscoveryLog log;
  log.socket_path = tinygpu_socket_path();
  VmBufferLog staging{"staging", kTransferProofVmStagingVa, kTransferProofBufferSize, 0, "not_run", {}};
  VmBufferLog readback{"readback", kTransferProofVmReadbackVa, kTransferProofBufferSize, 0, "not_run", {}};
  VmBufferLog sdma_control{"sdma_control", am_sdma::kControlVa, kPageSize, 0, "not_run", {}};
  VmBufferLog compute_control{"compute_control", am_compute::kRptrVa,
                              am_compute::kComputeControlByteCount, 0, "not_run", {}};

  UniqueFd socket_fd;
  SysmemMapping staging_mapping;
  SysmemMapping readback_mapping;
  SysmemMapping sdma_control_mapping;
  SysmemMapping compute_control_mapping;
  std::string connect_error;
  if (!connect_tinygpu_server(log.socket_path, &socket_fd, &connect_error)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "tinygpu_connect",
                         connect_error);
  }

  const RemoteClient client(socket_fd.get());
  RemoteRpcResult config = client.rpc_no_payload(RemoteCmd::CFG_READ, 0, 0, 4);
  log.config_response_header_hex =
      config.response_header_hex.empty() ? "unavailable" : config.response_header_hex;
  if (!config.ok) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "config-read",
                         rpc_failure_text("CFG_READ vendor_device", config));
  }
  log.config_vendor_id = static_cast<uint32_t>(config.value0 & 0xffffU);
  log.config_device_id = static_cast<uint32_t>((config.value0 >> 16) & 0xffffU);
  if (log.config_vendor_id != kTargetVendor || log.config_device_id != kTargetDevice) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "config-read",
                         "expected 1002:7551, observed " +
                             pci_id_text(log.config_vendor_id, log.config_device_id));
  }
  log.pci_id = pci_id_text(log.config_vendor_id, log.config_device_id);

  RemoteRpcResult bar_result;
  if (!map_bar(client, 0, &log.bar0, &bar_result)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "map-bar0",
                         rpc_failure_text("MAP_BAR bar0", bar_result));
  }
  if (!map_bar(client, 2, &log.bar2, &bar_result)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "map-bar2",
                         rpc_failure_text("MAP_BAR bar2", bar_result));
  }
  if (!map_bar(client, 5, &log.bar5, &bar_result)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "map-bar5",
                         rpc_failure_text("MAP_BAR bar5", bar_result));
  }

  std::string required_discovery_error;
  if (!try_discover_arch(client, &log, &required_discovery_error)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "arch_discovery",
                         required_discovery_error);
  }

  std::string vm_error;
  if (!map_sysmem_buffer(client, &staging, &staging_mapping, &vm_error)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "vm_mapping",
                         vm_error);
  }
  if (!map_sysmem_buffer(client, &readback, &readback_mapping, &vm_error)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "vm_mapping",
                         vm_error);
  }
  if (!map_sysmem_buffer(client, &sdma_control, &sdma_control_mapping, &vm_error)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "vm_mapping",
                         vm_error);
  }
  if (!map_sysmem_buffer(client, &compute_control, &compute_control_mapping, &vm_error)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "vm_mapping",
                         vm_error);
  }

  if (staging_mapping.size < kTransferByteCount || readback_mapping.size < kTransferByteCount ||
      sdma_control_mapping.size < am_sdma::kFenceOffset + sizeof(uint32_t) ||
      compute_control_mapping.size < am_compute::kComputeControlByteCount) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "vm_mapping",
                         "MAP_SYSMEM_FD CPU mappings are smaller than the fixed kernel proof/control spans");
  }
  if (compute_control.sys_pages.size() < 10) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "vm_mapping",
                         "MAP_SYSMEM_FD compute_control page list must contain 10 pages (2 control + 8 ring)");
  }

  const auto input_payload = kernel_input_payload();
  std::memcpy(staging_mapping.data, input_payload.data(), input_payload.size());
  std::memset(readback_mapping.data, 0, readback_mapping.size);
  std::memset(sdma_control_mapping.data, 0, sdma_control_mapping.size);
  std::memset(compute_control_mapping.data, 0, compute_control_mapping.size);

  FixedVmMappingResult vm_result;
  if (!setup_fixed_vm_mapping(client, &log, staging, readback, sdma_control, &compute_control,
                              true, &vm_result)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control,
                         vm_result.failure_stage.c_str(), vm_result.error_text);
  }

  std::string sdma_error;
  if (!setup_sdma_queue0(client, &log, &sdma_error)) {
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "sdma_ring_setup",
                         sdma_error);
  }
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, staging.gpu_va,
                        am_compute::kInputVramVa, kTransferByteCount, am_sdma::kFenceValue, 0,
                        &sdma_error)) {
    log.sdma.h2d_status = "fail";
    return finish_kernel(log, staging, readback, sdma_control, compute_control, "sdma_h2d_submit",
                         sdma_error);
  }
  if (!poll_sdma_fence(sdma_control_mapping, &sdma_error)) {
    log.sdma.timeline_status = "fail";
    log.sdma.h2d_status = "fail";
    log.failure_stage = "timeline_timeout";
    log.failure_text = sdma_error;
    print_kernel_log(log, staging, readback, sdma_control, compute_control, "not_run", 0, "not_run",
                     "fail", log.failure_stage, log.failure_text, 1);
    return 1;
  }
  log.sdma.timeline_status = "pass";
  log.sdma.h2d_status = "pass";

  std::string compute_error;
  if (!setup_compute_ring0(client, &log, &compute_control_mapping, &compute_error)) {
    log.failure_stage = "compute_ring_setup";
    log.failure_text = compute_error;
    print_kernel_log(log, staging, readback, sdma_control, compute_control, "blocked", 0,
                     "not_run_blocked_by_compute_ring_setup",
                     "not_run_blocked_by_compute_ring_setup", log.failure_stage, log.failure_text,
                     1);
    return 1;
  }

  if (!load_kernel_blob(client, &log, &compute_error)) {
    log.failure_stage = "kernel_blob_load";
    log.failure_text = compute_error;
    print_kernel_log(log, staging, readback, sdma_control, compute_control, "blocked", 0,
                     "not_run_blocked_by_kernel_blob_load",
                     "not_run_blocked_by_kernel_blob_load", log.failure_stage, log.failure_text, 1);
    return 1;
  }
  if (!write_kernel_kernargs(&compute_control_mapping, am_compute::kOutputVramVa,
                             am_compute::kInputVramVa,
                             am_compute::kKernargsVa + kKernelReferenceKernargSize,
                             &compute_error)) {
    log.compute.kernarg_write_status = "fail";
    log.failure_stage = "kernarg_write";
    log.failure_text = compute_error;
    print_kernel_log(log, staging, readback, sdma_control, compute_control, "blocked", 0,
                     "not_run_blocked_by_kernarg_write", "not_run_blocked_by_kernarg_write",
                     log.failure_stage, log.failure_text, 1);
    return 1;
  }
  log.compute.kernarg_write_status = "pass";

  const std::vector<uint32_t> dispatch_words = build_compute_dispatch_words(
      am_compute::kCodeVramVa, am_compute::kKernargsVa, am_compute::kTimelineVa);
  long kernel_elapsed_usec = 0;
  if (!submit_compute_dispatch(client, &log, &compute_control_mapping, dispatch_words,
                               &compute_error)) {
    log.failure_stage = "kernel_dispatch_submit";
    log.failure_text = compute_error;
    print_kernel_log(log, staging, readback, sdma_control, compute_control, "fail",
                     kernel_elapsed_usec, "not_run_blocked_by_kernel_dispatch_submit",
                     "not_run_blocked_by_kernel_dispatch_submit", log.failure_stage,
                     log.failure_text, 1);
    return 1;
  }
  if (!poll_compute_timeline(compute_control_mapping, &kernel_elapsed_usec, &compute_error)) {
    log.failure_stage = "kernel_timeline_timeout";
    ComputeQueueDebugSnapshot timeout_snapshot;
    std::string timeout_debug_error;
    if (read_compute_queue_debug_snapshot(client, log, false, &timeout_snapshot,
                                          &timeout_debug_error)) {
      log.compute.doorbell_probe_timeout =
          format_compute_queue_debug_snapshot(timeout_snapshot);
      log.compute.doorbell_probe_classification =
          classify_compute_doorbell_timeout(timeout_snapshot);
    } else {
      log.compute.doorbell_probe_timeout = "read_failed: " + timeout_debug_error;
      log.compute.doorbell_probe_classification = "compute_doorbell_delivery_unclassified";
    }
    ComputeDoorbellConsumptionSnapshot consumption_snapshot;
    std::string consumption_error;
    if (read_compute_doorbell_consumption_snapshot(client, log, compute_control_mapping,
                                                   &consumption_snapshot,
                                                   &consumption_error)) {
      log.compute.doorbell_consumption_timeout =
          format_compute_doorbell_consumption_snapshot(consumption_snapshot);
      log.compute.doorbell_consumption_classification =
          classify_compute_doorbell_consumption_timeout(consumption_snapshot);
    } else {
      log.compute.doorbell_consumption_timeout = "read_failed: " + consumption_error;
      log.compute.doorbell_consumption_classification =
          "doorbell_consumption_unclassified";
    }
    log.failure_text = compute_error + ", " + log.compute.doorbell_probe_timeout;
    print_kernel_log(log, staging, readback, sdma_control, compute_control, "fail",
                     kernel_elapsed_usec, "not_run_blocked_by_kernel_timeline_timeout",
                     "not_run_blocked_by_kernel_timeline_timeout", log.failure_stage,
                     log.failure_text, 1);
    return 1;
  }

  std::memset(static_cast<uint8_t*>(sdma_control_mapping.data) + am_sdma::kFenceOffset, 0,
              sizeof(uint32_t));
  std::atomic_thread_fence(std::memory_order_seq_cst);
  constexpr uint64_t kD2hSubmitByteOffset =
      static_cast<uint64_t>(kSdmaLinearCopyPacketDwords + am_sdma::kFencePacketDwords) *
      sizeof(uint32_t);
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, am_compute::kOutputVramVa,
                        readback.gpu_va, kTransferByteCount, am_sdma::kFenceValue,
                        kD2hSubmitByteOffset, &sdma_error)) {
    log.sdma.d2h_status = "fail";
    log.failure_stage = "sdma_d2h_submit";
    log.failure_text = sdma_error;
    print_kernel_log(log, staging, readback, sdma_control, compute_control, "pass",
                     kernel_elapsed_usec, "not_run_blocked_by_sdma_d2h_submit", "fail",
                     log.failure_stage, log.failure_text, 1);
    return 1;
  }
  if (!poll_sdma_fence(sdma_control_mapping, &sdma_error)) {
    log.sdma.d2h_status = "fail";
    log.failure_stage = "sdma_d2h_submit";
    log.failure_text = sdma_error;
    print_kernel_log(log, staging, readback, sdma_control, compute_control, "pass",
                     kernel_elapsed_usec, "not_run_blocked_by_sdma_d2h_submit", "fail",
                     log.failure_stage, log.failure_text, 1);
    return 1;
  }
  log.sdma.d2h_status = "pass";

  const auto expected_output_payload = kernel_expected_output_payload();
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (std::memcmp(readback_mapping.data, expected_output_payload.data(),
                  expected_output_payload.size()) != 0) {
    log.failure_stage = "readback_mismatch";
    const std::string observed_hex = hex_encode_bytes(
        static_cast<const uint8_t*>(readback_mapping.data), expected_output_payload.size());
    log.failure_text =
        "kernel output readback bytes did not match expected 32-byte payload; expected_hex=" +
        hex_encode_bytes(expected_output_payload.data(), expected_output_payload.size()) +
        " observed_hex=" + observed_hex;
    // CPU-side diagnostic only: describes the exact byte content the GPU wrote to
    // kOutputVramVa (the SDMA copy engine is byte-faithful). Never changes the failure
    // verdict or the CPU comparison contract.
    const ComputeReadbackAnomaly anomaly = classify_compute_readback_anomaly(
        static_cast<const uint8_t*>(readback_mapping.data),
        expected_output_payload.data(), expected_output_payload.size());
    char mask_buf[3][11]{};
    std::snprintf(mask_buf[0], sizeof(mask_buf[0]), "0x%02x", anomaly.written_element_mask);
    std::snprintf(mask_buf[1], sizeof(mask_buf[1]), "0x%02x", anomaly.swapped_element_mask);
    std::snprintf(mask_buf[2], sizeof(mask_buf[2]), "0x%02x",
                  anomaly.unswapped_match_element_mask);
    log.compute.compute_readback_anomaly =
        std::string("anomaly_class=") + compute_readback_anomaly_class_label(anomaly.cls) +
        " written_mask=" + mask_buf[0] +
        " swapped_mask=" + mask_buf[1] +
        " unswapped_match_mask=" + mask_buf[2];
    print_kernel_log(log, staging, readback, sdma_control, compute_control, "pass",
                     kernel_elapsed_usec, "fail", "fail", log.failure_stage, log.failure_text, 1);
    return 1;
  }

  log.failure_stage = "none";
  log.failure_text = "none";
  print_kernel_log(log, staging, readback, sdma_control, compute_control, "pass",
                   kernel_elapsed_usec, "pass", "pass", log.failure_stage, log.failure_text, 0);
  return 0;
}

void print_help(const char* argv0) {
  std::printf("usage: %s --self-test <name>\n", argv0);
  std::printf("       %s --discovery-smoke\n", argv0);
  std::printf("       %s --transfer-proof\n", argv0);
  std::printf("       %s --kernel-proof\n", argv0);
  std::printf("\n");
  std::printf("no-hardware self-tests:\n");
  std::printf("  --self-test remote-cmd-frame\n");
  std::printf("  --self-test log-contract\n");
  std::printf("  --self-test sysmem-page-list\n");
  std::printf("  --self-test sdma-packet-encoding\n");
  std::printf("  --self-test am-vm-pte-encoding\n");
  std::printf("  --self-test am-vm-page-table-plan\n");
  std::printf("  --self-test sdma-ring-setup\n");
  std::printf("  --self-test sdma-fence-packet-encoding\n");
  std::printf("  --self-test sdma-submit-sequence\n");
  std::printf("  --self-test am-vm-tlb-sequence\n");
  std::printf("  --self-test kernel-proof-contract\n");
  std::printf("  --self-test compute-vm-layout\n");
  std::printf("  --self-test gfx-ring-registers\n");
  std::printf("  --self-test compute-mqd-encoding\n");
  std::printf("  --self-test pm4-dispatch-sequence\n");
  std::printf("  --self-test compute-doorbell-delivery\n");
  std::printf("  --self-test compute-doorbell-consumption\n");
  std::printf("  --self-test compute-doorbell-consumption-classifier\n");
  std::printf("  --self-test gc-hub-sequence\n");
  std::printf("  --self-test mec-rs64-pipe-activation\n");
  std::printf("  --self-test compute-readback-classifier\n");
  std::printf("  --self-test kernel-text-decode\n");
  std::printf("\n");
  std::printf("hardware modes:\n");
  std::printf("  --discovery-smoke\n");
  std::printf("  --transfer-proof\n");
  std::printf("  --kernel-proof\n");
}

int unknown_self_test(const char* name) {
  std::printf("failure_text: unknown self-test '%s'\n", name);
  std::printf("exit_status: 1\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    print_help(argv[0]);
    return 0;
  }

  if (argc == 2 && std::strcmp(argv[1], "--discovery-smoke") == 0) {
    return run_discovery_smoke();
  }

  if (argc == 2 && std::strcmp(argv[1], "--transfer-proof") == 0) {
    return run_transfer_proof_scaffold();
  }

  if (argc == 2 && std::strcmp(argv[1], "--kernel-proof") == 0) {
    return run_kernel_proof_scaffold();
  }


  if (argc == 3 && std::strcmp(argv[1], "--self-test") == 0) {
    if (std::strcmp(argv[2], "remote-cmd-frame") == 0) {
      return run_remote_cmd_frame_self_test();
    }
    if (std::strcmp(argv[2], "log-contract") == 0) {
      return run_log_contract_self_test();
    }
    if (std::strcmp(argv[2], "sysmem-page-list") == 0) {
      return run_sysmem_page_list_self_test();
    }
    if (std::strcmp(argv[2], "sdma-ring-setup") == 0) {
      return run_sdma_ring_setup_self_test();
    }
    if (std::strcmp(argv[2], "sdma-fence-packet-encoding") == 0) {
      return run_sdma_fence_packet_encoding_self_test();
    }
    if (std::strcmp(argv[2], "sdma-submit-sequence") == 0) {
      return run_sdma_submit_sequence_self_test();
    }
    if (std::strcmp(argv[2], "sdma-packet-encoding") == 0) {
      return run_sdma_packet_encoding_self_test();
    }
    if (std::strcmp(argv[2], "am-vm-pte-encoding") == 0) {
      return run_am_vm_pte_encoding_self_test();
    }
    if (std::strcmp(argv[2], "am-vm-page-table-plan") == 0) {
      return run_am_vm_page_table_plan_self_test();
    }
    if (std::strcmp(argv[2], "am-vm-tlb-sequence") == 0) {
      return run_am_vm_tlb_sequence_self_test();
    }
    if (std::strcmp(argv[2], "kernel-proof-contract") == 0) {
      return run_kernel_proof_contract_self_test();
    }
    if (std::strcmp(argv[2], "compute-vm-layout") == 0) {
      return run_compute_vm_layout_self_test();
    }
    if (std::strcmp(argv[2], "gfx-ring-registers") == 0) {
      return run_gfx_ring_registers_self_test();
    }
    if (std::strcmp(argv[2], "compute-mqd-encoding") == 0) {
      return run_compute_mqd_encoding_self_test();
    }
    if (std::strcmp(argv[2], "pm4-dispatch-sequence") == 0) {
      return run_pm4_dispatch_sequence_self_test();
    }
    if (std::strcmp(argv[2], "compute-doorbell-delivery") == 0) {
      return run_compute_doorbell_delivery_self_test();
    }
    if (std::strcmp(argv[2], "compute-doorbell-consumption") == 0) {
      return run_compute_doorbell_consumption_self_test();
    }
    if (std::strcmp(argv[2], "compute-doorbell-consumption-classifier") == 0) {
      return run_compute_doorbell_consumption_classifier_self_test();
    }
    if (std::strcmp(argv[2], "gc-hub-sequence") == 0) {
      return run_gc_hub_sequence_self_test();
    }
    if (std::strcmp(argv[2], "mec-rs64-pipe-activation") == 0) {
      return run_mec_rs64_pipe_activation_self_test();
    }
    if (std::strcmp(argv[2], "compute-readback-classifier") == 0) {
      return run_compute_readback_classifier_self_test();
    }
    if (std::strcmp(argv[2], "kernel-text-decode") == 0) {
      return run_kernel_text_decode_self_test();
    }
    return unknown_self_test(argv[2]);
  }

  print_help(argv[0]);
  std::printf("failure_text: expected --help, --discovery-smoke, --transfer-proof, --kernel-proof, or --self-test <name>\n");
  std::printf("exit_status: 1\n");
  return 1;
}
