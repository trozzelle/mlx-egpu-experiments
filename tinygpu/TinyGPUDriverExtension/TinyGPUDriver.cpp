#include "TinyGPUDriver.h"
#include "TinyGPUInferenceUserClient.h"
#include "TinyGPURecoveryUserClient.h"
#include "TinyGPUDiagnosticUserClient.h"
#include "TGPUABI.h"
#include "TGPUColdLifecycle.h"
#include "TGPUFramebufferDecoder.h"
#include <PCIDriverKit/PCIDriverKit.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t kTargetVendor = 0x1002;
constexpr uint32_t kTargetDevice = 0x7551;
constexpr const char *kServiceIdentity = "org.tinygrad.tinygpu.driver2";
constexpr uint32_t kDiscoveryBinarySignature = 0x28211407;
constexpr uint32_t kDiscoveryTableSignature = 0x53445049;
constexpr uint32_t kDiscoveryVramBackoff = 64U << 10;
constexpr uint32_t kDiscoveryTableBytes = 10U << 10;
constexpr uint32_t kDiscoveryHeaderBytes = 60;
constexpr uint32_t kRccConfigMemsizeDword = 0x0de3;
constexpr uint32_t kBarMmIndexDword = 0;
constexpr uint32_t kBarMmDataDword = 1;
constexpr uint32_t kBarMmIndexHighDword = 6;
constexpr uint32_t kR9700GcHwId = 11;
constexpr uint32_t kR9700MmhubsHwId = 34;
constexpr uint32_t kR9700Sdma0HwId = 42;
constexpr uint32_t kR9700Mp1HwId = 1;
constexpr uint32_t kR9700Mp0HwId = 255;
constexpr uint32_t kR9700NbifHwId = 108;
constexpr uint32_t kMaxDiscoveryBases = 8;
constexpr uint64_t kMaxHostVisibleBufferBytes = 1ULL << 30;
constexpr uint64_t kMinHostVisibleBufferAlignment = 4096;
constexpr uint32_t kHostVisibleMemoryDomainBits = TGPU_MEMORY_HOST_VISIBLE;

enum TinyGPUIpSlot : uint32_t {
	kIpGc = 0,
	kIpMmhubs,
	kIpSdma0,
	kIpMp1,
	kIpMp0,
	kIpNbif,
	kIpCount,
};

struct TinyGPUIpBlock {
	bool found = false;
	uint16_t hw_id = 0;
	uint8_t instance = 0;
	uint8_t major = 0;
	uint8_t minor = 0;
	uint8_t revision = 0;
	uint8_t base_count = 0;
	uint64_t bases[kMaxDiscoveryBases] = {};
};

}  // namespace

struct TinyGPUDriver_IVars
{
	IOPCIDevice *pci = nullptr;
	uint8_t bar_index[6] = {};
	uint64_t bar_size[6] = {};
	TinyGPUIpBlock ip[kIpCount] = {};
	uint32_t gc_instance_count = 0;
	uint64_t vram_size = 0;
	uint16_t vendor_id = 0;
	uint16_t device_id = 0;
	uint64_t device_epoch = 1;
	uint64_t next_connection_epoch = 1;
	uint32_t health_state = TGPU_HEALTH_INITIALIZING;
	uint32_t failure_stage = TGPU_FAILURE_NONE;
	uint32_t failure_status = TGPU_STATUS_OK;
	uint32_t cold_stage = TGPU_FAILURE_NONE;
	uint32_t last_stage_status = TGPU_STATUS_OK;
	bool discovery_ready = false;
	bool cold_ready = false;
	char failure_text[TGPU_MAX_FAULT_TEXT_BYTES] = {};
};
namespace {

static uint16_t LoadU16(const uint8_t *bytes) {
	return static_cast<uint16_t>(bytes[0]) |
	       static_cast<uint16_t>(bytes[1] << 8);
}

static uint32_t LoadU32(const uint8_t *bytes) {
	return static_cast<uint32_t>(bytes[0]) |
	       (static_cast<uint32_t>(bytes[1]) << 8) |
	       (static_cast<uint32_t>(bytes[2]) << 16) |
	       (static_cast<uint32_t>(bytes[3]) << 24);
}

static uint64_t LoadU64(const uint8_t *bytes) {
	return static_cast<uint64_t>(LoadU32(bytes)) |
	       (static_cast<uint64_t>(LoadU32(bytes + 4)) << 32);
}

static void SetFailureText(TinyGPUDriver_IVars *ivars, const char *text) {
	if (!ivars) return;
	std::memset(ivars->failure_text, 0, sizeof(ivars->failure_text));
	if (!text) return;
	std::strncpy(ivars->failure_text, text, sizeof(ivars->failure_text) - 1);
}

static kern_return_t BarRead32(TinyGPUDriver_IVars *ivars, uint32_t bar,
	                            uint64_t offset, uint32_t *value) {
	if (!ivars || !ivars->pci || !value || bar >= 6 ||
	    offset > UINT64_MAX - sizeof(uint32_t) ||
	    offset + sizeof(uint32_t) > ivars->bar_size[bar]) {
		return kIOReturnBadArgument;
	}
	ivars->pci->MemoryRead32(ivars->bar_index[bar], offset, value);
	return kIOReturnSuccess;
}

static kern_return_t BarWrite32(TinyGPUDriver_IVars *ivars, uint32_t bar,
	                             uint64_t offset, uint32_t value) {
	if (!ivars || !ivars->pci || bar >= 6 ||
	    offset > UINT64_MAX - sizeof(uint32_t) ||
	    offset + sizeof(uint32_t) > ivars->bar_size[bar]) {
		return kIOReturnBadArgument;
	}
	ivars->pci->MemoryWrite32(ivars->bar_index[bar], offset, value);
	return kIOReturnSuccess;
}

static kern_return_t DiscoverBars(TinyGPUDriver_IVars *ivars) {
	if (!ivars || !ivars->pci) return kIOReturnNotAttached;
	for (uint32_t bar = 0; bar < 6; ++bar) {
		uint8_t memory_index = 0;
		uint8_t memory_type = 0;
		uint64_t memory_size = 0;
		const kern_return_t err =
		    ivars->pci->GetBARInfo(bar, &memory_index, &memory_size, &memory_type);
		if (err == kIOReturnSuccess && memory_size != 0) {
			ivars->bar_index[bar] = memory_index;
			ivars->bar_size[bar] = memory_size;
		}
	}
	if (ivars->bar_size[5] < (kRccConfigMemsizeDword + 1ULL) * sizeof(uint32_t)) {
		return kIOReturnNotReady;
	}
	return kIOReturnSuccess;
}

static kern_return_t ReadVram32(TinyGPUDriver_IVars *ivars, uint64_t address,
	                             uint32_t *value) {
	if (!ivars || !value || ivars->vram_size == 0 ||
	    address > ivars->vram_size - sizeof(uint32_t)) {
		return kIOReturnBadArgument;
	}
	// AMDev uses the directly mapped BAR0 window when it spans all VRAM.
	if (ivars->bar_size[0] >= ivars->vram_size) {
		return BarRead32(ivars, 0, address, value);
	}
	// Otherwise use the documented MM_INDEX/MM_DATA aperture.  The high
	// register is written first, matching tinygrad amdev.py:279-286.
	kern_return_t err = BarWrite32(ivars, 5, kBarMmIndexHighDword * 4ULL,
	                               static_cast<uint32_t>(address >> 31));
	if (err != kIOReturnSuccess) return err;
	err = BarWrite32(ivars, 5, kBarMmIndexDword * 4ULL,
	                 static_cast<uint32_t>((address & 0x7fffffffULL) |
	                                       0x80000000ULL));
	if (err != kIOReturnSuccess) return err;
	return BarRead32(ivars, 5, kBarMmDataDword * 4ULL, value);
}

static kern_return_t ReadDiscovery(TinyGPUDriver_IVars *ivars, uint8_t *table,
	                               size_t table_bytes) {
	if (!ivars || !table || table_bytes < kDiscoveryTableBytes) {
		return kIOReturnBadArgument;
	}
	uint32_t vram_megabytes = 0;
	kern_return_t err =
	    BarRead32(ivars, 5, kRccConfigMemsizeDword * 4ULL, &vram_megabytes);
	if (err != kIOReturnSuccess || vram_megabytes == 0 ||
	    vram_megabytes > (1U << 20)) {
		SetFailureText(ivars, "RCC_CONFIG_MEMSIZE is unavailable or invalid");
		return err != kIOReturnSuccess ? err : kIOReturnNotReady;
	}
	ivars->vram_size = static_cast<uint64_t>(vram_megabytes) << 20;
	if (ivars->vram_size < kDiscoveryVramBackoff + kDiscoveryTableBytes) {
		SetFailureText(ivars, "VRAM is too small for the IP discovery table");
		return kIOReturnNotReady;
	}

	const uint64_t table_offset = ivars->vram_size - kDiscoveryVramBackoff;
	for (uint32_t offset = 0; offset < kDiscoveryTableBytes; offset += 4) {
		uint32_t word = 0;
		err = ReadVram32(ivars, table_offset + offset, &word);
		if (err != kIOReturnSuccess) {
			SetFailureText(ivars, "IP discovery table read failed");
			return err;
		}
		std::memcpy(table + offset, &word, sizeof(word));
	}
	return kIOReturnSuccess;
}

static TinyGPUIpSlot SlotForHardwareId(uint16_t hardware_id) {
	switch (hardware_id) {
	case kR9700GcHwId: return kIpGc;
	case kR9700MmhubsHwId: return kIpMmhubs;
	case kR9700Sdma0HwId: return kIpSdma0;
	case kR9700Mp1HwId: return kIpMp1;
	case kR9700Mp0HwId: return kIpMp0;
	case kR9700NbifHwId: return kIpNbif;
	default: return kIpCount;
	}
}

static kern_return_t ParseDiscovery(TinyGPUDriver_IVars *ivars,
	                                const uint8_t *table, size_t table_bytes) {
	if (!ivars || !table || table_bytes < kDiscoveryHeaderBytes ||
	    LoadU32(table) != kDiscoveryBinarySignature) {
		SetFailureText(ivars, "IP discovery binary signature mismatch");
		return kIOReturnNotReady;
	}
	std::memset(ivars->ip, 0, sizeof(ivars->ip));
	ivars->gc_instance_count = 0;
	const size_t table_info_offset = 12;
	if (table_bytes < table_info_offset + 8) {
		SetFailureText(ivars, "IP discovery table-info record is missing");
		return kIOReturnNotReady;
	}
	const size_t ip_table_offset = LoadU16(table + table_info_offset);
	if (ip_table_offset > table_bytes || table_bytes - ip_table_offset < 80) {
		SetFailureText(ivars, "IP discovery header is outside its bounded table");
		return kIOReturnNotReady;
	}
	const uint8_t *ip_header = table + ip_table_offset;
	if (LoadU32(ip_header) != kDiscoveryTableSignature) {
		SetFailureText(ivars, "IP discovery table signature mismatch");
		return kIOReturnNotReady;
	}
	const uint16_t die_count = LoadU16(ip_header + 12);
	const bool base_address_64 = (ip_header[78] & 1U) != 0;
	if (die_count == 0 || die_count > 16) {
		SetFailureText(ivars, "IP discovery die count is invalid");
		return kIOReturnNotReady;
	}

	for (uint16_t die = 0; die < die_count; ++die) {
		const size_t die_info_offset = ip_table_offset + 14 + die * 4ULL;
		if (die_info_offset > table_bytes || table_bytes - die_info_offset < 4) {
			SetFailureText(ivars, "IP discovery die record is truncated");
			return kIOReturnNotReady;
		}
		const size_t die_offset = LoadU16(table + die_info_offset + 2);
		if (die_offset > table_bytes || table_bytes - die_offset < 4) {
			SetFailureText(ivars, "IP discovery die header is truncated");
			return kIOReturnNotReady;
		}
		const uint16_t ip_count = LoadU16(table + die_offset + 2);
		size_t ip_offset = die_offset + 4;
		for (uint16_t ip_index = 0; ip_index < ip_count; ++ip_index) {
			if (ip_offset > table_bytes || table_bytes - ip_offset < 8) {
				SetFailureText(ivars, "IP discovery IP record is truncated");
				return kIOReturnNotReady;
			}
			const uint16_t hardware_id = LoadU16(table + ip_offset);
			const uint8_t base_count = table[ip_offset + 3];
			const size_t base_width = base_address_64 ? 8 : 4;
			if (base_count > kMaxDiscoveryBases ||
			    base_count > (table_bytes - (ip_offset + 8)) / base_width) {
				SetFailureText(ivars, "IP discovery base-address list is invalid");
				return kIOReturnNotReady;
			}
			const TinyGPUIpSlot slot = SlotForHardwareId(hardware_id);
			if (slot != kIpCount) {
				if (slot == kIpGc) ++ivars->gc_instance_count;
				if (!ivars->ip[slot].found) {
					TinyGPUIpBlock &block = ivars->ip[slot];
					block.found = true;
					block.hw_id = hardware_id;
					block.instance = table[ip_offset + 2];
					block.major = table[ip_offset + 4];
					block.minor = table[ip_offset + 5];
					block.revision = table[ip_offset + 6];
					block.base_count = base_count;
					for (uint8_t base = 0; base < base_count; ++base) {
						const uint8_t *base_bytes =
						    table + ip_offset + 8 + base * base_width;
						block.bases[base] =
						    base_address_64 ? LoadU64(base_bytes) : LoadU32(base_bytes);
					}
				}
			}
			ip_offset += 8 + base_count * base_width;
		}
	}

	const TinyGPUIpBlock &gc = ivars->ip[kIpGc];
	if (!gc.found || gc.major != 12 || gc.minor != 0 ||
	    gc.revision != 1 || gc.instance != 0 || ivars->gc_instance_count != 1) {
		SetFailureText(ivars, "IP discovery did not identify exactly one gfx1201 GC");
		return kIOReturnUnsupported;
	}
	for (uint32_t slot = 0; slot < kIpCount; ++slot) {
		if (!ivars->ip[slot].found) {
			SetFailureText(ivars, "IP discovery omitted a required R9700 IP block");
			return kIOReturnNotReady;
		}
	}
	ivars->discovery_ready = true;
	return kIOReturnSuccess;
}

static kern_return_t DiscoverIPBlocks(TinyGPUDriver_IVars *ivars) {
	uint8_t table[kDiscoveryTableBytes] = {};
	kern_return_t err = DiscoverBars(ivars);
	if (err != kIOReturnSuccess) {
		SetFailureText(ivars, "required BAR5 MMIO window is unavailable");
		return err;
	}
	err = ReadDiscovery(ivars, table, sizeof(table));
	if (err != kIOReturnSuccess) return err;
	return ParseDiscovery(ivars, table, sizeof(table));
}

static kern_return_t ResolveRegister(TinyGPUDriver_IVars *ivars,
	                                 TinyGPUIpSlot slot, uint32_t segment,
	                                 uint32_t offset, uint64_t *dword) {
	if (!ivars || !dword || slot >= kIpCount || !ivars->ip[slot].found ||
	    segment >= ivars->ip[slot].base_count) {
		return kIOReturnNotReady;
	}
	const uint64_t base = ivars->ip[slot].bases[segment];
	if (base > UINT64_MAX - offset) return kIOReturnBadArgument;
	*dword = base + offset;
	return *dword > UINT32_MAX ? kIOReturnBadArgument : kIOReturnSuccess;
}

static kern_return_t ReadIPRegister(TinyGPUDriver_IVars *ivars, TinyGPUIpSlot slot,
	                                uint32_t segment, uint32_t offset,
	                                uint32_t *value) {
	uint64_t dword = 0;
	kern_return_t err = ResolveRegister(ivars, slot, segment, offset, &dword);
	if (err != kIOReturnSuccess) return err;
	if (dword < ivars->bar_size[5] / sizeof(uint32_t)) {
		return BarRead32(ivars, 5, dword * sizeof(uint32_t), value);
	}
	// Large register addresses use the NBIF RSMU index/data pair, exactly as
	// tinygrad amdev.py:264-270 and the native gfx1201 probe.
	const TinyGPUIpBlock &nbif = ivars->ip[kIpNbif];
	if (!nbif.found || nbif.base_count <= 1) return kIOReturnNotReady;
	const uint64_t index_dword = nbif.bases[1];
	const uint64_t data_dword = nbif.bases[1] + 1;
	if (index_dword >= ivars->bar_size[5] / sizeof(uint32_t) ||
	    data_dword >= ivars->bar_size[5] / sizeof(uint32_t)) {
		return kIOReturnNotReady;
	}
	err = BarWrite32(ivars, 5, index_dword * sizeof(uint32_t),
	                 static_cast<uint32_t>(dword * sizeof(uint32_t)));
	if (err != kIOReturnSuccess) return err;
	return BarRead32(ivars, 5, data_dword * sizeof(uint32_t), value);
}

static kern_return_t WriteIPRegister(TinyGPUDriver_IVars *ivars, TinyGPUIpSlot slot,
	                                 uint32_t segment, uint32_t offset,
	                                 uint32_t value) {
	uint64_t dword = 0;
	kern_return_t err = ResolveRegister(ivars, slot, segment, offset, &dword);
	if (err != kIOReturnSuccess) return err;
	if (dword < ivars->bar_size[5] / sizeof(uint32_t)) {
		return BarWrite32(ivars, 5, dword * sizeof(uint32_t), value);
	}
	const TinyGPUIpBlock &nbif = ivars->ip[kIpNbif];
	if (!nbif.found || nbif.base_count <= 1) return kIOReturnNotReady;
	const uint64_t index_dword = nbif.bases[1];
	const uint64_t data_dword = nbif.bases[1] + 1;
	if (index_dword >= ivars->bar_size[5] / sizeof(uint32_t) ||
	    data_dword >= ivars->bar_size[5] / sizeof(uint32_t)) {
		return kIOReturnNotReady;
	}
	err = BarWrite32(ivars, 5, index_dword * sizeof(uint32_t),
	                 static_cast<uint32_t>(dword * sizeof(uint32_t)));
	if (err != kIOReturnSuccess) return err;
	return BarWrite32(ivars, 5, data_dword * sizeof(uint32_t), value);
}

static kern_return_t ReadbackIPRegister(TinyGPUDriver_IVars *ivars,
	                                    TinyGPUIpSlot slot, uint32_t segment,
	                                    uint32_t offset, uint32_t expected,
	                                    uint32_t mask) {
	uint32_t observed = 0;
	kern_return_t err = ReadIPRegister(ivars, slot, segment, offset, &observed);
	if (err != kIOReturnSuccess) return err;
	return (observed & mask) == (expected & mask) ? kIOReturnSuccess
	                                               : kIOReturnNotReady;
}

static kern_return_t RunPspSosTmr(TinyGPUDriver_IVars *ivars) {
	// No approved provenance-bound PSP/SOS/TMR firmware or cold transition
	// input is present in this checkout.  Warm register observations are not
	// evidence of ownership and must never establish driver readiness.
	SetFailureText(
	    ivars,
	    "cold_stage=PspSosTmr: approved cold-ownership firmware transition path unavailable");
	return kIOReturnNotReady;
}

static kern_return_t RunSmu(TinyGPUDriver_IVars *ivars) {
	// SMU v14 TestMessage mailbox sequence from tinygrad
	// runtime/support/am/ip.py:235-244.  C2PMSG_90 is the response register;
	// the generated MP1 register map resolves it to offset 666.  A response is
	// polled only for a bounded interval; absence is a stage failure, never a
	// successful guess.
	if (WriteIPRegister(ivars, kIpMp1, 1, 666, 0) != kIOReturnSuccess ||
	    WriteIPRegister(ivars, kIpMp1, 1, 658, 0) != kIOReturnSuccess ||
	    WriteIPRegister(ivars, kIpMp1, 1, 642, 1) != kIOReturnSuccess) {
		SetFailureText(ivars, "cold_stage=Smu: SMU TestMessage mailbox write failed");
		return kIOReturnNotReady;
	}
	for (uint32_t attempt = 0; attempt < 256; ++attempt) {
		IODelay(1000U);
		uint32_t response = 0;
		if (ReadIPRegister(ivars, kIpMp1, 1, 666, &response) != kIOReturnSuccess) {
			SetFailureText(ivars, "cold_stage=Smu: SMU TestMessage response read failed");
			return kIOReturnNotReady;
		}
		if (response == 1) return kIOReturnSuccess;
	}
	SetFailureText(ivars, "cold_stage=Smu: SMU TestMessage response timed out");
	return kIOReturnTimeout;
}
static kern_return_t RunImu(TinyGPUDriver_IVars *ivars) {
	uint32_t boot_status = 0;
	if (ReadIPRegister(ivars, kIpGc, 0, 20092, &boot_status) != kIOReturnSuccess ||
	    (boot_status & (1U << 2 | 1U << 3)) != (1U << 2 | 1U << 3)) {
		SetFailureText(ivars,
		               "cold_stage=Imu: GC IMU security-policy firmware is not ready");
		return kIOReturnNotReady;
	}
	return kIOReturnSuccess;
}
static kern_return_t RunRlc(TinyGPUDriver_IVars *ivars) {
	uint32_t boot_status = 0;
	if (ReadIPRegister(ivars, kIpGc, 0, 20092, &boot_status) != kIOReturnSuccess ||
	    (boot_status & (1U << 31 | 1U << 4 | 1U << 5)) !=
	        (1U << 31 | 1U << 4 | 1U << 5)) {
		SetFailureText(ivars,
		               "cold_stage=Rlc: RLC bootload status did not reach complete");
		return kIOReturnNotReady;
	}
	return kIOReturnSuccess;
}
static kern_return_t RunCpMesGfxSdma(TinyGPUDriver_IVars *ivars) {
	uint32_t cp_status = 0;
	if (ReadIPRegister(ivars, kIpGc, 0, 3904, &cp_status) != kIOReturnSuccess ||
	    (cp_status & 0x01fffe00U) != 0) {
		SetFailureText(ivars,
		               "cold_stage=CpMesGfxSdma: CP_STAT reports a busy or incomplete engine");
		return kIOReturnNotReady;
	}
	uint32_t sdma_control = 0;
	if (ReadIPRegister(ivars, kIpSdma0, 0, 13, &sdma_control) != kIOReturnSuccess ||
	    (sdma_control & ((1U << 29) | (1U << 30))) != 0) {
		SetFailureText(ivars,
		               "cold_stage=CpMesGfxSdma: SDMA0 control reports a stopped engine");
		return kIOReturnNotReady;
	}
	uint32_t sdma_mcu = 0;
	if (ReadIPRegister(ivars, kIpSdma0, 1, 22670, &sdma_mcu) != kIOReturnSuccess ||
	    (sdma_mcu & 0x3U) != 0) {
		SetFailureText(ivars,
		               "cold_stage=CpMesGfxSdma: SDMA0 MCU is halted or held in reset");
		return kIOReturnNotReady;
	}
	return kIOReturnSuccess;
}
static kern_return_t RunGmcGartVm(TinyGPUDriver_IVars *ivars) {
	uint32_t fb_base = 0;
	uint32_t fb_top = 0;
	if (ReadIPRegister(ivars, kIpMmhubs, 0, 1364, &fb_base) != kIOReturnSuccess ||
	    ReadIPRegister(ivars, kIpMmhubs, 0, 1365, &fb_top) != kIOReturnSuccess) {
		SetFailureText(ivars,
		               "cold_stage=GmcGartVm: MMHUB framebuffer location read failed");
		return kIOReturnNotReady;
	}

	TGPUFramebufferDecodeResult framebuffer{};
	if (TGPUDecodeFramebufferLocation(fb_base, fb_top, &framebuffer) !=
	    TGPU_STATUS_OK) {
		SetFailureText(ivars,
		               "cold_stage=GmcGartVm: MMHUB framebuffer aperture is invalid");
		return kIOReturnNotReady;
	}
	if (WriteIPRegister(ivars, kIpMmhubs, 0, 1369,
	                    framebuffer.base_aperture_register) != kIOReturnSuccess ||
	    WriteIPRegister(ivars, kIpMmhubs, 0, 1370,
	                    framebuffer.top_aperture_register) != kIOReturnSuccess ||
	    ReadbackIPRegister(ivars, kIpMmhubs, 0, 1369,
	                       framebuffer.base_aperture_register,
	                       0xffffffffU) != kIOReturnSuccess ||
	    ReadbackIPRegister(ivars, kIpMmhubs, 0, 1370,
	                       framebuffer.top_aperture_register,
	                       0xffffffffU) != kIOReturnSuccess) {
		SetFailureText(ivars,
		               "cold_stage=GmcGartVm: MMHUB system aperture programming failed");
		return kIOReturnNotReady;
	}
	uint32_t l2_control = 0;
	if (ReadIPRegister(ivars, kIpMmhubs, 0, 1252, &l2_control) !=
	        kIOReturnSuccess ||
	    (l2_control & (1U | (1U << 11))) != (1U | (1U << 11))) {
		SetFailureText(ivars,
		               "cold_stage=GmcGartVm: MMHUB VM L2 is not enabled");
		return kIOReturnNotReady;
	}
	uint32_t context_control = 0;
	if (ReadIPRegister(ivars, kIpMmhubs, 0, 1380, &context_control) !=
	        kIOReturnSuccess ||
	    (context_control & 1U) == 0) {
		SetFailureText(ivars,
		               "cold_stage=GmcGartVm: MMHUB VMID0 context is not enabled");
		return kIOReturnNotReady;
	}
	return kIOReturnSuccess;
}


static TGPUFailureStage FailureClassForColdStage(TGPUColdStage stage) {
	switch (stage) {
	case TGPUColdStage::PspSosTmr:
	case TGPUColdStage::Imu:
	case TGPUColdStage::Rlc:
		return TGPU_FAILURE_FIRMWARE;
	case TGPUColdStage::Smu:
		return TGPU_FAILURE_POWER;
	case TGPUColdStage::CpMesGfxSdma:
		return TGPU_FAILURE_QUEUE;
	case TGPUColdStage::GmcGartVm:
		return TGPU_FAILURE_MEMORY;
	case TGPUColdStage::None:
		return TGPU_FAILURE_NONE;
	}
	return TGPU_FAILURE_DIAGNOSTIC;
}

class DriverColdStageExecutor final : public TGPUColdStageExecutor {
 public:
	explicit DriverColdStageExecutor(TinyGPUDriver &driver) : driver_(driver) {}
	bool execute(TGPUColdStage stage) override {
		return driver_.ExecuteColdStage(static_cast<uint32_t>(stage)) ==
		       kIOReturnSuccess;
	}

 private:
	TinyGPUDriver &driver_;
};

}  // namespace

bool TinyGPUDriver::init()
{
	os_log(OS_LOG_DEFAULT, "tinygpu: init");

	auto answer = super::init();
	if (!answer) {
		return false;
	}

	ivars = new TinyGPUDriver_IVars();
	if (ivars == nullptr) {
		return false;
	}

	return true;
}

void TinyGPUDriver::free()
{
	IOSafeDeleteNULL(ivars, TinyGPUDriver_IVars, 1);
	super::free();
}

kern_return_t TinyGPUDriver::Start_Impl(IOService* in_provider)
{
	IOServiceName service_name;
	os_log(OS_LOG_DEFAULT, "tinygpu: on gpu detected");

	kern_return_t err = Start(in_provider, SUPERDISPATCH);
	if (err) return err;

	ivars->pci = OSDynamicCast(IOPCIDevice, in_provider);
	if (!ivars->pci) {
		ivars->health_state = TGPU_HEALTH_UNAVAILABLE;
		ivars->failure_stage = TGPU_FAILURE_ATTACH;
		ivars->failure_status = TGPU_STATUS_DEVICE_LOST;
		SetFailureText(ivars, "provider is not an IOPCIDevice");
		return kIOReturnNoDevice;
	}

	err = ivars->pci->Open(this, 0);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: Open() failed 0x%08x", err);
		ivars->pci = nullptr;
		ivars->health_state = TGPU_HEALTH_UNAVAILABLE;
		ivars->failure_stage = TGPU_FAILURE_ATTACH;
		ivars->failure_status = TGPU_STATUS_DEVICE_LOST;
		SetFailureText(ivars, "PCI device could not be opened");
		return err;
	}

	uint16_t ven = 0, dev = 0;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetVendorID, &ven);
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetDeviceID, &dev);
	ivars->vendor_id = ven;
	ivars->device_id = dev;
	os_log(OS_LOG_DEFAULT, "tinygpu: opened device ven=0x%04x dev=0x%04x", ven, dev);
	if (ven != kTargetVendor || dev != kTargetDevice) {
		SetFailureText(ivars, "PCI identity is outside the R9700 product scope");
		ivars->health_state = TGPU_HEALTH_UNAVAILABLE;
		ivars->failure_stage = TGPU_FAILURE_ATTACH;
		ivars->failure_status = TGPU_STATUS_PERMISSION_DENIED;
		ivars->pci->Close(this, 0);
		ivars->pci = nullptr;
		return kIOReturnNoDevice;
	}

	uint16_t commandRegister = 0;
	ivars->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &commandRegister);
	commandRegister |= (kIOPCICommandIOSpace | kIOPCICommandBusMaster |
	                    kIOPCICommandMemorySpace);
	ivars->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand,
	                                 commandRegister);

	// Discovery is a real BAR0/BAR5 read and bounded parser, not a CPU-side
	// architecture hint.  A failed discovery is exposed through health so the
	// direct conformance client can report the attach failure without claiming
	// readiness.
	err = DiscoverIPBlocks(ivars);
	if (err != kIOReturnSuccess) {
		ivars->health_state = TGPU_HEALTH_FAULTED;
		ivars->failure_stage = TGPU_FAILURE_ATTACH;
		ivars->failure_status = TGPU_STATUS_DEVICE_LOST;
		if (ivars->failure_text[0] == '\0') {
			SetFailureText(ivars, "R9700 IP discovery failed");
		}
	}

	if (err == kIOReturnSuccess) {
		DriverColdStageExecutor executor(*this);
		TGPUColdLifecycle lifecycle(executor);
		const TGPUColdLifecycleResult result = lifecycle.initialize();
		if (result.ready) {
			ivars->cold_ready = true;
			ivars->health_state = TGPU_HEALTH_READY;
			ivars->failure_stage = TGPU_FAILURE_NONE;
			ivars->failure_status = TGPU_STATUS_OK;
			ivars->cold_stage = TGPU_FAILURE_NONE;
			ivars->last_stage_status = TGPU_STATUS_OK;
			SetFailureText(ivars, nullptr);
		} else {
			ivars->cold_ready = false;
			ivars->health_state = TGPU_HEALTH_FAULTED;
			ivars->failure_stage = FailureClassForColdStage(result.failure_stage);
			ivars->failure_status = TGPU_STATUS_DEVICE_LOST;
			ivars->cold_stage = static_cast<uint32_t>(result.failure_stage);
			ivars->last_stage_status = TGPU_STATUS_DEVICE_LOST;
			os_log(OS_LOG_DEFAULT,
			       "tinygpu: cold lifecycle stopped at stage %u",
			       static_cast<unsigned>(result.failure_stage));
		}
	}
	std::memset(service_name, 0, sizeof(service_name));
	std::memcpy(service_name, kServiceIdentity, sizeof(kServiceIdentity) - 1);
	SetName(service_name);
	RegisterService();

	os_log(OS_LOG_DEFAULT, "tinygpu: service started health=%u failure_stage=%u",
	       ivars->health_state, ivars->failure_stage);
	// A faulted service remains queryable so the structured health record can
	// attribute the exact stage.  It never advertises ready state.
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDriver::Stop_Impl(IOService* in_provider)
{
	if (ivars) {
		ivars->health_state = TGPU_HEALTH_DISCONNECTED;
		ivars->cold_ready = false;
		if (ivars->pci) {
			ivars->pci->Close(this, 0);
			ivars->pci = nullptr;
		}
	}
	return Stop(in_provider, SUPERDISPATCH);
}



kern_return_t TinyGPUDriver::NewUserClient_Impl(
    uint32_t in_type, IOUserClient **out_user_client)
{
	if (!out_user_client) return kIOReturnBadArgument;
	*out_user_client = nullptr;

	const char *properties = nullptr;
	switch (in_type) {
	case 0:
		properties = "TinyGPUInferenceUserClientProperties";
		break;
	case 1:
		properties = "TinyGPURecoveryUserClientProperties";
		break;
	case 2:
		properties = "TinyGPUDiagnosticUserClientProperties";
		break;
	default:
		os_log(OS_LOG_DEFAULT, "tinygpu: unsupported user-client type %u",
		       in_type);
		return kIOReturnUnsupported;
	}

	IOService *user_client_service = nullptr;
	kern_return_t err = Create(this, properties, &user_client_service);
	if (err != kIOReturnSuccess) {
		os_log(OS_LOG_DEFAULT, "tinygpu: failed to create user-client type %u",
		       in_type);
		return err;
	}

	IOUserClient *typed_client = nullptr;
	switch (in_type) {
	case 0:
		typed_client =
		    OSDynamicCast(TinyGPUInferenceUserClient, user_client_service);
		break;
	case 1:
		typed_client =
		    OSDynamicCast(TinyGPURecoveryUserClient, user_client_service);
		break;
	case 2:
		typed_client =
		    OSDynamicCast(TinyGPUDiagnosticUserClient, user_client_service);
		break;
	default:
		break;
	}
	if (!typed_client) {
		if (user_client_service) user_client_service->release();
		os_log(OS_LOG_DEFAULT,
		       "tinygpu: user-client property/class mismatch for type %u",
		       in_type);
		return kIOReturnUnsupported;
	}
	// Create returns a retained service. Success transfers that retain to the
	// NewUserClient output as required by the DriverKit contract.
	*out_user_client = typed_client;
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDriver::ExecuteColdStage(uint32_t stage)
{
	if (!ivars || !ivars->discovery_ready) {
		SetFailureText(ivars, "cold stage requested before IP discovery");
		return kIOReturnNotReady;
	}
	const TGPUColdStage cold_stage = static_cast<TGPUColdStage>(stage);
	kern_return_t err = kIOReturnUnsupported;
	switch (cold_stage) {
	case TGPUColdStage::PspSosTmr:
		err = RunPspSosTmr(ivars);
		break;
	case TGPUColdStage::Smu:
		err = RunSmu(ivars);
		break;
	case TGPUColdStage::Imu:
		err = RunImu(ivars);
		break;
	case TGPUColdStage::Rlc:
		err = RunRlc(ivars);
		break;
	case TGPUColdStage::CpMesGfxSdma:
		err = RunCpMesGfxSdma(ivars);
		break;
	case TGPUColdStage::GmcGartVm:
		err = RunGmcGartVm(ivars);
		break;
	case TGPUColdStage::None:
		err = kIOReturnBadArgument;
		break;
	}
	ivars->last_stage_status = err == kIOReturnSuccess
	                               ? TGPU_STATUS_OK
	                               : TGPU_STATUS_DEVICE_LOST;
	if (err != kIOReturnSuccess) {
		ivars->cold_stage = stage;
		ivars->failure_stage = FailureClassForColdStage(cold_stage);
		ivars->failure_status = TGPU_STATUS_DEVICE_LOST;
	}
	return err;
}

kern_return_t TinyGPUDriver::AllocateConnectionEpoch(uint64_t *epoch)
{
	if (!ivars || !epoch) return kIOReturnBadArgument;
	*epoch = 0;
	// Resource-table tokens carry a 32-bit epoch.  Stop before the
	// representable range is exhausted; never wrap or reuse an epoch.
	if (ivars->next_connection_epoch == 0 ||
	    ivars->next_connection_epoch > 0xffffffffULL) {
		return kIOReturnNoResources;
	}
	*epoch = ivars->next_connection_epoch;
	++ivars->next_connection_epoch;
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDriver::GetBufferBackingLimits(
    uint64_t *max_buffer_bytes, uint64_t *min_buffer_alignment,
    uint32_t *memory_domain_bits)
{
	if (!ivars || !max_buffer_bytes || !min_buffer_alignment ||
	    !memory_domain_bits) {
		return kIOReturnBadArgument;
	}
	if (!ivars->pci) return kIOReturnNotAttached;
	*max_buffer_bytes = kMaxHostVisibleBufferBytes;
	*min_buffer_alignment = kMinHostVisibleBufferAlignment;
	*memory_domain_bits = kHostVisibleMemoryDomainBits;
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDriver::QueryCapabilities(TGPUCapabilitiesResponse *response)
{
	if (!ivars || !response) return kIOReturnBadArgument;
	std::memset(response, 0, sizeof(*response));
	response->feature_bits = TGPU_FEATURE_FAULT_QUERY |
	                         TGPU_FEATURE_BUFFER_ALLOCATE;
	response->memory_domain_bits = kHostVisibleMemoryDomainBits;
	response->vendor_id = ivars->vendor_id;
	response->device_id = ivars->device_id;
	if (ivars->discovery_ready) {
		const char architecture[] = "gfx1201";
		response->architecture_length = sizeof(architecture) - 1;
		std::memcpy(response->architecture, architecture,
		            response->architecture_length);
	}
	response->max_queues = 0;
	response->max_inflight_submissions = 0;
	response->max_buffer_bytes = kMaxHostVisibleBufferBytes;
	response->max_mapping_bytes = kMaxHostVisibleBufferBytes;
	response->max_executable_bytes = 0;
	response->min_buffer_alignment = kMinHostVisibleBufferAlignment;
	response->min_mapping_alignment = kMinHostVisibleBufferAlignment;
	response->timestamp_frequency_hz = 0;
	response->device_epoch = ivars->device_epoch;
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDriver::QueryHealth(TGPUHealthFaultQueryResponse *response)
{
	if (!ivars || !response) return kIOReturnBadArgument;
	std::memset(response, 0, sizeof(*response));
	response->health_state = ivars->health_state;
	response->fault_kind =
	    ivars->health_state == TGPU_HEALTH_READY ? TGPU_FAULT_NONE
	                                             : TGPU_FAULT_DEVICE_FAULT;
	response->fault_id = ivars->failure_stage == TGPU_FAILURE_NONE ? 0 : 1;
	response->failure_stage = ivars->failure_stage;
	response->terminal_status = ivars->failure_status;
	response->text_length = 0;
	while (response->text_length < TGPU_MAX_FAULT_TEXT_BYTES &&
	       ivars->failure_text[response->text_length] != '\0') {
		++response->text_length;
	}
	std::memcpy(response->failure_text, ivars->failure_text,
	            response->text_length);
	response->device_epoch = ivars->device_epoch;
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDriver::MapBar(uint32_t bar, IOMemoryDescriptor** memory)
{
	uint8_t barMemoryIndex, barMemoryType;
	uint64_t barMemorySize;
	kern_return_t err = ivars->pci->GetBARInfo(bar, &barMemoryIndex, &barMemorySize, &barMemoryType);
	if (err) return err;
	os_log(OS_LOG_DEFAULT, "tinygpu: bar mapping %d idx=%d", bar, barMemoryIndex);
	return ivars->pci->_CopyDeviceMemoryWithIndex(barMemoryIndex, memory, this);
}

static kern_return_t WriteDMASegments(IOMemoryDescriptor* mem, IOAddressSegment* segments, uint32_t segCount,
                                      uint64_t mapOffset = 0, uint64_t mapSize = 0)
{
	// write dma segments to mapped memory as [addr0, len0, addr1, len1, ..., 0, 0]

	IOMemoryMap* map = nullptr;
	kern_return_t err = mem->CreateMapping(0, 0, 0, mapOffset, mapSize, &map);
	if (err || !map) return err ?: kIOReturnError;

	uint64_t* out = (uint64_t*)map->GetAddress();
	for (uint32_t i = 0; i < segCount; i++) { out[i * 2] = segments[i].address; out[i * 2 + 1] = segments[i].length; }
	out[segCount * 2] = 0; out[segCount * 2 + 1] = 0;
	map->release();
	return 0;
}

kern_return_t TinyGPUDriver::SetupDMA(IOMemoryDescriptor* memory, uint64_t size, IODMACommand** outCmd,
                                       IOAddressSegment* segments, uint32_t* segCount)
{
	IODMACommandSpecification dmaSpec = {.options = 0, .maxAddressBits = 40};
	IODMACommand* dmaCmd = nullptr;

	kern_return_t err = IODMACommand::Create(ivars->pci, kIODMACommandCreateNoOptions, &dmaSpec, &dmaCmd);
	if (err) { os_log(OS_LOG_DEFAULT, "tinygpu: DMA create failed err=%d", err); return err; }

	uint64_t flags = kIOMemoryDirectionInOut;
	err = dmaCmd->PrepareForDMA(kIODMACommandPrepareForDMANoOptions, memory, 0, size, &flags, segCount, segments);
	if (err) { os_log(OS_LOG_DEFAULT, "tinygpu: PrepareForDMA failed err=%d", err); dmaCmd->release(); return err; }

	*outCmd = dmaCmd;
	return 0;
}

kern_return_t TinyGPUDriver::CreateDMA(size_t size, TinyGPUCreateDMAResp* dmaDesc)
{
	IOBufferMemoryDescriptor* sharedBuf = nullptr;
	kern_return_t err = IOBufferMemoryDescriptor::Create(kIOMemoryDirectionInOut, size, IOVMPageSize, &sharedBuf);
	if (err) { os_log(OS_LOG_DEFAULT, "tinygpu: alloc failed err=%d", err); return err; }

	IODMACommand* dmaCmd = nullptr;
	IOAddressSegment segments[32];
	uint32_t segCount = 32;
	err = SetupDMA(sharedBuf, size, &dmaCmd, segments, &segCount);
	if (err) { sharedBuf->release(); return err; }

	err = WriteDMASegments(sharedBuf, segments, segCount, IOVMPageSize, IOVMPageSize);
	if (err) { dmaCmd->CompleteDMA(kIODMACommandCompleteDMANoOptions); dmaCmd->release(); sharedBuf->release(); return err; }

	dmaDesc->sharedBuf = sharedBuf;
	dmaDesc->dmaCmd = dmaCmd;
	os_log(OS_LOG_DEFAULT, "tinygpu: CreateDMA size=0x%zx segs=%u", size, segCount);
	return 0;
}

kern_return_t TinyGPUDriver::CfgRead(uint32_t off, uint32_t size, uint32_t* outVal)
{
  if (!ivars->pci || !outVal) return kIOReturnNotReady;

  if (size == 1) {
	uint8_t v8 = 0;
	ivars->pci->ConfigurationRead8(off, &v8);
	*outVal = v8;
  } else if (size == 2) {
	uint16_t v16 = 0;
	ivars->pci->ConfigurationRead16(off, &v16);
	*outVal = v16;
  } else if (size == 4) {
	uint32_t v32 = 0;
	ivars->pci->ConfigurationRead32(off, &v32);
	*outVal = v32;
  }
  return 0;
}

kern_return_t TinyGPUDriver::CfgWrite(uint32_t off, uint32_t size, uint32_t val)
{
  if (!ivars->pci) return kIOReturnNotReady;
  if (size == 1) ivars->pci->ConfigurationWrite8 (off, (uint8_t)val);
  else if (size == 2) ivars->pci->ConfigurationWrite16(off, (uint16_t)val);
  else if (size == 4) ivars->pci->ConfigurationWrite32(off, (uint32_t)val);
  return 0;
}

kern_return_t TinyGPUDriver::ResetDevice()
{
	if (!ivars->pci) return kIOReturnNotReady;
	kern_return_t ret = ivars->pci->Reset(kIOPCIDeviceResetTypeFunctionReset);
	return ret == kIOReturnSuccess ? ret : ivars->pci->Reset(kIOPCIDeviceResetTypeHotReset);
}

IOPCIDevice* TinyGPUDriver::GetPCI()
{
	return ivars->pci;
}
