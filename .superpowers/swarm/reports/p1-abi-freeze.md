# P1 TinyGPU user-client ABI freeze

**Task set:** P1 / 1 — ABI, security, entitlement, and command freeze  
**Status:** Done
**Owner:** `P1ABI`  
**Report:** `.superpowers/swarm/reports/p1-abi-freeze.md`  
**Review date:** 2026-08-25  
**Implementation boundary:** This report freezes the contract only. It does not edit the in-repository TinyGPU source tree, implement DEXT behavior, or edit the shared validation ledger. Later TinyGPU source implementation is isolated at `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu` on branch `feature/r9700-products-wave-a`. The in-repository source tree is read-only for this correction. No validation, build, install, hardware, or package-manager command was run.

## Decision summary

TinyGPU's public user-client contract is a fixed-width, little-endian, naturally 8-byte-aligned structured ABI named `TGPU`, **major 1, minor 0**. Every request and response starts with a common header. A driver validates the complete header, selector, structure size, reserved fields, counts, ranges, and checked arithmetic before reading an operation field or mutating hardware. Responses never contain a pointer, physical address, BAR mapping, GPU virtual address, raw register value, PM4 packet, doorbell, MQD, or HQD.

The old `ReadCfg`/`WriteCfg`/`Reset`/`PrepareDMA` selector meanings are not compatibility aliases. The unauthenticated `tinygpu/Shared/server.c` socket/proxy is quarantined historical diagnostic source only: it is excluded from every P1 product, build, conformance, validation, and Release path, is never launched by a P1 command, and cannot be used as an inference or acceptance transport. Normal access is the entitlement-gated DriverKit user client. Any future diagnostic transport is a separate user-client class with a separate exact entitlement and authenticated owner policy; it is not the legacy proxy.

Executable admission is authorized only for an entitled client role and an image digest recomputed by the driver from an immutable private image copy. The driver validates the complete closed code-object, descriptor, relocation, kernarg, wave, and resource set against the R9700 policy before admission and installs the executable in a per-client VM context. A client-supplied pack ID or digest is audit metadata and a consistency check only; it is never the authorization trust anchor. Kernel Pack/P3 records are not parsed in DriverKit.

Queue rings, MQDs, HQDs, doorbells, fence-control storage, and all hardware-consumed control state are driver-owned and non-client-mappable. Client command bytes and binding records are copied and validated into driver-owned submission storage before queue mutation; hardware never consumes a mutable client page. Handles are mandatory typed, nonzero, per-connection slot-plus-generation/epoch capabilities. Generation exhaustion fails closed and generation/epoch values never wrap or reissue.

Queue reset is owner-only. Device reset and device-wide fault detail require distinct recovery/diagnostic user-client roles and exact entitlements, serialized state preconditions, a rate limit, and bounded redacted output. Normal inference clients cannot reset the device or read another client's device-wide fault detail. Release transport and personality scope is exactly AMD PCI `1002:7551`; wildcard and vendor-wide matches are local NoSIP development-only and must be absent from Release.

Task set 1 is **Done**. Focused security re-review found zero remaining Critical/Important findings. On 2026-08-26 the supervisor selected Xcode 26.6 build `17F113` with DriverKit SDK 25.5, clearing the implementation prerequisite for task sets 2–4. External production-distribution credentials remain a separate promotion gate.

## Normative ABI definition

### Encoding, alignment, and bounds

The following declarations are the canonical byte contract to place in the separately authorized TinyGPU source boundary (`TinyGPUDriver.iig` and `TinyGPUDriverUserClient.iig`) and consume from the one future conformance client. They are recorded here so both repositories can implement the same ABI; this task does not add a duplicate source header. The wire encoding is little-endian. Public structures use ordinary C layout with maximum alignment 8 and explicit reserved fields. No public structure contains `pointer`, `size_t`, `bool`, a flexible array, a file descriptor integer, a physical address, or an implicit address-bearing field.

```cpp
#include <stddef.h>
#include <stdint.h>

constexpr uint32_t TGPU_ABI_MAJOR = 1;
constexpr uint32_t TGPU_ABI_MINOR = 0;
constexpr uint32_t TGPU_MAX_STRUCT_BYTES = 4096;
constexpr uint32_t TGPU_MAX_COMMAND_RECORDS = 256;
constexpr uint32_t TGPU_MAX_COMMAND_RECORD_BYTES = 128;
constexpr uint32_t TGPU_MAX_COMMAND_BYTES = 65536;
constexpr uint32_t TGPU_MAX_BINDINGS = 64;
constexpr uint32_t TGPU_MAX_WAIT_FENCES = 64;
constexpr uint32_t TGPU_MAX_FAULT_TEXT_BYTES = 192;
constexpr uint32_t TGPU_MAX_ENTRY_ID_BYTES = 16;
constexpr uint32_t TGPU_MAX_TARGET_BYTES = 16;
constexpr uint32_t TGPU_MAX_TIMESTAMP_LABEL_BYTES = 32;
constexpr uint64_t TGPU_MAX_WAIT_NS = 60000000000ULL;
constexpr uint64_t TGPU_MIN_DEVICE_RESET_INTERVAL_NS = 1000000000ULL;

// Every v1.0 request-header flags field is zero. Future minor versions may
// allocate an explicitly named bit only after reopening this ABI review.
constexpr uint32_t TGPU_REQUEST_FLAGS_V1_0 = 0;
constexpr uint32_t TGPU_RECORD_FLAGS_V1_0 = 0;
constexpr uint32_t TGPU_IMPORT_FLAGS_V1_0 = 0;
constexpr uint32_t TGPU_MAP_FLAGS_V1_0 = 0;
constexpr uint32_t TGPU_QUEUE_FLAGS_V1_0 = 0;
constexpr uint32_t TGPU_QUERY_FLAGS_V1_0 = 0;
constexpr uint32_t TGPU_RESET_FLAGS_V1_0 = 0;

enum TGPUStatus : uint32_t {
  TGPU_STATUS_OK = 0,
  TGPU_STATUS_INVALID_REQUEST = 1,
  TGPU_STATUS_ABI_MISMATCH = 2,
  TGPU_STATUS_UNSUPPORTED = 3,
  TGPU_STATUS_PERMISSION_DENIED = 4,
  TGPU_STATUS_INVALID_HANDLE = 5,
  TGPU_STATUS_RANGE = 6,
  TGPU_STATUS_ALIGNMENT = 7,
  TGPU_STATUS_RESOURCE_EXHAUSTED = 8,
  TGPU_STATUS_BUSY = 9,
  TGPU_STATUS_TIMEOUT = 10,
  TGPU_STATUS_EXECUTABLE_REJECTED = 11,
  TGPU_STATUS_SUBMISSION_REJECTED = 12,
  TGPU_STATUS_DEVICE_FAULT = 13,
  TGPU_STATUS_DEVICE_LOST = 14,
  TGPU_STATUS_RESET_REQUIRED = 15,
  TGPU_STATUS_CANCELED = 16,
  TGPU_STATUS_INTERNAL = 17,
};

enum TGPUFailureStage : uint32_t {
  TGPU_FAILURE_NONE = 0,
  TGPU_FAILURE_ATTACH = 1,
  TGPU_FAILURE_POWER = 2,
  TGPU_FAILURE_FIRMWARE = 3,
  TGPU_FAILURE_MEMORY = 4,
  TGPU_FAILURE_QUEUE = 5,
  TGPU_FAILURE_EXECUTABLE = 6,
  TGPU_FAILURE_SUBMIT = 7,
  TGPU_FAILURE_FENCE = 8,
  TGPU_FAILURE_INTERRUPT = 9,
  TGPU_FAILURE_RESET = 10,
  TGPU_FAILURE_TEARDOWN = 11,
  TGPU_FAILURE_DIAGNOSTIC = 12,
};

enum TGPUClientRole : uint32_t {
  TGPU_CLIENT_INFERENCE = 1,
  TGPU_CLIENT_RECOVERY = 2,
  TGPU_CLIENT_DIAGNOSTIC = 3,
};

enum TGPUHandleKind : uint32_t {
  TGPU_HANDLE_BUFFER = 1,
  TGPU_HANDLE_MAPPING = 2,
  TGPU_HANDLE_QUEUE = 3,
  TGPU_HANDLE_EXECUTABLE = 4,
  TGPU_HANDLE_FENCE = 5,
  TGPU_HANDLE_SUBMISSION = 6,
};

enum TGPUHealthState : uint32_t {
  TGPU_HEALTH_DISCONNECTED = 0,
  TGPU_HEALTH_INITIALIZING = 1,
  TGPU_HEALTH_READY = 2,
  TGPU_HEALTH_DEGRADED = 3,
  TGPU_HEALTH_FAULTED = 4,
  TGPU_HEALTH_RESETTING = 5,
  TGPU_HEALTH_UNAVAILABLE = 6,
};

enum TGPUFaultKind : uint32_t {
  TGPU_FAULT_NONE = 0,
  TGPU_FAULT_TIMEOUT = 1,
  TGPU_FAULT_PAGE_FAULT = 2,
  TGPU_FAULT_ILLEGAL_COMMAND = 3,
  TGPU_FAULT_EXECUTABLE_REJECTED = 4,
  TGPU_FAULT_QUEUE_HANG = 5,
  TGPU_FAULT_DEVICE_FAULT = 6,
  TGPU_FAULT_RESET_REQUIRED = 7,
};

enum TGPUResetReason : uint32_t {
  TGPU_RESET_REASON_FAULT = 1,
  TGPU_RESET_REASON_TIMEOUT = 2,
  TGPU_RESET_REASON_ADMIN = 3,
};
enum TGPUExecutionClass : uint32_t {
  TGPU_EXECUTION_SDMA = 1,
  TGPU_EXECUTION_COMPUTE = 2,
};

enum TGPUCommandKind : uint16_t {
  TGPU_COMMAND_COPY = 1,
  TGPU_COMMAND_FILL = 2,
  TGPU_COMMAND_DISPATCH = 3,
  TGPU_COMMAND_BARRIER = 4,
  TGPU_COMMAND_TIMESTAMP = 5,
};

constexpr uint64_t TGPU_FEATURE_BUFFER_ALLOCATE = 1ULL << 0;
constexpr uint64_t TGPU_FEATURE_BUFFER_IMPORT = 1ULL << 1;
constexpr uint64_t TGPU_FEATURE_BUFFER_MAP = 1ULL << 2;
constexpr uint64_t TGPU_FEATURE_QUEUE = 1ULL << 3;
constexpr uint64_t TGPU_FEATURE_EXECUTABLE = 1ULL << 4;
constexpr uint64_t TGPU_FEATURE_SUBMIT = 1ULL << 5;
constexpr uint64_t TGPU_FEATURE_FENCE = 1ULL << 6;
constexpr uint64_t TGPU_FEATURE_TIMESTAMP = 1ULL << 7;
constexpr uint64_t TGPU_FEATURE_FAULT_QUERY = 1ULL << 8;
constexpr uint64_t TGPU_FEATURE_QUEUE_RESET = 1ULL << 9;
constexpr uint64_t TGPU_FEATURE_DEVICE_RESET = 1ULL << 10;
constexpr uint64_t TGPU_FEATURE_DIAGNOSTIC_MMIO = 1ULL << 11;
constexpr uint64_t TGPU_FEATURE_MASK_V1_0 = (1ULL << 12) - 1;

constexpr uint32_t TGPU_MEMORY_HOST_VISIBLE = 1U << 0;
constexpr uint32_t TGPU_MEMORY_DEVICE_LOCAL = 1U << 1;
constexpr uint32_t TGPU_MEMORY_MASK_V1_0 = 0x3U;

constexpr uint32_t TGPU_ACCESS_READ = 1U << 0;
constexpr uint32_t TGPU_ACCESS_WRITE = 1U << 1;
constexpr uint32_t TGPU_ACCESS_EXECUTE = 1U << 2;
constexpr uint32_t TGPU_ACCESS_MASK_V1_0 = 0x7U;

constexpr uint32_t TGPU_RESOURCE_IMAGE = 1U << 0;
constexpr uint32_t TGPU_RESOURCE_KERNARG = 1U << 1;
constexpr uint32_t TGPU_RESOURCE_COMMAND = 1U << 2;
constexpr uint32_t TGPU_RESOURCE_STAGING = 1U << 3;
constexpr uint32_t TGPU_RESOURCE_MASK_V1_0 = 0xFU;
// Queue-control, fence-control, MQD/HQD, and doorbell storage have no public
// resource flag. Any bit outside TGPU_RESOURCE_MASK_V1_0 is rejected.

constexpr uint32_t TGPU_HEALTH_SCOPE_CLIENT = 1;
constexpr uint32_t TGPU_HEALTH_SCOPE_QUEUE = 2;
constexpr uint32_t TGPU_HEALTH_SCOPE_DEVICE = 3;
constexpr uint32_t TGPU_HEALTH_SCOPE_MASK_V1_0 = 0x3U;

constexpr uint32_t TGPU_QUEUE_STATE_IDLE = 0;
constexpr uint32_t TGPU_QUEUE_STATE_READY = 1;
constexpr uint32_t TGPU_QUEUE_STATE_RESETTING = 2;
constexpr uint32_t TGPU_QUEUE_STATE_FAULTED = 3;
constexpr uint32_t TGPU_QUEUE_STATE_CLOSED = 4;

using TGPUBufferHandle = uint64_t;
using TGPUMappingHandle = uint64_t;
using TGPUQueueHandle = uint64_t;
using TGPUExecutableHandle = uint64_t;
using TGPUFenceHandle = uint64_t;
using TGPUSubmissionHandle = uint64_t;

struct TGPURequestHeader {
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t struct_size;
  uint32_t flags;
  uint64_t request_id;
}; // 24 bytes

struct TGPUResponseHeader {
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t struct_size;
  uint32_t flags;
  uint32_t status;
  uint32_t failure_stage;
  uint64_t request_id;
}; // 32 bytes

struct TGPUQueryCapabilitiesRequest {
  TGPURequestHeader header;
}; // 24

struct TGPUCapabilitiesResponse {
  TGPUResponseHeader header;
  uint64_t feature_bits;
  uint32_t memory_domain_bits;
  uint32_t vendor_id;
  uint32_t device_id;
  uint32_t architecture_length;
  uint8_t architecture[16];
  uint32_t max_queues;
  uint32_t max_inflight_submissions;
  uint64_t max_buffer_bytes;
  uint64_t max_mapping_bytes;
  uint64_t max_executable_bytes;
  uint64_t min_buffer_alignment;
  uint64_t min_mapping_alignment;
  uint64_t timestamp_frequency_hz;
  uint64_t device_epoch;
  uint64_t reserved0;
}; // 144

struct TGPUBufferAllocateRequest {
  TGPURequestHeader header;
  uint64_t size;
  uint64_t alignment;
  uint32_t memory_domain;
  uint32_t access_flags;
  uint32_t resource_flags;
  uint32_t reserved0;
}; // 56

struct TGPUBufferAllocateResponse {
  TGPUResponseHeader header;
  TGPUBufferHandle buffer_handle;
  uint64_t committed_size;
  uint32_t granted_access;
  uint32_t memory_domain;
  uint64_t reserved0;
}; // 64

struct TGPUBufferImportRequest {
  TGPURequestHeader header;
  uint64_t requested_size;
  uint32_t memory_domain;
  uint32_t access_flags;
  uint32_t import_flags;
  uint32_t reserved0;
}; // 48

struct TGPUBufferImportResponse {
  TGPUResponseHeader header;
  TGPUBufferHandle buffer_handle;
  uint64_t imported_size;
  uint32_t granted_access;
  uint32_t memory_domain;
  uint64_t reserved0;
}; // 64

struct TGPUBufferMapRequest {
  TGPURequestHeader header;
  TGPUBufferHandle buffer_handle;
  uint64_t offset;
  uint64_t length;
  uint32_t access_flags;
  uint32_t map_flags;
  uint64_t reserved0;
}; // 64

struct TGPUBufferMapResponse {
  TGPUResponseHeader header;
  TGPUMappingHandle mapping_handle;
  TGPUBufferHandle buffer_handle;
  uint64_t offset;
  uint64_t length;
  uint32_t granted_access;
  uint32_t reserved0;
}; // 72

struct TGPUBufferUnmapRequest {
  TGPURequestHeader header;
  TGPUMappingHandle mapping_handle;
  uint64_t reserved0;
}; // 40

struct TGPUBufferReleaseRequest {
  TGPURequestHeader header;
  TGPUBufferHandle buffer_handle;
  uint64_t reserved0;
}; // 40

struct TGPUStatusResponse {
  TGPUResponseHeader header;
}; // 32

struct TGPUQueueCreateRequest {
  TGPURequestHeader header;
  uint32_t execution_class;
  uint32_t queue_flags;
  uint32_t requested_depth;
  uint32_t reserved0;
}; // 40

struct TGPUQueueCreateResponse {
  TGPUResponseHeader header;
  TGPUQueueHandle queue_handle;
  uint32_t effective_depth;
  uint32_t effective_inflight;
  uint64_t queue_epoch;
  uint64_t reserved0;
}; // 64

struct TGPUQueueDestroyRequest {
  TGPURequestHeader header;
  TGPUQueueHandle queue_handle;
  uint64_t reserved0;
}; // 40

struct TGPUExecutableAdmitRequest {
  TGPURequestHeader header;
  TGPUBufferHandle image_buffer_handle;
  uint64_t image_offset;
  uint64_t image_length;
  uint8_t target[16];
  uint32_t pack_schema_major;
  uint32_t pack_schema_minor;
  uint32_t code_object_version;
  uint8_t entry_id[16];
  uint8_t pack_id[16];
  uint8_t claimed_image_digest[32];
  uint32_t reserved0;
}; // 144

struct TGPUExecutableAdmitResponse {
  TGPUResponseHeader header;
  TGPUExecutableHandle executable_handle;
  uint8_t recomputed_image_digest[32];
  uint8_t pack_id[16];
  uint8_t entry_id[16];
  uint8_t target[16];
  uint64_t device_epoch;
}; // 128

struct TGPUExecutableReleaseRequest {
  TGPURequestHeader header;
  TGPUExecutableHandle executable_handle;
  uint64_t reserved0;
}; // 40

struct TGPUFenceWaitRef {
  TGPUFenceHandle fence_handle;
  uint64_t fence_value;
}; // 16

struct TGPUCommandRecordHeader {
  uint16_t command_kind;
  uint16_t flags;
  uint32_t record_size;
  uint64_t sequence;
}; // 16

struct TGPUCommandBinding {
  TGPUBufferHandle buffer_handle;
  uint64_t offset;
  uint64_t length;
  uint32_t access_flags;
  uint32_t reserved0;
}; // 32

struct TGPUCopyCommandRecord {
  TGPUCommandRecordHeader header;
  TGPUBufferHandle source_buffer;
  uint64_t source_offset;
  TGPUBufferHandle destination_buffer;
  uint64_t destination_offset;
  uint64_t length;
  uint32_t access_flags;
  uint32_t reserved0;
  uint64_t reserved[8];
}; // 128

struct TGPUFillCommandRecord {
  TGPUCommandRecordHeader header;
  TGPUBufferHandle destination_buffer;
  uint64_t destination_offset;
  uint64_t length;
  uint32_t pattern_width;
  uint32_t access_flags;
  uint8_t pattern[16];
  uint8_t reserved[64];
}; // 128

struct TGPUDispatchCommandRecord {
  TGPUCommandRecordHeader header;
  TGPUExecutableHandle executable_handle;
  uint8_t entry_id[16];
  uint32_t first_binding;
  uint32_t binding_count;
  uint32_t kernarg_binding_index;
  uint32_t reserved0;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  uint32_t block_x;
  uint32_t block_y;
  uint32_t block_z;
  uint32_t dynamic_lds_bytes;
  uint32_t reserved[11];
}; // 128

struct TGPUBarrierCommandRecord {
  TGPUCommandRecordHeader header;
  uint32_t visibility_scope;
  uint32_t access_scope;
  uint32_t reserved0;
  uint32_t reserved1;
  uint64_t reserved[12];
}; // 128

struct TGPUTimestampCommandRecord {
  TGPUCommandRecordHeader header;
  uint8_t label[32];
  uint32_t point;
  uint32_t reserved0;
  uint64_t reserved[9];
}; // 128

struct TGPUSubmitRequest {
  TGPURequestHeader header;
  TGPUQueueHandle queue_handle;
  TGPUExecutableHandle executable_handle;
  TGPUBufferHandle command_buffer_handle;
  uint64_t command_offset;
  uint64_t command_bytes;
  uint32_t command_count;
  uint32_t binding_table_offset;
  uint32_t binding_count;
  uint32_t wait_count;
  uint64_t signal_value;
  TGPUFenceWaitRef waits[64];
  uint64_t reserved0;
}; // 1120

struct TGPUSubmitResponse {
  TGPUResponseHeader header;
  TGPUSubmissionHandle submission_handle;
  TGPUFenceHandle fence_handle;
  uint64_t signal_value;
  uint64_t queue_epoch;
  uint64_t device_epoch;
  uint64_t reserved0;
}; // 80

struct TGPUFenceWaitRequest {
  TGPURequestHeader header;
  TGPUFenceHandle fence_handle;
  uint64_t fence_value;
  uint64_t timeout_ns;
  uint64_t reserved0;
}; // 56

struct TGPUFenceWaitResponse {
  TGPUResponseHeader header;
  uint64_t completed_value;
  uint64_t requested_value;
  uint32_t terminal_status;
  uint32_t reserved0;
  uint64_t queue_epoch;
}; // 64

struct TGPUTimestampQueryRequest {
  TGPURequestHeader header;
  TGPUSubmissionHandle submission_handle;
  TGPUFenceHandle fence_handle;
  uint32_t point;
  uint32_t reserved0;
}; // 48

struct TGPUTimestampQueryResponse {
  TGPUResponseHeader header;
  TGPUSubmissionHandle submission_handle;
  TGPUFenceHandle fence_handle;
  uint64_t start_ticks;
  uint64_t end_ticks;
  uint64_t timestamp_frequency_hz;
  uint64_t conversion_numerator;
  uint64_t conversion_denominator;
  uint64_t reserved0;
}; // 96

struct TGPUHealthFaultQueryRequest {
  TGPURequestHeader header;
  uint32_t scope;
  uint32_t query_flags;
  uint64_t cursor;
  TGPUQueueHandle queue_handle;
  TGPUSubmissionHandle submission_handle;
  uint64_t reserved[2];
}; // 72

struct TGPUHealthFaultQueryResponse {
  TGPUResponseHeader header;
  uint32_t health_state;
  uint32_t fault_kind;
  uint64_t fault_id;
  uint32_t failure_stage;
  uint32_t reserved0;
  TGPUQueueHandle queue_handle;
  TGPUSubmissionHandle submission_handle;
  TGPUExecutableHandle executable_handle;
  uint32_t terminal_status;
  uint32_t text_length;
  uint8_t failure_text[192];
  uint64_t device_epoch;
  uint64_t reserved1;
}; // 296

struct TGPUQueueResetRequest {
  TGPURequestHeader header;
  TGPUQueueHandle queue_handle;
  uint32_t reset_flags;
  uint32_t reserved0;
}; // 40

struct TGPUQueueResetResponse {
  TGPUResponseHeader header;
  TGPUQueueHandle queue_handle;
  uint32_t queue_state;
  uint32_t reserved0;
  uint64_t device_epoch;
  uint32_t canceled_submissions;
  uint32_t reserved1;
  uint64_t queue_epoch;
  uint64_t reserved2;
}; // 80

struct TGPUDeviceResetRequest {
  TGPURequestHeader header;
  uint32_t reset_flags;
  uint32_t reason_code;
  uint64_t reserved0;
}; // 40

struct TGPUDeviceResetResponse {
  TGPUResponseHeader header;
  uint32_t health_state;
  uint32_t reserved0;
  uint64_t previous_device_epoch;
  uint64_t device_epoch;
}; // 56

struct TGPUDiagnosticMMIOReadRequest {
  TGPURequestHeader header;
  uint64_t bar_relative_offset;
  uint32_t width_bytes;
  uint32_t reserved0;
}; // 40

struct TGPUDiagnosticMMIOReadResponse {
  TGPUResponseHeader header;
  uint64_t value;
}; // 40

struct TGPUDiagnosticMMIOWriteRequest {
  TGPURequestHeader header;
  uint64_t bar_relative_offset;
  uint32_t width_bytes;
  uint32_t reserved0;
  uint64_t value;
}; // 48
```

`TGPUBufferImportRequest` uses the DriverKit `structureInputDescriptor` sideband for the checked `IOMemoryDescriptor`; that sideband is not a wire pointer, integer descriptor, or segment array. The driver obtains and validates the descriptor supplied for this call, retains it only while the buffer handle is live, and never serializes its address or physical segments.

The canonical declarations include every selector request, response, wait element, binding element, and command record. The required compile-time layout assertions are:

```cpp
static_assert(sizeof(TGPURequestHeader) == 24);
static_assert(sizeof(TGPUResponseHeader) == 32);
static_assert(offsetof(TGPURequestHeader, request_id) == 16);
static_assert(offsetof(TGPUResponseHeader, status) == 16);
static_assert(offsetof(TGPUResponseHeader, request_id) == 24);

static_assert(sizeof(TGPUQueryCapabilitiesRequest) == 24);
static_assert(sizeof(TGPUCapabilitiesResponse) == 144);
static_assert(offsetof(TGPUCapabilitiesResponse, feature_bits) == 32);
static_assert(offsetof(TGPUCapabilitiesResponse, architecture) == 56);
static_assert(offsetof(TGPUCapabilitiesResponse, device_epoch) == 128);
static_assert(sizeof(TGPUBufferAllocateRequest) == 56);
static_assert(sizeof(TGPUBufferAllocateResponse) == 64);
static_assert(sizeof(TGPUBufferImportRequest) == 48);
static_assert(sizeof(TGPUBufferImportResponse) == 64);
static_assert(sizeof(TGPUBufferMapRequest) == 64);
static_assert(sizeof(TGPUBufferMapResponse) == 72);
static_assert(offsetof(TGPUBufferMapRequest, offset) == 32);
static_assert(offsetof(TGPUBufferMapResponse, granted_access) == 64);
static_assert(sizeof(TGPUBufferUnmapRequest) == 40);
static_assert(sizeof(TGPUBufferReleaseRequest) == 40);
static_assert(sizeof(TGPUStatusResponse) == 32);

static_assert(sizeof(TGPUQueueCreateRequest) == 40);
static_assert(sizeof(TGPUQueueCreateResponse) == 64);
static_assert(sizeof(TGPUQueueDestroyRequest) == 40);
static_assert(sizeof(TGPUExecutableAdmitRequest) == 144);
static_assert(offsetof(TGPUExecutableAdmitRequest, image_buffer_handle) == 24);
static_assert(offsetof(TGPUExecutableAdmitRequest, target) == 48);
static_assert(offsetof(TGPUExecutableAdmitRequest, claimed_image_digest) == 108);
static_assert(sizeof(TGPUExecutableAdmitResponse) == 128);
static_assert(offsetof(TGPUExecutableAdmitResponse, recomputed_image_digest) == 40);
static_assert(sizeof(TGPUExecutableReleaseRequest) == 40);

static_assert(sizeof(TGPUFenceWaitRef) == 16);
static_assert(sizeof(TGPUCommandRecordHeader) == 16);
static_assert(sizeof(TGPUCommandBinding) == 32);
static_assert(sizeof(TGPUCopyCommandRecord) == 128);
static_assert(sizeof(TGPUFillCommandRecord) == 128);
static_assert(sizeof(TGPUDispatchCommandRecord) == 128);
static_assert(sizeof(TGPUBarrierCommandRecord) == 128);
static_assert(sizeof(TGPUTimestampCommandRecord) == 128);
static_assert(offsetof(TGPUDispatchCommandRecord, first_binding) == 40);
static_assert(offsetof(TGPUDispatchCommandRecord, kernarg_binding_index) == 48);
static_assert(sizeof(TGPUSubmitRequest) == 1120);
static_assert(offsetof(TGPUSubmitRequest, command_offset) == 48);
static_assert(offsetof(TGPUSubmitRequest, binding_table_offset) == 68);
static_assert(offsetof(TGPUSubmitRequest, waits) == 88);
static_assert(sizeof(TGPUSubmitResponse) == 80);
static_assert(sizeof(TGPUFenceWaitRequest) == 56);
static_assert(sizeof(TGPUFenceWaitResponse) == 64);
static_assert(sizeof(TGPUTimestampQueryRequest) == 48);
static_assert(sizeof(TGPUTimestampQueryResponse) == 96);
static_assert(sizeof(TGPUHealthFaultQueryRequest) == 72);
static_assert(sizeof(TGPUHealthFaultQueryResponse) == 296);
static_assert(offsetof(TGPUHealthFaultQueryResponse, failure_text) == 88);
static_assert(sizeof(TGPUQueueResetRequest) == 40);
static_assert(sizeof(TGPUQueueResetResponse) == 80);
static_assert(sizeof(TGPUDeviceResetRequest) == 40);
static_assert(sizeof(TGPUDeviceResetResponse) == 56);
static_assert(sizeof(TGPUDiagnosticMMIOReadRequest) == 40);
static_assert(sizeof(TGPUDiagnosticMMIOReadResponse) == 40);
static_assert(sizeof(TGPUDiagnosticMMIOWriteRequest) == 48);
```
```cpp
static_assert(offsetof(TGPUCapabilitiesResponse, memory_domain_bits) == 40);
static_assert(offsetof(TGPUCapabilitiesResponse, vendor_id) == 44);
static_assert(offsetof(TGPUCapabilitiesResponse, device_id) == 48);
static_assert(offsetof(TGPUCapabilitiesResponse, architecture_length) == 52);
static_assert(offsetof(TGPUCapabilitiesResponse, max_queues) == 72);
static_assert(offsetof(TGPUCapabilitiesResponse, max_inflight_submissions) == 76);
static_assert(offsetof(TGPUCapabilitiesResponse, max_buffer_bytes) == 80);
static_assert(offsetof(TGPUCapabilitiesResponse, max_mapping_bytes) == 88);
static_assert(offsetof(TGPUCapabilitiesResponse, max_executable_bytes) == 96);
static_assert(offsetof(TGPUCapabilitiesResponse, min_buffer_alignment) == 104);
static_assert(offsetof(TGPUCapabilitiesResponse, min_mapping_alignment) == 112);
static_assert(offsetof(TGPUCapabilitiesResponse, timestamp_frequency_hz) == 120);
static_assert(offsetof(TGPUCapabilitiesResponse, reserved0) == 136);
static_assert(offsetof(TGPUBufferAllocateRequest, size) == 24);
static_assert(offsetof(TGPUBufferAllocateRequest, alignment) == 32);
static_assert(offsetof(TGPUBufferAllocateRequest, memory_domain) == 40);
static_assert(offsetof(TGPUBufferAllocateRequest, access_flags) == 44);
static_assert(offsetof(TGPUBufferAllocateRequest, resource_flags) == 48);
static_assert(offsetof(TGPUBufferAllocateResponse, buffer_handle) == 32);
static_assert(offsetof(TGPUBufferAllocateResponse, committed_size) == 40);
static_assert(offsetof(TGPUBufferAllocateResponse, granted_access) == 48);
static_assert(offsetof(TGPUBufferImportRequest, requested_size) == 24);
static_assert(offsetof(TGPUBufferImportRequest, memory_domain) == 32);
static_assert(offsetof(TGPUBufferImportRequest, access_flags) == 36);
static_assert(offsetof(TGPUBufferImportRequest, import_flags) == 40);
static_assert(offsetof(TGPUBufferMapRequest, buffer_handle) == 24);
static_assert(offsetof(TGPUBufferMapRequest, length) == 40);
static_assert(offsetof(TGPUBufferMapRequest, access_flags) == 48);
static_assert(offsetof(TGPUBufferMapResponse, mapping_handle) == 32);
static_assert(offsetof(TGPUBufferMapResponse, buffer_handle) == 40);
static_assert(offsetof(TGPUBufferMapResponse, offset) == 48);
static_assert(offsetof(TGPUBufferMapResponse, length) == 56);
static_assert(offsetof(TGPUBufferUnmapRequest, mapping_handle) == 24);
static_assert(offsetof(TGPUBufferReleaseRequest, buffer_handle) == 24);
static_assert(offsetof(TGPUQueueCreateRequest, execution_class) == 24);
static_assert(offsetof(TGPUQueueCreateRequest, queue_flags) == 28);
static_assert(offsetof(TGPUQueueCreateRequest, requested_depth) == 32);
static_assert(offsetof(TGPUQueueCreateResponse, queue_handle) == 32);
static_assert(offsetof(TGPUQueueCreateResponse, effective_depth) == 40);
static_assert(offsetof(TGPUQueueCreateResponse, effective_inflight) == 44);
static_assert(offsetof(TGPUQueueCreateResponse, queue_epoch) == 48);
static_assert(offsetof(TGPUExecutableAdmitRequest, image_offset) == 32);
static_assert(offsetof(TGPUExecutableAdmitRequest, image_length) == 40);
static_assert(offsetof(TGPUExecutableAdmitRequest, pack_schema_major) == 64);
static_assert(offsetof(TGPUExecutableAdmitRequest, pack_schema_minor) == 68);
static_assert(offsetof(TGPUExecutableAdmitRequest, code_object_version) == 72);
static_assert(offsetof(TGPUExecutableAdmitRequest, entry_id) == 76);
static_assert(offsetof(TGPUExecutableAdmitRequest, pack_id) == 92);
static_assert(offsetof(TGPUExecutableAdmitRequest, reserved0) == 140);
static_assert(offsetof(TGPUExecutableAdmitResponse, pack_id) == 72);
static_assert(offsetof(TGPUExecutableAdmitResponse, entry_id) == 88);
static_assert(offsetof(TGPUExecutableAdmitResponse, target) == 104);
static_assert(offsetof(TGPUExecutableAdmitResponse, device_epoch) == 120);
static_assert(offsetof(TGPUFenceWaitRef, fence_value) == 8);
static_assert(offsetof(TGPUCommandRecordHeader, command_kind) == 0);
static_assert(offsetof(TGPUCommandRecordHeader, record_size) == 4);
static_assert(offsetof(TGPUCommandRecordHeader, sequence) == 8);
static_assert(offsetof(TGPUCommandBinding, offset) == 8);
static_assert(offsetof(TGPUCommandBinding, length) == 16);
static_assert(offsetof(TGPUCommandBinding, access_flags) == 24);
static_assert(offsetof(TGPUCopyCommandRecord, source_buffer) == 16);
static_assert(offsetof(TGPUCopyCommandRecord, source_offset) == 24);
static_assert(offsetof(TGPUCopyCommandRecord, destination_buffer) == 32);
static_assert(offsetof(TGPUCopyCommandRecord, destination_offset) == 40);
static_assert(offsetof(TGPUCopyCommandRecord, length) == 48);
static_assert(offsetof(TGPUCopyCommandRecord, access_flags) == 56);
static_assert(offsetof(TGPUFillCommandRecord, destination_buffer) == 16);
static_assert(offsetof(TGPUFillCommandRecord, destination_offset) == 24);
static_assert(offsetof(TGPUFillCommandRecord, length) == 32);
static_assert(offsetof(TGPUFillCommandRecord, pattern_width) == 40);
static_assert(offsetof(TGPUFillCommandRecord, pattern) == 48);
static_assert(offsetof(TGPUDispatchCommandRecord, executable_handle) == 16);
static_assert(offsetof(TGPUDispatchCommandRecord, entry_id) == 24);
static_assert(offsetof(TGPUDispatchCommandRecord, binding_count) == 44);
static_assert(offsetof(TGPUDispatchCommandRecord, grid_x) == 56);
static_assert(offsetof(TGPUDispatchCommandRecord, dynamic_lds_bytes) == 80);
static_assert(offsetof(TGPUBarrierCommandRecord, visibility_scope) == 16);
static_assert(offsetof(TGPUBarrierCommandRecord, access_scope) == 20);
static_assert(offsetof(TGPUTimestampCommandRecord, label) == 16);
static_assert(offsetof(TGPUTimestampCommandRecord, point) == 48);
static_assert(offsetof(TGPUSubmitRequest, queue_handle) == 24);
static_assert(offsetof(TGPUSubmitRequest, executable_handle) == 32);
static_assert(offsetof(TGPUSubmitRequest, command_buffer_handle) == 40);
static_assert(offsetof(TGPUSubmitRequest, command_bytes) == 56);
static_assert(offsetof(TGPUSubmitRequest, command_count) == 64);
static_assert(offsetof(TGPUSubmitRequest, binding_count) == 72);
static_assert(offsetof(TGPUSubmitRequest, wait_count) == 76);
static_assert(offsetof(TGPUSubmitRequest, signal_value) == 80);
static_assert(offsetof(TGPUSubmitResponse, submission_handle) == 32);
static_assert(offsetof(TGPUSubmitResponse, fence_handle) == 40);
static_assert(offsetof(TGPUSubmitResponse, signal_value) == 48);
static_assert(offsetof(TGPUSubmitResponse, queue_epoch) == 56);
static_assert(offsetof(TGPUSubmitResponse, device_epoch) == 64);
static_assert(offsetof(TGPUFenceWaitRequest, fence_handle) == 24);
static_assert(offsetof(TGPUFenceWaitRequest, fence_value) == 32);
static_assert(offsetof(TGPUFenceWaitRequest, timeout_ns) == 40);
static_assert(offsetof(TGPUFenceWaitResponse, completed_value) == 32);
static_assert(offsetof(TGPUFenceWaitResponse, requested_value) == 40);
static_assert(offsetof(TGPUFenceWaitResponse, terminal_status) == 48);
static_assert(offsetof(TGPUFenceWaitResponse, queue_epoch) == 56);
static_assert(offsetof(TGPUTimestampQueryRequest, submission_handle) == 24);
static_assert(offsetof(TGPUTimestampQueryRequest, fence_handle) == 32);
static_assert(offsetof(TGPUTimestampQueryRequest, point) == 40);
static_assert(offsetof(TGPUTimestampQueryResponse, submission_handle) == 32);
static_assert(offsetof(TGPUTimestampQueryResponse, start_ticks) == 48);
static_assert(offsetof(TGPUTimestampQueryResponse, conversion_denominator) == 80);
static_assert(offsetof(TGPUHealthFaultQueryRequest, scope) == 24);
static_assert(offsetof(TGPUHealthFaultQueryRequest, cursor) == 32);
static_assert(offsetof(TGPUHealthFaultQueryRequest, queue_handle) == 40);
static_assert(offsetof(TGPUHealthFaultQueryResponse, health_state) == 32);
static_assert(offsetof(TGPUHealthFaultQueryResponse, fault_id) == 40);
static_assert(offsetof(TGPUHealthFaultQueryResponse, queue_handle) == 56);
static_assert(offsetof(TGPUHealthFaultQueryResponse, terminal_status) == 80);
static_assert(offsetof(TGPUHealthFaultQueryResponse, text_length) == 84);
static_assert(offsetof(TGPUHealthFaultQueryResponse, device_epoch) == 280);
static_assert(offsetof(TGPUQueueResetRequest, queue_handle) == 24);
static_assert(offsetof(TGPUQueueResetRequest, reset_flags) == 32);
static_assert(offsetof(TGPUQueueResetResponse, queue_handle) == 32);
static_assert(offsetof(TGPUQueueResetResponse, queue_state) == 40);
static_assert(offsetof(TGPUQueueResetResponse, device_epoch) == 48);
static_assert(offsetof(TGPUQueueResetResponse, canceled_submissions) == 56);
static_assert(offsetof(TGPUQueueResetResponse, queue_epoch) == 64);
static_assert(offsetof(TGPUQueueResetResponse, reserved2) == 72);
static_assert(offsetof(TGPUDeviceResetRequest, reset_flags) == 24);
static_assert(offsetof(TGPUDeviceResetRequest, reason_code) == 28);
static_assert(offsetof(TGPUDeviceResetResponse, health_state) == 32);
static_assert(offsetof(TGPUDeviceResetResponse, previous_device_epoch) == 40);
static_assert(offsetof(TGPUDiagnosticMMIOReadRequest, bar_relative_offset) == 24);
static_assert(offsetof(TGPUDiagnosticMMIOReadRequest, width_bytes) == 32);
static_assert(offsetof(TGPUDiagnosticMMIOReadResponse, value) == 32);
static_assert(offsetof(TGPUDiagnosticMMIOWriteRequest, value) == 40);
```

The assertions above are required in both generated/client-facing declarations and the DriverKit declaration boundary; a mismatch is an ABI review failure, not an implementation detail.

The layout table is the selector-to-type and size contract. For v1.0, `min_request`/`min_response` are the exact declaration sizes below; a request's supplied `struct_size` may include only zero trailing bytes up to `TGPU_MAX_STRUCT_BYTES`, while a response always writes exactly the listed response size.

| Selector | Request type / `min_request` | Response type / `min_response` | Offset bases and bounded elements |
|---:|---|---|---|
| `0x00` | `TGPUQueryCapabilitiesRequest` / 24 | `TGPUCapabilitiesResponse` / 144 | Fixed struct; architecture is 16 bytes, `architecture_length <= 15` and remaining bytes are zero. |
| `0x01` | `TGPUBufferAllocateRequest` / 56 | `TGPUBufferAllocateResponse` / 64 | Fixed struct; one buffer; byte size/alignment are checked `uint64_t`. |
| `0x02` | `TGPUBufferImportRequest` / 48 | `TGPUBufferImportResponse` / 64 | Fixed struct plus one checked `structureInputDescriptor` sideband; no pointer field. |
| `0x03` | `TGPUBufferMapRequest` / 64 | `TGPUBufferMapResponse` / 72 | `offset`/`length` are bytes relative to buffer start; range is half-open. |
| `0x04` | `TGPUBufferUnmapRequest` / 40 | `TGPUStatusResponse` / 32 | One mapping handle. |
| `0x05` | `TGPUBufferReleaseRequest` / 40 | `TGPUStatusResponse` / 32 | One buffer handle. |
| `0x06` | `TGPUQueueCreateRequest` / 40 | `TGPUQueueCreateResponse` / 64 | Driver-owned controls; no client storage handle or control address. |
| `0x07` | `TGPUQueueDestroyRequest` / 40 | `TGPUStatusResponse` / 32 | One idle queue. |
| `0x08` | `TGPUExecutableAdmitRequest` / 144 | `TGPUExecutableAdmitResponse` / 128 | Image range is relative to image buffer; fixed 16-byte target/entry/pack ID and 32-byte claimed digest. |
| `0x09` | `TGPUExecutableReleaseRequest` / 40 | `TGPUStatusResponse` / 32 | One executable handle. |
| `0x0a` | `TGPUSubmitRequest` / 1120 | `TGPUSubmitResponse` / 80 | Command range is relative to command buffer; records are 128 bytes; binding table elements are 32 bytes; waits are 16 bytes. |
| `0x0b` | `TGPUFenceWaitRequest` / 56 | `TGPUFenceWaitResponse` / 64 | Fence value and timeout are separate `uint64_t` fields. |
| `0x0c` | `TGPUTimestampQueryRequest` / 48 | `TGPUTimestampQueryResponse` / 96 | One bounded interval; ticks and conversion values are `uint64_t`. |
| `0x0d` | `TGPUHealthFaultQueryRequest` / 72 | `TGPUHealthFaultQueryResponse` / 296 | Scope is client/queue/device; fault text is exactly 192 bytes maximum. |
| `0x0e` | `TGPUQueueResetRequest` / 40 | `TGPUQueueResetResponse` / 80 | One owner queue; old queue/submission/fence tokens are invalidated and a new queue handle is returned. |
| `0x0f` | `TGPUDeviceResetRequest` / 40 | `TGPUDeviceResetResponse` / 56 | One serialized device transition; recovery role only. |
| `0x80` | `TGPUDiagnosticMMIOReadRequest` / 40 | `TGPUDiagnosticMMIOReadResponse` / 40 | Diagnostic role only; BAR-relative offset and width are checked. |
| `0x81` | `TGPUDiagnosticMMIOWriteRequest` / 48 | `TGPUStatusResponse` / 32 | Diagnostic role only; allowlisted write and checked width. |

### Selector namespace

`ExternalMethod` dispatches only the following selectors after validating the request header and before calling an owner helper. The former raw four-selector values are not aliases.

| Selector | Name | Authorization and meaning |
|---:|---|---|
| `0x00` | `TGPU_QUERY_CAPABILITIES` | All entitled roles; read-only device identity, features, limits, timestamp conversion, and epoch. |
| `0x01` | `TGPU_BUFFER_ALLOCATE` | Inference role for its connection; driver-owned buffer. |
| `0x02` | `TGPU_BUFFER_IMPORT` | Inference role for its connection; checked DriverKit descriptor sideband. |
| `0x03` | `TGPU_BUFFER_MAP` | Inference role for its connection; opaque mapping handle only. |
| `0x04` | `TGPU_BUFFER_UNMAP` | Owner of mapping; `BUSY` while referenced. |
| `0x05` | `TGPU_BUFFER_RELEASE` | Owner of buffer; `BUSY` while referenced. |
| `0x06` | `TGPU_QUEUE_CREATE` | Inference role; queue controls are driver-owned. |
| `0x07` | `TGPU_QUEUE_DESTROY` | Queue owner; queue must be idle. |
| `0x08` | `TGPU_EXECUTABLE_ADMIT` | Entitled inference role; driver-owned image policy and recomputed digest required. |
| `0x09` | `TGPU_EXECUTABLE_RELEASE` | Executable owner; no queue/submission references. |
| `0x0a` | `TGPU_SUBMIT` | Queue/executable owner; all records and bindings validated before mutation. |
| `0x0b` | `TGPU_FENCE_WAIT` | Fence owner; bounded poll/wait only. |
| `0x0c` | `TGPU_TIMESTAMP_QUERY` | Submission/fence owner; bounded device-domain data. |
| `0x0d` | `TGPU_HEALTH_FAULT_QUERY` | Inference sees own client/queue detail; recovery/diagnostic role may see bounded device scope. |
| `0x0e` | `TGPU_QUEUE_RESET` | Owner of the queue only; no cross-client reset. |
| `0x0f` | `TGPU_DEVICE_RESET` | Recovery role and exact recovery entitlement only; never normal inference. |
| `0x80` | `TGPU_DIAGNOSTIC_MMIO_READ` | Diagnostic role and exact diagnostic entitlement only; not in normal inference dispatch. |
| `0x81` | `TGPU_DIAGNOSTIC_MMIO_WRITE` | Diagnostic role and exact diagnostic entitlement only; allowlist required. |
| `0xff` | reserved | Always `TGPU_STATUS_UNSUPPORTED`; never a raw-RPC compatibility path. |

### Versioning, structure validation, and offset rules

These rules are part of the ABI and apply before operation-specific validation:

1. `abi_major` must equal `1`; otherwise the response is `TGPU_STATUS_ABI_MISMATCH` and no operation executes. The current implementation accepts only `abi_minor == 0`; a request with `abi_minor > 0` is `TGPU_STATUS_ABI_MISMATCH`.
2. For v1.0, `struct_size` must be at least the selector's `min_request` and at most `TGPU_MAX_STRUCT_BYTES`. All bytes from the declaration size through `struct_size` are unknown trailing bytes and must be zero. A field outside the supplied size is absent and cannot be read. A response is written at exactly the listed `min_response` size.
3. A future minor may append fields only after the v1.0 prefix; it may not insert, reorder, resize, or reinterpret a v1.0 field. A driver supporting minor `N` must accept a minor-0 request using only the v1.0 prefix and must return `abi_minor == 0` for that request. It may accept minor `M <= N` only when every field through the supplied `struct_size` has identical meaning. A required field introduced by minor `M` must be present through its `offsetof + sizeof` or the request is `TGPU_STATUS_INVALID_REQUEST`.
4. Every request header `flags`, every command-header `flags`, every reserved scalar, every reserved array byte, and every unknown bit outside the masks above must be zero. The v1.0 `import_flags`, `map_flags`, `queue_flags`, `query_flags`, and `reset_flags` fields must equal their corresponding `*_FLAGS_V1_0 == 0` constants; v1.0 does not negotiate operation flags.
5. Output capacity below `sizeof(TGPUResponseHeader)` is a DriverKit transport argument failure and writes nothing. Capacity at least 32 but below the selector's response minimum receives only a 32-byte response header with `TGPU_STATUS_INVALID_REQUEST`; a complete response requires the listed minimum capacity. No response writes beyond capacity.
6. Counts are bounded before multiplication. `count * element_size`, `offset + length`, `base + size`, and `binding_table_offset + binding_count * sizeof(TGPUCommandBinding)` use checked arithmetic and reject overflow. `command_count <= 256`, `binding_count <= 64`, `wait_count <= 64`, and `command_bytes <= 65536`; a record's `record_size` must equal 128 for v1.0 and `command_count * 128` must fit in `command_bytes`.
7. Operation byte offsets are relative to the named owned object, never process or GPU addresses. Submit `command_offset` is relative to the command buffer; `binding_table_offset` is relative to that submitted command range. Records start at command-range offset zero, the binding table must be in range and non-overlapping, and all dispatch binding indices must be within the table. All ranges are half-open `[offset, offset + length)`.
8. DriverKit's `kern_return_t` reports only transport/argument failure. The structured status is the semantic result and is populated whenever a response header can be formed.

### Operation semantics and security invariants

**Capabilities (`0x00`).** The response reports `vendor_id == 0x1002`, `device_id == 0x7551`, and `architecture == "gfx1201"` only when the R9700 target is attached. It reports only advertised feature/memory bits, alignments, limits, timestamp frequency, and `device_epoch`; it never reports BAR size, physical segment, GPU VA, register offset, or unrestricted capability.

**Buffers (`0x01`–`0x05`).** Allocation requires nonzero size, power-of-two alignment within capability limits, a supported domain, access subset, and resource flags within `TGPU_RESOURCE_MASK_V1_0`. Import accepts only the checked `structureInputDescriptor` sideband. The driver validates descriptor length, page/range overflow, direction, lifetime, and connection ownership. Mapping checks `offset <= size`, `length > 0`, `length <= size - offset`, alignment, and access subset, then returns only an opaque mapping handle. No public resource flag or mapping can request queue-control, fence-control, MQD/HQD, doorbell, or other hardware-consumed storage. Unmap/release returns `TGPU_STATUS_BUSY` while pinned by a queue, executable, kernarg, command, or in-flight submission. Stale, zero, cross-client, wrong-kind, or double-release handles fail without mutation.

**Queues (`0x06`–`0x07`).** Queue creation has no client control-buffer handle or address. Rings, MQDs, HQDs, doorbells, fence storage, and queue state are allocated and mapped only by the driver and are never client-mappable. If a future implementation needs a producer page, it is at most one bounded page per call, copied under the connection lock, fully validated, and never hardware-consumed after validation; no such page is part of the v1.0 queue ABI. Queue depth/class are capability-bounded. Destroy requires no pending work and never silently cancels it.

**Executable admission (`0x08`–`0x09`).** The inference/recovery caller must be authorized by its connection role; a caller-supplied role bit is not trusted. The driver copies the owned image range to immutable private storage, recomputes SHA-256 over exactly those bytes, and rejects any overflow, writable alias, target mismatch, code-object mismatch, or digest mismatch. The recomputed digest must match the driver-owned immutable R9700 image authorization table; client `pack_id` and `claimed_image_digest` are retained only as audit metadata/consistency checks. The driver validates code-object structure, symbols, descriptors, relocations, kernarg offsets/alignment, wave mode, SGPR/VGPR/LDS/private-memory limits, expected ISA/resource categories, and entry identity. It does not parse P3 Kernel Pack records or trust a P3 schema as an authorization decision. The executable and all future translations belong to the connection's private VM context; absolute addresses, unbound addresses, raw pointers, and client-provided relocations reject. Release is refused while referenced.

**Submit (`0x0a`).** The driver first snapshots the complete command range and binding table into private storage. It then validates every fixed 128-byte record, sequence, handle namespace, range, alignment, access, executable/entry identity, kernarg binding, grid/block limits, dynamic LDS, and queue state. The only record kinds are `COPY`, `FILL`, `DISPATCH`, `BARRIER`, and `TIMESTAMP`. `COPY` and `FILL` use owned ranges; `DISPATCH` names an admitted executable entry and a binding-table range, including exactly one read-only kernarg binding, and the driver builds kernargs/relocations from `{buffer_handle, offset, length, access}` bindings. `BARRIER` is an ordering declaration; `TIMESTAMP` is a bounded label/point. A PM4 packet, doorbell, MQD, HQD, GPU VA, physical address, absolute address, or unbound memory operand is invalid. A failure rejects the entire submission before queue mutation, signal allocation, or reference-count changes. On success the driver emits hardware packets privately and returns opaque submission/fence handles and the assigned nonzero monotonic signal value.

**Fences/timestamps (`0x0b`–`0x0c`).** Fence values are per queue, start at zero, increase strictly, and never wrap; a submit signal must be nonzero and greater than the queue's last signal. `timeout_ns == 0` **polls the requested `fence_value`** and never blocks; only `timeout_ns` is bounded by `TGPU_MAX_WAIT_NS`. `fence_value == 0` is explicitly valid as the initial-state query: it returns immediately with completed value zero when the fence is healthy, or its terminal failure status, and is never a submission signal. A nonzero target with zero timeout returns the current completed/terminal state without waiting. A nonzero timeout waits up to the requested duration and returns `TGPU_STATUS_TIMEOUT` without queue mutation if the target has not completed. Fence failure is explicit, not fabricated completion. Timestamp data contains only ticks, frequency/conversion, and associated opaque identities.

**Health/fault (`0x0d`).** Fault output uses the numeric health/fault/stage enums above, opaque queue/submission/executable handles that are zero when unknown, terminal status, `text_length <= 192`, and redacted UTF-8 text. Bytes after `text_length` are zero. Inference scope is limited to the requesting connection and its queues. Device scope is available only to recovery/diagnostic roles and still omits raw registers, BAR contents, physical addresses, prompt/token data, and unbounded logs.

**Reset (`0x0e`–`0x0f`).** Queue reset is owner-only. It serializes on the queue, cancels pending work, marks pending fences `TGPU_STATUS_CANCELED`, records a queue-scoped bounded fault, invalidates the old queue, submission, and fence tokens, increments the queue epoch, reinitializes driver-owned controls, and returns a new queue handle in `TGPUQueueResetResponse`; it never retries or replays accepted work. Device reset is accepted only from `TinyGPURecoveryUserClient` with role `TGPU_CLIENT_RECOVERY` and entitlement `org.tinygrad.tinygpu.recovery`. It requires the global recovery lock, state `DEGRADED`, `FAULTED`, or `RESET_REQUIRED`, no concurrent initialization/reset, and at least `TGPU_MIN_DEVICE_RESET_INTERVAL_NS` since the previous accepted attempt. `READY` reset, a normal inference caller, a diagnostic caller, or a missing exact entitlement returns `TGPU_STATUS_PERMISSION_DENIED`/`TGPU_STATUS_BUSY` before mutation. An accepted reset transitions through `RESETTING`, invalidates affected handles and all in-flight submissions, increments `device_epoch` without wrap, and returns `READY` or `UNAVAILABLE`. A new epoch requires capability re-query and resource recreation.

**Diagnostics (`0x80`–`0x81`).** These selectors are absent from the normal inference dispatch table. They are implemented only by `TinyGPUDiagnosticUserClient` for role `TGPU_CLIENT_DIAGNOSTIC` with entitlement `org.tinygrad.tinygpu.diagnostic`, a separately authenticated owner policy, capability bit, checked BAR-relative offset/width (`1`, `2`, `4`, or `8` bytes as explicitly allowlisted), bounded output, and redacted evidence labels. They never return a mapping and never feed producer correctness. The legacy socket/proxy cannot satisfy this boundary.

## Handle and lifetime contract

The public typedefs are all `uint64_t`, but a nonzero token is valid only when the per-connection serialized handle table resolves all of these mandatory fields:

* `connection_epoch` — nonzero connection instance epoch, invalidated on close and device/queue reset as specified;
* `slot` — bounded table slot;
* `generation` — nonzero slot generation, incremented before every slot reuse;
* `kind` — exact `TGPUHandleKind`, checked before any object access.

The encoded token is opaque to the client, but the driver must bind it to `(connection_epoch, slot, generation, kind)` and must not accept a token from another connection. Generation never wraps and is never reissued. If the next generation or epoch cannot be represented, minting fails closed with `TGPU_STATUS_RESOURCE_EXHAUSTED`; the slot is not reused. Close, queue reset, and device reset invalidate the relevant epoch/tokens before releasing owner objects. Zero is always invalid. These rules apply equally to buffer, mapping, queue, executable, fence, and submission handles.

Close is idempotent and ordered: mark the connection closing and reject new calls; task-set-4 queue/submission/fence hooks cancel and retire hardware work; task-set-4 executable hooks release executable references; task-set-3 buffer/import/mapping hooks unpin and release descriptors and buffers; invalidate and clear the token table; then release provider state. Every resource task owns an idempotent bounded cleanup hook for the objects it introduces. Task set 5 integrates this ordering with reset/recovery; it does not defer buffer or queue cleanup implementation to a later task. A later connection receives a new empty namespace and epoch and cannot inherit stale state.

## Current TinyGPU source versus the required boundary

| Current source/symbol | Current exposure/risk | Frozen required mapping and owner |
|---|---|---|
| `TinyGPUDriver.iig`, `TinyGPUCreateDMAResp` | Existing declarations expose `IOBufferMemoryDescriptor*` and `IODMACommand*` object pointers. | Keep such objects DEXT-internal only. Public allocation/import/map/release use the fixed opaque handle structures above. Task set 1 owns the declarations; later tasks cannot change them. |
| `TinyGPUDriver.iig`, `MapBar`, `CreateDMA`, `SetupDMA`, `CfgRead`, `CfgWrite`, `ResetDevice`, `GetPCI` | Owner helpers include BAR/config/DMA/reset authority. | Remain private owner helpers. No normal request reaches them with raw addresses or client-controlled device state. |
| `TinyGPUDriverUserClient.iig`, `TinyGPURPC::{ReadCfg,WriteCfg,Reset,PrepareDMA}` | Four unversioned raw selectors have unchecked scalar/descriptor behavior. | Replace their public meanings with selectors `0x00`–`0x0f` and `0x80`–`0x81`; no compatibility alias preserves raw RPC meanings. |
| `TinyGPUDriverUserClient.cpp`, `ExternalMethod` and `CopyClientMemoryForType` | Existing dispatch/memory-copy logic can map BARs or interpret a type as DMA size. | New dispatch validates the canonical header and calls only structured owner helpers. Normal methods never map a BAR, return a physical segment, or turn a type into an allocation size. |
| `TinyGPUDriverUserClient.cpp`, DMA table and `ensureDMACap` | Existing per-client array is manually grown, not thread-safe, and stores raw pointers. | Use bounded serialized type-tagged handle tables with mandatory generation/epoch checks and per-resource cleanup hooks. No unbounded client-controlled growth. |
| `TinyGPUDriverUserClient.cpp`, `Stop_Impl`/`free` | Existing teardown has no complete queue/executable/fence/buffer ordering. | Task set 3 owns idempotent buffer/import/mapping cleanup hooks; task set 4 owns queue/executable/submission/fence hooks; task set 5 integrates close/reset ordering and provider release. |
| `Shared/server.c`, packed `request_t`/`response_t` and `CMD_*` | Unauthenticated local socket path includes BAR mapping, config writes, reset, MMIO, and overflow-prone range checks. | Quarantine the entire source and protocol as historical local diagnostic material. It is excluded from every P1 product/build/conformance/validation/Release target, is not linked or launched by any P1 command, is not an inference acceptance path, and is removed or separately archived before Release. |
| `server.c`, `map_bar`/`map_sysmem_fd`/`validate_bar` | Returns process mappings and physical segments and accepts unauthenticated peers. | No P1 evidence or client may use these sinks. Future diagnostics use only the separate entitled diagnostic user client and checked BAR-relative offsets; physical values never become producer evidence. |
| `TinyGPUDriver.cpp`, `Start_Impl` and `ResetDevice` | Existing lifecycle/reset helpers have no ABI epoch or role policy. | Task set 2 owns cold lifecycle/capabilities; task set 5 integrates recovery and epoch invalidation around the private helpers. Neither changes selectors or resource semantics. |
| `TinyGPUDriverExtension/Info.plist` | Existing personality matches display-class PCI devices generally and has one generic user-client class. | Release personality must match only AMD `1002:7551` and expose separate inference/recovery/diagnostic user-client classes with exact role authorization. A class-wide or vendor-wide Release personality is forbidden. |
### Required source/package cutover before any install

The current source still contains a buildable legacy path (`TinyGPUDriverExtension.xcodeproj/project.pbxproj` includes `server.c` in the app Sources phase) and the current app CLI still exposes `server <path>`. The quarantine is therefore an owned source cutover, not a prose exception. **P1 task set 2 owns this cutover in the in-repository `tinygpu/` source tree before any P1 install or cold validation:**

| Exact path/symbol | Required cutover and acceptance check |
|---|---|
| `TinyGPUDriverExtension.xcodeproj/project.pbxproj`: PBXBuildFile `0A5C11DD2F18E466006DBBCA`, file reference `0A5C11DC2F18E461006DBBCA`, and app `PBXSourcesBuildPhase` entry `server.c in Sources` | Remove the file reference/build-file/source-phase entry. A source/package review must find no `server.c` in any target Sources/Compile Sources list. The retained `Shared/server.c` file is historical quarantine material only and is not compiled or linked. |
| `Shared/TinyGPUCLIRunner.swift`: `case "server"`, `run_server`, and `server <path>` usage | Remove the command, call, and usage line. `TinyGPU status`, `install`, and `uninstall` remain the only app CLI commands; no app command accepts a socket path. |
| `Shared/TinyGPUApp.swift`: user-visible `Remote PCI Device Server` text and any server launch wording | Replace with a DriverKit controller/status description. The app must not advertise or launch a raw server. |
| `TinyGPUDriverExtension/Info.plist` | Release personality uses only `IOPCIPrimaryMatch = 0x75511002&0xFFFFFFFF` (AMD vendor `0x1002`, device `0x7551`), not `IOPCIClassMatch` or a vendor-wide match. Its user-client properties name the separately authorized inference, recovery, and diagnostic classes. |
| `TinyGPUDriverExtension/TinyGPUDriver.entitlements` and `TinyGPUDriver.Release.entitlements` | Both production-selected entitlement files use only `0x75511002&0xFFFFFFFF`; neither contains wildcard, vendor-wide AMD, or NVIDIA access. |
| `TinyGPUDriverExtension/TinyGPUDriver.NoSIP.entitlements` | Retain wildcard/allow-any access only for the explicitly local NoSIP configuration. It is never selected by Release and is never accepted as product evidence. |
| `TinyGPUDriverExtension/TinyGPUDriver.NV.Release.entitlements` and `build_and_sign_nv.sh` | Retire/remove from all Release/product paths; NVIDIA is not a P1 product. The only Release signing path is the R9700 AMD path. |

The cutover acceptance is fail-closed: any `server.c` target reference, `server` CLI branch/usage, raw socket launch, broad Info personality, wildcard/vendor-wide Release entitlement, or NVIDIA Release path blocks install and P1 validation. This is separate from the allowed NoSIP local wildcard and from external Apple distribution credentials.

## Entitlement, role, and Release scope

The role is selected by the user-client class and signed code requirement, never by a request field:

| Role | User-client class | Exact product entitlement | Allowed authority |
|---|---|---|---|
| Inference | `TinyGPUInferenceUserClient` | `org.tinygrad.tinygpu.inference` | Own buffers, mappings, queues, executables, submissions, fences, timestamps, and client/queue fault detail; own queue reset. |
| Recovery | `TinyGPURecoveryUserClient` | `org.tinygrad.tinygpu.recovery` | All inference operations as separately authorized plus serialized device reset and bounded device-scope recovery detail. |
| Diagnostic | `TinyGPUDiagnosticUserClient` | `org.tinygrad.tinygpu.diagnostic` | Bounded device fault detail and allowlisted diagnostic MMIO only; no inference acceptance and no device reset. |

`org.tinygrad.tinygpu.driver2` remains the DriverKit service/bundle identity and the app's user-client access entry. The role entitlements above are exact additional product authorization requirements; `com.apple.developer.driverkit.allow-any-userclient-access` is never a Release substitute. A normal inference connection cannot request recovery or diagnostic role by setting a flag.

The Release transport entitlement and Release personality both match exactly the PCI tuple `vendor_id=0x1002`, `device_id=0x7551` (`IOPCIPrimaryMatch` combined value `0x75511002&0xFFFFFFFF`, with no vendor-only mask). Release contains no `0xFFFFFFFF&0x00000000`, no `0x00001002&0x0000FFFF`, no `0x000010de&0x0000FFFF`, no wildcard class match, and no unrelated device match. The existing wildcard `TinyGPUDriver.NoSIP.entitlements` is explicitly local development only. Any retained wildcard default entitlement is also local-only and must not be selected by a Release configuration. A Release package check must inspect the signed entitlements and Info personality and fail closed on wildcard, NVIDIA, vendor-wide AMD, or display-class-only matching.

Local development is separate: `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/install_nosip.sh` may use SIP-disabled ad-hoc signing and NoSIP wildcard access solely for local development. It cannot launch the quarantined legacy proxy or be used as Release evidence. Apple PCI distribution entitlement, matching profiles, Developer ID/notarization credentials, and the approved external signing invocation remain a **promotion-only** blocker; they do not authorize broad device matching and do not block source implementation once the local SDK gate is satisfied.

## SDK gate and implementation status

The supervisor's refreshed prerequisite observation on 2026-08-26 is recorded exactly:

* active developer directory: `/Applications/Xcode.app/Contents/Developer`;
* `xcodebuild -version`: Xcode 26.6, build `17F113`;
* `xcrun --sdk driverkit --show-sdk-version`: `25.5`;
* selected SDK path: `/Applications/Xcode.app/Contents/Developer/Platforms/DriverKit.platform/Developer/SDKs/DriverKit25.5.sdk`.

The SDK/source gate and focused security-review gate are clear for task sets 2–4. The external distribution-entitlement/credential gate remains independent and promotion-only.

## Later task ownership matrix

Later tasks may implement this contract but may not renumber selectors, change structure prefixes or layout values, add address fields, weaken role/entitlement checks, parse P3 Kernel Packs in DriverKit, or weaken handle/range/permission rules without reopening P1 review.

| Owner/task | Exact files/symbols | Owns | Must not edit/claim |
|---|---|---|---|
| P1 task set 1 / ABI owner | TinyGPU declarations through the separately authorized source boundary; this report and the P1 ledger row | Canonical declarations, selector/status/role/handle values, layout assertions, version/bounds/security record, command contract, validation-client CLI freeze | TinyGPU implementation in this batch; shared validation ledger; service/model lifecycle |
| P1 task set 2 | Later in-repository source tree `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu` (`feature/r9700-products-wave-a`), `TinyGPUDriver.cpp` `init`/`Start_Impl`/cold-stage helpers, packaging files named in the source cutover table, and `Conformance/tgpu_conformance_client.cpp`; P1 cold-lifecycle report | Cold lifecycle, firmware provenance, capability/health population, removal of `server.c` from all targets/CLI paths, exact R9700 Release Info/entitlement scope, common conformance-client transport/entry point, and the `cold-lifecycle` subcommand | `.iig` layout/selector changes; buffer/resource tables; queue/submit/reset policy; service/model files; any legacy proxy path |
| P1 task set 3 | Same in-repository `tinygpu/` source tree/branch; TinyGPU buffer/VA owner helpers and `TinyGPUDriverUserClient.cpp` buffer selectors/resource tables; `Conformance/tgpu_conformance_client.cpp` client-death extension; named buffer tests | Buffer allocate/import/map/unmap/release, opaque mapping handles, per-client VM/range/alignment/permission checks, idempotent buffer/import/mapping client-death cleanup hooks, and the `client-death` subcommand | Cold stages/package cutover; queue/executable/submit/fence implementation; service/model; raw BAR/MMIO; deferring buffer cleanup to task 5 |
| P1 task set 4 | Same in-repository `tinygpu/` source tree/branch; `TinyGPUDriverUserClient.cpp` queue/executable/submit/fence/timestamp/health cases and narrow owner helpers; sequential `Conformance/tgpu_conformance_client.cpp` extensions; named queue/executable tests | Driver-owned queue controls, executable admission and closed validation, binding snapshots/driver-built kernargs, submission, monotonic fences/timestamps, bounded fault attribution, idempotent queue/executable/submission/fence cleanup hooks, and `malformed-submit`, `queue-reset`, `fault-query`, and `g0-binding` subcommands | Buffer ABI/table changes; reset/recovery integration; portable HAL; raw PM4 from clients; deferring execution-resource cleanup to task 5 |
| P1 task set 5 | Same in-repository `tinygpu/` source tree/branch; `ResetDevice`/recovery helpers, `TGPU_QUEUE_RESET`/`TGPU_DEVICE_RESET`, `Stop_Impl`/`free` orchestration, and sequential `Conformance/tgpu_conformance_client.cpp` extension | Integrate task-3 and task-4 cleanup hooks in close/reset order; queue/device reset, epoch invalidation, recovery/unavailable state, client-death sequencing, and the `device-recovery` subcommand | Selector/struct changes; replacing per-resource hooks; hidden retries; multi-client scheduler; P2 HAL reset policy; legacy proxy |
| P1 task set 6 | `logs/p1-tinygpu-owner/`, promotion report, and task/ledger evidence | Fresh cold → BO/VA → TGPU queue → exact G0 → sustained inference → fault/reset evidence; G0 consumption | Regenerating G0; changing service/model; changing P1 ABI or launching legacy proxy |
| P2 consumers | `native_r9700/hal.h`, `hal.cpp`, `hal_amdev.h`, `hal_amdev.cpp` and P2 tests | Portable Device/Buffer/Executable/Queue/Fence over TGPU with no AMD register leakage | TinyGPU source or TGPU selector/layout changes |
| F1/Q1/F2/P3 owners | Their existing phase-owned source and artifact paths | Service/model, Qwen, WMMA/G0, and generic Kernel Pack records respectively | P1 selectors, handle ABI, or DriverKit parsing of P3 records |
| Supervisor-selected integration owner | `native_r9700/kernel_assets.cpp`, `native_r9700/kernel_catalog.cpp`, generated catalogs | One serialized F2/P3 integration boundary consumed as opaque admission metadata | Concurrent edits by P1/F2/P3 |

The single `TGPUConformanceClient` target is intentionally sequential: task set 2 creates the common transport/entry point and `cold-lifecycle` subcommand while performing the source/package cutover; task set 3 adds `client-death`; task set 4 adds `malformed-submit`, `queue-reset`, `fault-query`, and `g0-binding`; task set 5 adds `device-recovery`. Later tasks may extend this source only after the predecessor's gate; no client-source edits are concurrent. Thus task set 2 can build and run the cold-lifecycle command immediately after its own implementation, without waiting for task set 5.
## Fixed future conformance client and recorded validation commands

These commands are a recorded future contract, not executed by this task and not an assertion that the client or binary exists today. The single implementation target is:

* source: `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/Conformance/tgpu_conformance_client.cpp`;
* Xcode target: `TGPUConformanceClient`;
* binary: `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client`.

Task set 2 creates the common client target and `cold-lifecycle`; task sets 3–5 extend that same target with their assigned subcommands after their respective gates. A different executable, raw socket command, or placeholder shell fragment is not a substitute. Every invocation writes a bounded log containing `abi_major`, `abi_minor`, `selector`, `status`, `failure_stage`, `device_epoch`, and `exit_status`.

### SDK/build/install preflight and local install

```sh
xcode-select -p
xcrun --sdk driverkit --show-sdk-version
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -target TGPUConformanceClient -configuration Debug \
  CONFIGURATION_BUILD_DIR=${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug
./install_nosip.sh
```

The currently verified preflight result is blocked: active developer directory is CommandLineTools, DriverKit SDK version is unavailable, and full Xcode is absent. After full Xcode is installed/selected, the first command must print the selected developer directory and the second must print an exact DriverKit SDK version; that version is recorded before any source build. The local installer may use only the NoSIP development entitlement. No command in this section launches or links `Shared/server.c`.

### P1 cold lifecycle

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client \
  cold-lifecycle --service org.tinygrad.tinygpu.driver2 \
  --pci-id 1002:7551 --architecture gfx1201 \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/cold-lifecycle.log
```

Required observations are a fresh DEXT cold attach, ordered lifecycle stages, `TGPU_QUERY_CAPABILITIES` with `abi_major: 1`, `abi_minor: 0`, exact PCI/architecture identity, and a ready health record. The command uses the DriverKit user client directly and never the legacy proxy. A missing DriverKit SDK or DEXT is a blocked prerequisite, not a pass or a reason to fall back to a raw socket.

### P1 malformed submission, stale/client-death, queue reset, and bounded fault query

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client \
  malformed-submit --service org.tinygrad.tinygpu.driver2 \
  --cases wrong-record-size,absolute-address,unbound-binding,stale-handle \
  --expect-status TGPU_STATUS_SUBMISSION_REJECTED \
  --expect-no-queue-mutation \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/malformed-submit.log

${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client \
  client-death --service org.tinygrad.tinygpu.driver2 \
  --close-with-live-resources --reopen --replay-handles \
  --expect-status TGPU_STATUS_INVALID_HANDLE \
  --expect-empty-new-namespace \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/client-death.log

${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client \
  queue-reset --service org.tinygrad.tinygpu.driver2 \
  --owner-only --expect-pending-fence-status TGPU_STATUS_CANCELED \
  --expect-no-replay \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/queue-reset.log

${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client \
  fault-query --service org.tinygrad.tinygpu.driver2 \
  --scope client --max-text-bytes 192 --expect-redacted \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/fault-query.log
```

The malformed cases must return status 12 before queue mutation or signal allocation; stale/cross-client handles must return status 5; queue reset must be accepted only for the queue owner and must explicitly fail pending fences; fault text must be at most 192 bytes and must not contain raw registers, addresses, prompts, or tokens. No invocation uses a socket/proxy command.

### P1 device recovery

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client \
  device-recovery --service org.tinygrad.tinygpu.driver2 \
  --recovery-service org.tinygrad.tinygpu.recovery \
  --preflight-normal-reset-denied --fault-source physical \
  --expect-device-epoch-increment --expect-stale-handle-rejection \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/device-recovery.log
```

The fixed client first proves that a normal inference connection receives `TGPU_STATUS_PERMISSION_DENIED` for device reset, then uses the exact recovery role and entitlement. If physical fault injection is unavailable, the command must record `physical_fault_injection: unavailable` and `status: blocked`; it must not claim recovery success and must not substitute the old kernel-proof/raw-proxy control. Once hardware injection exists, the command requires a bounded fault, serialized recovery, incremented `device_epoch`, stale-handle rejection, and a clean new-client capability query. The fixed CLI remains the task-5 implementation target even while the physical injector is unavailable.

### P1 exact G0 binding

```sh
${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/tinygpu/build/Debug/tgpu-conformance-client \
  g0-binding --service org.tinygrad.tinygpu.driver2 \
  --g0-report ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/.superpowers/swarm/reports/g0-wmma-conformance.md \
  --require-status-field g0_status=pass \
  --require-record-id-field g0_record_id \
  --require-image-sha256-field g0_image_sha256 \
  --require-target-field g0_target \
  --require-entry-field g0_entry \
  --require-pci-id 1002:7551 --require-architecture gfx1201 \
  --expect-recomputed-digest-match --expect-no-fallback \
  --log ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a/logs/p1-tinygpu-owner/g0-binding.log
```

The client reads exactly the accepted report fields `g0_record_id`, `g0_image_sha256`, `g0_target`, and `g0_entry`, passes only opaque audit metadata plus image bytes to the TGPU client, and requires the executable-admission response's recomputed digest/target/entry to match those values. The DriverKit implementation does not parse a P3 Kernel Pack. Missing report, non-pass status, digest/target/entry mismatch, target other than `1002:7551`/`gfx1201`, or fallback is fail-closed. No command regenerates G0.

## Unresolved blockers

* **G0 consumption:** task set 6 and P1 promotion require the still-blocked exact F2/G0 WMMA record; task sets 2–5 may proceed independently.
* **Physical fault injection:** the fixed `device-recovery` CLI exists as the task-5 target; physical injection may remain unavailable and must report blocked rather than be replaced by a legacy control.
* **Distribution promotion:** Apple PCI distribution entitlement, profiles, Developer ID/notarization credentials, and approved external signing invocation remain promotion-only inputs. They are separate from the repository-controlled R9700-only Release scope and do not block task sets 2–5.

## Review corrections mapping

The following table maps every finding from `agent://P1SecurityReview` to the frozen correction. The mapping is contract evidence, not a claim that the focused re-review has already accepted the changes.

| Finding | Severity | Correction now frozen |
|---|---|---|
| `P1-SEC-001` legacy socket proxy bypass | Critical | `Shared/server.c` and its raw protocol are historical quarantine only; no P1 product/build/conformance/validation/Release target links or launches it. Task set 2 owns the pre-install project/CLI source cutover and must remove every target/CLI reference. Normal access is the entitled DriverKit user client; future diagnostics use a separate role/class/entitlement with checked offsets. |
| `P1-EXEC-001` executable/resource binding | Important | Admission hashes an immutable private image copy and checks a driver-owned R9700 digest policy plus closed code-object/resource validation. P3 packs are not parsed in DriverKit. Submit snapshots bounded `{buffer_handle, offset, length, access}` bindings, builds kernargs/relocations, enforces per-client VM isolation, and rejects absolute/unbound addresses. |
| `P1-QUEUE-001` mutable queue controls | Important | Queue/MQD/HQD/doorbell/fence controls are driver-owned and non-client-mappable. Client command/producer data is copied and validated before hardware consumption; a mutable producer page is not part of v1.0 and any future page is bounded and copied. |
| `P1-HANDLE-001` optional generation | Important | Every nonzero typed handle resolves mandatory connection epoch, slot, generation, and kind. Generation/epoch never wrap or reissue; exhaustion fails closed; close/reset invalidates tokens. |
| `P1-AUTH-001` reset/fault authorization | Important | Roles/classes/entitlements are exact. Queue reset is owner-only. Device reset is recovery-role-only with `org.tinygrad.tinygpu.recovery`, serialized degraded/faulted preconditions, one-second rate limit, epoch invalidation, and bounded redacted output; inference clients cannot invoke it. |
| `P1-ABI-001` prose-only operation ABI | Important | Every selector now has canonical fixed-width request/response declarations, command/wait/binding elements, numeric enums/masks, reserved fields, sizes, offsets, alignment, offset bases, static assertions, max counts, and v1.0 append-only minor rules. |
| `P1-LIFE-001` cleanup assigned too late | Important | Task set 3 owns idempotent buffer/import/mapping client-death hooks; task set 4 owns queue/executable/submission/fence hooks; task set 5 only integrates their ordered close/reset calls and recovery. |
| `P1-VAL-001` wrong checkout/placeholders | Important | Commands point to the in-repository `tinygpu/` source tree and one fixed `TGPUConformanceClient` source/binary whose ownership is sequential: task 2 common/cold, task 3 client-death, task 4 malformed/queue/fault/G0, task 5 recovery. Exact CLIs are recorded; no command launches the legacy proxy. Physical injection is explicitly blocked rather than replaced. |
| `P1-SIGN-001` broad Release PCI scope | Important | Task set 2 owns the pre-install package cutover; Release transport/personality is exactly AMD `1002:7551` (`0x75511002&0xFFFFFFFF`); wildcard, vendor-wide, NVIDIA, and class-wide matches are forbidden in Release. Wildcard/allow-any access is local NoSIP only. |
| `P1-ABI-002` fence polling contradiction | Minor | `timeout_ns == 0` polls the requested `fence_value`; only timeout is capped. `fence_value == 0` is explicitly an immediate initial-state query and never a signal value. |
| `P1-SDK-001` missing DriverKit SDK datum | Minor | Cleared 2026-08-26: Xcode 26.6 build `17F113`, DriverKit SDK `25.5`, and the exact selected SDK path are recorded before source/build work resumes. |

Task set 1 is therefore **Done**. Task sets 2–4 may proceed with the selected Xcode/DriverKit toolchain; task set 6 and phase promotion remain blocked on G0, with distribution credentials tracked separately.
