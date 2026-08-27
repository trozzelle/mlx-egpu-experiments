#ifndef TGPU_ABI_H
#define TGPU_ABI_H

#include <stddef.h>
#include <stdint.h>

// TGPU v1.0 is a fixed-width, little-endian, naturally aligned ABI.  The
// DriverKit user-client and the host conformance client include this one
// declaration boundary; no public structure contains an address or pointer.
#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
static constexpr uint32_t TGPU_ABI_MAJOR = 1;
static constexpr uint32_t TGPU_ABI_MINOR = 0;
static constexpr uint32_t TGPU_MAX_STRUCT_BYTES = 4096;
static constexpr uint32_t TGPU_MAX_COMMAND_RECORDS = 256;
static constexpr uint32_t TGPU_MAX_COMMAND_RECORD_BYTES = 128;
static constexpr uint32_t TGPU_MAX_COMMAND_BYTES = 65536;
static constexpr uint32_t TGPU_MAX_BINDINGS = 64;
static constexpr uint32_t TGPU_MAX_WAIT_FENCES = 64;
static constexpr uint32_t TGPU_MAX_FAULT_TEXT_BYTES = 192;
static constexpr uint32_t TGPU_MAX_ENTRY_ID_BYTES = 16;
static constexpr uint32_t TGPU_MAX_TARGET_BYTES = 16;
static constexpr uint32_t TGPU_MAX_TIMESTAMP_LABEL_BYTES = 32;
static constexpr uint64_t TGPU_MAX_WAIT_NS = 60000000000ULL;
static constexpr uint64_t TGPU_MIN_DEVICE_RESET_INTERVAL_NS = 1000000000ULL;
#else
#define TGPU_ABI_MAJOR 1U
#define TGPU_ABI_MINOR 0U
#define TGPU_MAX_STRUCT_BYTES 4096U
#define TGPU_MAX_COMMAND_RECORDS 256U
#define TGPU_MAX_COMMAND_RECORD_BYTES 128U
#define TGPU_MAX_COMMAND_BYTES 65536U
#define TGPU_MAX_BINDINGS 64U
#define TGPU_MAX_WAIT_FENCES 64U
#define TGPU_MAX_FAULT_TEXT_BYTES 192U
#define TGPU_MAX_ENTRY_ID_BYTES 16U
#define TGPU_MAX_TARGET_BYTES 16U
#define TGPU_MAX_TIMESTAMP_LABEL_BYTES 32U
#define TGPU_MAX_WAIT_NS 60000000000ULL
#define TGPU_MIN_DEVICE_RESET_INTERVAL_NS 1000000000ULL
#endif

#ifdef __cplusplus
static constexpr uint32_t TGPU_REQUEST_FLAGS_V1_0 = 0;
static constexpr uint32_t TGPU_RECORD_FLAGS_V1_0 = 0;
static constexpr uint32_t TGPU_IMPORT_FLAGS_V1_0 = 0;
static constexpr uint32_t TGPU_MAP_FLAGS_V1_0 = 0;
static constexpr uint32_t TGPU_QUEUE_FLAGS_V1_0 = 0;
static constexpr uint32_t TGPU_QUERY_FLAGS_V1_0 = 0;
static constexpr uint32_t TGPU_RESET_FLAGS_V1_0 = 0;
#else
#define TGPU_REQUEST_FLAGS_V1_0 0U
#define TGPU_RECORD_FLAGS_V1_0 0U
#define TGPU_IMPORT_FLAGS_V1_0 0U
#define TGPU_MAP_FLAGS_V1_0 0U
#define TGPU_QUEUE_FLAGS_V1_0 0U
#define TGPU_QUERY_FLAGS_V1_0 0U
#define TGPU_RESET_FLAGS_V1_0 0U
#endif

typedef enum TGPUStatus : uint32_t {
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
} TGPUStatus;

typedef enum TGPUFailureStage : uint32_t {
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
} TGPUFailureStage;

typedef enum TGPUClientRole : uint32_t {
  TGPU_CLIENT_INFERENCE = 1,
  TGPU_CLIENT_RECOVERY = 2,
  TGPU_CLIENT_DIAGNOSTIC = 3,
} TGPUClientRole;

typedef enum TGPUHandleKind : uint32_t {
  TGPU_HANDLE_BUFFER = 1,
  TGPU_HANDLE_MAPPING = 2,
  TGPU_HANDLE_QUEUE = 3,
  TGPU_HANDLE_EXECUTABLE = 4,
  TGPU_HANDLE_FENCE = 5,
  TGPU_HANDLE_SUBMISSION = 6,
} TGPUHandleKind;

typedef enum TGPUHealthState : uint32_t {
  TGPU_HEALTH_DISCONNECTED = 0,
  TGPU_HEALTH_INITIALIZING = 1,
  TGPU_HEALTH_READY = 2,
  TGPU_HEALTH_DEGRADED = 3,
  TGPU_HEALTH_FAULTED = 4,
  TGPU_HEALTH_RESETTING = 5,
  TGPU_HEALTH_UNAVAILABLE = 6,
} TGPUHealthState;

typedef enum TGPUFaultKind : uint32_t {
  TGPU_FAULT_NONE = 0,
  TGPU_FAULT_TIMEOUT = 1,
  TGPU_FAULT_PAGE_FAULT = 2,
  TGPU_FAULT_ILLEGAL_COMMAND = 3,
  TGPU_FAULT_EXECUTABLE_REJECTED = 4,
  TGPU_FAULT_QUEUE_HANG = 5,
  TGPU_FAULT_DEVICE_FAULT = 6,
  TGPU_FAULT_RESET_REQUIRED = 7,
} TGPUFaultKind;

typedef enum TGPUResetReason : uint32_t {
  TGPU_RESET_REASON_FAULT = 1,
  TGPU_RESET_REASON_TIMEOUT = 2,
  TGPU_RESET_REASON_ADMIN = 3,
} TGPUResetReason;

typedef enum TGPUExecutionClass : uint32_t {
  TGPU_EXECUTION_SDMA = 1,
  TGPU_EXECUTION_COMPUTE = 2,
} TGPUExecutionClass;

typedef enum TGPUCommandKind : uint16_t {
  TGPU_COMMAND_COPY = 1,
  TGPU_COMMAND_FILL = 2,
  TGPU_COMMAND_DISPATCH = 3,
  TGPU_COMMAND_BARRIER = 4,
  TGPU_COMMAND_TIMESTAMP = 5,
} TGPUCommandKind;

typedef enum TGPUSelector : uint32_t {
  TGPU_QUERY_CAPABILITIES = 0x00,
  TGPU_BUFFER_ALLOCATE = 0x01,
  TGPU_BUFFER_IMPORT = 0x02,
  TGPU_BUFFER_MAP = 0x03,
  TGPU_BUFFER_UNMAP = 0x04,
  TGPU_BUFFER_RELEASE = 0x05,
  TGPU_QUEUE_CREATE = 0x06,
  TGPU_QUEUE_DESTROY = 0x07,
  TGPU_EXECUTABLE_ADMIT = 0x08,
  TGPU_EXECUTABLE_RELEASE = 0x09,
  TGPU_SUBMIT = 0x0a,
  TGPU_FENCE_WAIT = 0x0b,
  TGPU_TIMESTAMP_QUERY = 0x0c,
  TGPU_HEALTH_FAULT_QUERY = 0x0d,
  TGPU_QUEUE_RESET = 0x0e,
  TGPU_DEVICE_RESET = 0x0f,
  TGPU_DIAGNOSTIC_MMIO_READ = 0x80,
  TGPU_DIAGNOSTIC_MMIO_WRITE = 0x81,
  TGPU_SELECTOR_RESERVED = 0xff,

} TGPUSelector;
#ifdef __cplusplus
static constexpr uint64_t TGPU_FEATURE_BUFFER_ALLOCATE = 1ULL << 0;
static constexpr uint64_t TGPU_FEATURE_BUFFER_IMPORT = 1ULL << 1;
static constexpr uint64_t TGPU_FEATURE_BUFFER_MAP = 1ULL << 2;
static constexpr uint64_t TGPU_FEATURE_QUEUE = 1ULL << 3;
static constexpr uint64_t TGPU_FEATURE_EXECUTABLE = 1ULL << 4;
static constexpr uint64_t TGPU_FEATURE_SUBMIT = 1ULL << 5;
static constexpr uint64_t TGPU_FEATURE_FENCE = 1ULL << 6;
static constexpr uint64_t TGPU_FEATURE_TIMESTAMP = 1ULL << 7;
static constexpr uint64_t TGPU_FEATURE_FAULT_QUERY = 1ULL << 8;
static constexpr uint64_t TGPU_FEATURE_QUEUE_RESET = 1ULL << 9;
static constexpr uint64_t TGPU_FEATURE_DEVICE_RESET = 1ULL << 10;
static constexpr uint64_t TGPU_FEATURE_DIAGNOSTIC_MMIO = 1ULL << 11;
static constexpr uint64_t TGPU_FEATURE_MASK_V1_0 = (1ULL << 12) - 1;
static constexpr uint32_t TGPU_MEMORY_HOST_VISIBLE = 1U << 0;
static constexpr uint32_t TGPU_MEMORY_DEVICE_LOCAL = 1U << 1;
static constexpr uint32_t TGPU_MEMORY_MASK_V1_0 = 0x3U;
static constexpr uint32_t TGPU_ACCESS_READ = 1U << 0;
static constexpr uint32_t TGPU_ACCESS_WRITE = 1U << 1;
static constexpr uint32_t TGPU_ACCESS_EXECUTE = 1U << 2;
static constexpr uint32_t TGPU_ACCESS_MASK_V1_0 = 0x7U;
static constexpr uint32_t TGPU_RESOURCE_IMAGE = 1U << 0;
static constexpr uint32_t TGPU_RESOURCE_KERNARG = 1U << 1;
static constexpr uint32_t TGPU_RESOURCE_COMMAND = 1U << 2;
static constexpr uint32_t TGPU_RESOURCE_STAGING = 1U << 3;
static constexpr uint32_t TGPU_RESOURCE_MASK_V1_0 = 0xFU;
static constexpr uint32_t TGPU_HEALTH_SCOPE_CLIENT = 1;
static constexpr uint32_t TGPU_HEALTH_SCOPE_QUEUE = 2;
static constexpr uint32_t TGPU_HEALTH_SCOPE_DEVICE = 3;
static constexpr uint32_t TGPU_HEALTH_SCOPE_MASK_V1_0 = 0x3U;
static constexpr uint32_t TGPU_QUEUE_STATE_IDLE = 0;
static constexpr uint32_t TGPU_QUEUE_STATE_READY = 1;
static constexpr uint32_t TGPU_QUEUE_STATE_RESETTING = 2;
static constexpr uint32_t TGPU_QUEUE_STATE_FAULTED = 3;
static constexpr uint32_t TGPU_QUEUE_STATE_CLOSED = 4;
#else
#define TGPU_FEATURE_BUFFER_ALLOCATE (1ULL << 0)
#define TGPU_FEATURE_BUFFER_IMPORT (1ULL << 1)
#define TGPU_FEATURE_BUFFER_MAP (1ULL << 2)
#define TGPU_FEATURE_QUEUE (1ULL << 3)
#define TGPU_FEATURE_EXECUTABLE (1ULL << 4)
#define TGPU_FEATURE_SUBMIT (1ULL << 5)
#define TGPU_FEATURE_FENCE (1ULL << 6)
#define TGPU_FEATURE_TIMESTAMP (1ULL << 7)
#define TGPU_FEATURE_FAULT_QUERY (1ULL << 8)
#define TGPU_FEATURE_QUEUE_RESET (1ULL << 9)
#define TGPU_FEATURE_DEVICE_RESET (1ULL << 10)
#define TGPU_FEATURE_DIAGNOSTIC_MMIO (1ULL << 11)
#define TGPU_FEATURE_MASK_V1_0 ((1ULL << 12) - 1)
#define TGPU_MEMORY_HOST_VISIBLE (1U << 0)
#define TGPU_MEMORY_DEVICE_LOCAL (1U << 1)
#define TGPU_MEMORY_MASK_V1_0 0x3U
#define TGPU_ACCESS_READ (1U << 0)
#define TGPU_ACCESS_WRITE (1U << 1)
#define TGPU_ACCESS_EXECUTE (1U << 2)
#define TGPU_ACCESS_MASK_V1_0 0x7U
#define TGPU_RESOURCE_IMAGE (1U << 0)
#define TGPU_RESOURCE_KERNARG (1U << 1)
#define TGPU_RESOURCE_COMMAND (1U << 2)
#define TGPU_RESOURCE_STAGING (1U << 3)
#define TGPU_RESOURCE_MASK_V1_0 0xFU
#define TGPU_HEALTH_SCOPE_CLIENT 1U
#define TGPU_HEALTH_SCOPE_QUEUE 2U
#define TGPU_HEALTH_SCOPE_DEVICE 3U
#define TGPU_HEALTH_SCOPE_MASK_V1_0 0x3U
#define TGPU_QUEUE_STATE_IDLE 0U
#define TGPU_QUEUE_STATE_READY 1U
#define TGPU_QUEUE_STATE_RESETTING 2U
#define TGPU_QUEUE_STATE_FAULTED 3U
#define TGPU_QUEUE_STATE_CLOSED 4U
#endif

typedef uint64_t TGPUBufferHandle;
typedef uint64_t TGPUMappingHandle;
typedef uint64_t TGPUQueueHandle;
typedef uint64_t TGPUExecutableHandle;
typedef uint64_t TGPUFenceHandle;
typedef uint64_t TGPUSubmissionHandle;

typedef struct TGPURequestHeader {
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t struct_size;
  uint32_t flags;
  uint64_t request_id;
} TGPURequestHeader;

typedef struct TGPUResponseHeader {
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t struct_size;
  uint32_t flags;
  uint32_t status;
  uint32_t failure_stage;
  uint64_t request_id;
} TGPUResponseHeader;

typedef struct TGPUQueryCapabilitiesRequest {
  TGPURequestHeader header;
} TGPUQueryCapabilitiesRequest;

typedef struct TGPUCapabilitiesResponse {
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
} TGPUCapabilitiesResponse;

typedef struct TGPUBufferAllocateRequest {
  TGPURequestHeader header;
  uint64_t size;
  uint64_t alignment;
  uint32_t memory_domain;
  uint32_t access_flags;
  uint32_t resource_flags;
  uint32_t reserved0;
} TGPUBufferAllocateRequest;

typedef struct TGPUBufferAllocateResponse {
  TGPUResponseHeader header;
  TGPUBufferHandle buffer_handle;
  uint64_t committed_size;
  uint32_t granted_access;
  uint32_t memory_domain;
  uint64_t reserved0;
} TGPUBufferAllocateResponse;

typedef struct TGPUBufferImportRequest {
  TGPURequestHeader header;
  uint64_t requested_size;
  uint32_t memory_domain;
  uint32_t access_flags;
  uint32_t import_flags;
  uint32_t reserved0;
} TGPUBufferImportRequest;

typedef struct TGPUBufferImportResponse {
  TGPUResponseHeader header;
  TGPUBufferHandle buffer_handle;
  uint64_t imported_size;
  uint32_t granted_access;
  uint32_t memory_domain;
  uint64_t reserved0;
} TGPUBufferImportResponse;

typedef struct TGPUBufferMapRequest {
  TGPURequestHeader header;
  TGPUBufferHandle buffer_handle;
  uint64_t offset;
  uint64_t length;
  uint32_t access_flags;
  uint32_t map_flags;
  uint64_t reserved0;
} TGPUBufferMapRequest;

typedef struct TGPUBufferMapResponse {
  TGPUResponseHeader header;
  TGPUMappingHandle mapping_handle;
  TGPUBufferHandle buffer_handle;
  uint64_t offset;
  uint64_t length;
  uint32_t granted_access;
  uint32_t reserved0;
} TGPUBufferMapResponse;

typedef struct TGPUBufferUnmapRequest {
  TGPURequestHeader header;
  TGPUMappingHandle mapping_handle;
  uint64_t reserved0;
} TGPUBufferUnmapRequest;

typedef struct TGPUBufferReleaseRequest {
  TGPURequestHeader header;
  TGPUBufferHandle buffer_handle;
  uint64_t reserved0;
} TGPUBufferReleaseRequest;

typedef struct TGPUStatusResponse {
  TGPUResponseHeader header;
} TGPUStatusResponse;

typedef struct TGPUQueueCreateRequest {
  TGPURequestHeader header;
  uint32_t execution_class;
  uint32_t queue_flags;
  uint32_t requested_depth;
  uint32_t reserved0;
} TGPUQueueCreateRequest;

typedef struct TGPUQueueCreateResponse {
  TGPUResponseHeader header;
  TGPUQueueHandle queue_handle;
  uint32_t effective_depth;
  uint32_t effective_inflight;
  uint64_t queue_epoch;
  uint64_t reserved0;
} TGPUQueueCreateResponse;

typedef struct TGPUQueueDestroyRequest {
  TGPURequestHeader header;
  TGPUQueueHandle queue_handle;
  uint64_t reserved0;
} TGPUQueueDestroyRequest;

typedef struct TGPUExecutableAdmitRequest {
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
} TGPUExecutableAdmitRequest;

typedef struct TGPUExecutableAdmitResponse {
  TGPUResponseHeader header;
  TGPUExecutableHandle executable_handle;
  uint8_t recomputed_image_digest[32];
  uint8_t pack_id[16];
  uint8_t entry_id[16];
  uint8_t target[16];
  uint64_t device_epoch;
} TGPUExecutableAdmitResponse;

typedef struct TGPUExecutableReleaseRequest {
  TGPURequestHeader header;
  TGPUExecutableHandle executable_handle;
  uint64_t reserved0;
} TGPUExecutableReleaseRequest;

typedef struct TGPUFenceWaitRef {
  TGPUFenceHandle fence_handle;
  uint64_t fence_value;
} TGPUFenceWaitRef;

typedef struct TGPUCommandRecordHeader {
  uint16_t command_kind;
  uint16_t flags;
  uint32_t record_size;
  uint64_t sequence;
} TGPUCommandRecordHeader;

typedef struct TGPUCommandBinding {
  TGPUBufferHandle buffer_handle;
  uint64_t offset;
  uint64_t length;
  uint32_t access_flags;
  uint32_t reserved0;
} TGPUCommandBinding;

typedef struct TGPUCopyCommandRecord {
  TGPUCommandRecordHeader header;
  TGPUBufferHandle source_buffer;
  uint64_t source_offset;
  TGPUBufferHandle destination_buffer;
  uint64_t destination_offset;
  uint64_t length;
  uint32_t access_flags;
  uint32_t reserved0;
  uint64_t reserved[8];
} TGPUCopyCommandRecord;

typedef struct TGPUFillCommandRecord {
  TGPUCommandRecordHeader header;
  TGPUBufferHandle destination_buffer;
  uint64_t destination_offset;
  uint64_t length;
  uint32_t pattern_width;
  uint32_t access_flags;
  uint8_t pattern[16];
  uint8_t reserved[64];
} TGPUFillCommandRecord;

typedef struct TGPUDispatchCommandRecord {
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
} TGPUDispatchCommandRecord;

typedef struct TGPUBarrierCommandRecord {
  TGPUCommandRecordHeader header;
  uint32_t visibility_scope;
  uint32_t access_scope;
  uint32_t reserved0;
  uint32_t reserved1;
  uint64_t reserved[12];
} TGPUBarrierCommandRecord;

typedef struct TGPUTimestampCommandRecord {
  TGPUCommandRecordHeader header;
  uint8_t label[32];
  uint32_t point;
  uint32_t reserved0;
  uint64_t reserved[9];
} TGPUTimestampCommandRecord;

typedef struct TGPUSubmitRequest {
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
} TGPUSubmitRequest;

typedef struct TGPUSubmitResponse {
  TGPUResponseHeader header;
  TGPUSubmissionHandle submission_handle;
  TGPUFenceHandle fence_handle;
  uint64_t signal_value;
  uint64_t queue_epoch;
  uint64_t device_epoch;
  uint64_t reserved0;
} TGPUSubmitResponse;

typedef struct TGPUFenceWaitRequest {
  TGPURequestHeader header;
  TGPUFenceHandle fence_handle;
  uint64_t fence_value;
  uint64_t timeout_ns;
  uint64_t reserved0;
} TGPUFenceWaitRequest;

typedef struct TGPUFenceWaitResponse {
  TGPUResponseHeader header;
  uint64_t completed_value;
  uint64_t requested_value;
  uint32_t terminal_status;
  uint32_t reserved0;
  uint64_t queue_epoch;
} TGPUFenceWaitResponse;

typedef struct TGPUTimestampQueryRequest {
  TGPURequestHeader header;
  TGPUSubmissionHandle submission_handle;
  TGPUFenceHandle fence_handle;
  uint32_t point;
  uint32_t reserved0;
} TGPUTimestampQueryRequest;

typedef struct TGPUTimestampQueryResponse {
  TGPUResponseHeader header;
  TGPUSubmissionHandle submission_handle;
  TGPUFenceHandle fence_handle;
  uint64_t start_ticks;
  uint64_t end_ticks;
  uint64_t timestamp_frequency_hz;
  uint64_t conversion_numerator;
  uint64_t conversion_denominator;
  uint64_t reserved0;
} TGPUTimestampQueryResponse;

typedef struct TGPUHealthFaultQueryRequest {
  TGPURequestHeader header;
  uint32_t scope;
  uint32_t query_flags;
  uint64_t cursor;
  TGPUQueueHandle queue_handle;
  TGPUSubmissionHandle submission_handle;
  uint64_t reserved[2];
} TGPUHealthFaultQueryRequest;

typedef struct TGPUHealthFaultQueryResponse {
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
} TGPUHealthFaultQueryResponse;

typedef struct TGPUQueueResetRequest {
  TGPURequestHeader header;
  TGPUQueueHandle queue_handle;
  uint32_t reset_flags;
  uint32_t reserved0;
} TGPUQueueResetRequest;

typedef struct TGPUQueueResetResponse {
  TGPUResponseHeader header;
  TGPUQueueHandle queue_handle;
  uint32_t queue_state;
  uint32_t reserved0;
  uint64_t device_epoch;
  uint32_t canceled_submissions;
  uint32_t reserved1;
  uint64_t queue_epoch;
  uint64_t reserved2;
} TGPUQueueResetResponse;

typedef struct TGPUDeviceResetRequest {
  TGPURequestHeader header;
  uint32_t reset_flags;
  uint32_t reason_code;
  uint64_t reserved0;
} TGPUDeviceResetRequest;

typedef struct TGPUDeviceResetResponse {
  TGPUResponseHeader header;
  uint32_t health_state;
  uint32_t reserved0;
  uint64_t previous_device_epoch;
  uint64_t device_epoch;
} TGPUDeviceResetResponse;

typedef struct TGPUDiagnosticMMIOReadRequest {
  TGPURequestHeader header;
  uint64_t bar_relative_offset;
  uint32_t width_bytes;
  uint32_t reserved0;
} TGPUDiagnosticMMIOReadRequest;

typedef struct TGPUDiagnosticMMIOReadResponse {
  TGPUResponseHeader header;
  uint64_t value;
} TGPUDiagnosticMMIOReadResponse;

typedef struct TGPUDiagnosticMMIOWriteRequest {
  TGPURequestHeader header;
  uint64_t bar_relative_offset;
  uint32_t width_bytes;
  uint32_t reserved0;
  uint64_t value;
} TGPUDiagnosticMMIOWriteRequest;

#ifdef __cplusplus
}  // extern "C"

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
#endif

#endif  // TGPU_ABI_H
