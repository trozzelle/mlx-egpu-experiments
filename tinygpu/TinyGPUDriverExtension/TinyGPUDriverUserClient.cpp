#include "TinyGPUInferenceUserClient.h"
#include "TinyGPURecoveryUserClient.h"
#include "TinyGPUDiagnosticUserClient.h"
#include "TinyGPUDriver.h"
#include "TinyGPUDriverBackingProvider.h"
#include "TGPUHealthRequestValidator.h"
#include "TGPUFixedTransport.h"
#include <DriverKit/DriverKit.h>
#include <DriverKit/OSBoolean.h>
#include <DriverKit/OSData.h>
#include <DriverKit/OSDictionary.h>
#include <DriverKit/OSSharedPtr.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

struct TinyGPUInferenceUserClient_IVars {
	OSSharedPtr<TinyGPUDriver> provider = nullptr;
	TinyGPUDriverBackingProvider *backing_provider = nullptr;
	TinyGPUBufferOwner *buffer_owner = nullptr;
	TGPUBufferValidationLimits limits = {};
	uint64_t connection_epoch = 0;
	bool started = false;
};

struct TinyGPURecoveryUserClient_IVars {
	OSSharedPtr<TinyGPUDriver> provider = nullptr;
};

struct TinyGPUDiagnosticUserClient_IVars {
	OSSharedPtr<TinyGPUDriver> provider = nullptr;
};

namespace {

enum class UserClientRole : uint8_t {
	Inference,
	Recovery,
	Diagnostic,
};

static bool HasRoleEntitlement(IOUserClient *client,
	                           const char *entitlement) {
	if (!client || !entitlement) return false;
	OSDictionary *entitlements = nullptr;
	if (client->CopyClientEntitlements(&entitlements) != kIOReturnSuccess ||
	    !entitlements) {
		return false;
	}
	OSBoolean *allowed =
	    OSDynamicCast(OSBoolean, entitlements->getObject(entitlement));
	const bool result = allowed == kOSBooleanTrue;
	entitlements->release();
	return result;
}

template <typename Request>
static bool CopyValidatedRequest(const std::uint8_t *bytes, size_t length,
                                 const TGPUFixedInputPlan &plan,
                                 Request *request) {
	if (!bytes || !request ||
	    plan.disposition != TGPUFixedTransportDisposition::kExecute ||
	    !plan.execute || plan.request_length != length ||
	    length < sizeof(Request) ||
	    plan.header.struct_size < sizeof(Request)) {
		return false;
	}
	std::memset(request, 0, sizeof(*request));
	// The pure transport seam has already checked the complete byte span,
	// including any zero extension.  Only the declared typed prefix is copied.
	std::memcpy(request, bytes, sizeof(*request));
	return true;
}

static bool GetOutputCapacity(const IOUserClientMethodArguments *args,
                              size_t *capacity) {
	if (!args || !capacity) return false;
	if (args->structureOutputMaximumSize ==
	    kIOUserClientVariableStructureSize) {
		// Variable output is still bounded by the frozen fixed-transport limit.
		*capacity = TGPU_MAX_STRUCT_BYTES;
		return true;
	}
	const std::uint64_t value = args->structureOutputMaximumSize;
	if (value > static_cast<std::uint64_t>(
	                 SIZE_MAX)) {
		return false;
	}
	*capacity = static_cast<size_t>(value);
	return true;
}

static kern_return_t ValidateFixedTransport(
    IOUserClientMethodArguments *args, size_t *output_capacity) {
	if (!args || args->scalarInputCount != 0 ||
	    args->scalarOutputCount != 0 || !args->structureInput ||
	    args->structureInputDescriptor || args->structureOutputDescriptor ||
	    !output_capacity || !GetOutputCapacity(args, output_capacity) ||
	    *output_capacity < sizeof(TGPUResponseHeader)) {
		return kIOReturnBadArgument;
	}
	args->structureOutput = nullptr;
	return kIOReturnSuccess;
}

template <typename Response>
static void ClearResponseAndSetHeader(Response *response, uint64_t request_id,
                                      uint32_t status,
                                      uint32_t failure_stage) {
	if (!response) return;
	std::memset(response, 0, sizeof(*response));
	TGPUSetResponseHeader(&response->header, request_id,
	                      static_cast<uint32_t>(sizeof(*response)), status,
	                      failure_stage);
}

static kern_return_t ReturnResponseBytes(IOUserClientMethodArguments *args,
                                         const void *bytes, size_t length) {
	if (!args || !bytes || length < sizeof(TGPUResponseHeader) ||
	    args->structureOutputDescriptor) {
		return kIOReturnBadArgument;
	}
	if (args->structureOutputMaximumSize !=
	        kIOUserClientVariableStructureSize &&
	    args->structureOutputMaximumSize < length) {
		return kIOReturnBadArgument;
	}
	OSData *output = OSData::withBytes(bytes, length);
	if (!output) return kIOReturnNoMemory;
	args->structureOutput = output;
	return kIOReturnSuccess;
}

static kern_return_t ReturnHeaderResponse(IOUserClientMethodArguments *args,
                                          uint64_t request_id,
                                          TGPUStatus status) {
	TGPUResponseHeader response{};
	TGPUSetResponseHeader(&response, request_id,
	                      static_cast<uint32_t>(sizeof(response)),
	                      static_cast<uint32_t>(status), TGPU_FAILURE_NONE);
	return ReturnResponseBytes(args, &response, sizeof(response));
}

static kern_return_t ReturnTypedResponseError(
    IOUserClientMethodArguments *args, size_t response_size,
    uint64_t request_id, TGPUStatus status) {
	if (response_size < sizeof(TGPUResponseHeader) ||
	    response_size > TGPU_MAX_STRUCT_BYTES) {
		return kIOReturnBadArgument;
	}
	std::uint8_t response[TGPU_MAX_STRUCT_BYTES] = {};
	TGPUResponseHeader header{};
	TGPUSetResponseHeader(&header, request_id,
	                      static_cast<uint32_t>(response_size),
	                      static_cast<uint32_t>(status), TGPU_FAILURE_NONE);
	std::memcpy(response, &header, sizeof(header));
	return ReturnResponseBytes(args, response, response_size);
}


static kern_return_t ReturnResponsePlan(
    IOUserClientMethodArguments *args, const TGPUFixedResponsePlan &plan) {
	if (plan.disposition == TGPUFixedTransportDisposition::kTransportError ||
	    plan.response_bytes < sizeof(TGPUResponseHeader)) {
		return kIOReturnBadArgument;
	}
	return ReturnTypedResponseError(args, plan.response_bytes,
	                                plan.header.request_id, plan.status);
}
static uint32_t HealthStatus(const TGPUHealthFaultQueryResponse &health) {
	return health.health_state == TGPU_HEALTH_READY
	           ? TGPU_STATUS_OK
	           : (health.terminal_status == TGPU_STATUS_OK
	                  ? TGPU_STATUS_DEVICE_LOST
	                  : health.terminal_status);
}

static TGPUStatus StatusForKernelError(kern_return_t err) {
	if (err == kIOReturnNoMemory || err == kIOReturnNoResources) {
		return TGPU_STATUS_RESOURCE_EXHAUSTED;
	}
	if (err == kIOReturnNotPermitted || err == kIOReturnNotPrivileged) {
		return TGPU_STATUS_PERMISSION_DENIED;
	}
	if (err == kIOReturnNoDevice || err == kIOReturnNotAttached) {
		return TGPU_STATUS_DEVICE_LOST;
	}
	return TGPU_STATUS_INTERNAL;
}

enum class FixedSelectorOperation : uint8_t {
	Unsupported,
	QueryCapabilities,
	Health,
	Allocate,
	Import,
	Map,
	Unmap,
	Release,
};

struct FixedSelectorInfo {
	FixedSelectorOperation operation;
	size_t request_size;
	size_t response_size;
};

static FixedSelectorInfo GetSelectorInfo(uint64_t selector) {
	if (selector > UINT32_MAX) {
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPURequestHeader), sizeof(TGPUResponseHeader)};
	}
	switch (static_cast<uint32_t>(selector)) {
	case TGPU_QUERY_CAPABILITIES:
		return {FixedSelectorOperation::QueryCapabilities,
		        sizeof(TGPUQueryCapabilitiesRequest),
		        sizeof(TGPUCapabilitiesResponse)};
	case TGPU_HEALTH_FAULT_QUERY:
		return {FixedSelectorOperation::Health, sizeof(TGPUHealthFaultQueryRequest),
		        sizeof(TGPUHealthFaultQueryResponse)};
	case TGPU_BUFFER_ALLOCATE:
		return {FixedSelectorOperation::Allocate,
		        sizeof(TGPUBufferAllocateRequest),
		        sizeof(TGPUBufferAllocateResponse)};
	case TGPU_BUFFER_IMPORT:
		return {FixedSelectorOperation::Import, sizeof(TGPUBufferImportRequest),
		        sizeof(TGPUBufferImportResponse)};
	case TGPU_BUFFER_MAP:
		return {FixedSelectorOperation::Map, sizeof(TGPUBufferMapRequest),
		        sizeof(TGPUBufferMapResponse)};
	case TGPU_BUFFER_UNMAP:
		return {FixedSelectorOperation::Unmap, sizeof(TGPUBufferUnmapRequest),
		        sizeof(TGPUStatusResponse)};
	case TGPU_BUFFER_RELEASE:
		return {FixedSelectorOperation::Release,
		        sizeof(TGPUBufferReleaseRequest), sizeof(TGPUStatusResponse)};
	case TGPU_QUEUE_CREATE:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUQueueCreateRequest), sizeof(TGPUQueueCreateResponse)};
	case TGPU_QUEUE_DESTROY:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUQueueDestroyRequest), sizeof(TGPUStatusResponse)};
	case TGPU_EXECUTABLE_ADMIT:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUExecutableAdmitRequest),
		        sizeof(TGPUExecutableAdmitResponse)};
	case TGPU_EXECUTABLE_RELEASE:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUExecutableReleaseRequest), sizeof(TGPUStatusResponse)};
	case TGPU_SUBMIT:
		return {FixedSelectorOperation::Unsupported, sizeof(TGPUSubmitRequest),
		        sizeof(TGPUSubmitResponse)};
	case TGPU_FENCE_WAIT:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUFenceWaitRequest), sizeof(TGPUFenceWaitResponse)};
	case TGPU_TIMESTAMP_QUERY:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUTimestampQueryRequest),
		        sizeof(TGPUTimestampQueryResponse)};
	case TGPU_QUEUE_RESET:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUQueueResetRequest), sizeof(TGPUQueueResetResponse)};
	case TGPU_DEVICE_RESET:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUDeviceResetRequest), sizeof(TGPUDeviceResetResponse)};
	case TGPU_DIAGNOSTIC_MMIO_READ:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUDiagnosticMMIOReadRequest),
		        sizeof(TGPUDiagnosticMMIOReadResponse)};
	case TGPU_DIAGNOSTIC_MMIO_WRITE:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPUDiagnosticMMIOWriteRequest),
		        sizeof(TGPUResponseHeader)};
	default:
		return {FixedSelectorOperation::Unsupported,
		        sizeof(TGPURequestHeader), sizeof(TGPUResponseHeader)};
	}
}

static bool IsBufferOperation(FixedSelectorOperation operation) {
	return operation == FixedSelectorOperation::Allocate ||
	       operation == FixedSelectorOperation::Import ||
	       operation == FixedSelectorOperation::Map ||
	       operation == FixedSelectorOperation::Unmap ||
	       operation == FixedSelectorOperation::Release;
}

static bool IsImplementedOperation(FixedSelectorOperation operation) {
	return operation == FixedSelectorOperation::QueryCapabilities ||
	       operation == FixedSelectorOperation::Health ||
	       operation == FixedSelectorOperation::Allocate ||
	       operation == FixedSelectorOperation::Release;
}

static TGPUStatus ValidateImportTransportBody(
    const TGPUBufferImportRequest &request,
    const TGPUBufferValidationLimits &limits) {
	if (request.import_flags != TGPU_IMPORT_FLAGS_V1_0 ||
	    request.reserved0 != 0) {
		return TGPU_STATUS_INVALID_REQUEST;
	}
	if (request.requested_size == 0 ||
	    request.requested_size > limits.max_buffer_bytes) {
		return TGPU_STATUS_RANGE;
	}
	if ((request.memory_domain & ~TGPU_MEMORY_MASK_V1_0) != 0) {
		return TGPU_STATUS_INVALID_REQUEST;
	}
	if (request.memory_domain == 0 ||
	    (request.memory_domain & ~limits.memory_domain_bits) != 0) {
		return TGPU_STATUS_UNSUPPORTED;
	}
	if ((request.access_flags & ~TGPU_ACCESS_MASK_V1_0) != 0 ||
	    request.access_flags == 0) {
		return TGPU_STATUS_INVALID_REQUEST;
	}
	return TGPU_STATUS_OK;
}

static kern_return_t ExternalMethodForRole(
    uint64_t selector, IOUserClientMethodArguments *args,
    const IOUserClientMethodDispatch *in_dispatch, OSObject *in_target,
    void *in_reference, const OSSharedPtr<TinyGPUDriver> &provider,
    TinyGPUBufferOwner *owner, const TGPUBufferValidationLimits *limits,
    UserClientRole role) {
	(void)in_dispatch;
	(void)in_target;
	(void)in_reference;
	if (!provider.get()) return kIOReturnNotAttached;

	const FixedSelectorInfo info = GetSelectorInfo(selector);
	const bool is_buffer = IsBufferOperation(info.operation);
	size_t output_capacity = 0;
	kern_return_t err = ValidateFixedTransport(args, &output_capacity);
	if (err != kIOReturnSuccess) return err;
	const void *input = args->structureInput->getBytesNoCopy();
	const size_t input_length = args->structureInput->getLength();
	const auto *input_bytes = static_cast<const std::uint8_t *>(input);
	const TGPUFixedInputPlan input_plan =
	    TGPUValidateFixedInput(input_bytes, input_length, info.request_size);
	if (input_plan.disposition ==
	    TGPUFixedTransportDisposition::kTransportError) {
		return kIOReturnBadArgument;
	}
	const TGPUFixedResponsePlan response_plan =
	    TGPUPlanFixedResponse(output_capacity, info.response_size,
	                          input_plan.header.request_id);

	const bool role_denied =
	    role == UserClientRole::Diagnostic ||
	    (role != UserClientRole::Inference && is_buffer) ||
	    (role == UserClientRole::Recovery &&
	     !IsImplementedOperation(info.operation));
	const bool unsupported_operation =
	    role_denied || !IsImplementedOperation(info.operation);

	if (unsupported_operation) {
		if (input_plan.execute &&
		    info.operation == FixedSelectorOperation::Unsupported &&
		    info.request_size == sizeof(TGPURequestHeader) &&
		    info.response_size == sizeof(TGPUResponseHeader)) {
			const uint32_t planner_selector =
			    selector <= UINT32_MAX
			        ? static_cast<uint32_t>(selector)
			        : static_cast<uint32_t>(TGPU_SELECTOR_RESERVED);
			const TGPUSelectorPlan selector_plan = TGPUPlanUnsupportedSelector(
			    planner_selector, input_plan, output_capacity);
			if (selector_plan.response_bytes < sizeof(TGPUResponseHeader)) {
				return kIOReturnBadArgument;
			}
			return ReturnHeaderResponse(args, input_plan.header.request_id,
			                            selector_plan.status);
		}

		TGPUStatus typed_status = TGPU_STATUS_OK;
		if (input_plan.execute && !role_denied && limits) {
			if (info.operation == FixedSelectorOperation::Import) {
				TGPUBufferImportRequest request{};
				if (!CopyValidatedRequest(input_bytes, input_length, input_plan,
				                          &request)) {
					typed_status = TGPU_STATUS_INVALID_REQUEST;
				} else {
					typed_status =
					    ValidateImportTransportBody(request, *limits);
				}
			} else if (info.operation == FixedSelectorOperation::Map) {
				TGPUBufferMapRequest request{};
				if (!CopyValidatedRequest(input_bytes, input_length, input_plan,
				                          &request)) {
					typed_status = TGPU_STATUS_INVALID_REQUEST;
				} else {
					typed_status = TGPUValidateBufferMapRequest(
					    request, *limits,
					    static_cast<uint32_t>(
					        sizeof(TGPUBufferMapResponse)));
				}
			} else if (info.operation == FixedSelectorOperation::Unmap) {
				TGPUBufferUnmapRequest request{};
				if (!CopyValidatedRequest(input_bytes, input_length, input_plan,
				                          &request)) {
					typed_status = TGPU_STATUS_INVALID_REQUEST;
				} else {
					typed_status = TGPUValidateBufferUnmapRequest(
					    request,
					    static_cast<uint32_t>(sizeof(TGPUStatusResponse)));
				}
			}
		}
		const TGPUFixedResponsePlan unsupported_plan =
		    TGPUPlanUnsupportedOperation(input_plan, response_plan, typed_status);
		return ReturnResponsePlan(args, unsupported_plan);
	}

	if (!input_plan.execute) {
		if (response_plan.disposition ==
		    TGPUFixedTransportDisposition::kTransportError) {
			return kIOReturnBadArgument;
		}
		if (response_plan.disposition ==
		    TGPUFixedTransportDisposition::kStructuredResponse) {
			return ReturnResponsePlan(args, response_plan);
		}
		return ReturnTypedResponseError(args, info.response_size,
		                                input_plan.header.request_id,
		                                input_plan.status);
	}

	if (response_plan.disposition !=
	    TGPUFixedTransportDisposition::kExecute) {
		return ReturnResponsePlan(args, response_plan);
	}
	if (role == UserClientRole::Inference && is_buffer &&
	    (!owner || !limits)) {
		return kIOReturnNotAttached;
	}

	if (info.operation == FixedSelectorOperation::QueryCapabilities) {
		TGPUQueryCapabilitiesRequest request{};
		TGPUCapabilitiesResponse response{};
		TGPUStatus status =
		    CopyValidatedRequest(input_bytes, input_length, input_plan, &request)
		        ? TGPU_STATUS_OK
		        : TGPU_STATUS_INVALID_REQUEST;
		if (status == TGPU_STATUS_OK) {
			err = provider->QueryCapabilities(&response);
			if (err != kIOReturnSuccess) status = StatusForKernelError(err);
		}
		if (status == TGPU_STATUS_OK) {
			TGPUSetResponseHeader(
			    &response.header, request.header.request_id,
			    static_cast<uint32_t>(sizeof(response)), TGPU_STATUS_OK,
			    TGPU_FAILURE_NONE);
		} else {
			ClearResponseAndSetHeader(&response, request.header.request_id, status,
			                          TGPU_FAILURE_NONE);
		}
		return ReturnResponseBytes(args, &response, sizeof(response));
	}

	if (info.operation == FixedSelectorOperation::Health) {
		TGPUHealthFaultQueryRequest request{};
		TGPUHealthFaultQueryResponse response{};
		TGPUStatus status =
		    CopyValidatedRequest(input_bytes, input_length, input_plan, &request)
		        ? TGPU_STATUS_OK
		        : TGPU_STATUS_INVALID_REQUEST;
		if (status == TGPU_STATUS_OK) {
			status = static_cast<TGPUStatus>(
			    TGPUValidateInferenceHealthRequest(request));
		}
		if (status == TGPU_STATUS_OK) {
			err = provider->QueryHealth(&response);
			if (err != kIOReturnSuccess) {
				status = StatusForKernelError(err);
			} else {
				TGPUSetResponseHeader(
				    &response.header, request.header.request_id,
				    static_cast<uint32_t>(sizeof(response)),
				    HealthStatus(response), response.failure_stage);
			}
		}
		if (status != TGPU_STATUS_OK) {
			ClearResponseAndSetHeader(&response, request.header.request_id, status,
			                          TGPU_FAILURE_NONE);
		}
		return ReturnResponseBytes(args, &response, sizeof(response));
	}

	if (info.operation == FixedSelectorOperation::Allocate) {
		TGPUBufferAllocateRequest request{};
		TGPUBufferAllocateResponse response{};
		TGPUStatus status =
		    CopyValidatedRequest(input_bytes, input_length, input_plan, &request)
		        ? owner->Allocate(request, &response)
		        : TGPU_STATUS_INVALID_REQUEST;
		if (status != TGPU_STATUS_OK) {
			ClearResponseAndSetHeader(&response, request.header.request_id, status,
			                          TGPU_FAILURE_NONE);
		}
		return ReturnResponseBytes(args, &response, sizeof(response));
	}

	TGPUBufferReleaseRequest request{};
	TGPUStatusResponse response{};
	TGPUStatus status =
	    CopyValidatedRequest(input_bytes, input_length, input_plan, &request)
	        ? owner->Release(request, &response)
	        : TGPU_STATUS_INVALID_REQUEST;
	if (status != TGPU_STATUS_OK) {
		ClearResponseAndSetHeader(&response, request.header.request_id, status,
		                          TGPU_FAILURE_NONE);
	}
	return ReturnResponseBytes(args, &response, sizeof(response));
}

static kern_return_t CopyNoMemory(uint64_t type, uint64_t *options,
	                              IOMemoryDescriptor **memory) {
	(void)type;
	(void)options;
	if (memory) *memory = nullptr;
	// BARs, DMA segments, queue controls, and all hardware-consumed storage
	// remain DEXT-owned. No normal TGPU selector exposes a mapping.
	return kIOReturnUnsupported;
}

static void DestroyBufferOwner(TinyGPUBufferOwner *owner) {
	if (!owner) return;
	owner->~TinyGPUBufferOwner();
	IOFree(owner, sizeof(TinyGPUBufferOwner));
}

static void DestroyBackingProvider(
    TinyGPUDriverBackingProvider *backing_provider) {
	if (!backing_provider) return;
	backing_provider->Reset();
	backing_provider->~TinyGPUDriverBackingProvider();
	IOFree(backing_provider, sizeof(TinyGPUDriverBackingProvider));
}

static void CleanupInferenceClient(TinyGPUInferenceUserClient_IVars *ivars) {
	if (!ivars) return;
	ivars->started = false;
	if (ivars->buffer_owner) {
		// CleanupClient invalidates public table tokens before provider
		// teardown.  The destructor repeats the bounded idempotent pass.
		(void)ivars->buffer_owner->CleanupClient();
		DestroyBufferOwner(ivars->buffer_owner);
		ivars->buffer_owner = nullptr;
	}
	if (ivars->backing_provider) {
		DestroyBackingProvider(ivars->backing_provider);
		ivars->backing_provider = nullptr;
	}
	ivars->connection_epoch = 0;
	ivars->limits = {};
	ivars->provider.reset();
}
}  // namespace

bool TinyGPUInferenceUserClient::init() {
	auto ok = super::init();
	if (!ok) return false;
	ivars = IONewZero(TinyGPUInferenceUserClient_IVars, 1);
	return ivars != nullptr;
}

void TinyGPUInferenceUserClient::free() {
	if (ivars) {
		CleanupInferenceClient(ivars);
		IOSafeDeleteNULL(ivars, TinyGPUInferenceUserClient_IVars, 1);
	}
	super::free();
}

kern_return_t TinyGPUInferenceUserClient::Start_Impl(IOService *in_provider) {
	if (!ivars) return kIOReturnNoMemory;
	if (ivars->started) return kIOReturnBusy;
	kern_return_t err = Start(in_provider, SUPERDISPATCH);
	if (err != kIOReturnSuccess) return err;

	TinyGPUDriver *typed_provider =
	    OSDynamicCast(TinyGPUDriver, in_provider);
	if (!typed_provider) {
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNotAttached;
	}
	if (!HasRoleEntitlement(this, "org.tinygrad.tinygpu.inference")) {
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNotPermitted;
	}

	uint64_t max_buffer_bytes = 0;
	uint64_t min_buffer_alignment = 0;
	uint32_t memory_domain_bits = 0;
	err = typed_provider->GetBufferBackingLimits(
	    &max_buffer_bytes, &min_buffer_alignment, &memory_domain_bits);
	if (err != kIOReturnSuccess) {
		(void)Stop(in_provider, SUPERDISPATCH);
		return err;
	}

	void *backing_storage =
	    IOMallocZero(sizeof(TinyGPUDriverBackingProvider));
	if (!backing_storage) {
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNoMemory;
	}
	auto *backing_provider = new (backing_storage)
	    TinyGPUDriverBackingProvider(TinyGPUDriverBackingProvider::kCapacity,
	                                 max_buffer_bytes, min_buffer_alignment,
	                                 memory_domain_bits);
	if (!backing_provider->IsReady()) {
		DestroyBackingProvider(backing_provider);
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNoMemory;
	}

	uint64_t connection_epoch = 0;
	err = typed_provider->AllocateConnectionEpoch(&connection_epoch);
	if (err != kIOReturnSuccess || connection_epoch == 0 ||
	    connection_epoch > 0xffffffffULL) {
		DestroyBackingProvider(backing_provider);
		(void)Stop(in_provider, SUPERDISPATCH);
		return err == kIOReturnSuccess ? kIOReturnNoResources : err;
	}

	TGPUBufferValidationLimits limits{};
	limits.connection_epoch = connection_epoch;
	limits.max_buffer_bytes = max_buffer_bytes;
	limits.max_mapping_bytes = max_buffer_bytes;
	limits.min_buffer_alignment = min_buffer_alignment;
	limits.min_mapping_alignment = min_buffer_alignment;
	limits.memory_domain_bits = memory_domain_bits;

	void *owner_storage = IOMallocZero(sizeof(TinyGPUBufferOwner));
	if (!owner_storage) {
		DestroyBackingProvider(backing_provider);
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNoMemory;
	}
	auto *owner = new (owner_storage)
	    TinyGPUBufferOwner(connection_epoch,
	                       TinyGPUDriverBackingProvider::kCapacity,
	                       *backing_provider, limits);
	if (!owner->IsReady()) {
		DestroyBufferOwner(owner);
		DestroyBackingProvider(backing_provider);
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNoResources;
	}

	ivars->provider = OSSharedPtr<TinyGPUDriver>(typed_provider, OSRetain);
	ivars->backing_provider = backing_provider;
	ivars->buffer_owner = owner;
	ivars->limits = limits;
	ivars->connection_epoch = connection_epoch;
	ivars->started = true;
	return kIOReturnSuccess;
}

kern_return_t TinyGPUInferenceUserClient::Stop_Impl(IOService *in_provider) {
	if (!ivars) return kIOReturnNotAttached;
	CleanupInferenceClient(ivars);
	return Stop(in_provider, SUPERDISPATCH);
}

kern_return_t TinyGPUInferenceUserClient::ExternalMethod(
    uint64_t selector, IOUserClientMethodArguments *args,
    const IOUserClientMethodDispatch *in_dispatch, OSObject *in_target,
    void *in_reference) {
	if (!ivars || !ivars->started || !ivars->provider.get() ||
	    !ivars->buffer_owner || !ivars->backing_provider) {
		return kIOReturnNotAttached;
	}
	return ExternalMethodForRole(
	    selector, args, in_dispatch, in_target, in_reference, ivars->provider,
	    ivars->buffer_owner, &ivars->limits, UserClientRole::Inference);
}

kern_return_t IMPL(TinyGPUInferenceUserClient, CopyClientMemoryForType) {
	return CopyNoMemory(type, options, memory);
}

bool TinyGPURecoveryUserClient::init() {
	auto ok = super::init();
	if (!ok) return false;
	ivars = IONewZero(TinyGPURecoveryUserClient_IVars, 1);
	return ivars != nullptr;
}

void TinyGPURecoveryUserClient::free() {
	if (ivars) {
		ivars->provider.reset();
		IOSafeDeleteNULL(ivars, TinyGPURecoveryUserClient_IVars, 1);
	}
	super::free();
}

kern_return_t TinyGPURecoveryUserClient::Start_Impl(IOService *in_provider) {
	if (!ivars) return kIOReturnNoMemory;
	kern_return_t err = Start(in_provider, SUPERDISPATCH);
	if (err != kIOReturnSuccess) return err;
	TinyGPUDriver *typed_provider =
	    OSDynamicCast(TinyGPUDriver, in_provider);
	if (!typed_provider) {
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNotAttached;
	}
	if (!HasRoleEntitlement(this, "org.tinygrad.tinygpu.recovery")) {
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNotPermitted;
	}
	ivars->provider = OSSharedPtr<TinyGPUDriver>(typed_provider, OSRetain);
	return kIOReturnSuccess;
}

kern_return_t TinyGPURecoveryUserClient::Stop_Impl(IOService *in_provider) {
	if (!ivars) return kIOReturnNotAttached;
	ivars->provider.reset();
	return Stop(in_provider, SUPERDISPATCH);
}

kern_return_t TinyGPURecoveryUserClient::ExternalMethod(
    uint64_t selector, IOUserClientMethodArguments *args,
    const IOUserClientMethodDispatch *in_dispatch, OSObject *in_target,
    void *in_reference) {
	if (!ivars) return kIOReturnNotAttached;
	return ExternalMethodForRole(selector, args, in_dispatch, in_target,
	                             in_reference, ivars->provider, nullptr,
	                             nullptr, UserClientRole::Recovery);
}

kern_return_t IMPL(TinyGPURecoveryUserClient, CopyClientMemoryForType) {
	return CopyNoMemory(type, options, memory);
}

bool TinyGPUDiagnosticUserClient::init() {
	auto ok = super::init();
	if (!ok) return false;
	ivars = IONewZero(TinyGPUDiagnosticUserClient_IVars, 1);
	return ivars != nullptr;
}

void TinyGPUDiagnosticUserClient::free() {
	if (ivars) {
		ivars->provider.reset();
		IOSafeDeleteNULL(ivars, TinyGPUDiagnosticUserClient_IVars, 1);
	}
	super::free();
}

kern_return_t TinyGPUDiagnosticUserClient::Start_Impl(IOService *in_provider) {
	if (!ivars) return kIOReturnNoMemory;
	kern_return_t err = Start(in_provider, SUPERDISPATCH);
	if (err != kIOReturnSuccess) return err;
	TinyGPUDriver *typed_provider =
	    OSDynamicCast(TinyGPUDriver, in_provider);
	if (!typed_provider) {
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNotAttached;
	}
	if (!HasRoleEntitlement(this, "org.tinygrad.tinygpu.diagnostic")) {
		(void)Stop(in_provider, SUPERDISPATCH);
		return kIOReturnNotPermitted;
	}
	ivars->provider = OSSharedPtr<TinyGPUDriver>(typed_provider, OSRetain);
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDiagnosticUserClient::Stop_Impl(IOService *in_provider) {
	if (!ivars) return kIOReturnNotAttached;
	ivars->provider.reset();
	return Stop(in_provider, SUPERDISPATCH);
}

kern_return_t TinyGPUDiagnosticUserClient::ExternalMethod(
    uint64_t selector, IOUserClientMethodArguments *args,
    const IOUserClientMethodDispatch *in_dispatch, OSObject *in_target,
    void *in_reference) {
	if (!ivars) return kIOReturnNotAttached;
	return ExternalMethodForRole(selector, args, in_dispatch, in_target,
	                             in_reference, ivars->provider, nullptr,
	                             nullptr, UserClientRole::Diagnostic);
}

kern_return_t IMPL(TinyGPUDiagnosticUserClient, CopyClientMemoryForType) {
	return CopyNoMemory(type, options, memory);
}
