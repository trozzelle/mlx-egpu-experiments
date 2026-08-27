#include "TGPUABI.h"
#include "TGPUResponseValidator.h"
#include "TGPUEvidenceLog.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
constexpr const char *kExpectedService = "org.tinygrad.tinygpu.driver2";
constexpr const char *kExpectedPciId = "1002:7551";
constexpr const char *kExpectedArchitecture = "gfx1201";

constexpr uint32_t kInferenceConnectionType = 0;
constexpr uint64_t kRequestCapabilitiesId = 1;
constexpr uint64_t kRequestHealthId = 2;
constexpr uint64_t kRequestBufferAllocateId = 3;
constexpr uint64_t kRequestBufferReleaseId = 4;
constexpr uint64_t kClientDeathBufferBytes = 4096;
constexpr size_t kMaxLogPathBytes = 1024;

enum class ClientCommand : uint8_t {
	ColdLifecycle,
	ClientDeath,
};

struct ClientOptions {
	ClientCommand command = ClientCommand::ColdLifecycle;
	const char *service = nullptr;
	const char *pci_id = nullptr;
	const char *architecture = nullptr;
	const char *log_path = nullptr;
	bool close_with_live_resources = false;
	bool reopen = false;
	bool replay_handles = false;
	bool expect_empty_new_namespace = false;
	bool expect_invalid_handle = false;
};

struct Observation {
	uint32_t abi_major = TGPU_ABI_MAJOR;
	uint32_t abi_minor = TGPU_ABI_MINOR;
	uint32_t selector = TGPU_QUERY_CAPABILITIES;
	uint32_t status = TGPU_STATUS_INTERNAL;
	uint32_t failure_stage = TGPU_FAILURE_ATTACH;
	uint64_t device_epoch = 0;
	uint32_t exit_status = 1;
	uint8_t failure_text[TGPU_MAX_FAULT_TEXT_BYTES] = {};
};

static void PrintUsage(const char *program) {
	std::fprintf(stderr,
	             "Usage: %s cold-lifecycle --service <bundle-id> "
	             "--pci-id 1002:7551 --architecture gfx1201 --log <path>\n"
	             "       %s client-death --service <bundle-id> "
	             "--close-with-live-resources --reopen --replay-handles "
	             "--expect-status TGPU_STATUS_INVALID_HANDLE "
	             "--expect-empty-new-namespace --log <path>\n",
	             program, program);
}

static bool ParseOptions(int argc, char **argv, ClientOptions *options) {
	if (!options || argc < 2) return false;
	if (std::strcmp(argv[1], "cold-lifecycle") == 0) {
		options->command = ClientCommand::ColdLifecycle;
	} else if (std::strcmp(argv[1], "client-death") == 0) {
		options->command = ClientCommand::ClientDeath;
	} else {
		return false;
	}

	for (int index = 2; index < argc; ++index) {
		const char *key = argv[index];
		if (std::strcmp(key, "--close-with-live-resources") == 0) {
			if (options->close_with_live_resources) return false;
			options->close_with_live_resources = true;
		} else if (std::strcmp(key, "--reopen") == 0) {
			if (options->reopen) return false;
			options->reopen = true;
		} else if (std::strcmp(key, "--replay-handles") == 0) {
			if (options->replay_handles) return false;
			options->replay_handles = true;
		} else if (std::strcmp(key, "--expect-empty-new-namespace") == 0) {
			if (options->expect_empty_new_namespace) return false;
			options->expect_empty_new_namespace = true;
		} else {
			if (index + 1 >= argc) return false;
			const char *value = argv[++index];
			if (std::strcmp(key, "--service") == 0) {
				if (options->service) return false;
				options->service = value;
			} else if (std::strcmp(key, "--pci-id") == 0) {
				if (options->pci_id) return false;
				options->pci_id = value;
			} else if (std::strcmp(key, "--architecture") == 0) {
				if (options->architecture) return false;
				options->architecture = value;
			} else if (std::strcmp(key, "--log") == 0) {
				if (options->log_path) return false;
				options->log_path = value;
			} else if (std::strcmp(key, "--expect-status") == 0) {
				if (options->expect_invalid_handle ||
				    std::strcmp(value, "TGPU_STATUS_INVALID_HANDLE") != 0) {
					return false;
				}
				options->expect_invalid_handle = true;
			} else {
				return false;
			}
		}
	}

	if (!options->service || !options->log_path ||
	    std::strlen(options->log_path) >= kMaxLogPathBytes) {
		return false;
	}
	if (options->command == ClientCommand::ColdLifecycle) {
		return options->pci_id && options->architecture &&
		       !options->close_with_live_resources && !options->reopen &&
		       !options->replay_handles &&
		       !options->expect_empty_new_namespace &&
		       !options->expect_invalid_handle;
	}
	return !options->pci_id && !options->architecture &&
	       options->close_with_live_resources && options->reopen &&
	       options->replay_handles && options->expect_invalid_handle &&
	       options->expect_empty_new_namespace;
}

static void SetFailureTextFromHealth(
    Observation *observation, const TGPUHealthFaultQueryResponse &health) {
	if (!observation) return;
	std::memset(observation->failure_text, 0,
	            sizeof(observation->failure_text));
	const size_t bytes = std::min<size_t>(
	    health.text_length,
	    std::min(sizeof(observation->failure_text) - 1,
	             sizeof(health.failure_text)));
	std::memcpy(observation->failure_text, health.failure_text, bytes);
}

static bool EmitObservation(const Observation &observation, const char *path) {
	if (!path) return false;
	TGPUEvidenceRecord record{};
	record.abi_major = observation.abi_major;
	record.abi_minor = observation.abi_minor;
	record.selector = observation.selector;
	record.status = observation.status;
	record.failure_stage = observation.failure_stage;
	record.device_epoch = observation.device_epoch;
	record.exit_status = observation.exit_status;
	std::memcpy(record.failure_text, observation.failure_text,
	            sizeof(record.failure_text));
	return TGPUEvidenceLog::Write(path, record);
}

static bool RegistryStringMatches(io_registry_entry_t entry,
	                              CFStringRef key, const char *expected) {
	if (!entry || !key || !expected) return false;
	CFTypeRef value =
	    IORegistryEntryCreateCFProperty(entry, key, kCFAllocatorDefault, 0);
	if (!value || CFGetTypeID(value) != CFStringGetTypeID()) {
		if (value) CFRelease(value);
		return false;
	}
	CFStringRef expected_value =
	    CFStringCreateWithCString(kCFAllocatorDefault, expected,
	                              kCFStringEncodingUTF8);
	const bool matches =
	    expected_value &&
	    CFStringCompare(reinterpret_cast<CFStringRef>(value), expected_value, 0) ==
	        kCFCompareEqualTo;
	if (expected_value) CFRelease(expected_value);
	CFRelease(value);
	return matches;
}

static bool OpenDriver(const char *requested_service,
	                   io_connect_t *connection) {
	if (!connection || !requested_service ||
	    std::strcmp(requested_service, kExpectedService) != 0) {
		return false;
	}
	*connection = IO_OBJECT_NULL;

	CFMutableDictionaryRef matching = IOServiceMatching("IOUserService");
	if (!matching) return false;
	io_iterator_t iterator = IO_OBJECT_NULL;
	const kern_return_t matching_error =
	    IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
	if (matching_error != KERN_SUCCESS || iterator == IO_OBJECT_NULL) {
		if (iterator != IO_OBJECT_NULL) IOObjectRelease(iterator);
		return false;
	}

	io_service_t candidate = IO_OBJECT_NULL;
	bool ambiguous = false;
	io_service_t service = IOIteratorNext(iterator);
	while (service != IO_OBJECT_NULL) {
		const bool identity_matches = RegistryStringMatches(
		    service, CFSTR("IOUserServerName"), requested_service);
		if (identity_matches) {
			if (candidate == IO_OBJECT_NULL) {
				candidate = service;
			} else {
				ambiguous = true;
				IOObjectRelease(service);
			}
		} else {
			IOObjectRelease(service);
		}
		service = IOIteratorNext(iterator);
	}
	IOObjectRelease(iterator);
	if (ambiguous || candidate == IO_OBJECT_NULL) {
		if (candidate != IO_OBJECT_NULL) IOObjectRelease(candidate);
		return false;
	}

	const kern_return_t open_error = IOServiceOpen(
	    candidate, mach_task_self(), kInferenceConnectionType, connection);
	IOObjectRelease(candidate);
	return open_error == KERN_SUCCESS && *connection != IO_OBJECT_NULL;
}

static bool CallCapabilities(io_connect_t connection,
                             TGPUCapabilitiesResponse *response) {
	if (!response) return false;
	TGPUQueryCapabilitiesRequest request{};
	request.header.abi_major = TGPU_ABI_MAJOR;
	request.header.abi_minor = TGPU_ABI_MINOR;
	request.header.struct_size = sizeof(request);
	request.header.flags = TGPU_REQUEST_FLAGS_V1_0;
	request.header.request_id = kRequestCapabilitiesId;
	size_t response_size = sizeof(*response);
	if (IOConnectCallStructMethod(
	        connection, TGPU_QUERY_CAPABILITIES, &request, sizeof(request),
	        response, &response_size) != KERN_SUCCESS ||
	    response_size != sizeof(*response)) {
		return false;
	}
	const TGPUResponseHeaderValidation validation =
	    TGPUValidateResponseHeader(
	        reinterpret_cast<const std::uint8_t *>(response), response_size,
	        sizeof(*response), kRequestCapabilitiesId);
	return validation.body_usable;
}

static bool CallHealth(io_connect_t connection,
                       TGPUHealthFaultQueryResponse *response) {
	if (!response) return false;
	TGPUHealthFaultQueryRequest request{};
	request.header.abi_major = TGPU_ABI_MAJOR;
	request.header.abi_minor = TGPU_ABI_MINOR;
	request.header.struct_size = sizeof(request);
	request.header.flags = TGPU_REQUEST_FLAGS_V1_0;
	request.header.request_id = kRequestHealthId;
	request.scope = TGPU_HEALTH_SCOPE_CLIENT;
	request.query_flags = TGPU_QUERY_FLAGS_V1_0;
	size_t response_size = sizeof(*response);
	if (IOConnectCallStructMethod(
	        connection, TGPU_HEALTH_FAULT_QUERY, &request, sizeof(request),
	        response, &response_size) != KERN_SUCCESS ||
	    response_size != sizeof(*response)) {
		return false;
	}
	const TGPUResponseHeaderValidation validation =
	    TGPUValidateResponseHeader(
	        reinterpret_cast<const std::uint8_t *>(response), response_size,
	        sizeof(*response), kRequestHealthId);
	return validation.body_usable;
}

static bool CallBufferAllocate(io_connect_t connection,
                               TGPUBufferAllocateResponse *response) {
	if (!response) return false;
	TGPUBufferAllocateRequest request{};
	request.header.abi_major = TGPU_ABI_MAJOR;
	request.header.abi_minor = TGPU_ABI_MINOR;
	request.header.struct_size = sizeof(request);
	request.header.flags = TGPU_REQUEST_FLAGS_V1_0;
	request.header.request_id = kRequestBufferAllocateId;
	request.size = kClientDeathBufferBytes;
	request.alignment = 4096;
	request.memory_domain = TGPU_MEMORY_HOST_VISIBLE;
	request.access_flags = TGPU_ACCESS_READ | TGPU_ACCESS_WRITE;
	request.resource_flags = TGPU_RESOURCE_STAGING;
	size_t response_size = sizeof(*response);
	if (IOConnectCallStructMethod(
	        connection, TGPU_BUFFER_ALLOCATE, &request, sizeof(request),
	        response, &response_size) != KERN_SUCCESS ||
	    response_size != sizeof(*response)) {
		return false;
	}
	const TGPUResponseHeaderValidation validation =
	    TGPUValidateResponseHeader(
	        reinterpret_cast<const std::uint8_t *>(response), response_size,
	        sizeof(*response), kRequestBufferAllocateId);
	if (!validation.body_usable) return false;
	return validation.status == TGPU_STATUS_OK &&
	       response->buffer_handle != 0 &&
	       response->committed_size == kClientDeathBufferBytes;
}

static bool CallBufferRelease(io_connect_t connection, uint64_t buffer_handle,
                              TGPUStatus *status) {
	TGPUBufferReleaseRequest request{};
	request.header.abi_major = TGPU_ABI_MAJOR;
	request.header.abi_minor = TGPU_ABI_MINOR;
	request.header.struct_size = sizeof(request);
	request.header.flags = TGPU_REQUEST_FLAGS_V1_0;
	request.header.request_id = kRequestBufferReleaseId;
	request.buffer_handle = buffer_handle;
	TGPUStatusResponse response{};
	size_t response_size = sizeof(response);
	if (IOConnectCallStructMethod(
	        connection, TGPU_BUFFER_RELEASE, &request, sizeof(request),
	        &response, &response_size) != KERN_SUCCESS ||
	    response_size != sizeof(response)) {
		return false;
	}
	const TGPUResponseHeaderValidation validation =
	    TGPUValidateResponseHeader(
	        reinterpret_cast<const std::uint8_t *>(&response), response_size,
	        sizeof(response), kRequestBufferReleaseId);
	if (!validation.body_usable) return false;
	if (status) *status = validation.status;
	return true;
}

static bool WriteAll(int descriptor, const void *bytes, size_t length) {
	const uint8_t *cursor = static_cast<const uint8_t *>(bytes);
	while (length != 0) {
		const ssize_t count = write(descriptor, cursor, length);
		if (count < 0 && errno == EINTR) continue;
		if (count <= 0) return false;
		cursor += count;
		length -= static_cast<size_t>(count);
	}
	return true;
}

static bool ReadAll(int descriptor, void *bytes, size_t length) {
	uint8_t *cursor = static_cast<uint8_t *>(bytes);
	while (length != 0) {
		const ssize_t count = read(descriptor, cursor, length);
		if (count < 0 && errno == EINTR) continue;
		if (count <= 0) return false;
		cursor += count;
		length -= static_cast<size_t>(count);
	}
	return true;
}

static bool RunLiveAllocationChild(const char *service,
	                               uint64_t *stale_handle) {
	if (!service || !stale_handle) return false;
	*stale_handle = 0;
	int handles[2] = {-1, -1};
	if (pipe(handles) != 0) return false;

	const pid_t child = fork();
	if (child < 0) {
		close(handles[0]);
		close(handles[1]);
		return false;
	}
	if (child == 0) {
		close(handles[0]);
		io_connect_t connection = IO_OBJECT_NULL;
		TGPUBufferAllocateResponse response{};
		const bool opened = OpenDriver(service, &connection);
		const bool allocated = opened && CallBufferAllocate(connection, &response);
		bool sent = false;
		if (allocated) {
			sent = WriteAll(handles[1], &response.buffer_handle,
			                sizeof(response.buffer_handle));
		}
		if (connection != IO_OBJECT_NULL && !allocated) {
			IOServiceClose(connection);
		}
		close(handles[1]);
		// A successful child intentionally exits with the connection and live
		// backing still open.  The kernel then drives user-client Stop/free.
		_exit(allocated && sent ? 0 : 1);
	}

	close(handles[1]);
	uint64_t handle = 0;
	const bool received = ReadAll(handles[0], &handle, sizeof(handle));
	close(handles[0]);
	int wait_status = 0;
	const bool waited = waitpid(child, &wait_status, 0) == child;
	if (!received || !waited || !WIFEXITED(wait_status) ||
	    WEXITSTATUS(wait_status) != 0 || handle == 0) {
		return false;
	}
	*stale_handle = handle;
	return true;
}

static void SetFailureTextLiteral(Observation *observation, const char *text) {
	if (!observation) return;
	std::memset(observation->failure_text, 0,
	            sizeof(observation->failure_text));
	if (!text) return;
	const size_t bytes =
	    std::min(sizeof(observation->failure_text) - 1, std::strlen(text));
	std::memcpy(observation->failure_text, text, bytes);
}

static bool RunClientDeath(const ClientOptions &options,
	                       Observation *observation) {
	if (!observation) return false;

	io_connect_t probe = IO_OBJECT_NULL;
	TGPUCapabilitiesResponse capabilities{};
	if (!OpenDriver(options.service, &probe) ||
	    !CallCapabilities(probe, &capabilities)) {
		if (probe != IO_OBJECT_NULL) IOServiceClose(probe);
		observation->status = TGPU_STATUS_DEVICE_LOST;
		observation->failure_stage = TGPU_FAILURE_ATTACH;
		SetFailureTextLiteral(observation,
		                      "DriverKit inference service unavailable");
		return false;
	}
	observation->abi_major = capabilities.header.abi_major;
	observation->abi_minor = capabilities.header.abi_minor;
	observation->selector = TGPU_QUERY_CAPABILITIES;
	observation->status = capabilities.header.status;
	observation->failure_stage = capabilities.header.failure_stage;
	observation->device_epoch = capabilities.device_epoch;
	const bool identity_matches =
	    capabilities.vendor_id == 0x1002 && capabilities.device_id == 0x7551 &&
	    capabilities.architecture_length == 7 &&
	    std::memcmp(capabilities.architecture, kExpectedArchitecture, 7) == 0;
	const bool backing_capability =
	    (capabilities.feature_bits & TGPU_FEATURE_BUFFER_ALLOCATE) != 0 &&
	    (capabilities.memory_domain_bits & TGPU_MEMORY_HOST_VISIBLE) != 0 &&
	    capabilities.max_buffer_bytes >= kClientDeathBufferBytes &&
	    capabilities.min_buffer_alignment <= 4096;
	if (capabilities.header.status != TGPU_STATUS_OK ||
	    !identity_matches || !backing_capability) {
		IOServiceClose(probe);
		observation->status = capabilities.header.status == TGPU_STATUS_OK
		                         ? TGPU_STATUS_UNSUPPORTED
		                         : capabilities.header.status;
		observation->failure_stage = TGPU_FAILURE_ATTACH;
		SetFailureTextLiteral(observation,
		                      "host-visible buffer capability unavailable");
		return false;
	}

	TGPUHealthFaultQueryResponse health{};
	if (!CallHealth(probe, &health)) {
		IOServiceClose(probe);
		observation->selector = TGPU_HEALTH_FAULT_QUERY;
		observation->status = TGPU_STATUS_DEVICE_LOST;
		observation->failure_stage = TGPU_FAILURE_ATTACH;
		SetFailureTextLiteral(observation,
		                      "DriverKit health query unavailable");
		return false;
	}
	observation->selector = TGPU_HEALTH_FAULT_QUERY;
	observation->status = health.header.status;
	observation->failure_stage = health.header.failure_stage;
	observation->device_epoch = health.device_epoch;
	SetFailureTextFromHealth(observation, health);
	if (health.header.status != TGPU_STATUS_OK ||
	    health.health_state != TGPU_HEALTH_READY) {
		IOServiceClose(probe);
		if (observation->status == TGPU_STATUS_OK) {
			observation->status = TGPU_STATUS_DEVICE_LOST;
		}
		return false;
	}
	IOServiceClose(probe);

	uint64_t stale_handle = 0;
	if (!RunLiveAllocationChild(options.service, &stale_handle)) {
		observation->selector = TGPU_BUFFER_ALLOCATE;
		observation->status = TGPU_STATUS_RESOURCE_EXHAUSTED;
		observation->failure_stage = TGPU_FAILURE_MEMORY;
		SetFailureTextLiteral(observation,
		                      "child could not allocate and close live backing");
		return false;
	}

	io_connect_t reopened = IO_OBJECT_NULL;
	if (!OpenDriver(options.service, &reopened)) {
		observation->selector = TGPU_BUFFER_RELEASE;
		observation->status = TGPU_STATUS_DEVICE_LOST;
		observation->failure_stage = TGPU_FAILURE_ATTACH;
		SetFailureTextLiteral(observation,
		                      "DriverKit inference service did not reopen");
		return false;
	}
	TGPUStatus stale_status = TGPU_STATUS_INTERNAL;
	if (!CallBufferRelease(reopened, stale_handle, &stale_status) ||
	    stale_status != TGPU_STATUS_INVALID_HANDLE) {
		IOServiceClose(reopened);
		observation->selector = TGPU_BUFFER_RELEASE;
		observation->status = stale_status;
		observation->failure_stage = TGPU_FAILURE_MEMORY;
		SetFailureTextLiteral(observation,
		                      "stale release was not rejected as INVALID_HANDLE");
		return false;
	}

	TGPUBufferAllocateResponse fresh{};
	if (!CallBufferAllocate(reopened, &fresh) ||
	    fresh.buffer_handle == stale_handle) {
		IOServiceClose(reopened);
		observation->selector = TGPU_BUFFER_ALLOCATE;
		observation->status = TGPU_STATUS_RESOURCE_EXHAUSTED;
		observation->failure_stage = TGPU_FAILURE_MEMORY;
		SetFailureTextLiteral(observation,
		                      "new connection namespace did not allocate fresh handle");
		return false;
	}
	TGPUStatus fresh_status = TGPU_STATUS_INTERNAL;
	if (!CallBufferRelease(reopened, fresh.buffer_handle, &fresh_status) ||
	    fresh_status != TGPU_STATUS_OK) {
		IOServiceClose(reopened);
		observation->selector = TGPU_BUFFER_RELEASE;
		observation->status = fresh_status;
		observation->failure_stage = TGPU_FAILURE_MEMORY;
		SetFailureTextLiteral(observation,
		                      "fresh namespace buffer release failed");
		return false;
	}
	IOServiceClose(reopened);

	observation->selector = TGPU_BUFFER_RELEASE;
	observation->status = TGPU_STATUS_INVALID_HANDLE;
	observation->failure_stage = TGPU_FAILURE_NONE;
	observation->exit_status = 0;
	return true;
}

}  // namespace

int main(int argc, char **argv) {
	ClientOptions options;
	Observation observation;
	const bool parsed = ParseOptions(argc, argv, &options);
	const bool service_matches =
	    options.service && std::strcmp(options.service, kExpectedService) == 0;
	const bool cold_identity_matches =
	    options.command == ClientCommand::ColdLifecycle && options.pci_id &&
	    options.architecture &&
	    std::strcmp(options.pci_id, kExpectedPciId) == 0 &&
	    std::strcmp(options.architecture, kExpectedArchitecture) == 0;
	if (!parsed || !service_matches ||
	    (options.command == ClientCommand::ColdLifecycle &&
	     !cold_identity_matches)) {
		PrintUsage(argc > 0 ? argv[0] : "tgpu-conformance-client");
		observation.status = TGPU_STATUS_INVALID_REQUEST;
		observation.failure_stage = TGPU_FAILURE_ATTACH;
		observation.exit_status = 2;
		if (options.log_path && std::strlen(options.log_path) < kMaxLogPathBytes) {
			(void)EmitObservation(observation, options.log_path);
		}
		return observation.exit_status;
	}

	if (options.command == ClientCommand::ClientDeath) {
		(void)RunClientDeath(options, &observation);
		const bool log_written = EmitObservation(observation, options.log_path);
		return log_written ? observation.exit_status : 1;
	}

	io_connect_t connection = IO_OBJECT_NULL;
	TGPUCapabilitiesResponse capabilities{};
	TGPUHealthFaultQueryResponse health{};
	if (!OpenDriver(options.service, &connection) ||
	    !CallCapabilities(connection, &capabilities)) {
		if (connection != IO_OBJECT_NULL) IOServiceClose(connection);
		const bool log_written = EmitObservation(observation, options.log_path);
		return log_written ? observation.exit_status : 1;
	}

	observation.abi_major = capabilities.header.abi_major;
	observation.abi_minor = capabilities.header.abi_minor;
	observation.status = capabilities.header.status;
	observation.failure_stage = capabilities.header.failure_stage;
	observation.device_epoch = capabilities.device_epoch;
	const bool abi_matches =
	    capabilities.header.abi_major == TGPU_ABI_MAJOR &&
	    capabilities.header.abi_minor == TGPU_ABI_MINOR &&
	    capabilities.header.struct_size == sizeof(capabilities);
	const bool capabilities_ok = capabilities.header.status == TGPU_STATUS_OK;
	const bool identity_matches =
	    capabilities.vendor_id == 0x1002 && capabilities.device_id == 0x7551 &&
	    capabilities.architecture_length == 7 &&
	    std::memcmp(capabilities.architecture, kExpectedArchitecture, 7) == 0;
	bool health_called = false;
	if (abi_matches && capabilities_ok && identity_matches) {
		health_called = CallHealth(connection, &health);
		if (health_called) {
			observation.selector = TGPU_HEALTH_FAULT_QUERY;
			observation.status = health.header.status;
			observation.failure_stage = health.header.failure_stage;
			observation.device_epoch = health.device_epoch;
			SetFailureTextFromHealth(&observation, health);
		}
	}
	if (!abi_matches || !capabilities_ok || !identity_matches ||
	    !health_called) {
		observation.status = abi_matches && capabilities_ok && identity_matches
		                         ? TGPU_STATUS_DEVICE_LOST
		                         : (capabilities_ok ? TGPU_STATUS_PERMISSION_DENIED
		                                            : capabilities.header.status);
		if (observation.failure_stage == TGPU_FAILURE_NONE) {
			observation.failure_stage = TGPU_FAILURE_ATTACH;
		}
		observation.exit_status = 1;
	} else {
		observation.exit_status =
		    health.health_state == TGPU_HEALTH_READY &&
		            health.header.status == TGPU_STATUS_OK &&
		            health.failure_stage == TGPU_FAILURE_NONE
		        ? 0
		        : 1;
	}
	IOServiceClose(connection);
	const bool log_written = EmitObservation(observation, options.log_path);
	return log_written ? observation.exit_status : 1;
}
