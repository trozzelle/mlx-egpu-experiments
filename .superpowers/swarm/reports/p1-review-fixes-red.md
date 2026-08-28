# P1 review-fixes RED contracts

## Status and scope

This is a RED-only contract packet for the validated P1 Wave A review findings. It adds no production implementation, DriverKit behavior, package/configuration assertion, service-registry test, firmware fake, or hardware success path.

No validation command, test, build, formatter, linter, package-manager, install/signing, Xcode, or hardware command was run. The commands below are exact supervisor commands recorded for later execution; their expected failures are not observed output.

## Changed files

- Historical execution/provenance boundary: TinyGPU source changes were executed/reviewed in former external checkout `<former-tinygpu-worktree>` on branch `feature/r9700-device-owner`; changed-file paths below are provenance only.
  - `extra/usbgpu/tbgpu/installer/Conformance/tests/tgpu_resource_table_contract.cpp`
  - `extra/usbgpu/tbgpu/installer/Conformance/tests/test_tgpu_framebuffer_contract.cpp`
  - `extra/usbgpu/tbgpu/installer/Conformance/tests/test_tgpu_health_request_contract.cpp`
  - `extra/usbgpu/tbgpu/installer/Conformance/tests/test_tgpu_evidence_log_contract.cpp`
- Current source authority and reproduction root: `<repo-root>/tinygpu` on branch `feature/r9700-products-wave-a`; current commands run from `tinygpu/`.
- Current orchestration/evidence file: products checkout `feature/r9700-products-wave-a` — `.superpowers/swarm/reports/p1-review-fixes-red.md`.

Only the four conformance test files and this report were edited. No production header/source, Xcode/project/package/app file, or existing Python test was changed.

## Contract matrix and mutations caught

| Review finding | RED behavior | Mutation caught |
|---|---|---|
| P1A-TOKEN-001 / P1-BUFFER-001 | `tgpu_resource_table_contract.cpp` creates table A at epoch 1 and allocates once, then table B at epoch 2 and allocates twice. The old A token must remain `TGPU_STATUS_INVALID_HANDLE` in B. The historical case, recorded in the former checkout before the 2026-08-27 migration checkpoint `9d83a0a`, deliberately reaches the concrete epoch-1/nonce-1 versus epoch-2/nonce-2 alias in that then-current XOR mixer without asserting an opaque token layout. | Reusing `MixToken(connection_epoch ^ salt ^ nonce)` across table lifetimes, omitting the full connection epoch from the capability namespace, or accepting a stale token as a live same-kind resource in the new table. The former implementation was expected to compile and then fail behaviorally at `epoch-one token remains invalid in epoch-two namespace` because its second epoch-two allocation aliased the epoch-one input and resolved as `OK`; the current products outcome is recorded below. |
| P1-COLD-002 | `test_tgpu_framebuffer_contract.cpp` uses upper-bit-bearing raw fields whose low 24-bit values are base `0x00008000` and top `0x00009000`. It requires base bytes `0x0000008000000000`, top bytes `0x0000009000000000`, and aperture register values equal to each decoded address shifted right by 18. A masked wrapped/descending range must return `RANGE` without changing the output; a null output is `INVALID_REQUEST`. | Shifting raw register values without the `0x00ffffff` mask and `<< 24` expansion, programming aperture values from the unexpanded raw fields, accepting `top < base`, unchecked malformed ranges, or writing output on rejection. At contract-authoring time, the former checkout had no requested pure decoder seam, so this historical contract was expected to stop at a missing-header/source compile failure before runtime. |
| P1A-UC-001 / P1-ABI-001 | `test_tgpu_health_request_contract.cpp` accepts exactly a v1.0 client-scope request with common header flags and typed query flags, cursor, queue handle, submission handle, and both reserved words zero. It rejects device scope, queue scope, unknown scope, and each individual nonzero unsupported field. The seam is pure so the user-client can invoke it after common header checks and before provider `QueryHealth` access. | Calling the provider for device/queue scope from an inference client, accepting nonzero reserved/unknown flags or cursor/handles, or validating only the common header/trailing bytes while ignoring the typed request body. At contract-authoring time, the former checkout had no requested pure validator seam, so this historical contract was expected to stop at a missing-header/source compile failure before runtime. |
| P1-COLD-004 / P1-EVIDENCE-001 / P1-EVIDENCE-002 | `test_tgpu_evidence_log_contract.cpp` builds a real temporary filesystem path with several absent parent directories and requires durable creation. It compares the complete eight-line record exactly: the seven required bounded fields (`abi_major`, `abi_minor`, `selector`, `status`, `failure_stage`, `device_epoch`, `exit_status`) plus one `failure_text=` line. The health selector is `13`; the failure text is the exact private label `cold_stage=PspSosTmr`, preserving stage diagnosis when the numeric stage is the generic firmware class. Newline/ESC injection must remain one sanitized line (`?` replacement), a full unterminated 192-byte field must emit at most 191 text bytes, and an existing regular file used as a parent must make `Write` report failure. | Losing the requested artifact when a parent directory is absent, emitting health-derived fields under the wrong selector, accepting newline/control injection, reading beyond the frozen 192-byte health field, silently logging open/write/close failures, or returning success without durable evidence. At contract-authoring time, the former checkout had no requested evidence seam, so this historical contract was expected to stop at a missing-header/source compile failure before runtime. |
Cold firmware ownership itself remains an external/provenance blocker. No fake firmware load or warm-state success test was added; the existing fail-closed cold ownership review remains authoritative.

## Current in-repository outcome

As recorded after the 2026-08-27 migration checkpoint `9d83a0a`, `TinyGPUResourceTable.{h,cpp}`, `TGPUFramebufferDecoder.{h,cpp}`, `TGPUHealthRequestValidator.{h,cpp}`, and `TGPUEvidenceLog.{h,cpp}` are present under `tinygpu/TinyGPUDriverExtension/`. The named resource-table, framebuffer, health-validator, and evidence-log host contracts, together with the other named host C++ contracts, compiled from `tinygpu/` and exited `0` in the recorded Task 3 verification under the selected Xcode 26.6 build `17F113` / DriverKit SDK `25.5` source gate. The proxy cutover is complete: the direct DriverKit client has no socket/proxy fallback and `server.c` is excluded from the verification. The RED commands below remain historical contract commands and were not rerun by this report. Cold firmware ownership, import/private-VA completion, and hardware acceptance remain externally blocked; no hardware success claim is made.

## Smallest wished-for production seams

These are declarations required by the RED tests only; they are not implemented in this packet and are not generic HAL/logger abstractions.

### Framebuffer decoder

`TinyGPUDriverExtension/TGPUFramebufferDecoder.h` and `.cpp` should expose a DriverKit-independent result and function equivalent to:

```cpp
struct TGPUFramebufferDecodeResult {
  uint64_t base_bytes;
  uint64_t top_bytes;
  uint32_t base_aperture_register;
  uint32_t top_aperture_register;
};

TGPUStatus TGPUDecodeFramebufferLocation(
    uint32_t raw_base, uint32_t raw_top,
    TGPUFramebufferDecodeResult* out);
```

The implementation must mask each raw field to `0x00ffffff`, expand by 24 bits, reject invalid/checked ranges before publishing output, and derive each aperture register from the expanded byte address with `>> 18`.

### Inference health validator

`TinyGPUDriverExtension/TGPUHealthRequestValidator.h` and `.cpp` should expose a pure typed check equivalent to:

```cpp
TGPUStatus TGPUValidateInferenceHealthRequest(
    const TGPUHealthFaultQueryRequest& request);
```

The user-client must call it before provider access. It must allow only `TGPU_HEALTH_SCOPE_CLIENT` for the inference class and require v1.0 request/query flags, cursor, queue/submission handles, and both reserved words to be zero. Common ABI major/minor/size validation remains part of the existing request boundary.

### Evidence log

`TinyGPUDriverExtension/TGPUEvidenceLog.h` and `.cpp` should expose a TinyGPU-specific bounded record and checked writer equivalent to:

```cpp
struct TGPUEvidenceRecord {
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t selector;
  uint32_t status;
  uint32_t failure_stage;
  uint64_t device_epoch;
  uint32_t exit_status;
  uint8_t failure_text[TGPU_MAX_FAULT_TEXT_BYTES];
};

class TGPUEvidenceLog final {
 public:
  static bool Write(const char* path, const TGPUEvidenceRecord& record);
};
```

`failure_text` is sourced from the frozen health response field and remains bounded to `TGPU_MAX_FAULT_TEXT_BYTES` (192 bytes including its terminator). `Write` must create missing parent directories, emit exactly one sanitized `failure_text=` line, replace control bytes deterministically, and return false for directory/open/write/close failure. The required private-stage case is the value `cold_stage=PspSosTmr` inside that one line; no separate `cold_stage` key is added.

## Exact RED commands (supervisor only; do not run in this packet)

Run from the products worktree's `tinygpu/` directory, not the repository root.

### Token replay contract

```sh
cd <repo-root>/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TinyGPUResourceTable.cpp \
  Conformance/tests/tgpu_resource_table_contract.cpp \
  -o /tmp/tgpu_resource_table_contract \
  && /tmp/tgpu_resource_table_contract
```

Historical expected result (former checkout, recorded before the 2026-08-27 migration checkpoint `9d83a0a`; not executed here): compilation succeeded against that checkout's resource-table seam, then the binary was expected to exit nonzero because the then-current XOR inputs aliased. The expected first diagnostic was conceptually:

```text
FAIL: epoch-one token remains invalid in epoch-two namespace (observed=0 expected=5)
```

### Framebuffer decoder contract

```sh
cd <repo-root>/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUFramebufferDecoder.cpp \
  Conformance/tests/test_tgpu_framebuffer_contract.cpp \
  -o /tmp/tgpu_framebuffer_contract \
  && /tmp/tgpu_framebuffer_contract
```

Historical expected result (former checkout, recorded before the 2026-08-27 migration checkpoint `9d83a0a`; not executed here): compilation stopped before the run because `TGPUFramebufferDecoder.cpp` and `TGPUFramebufferDecoder.h` were absent. This was an intentional historical missing-seam failure, not a syntax or source-text assertion. The current products seam and host-contract result are recorded above.

### Inference health validator contract

```sh
cd <repo-root>/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUHealthRequestValidator.cpp \
  Conformance/tests/test_tgpu_health_request_contract.cpp \
  -o /tmp/tgpu_health_request_contract \
  && /tmp/tgpu_health_request_contract
```

Historical expected result (former checkout, recorded before the 2026-08-27 migration checkpoint `9d83a0a`; not executed here): compilation stopped before the run because `TGPUHealthRequestValidator.cpp` and `TGPUHealthRequestValidator.h` were absent. This was an intentional historical missing-seam failure, not a syntax or mock-provider assertion. The current products seam and host-contract result are recorded above.

### Evidence-log contract

```sh
cd <repo-root>/tinygpu
xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror \
  -I TinyGPUDriverExtension \
  TinyGPUDriverExtension/TGPUEvidenceLog.cpp \
  Conformance/tests/test_tgpu_evidence_log_contract.cpp \
  -o /tmp/tgpu_evidence_log_contract \
  && /tmp/tgpu_evidence_log_contract
```

Historical expected result (former checkout, recorded before the 2026-08-27 migration checkpoint `9d83a0a`; not executed here): compilation stopped before the run because `TGPUEvidenceLog.cpp` and `TGPUEvidenceLog.h` were absent. This was an intentional historical missing-seam failure, not a syntax or generic-logger test. The current products seam and host-contract result are recorded above.
