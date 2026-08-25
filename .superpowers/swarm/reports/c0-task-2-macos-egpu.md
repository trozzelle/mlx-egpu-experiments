# C0 Task Set 2 — macOS eGPU minimal runtime probe

## Implementation

- Added `experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp`.
- The probe is tinygrad-free C++17 and uses only libusb plus the C++ standard library.
- It enumerates the pinned TinyGPU USB IDs from `docs/pinned-upstream-interfaces.md` / `docs/egpu-prefill-offload-reference.md`:
  - `0xADD1:0x0001`
  - `0x3801:0x0001`
- It prints the C0 log-policy fields available before a native TinyGPU transport exists:
  - source name and command placeholder;
  - runtime substrate (`macOS TinyGPU USB/libusb tinygrad-free probe`);
  - no-model note;
  - known USB IDs and matched device VID/PID, bus/address, USB class, and strings when the device can be opened;
  - deterministic vector-add input samples and CPU expected digest;
  - device output and CPU comparison status as unavailable/not-run;
  - host/device transfer status, kernel launch status, elapsed time, failure text, and exit status semantics.

## Expected supervisor command

From `docs/tasks/native-r9700-producer/validation-commands.md`:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -I/opt/homebrew/include/libusb-1.0 experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -lusb-1.0 -o build/native-r9700-runtime/macos_tinygpu_minimal && ./build/native-r9700-runtime/macos_tinygpu_minimal"; date -u +"timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -I/opt/homebrew/include/libusb-1.0 experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp -L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib -lusb-1.0 -o build/native-r9700-runtime/macos_tinygpu_minimal && ./build/native-r9700-runtime/macos_tinygpu_minimal; status=$?; printf "exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected log path: `logs/c0-macos-egpu-minimal-runtime.log`.

## Success/blocker interpretation

Current expected outcome is a loud blocker, not a fake GPU pass:

- `exit_status: 1` means libusb setup/device-listing failed before safe TinyGPU discovery.
- `exit_status: 2` means one or more TinyGPU USB devices were discovered, but host/device transfer and kernel launch were intentionally not attempted because the safe native TinyGPU DMA mapping, command queue, and kernel dispatch ABI are not pinned for tinygrad-free use in this repo.
- `exit_status: 3` means no USB device matched the pinned TinyGPU IDs; the same native DMA/queue/kernel ABI blocker still applies.

A promotable macOS success would require extending this probe with a tinygrad-free implementation of host→device write, deterministic kernel launch, device→host readback, and CPU comparison for the vector-add sample. That capability is not safely available from the pinned docs without importing/calling tinygrad or reimplementing unpinned TinyGPU internals.

Linux ROCm/HIP can proceed independently as the production-candidate/reference lane; this macOS blocker does not decide the final C0 substrate.

## Task doc update

Updated only task set 2's row in `docs/archive/tasks/native-r9700-producer/phase-c0-runtime-discovery.md` with the source path, exact blocker, and Linux-lane independence.

## Validation not run

Per OMP task-executor policy, I did not run the build/run command, tests, linters, formatters, package managers, git commands, or project-wide suites. Supervisor should run the recorded command above.
