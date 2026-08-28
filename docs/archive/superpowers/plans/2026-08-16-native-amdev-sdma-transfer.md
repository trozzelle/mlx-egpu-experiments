# Native AMDev/SDMA Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tinygrad-free macOS TinyGPU.app/APLRemotePCIDevice/PCIIface host-device transfer proof for the local R9700/gfx1201.

**Architecture:** Add an experiment-only C++ probe with four local units: TinyGPU.app RemoteCmd transport, minimal AMD discovery/VM mapping, SDMA queue submission, and a structured transfer harness. Port only the required MIT-licensed tinygrad AMDev/SDMA mechanics, with provenance and license notice, and keep libusb as a negative control only.

**Tech Stack:** C++17, macOS `xcrun --sdk macosx clang++`, Python 3.12 pytest for no-hardware contract tests, TinyGPU.app local UNIX socket, existing `docs/tasks/native-r9700-producer/validation-commands.md` command ledger.

## Global Constraints

- Required shared work boundary: `<former-native-r9700-worktree>` on branch `feature/native-r9700-producer`.
- Native proof code must not import, shell out to, or dynamically depend on tinygrad.
- Substantial tinygrad-derived C++ must include MIT notice and file/line provenance comments.
- No libusb/`USBIface` acceptance path; `macos_tinygpu_minimal.cpp` remains a stale negative control unless explicitly labeled as such.
- No model code, C1 runtime wrapper, generic backend framework, mlx-lm/oMLX integration, compute kernel dispatch, TCP transport, multi-device support, or non-macOS backend support in this transfer proof.
- Every OMP executor records recommended commands but does not run tests, linters, formatters, package managers, git commands, or project-wide suites; the supervisor runs verification after each wave.

---

## File structure

- Create `tests/test_native_amdev_transfer_contract.py`: no-hardware pytest contract for the native C++ probe. It compiles `native_amdev_transfer_probe.cpp` and exercises self-test modes for RemoteCmd framing and required log field declarations.
- Create `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`: single-file experiment until the transfer proof passes. It contains local `remote_pci`, `amd_discovery`, `amd_vm`, `sdma_queue`, and `transfer_probe` sections with provenance comments.
- Modify `docs/tasks/native-r9700-producer/validation-commands.md`: add the exact no-hardware contract-test command first, then the exact hardware build/run/log command when the probe is ready.
- Create `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`: executable task ledger for this phase.
- Update `.superpowers/swarm/progress.md`: append C0B task rows without changing prior Done rows.
- Write reports under `.superpowers/swarm/reports/` for every C0B task.

---

### Task 1: Red contract tests and phase task doc

**Files:**
- Create: `tests/test_native_amdev_transfer_contract.py`
- Create/update: `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Modify: `.superpowers/swarm/progress.md`
- Report: `.superpowers/swarm/reports/c0b-task-1-red-contract.md`

**Interfaces:**
- Consumes: native boundary spec `docs/archive/superpowers/specs/2026-08-16-native-amdev-sdma-boundary-design.md`.
- Produces: pytest command `${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v`; expected RED failure before production source exists.

- [ ] **Step 1: Write the failing test**

Create `tests/test_native_amdev_transfer_contract.py` with these behaviors:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp"


def compile_probe(tmp_path: Path) -> Path:
    assert SOURCE.exists(), "native transfer probe source missing"
    exe = tmp_path / "native_amdev_transfer_probe"
    subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(SOURCE),
            "-o",
            str(exe),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return exe


def run_self_test(exe: Path, name: str) -> str:
    result = subprocess.run(
        [str(exe), "--self-test", name],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def test_remote_cmd_frame_self_test_passes(tmp_path):
    exe = compile_probe(tmp_path)
    out = run_self_test(exe, "remote-cmd-frame")
    assert "self_test: remote-cmd-frame" in out
    assert "status: pass" in out


def test_log_contract_self_test_lists_required_fields(tmp_path):
    exe = compile_probe(tmp_path)
    out = run_self_test(exe, "log-contract")
    required = [
        "runtime_substrate",
        "pci_id",
        "arch",
        "transfer_byte_count",
        "cpu_comparison_status",
        "host_device_transfer_status",
        "failure_stage",
        "failure_text",
        "exit_status",
    ]
    for field in required:
        assert f"required_log_field: {field}" in out
    assert "status: pass" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected RED result before Task 2: FAIL with `AssertionError: native transfer probe source missing`.

- [ ] **Step 3: Record the command**

Add the exact pytest command and RED expectation to `validation-commands.md` under a C0B section. Mark the final hardware transfer command as owned by Task 5 until discovered.

- [ ] **Step 4: Commit after supervisor verification**

Supervisor only: after reviewing the report and seeing the expected RED failure, commit the test/task-doc wave.

---

### Task 2: RemoteCmd transport and no-hardware self-tests

**Files:**
- Create: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `.superpowers/swarm/progress.md`
- Report: `.superpowers/swarm/reports/c0b-task-2-remote-pci.md`

**Interfaces:**
- Consumes: Task 1 pytest contract.
- Produces: C++ self-test modes `--self-test remote-cmd-frame` and `--self-test log-contract`.

- [ ] **Step 1: Implement only enough C++ for the two no-hardware tests**

The first implementation includes:

```cpp
// Substantial AMD/TinyGPU mechanics in this file are derived from tinygrad
// under the MIT license. Source provenance is recorded beside each ported
// section. See <tinygrad-checkout>/LICENSE.
```

Implement:

- `enum class RemoteCmd : uint8_t` with command numbers matching tinygrad `RemoteCmd` order;
- a packed request builder equivalent to `<BIIQQQ` for `cmd, dev_id, bar, arg0, arg1, arg2`;
- `--self-test remote-cmd-frame` that builds a `MAP_SYSMEM_FD` frame and checks byte count, command id, `dev_id`, and little-endian arguments;
- `--self-test log-contract` that prints every required log field name;
- `main` argument parsing that returns nonzero for unknown self-tests.

- [ ] **Step 2: Supervisor runs GREEN command**

Run:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Expected after Task 2: 2 passed.

- [ ] **Step 3: Review and commit**

Reviewer checks provenance, no tinygrad runtime dependency, no libusb path, and no broad runtime abstraction. Supervisor commits after review and green pytest.

---

### Task 3: TinyGPU.app connection and AMD discovery smoke

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Report: `.superpowers/swarm/reports/c0b-task-3-discovery.md`

**Interfaces:**
- Consumes: `RemoteCmd` request/response helpers from Task 2.
- Produces: `--discovery-smoke` mode that connects to TinyGPU.app, maps BAR0/BAR2/BAR5, reads discovery facts, and logs `runtime_substrate`, `pci_id`, `arch` when known, BAR sizes, and precise failure stage.

- [ ] **Step 1: Add a RED pytest or self-test expectation**

Extend `tests/test_native_amdev_transfer_contract.py` with a no-hardware assertion for argument support:

```python
def test_discovery_smoke_help_is_declared(tmp_path):
    exe = compile_probe(tmp_path)
    result = subprocess.run([str(exe), "--help"], cwd=ROOT, check=True, text=True, capture_output=True)
    assert "--discovery-smoke" in result.stdout
    assert "--transfer-proof" in result.stdout
```

Run the focused pytest and confirm it fails before adding `--help` output.

- [ ] **Step 2: Implement discovery mode**

Port only the TinyGPU.app connection and discovery operations needed for BAR mapping and IP discovery. Failure must log one of: `tinygpu_socket_connect`, `tinygpu_server_launch`, `remote_cmd`, `bar_map`, `pci_config`, or `amd_discovery`.

- [ ] **Step 3: Supervisor runs focused commands**

Run pytest, then run the discovery smoke command recorded in `validation-commands.md`. The discovery smoke may pass or block with exact failure text; it must not claim transfer success.

---

### Task 4: VM/sysmem mapping port

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Report: `.superpowers/swarm/reports/c0b-task-4-vm-sysmem.md`

**Interfaces:**
- Consumes: BAR/MMIO/discovery handles from Task 3.
- Produces: internal functions that allocate/map one VRAM buffer and one or two CPU-visible sysmem buffers through `MAP_SYSMEM_FD`, with GPU virtual addresses and CPU views available to SDMA.

- [ ] **Step 1: Add a RED self-test for page-list parsing**

Add `--self-test sysmem-page-list` and a pytest assertion that the mode passes. The self-test feeds a synthetic `(paddr, size)` list ending with `(0, 0)` and checks page expansion at 4 KiB granularity.

- [ ] **Step 2: Port minimal VM/sysmem logic**

Port the smallest page-list parsing, PTE construction, and mapping scaffolding required by the transfer proof. Keep fixed sizes and one queue-lifetime arena. Do not implement a general allocator.

- [ ] **Step 3: Supervisor runs focused pytest**

Run the native contract pytest. Expected after Task 4: all self-tests pass.

---

### Task 5: SDMA queue and transfer proof integration

**Files:**
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- Modify: `docs/tasks/native-r9700-producer/validation-commands.md`
- Report: `.superpowers/swarm/reports/c0b-task-5-transfer-proof.md`

**Interfaces:**
- Consumes: RemoteCmd transport, discovery facts, VM/sysmem mappings.
- Produces: `--transfer-proof` mode and exact supervisor build/run/log command under `validation-commands.md`.

- [ ] **Step 1: Add RED self-test for SDMA packet encoding**

Add `--self-test sdma-packet-encoding` and a pytest assertion that it passes. The self-test checks a 32-byte linear-copy packet encodes source and destination addresses little-endian and `count == 31`.

- [ ] **Step 2: Port SDMA queue setup and transfer flow**

Implement SDMA queue 0 setup, ring write, BAR2 doorbell, linear-copy packets, completion/fence/timeline polling, and bounded timeout. The transfer flow copies 32 bytes staging -> VRAM -> staging and compares bytes exactly.

- [ ] **Step 3: Add exact hardware validation command**

Add a command to `validation-commands.md` that compiles the probe, writes `logs/c0b-native-amdev-sdma-transfer.log`, includes command line and UTC timestamp, runs `--transfer-proof`, prints wrapper `exit_status`, and exits with the probe status.

- [ ] **Step 4: Supervisor runs focused commands**

Run native contract pytest and the exact hardware transfer command. Passing transfer requires process exit 0, `host_device_transfer_status: pass`, `transfer_byte_count: 32`, and CPU comparison success.

---

### Task 6: Review, C0 ledger update, and handoff

**Files:**
- Modify: `.superpowers/swarm/progress.md`
- Modify: `docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md`
- Modify: `docs/archive/tasks/native-r9700-producer/README.md`
- Modify: `.superpowers/swarm/native-r9700-producer-supervisor.md`
- Report: `.superpowers/swarm/reports/c0b-task-6-review-handoff.md`

**Interfaces:**
- Consumes: Task 5 report and supervisor validation output.
- Produces: final C0B transfer status and next gate decision for C0A kernel proof.

- [ ] **Step 1: Dispatch reviewer**

Reviewer checks correctness, maintainability, architectural fit, simplicity, provenance/MIT notice, no hidden tinygrad dependency, no stale libusb acceptance, and exact validation evidence.

- [ ] **Step 2: Update ledgers**

If transfer passed, mark C0B transfer Done and unblock C0A kernel proof. If transfer failed with a precise hardware blocker, record Blocked with exact stage/log path. Do not mark C0 substrate selected until kernel proof and C0 decision rerun pass.

- [ ] **Step 3: Supervisor verification**

Run:

```sh
git diff --check docs/archive/tasks/native-r9700-producer/README.md docs/archive/tasks/native-r9700-producer/phase-c0a-macos-egpu-runtime-focus.md docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/reports/c0b-task-6-review-handoff.md
```

- [ ] **Step 4: Commit**

Supervisor commits the reviewed/verified C0B state. Push remains the user's responsibility.
