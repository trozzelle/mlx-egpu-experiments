# C1R-2 runtime hardware proof

Status: **Done**

## Decision

`native_r9700::RuntimeSession` now exposes `kernel_proof()` and `native_r9700/runner.cpp --kernel-proof` calls it. The method wraps the frozen C0A25 probe source instead of mutating it:

- default path: build `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` to `logs/native-r9700-c0a25-probe`, then run `--kernel-proof`;
- no-hardware test path: run executable from `NATIVE_R9700_C0_PROBE`;
- every run emits `producer_kind: hardware_probe`, C0 pass/fail fields, `kernel_proof_wrapper_status`, `wrapper_exit_status`, and writes a timestamped log under `logs/`.

This is a reusable C1R boundary only for the known C0A25 minimal `uint32_t add-one` proof. Llama model-forward buffers/kernels remain C1R-3 through C1R-7.

## Files changed

- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `tests/native_r9700/test_runtime_contract.py`

## Verification

Focused no-hardware contract:

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -q
```

Result: `8 passed in 5.99s`.

Real hardware proof:

```sh
mkdir -p build/native-r9700-runtime && \
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && \
build/native-r9700-runtime/native_r9700_runner --kernel-proof
```

Result: exit `0`; wrapper log `logs/c1-runner-kernel-proof-2026-08-19T11:09:34Z.log`.

Observed hardware pass markers:

- `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`
- `pci_id: 1002:7551`
- `arch: gfx1201`
- `kernel_blob_load_status: pass`
- `kernarg_write_status: pass`
- `sdma_h2d_status: pass`
- `sdma_d2h_status: pass`
- `kernel_launch_status: pass`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `exit_status: 0`
- `kernel_proof_wrapper_status: pass`
- `wrapper_exit_status: 0`
