# F1 multi-PDB1 Task 4: real process smoke

**Status:** Done  
**Owner:** Main  
**Hardware:** AMD Radeon AI PRO R9700, TinyGPU.app/APLRemotePCIDevice/PCIIface

## Implemented prerequisite

- Large- and small-BAR resident VA windows cover allocator-visible VRAM within one PDB2 entry.
- Small-BAR page-table pages remain in the BAR-visible pool; resident payloads use full VRAM through SDMA.
- DynamicPageTable owns PDB0/PTB pages across PDB1 slots with child-before-parent cleanup.
- Resident VA starts at the second PDB1 entry, preserving C0's fixed staging PDB0/PTB mapping in the first entry.
- Obsolete boundary-guard allocations/checks were removed from resident HSA planning/preparation.

## Real smoke evidence

Exact command: `native_r9700.native_worker --smoke-load-unload-reload`, accepted model, `prompt-128`, `--samples 10`, explicit branch runner.

Observed `logs/f1-persistent-worker/process-smoke/result.json`:

- `status=pass`, `exit_status=0`, `sample_count=10`
- operations: `LoadModel`, ten `Prefill`, `UnloadModel`, `LoadModel`, `UnloadModel`
- `load_preparation_count=2`, `prefill_count=10`, `warm_prefill_weight_reload_count=0`
- every sample: `S=129`, `N=128`, `route=native_producer`, `accepted_cache=true`, empty fallback, `producer_kind=r9700_native`, full-layer/acceptance pass, `kernel_count=640`, `block_tokens=32`, `block_count=4`, `exit_status=0`

## Acceptance gate

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_vram_layout.py \
  tests/native_r9700/test_dynamic_page_table_contract.py \
  tests/native_r9700/test_resident_memory_contract.py \
  tests/native_r9700/test_runtime_vram_contract.py \
  tests/native_r9700/test_native_resource_client.py \
  tests/native_r9700/test_native_resource_worker_contract.py \
  tests/native_r9700/test_model_service.py -q
# 125 passed
```

The multi-PDB1 prerequisite is accepted. F1 warm benchmark promotion remains the next gate.
