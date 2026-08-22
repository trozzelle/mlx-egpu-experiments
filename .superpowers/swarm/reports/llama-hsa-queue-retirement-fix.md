# Llama HSA Queue Retirement Fix

## Scope

- `native_r9700/amdev_session.cpp`
- This report

No hardware run, test command, probe, build, formatter, or git command was run.

## Source-grounded retirement

`TerminalComputeQueue0Retirement` centralizes terminal queue-0 retirement around the existing included C0 helper:

```cpp
reset_compute_queue0(const RemoteClient&, const DiscoveryLog&, std::string*)
```

That helper selects MEC pipe 0 / queue 0, reads `regCP_HQD_ACTIVE`, dequeues and resets only an active HQD, polls until the active bit clears, and restores GRBM selection. The new lifecycle wrapper neither reprograms HQD registers nor writes the active bit.

Its state is armed only after `setup_compute_ring0` returns successfully. Retirement records a failed attempt as terminal, so a later call returns the saved failure rather than issuing a fresh hardware retry.

## Lifecycle coverage

The wrapper is used by all three native C0 compute-ring setup callers:

- `run_vram_smoke`
- `run_llama_embed_smoke`
- `run_resident_kernel_dispatch`

For the VRAM and Llama smokes, every post-setup failure path retires queue 0 before `release_resident()`. Normal completion also retires before resident/PTE cleanup. A retirement failure returns `failure_stage: compute_queue_retirement`, preserves the prior failure text by prefixing it to the retirement detail, and deliberately skips `release_resident()` so resident mappings remain quarantined.

For resident-kernel dispatch, every post-setup error exit and normal return retires queue 0 before scope teardown releases the fixed mappings.

## Static source audit

Source inspection confirms each of the three `setup_compute_ring0` success paths is immediately followed by `compute_queue_retirement.arm()`. All subsequent dispatch, fence, readback, and normal-completion paths reach either `fail_after_compute_queue_setup` or an explicit terminal `retire()` before resident release or function return.
