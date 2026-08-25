# LN-2 recovery-state audit

## Scope and evidence boundary

This is a read-only audit of the checked-in recovery scripts, retained hardware logs, and the documented TinyGPU discovery/recovery sequence. No recovery, build, test, reset, or device command was issued for this audit.

The current assertion text, `discovery signatures mismatch`, is the reported post-reset state. There is no scoped retained console log for that particular `tinygpu_amdev_full_boot.py` invocation; consequently this report does not invent its observed discovery bytes or a hardware root cause.

## Meaning of `discovery signatures mismatch`

This is an **AMDev IP-discovery validation failure**, not an RMSNorm numerical result and not an HSA-image admission result.

The documented initialization sequence is:

1. `APLRemotePCIDevice` connects to TinyGPU.app's Unix-socket RPC service and presents the device as `pcibus='usb4'` (`docs/archive/tasks/native-r9700-producer/macos-tinygpu-abi-notes.md:31-47`).
2. AMDev maps BAR0 (VRAM), BAR2 (doorbells), and BAR5 (MMIO), establishes its AMD boot policy, and then performs software/memory initialization (`macos-tinygpu-abi-notes.md:51-58`).
3. During discovery, AMDev reads VRAM size from register `0xde3`, verifies that BAR0 is sufficient, reads the IP-discovery table from VRAM or through MMIO-backed `_read_vram`, and validates that table's discovery signatures (`macos-tinygpu-abi-notes.md:54`).

Therefore the assertion means the bytes AMDev obtained for the IP-discovery table after reset did **not** satisfy the expected discovery signatures. It prevents AMDev from accepting the IP table and proceeding with normal IP-block identification/initialization. It does **not** establish which discovery byte was wrong, whether the invalid read originated in VRAM/MMIO transport or device state, or that the R9700 is absent: those facts are not captured by the scoped sources.

The retained healthy C0 log demonstrates the contrasting successful state: it reports `arch_discovery_status: discovered_from_ip_table`, `gfx1201`, GC 12.0.1, MMHUB 4.1.0, and SDMA 7.0.1 before completing the proof (`logs/c1-runner-kernel-proof-2026-08-23T02:57:12Z.log:13-30,147-153`). The later retained C0 log still completed discovery but then failed during compute-ring setup because `regCP_HQD_ACTIVE` would not become active (`logs/c1-runner-kernel-proof-2026-08-23T04:06:03Z.log:13-30,141-153`). The newly reported signature failure is thus a further, earlier initialization blocker after the reset; it is not evidence that the earlier discovery had been invalid.

## Why this does not invalidate the epsilon-arithmetic asset

The original RMSNorm observation is separately recorded as `trace_nonfinite`, with `trace output contains NaN or infinity`, the original RMSNorm kernargs, three resident-buffer bindings, and the stage PM4 resources (`logs/ln-2-native/layer0-token0-normalized.failure.json:1`). The all-zero-input/unit-scale variant also produced the same `trace_nonfinite` result (`logs/ln-2-zero-scale/layer0-token0-normalized.failure.json:1`). Those are executions that reached the trace result check; they are the original RMSNorm nonfinite behavior.

By contrast, the epsilon-arithmetic probe is a diagnostic asset intended to isolate `1 / sqrt(epsilon)` from the original reduction and input/scale multiplication. Its documented expected output is repeated fp16 `0x5cf1` for epsilon `1e-5`, and its contract states that it is trace-only and cannot affect the persistent/production asset path (`ln-2-rmsnorm-epsilon-arithmetic-probe.md:5-32`). A timeline timeout during that probe supplies no output payload with which to apply the probe's exact-output discriminator. The subsequent discovery-signature assertion occurs during AMDev initialization, before an IP table is accepted and before any selected HSA stage can be prepared or dispatched.

Accordingly:

- The probe timeout is compatible with the already documented queue-health failure domain; it is **not a successful or failed arithmetic comparison**.
- The post-reset signature assertion is a new substrate-initialization blocker; it is **not evidence that the generated epsilon-arithmetic image is invalid**.
- Neither event resolves the original RMSNorm `trace_nonfinite` observation. In particular, native acceptance of neither the original asset nor the diagnostic asset can be inferred.

A useful contrast is the retained zero-store diagnostic artifact: with the same constrained zero-input/unit-scale trace shape, it records `finite_count:2048`, epsilon `0.000010`, and the expected zero-store kernel identity (`logs/ln-2-zero-store/layer0-token0-normalized/layer0-token0-normalized.json:1`). That is evidence for that zero-store diagnostic's completed output path only; it does not convert the original arithmetic nonfinite result into an accepted RMSNorm result.

## Source-grounded non-destructive actions already available

All available recovery actions have already been exercised or documented; none is an untried register-poke recommendation.

| Action | Evidence | Result / limit |
| --- | --- | --- |
| TinyGPU RPC reset | `logs/tinygpu_reset.py:1-5` instantiates `APLRemotePCIDevice("AMD", "usb4")` and calls `device.reset()`. `RESET` is part of the documented TinyGPU RPC client ABI (`macos-tinygpu-abi-notes.md:38-45`). | The handoff records that it completed but did not clear the HQD (`2026-08-22-native-producer-handoff.md:185-190`). The current reported signature failure followed this reset, so it is not a proof of restored AMDev discoverability. |
| Full AMDev boot/finalization | `logs/tinygpu_amdev_full_boot.py:1-12` constructs `AMDev(pci)`, then calls `am.fini()` and closes the socket. | The handoff records completion without clearing the HQD. It is now the path that reports the signature assertion, so it cannot presently advance to queue recovery. |
| Live AMDev boot/finalization | `logs/tinygpu_amdev_boot_live.py:1-15` constructs AMDev, retains it only while paused, then finalizes/closes. | The handoff records failure in Tinygrad `_dequeue_hqds()` with a 10-second HQD dequeue timeout (`2026-08-22-native-producer-handoff.md:179-190`). |
| Native C0 control/health gate | The latest retained native control reaches queue setup, then fails because `regCP_HQD_ACTIVE` does not reach the required state (`logs/c1-runner-kernel-proof-2026-08-23T04:06:03Z.log:141-153`). | This is a health check, not a recovery mechanism. It must pass in full before any model diagnostic is resumed. |
| Gfx12 non-destructive queue sequence | The handoff's Tinygrad-source review identifies the complete available sequence: dequeue HQDs/wait for `HQD_ACTIVE=0`; reset/configure MEC; replay RS64 program start; enable MEC (`2026-08-22-native-producer-handoff.md:191-195`). | The handoff explicitly states there is no additional source-grounded non-destructive register poke once `HQD_ACTIVE` refuses to clear (`:197`). Do not invent an engine reset, register write, or retry loop. |

## External intervention required

The physical TinyGPU/R9700 command-queue and, now, accepted discovery-table state must be restored outside the repository's client-side controls. The documentation assigns privileged PCI/BAR/config/reset/reBAR operation, host-side transport, and device mediation to TinyGPU.app and its installed extension; tinygrad exposes only the client ABI (`macos-tinygpu-abi-notes.md:40-47`). The handoff consequently requires an external device/app recovery or a successful external queue dequeue/reset that restores a healthy C0 control (`2026-08-22-native-producer-handoff.md:199-201`).

External recovery is sufficient for resumption only when it restores both of these observable preconditions:

1. AMDev accepts the IP-discovery table rather than asserting a signature mismatch.
2. The documented C0 health gate again records all of `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `exit_status: 0` (`2026-08-22-native-producer-handoff.md:203-210`).

Until then, no further RMSNorm probe—especially not the epsilon-arithmetic discriminator—can be treated as a meaningful asset result. The documented resume sequence deliberately places the C0 health gate before resident-VRAM smoke and native prefill work (`2026-08-22-native-producer-handoff.md:212-232`).
