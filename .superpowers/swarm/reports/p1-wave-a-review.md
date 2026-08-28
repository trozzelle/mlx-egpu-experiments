# P1 Wave A review gate

## Review inputs

- Code/architecture review: `agent://P1CodeReview`
- Focused security review: `agent://P1SecurityReview-2`
- Frozen contract: `.superpowers/swarm/reports/p1-abi-freeze.md`
**Historical execution/provenance boundary:** This report records source changes executed/reviewed in the former external TinyGPU checkout `<former-tinygpu-worktree>` on branch `feature/r9700-device-owner`; original changed-file paths below retain their former locations as provenance only and never authorize edits.
**Current source authority and reproduction root:** Active TinyGPU source/build/task authority is `<repo-root>/tinygpu` on branch `feature/r9700-products-wave-a`; current commands below run from this root and write binaries under `<repo-root>/tinygpu/build/`.

## Supervisor evaluation

Every Critical/Important finding was checked against current source, the DriverKit 25.5 headers, the frozen ABI report, or pinned/local source evidence. No Critical/Important finding was rejected.

| Finding | Severity | Decision | Required disposition |
|---|---|---|---|
| P1-COLD-001 | Critical | Valid | Current functions inspect warm predicates rather than establish cold firmware/IP ownership. No approved firmware bundle exists in the checkout, so TinyGPU must remain non-ready and task-set-2 cold acceptance remains Blocked until provenance-bound firmware/transition inputs exist. Never accept pre-warmed state. |
| P1-COLD-002 | Critical | Valid | Decode `MMMC_VM_FB_LOCATION_*` as 24-bit fields shifted by 24, validate the expanded range, then program/read back the derived aperture. |
| P1-COLD-003 | Important | Valid | Use source-grounded MP1 C2PMSG_90 offset 666 and a bounded delayed poll; 154 is unsupported. |
| P1-COLD-004 | Important | Valid | Preserve the generic frozen failure enum while carrying the exact private cold-stage label through bounded redacted health text/evidence. |
| P1A-UC-001 / P1-ABI-001 | Important | Valid duplicate | Validate the complete typed health request and inference-only client scope before provider access. |
| P1A-TOKEN-001 / P1-BUFFER-001 | Important | Valid duplicate | XOR epoch/nonce mixing admits deterministic cross-epoch token aliasing. Use a collision-free full epoch/slot/generation/kind binding and add the explicit replay contract. |
| P1A-CLIENT-001 | Important | Valid | The validated `--service` argument is ignored by `OpenDriver`; bind registry matching to the exact requested DriverKit service identity and reject ambiguity. |
| P1A-ENT-001 / P1-PACKAGE-002 | Important | Valid duplicate | Give the conformance client a dedicated least-privilege entitlement file; do not grant system-extension install authority. |
| P1A-LIFE-001 | Important | Valid | Extension activation is not device readiness. App/CLI status must say health is unchecked unless the structured health contract passes. |
| P1-PACKAGE-001 | Important | Valid | Task-set-2-owned packaging must define distinct inference/recovery/diagnostic user-client classes/entitlements; later selectors may remain explicitly unsupported. |
| P1-EVIDENCE-001 | Minor | Valid | Associate health-derived status with the health selector in the final record. |
| P1-EVIDENCE-002 | Minor in review; promotion-blocking in observed smoke | Valid | The exact preinstall smoke printed to stderr but did not create the requested log because its parent directory was absent. Log directory/open/write/close failures must produce nonzero exit and never lose required evidence. |

## Observed supervisor evidence

- Host cold coordinator contract: pass.
- Host resource-table contract: pass after compile fix.
- Unsigned `TGPUConformanceClient` build: pass.
- Unsigned `TinyGPUDriver` build: pass after adapting DriverKit 25.5 void MMIO signatures.
- Signed client build: blocked by missing selected development team/profile.
- Preinstall direct-client smoke: fail closed (`status=17`, `failure_stage=1`, `exit_status=1`) but failed to create the requested log path; this confirms P1-EVIDENCE-002.

## Gate result

**Needs fixes.** No source wave may be accepted or checkpointed until all Critical/Important findings are fixed and re-reviewed. Task-set-2 cold hardware acceptance will remain explicitly Blocked after source fixes because the required provenance-bound firmware/transition path is unavailable; this is not replaceable with warm-state observation.
