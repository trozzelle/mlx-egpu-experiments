# C1R-6a fp16-to-fp32 primitive review

Reviewer: `agent://C1R6Review`

Initial verdict: CHANGES_REQUIRED
Final verdict: APPROVE (`agent://C1R6ReReview`, no findings)

Finding:
- P2 `tests/native_r9700/test_runtime_contract.py`: the embedded-kernel SHA regression hashed only `kC1PrimitiveKernelText`, so `kC1Fp16ToFp32KernelSha256` could drift from `kC1Fp16ToFp32KernelText` while wrapper/fake-bridge tests still passed.

Fix:
- Changed `test_primitive_kernel_sha_matches_embedded_text` to iterate both embedded kernel symbols:
  - `kC1PrimitiveKernelText` -> `FIRST_PRIMITIVE_SHA256`
  - `kC1Fp16ToFp32KernelText` -> `FP16_TO_FP32_PRIMITIVE_SHA256`
- Preserved the 64-byte length assertion for both kernels.

Verification:
- First looped-regex run failed because the regex was over-escaped and matched no kernel symbol.
- Regex corrected to `rf"{symbol}\s*=\s*\{{\{{(?P<body>.*?)\}}\}};"`.
- `${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_kernel_sha_matches_embedded_text -q` -> 1 passed in 0.03s.
- `${PY} -m pytest tests/native_r9700/test_runtime_contract.py -q` -> 19 passed in 18.90s.
- `${PY} -m pytest tests/native_r9700 -q` -> 138 passed, 2 warnings in 23.35s.
- `${PY} -m pytest tests -q` -> 178 passed, 2 warnings in 57.04s.
- `git diff --check native_r9700/runtime.h native_r9700/runtime.cpp native_r9700/runner.cpp native_r9700/c1_primitive_bridge.cpp tests/native_r9700/test_runtime_contract.py docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/reports/c1r-task-6-fp16-cast.md .superpowers/swarm/reports/c1r-task-6-review.md` -> exit 0.

Status: Fixed; focused re-review approved.
