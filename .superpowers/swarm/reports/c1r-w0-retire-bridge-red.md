# C1R-W0-1 legacy bridge RED contract

## Changed test
`test_native_prefill_proof_reports_legacy_proof_unavailable_without_primitive_bridge`

## Why it is RED before cutover
With `NATIVE_R9700_C1_PRIMITIVE_BRIDGE` absent, the active `--native-prefill-proof` diagnostic currently follows its archived-source fallback rather than reporting the explicit unavailable-legacy-proof outcome. The test observes only the runner command's exit status and structured log output, requiring a nonzero exit, `failure_stage: legacy_proof_unavailable`, and no `native_prefill_acceptance: pass` claim. It therefore fails until the fallback is removed and the diagnostic fails loudly.

## Supervisor commands

RED (before production cutover):

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k legacy_proof_unavailable -q
```

GREEN (after production cutover):

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k legacy_proof_unavailable -q
```
