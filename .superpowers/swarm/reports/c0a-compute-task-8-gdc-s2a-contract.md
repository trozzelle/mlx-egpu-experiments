# C0A Compute Task 8 GDC/S2A RED Contract

## Changed files
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0a-compute-task-8-gdc-s2a-contract.md`

## New expected self-test lines
Added to `EXPECTED_COMPUTE_DOORBELL_DELIVERY_LINES` immediately before `status: pass`:

```text
compute_doorbell_route_readback_field: compute_doorbell_route_readback
compute_doorbell_route_classification_field: compute_doorbell_route_classification
route_readback_registers: regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN,regRCC_DEV0_EPF2_STRAP2,regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL,regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL
route_expected_entry0_ctrl: 0x30000007
route_expected_entry3_ctrl: 0x3000000d
route_classification_values: gdc_s2a_route_readback_matches,gdc_s2a_route_readback_mismatch,gdc_s2a_route_readback_unclassified
```

## RED reason
This is RED before the C++ implementation because the Python contract now requires the `compute-doorbell-delivery` self-test output to report the GDC/S2A route-readback field names, register list, expected raw entry control values, and route classification values. The C++ self-test has not been updated in this slice, so the focused contract test should fail until that instrumentation output is implemented.

## Supervisor RED command
```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py::test_compute_doorbell_delivery_self_test_reports_diagnostic_contract -v
```

No validation commands were run by this agent.
