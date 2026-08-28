# F1 multi-PDB1 Task 3: multi-GiB resident proof

**Status:** Done  
**Owner:** Main  
**Scope:** No-hardware `ResidentMemory` integration proof.

The focused C++ probe reserves three named resident ranges totaling exactly 4 GiB (`1 GiB + 4 KiB`, `2 GiB`, `1 GiB - 4 KiB`) without materializing host payloads. It verifies page-aligned contiguous GPU VAs across multiple PDB1 boundaries, one map callback per 4 KiB page, complete unmap, zero remaining mappings, and reuse of the first GPU VA/physical page.

```sh
${PY} -m pytest \
  tests/native_r9700/test_resident_memory_contract.py::test_resident_memory_maps_and_reclaims_four_gibibytes_without_host_payloads -q
# 1 passed
```
