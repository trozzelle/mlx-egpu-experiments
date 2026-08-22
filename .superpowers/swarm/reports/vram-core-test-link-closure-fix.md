# VRAM Core Native Test Link Closure Fix

Updated the legacy no-hardware native C++ compile helpers in `test_runtime_lifecycle.py`, `test_runtime_protocol.py`, `test_resident_kernel_dispatch_contract.py`, `test_device_memory_contract.py`, and `test_layer0_executor_contract.py` to link the full direct VRAM closure: `vram_layout.cpp`, `vram_allocator.cpp`, `dynamic_page_table.cpp`, `resident_memory.cpp`, and `vram_smoke_asset.cpp`. Runner and session probes retain their required `kernel_catalog.cpp` linkage.

The runtime-protocol transfer-bridge helper augments the runtime-derived source list only with closure members absent from that list, so a later production command update cannot create duplicate C++ translation units.

No test execution, compiler invocation, hardware access, or `--vram-smoke` invocation was performed, as required.
