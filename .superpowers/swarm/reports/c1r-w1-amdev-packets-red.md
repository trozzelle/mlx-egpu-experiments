# C1R-W1 AMDev packet RED contracts

## Selectors

- `tests/native_r9700/test_amdev_packets.py::test_sdma_copy_words_preserve_linear_opcode_length_addresses_and_fence`
- `tests/native_r9700/test_amdev_packets.py::test_pm4_dispatch_words_preserve_the_frozen_59_dword_c0a25_stream`

## Public contract

- `native_r9700::build_sdma_copy_words(0x0102030405060708, 0x1122334455667788, 32, 0xa1b2c3d4)` emits the exact 11-word C0 SDMA linear-copy and fence stream: opcode `0x00000001`, count-minus-one `0x0000001f`, low/high source and destination words, and a zero-address fence carrying `0xa1b2c3d4`.
- `native_r9700::build_pm4_dispatch_words(0x0000200000005000, 0x0000200000006000, 0x000020000000f010)` emits the exact 59-word C0A25 PM4 stream, including all packet-3 headers, register payloads, dispatch dimensions, partial flush, and timeline release fence.
- The generated C++ probe links only `native_r9700/amdev_packets.cpp` and includes only `amdev_packets.h`; it performs no TinyGPU socket, BAR, buffer, or hardware operation.

The selectors were not executed because the assignment explicitly prohibits running commands. They are RED: `native_r9700/amdev_packets.h` and `native_r9700/amdev_packets.cpp` do not yet exist, so the test's compile assertion will fail until the pure encoders are extracted.
