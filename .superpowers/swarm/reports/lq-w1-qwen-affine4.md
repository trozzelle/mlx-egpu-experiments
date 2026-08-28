# LQ-W1 Qwen affine4 linear source

`qwen_affine4_linear` is a text-only HSA source primitive for one fp16 input vector and a bounded affine4 weight window. Launch one workgroup per output row; lane 0 decodes each packed 4-bit weight on device, applies `(q * scale + bias)` for its group of 64 input elements, accumulates in fp32, and writes one fp16 output. A nonempty input extent must be divisible by 64.

## Kernel argument ABI

| Index | Argument | Type | Meaning |
| --- | --- | --- | --- |
| 0 | `input` | `const unsigned short*` | fp16 input vector with `input_features` elements |
| 1 | `packed_weight` | `const unsigned char*` | raw two-nibbles-per-byte affine4 matrix, output-major |
| 2 | `scales` | `const unsigned short*` | fp16 scale per output-major group of 64 inputs |
| 3 | `biases` | `const unsigned short*` | fp16 affine bias per output-major group of 64 inputs |
| 4 | `output` | `unsigned short*` | fp16 output vector with `output_features` elements |
| 5 | `input_features` | `unsigned long long` | requested input-column extent |
| 6 | `output_features` | `unsigned long long` | requested output-row extent |
| 7 | `input_capacity_elements` | `unsigned long long` | validated readable fp16 input capacity |
| 8 | `packed_weight_capacity_bytes` | `unsigned long long` | validated readable packed affine4 capacity |
| 9 | `affine_group_capacity` | `unsigned long long` | validated readable scale/bias group capacity |
| 10 | `output_capacity_elements` | `unsigned long long` | validated writable fp16 output capacity |

The ABI is five 64-bit pointers followed by six 64-bit extents (88 bytes). Before
dereferencing a pointer, the kernel validates input/output capacities, checked
matrix/group products, and packed-byte rounding. It does not accept a group-size
argument: group size is the fixed Qwen affine4 contract of 64.

## Supervisor command

```sh
 ${PY} -m pytest -q tests/native_r9700/test_qwen_affine4_source.py
```
