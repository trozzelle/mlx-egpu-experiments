# Qwen stage executor

`plan_qwen_text_layer` is a metadata-only, fail-closed boundary. It accepts only caller-marked validated text tokens, one layer index, a validated `QwenAffineBinding`, and a borrowed 64-entry `QwenHybridCacheMetadata` view. It neither reads affine payloads nor cache bytes, allocates device buffers, performs model math, or dispatches a runner.

## Runtime order and assets

Each plan validates all 64 cache entries in layer-index order before selecting a stage:

| Cache entry | Selected assets, in launch order |
| --- | --- |
| `ArraysCache` at every layer other than `3, 7, ..., 63` | `qwen_affine4_linear` (`kAffine4Linear`), then `qwen_deltanet_state` (`kDeltaNetState`) |
| `KVCache` at layers `3, 7, ..., 63` | `qwen_affine4_linear` (`kAffine4Linear`), then `qwen_full_attention` (`kFullAttention`) |

The returned plan borrows the exact affine raw-window binding and selected cache-entry metadata, including its opaque spill-state identity. It retains state-buffer GPU VA/size metadata without copying or interpreting contents. Cache entries require an opaque spill-state identity and two nonzero resident-state buffers; full-attention entries additionally require an offset exactly equal to the cache committed position. This is compatible with later persistent resident-stage wiring while preventing CPU reupload or cache-byte fabrication at the planning boundary.

## Rejection boundary

Planning rejects unvalidated/multimodal input before cache or asset selection; invalid layer indices; mismatched affine layer metadata; invalid affine-4/group-64 byte windows; non-64-entry or incorrectly interleaved cache metadata; missing spill metadata; misplaced KV offsets; and missing/invalid resident state buffers.

The current reviewed Qwen HSA images are not dispatched or admitted here. In particular, the DeltaNet source ABI is newer than its existing image/manifest, so resident execution remains fail-closed until the image is regenerated and admitted by its integration owner.

## Focused test

`tests/native_r9700/test_qwen_layer_executor_contract.py` compiles a narrow C++ probe and covers exact ArraysCache/DeltaNet and KVCache/full-attention selection plus multimodal, cache-order, resident-buffer, and affine-binding rejection. Per instruction, no commands were run in this wave.
