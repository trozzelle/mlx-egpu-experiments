# Q1 native shape map

**Status:** Read-only Q1 task-set-5 handoff for F6; contract/oracle evidence only.
**Owner:** `Q1ShapeMap`.
**Scope:** The fixed Qwen3.8-27B text graph, its quantized tensor layouts, recurrent/full-attention state, and the exact boundaries at which F2/F3/F4 contracts can be reused or must be adapted.

No production file, fixture, catalog, Kernel Pack, or validation command was changed by this lane. This report makes no native-execution, native-acceptance, or performance claim. Q1 artifacts remain `producer_kind=cpu_reference` and `native_evidence=false`.

## 1. Immutable identity and authority

The only model identity admitted by Q1 is:

| Field | Exact value |
|---|---|
| Model snapshot | `mlx-community/Qwen3.8-27B-4bit`, revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| MLX-VLM source | `Blaizzy/mlx-vlm`, revision `2b31570bdee86e2cdeea049761885aeed524a98c` (`mlx-vlm-qwen3-5`) |
| mlx-lm cache source | `ml-explore/mlx-lm`, revision `e2f2fb2aef987f86878d17638446183cffe21fe4` (`mlx-lm-cache`) |
| Model fingerprint | `4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371` |
| Base model revision | `unavailable_in_pinned_conversion_metadata` |
| Promotion state | `blocked_base_model_revision` |

The fingerprint, provenance marker, local shard digests, and no-fallback rule are frozen in `.superpowers/swarm/reports/q1-identity-freeze.md:36-145`; the header-only inventory counts and `inventory_sha256=508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4` are frozen in `.superpowers/swarm/reports/q1-tensor-inventory.md:40-62` and the identity report `:191-271`.

The language boundary is `model.language_model`/`LanguageModel`, not the VLM image/video merge path. Accepted token inputs are integer text IDs only. Q1 rejects `vision_start_token_id=248053`, `vision_end_token_id=248054`, `image_token_id=248056`, and `video_token_id=248057`; it does not strip or remap them. This is the pinned text-only rule in `.superpowers/swarm/reports/q1-identity-freeze.md:147-178` and `native_r9700/qwen_text_adapter.py:168-178`.

### Classification vocabulary

* `exact_shared`: the existing contract can be reused without changing operation shape, dtype, packing, state ownership, or acceptance semantics.
* `adaptable_new_shape_or_packing`: the operation family/algorithm is reusable only after a new declared shape, dtype, physical pack, numerical policy, and evidence record. It is not an alias for an existing Llama family.
* `qwen_specific`: the operation or state semantics are owned by the Qwen3.5-family graph and must have a Qwen-specific source, ABI, and oracle boundary.

No Qwen tensor-compute row below is `exact_shared`: Qwen is BF16 plus affine-4bit with 5120-wide hidden states, while the frozen F2/F3/F4 product controls are Llama FP16 families and Llama cache/attention geometry. `exact_shared` is used below only for the generic metadata/evidence and bounded-span shells, not for a Qwen math kernel.

## 2. Fixed Qwen geometry

All dimensions below are model-config values, not inferred runtime values. `B` is batch, `S` is the current token block, and `N` is the committed prefix position (`N=S-1` for the Q1 capture boundary).

| Quantity | Exact value / shape | Dtype and source |
|---|---:|---|
| Text hidden width `H` | `5120` | Model text dtype `bfloat16` |
| MLP width `I` | `17408` | Model text dtype `bfloat16` |
| Vocabulary | `248320` | Token IDs are integer; embedding/output activations are BF16 |
| Layers | `64` | 48 `linear_attention`, 16 `full_attention` |
| Full-attention Q heads / KV heads | `24 / 4` | GQA ratio `6` Q heads per KV head |
| Full-attention head dimension | `256` | Q/K/V cache leaves are BF16 |
| Full-attention interval | every fourth layer | full layers `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]` |
| DeltaNet key heads / value heads | `16 / 48` | grouped recurrence ratio `3` value heads per key head |
| DeltaNet key/value head dimensions | `d_k=128`, `d_v=128` | recurrent state is FP32; transient q/k/v are BF16 |
| DeltaNet convolution width | `4` | retained ring has `K-1=3` rows |
| DeltaNet mixed-QKV width | `2(16*128)+(48*128)=10240` | BF16 |
| DeltaNet value width | `48*128=6144` | BF16 transient/output before `out_proj` |
| Max positions | `262144` | `rope_theta=10000000`, partial rotary factor `0.25`, mRoPE sections `[11,11,10]` |
| Norm epsilon | `1e-6` | Qwen-specific; not the Llama `1e-5` control |
| Quantization | `mode=affine`, `bits=4`, `group_size=64` | `.weight=U32`, `.scales=BF16`, `.biases=BF16` in the MLX safetensors headers |

The exact source values and layer list are in the pinned model `config.json`, the adapter constants `native_r9700/qwen_text_adapter.py:28-39`, and `.superpowers/swarm/reports/q1-identity-freeze.md:147-163`. The pinned MLX-VLM declarations are `qwen3_5/config.py:61-89` and `qwen3_5/language.py:1357-1760` at the MLX-VLM revision above.

### Affine-4 source representation

For every complete language-model affine stem, the logical source matrix is output-major `[O,I]` and the model header representation is:

* packed weight: `U32[O,I/8]` because 32-bit words hold eight 4-bit values;
* scales: `BF16[O,I/64]`;
* biases: `BF16[O,I/64]`;
* each scale/bias group covers 64 logical input elements;
* a raw upload window is the packed byte stream, two low/high nibbles per byte, with row-major output order; the companion scale/bias windows remain one BF16 value per output-major group.

All Qwen input/output activations in the text model are BF16 unless a state or scalar policy below explicitly says FP32. The header-only inventory is the authority for the six-field records and source spans; it does not decode payloads. This is `.superpowers/swarm/reports/q1-identity-freeze.md:191-271` and `.superpowers/swarm/reports/q1-tensor-inventory.md:40-62`.

The following exact logical matrices and packed header shapes are the minimum F6 map. `SB` means both the scales and biases shape.

| Tensor family | Logical matrix `[O,I]` | Packed `.weight` | `.scales`/`.biases` (`SB`) | Source names |
|---|---:|---:|---:|---|
| Input embedding | `[248320,5120]` | `U32[248320,640]` | `BF16[248320,80]` | `language_model.model.embed_tokens.{weight,scales,biases}` |
| LM head (untied) | `[248320,5120]` | `U32[248320,640]` | `BF16[248320,80]` | `language_model.lm_head.{weight,scales,biases}` |
| DeltaNet `in_proj_qkv` | `[10240,5120]` | `U32[10240,640]` | `BF16[10240,80]` | `layers.i.linear_attn.in_proj_qkv.*` |
| DeltaNet `in_proj_z` | `[6144,5120]` | `U32[6144,640]` | `BF16[6144,80]` | `layers.i.linear_attn.in_proj_z.*` |
| DeltaNet `in_proj_a`, `in_proj_b` | `[48,5120]` each | `U32[48,640]` each | `BF16[48,80]` each | `layers.i.linear_attn.in_proj_a.*`, `.in_proj_b.*` |
| DeltaNet `out_proj` | `[5120,6144]` | `U32[5120,768]` | `BF16[5120,96]` | `layers.i.linear_attn.out_proj.*` |
| Full-attention `q_proj` (query plus gate) | `[12288,5120]` | `U32[12288,640]` | `BF16[12288,80]` | `layers.i.self_attn.q_proj.*` |
| Full-attention `k_proj`, `v_proj` | `[1024,5120]` each | `U32[1024,640]` each | `BF16[1024,80]` each | `layers.i.self_attn.k_proj.*`, `.v_proj.*` |
| Full-attention `o_proj` | `[5120,6144]` | `U32[5120,768]` | `BF16[5120,96]` | `layers.i.self_attn.o_proj.*` |
| MLP `gate_proj`, `up_proj` | `[17408,5120]` each | `U32[17408,640]` each | `BF16[17408,80]` each | `layers.i.mlp.gate_proj.*`, `.up_proj.*` |
| MLP `down_proj` | `[5120,17408]` | `U32[5120,2176]` | `BF16[5120,272]` | `layers.i.mlp.down_proj.*` |

The packed shapes follow the frozen model relation `I/8` and `I/64`; they are not a new kernel-family selection. The first fixture windows are deliberately smaller bounded windows for layer 0 `linear_attn.in_proj_qkv` and layer 3 `self_attn.q_proj`, as frozen in `.superpowers/swarm/reports/q1-identity-freeze.md:323-333` and the oracle RED handoff.

## 3. Operation and state map

### 3.1 Affine-4 GEMM — `adaptable_new_shape_or_packing`

**Exact operation and IO.** A Qwen affine linear consumes an activation `X[B,S,I]` in BF16 and a logical output-major matrix `[O,I]` represented by the U32/BF16/BF16 triplet above. It computes the affine group-64 dequantized dot and returns `Y[B,S,O]` in BF16. The F6 implementation must accumulate in a declared wider type and bind the exact output cast in its Qwen numerical policy; Q1 does not choose that policy. The matrix instances are the exact rows in the packed-shape table: `I=5120` for all hidden-input projections, `I=6144` for `out_proj`, `I=17408` for `down_proj`, and `I=5120` for the embedding/head matrix.

**Current local source boundary.** `native_r9700/kernels/qwen_affine4_linear.cpp` / `.superpowers/swarm/reports/lq-w1-qwen-affine4.md:1-24` exposes one output-major window with raw two-nibble bytes, `input_features`, `output_features`, four capacity extents, and fixed `group_size=64`; one workgroup owns one output row. It rejects non-divisible input extents and all capacity/overflow violations before reads. The local source currently bit-casts `input`, `scales`, and `biases` as IEEE FP16 and stores FP16 (`qwen_affine4_linear.cpp:40-55`). That is not the pinned Qwen representation: the model config and inventory require BF16 activations, BF16 scales, and BF16 biases. Therefore this source ABI is a bounded affine4 shape probe, not an exact Qwen model contract. F6 must make the BF16 decode/output representation explicit in a new reviewed source/packing identity; it must not reinterpret Qwen BF16 words as FP16.

**Reuse boundary.** The dot-product shape is adaptable from the F2 matrix family. The exact F2 first family is `A[M<=128,K=2048] x B[K=2048,N=8192] -> D[M,N]`, FP16 inputs/weights, FP32 accumulation, FP16 output, physical pack `f2-wmma-physical-tile-v1`, and `f2-wmma-64x64-m-tail-v1` (`.superpowers/swarm/reports/f2-contract-freeze.md:98-147`). None of those dimensions, dtypes, or packs are exact for Qwen affine4. The quantized kernel needs a Qwen pack version, affine scale/bias binding, BF16 activation/output policy, and separate shape-family records in P3. It cannot be cataloged as an F2 alias.

**F2/F3/F4 dependency.** F2/G0 supplies the admitted matrix execution/evidence discipline and P3 supplies the closed compatibility/numerics/evidence record, but Q1 does not select the quantized family. F3's Llama projection order (gate/up, down, QKV, O) is a graph precedent only; Qwen uses the matrix rows above and its own graph. F4 is not a dependency for the affine math itself, but F6 promotion consumes the completed F4 architecture and must keep affine stages separate from recurrent state and periodic attention. See `phase-f3-matrix-projection-graph.md:12-36`, `phase-f6-quantized-model-promotion.md:12-36`, and `docs/ROADMAP.md:216-242`.

### 3.2 Recurrent DeltaNet — `qwen_specific`

**Pinned graph.** Every layer other than `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]` is `Qwen3_5GatedDeltaNet`. For `B,S` input, its exact transient widths are:

* `in_proj_qkv`: `[B,S,10240]`, split into Q `[B,S,16,128]`, K `[B,S,16,128]`, and V `[B,S,48,128]` after depthwise causal convolution;
* `in_proj_z`: `[B,S,6144]`, reshaped to `[B,S,48,128]` for the gated output norm;
* `in_proj_a` and `in_proj_b`: `[B,S,48]` each; `A_log` and `dt_bias` are `[48]` FP32 recurrence parameters under `mamba_ssm_dtype="float32"`;
* convolution state: `[B,3,10240]` BF16, retaining the last three mixed-QKV rows from kernel width four;
* depthwise `conv1d.weight`: logical coefficients `[out_channels=10240,kernel=4,in_channels_per_group=1]` in BF16, with `in_channels=out_channels=groups=10240` and no bias; it consumes `[B,S+3,10240]` with the retained three-row prefix and returns `[B,S,10240]` before SiLU;
* recurrent state: `[B,48,128,128]` FP32, one value-head/value-channel/key-channel matrix per value head;
* recurrence output: `[B,S,48,128]` BF16, followed by gated RMSNorm over the 128-wide value head and `out_proj` `[6144 -> 5120]` to `[B,S,5120]` BF16.

The pinned source is `mlx_vlm/models/qwen3_5/language.py:1525-1714` plus `qwen3_5/gated_delta.py:19-39,151-176,179-235,484-534` at the MLX-VLM revision above. The recurrent state is initialized as FP32 `[B,Hv,Dv,Dk]`, decays, applies the key/value delta update, and emits the Q-weighted state; it is not a KV sequence.

**Reuse boundary.** The recurrence, ring state, value/key head ratio, and state ownership are Qwen-specific. F2 can provide a matrix execution substrate for the four affine projections only after a new quantized pack is admitted. F3 projection graph does not define a recurrent state ABI. F4 tiled attention does not apply to this state update and must not be used to infer its cache or positions.

### 3.3 Periodic full attention core — `adaptable_new_shape_or_packing`

**Layer placement and tensor IO.** Full attention runs only at the 16 explicit indices above. For `[B,S,5120]` input:

* `q_proj` returns `[B,S,12288]`, split into queries `[B,S,24,256]` and an output gate `[B,S,24,256]`;
* `k_proj` and `v_proj` return `[B,S,1024]`, reshaped to `[B,4,S,256]` after transpose;
* Q/K are RMS-normalized per 256-wide head, then Qwen partial/mRoPE is applied to Q/K;
* attention maps six query heads to each KV head (`kv_head=query_head/(24/4)`), uses causal positions, and returns context `[B,S,24,256]` BF16;
* context is transposed/flattened to `[B,S,6144]`, multiplied by `sigmoid(gate)`, and passed through `o_proj` `[6144 -> 5120]` to `[B,S,5120]` BF16;
* full-attention cache leaves are K/V `[1,4,N,256]` BF16, with `KVCache.offset=N`.

The pinned implementation is `mlx_vlm/models/qwen3_5/language.py:1357-1508`; the exact model geometry and layer schedule are also in `q1-identity-freeze.md:156-189`.

**Current local source boundary.** `native_r9700/kernels/qwen_full_attention.cpp:1-95` consumes byte-addressed BF16 query/K/V buffers, `query_heads`, `kv_heads`, `head_dimension`, `query_length`, `key_length`, `position`, and bounded K/V capacities. It performs causal dot products with FP32 score/max/normalizer/context accumulators and writes BF16 context; it does not materialize a score/probability tensor and does not own K/V capture. The source maps exactly the Qwen `24/4=6` grouping when called with the model values. It is an operation boundary, not a promoted F4 asset.

**Reuse boundary.** Online causal score/softmax/context is adaptable from F4's tiled-attention algorithm, but the F4 contract is frozen around Llama's four-Q-heads-per-KV-head mapping, Llama K/V geometry, and Llama context gates (`phase-f4-tiled-attention-context.md:12-35,68-78`). Qwen requires a new Q/K/V layout and six-Q-per-KV mapping, BF16 policy, partial/mRoPE semantics, and hybrid-layer position handling. No Qwen full-attention stage may be admitted as an F4 alias.

### 3.4 Qwen full-attention modifiers — `qwen_specific`

These are part of the periodic full-attention layer but are not supplied by the generic F4 score/context core:

* `q_norm` and `k_norm`: RMSNorm weights `[256]`, epsilon `1e-6`, applied independently to each Q/K head;
* rotary embedding: `head_dim*partial_rotary_factor=64` rotary width, `rope_theta=10000000`, `rope_type=default`, interleaved mRoPE sections `[11,11,10]`; Qwen's language source uses `Qwen3_5RotaryEmbedding`/`MRoPERotaryEmbedding` rather than the Llama-3 RoPE sidecar;
* output gate: q projection's second `[B,S,24,256]` split is sigmoid-gated into the context before `o_proj`;
* the `KVCache` position is the committed absolute prefix `N`, and the full-layer update is independent of recurrent `ArraysCache` state.

The source symbols are `Qwen3_5Attention.__init__` and `__call__` (`language.py:1357-1508`) and the model `config.json:132-145`. These fields require a Qwen-specific oracle trace and must not borrow Llama RoPE, head count, or epsilon.

### 3.5 Norms, activations, and residuals — `adaptable_new_shape_or_packing`

The operation semantics are common transformer primitives, but the contract is not exact-shared:

| Stage | Exact IO and policy | Classification reason |
|---|---|---|
| Layer input RMSNorm | `[B,S,5120]` BF16 -> `[B,S,5120]` BF16; weight `[5120]` BF16; epsilon `1e-6` | Shape/epsilon/dtype differ from B0/F2 scalar Llama control (`H=2048`, epsilon `1e-5`) |
| Layer post-attention RMSNorm | `[B,S,5120]` BF16 -> `[B,S,5120]` BF16; weight `[5120]` BF16; epsilon `1e-6` | Same semantic operation, new Qwen geometry/policy |
| Final model RMSNorm | `[B,S,5120]` BF16 -> `[B,S,5120]` BF16; weight `[5120]` BF16; epsilon `1e-6` | Separate Qwen final-stage contract |
| Full-attention Q/K norms | `[B,S,24,256]` and `[B,S,4,256]` BF16; weights `[256]`; epsilon `1e-6` | Head-local norm absent from the frozen Llama F2/F4 ABI |
| DeltaNet Q/K norms | `[B,S,16,128]` BF16, normalized with the pinned recurrence scaling | Qwen recurrence-specific head dimensions |
| Gated DeltaNet output norm | value output and `z`, each `[B,S,48,128]` BF16; RMSNorm weight `[128]`, epsilon `1e-6`; SiLU gate in FP32, result cast to BF16 | Qwen-specific gated norm/activation boundary |
| MLP SwiGLU | `gate/up: [B,S,5120] -> [B,S,17408]`; SiLU/gate product `[B,S,17408]`; `down: [B,S,17408] -> [B,S,5120]`; BF16 activations | Reusable activation semantics, but new affine4 matrices and dimensions |
| Residual adds | `x+r` and `h+mlp(...)`, `[B,S,5120]` BF16 | Elementwise graph shape is adaptable; F2 image explicitly has no residual/activation epilogue (`f2-contract-freeze.md:141-147`) |

Pinned source refs are `Qwen3_5RMSNormGated`/`_precise_swiglu` (`language.py:47-66`), `Qwen3_5MLP` (`language.py:1511-1522`), `Qwen3_5DecoderLayer` (`language.py:1717-1760`), and the model config. F3 may supply a measured epilogue policy for its Llama graph, but Qwen must carry its own BF16/epsilon/gated-SwiGLU records; F6 cannot silently attach a F3 epilogue to the F2 standalone family.

### 3.6 Embeddings and LM head — `adaptable_new_shape_or_packing`

* Input token IDs `[B,S]` are looked up in the affine-4bit embedding matrix with logical shape `[248320,5120]` and packed U32/BF16/BF16 shapes `[248320,640]`, `[248320,80]`, `[248320,80]`; the resulting hidden tensor is `[B,S,5120]` BF16.
* `tie_word_embeddings=false`, so the LM head is a separate affine-4bit matrix with the same logical and packed shapes. It maps `[B,S,5120]` BF16 to logits `[B,S,248320]`; the exact logits dtype/cast must be carried by the Qwen numerical policy.
* The VLM top-level embedding helper can merge image/video features, but Q1 text mode never enters it. Special-token rejection is part of this boundary.

Pinned refs are `Qwen3_5Model.embed_tokens` (`language.py:1763-1790`), `LanguageModel.lm_head` (`language.py:1919-1930` and `:2561-2579`), `qwen3_5.py:36-86`, model `config.json:22-34,123-145`, and the text adapter. Embedding/head matrix semantics are adaptable to F2/P3 quantized execution only with their own family/pack/evidence; there is no exact Llama embedding/head pack.

### 3.7 Cache capture, raw transfer, and restore — `qwen_specific`

Qwen has two state classes in runtime order, never one homogeneous K/V list:

| Runtime layer set | Cache class | Leaf 0 | Leaf 1 | Position/trim |
|---|---|---|---|---|
| `i not in {3,7,...,63}` (48 layers) | `ArraysCache(size=2)` | `layer.i.arrays.conv_state`: `(1,3,10240)` BF16, owner `Qwen3_5GatedDeltaNet`, update `retain_last_3_mixed_qkv_rows` | `layer.i.arrays.delta_state`: `(1,48,128,128)` FP32, owner `gated_delta_update`, update `recurrent_delta_update` | `offset=null`; both leaves at `committed_position=N`; `trim_supported=false`; mutable list state |
| `i in {3,7,...,63}` (16 layers) | `KVCache` | `layer.i.full_attention.keys`: `(1,4,N,256)` BF16, owner `Qwen3_5Attention/KVCache`, update `KVCache.update_and_fetch` | `layer.i.full_attention.values`: `(1,4,N,256)` BF16, same owner/update | `offset=N`; `trim_supported=true` through `KVCache.trim`; immutable two-array tuple state |

The exact state shapes and ownership are frozen in `.superpowers/swarm/reports/q1-identity-freeze.md:180-189`, `.superpowers/swarm/reports/q1-hybrid-cache-green.md:23-31`, and `qwen_spill.py:351-370,433-473`. The fixture probe uses tokens `[760,6511,314,9338,369]`, `S=5`, `N=4`, so its full-layer state is `(1,4,4,256)` per K/V leaf and its linear leaves retain the same shapes above.

**Capture.** `capture_qwen_hybrid_state` takes the already materialized MLX-VLM cache list, validates the exact 64-entry config order, model fingerprint, class, leaf shape/dtype, and full-layer offset, then captures immutable leaf bytes and per-leaf digests. It does not construct tensors or recalculate the prefix. The capture boundary is after `T[:-1]`, `N=S-1`.

**Wire serialization.** `serialize_qwen_hybrid_state` emits version-1 `QWENSPIL1` metadata with `model_identity`, `committed_position`, the exact 64-class `runtime_layer_order`, 64 ordered entries, `offset=null` for ArraysCache and `offset=N` for KVCache, and two leaves per entry. Each leaf carries component ID, owner, update, position, trim flag, shape, dtype, byte count, digest, and opaque payload; the whole record has a checksum. Metadata and leaf payload mutations fail closed before state exposure.

**Raw transfer.** `upload_qwen_hybrid_state` validates a complete state, accepts only explicit unique layer indices in captured order, checks the resident window's integer capacity before upload, and streams raw leaf bytes at sequential window offsets. It does not decode BF16/FP32, allocate a complete cache, infer cache class, or perform model math. This is the only Q1 raw transfer surface.

**Executable restore.** `restore_qwen_hybrid_cache_into_mlx(model,state,cache=...)` is the sole conversion boundary. It validates all entries, finite BF16/FP32 words, target layer classes, shapes, and state metadata before any assignment; decodes canonical little-endian contiguous C-order BF16/FP32 bytes to real MLX arrays; assigns a mutable list to `ArraysCache.state`; assigns a two-array tuple to `KVCache.state` and sets `offset=N`; and commits only after all layers are prepared. It resolves `model.language_model.cache` or `language_model.make_cache()` when no explicit cache is supplied. The source is `native_r9700/qwen_hybrid_cache.py:75-136,139-211`; the ownership/report contract is `.superpowers/swarm/reports/q1-hybrid-cache-green.md:9-31`.

**Resume seam.** The final prompt token `T[-1]` is injected exactly once after restoring the S-1 state. Re-supplying the full prompt after cache acceptance is a duplicate-prefix error, not a repair path. Recurrent state is not recomputed by the consumer, and full-attention KV offset remains `N`.

### 3.8 Generic evidence/metadata shell — `exact_shared`

Qwen may reuse the non-tensor shell without changing its semantics:

* P3's allocation-free `KernelPackSpan`/`KernelPackOptional` view discipline;
* closed `KernelPackCompatibility` fields (`input_dtype`, `weight_dtype`, `output_dtype`, source layout version, shape family, physical packing version, tolerance policy);
* the exact five `EvidenceRef.record_kind` values and nine evidence slots, with `cpu_reference`/`offline_oracle` kept separate from request-bound `r9700_native` records;
* validate-before-allocation/submission and fail-closed rejection; no runtime parsing of the documentation manifest;
* Q1 source/model/inventory identity fields and the QWENSPIL1 digest/checksum shape.

P3 freezes the generic records and lookup boundary in `.superpowers/swarm/reports/p3-contract-freeze.md:38-52,199-236,238-312,330-404`. F2 owns WMMA-specific source/image evidence; P3 owns generic pack records; one supervisor-selected integration owner owns generated `kernel_assets.cpp`/`kernel_catalog.cpp`, as both F2 and P3 state. Q1 does not edit or create a parallel catalog. The Qwen values plugged into this shell remain model-specific and require a new pack identity.

## 4. Explicit F2/F3/F4 dependency map

### F2 — gfx1201 WMMA foundation

1. **Q1 research dependency:** none. The Q1 task packet explicitly permits task sets 1–5 to establish model/oracle truth before F2–F4 are complete (`phase-q1-qwen-contract-oracle.md:14-23`).
2. **F6 implementation dependency:** F6 consumes the accepted F2–F4 matrix/attention architecture and P3 metadata, but Qwen cannot reuse the F2 first family as an exact family. F2 is fixed to `M<=128`, `K=2048`, `N=8192`, FP16 A/B/D, FP32 accumulation, `v_wmma_f32_16x16x16_f16`, and `f2-wmma-physical-tile-v1` (`f2-contract-freeze.md:98-147,176-189`). Qwen's affine matrices have `K=5120`, `6144`, or `17408`, U32 affine4 weights, BF16 scales/biases/activations, and the distinct output widths in §2.
3. **Required adaptation:** a future F6 quantized family must have its own physical pack version, affine decode/scale/bias mapping, shape-family records, numerical policy, finite checks, and request-bound oracle/native evidence. The F2 atom/lane/descriptor discipline can be a substrate only after the selected pack has its own layout proof and admission record. Q1 does not select that family.

### F3 — matrix projection graph

1. F3 is Llama graph work in profile order gate/up -> down -> fused QKV -> O, with model-handle/prepacking identity and a selected WMMA family (`phase-f3-matrix-projection-graph.md:12-36,68-80`). It is not a Qwen graph contract.
2. Qwen's affine rows are: embedding/head; DeltaNet `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`; full-attention q/k/v/o; and MLP gate/up/down. Their dimensions, affine pack, BF16 numerics, recurrent ownership, and periodic layer placement must be carried as Qwen records. No Llama `gate_up`, homogeneous QKV, or O selection may be inferred from a Qwen tensor suffix.
3. F3 may supply matrix-family lifecycle and prepacking patterns after its accepted graph, but Qwen stage integration must remain a separate F6 task-set-3 boundary. F3's shared catalog/binder/layout/executor files remain outside Q1 ownership.

### F4 — tiled attention and context

1. F4's reusable algorithmic idea is causal tiled score/online-softmax/context with bounded prefix/live/chunk lengths and no full score/probability scratch (`phase-f4-tiled-attention-context.md:12-35,68-78,162-190`).
2. Qwen periodic full attention requires a new geometry: 24 query heads, 4 KV heads, six Q heads per KV head, head dimension 256, BF16 Q/K/V, Q/K norm, Qwen partial/mRoPE, output sigmoid gate, and K/V `(1,4,N,256)` state. F4's Llama four-Q-per-KV mapping and Llama RoPE/KV acceptance cannot be reused exactly.
3. F4 does not cover DeltaNet. Recurrent convolution/ring state, FP32 `[1,48,128,128]` recurrence, and non-trimmable ArraysCache leaves remain Qwen-specific even when full attention later uses an adapted tiled core.
4. F6's full Qwen path waits for the accepted F4 architecture per `phase-f6-quantized-model-promotion.md:12-36`; F6 must run Qwen-specific state/finite/numerical/token/stability/residency evidence rather than inherit F4's Llama context result.

### P3 — Kernel Pack boundary consumed by F6

P3's schema is exact-shared as a metadata contract, not as a Qwen pack. A Qwen pack must carry concrete source/image/build/provenance and accepted license fields, exact BF16/affine4 dtypes, logical source layout, a distinct physical `weight_packing_version`, all fixed/runtime dimensions, finite/tolerance policy, and the closed evidence references. A distinct physical pack requires an `offline_review/layout_proof` record; P3 does not regenerate F2 lane maps or reinterpret G0. Runtime receives generated allocation-free records and never reads `docs/upstream-reference-manifest.yaml` (`p3-contract-freeze.md:199-236,238-312,400-404`).

## 5. F6 implementation handoff without rediscovery

F6 should carry the following closed stage boundaries into its task-set-1 decision and then into task sets 2–5:

1. **Model admission:** verify the Q1 fingerprint, all source-pin/shard identity fields, text-only token policy, and the literal base revision marker before any tensor/state operation.
2. **Matrix family inputs:** bind only the affine triplets listed in §2. Source logical shapes and packed header shapes are fixed; raw windows remain bounded and output-major. Do not use the local FP16-cast affine source for BF16 model words without a new reviewed identity.
3. **Linear layer execution:** input RMSNorm `[B,S,5120]`; affine `in_proj_qkv/z/a/b`; BF16 depthwise causal convolution/ring update; Q/K normalization; gated DeltaNet recurrence with FP32 state `[B,48,128,128]`; gated RMSNorm/SiLU; affine `out_proj`; residual; post-attention RMSNorm; affine4 SwiGLU MLP; residual.
4. **Full layer execution:** input RMSNorm; affine q/k/v; Q/K head norms and Qwen mRoPE; causal full attention with Qwen 24/4/256 geometry; K/V append at absolute position `N`; sigmoid output gate; affine o projection; residual; post-attention RMSNorm; affine4 SwiGLU MLP; residual.
5. **Embedding/head:** text IDs only, embedding output `[B,S,5120]` BF16; final norm; untied affine4 head output `[B,S,248320]` with a declared logits cast.
6. **State boundary:** preserve 64 runtime entries in explicit model order. ArraysCache leaves are mutable `(1,3,10240)` BF16 plus `(1,48,128,128)` FP32, no offset and no trim. KVCache leaves are `(1,4,N,256)` BF16 plus offset `N`, trim only through KVCache semantics. Never collapse them into one cache type.
7. **Capture/restore/transfer:** capture only after `T[:-1]`; serialize QWENSPIL1 with leaf/component metadata and digests; raw upload only after capacity validation and in captured order; MLX restore decodes canonical little-endian bytes atomically; inject `T[-1]` exactly once.
8. **Evidence:** keep oracle fixtures/MLX outputs as `cpu_reference`/`offline_oracle`; native acceptance, if a later F6 task reaches it, requires separate request-bound `r9700_native` evidence with this exact model fingerprint and selected Qwen pack identity. Q1 itself supplies no such record.

## 6. Concrete blockers and boundary decisions

* **Provenance blocker:** `base_model_revision=unavailable_in_pinned_conversion_metadata` remains part of the fingerprint and keeps Q1/F6 promotion fail-closed until immutable base revision/license provenance is recorded or explicitly accepted by a human in the package.
* **Dtype boundary:** pinned model affine triplets and activations are BF16; the current `qwen_affine4_linear.cpp` FP16 bit-cast ABI is not the model representation. F6 must record and admit a BF16-correct affine source/pack before using any Qwen weight window.
* **Recurrence boundary:** `qwen_deltanet_state.cpp` is a bounded one-step state update. The full Qwen GatedDeltaNet contract also requires affine qkv/z/a/b, depthwise convolution, q/k normalization, `A_log`/`dt_bias` transforms, gated RMSNorm/SiLU, and `out_proj`; F6 must account for each named stage.
* **Attention boundary:** Qwen full attention is periodic and has six Q heads per KV head, head dimension 256, Qwen mRoPE, and a separate KVCache. F4's Llama four-Q mapping and Llama RoPE/cache geometry are not Qwen inputs.
* **Pack boundary:** F2 physical WMMA packing and P3 generated records are not Qwen quantized-family selection. The future F6 family decision must name the exact Qwen physical pack, source/image/license digests, shapes, scales/biases, numerical policy, and evidence records before graph integration.
* **Ownership boundary:** Q1 owns model/oracle truth only. F6 owns later native implementation, hardware evidence, residency, and performance. Q1 does not edit F2/P3 files, shared catalogs, or runtime selection.

No quantized family is selected here. No throughput, warm-performance, or native-acceptance statement is made.

## Source index

* `.superpowers/swarm/reports/q1-identity-freeze.md` — identity, model geometry, inventory schema, state ownership, S-1, fixture minimums, and Q1/F2/P3 ownership.
* `.superpowers/swarm/reports/q1-tensor-inventory.md` — schema-v2 header-only counts, affine classification, source/shard identity, and fail-closed inventory boundary.
* `.superpowers/swarm/reports/q1-hybrid-cache-green.md` and `q1-hybrid-cache-red.md` — 64-entry hybrid state, capture/restore, MLX assignment, CLI, and deterministic metadata.
* `.superpowers/swarm/reports/f2-contract-freeze.md` — F2 WMMA atom, first linear family, source/physical-layout split, numerics, evidence, and integration ownership.
* `.superpowers/swarm/reports/p3-contract-freeze.md` — closed Kernel Pack schema, compatibility, evidence, numerics, lookup, and runtime/offline boundary.
* `docs/pinned-upstream-interfaces.md:324-348` — pinned Qwen MLX-VLM/model/cache authority.
* `docs/tasks/r9700-products/phase-q1-qwen-contract-oracle.md:202-229` — task-set-5 target and acceptance.
* `docs/tasks/r9700-products/phase-f3-matrix-projection-graph.md` and `phase-f4-tiled-attention-context.md` — current Llama matrix/attention contracts and non-Qwen boundaries.
* `docs/tasks/r9700-products/phase-f6-quantized-model-promotion.md:12-36,115-138,187-214` and `docs/ROADMAP.md:216-242` — F6 dependencies and promotion gates.
* Pinned MLX-VLM source at revision `2b31570bdee86e2cdeea049761885aeed524a98c`: `mlx_vlm/models/qwen3_5/config.py`, `language.py`, `qwen3_5.py`, `cache.py`, and `qwen3_5/gated_delta.py`.
* Pinned mlx-lm source at revision `e2f2fb2aef987f86878d17638446183cffe21fe4`: `mlx_lm/models/cache.py` (`ArraysCache`, `KVCache`, `state`, `offset`, `trim`, `from_state`).
* Local source boundaries: `native_r9700/qwen_text_adapter.py`, `qwen_weight_binder.h/.cpp`, `kernels/qwen_affine4_linear.cpp`, `kernels/qwen_deltanet_state.cpp`, `kernels/qwen_full_attention.cpp`, `qwen_spill.py`, `qwen_hybrid_cache.py`, and `qwen_layer_executor.py`.
