"""C1 reference fixtures tests (Lane B2) — schema, determinism, size.

Marker: c1w2-lane-b2

Verifies the on-disk MLX oracle fixtures under ``tests/native_r9700/fixtures/``
that Lane A2 (primitive seam) and later task sets consume:

  prompts.json             prompt texts, mlx token ids, S per prompt
  baseline_r_tokens.json   mlx-lm native-baseline R token ids per prompt
  kv_state.npz             per-layer K/V for prompt-0, S-1 prefix, fp16 (1,8,N,64)
  primitives_fixtures.npz  deterministic small intermediate tensors (cast,
                           matmul, rms_norm, silu) per the Lane A2-agreed schema
  layer_trace_full_inner_projection_fixtures.npz  bulky K/V cols0:64 oracles
  layer_trace_q_full_inner_projection_fixtures.npz  bulky Q cols0:64 oracle
  fixtures_schema.json     documented schema + source digests

These are small, deterministic, committed-friendly files. The deterministic
numpy primitive tensors are recomputed in-process and compared to the on-disk
arrays to prove determinism without re-running the mlx oracle. The mlx-derived
oracle content (token ids, R tokens, KV) is checked for schema, shape, dtype,
size, and cross-file digest consistency against ``fixtures_schema.json``.

Per the Lane A2 seam contract, every test ``pytest.skip``s gracefully when the
fixture directory is absent, so each lane's focused suite stays independently
green before the supervisor combines them.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FIXTURES_DIR = os.path.join(_REPO_ROOT, "tests", "native_r9700", "fixtures")

from native_r9700 import ref_fixtures as rf  # noqa: E402

# Size ceiling (bytes) for a single committed fixture file. The KV npz is the
# largest (16 layers x K/V x (1,8,5,64) fp16); anything beyond this bound is a
# "large blob" the C1 contract forbids committing.
_MAX_FIXTURE_BYTES = 512 * 1024
_EXPECTED_TRACE_LAYERS = (0, 15)
_TRACE_TOKEN_COUNT = 2
_TRACE_DIM = 16
_TRACE_HEAD_COUNT = 2
_TRACE_SCORE_SOURCE_TOKENS = 5
_TILE_ROWS = 8
_TILE_INNER = 16
_TILE_COLS = 8
_TILE_VALID_ROWS = 5
_FULL_INNER = rf.HIDDEN_SIZE
_ROPE_PAIR_COUNT = 8
_ROPE_PAIR_START = 12
_ROPE_TOKEN_INDEX = 1
_ROPE_FULL_HEAD_PAIR_COUNT = rf.HEAD_DIM // 2
_ROPE_HEAD_INDEX = 0
_POST_ATTENTION_RMSNORM_COLS0_64_FIXTURE_SHA256 = (
    "58e213bed698fcf00584ab7e7a653f9a51d0c6cde4cfdde133ab995c863c6c59"
)
_POST_ATTENTION_RMSNORM_COLS0_64_EXPECTED_FP16_SHA256 = (
    "66d70f967e30b3ddad71dc8fadf7e9157d7badc2fd0d654e26177a908fddd903"
)
_MLP_FULL_INNER_PROJECTION_FIXTURE_SHA256 = (
    "b5a6a11d98cae23d1836366ec3584de516d66c169c277367f8adc74966ad10c1"
)
_MLP_GATE_PROJ_COLS0_64_EXPECTED_FP16_SHA256 = (
    "328f6449d994f8fb2e3c54991a536b2dc89188f8af16c0d8ce557a6321c84845"
)
_MLP_UP_PROJ_COLS0_64_EXPECTED_FP16_SHA256 = (
    "64c55737432e8eb4496048a52b3eb66d49a3348c0c573aa45b16032a33ac600a"
)
_MLP_DOWN_PROJ_INNER_COLS0_64_TO_COLS0_64_FIXTURE_SHA256 = (
    "62ba57e858a723ea0e326e6d3773d915e423c777f84b39bd6b322a39a98f30ad"
)
_MLP_DOWN_PROJ_INNER_COLS0_64_TO_COLS0_64_EXPECTED_FP32_SHA256 = (
    "64559386ab500f4807074afadb1878c50f14069a9c1ff4bd48c1931658ade390"
)
_MLP_ACTIVATION_FULL_INNER_FIXTURE_SHA256 = (
    "c4ac8b5c351d57097cc0fb6f68539f1aa2996591c13e27064f0a146b5e2d6ad9"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS0_64_FIXTURE_SHA256 = (
    "e3aab29d893f849fc4627e4781ca36fef1574ccf4d5dda562fcdacf3438bb338"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS0_64_EXPECTED_FP32_SHA256 = (
    "84f9ddf66e1e71849b928caa061b6abcca81d00bea59081635592ca7d58f4d7e"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS64_128_FIXTURE_SHA256 = (
    "d1242c9add185957e7c5cf8273d6f26d9eb4103786e0496bf1a0d5e29d9929f6"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS64_128_EXPECTED_FP32_SHA256 = (
    "cd7f1e930959cf668116142d14ad8374ddbf732d6b71134aa9550de9c277d21a"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS128_192_FIXTURE_SHA256 = (
    "ba75e101395b1682c92585b8030ea7d78431f15b3c58f40ea47564c28aac9b4d"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS128_192_EXPECTED_FP32_SHA256 = (
    "f70e9db966a3e63ca22a38f06c68b60b4910c2522055670404f4eb24405f89b4"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS192_256_FIXTURE_SHA256 = (
    "691e6c216090c5569a39177c532f3eca6b8e4792ef30656dbdeb4529495378f6"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS192_256_EXPECTED_FP32_SHA256 = (
    "f5ca75c595cfebb249605cb43788a2e64f6bd508f3cd5dd262a24b3101fa3533"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS256_320_FIXTURE_SHA256 = (
    "252fdad991788ef0caf826450e0e35058ad5913b943569cbaeeca2a606c264e2"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS256_320_EXPECTED_FP32_SHA256 = (
    "d9c9e4bf8a22f7c23842c9e7bc45eaf4160d02ffe853bf213f2137e4650ac3ea"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS320_384_FIXTURE_SHA256 = (
    "c80392d1613bffe36e0d910a17b909100f5a1ab3443a4b7d9d12fe9abd42ae35"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS320_384_EXPECTED_FP32_SHA256 = (
    "d778614a9ca5543a9b399379f6e9161af6e14722e74d1a204047ab8e0e17bc94"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS384_448_FIXTURE_SHA256 = (
    "ee04edb8003f6b7d90e6febb1493aeec40a5c44b5693f436fbf0f746c33c855d"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS384_448_EXPECTED_FP32_SHA256 = (
    "009dfb2599ca19db39614c27b269db20b4d29408a16e78ca3ff89037818fb4e6"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS448_512_FIXTURE_SHA256 = (
    "d617b8f69e5d484db6ec7abe96e888b990ba091b4e7b44c43318533e5035c069"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS448_512_EXPECTED_FP32_SHA256 = (
    "9460250f2aa90b73b75b74e904c3e049ab2e0d734001faa772b205debe279c26"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS512_576_FIXTURE_SHA256 = (
    "dba88e1d9feb9454c0cc9a510d705d133640a6e823467ea9c7ada4fab07ae12b"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS512_576_EXPECTED_FP32_SHA256 = (
    "1dbf59efedf86d3b58e67ebe5914c907720b91b36d358594e807813ce9b08e46"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS576_640_FIXTURE_SHA256 = (
    "649fd23988fe8f7a4c40f3ca09b2e3c04bdffa25f807e0ca81892effac815e77"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS576_640_EXPECTED_FP32_SHA256 = (
    "811bfd494dd3f5fbbe89281e90a85a31f03623fe3837d7bb6a12cb5d9dd7df55"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS640_704_FIXTURE_SHA256 = (
    "4f87494495e5d07765f76504711c810ee9bf20e680c83f26d9656e3ac4f7ba9c"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS640_704_EXPECTED_FP32_SHA256 = (
    "98d6b80871779d4c01cb6205d4d6b95e9a3a7d2ee66fdf5f8d864feacc3088f8"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS704_768_FIXTURE_SHA256 = (
    "bd14dc7b032e7c2fe2340743cbf360339f514c7ca0f938ea89665c473098f22e"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS704_768_EXPECTED_FP32_SHA256 = (
    "db8c4b0630d95561e8cc3dfebfbe2a3d0aa13b14116dbaaeb4fcf24efc45de0a"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS768_832_FIXTURE_SHA256 = (
    "9492bbaac58440e9495cf9d452c51f7889bc2035fca18dea8c4644001fe4178d"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS768_832_EXPECTED_FP32_SHA256 = (
    "efec5949b3122b9bb0e50ffd16d70619060448ad613e8808b117d4420d03d0d7"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS832_896_FIXTURE_SHA256 = (
    "ce7e699faee42b53bbd4b20f3517615113436af86296d39487a274b38cd1cca3"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS832_896_EXPECTED_FP32_SHA256 = (
    "371bd55cba32daf9d363aa8b1df52bda6db006ce82b333846829f143b9e14750"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS896_960_FIXTURE_SHA256 = (
    "0191316e76c2770adc3da5bdd3e6d67fd27199051b97ab3d6f62ed8c1ba228ff"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS896_960_EXPECTED_FP32_SHA256 = (
    "141ee577a5ec10d7fb2529f7f6f7aa9c398b39bbba25356c254138e1aa4d30b3"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS960_1024_FIXTURE_SHA256 = (
    "3d5ec83e98bd07500b1e40af66a1d117ef0d2b71901a1a2ecc0679fd5b1b51df"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS960_1024_EXPECTED_FP32_SHA256 = (
    "1ce368e6195745aa49d0569e17a28847c8086d59d8449cc9382bc56edf4a0830"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1024_1088_FIXTURE_SHA256 = (
    "1437af73e83249565ca7c4205d4bcae23a52c6cdd3a4fbb89bf5d7777fca4153"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1024_1088_EXPECTED_FP32_SHA256 = (
    "b477fecfa61b2e402af9e4fe4be4a3ee562defe89f7089d8daf0de37574f3f43"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1088_1152_FIXTURE_SHA256 = (
    "ec8b3887d7d19cc2b83260384e67eaef9fe38cfa6ad548eb0d94265ace88ed72"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1088_1152_EXPECTED_FP32_SHA256 = (
    "96a4426d900daec45deb30b32c628c2b70ea51972b2b841984f62500f0b1cb28"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1152_1216_FIXTURE_SHA256 = (
    "9185f0abdb25976573b1481a08dbe809bddee2e46fc44b9fb57fbc93ed669e5a"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1152_1216_EXPECTED_FP32_SHA256 = (
    "9d32308facc313a99452dc1266c2ec2d0a0bec9b0f371cbf72f93119a8a3eaea"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1216_1280_FIXTURE_SHA256 = (
    "610ded7944f4a92930cf6d610e9c3f5d4a857bb17d40b029be723cef96a8d84e"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1216_1280_EXPECTED_FP32_SHA256 = (
    "0474bf43af83683b747b2944216bbe77885ccd29f275bf6f1fc35fd8f25ae3aa"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1280_1344_FIXTURE_SHA256 = (
    "021dd5f080ccc96e73c05747d0c70d215687ee6a60cf5a2983ed0a692b897c68"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1280_1344_EXPECTED_FP32_SHA256 = (
    "f0129a3371ee92b6cc844bb3654e859e19f503e87c2d8ad8cbefea4880280112"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1344_1408_FIXTURE_SHA256 = (
    "30ea05c720ff1004a8c76c2db6ef869cebca003c63bb472457f9d5c19012cd13"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1344_1408_EXPECTED_FP32_SHA256 = (
    "c33442e0720192d8485a2d3d267bde9e1c391a1da741eae0b4e7ed9ee62284d7"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1408_1472_FIXTURE_SHA256 = (
    "c734eec72fd4784ab699e4f9654253130b3775f3240c42cd3dc857f3452acffc"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1408_1472_EXPECTED_FP32_SHA256 = (
    "d83fbf7899a07c74e9c064f799baa86eacbfbc25447047a62e966a31642660a0"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1472_1536_FIXTURE_SHA256 = (
    "1c71e4cc882f89cfb62882d3ff04e4380d36b88fa6890df024c276e30d1f85d8"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1472_1536_EXPECTED_FP32_SHA256 = (
    "23a1ab9ad45b36d9c6d46a2463e62a23f71cb1183ec2a91f1c4226f93557fba3"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1536_1600_FIXTURE_SHA256 = (
    "65df5bacd3d3fe1c26f76f86bb745fac4f83a98347e063a0760d8aed4a4b3cae"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1536_1600_EXPECTED_FP32_SHA256 = (
    "e63fa909ecc36084bb4fad28144f25d583898a9a5cfad55e310c90cd7da27b3e"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1856_1920_FIXTURE_SHA256 = (
    "7ea12733c622bfd2ed0dc3293c73c4c61a264e7730eb60e4768dd89b7b18f206"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1856_1920_EXPECTED_FP32_SHA256 = (
    "5ca70f7d6aee8c712acedef671c1eb8514cce0754318411afe90906eda956ea7"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1920_1984_FIXTURE_SHA256 = (
    "efabe98f441524e20aeed7810fa56c853f577b72ce1c531e1bfc403c11fa2cb2"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1920_1984_EXPECTED_FP32_SHA256 = (
    "7a78d5e6e416906132a279dcfe7a065c73594849361d2af7a9ea5cee37798bc0"
)

_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1984_2048_FIXTURE_SHA256 = (
    "4e431df60dae178163293afe2efa74f0ab9867e85e6bf68b60cee1efea4d186f"
)
_MLP_DOWN_PROJ_FULL_INNER_TO_COLS1984_2048_EXPECTED_FP32_SHA256 = (
    "a020ed331d7ba0e6c3b63a46c991fbe201e042e51ea36c641922b82111f31f79"
)



_MLP_DOWN_PROJ_FULL_INNER_CASES = (
    (
        "layer0_mlp_down_proj_full_inner_to_cols0_64",
        "layer_trace_mlp_down_projection_full_inner_to_cols0_64",
        (0, 64),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS0_64_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS0_64_EXPECTED_FP32_SHA256,
        True,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols64_128",
        "layer_trace_mlp_down_projection_full_inner_to_cols64_128",
        (64, 128),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS64_128_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS64_128_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols128_192",
        "layer_trace_mlp_down_projection_full_inner_to_cols128_192",
        (128, 192),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS128_192_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS128_192_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols192_256",
        "layer_trace_mlp_down_projection_full_inner_to_cols192_256",
        (192, 256),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS192_256_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS192_256_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols256_320",
        "layer_trace_mlp_down_projection_full_inner_to_cols256_320",
        (256, 320),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS256_320_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS256_320_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols320_384",
        "layer_trace_mlp_down_projection_full_inner_to_cols320_384",
        (320, 384),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS320_384_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS320_384_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols384_448",
        "layer_trace_mlp_down_projection_full_inner_to_cols384_448",
        (384, 448),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS384_448_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS384_448_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols448_512",
        "layer_trace_mlp_down_projection_full_inner_to_cols448_512",
        (448, 512),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS448_512_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS448_512_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols512_576",
        "layer_trace_mlp_down_projection_full_inner_to_cols512_576",
        (512, 576),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS512_576_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS512_576_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols576_640",
        "layer_trace_mlp_down_projection_full_inner_to_cols576_640",
        (576, 640),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS576_640_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS576_640_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols640_704",
        "layer_trace_mlp_down_projection_full_inner_to_cols640_704",
        (640, 704),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS640_704_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS640_704_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols704_768",
        "layer_trace_mlp_down_projection_full_inner_to_cols704_768",
        (704, 768),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS704_768_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS704_768_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols768_832",
        "layer_trace_mlp_down_projection_full_inner_to_cols768_832",
        (768, 832),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS768_832_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS768_832_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols832_896",
        "layer_trace_mlp_down_projection_full_inner_to_cols832_896",
        (832, 896),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS832_896_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS832_896_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols896_960",
        "layer_trace_mlp_down_projection_full_inner_to_cols896_960",
        (896, 960),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS896_960_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS896_960_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols960_1024",
        "layer_trace_mlp_down_projection_full_inner_to_cols960_1024",
        (960, 1024),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS960_1024_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS960_1024_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1024_1088",
        "layer_trace_mlp_down_projection_full_inner_to_cols1024_1088",
        (1024, 1088),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1024_1088_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1024_1088_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1088_1152",
        "layer_trace_mlp_down_projection_full_inner_to_cols1088_1152",
        (1088, 1152),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1088_1152_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1088_1152_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1152_1216",
        "layer_trace_mlp_down_projection_full_inner_to_cols1152_1216",
        (1152, 1216),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1152_1216_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1152_1216_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1216_1280",
        "layer_trace_mlp_down_projection_full_inner_to_cols1216_1280",
        (1216, 1280),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1216_1280_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1216_1280_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1280_1344",
        "layer_trace_mlp_down_projection_full_inner_to_cols1280_1344",
        (1280, 1344),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1280_1344_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1280_1344_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1344_1408",
        "layer_trace_mlp_down_projection_full_inner_to_cols1344_1408",
        (1344, 1408),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1344_1408_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1344_1408_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1408_1472",
        "layer_trace_mlp_down_projection_full_inner_to_cols1408_1472",
        (1408, 1472),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1408_1472_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1408_1472_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1472_1536",
        "layer_trace_mlp_down_projection_full_inner_to_cols1472_1536",
        (1472, 1536),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1472_1536_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1472_1536_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1536_1600",
        "layer_trace_mlp_down_projection_full_inner_to_cols1536_1600",
        (1536, 1600),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1536_1600_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1536_1600_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1600_1664",
        "layer_trace_mlp_down_projection_full_inner_to_cols1600_1664",
        (1600, 1664),
        "ef2858ea16d1651e7bee0f40e70ec493f5cd02857f06c404e3867c9c1b96fe20",
        "798d5abe872dc3a2ab90d0ecf8d3e726a099490d41beeb7664b175b36c082c67",
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1664_1728",
        "layer_trace_mlp_down_projection_full_inner_to_cols1664_1728",
        (1664, 1728),
        "038a847c96b6657fe529b3b25fc48bf73b0b0328c1c139b4463392656945d437",
        "b3dac35c0a94c400326c4065504781ca4ee51a5ee9a2cad421fd5d5cbebc7995",
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1728_1792",
        "layer_trace_mlp_down_projection_full_inner_to_cols1728_1792",
        (1728, 1792),
        "1b621038e1fd86431ad4e71ec5ae0e596cad4b12cdbe1aa3cfea495d774073b4",
        "8f56445ef4b2f1a53ac290a34fbe786a19667b9622443e78fe3e97a9fa912e00",
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1792_1856",
        "layer_trace_mlp_down_projection_full_inner_to_cols1792_1856",
        (1792, 1856),
        "381a14fcf4e909735f238cc0a346cd496644dc60f3ddcb6fc826a95e94341c9b",
        "f43a8b5ce52c594f6270ec62fe7e26e0e7e4a169573550280b4a0ff3cf69e66c",
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1856_1920",
        "layer_trace_mlp_down_projection_full_inner_to_cols1856_1920",
        (1856, 1920),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1856_1920_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1856_1920_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1920_1984",
        "layer_trace_mlp_down_projection_full_inner_to_cols1920_1984",
        (1920, 1984),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1920_1984_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1920_1984_EXPECTED_FP32_SHA256,
        False,
    ),
    (
        "layer0_mlp_down_proj_full_inner_to_cols1984_2048",
        "layer_trace_mlp_down_projection_full_inner_to_cols1984_2048",
        (1984, 2048),
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1984_2048_FIXTURE_SHA256,
        _MLP_DOWN_PROJ_FULL_INNER_TO_COLS1984_2048_EXPECTED_FP32_SHA256,
        False,
    ),
)




_LAYER0_POST_LAYER_HIDDEN_FIXTURE_SHA256 = (
    "feb3f5f10bca2182d677f0edb5f386270b2e1f91c21275d7ed95c419d14bc7a7"
)





_EXPECTED_TRACE_ARRAY_SPECS = {
    "hidden_in_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "input_norm_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "q_proj_fp16": ((1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "k_proj_fp16": ((1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "v_proj_fp16": ((1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "q_rope_fp16": ((1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "k_rope_fp16": ((1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "attention_scores_fp32": (
        (1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_SCORE_SOURCE_TOKENS),
        np.float32,
    ),
    "attention_probs_fp32": (
        (1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_SCORE_SOURCE_TOKENS),
        np.float32,
    ),
    "attention_context_fp16": ((1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "o_proj_output_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "attention_residual_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "post_norm_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "gate_proj_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "up_proj_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "silu_gate_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "gated_mlp_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "down_proj_output_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "mlp_residual_out_fp16": ((_TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "final_K_fp16": ((1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
    "final_V_fp16": ((1, _TRACE_HEAD_COUNT, _TRACE_TOKEN_COUNT, _TRACE_DIM), np.float16),
}
_EXPECTED_LAYER0_TILE_ARRAY_SPECS = {
    "layer0_k_proj_tile_a_fp16": ((_TILE_ROWS, _TILE_INNER), np.float16),
    "layer0_k_proj_tile_b_fp16": ((_TILE_INNER, _TILE_COLS), np.float16),
    "layer0_k_proj_tile_expected_fp32": ((_TILE_ROWS, _TILE_COLS), np.float32),
}
_EXPECTED_LAYER0_FULL_INNER_COLS8_ARRAY_SPECS = {
    "layer0_k_proj_full_inner_cols8_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_k_proj_full_inner_cols8_b_fp16": ((_FULL_INNER, _TILE_COLS), np.float16),
    "layer0_k_proj_full_inner_cols8_expected_fp32": ((_TILE_ROWS, _TILE_COLS), np.float32),
    "layer0_k_proj_full_inner_cols8_expected_fp16": ((_TILE_ROWS, _TILE_COLS), np.float16),
}
_EXPECTED_LAYER0_FULL_INNER_COLS0_16_ARRAY_SPECS = {
    "layer0_k_proj_full_inner_cols0_16_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_k_proj_full_inner_cols0_16_b_fp16": ((_FULL_INNER, 2 * _TILE_COLS), np.float16),
    "layer0_k_proj_full_inner_cols0_16_expected_fp32": (
        (_TILE_ROWS, 2 * _TILE_COLS),
        np.float32,
    ),
    "layer0_k_proj_full_inner_cols0_16_expected_fp16": (
        (_TILE_ROWS, 2 * _TILE_COLS),
        np.float16,
    ),
}

_EXPECTED_LAYER0_FULL_INNER_COLS0_64_ARRAY_SPECS = {
    "layer0_k_proj_full_inner_cols0_64_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_k_proj_full_inner_cols0_64_b_fp16": ((_FULL_INNER, 8 * _TILE_COLS), np.float16),
    "layer0_k_proj_full_inner_cols0_64_expected_fp32": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float32,
    ),
    "layer0_k_proj_full_inner_cols0_64_expected_fp16": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float16,
    ),
}


_EXPECTED_LAYER0_V_FULL_INNER_COLS8_ARRAY_SPECS = {
    "layer0_v_proj_full_inner_cols8_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_v_proj_full_inner_cols8_b_fp16": ((_FULL_INNER, _TILE_COLS), np.float16),
    "layer0_v_proj_full_inner_cols8_expected_fp32": ((_TILE_ROWS, _TILE_COLS), np.float32),
    "layer0_v_proj_full_inner_cols8_expected_fp16": ((_TILE_ROWS, _TILE_COLS), np.float16),
}

_EXPECTED_LAYER0_V_FULL_INNER_COLS0_64_ARRAY_SPECS = {
    "layer0_v_proj_full_inner_cols0_64_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_v_proj_full_inner_cols0_64_b_fp16": ((_FULL_INNER, 8 * _TILE_COLS), np.float16),
    "layer0_v_proj_full_inner_cols0_64_expected_fp32": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float32,
    ),
    "layer0_v_proj_full_inner_cols0_64_expected_fp16": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float16,
    ),
}

_EXPECTED_LAYER0_Q_FULL_INNER_COLS8_ARRAY_SPECS = {
    "layer0_q_proj_full_inner_cols8_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_q_proj_full_inner_cols8_b_fp16": ((_FULL_INNER, _TILE_COLS), np.float16),
    "layer0_q_proj_full_inner_cols8_expected_fp32": ((_TILE_ROWS, _TILE_COLS), np.float32),
    "layer0_q_proj_full_inner_cols8_expected_fp16": ((_TILE_ROWS, _TILE_COLS), np.float16),
}
_EXPECTED_LAYER0_Q_FULL_INNER_COLS0_64_ARRAY_SPECS = {
    "layer0_q_proj_full_inner_cols0_64_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_q_proj_full_inner_cols0_64_b_fp16": ((_FULL_INNER, 8 * _TILE_COLS), np.float16),
    "layer0_q_proj_full_inner_cols0_64_expected_fp32": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float32,
    ),
    "layer0_q_proj_full_inner_cols0_64_expected_fp16": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float16,
    ),
}

_EXPECTED_LAYER0_O_FULL_INNER_COLS0_64_ARRAY_SPECS = {
    "layer0_o_proj_full_inner_cols0_64_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_o_proj_full_inner_cols0_64_b_fp16": ((_FULL_INNER, 8 * _TILE_COLS), np.float16),
    "layer0_o_proj_full_inner_cols0_64_expected_fp32": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float32,
    ),
    "layer0_o_proj_full_inner_cols0_64_expected_fp16": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float16,
    ),
}

_EXPECTED_LAYER0_O_FULL_INNER_COLS64_128_ARRAY_SPECS = {
    "layer0_o_proj_full_inner_cols64_128_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_o_proj_full_inner_cols64_128_b_fp16": ((_FULL_INNER, 8 * _TILE_COLS), np.float16),
    "layer0_o_proj_full_inner_cols64_128_expected_fp32": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float32,
    ),
    "layer0_o_proj_full_inner_cols64_128_expected_fp16": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float16,
    ),
}

_EXPECTED_LAYER0_O_FULL_INNER_COLS128_192_ARRAY_SPECS = {
    "layer0_o_proj_full_inner_cols128_192_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_o_proj_full_inner_cols128_192_b_fp16": ((_FULL_INNER, 8 * _TILE_COLS), np.float16),
    "layer0_o_proj_full_inner_cols128_192_expected_fp32": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float32,
    ),
    "layer0_o_proj_full_inner_cols128_192_expected_fp16": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float16,
    ),
}

_EXPECTED_LAYER0_O_FULL_INNER_COLS192_256_ARRAY_SPECS = {
    "layer0_o_proj_full_inner_cols192_256_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_o_proj_full_inner_cols192_256_b_fp16": ((_FULL_INNER, 8 * _TILE_COLS), np.float16),
    "layer0_o_proj_full_inner_cols192_256_expected_fp32": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float32,
    ),
    "layer0_o_proj_full_inner_cols192_256_expected_fp16": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float16,
    ),
}

_EXPECTED_LAYER0_O_FULL_INNER_COLS256_320_ARRAY_SPECS = {
    "layer0_o_proj_full_inner_cols256_320_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_o_proj_full_inner_cols256_320_b_fp16": ((_FULL_INNER, 8 * _TILE_COLS), np.float16),
    "layer0_o_proj_full_inner_cols256_320_expected_fp32": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float32,
    ),
    "layer0_o_proj_full_inner_cols256_320_expected_fp16": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float16,
    ),
}

_EXPECTED_LAYER0_O_FULL_INNER_COLS320_384_ARRAY_SPECS = {
    "layer0_o_proj_full_inner_cols320_384_a_fp16": ((_TILE_ROWS, _FULL_INNER), np.float16),
    "layer0_o_proj_full_inner_cols320_384_b_fp16": ((_FULL_INNER, 8 * _TILE_COLS), np.float16),
    "layer0_o_proj_full_inner_cols320_384_expected_fp32": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float32,
    ),
    "layer0_o_proj_full_inner_cols320_384_expected_fp16": (
        (_TILE_ROWS, 8 * _TILE_COLS),
        np.float16,
    ),
}



_EXPECTED_LAYER0_ROPE_PAIR_ARRAY_SPECS = {
    "layer0_k_rope_pairs12_20_input_fp16": ((2, _ROPE_PAIR_COUNT), np.float16),
    "layer0_k_rope_pairs12_20_cos_fp32": ((_ROPE_PAIR_COUNT,), np.float32),
    "layer0_k_rope_pairs12_20_sin_fp32": ((_ROPE_PAIR_COUNT,), np.float32),
    "layer0_k_rope_pairs12_20_expected_fp16": ((2, _ROPE_PAIR_COUNT), np.float16),
    "layer0_q_rope_pairs12_20_input_fp16": ((2, _ROPE_PAIR_COUNT), np.float16),
    "layer0_q_rope_pairs12_20_cos_fp32": ((_ROPE_PAIR_COUNT,), np.float32),
    "layer0_q_rope_pairs12_20_sin_fp32": ((_ROPE_PAIR_COUNT,), np.float32),
    "layer0_q_rope_pairs12_20_expected_fp16": ((2, _ROPE_PAIR_COUNT), np.float16),
}
_EXPECTED_LAYER0_K_ROPE_FULL_HEAD_ARRAY_SPECS = {
    "layer0_k_rope_token1_head0_full_head_input_fp16": (
        (2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
    "layer0_k_rope_token1_head0_full_head_cos_fp32": (
        (_ROPE_FULL_HEAD_PAIR_COUNT,),
        np.float32,
    ),
    "layer0_k_rope_token1_head0_full_head_sin_fp32": (
        (_ROPE_FULL_HEAD_PAIR_COUNT,),
        np.float32,
    ),
    "layer0_k_rope_token1_head0_full_head_expected_fp16": (
        (2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
}

_EXPECTED_LAYER0_K_ROPE_PREFIX_HEAD0_ARRAY_SPECS = {
    "layer0_k_rope_tokens0_5_head0_full_head_input_fp16": (
        (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
    "layer0_k_rope_tokens0_5_head0_full_head_cos_fp32": (
        (5, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float32,
    ),
    "layer0_k_rope_tokens0_5_head0_full_head_sin_fp32": (
        (5, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float32,
    ),
    "layer0_k_rope_tokens0_5_head0_full_head_expected_fp16": (
        (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
}


_EXPECTED_LAYER0_Q_ROPE_FULL_HEAD_ARRAY_SPECS = {
    "layer0_q_rope_token1_head0_full_head_input_fp16": (
        (2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
    "layer0_q_rope_token1_head0_full_head_cos_fp32": (
        (_ROPE_FULL_HEAD_PAIR_COUNT,),
        np.float32,
    ),
    "layer0_q_rope_token1_head0_full_head_sin_fp32": (
        (_ROPE_FULL_HEAD_PAIR_COUNT,),
        np.float32,
    ),
    "layer0_q_rope_token1_head0_full_head_expected_fp16": (
        (2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
}

_EXPECTED_LAYER0_Q_ROPE_PREFIX_HEAD0_ARRAY_SPECS = {
    "layer0_q_rope_tokens0_5_head0_full_head_input_fp16": (
        (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
    "layer0_q_rope_tokens0_5_head0_full_head_cos_fp32": (
        (5, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float32,
    ),
    "layer0_q_rope_tokens0_5_head0_full_head_sin_fp32": (
        (5, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float32,
    ),
    "layer0_q_rope_tokens0_5_head0_full_head_expected_fp16": (
        (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
}

_EXPECTED_LAYER0_Q_ROPE_PREFIX_HEAD1_ARRAY_SPECS = {
    "layer0_q_rope_tokens0_5_head1_full_head_input_fp16": (
        (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
    "layer0_q_rope_tokens0_5_head1_full_head_cos_fp32": (
        (5, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float32,
    ),
    "layer0_q_rope_tokens0_5_head1_full_head_sin_fp32": (
        (5, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float32,
    ),
    "layer0_q_rope_tokens0_5_head1_full_head_expected_fp16": (
        (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
}

_EXPECTED_LAYER0_Q_ROPE_PREFIX_HEAD2_ARRAY_SPECS = {
    "layer0_q_rope_tokens0_5_head2_full_head_input_fp16": (
        (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
    "layer0_q_rope_tokens0_5_head2_full_head_cos_fp32": (
        (5, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float32,
    ),
    "layer0_q_rope_tokens0_5_head2_full_head_sin_fp32": (
        (5, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float32,
    ),
    "layer0_q_rope_tokens0_5_head2_full_head_expected_fp16": (
        (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT),
        np.float16,
    ),
}

_EXPECTED_LAYER0_ATTENTION_SCORE_RAW_HEAD0_ARRAY_SPECS = {
    "layer0_attention_score_raw_head0_tokens0_5_q_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_score_raw_head0_tokens0_5_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_score_raw_head0_tokens0_5_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}
_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD0_ARRAY_SPECS = {
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD0_ARRAY_SPECS = {
    "layer0_attention_probs_head0_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head0_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head0_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}
_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD1_ARRAY_SPECS = {
    "layer0_attention_scores_head1_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head1_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head1_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head1_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD1_ARRAY_SPECS = {
    "layer0_attention_probs_head1_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head1_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head1_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}





_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD2_ARRAY_SPECS = {
    "layer0_attention_scores_head2_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head2_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head2_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head2_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD2_ARRAY_SPECS = {
    "layer0_attention_probs_head2_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head2_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head2_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD3_ARRAY_SPECS = {
    "layer0_attention_scores_head3_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head3_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head3_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head3_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD3_ARRAY_SPECS = {
    "layer0_attention_probs_head3_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head3_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head3_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD4_ARRAY_SPECS = {
    "layer0_attention_scores_head4_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head4_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head4_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head4_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD4_ARRAY_SPECS = {
    "layer0_attention_probs_head4_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head4_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head4_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD5_ARRAY_SPECS = {
    "layer0_attention_scores_head5_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head5_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head5_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head5_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD5_ARRAY_SPECS = {
    "layer0_attention_probs_head5_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head5_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head5_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}



_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD6_ARRAY_SPECS = {
    "layer0_attention_scores_head6_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head6_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head6_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head6_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD7_ARRAY_SPECS = {
    "layer0_attention_scores_head7_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head7_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head7_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head7_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD8_ARRAY_SPECS = {
    "layer0_attention_scores_head8_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head8_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head8_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head8_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD10_ARRAY_SPECS = {
    "layer0_attention_scores_head10_tokens0_5_scaled_masked_q_scaled_fp16": ((_TILE_ROWS, rf.HEAD_DIM), np.float16),
    "layer0_attention_scores_head10_tokens0_5_scaled_masked_k_as_b_fp16": ((rf.HEAD_DIM, _TILE_ROWS), np.float16),
    "layer0_attention_scores_head10_tokens0_5_scaled_masked_seed_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    "layer0_attention_scores_head10_tokens0_5_scaled_masked_expected_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
}

_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD11_ARRAY_SPECS = {
    "layer0_attention_scores_head11_tokens0_5_scaled_masked_q_scaled_fp16": ((_TILE_ROWS, rf.HEAD_DIM), np.float16),
    "layer0_attention_scores_head11_tokens0_5_scaled_masked_k_as_b_fp16": ((rf.HEAD_DIM, _TILE_ROWS), np.float16),
    "layer0_attention_scores_head11_tokens0_5_scaled_masked_seed_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    "layer0_attention_scores_head11_tokens0_5_scaled_masked_expected_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
}

_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD12_ARRAY_SPECS = {
    "layer0_attention_scores_head12_tokens0_5_scaled_masked_q_scaled_fp16": ((_TILE_ROWS, rf.HEAD_DIM), np.float16),
    "layer0_attention_scores_head12_tokens0_5_scaled_masked_k_as_b_fp16": ((rf.HEAD_DIM, _TILE_ROWS), np.float16),
    "layer0_attention_scores_head12_tokens0_5_scaled_masked_seed_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    "layer0_attention_scores_head12_tokens0_5_scaled_masked_expected_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
}

_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD9_ARRAY_SPECS = {
    "layer0_attention_scores_head9_tokens0_5_scaled_masked_q_scaled_fp16": (
        (_TILE_ROWS, rf.HEAD_DIM),
        np.float16,
    ),
    "layer0_attention_scores_head9_tokens0_5_scaled_masked_k_as_b_fp16": (
        (rf.HEAD_DIM, _TILE_ROWS),
        np.float16,
    ),
    "layer0_attention_scores_head9_tokens0_5_scaled_masked_seed_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_scores_head9_tokens0_5_scaled_masked_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD6_ARRAY_SPECS = {
    "layer0_attention_probs_head6_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head6_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head6_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD7_ARRAY_SPECS = {
    "layer0_attention_probs_head7_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head7_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head7_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD8_ARRAY_SPECS = {
    "layer0_attention_probs_head8_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head8_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head8_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD10_ARRAY_SPECS = {
    "layer0_attention_probs_head10_tokens0_5_softmax_input_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    "layer0_attention_probs_head10_tokens0_5_softmax_expected_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    "layer0_attention_probs_head10_tokens0_5_softmax_row_sums_fp32": ((_TILE_ROWS,), np.float32),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD11_ARRAY_SPECS = {
    "layer0_attention_probs_head11_tokens0_5_softmax_input_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    "layer0_attention_probs_head11_tokens0_5_softmax_expected_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    "layer0_attention_probs_head11_tokens0_5_softmax_row_sums_fp32": ((_TILE_ROWS,), np.float32),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD12_ARRAY_SPECS = {
    "layer0_attention_probs_head12_tokens0_5_softmax_input_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    "layer0_attention_probs_head12_tokens0_5_softmax_expected_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    "layer0_attention_probs_head12_tokens0_5_softmax_row_sums_fp32": ((_TILE_ROWS,), np.float32),
}

_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD9_ARRAY_SPECS = {
    "layer0_attention_probs_head9_tokens0_5_softmax_input_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head9_tokens0_5_softmax_expected_fp32": (
        (_TILE_ROWS, _TILE_ROWS),
        np.float32,
    ),
    "layer0_attention_probs_head9_tokens0_5_softmax_row_sums_fp32": (
        (_TILE_ROWS,),
        np.float32,
    ),
}

_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD2_TOKENS0_5_COLS128_192_ARRAY_SPECS = {
    "layer0_attention_context_head2_tokens0_5_cols128_192_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head2_tokens0_5_cols128_192_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head2_tokens0_5_cols128_192_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head2_tokens0_5_cols128_192_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}

_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD3_TOKENS0_5_COLS192_256_ARRAY_SPECS = {
    "layer0_attention_context_head3_tokens0_5_cols192_256_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head3_tokens0_5_cols192_256_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head3_tokens0_5_cols192_256_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head3_tokens0_5_cols192_256_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}

_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD4_TOKENS0_5_COLS256_320_ARRAY_SPECS = {
    "layer0_attention_context_head4_tokens0_5_cols256_320_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head4_tokens0_5_cols256_320_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head4_tokens0_5_cols256_320_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head4_tokens0_5_cols256_320_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}

_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD5_TOKENS0_5_COLS320_384_ARRAY_SPECS = {
    "layer0_attention_context_head5_tokens0_5_cols320_384_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head5_tokens0_5_cols320_384_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head5_tokens0_5_cols320_384_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head5_tokens0_5_cols320_384_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}

_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD6_TOKENS0_5_COLS384_448_ARRAY_SPECS = {
    "layer0_attention_context_head6_tokens0_5_cols384_448_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head6_tokens0_5_cols384_448_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head6_tokens0_5_cols384_448_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head6_tokens0_5_cols384_448_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}


_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD7_TOKENS0_5_COLS448_512_ARRAY_SPECS = {
    "layer0_attention_context_head7_tokens0_5_cols448_512_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head7_tokens0_5_cols448_512_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head7_tokens0_5_cols448_512_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head7_tokens0_5_cols448_512_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}


_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD8_TOKENS0_5_COLS512_576_ARRAY_SPECS = {
    "layer0_attention_context_head8_tokens0_5_cols512_576_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head8_tokens0_5_cols512_576_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head8_tokens0_5_cols512_576_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head8_tokens0_5_cols512_576_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}


_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD10_TOKENS0_5_COLS640_704_ARRAY_SPECS = {
    "layer0_attention_context_head10_tokens0_5_cols640_704_probs_fp16": ((_TILE_ROWS, 16), np.float16),
    "layer0_attention_context_head10_tokens0_5_cols640_704_v_as_b_fp16": ((16, 64), np.float16),
    "layer0_attention_context_head10_tokens0_5_cols640_704_expected_fp32": ((_TILE_ROWS, 64), np.float32),
    "layer0_attention_context_head10_tokens0_5_cols640_704_expected_fp16": ((_TILE_ROWS, 64), np.float16),
}

_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD11_TOKENS0_5_COLS704_768_ARRAY_SPECS = {
    "layer0_attention_context_head11_tokens0_5_cols704_768_probs_fp16": ((_TILE_ROWS, 16), np.float16),
    "layer0_attention_context_head11_tokens0_5_cols704_768_v_as_b_fp16": ((16, 64), np.float16),
    "layer0_attention_context_head11_tokens0_5_cols704_768_expected_fp32": ((_TILE_ROWS, 64), np.float32),
    "layer0_attention_context_head11_tokens0_5_cols704_768_expected_fp16": ((_TILE_ROWS, 64), np.float16),
}

_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD12_TOKENS0_5_COLS768_832_ARRAY_SPECS = {
    "layer0_attention_context_head12_tokens0_5_cols768_832_probs_fp16": ((_TILE_ROWS, 16), np.float16),
    "layer0_attention_context_head12_tokens0_5_cols768_832_v_as_b_fp16": ((16, 64), np.float16),
    "layer0_attention_context_head12_tokens0_5_cols768_832_expected_fp32": ((_TILE_ROWS, 64), np.float32),
    "layer0_attention_context_head12_tokens0_5_cols768_832_expected_fp16": ((_TILE_ROWS, 64), np.float16),
}


def _attention_scores_scaled_masked_specs(head: int):
    return {
        f"layer0_attention_scores_head{head}_tokens0_5_scaled_masked_q_scaled_fp16": ((_TILE_ROWS, rf.HEAD_DIM), np.float16),
        f"layer0_attention_scores_head{head}_tokens0_5_scaled_masked_k_as_b_fp16": ((rf.HEAD_DIM, _TILE_ROWS), np.float16),
        f"layer0_attention_scores_head{head}_tokens0_5_scaled_masked_seed_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
        f"layer0_attention_scores_head{head}_tokens0_5_scaled_masked_expected_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
    }


def _attention_probs_softmax_specs(head: int):
    return {
        f"layer0_attention_probs_head{head}_tokens0_5_softmax_input_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
        f"layer0_attention_probs_head{head}_tokens0_5_softmax_expected_fp32": ((_TILE_ROWS, _TILE_ROWS), np.float32),
        f"layer0_attention_probs_head{head}_tokens0_5_softmax_row_sums_fp32": ((_TILE_ROWS,), np.float32),
    }


def _attention_context_specs(head: int, start: int, stop: int):
    prefix = f"layer0_attention_context_head{head}_tokens0_5_cols{start}_{stop}"
    return {
        f"{prefix}_probs_fp16": ((_TILE_ROWS, 16), np.float16),
        f"{prefix}_v_as_b_fp16": ((16, 64), np.float16),
        f"{prefix}_expected_fp32": ((_TILE_ROWS, 64), np.float32),
        f"{prefix}_expected_fp16": ((_TILE_ROWS, 64), np.float16),
    }


_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD13_ARRAY_SPECS = _attention_scores_scaled_masked_specs(13)
_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD14_ARRAY_SPECS = _attention_scores_scaled_masked_specs(14)
_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD15_ARRAY_SPECS = _attention_scores_scaled_masked_specs(15)
_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD16_ARRAY_SPECS = _attention_scores_scaled_masked_specs(16)
_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD17_ARRAY_SPECS = _attention_scores_scaled_masked_specs(17)
_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD18_ARRAY_SPECS = _attention_scores_scaled_masked_specs(18)
_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD19_ARRAY_SPECS = _attention_scores_scaled_masked_specs(19)
_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD20_ARRAY_SPECS = _attention_scores_scaled_masked_specs(20)
_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD13_ARRAY_SPECS = _attention_probs_softmax_specs(13)
_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD14_ARRAY_SPECS = _attention_probs_softmax_specs(14)
_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD15_ARRAY_SPECS = _attention_probs_softmax_specs(15)
_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD16_ARRAY_SPECS = _attention_probs_softmax_specs(16)
_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD17_ARRAY_SPECS = _attention_probs_softmax_specs(17)
_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD18_ARRAY_SPECS = _attention_probs_softmax_specs(18)
_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD19_ARRAY_SPECS = _attention_probs_softmax_specs(19)
_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD20_ARRAY_SPECS = _attention_probs_softmax_specs(20)
_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD13_TOKENS0_5_COLS832_896_ARRAY_SPECS = _attention_context_specs(13, 832, 896)
_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD14_TOKENS0_5_COLS896_960_ARRAY_SPECS = _attention_context_specs(14, 896, 960)
_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD15_TOKENS0_5_COLS960_1024_ARRAY_SPECS = _attention_context_specs(15, 960, 1024)
_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD16_TOKENS0_5_COLS1024_1088_ARRAY_SPECS = _attention_context_specs(16, 1024, 1088)
_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD17_TOKENS0_5_COLS1088_1152_ARRAY_SPECS = _attention_context_specs(17, 1088, 1152)
_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD18_TOKENS0_5_COLS1152_1216_ARRAY_SPECS = _attention_context_specs(18, 1152, 1216)
_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD19_TOKENS0_5_COLS1216_1280_ARRAY_SPECS = _attention_context_specs(19, 1216, 1280)
_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD20_TOKENS0_5_COLS1280_1344_ARRAY_SPECS = _attention_context_specs(20, 1280, 1344)

_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD9_TOKENS0_5_COLS576_640_ARRAY_SPECS = {
    "layer0_attention_context_head9_tokens0_5_cols576_640_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head9_tokens0_5_cols576_640_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head9_tokens0_5_cols576_640_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head9_tokens0_5_cols576_640_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}


_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD0_TOKENS0_5_COLS0_64_ARRAY_SPECS = {
    "layer0_attention_context_head0_tokens0_5_cols0_64_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head0_tokens0_5_cols0_64_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head0_tokens0_5_cols0_64_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head0_tokens0_5_cols0_64_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}
_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD1_TOKENS0_5_COLS64_128_ARRAY_SPECS = {
    "layer0_attention_context_head1_tokens0_5_cols64_128_probs_fp16": (
        (_TILE_ROWS, 16),
        np.float16,
    ),
    "layer0_attention_context_head1_tokens0_5_cols64_128_v_as_b_fp16": (
        (16, 64),
        np.float16,
    ),
    "layer0_attention_context_head1_tokens0_5_cols64_128_expected_fp32": (
        (_TILE_ROWS, 64),
        np.float32,
    ),
    "layer0_attention_context_head1_tokens0_5_cols64_128_expected_fp16": (
        (_TILE_ROWS, 64),
        np.float16,
    ),
}


_EXPECTED_PRIMITIVE_KEYS = {
    "cast_in_fp32": ((16,), np.float32),
    "cast_expected_fp16": ((16,), np.float16),
    "matmul_a_fp16": ((8, 16), np.float16),
    "matmul_b_fp16": ((16, 8), np.float16),
    "matmul_expected_fp16": ((8, 8), np.float16),
    "rms_x_fp16": ((1, 64), np.float16),
    "rms_weight_fp16": ((64,), np.float16),
    "rms_eps": ((), np.float32),
    "rms_expected_fp16": ((1, 64), np.float16),
    "silu_x_fp16": ((8, 8), np.float16),
    "silu_expected_fp16": ((8, 8), np.float16),
}


def _require_fixtures():
    """Skip the whole module gracefully when the fixture dir is absent."""
    if not os.path.isdir(_FIXTURES_DIR):
        pytest.skip("reference fixtures absent (run "
                    "`python3 -m native_r9700.ref_fixtures --generate`)")
    for name in ("prompts.json", "baseline_r_tokens.json", "kv_state.npz",
                 "primitives_fixtures.npz", "fixtures_schema.json"):
        if not os.path.isfile(os.path.join(_FIXTURES_DIR, name)):
            pytest.skip(f"fixture file {name} absent")


_require_fixtures()


def _load(name):
    return os.path.join(_FIXTURES_DIR, name)


# ---------------------------------------------------------------------------
# Global schema / size bounds
# ---------------------------------------------------------------------------
def test_all_fixture_files_small_enough():
    for name in sorted(os.listdir(_FIXTURES_DIR)):
        p = os.path.join(_FIXTURES_DIR, name)
        if not os.path.isfile(p):
            continue
        assert os.path.getsize(p) <= _MAX_FIXTURE_BYTES, (
            f"{name} is {os.path.getsize(p)} bytes, over the {_MAX_FIXTURE_BYTES} "
            "committed-blob bound")


def test_schema_json_matches_disk_digests():
    schema = json.load(open(_load("fixtures_schema.json")))
    expected_files = {
        "prompts.json",
        "baseline_r_tokens.json",
        "kv_state.npz",
        "primitives_fixtures.npz",
        "layer_trace_fixtures.npz",
        "layer_trace_full_inner_projection_fixtures.npz",
        "layer_trace_q_full_inner_projection_fixtures.npz",
        "layer_trace_o_full_inner_projection_fixtures.npz",
        "layer_trace_o_full_inner_projection_cols128_256_fixtures.npz",
        "layer_trace_o_full_inner_projection_cols256_384_fixtures.npz",
        "layer_trace_attention_residual_cols0_64_fixtures.npz",
        "layer_trace_post_attention_rmsnorm_cols0_64_fixtures.npz",
        "layer_trace_mlp_full_inner_projection_fixtures.npz",
        "layer_trace_mlp_activation_cols0_64_fixtures.npz",
        "layer_trace_mlp_activation_full_inner_fixtures.npz",
        "layer_trace_mlp_down_projection_inner_cols0_64_to_cols0_64_fixtures.npz",
        "layer_trace_layer0_post_layer_hidden_fixtures.npz",
    }
    for _case_name, _file_stem, *_rest in _MLP_DOWN_PROJ_FULL_INNER_CASES:
        expected_files.add(f"{_file_stem}_fixtures.npz")
        for _chunk_index in range(4):
            expected_files.add(f"{_file_stem}_chunk{_chunk_index}_fixtures.npz")
    assert set(schema["files"].keys()) == expected_files
    for name, meta in schema["files"].items():
        with open(os.path.join(_FIXTURES_DIR, name), "rb") as fh:
            assert rf.digest_bytes(fh.read()) == meta["sha256"], name
    # Geometry contract from the frozen config.
    g = schema["geometry"]
    assert g["intermediate_size"] == 8192
    assert g["num_layers"] == 16 and g["n_kv_heads"] == 8
    assert g["head_dim"] == 64 and g["hidden_size"] == 2048
    assert g["rms_norm_eps"] == 1e-05


# ---------------------------------------------------------------------------
# prompts.json
# ---------------------------------------------------------------------------
def test_prompts_json_schema_and_s():
    prompts = json.load(open(_load("prompts.json")))
    assert set(prompts.keys()) == {"prompt-0", "prompt-1", "prompt-2", "prompt-16",
                                   "prompt-64", "prompt-128"}
    for name, p in prompts.items():
        assert p["S"] == rf.EXPECTED_S[name] == len(p["token_ids"])
        assert p["text"] == rf.PROMPT_TEXTS[name]
        assert all(isinstance(i, int) for i in p["token_ids"])
    # Token ids are in-vocab (bos 128000 for Llama-3 tokenizer).
    assert prompts["prompt-0"]["token_ids"][0] == 128000

def test_benchmark_prompts_json_schema_and_monotonic_s():
    prompts = json.load(open(_load("benchmark_prompts.json")))
    assert list(prompts.keys()) == ["prompt-1", "prompt-2", "prompt-3", "prompt-4", "prompt-5"]

    core_prompts = json.load(open(_load("prompts.json")))
    assert prompts["prompt-1"] == core_prompts["prompt-1"]
    assert prompts["prompt-2"] == core_prompts["prompt-2"]

    previous_s = 0
    for name, prompt in prompts.items():
        assert prompt["S"] == len(prompt["token_ids"])
        assert prompt["S"] > previous_s, name
        assert prompt["text"]
        assert all(isinstance(token_id, int) for token_id in prompt["token_ids"])
        assert prompt["token_ids"][0] == core_prompts["prompt-0"]["token_ids"][0]
        previous_s = prompt["S"]


# ---------------------------------------------------------------------------
# baseline_r_tokens.json
# ---------------------------------------------------------------------------
def test_r_tokens_json_schema():
    rt = json.load(open(_load("baseline_r_tokens.json")))
    assert set(rt.keys()) == {"prompt-0", "prompt-1", "prompt-2", "prompt-16",
                              "prompt-64", "prompt-128", "_joint_r_tokens_digest"}
    for name in ("prompt-0", "prompt-1", "prompt-2", "prompt-16", "prompt-64", "prompt-128"):
        entry = rt[name]
        assert entry["S"] == rf.EXPECTED_S[name]
        assert entry["max_new_tokens"] == rf.DEFAULT_MAX_NEW_TOKENS
        r = entry["r_tokens"]
        assert isinstance(r, list) and 1 <= len(r) <= rf.DEFAULT_MAX_NEW_TOKENS
        assert all(isinstance(i, int) for i in r)


# ---------------------------------------------------------------------------
# kv_state.npz — per-layer (1,8,N,64) fp16, S-1 prefix + final-token injection
# ---------------------------------------------------------------------------
def test_kv_state_schema_shape_dtype():
    z = np.load(_load("kv_state.npz"))
    schema = json.load(open(_load("fixtures_schema.json")))["files"]["kv_state.npz"]
    n_layers = schema["n_layers"]
    n_prefix = schema["n_prefix"]
    assert n_layers == 16
    # S-1 prefix contract: n_prefix == S - 1 for prompt-0 (S=6).
    assert schema["S"] == 6 and n_prefix == 5
    assert schema["final_token_id"] == json.load(open(_load("prompts.json")))["prompt-0"]["token_ids"][-1]
    assert z.files == [f"layer{i}_{s}" for i in range(n_layers) for s in ("K", "V")]
    for i in range(n_layers):
        for s in ("K", "V"):
            arr = z[f"layer{i}_{s}"]
            assert arr.shape == (1, 8, n_prefix, rf.HEAD_DIM)
            assert arr.dtype == np.float16
    # Sanity: KV is non-trivial (not all zeros) at the first layer.
    assert float(np.abs(z["layer0_K"]).sum()) > 0.0
    assert float(np.abs(z["layer0_V"]).sum()) > 0.0


def test_layer_trace_fixtures_schema_shape_dtype():
    z = np.load(_load("layer_trace_fixtures.npz"))
    schema = json.load(open(_load("fixtures_schema.json")))["files"]["layer_trace_fixtures.npz"]
    assert schema["layers"] == list(_EXPECTED_TRACE_LAYERS)
    assert schema["prompt_name"] == "prompt-0"
    assert schema["token_slice"] == [0, _TRACE_TOKEN_COUNT]
    assert schema["hidden_dim_slice"] == [0, _TRACE_DIM]
    assert schema["head_slice"] == [0, _TRACE_HEAD_COUNT]
    assert schema["head_dim_slice"] == [0, _TRACE_DIM]
    assert schema["additional_trace_slices"] == [
        {
            "name": "layer0_attention_head0_tokens0_5_cols0_64",
            "token_slice": [0, 5],
            "query_head": 0,
            "kv_head": 0,
            "context_hidden_dim_slice": [0, 64],
        },
        {
            "name": "layer0_attention_head1_tokens0_5_cols64_128",
            "token_slice": [0, 5],
            "query_head": 1,
            "kv_head": 0,
            "context_hidden_dim_slice": [64, 128],
        },
        {
            "name": "layer0_attention_head2_tokens0_5_cols128_192",
            "token_slice": [0, 5],
            "query_head": 2,
            "kv_head": 0,
            "context_hidden_dim_slice": [128, 192],
        },
        {
            "name": "layer0_attention_head3_tokens0_5_cols192_256",
            "token_slice": [0, 5],
            "query_head": 3,
            "kv_head": 0,
            "context_hidden_dim_slice": [192, 256],
        },
        {
            "name": "layer0_attention_head4_tokens0_5_cols256_320",
            "token_slice": [0, 5],
            "query_head": 4,
            "kv_head": 1,
            "context_hidden_dim_slice": [256, 320],
        },
        {
            "name": "layer0_attention_head5_tokens0_5_cols320_384",
            "token_slice": [0, 5],
            "query_head": 5,
            "kv_head": 1,
            "context_hidden_dim_slice": [320, 384],
        },
        {
            "name": "layer0_attention_head6_tokens0_5_cols384_448",
            "token_slice": [0, 5],
            "query_head": 6,
            "kv_head": 1,
            "context_hidden_dim_slice": [384, 448],
        },
        {
            "name": "layer0_attention_head7_tokens0_5_cols448_512",
            "token_slice": [0, 5],
            "query_head": 7,
            "kv_head": 1,
            "context_hidden_dim_slice": [448, 512],
        },
        {
            "name": "layer0_attention_head8_tokens0_5_cols512_576",
            "token_slice": [0, 5],
            "query_head": 8,
            "kv_head": 2,
            "context_hidden_dim_slice": [512, 576],
        },
        {
            "name": "layer0_attention_head9_tokens0_5_cols576_640",
            "token_slice": [0, 5],
            "query_head": 9,
            "kv_head": 2,
            "context_hidden_dim_slice": [576, 640],
        },
        {
            "name": "layer0_attention_head10_tokens0_5_cols640_704",
            "token_slice": [0, 5],
            "query_head": 10,
            "kv_head": 2,
            "context_hidden_dim_slice": [640, 704],
        },
        {
            "name": "layer0_attention_head11_tokens0_5_cols704_768",
            "token_slice": [0, 5],
            "query_head": 11,
            "kv_head": 2,
            "context_hidden_dim_slice": [704, 768],
        },
        {
            "name": "layer0_attention_head12_tokens0_5_cols768_832",
            "token_slice": [0, 5],
            "query_head": 12,
            "kv_head": 3,
            "context_hidden_dim_slice": [768, 832],
        },
        {
            "name": "layer0_attention_head13_tokens0_5_cols832_896",
            "token_slice": [0, 5],
            "query_head": 13,
            "kv_head": 3,
            "context_hidden_dim_slice": [832, 896],
        },
        {
            "name": "layer0_attention_head14_tokens0_5_cols896_960",
            "token_slice": [0, 5],
            "query_head": 14,
            "kv_head": 3,
            "context_hidden_dim_slice": [896, 960],
        },
        {
            "name": "layer0_attention_head15_tokens0_5_cols960_1024",
            "token_slice": [0, 5],
            "query_head": 15,
            "kv_head": 3,
            "context_hidden_dim_slice": [960, 1024],
        },
        {
            "name": "layer0_attention_head16_tokens0_5_cols1024_1088",
            "token_slice": [0, 5],
            "query_head": 16,
            "kv_head": 4,
            "context_hidden_dim_slice": [1024, 1088],
        },
        {
            "name": "layer0_attention_head17_tokens0_5_cols1088_1152",
            "token_slice": [0, 5],
            "query_head": 17,
            "kv_head": 4,
            "context_hidden_dim_slice": [1088, 1152],
        },
        {
            "name": "layer0_attention_head18_tokens0_5_cols1152_1216",
            "token_slice": [0, 5],
            "query_head": 18,
            "kv_head": 4,
            "context_hidden_dim_slice": [1152, 1216],
        },
        {
            "name": "layer0_attention_head19_tokens0_5_cols1216_1280",
            "token_slice": [0, 5],
            "query_head": 19,
            "kv_head": 4,
            "context_hidden_dim_slice": [1216, 1280],
        },
        {
            "name": "layer0_attention_head20_tokens0_5_cols1280_1344",
            "token_slice": [0, 5],
            "query_head": 20,
            "kv_head": 5,
            "context_hidden_dim_slice": [1280, 1344],
        },
        {
            "name": "layer0_attention_head21_tokens0_5_cols1344_1408",
            "token_slice": [0, 5],
            "query_head": 21,
            "kv_head": 5,
            "context_hidden_dim_slice": [1344, 1408],
        },
        {
            "name": "layer0_attention_head22_tokens0_5_cols1408_1472",
            "token_slice": [0, 5],
            "query_head": 22,
            "kv_head": 5,
            "context_hidden_dim_slice": [1408, 1472],
        },
        {
            "name": "layer0_attention_head23_tokens0_5_cols1472_1536",
            "token_slice": [0, 5],
            "query_head": 23,
            "kv_head": 5,
            "context_hidden_dim_slice": [1472, 1536],
        },
        {
            "name": "layer0_attention_head24_tokens0_5_cols1536_1600",
            "token_slice": [0, 5],
            "query_head": 24,
            "kv_head": 6,
            "context_hidden_dim_slice": [1536, 1600],
        },
        {
            "name": "layer0_attention_head25_tokens0_5_cols1600_1664",
            "token_slice": [0, 5],
            "query_head": 25,
            "kv_head": 6,
            "context_hidden_dim_slice": [1600, 1664],
        },
        {
            "name": "layer0_attention_head26_tokens0_5_cols1664_1728",
            "token_slice": [0, 5],
            "query_head": 26,
            "kv_head": 6,
            "context_hidden_dim_slice": [1664, 1728],
        },
        {
            "name": "layer0_attention_head27_tokens0_5_cols1728_1792",
            "token_slice": [0, 5],
            "query_head": 27,
            "kv_head": 6,
            "context_hidden_dim_slice": [1728, 1792],
        },
        {
            "name": "layer0_attention_head28_tokens0_5_cols1792_1856",
            "token_slice": [0, 5],
            "query_head": 28,
            "kv_head": 7,
            "context_hidden_dim_slice": [1792, 1856],
        },
        {
            "name": "layer0_attention_head29_tokens0_5_cols1856_1920",
            "token_slice": [0, 5],
            "query_head": 29,
            "kv_head": 7,
            "context_hidden_dim_slice": [1856, 1920],
        },
        {
            "name": "layer0_attention_head30_tokens0_5_cols1920_1984",
            "token_slice": [0, 5],
            "query_head": 30,
            "kv_head": 7,
            "context_hidden_dim_slice": [1920, 1984],
        },
        {
            "name": "layer0_attention_head31_tokens0_5_cols1984_2048",
            "token_slice": [0, 5],
            "query_head": 31,
            "kv_head": 7,
            "context_hidden_dim_slice": [1984, 2048],
        },
    ]

    expected_keys = {
        f"layer{layer_index}_{name}"
        for layer_index in _EXPECTED_TRACE_LAYERS
        for name in _EXPECTED_TRACE_ARRAY_SPECS
    }
    expected_layer0_extra_keys = (
        set(_EXPECTED_LAYER0_TILE_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_FULL_INNER_COLS8_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_FULL_INNER_COLS0_16_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_V_FULL_INNER_COLS8_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_Q_FULL_INNER_COLS8_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ROPE_PAIR_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_K_ROPE_FULL_HEAD_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_K_ROPE_PREFIX_HEAD0_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_Q_ROPE_FULL_HEAD_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_Q_ROPE_PREFIX_HEAD0_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_Q_ROPE_PREFIX_HEAD1_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_Q_ROPE_PREFIX_HEAD2_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORE_RAW_HEAD0_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD0_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD0_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD1_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD1_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD0_TOKENS0_5_COLS0_64_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD1_TOKENS0_5_COLS64_128_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD2_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD2_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD2_TOKENS0_5_COLS128_192_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD3_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD3_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD3_TOKENS0_5_COLS192_256_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD4_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD4_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD4_TOKENS0_5_COLS256_320_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD5_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD5_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD5_TOKENS0_5_COLS320_384_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD6_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD6_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD6_TOKENS0_5_COLS384_448_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD7_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD7_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD7_TOKENS0_5_COLS448_512_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD8_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD8_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD8_TOKENS0_5_COLS512_576_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD9_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD9_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD9_TOKENS0_5_COLS576_640_ARRAY_SPECS)
    )
    expected_schema_keys = expected_keys | expected_layer0_extra_keys
    assert set(z.files) == set(schema["arrays"])
    assert expected_schema_keys <= set(schema["arrays"])
    for layer_index in _EXPECTED_TRACE_LAYERS:
        for name, (shape, dtype) in _EXPECTED_TRACE_ARRAY_SPECS.items():
            key = f"layer{layer_index}_{name}"
            arr = z[key]
            assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
            assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
            assert schema["arrays"][key]["shape"] == list(shape)
            assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_TILE_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_FULL_INNER_COLS8_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_FULL_INNER_COLS0_16_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_V_FULL_INNER_COLS8_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_Q_FULL_INNER_COLS8_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_ROPE_PAIR_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_K_ROPE_FULL_HEAD_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
    for key, (shape, dtype) in _EXPECTED_LAYER0_K_ROPE_PREFIX_HEAD0_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_Q_ROPE_FULL_HEAD_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_Q_ROPE_PREFIX_HEAD0_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_Q_ROPE_PREFIX_HEAD1_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_Q_ROPE_PREFIX_HEAD2_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_ATTENTION_SCORE_RAW_HEAD0_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD0_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for key, (shape, dtype) in _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD0_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))
    for spec in (
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD1_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD1_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD0_TOKENS0_5_COLS0_64_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD1_TOKENS0_5_COLS64_128_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD2_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD2_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD2_TOKENS0_5_COLS128_192_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD3_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD3_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD3_TOKENS0_5_COLS192_256_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD4_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD4_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD4_TOKENS0_5_COLS256_320_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD10_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD10_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD10_TOKENS0_5_COLS640_704_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD11_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD11_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD11_TOKENS0_5_COLS704_768_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD12_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD12_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD12_TOKENS0_5_COLS768_832_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD13_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD13_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD13_TOKENS0_5_COLS832_896_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD14_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD14_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD14_TOKENS0_5_COLS896_960_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD15_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD15_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD15_TOKENS0_5_COLS960_1024_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD16_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD16_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD16_TOKENS0_5_COLS1024_1088_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD17_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD17_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD17_TOKENS0_5_COLS1088_1152_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD18_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD18_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD18_TOKENS0_5_COLS1152_1216_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD19_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD19_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD19_TOKENS0_5_COLS1216_1280_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_SCORES_SCALED_MASKED_HEAD20_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_PROBS_SOFTMAX_HEAD20_ARRAY_SPECS,
        _EXPECTED_LAYER0_ATTENTION_CONTEXT_HEAD20_TOKENS0_5_COLS1280_1344_ARRAY_SPECS,
    ):
        for key, (shape, dtype) in spec.items():
            arr = z[key]
            assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
            assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
            assert schema["arrays"][key]["shape"] == list(shape)
            assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))





def test_layer_trace_full_inner_projection_fixtures_schema_shape_dtype():
    z = np.load(_load("layer_trace_full_inner_projection_fixtures.npz"))
    schema = json.load(open(_load("fixtures_schema.json")))[
        "files"
    ]["layer_trace_full_inner_projection_fixtures.npz"]
    expected_keys = (
        set(_EXPECTED_LAYER0_FULL_INNER_COLS0_64_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_V_FULL_INNER_COLS0_64_ARRAY_SPECS)
    )
    assert schema["layers"] == [0]
    assert schema["prompt_name"] == "prompt-0"
    assert schema["token_slice"] == [0, _TRACE_TOKEN_COUNT]
    assert schema["hidden_dim_slice"] == [0, _FULL_INNER]
    assert set(z.files) == expected_keys
    assert set(schema["arrays"]) == expected_keys
    for spec in (
        _EXPECTED_LAYER0_FULL_INNER_COLS0_64_ARRAY_SPECS,
        _EXPECTED_LAYER0_V_FULL_INNER_COLS0_64_ARRAY_SPECS,
    ):
        for key, (shape, dtype) in spec.items():
            arr = z[key]
            assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
            assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
            assert schema["arrays"][key]["shape"] == list(shape)
            assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))

def test_layer_trace_q_full_inner_projection_fixtures_schema_shape_dtype():
    z = np.load(_load("layer_trace_q_full_inner_projection_fixtures.npz"))
    schema = json.load(open(_load("fixtures_schema.json")))[
        "files"
    ]["layer_trace_q_full_inner_projection_fixtures.npz"]
    expected_keys = set(_EXPECTED_LAYER0_Q_FULL_INNER_COLS0_64_ARRAY_SPECS)
    assert schema["layers"] == [0]
    assert schema["prompt_name"] == "prompt-0"
    assert schema["token_slice"] == [0, _TRACE_TOKEN_COUNT]
    assert schema["hidden_dim_slice"] == [0, _FULL_INNER]
    assert set(z.files) == expected_keys
    assert set(schema["arrays"]) == expected_keys
    for key, (shape, dtype) in _EXPECTED_LAYER0_Q_FULL_INNER_COLS0_64_ARRAY_SPECS.items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))

def test_layer_trace_o_full_inner_projection_fixtures_schema_shape_dtype():
    z = np.load(_load("layer_trace_o_full_inner_projection_fixtures.npz"))
    schema = json.load(open(_load("fixtures_schema.json")))[
        "files"
    ]["layer_trace_o_full_inner_projection_fixtures.npz"]
    expected_keys = (
        set(_EXPECTED_LAYER0_O_FULL_INNER_COLS0_64_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_O_FULL_INNER_COLS64_128_ARRAY_SPECS)
    )
    assert schema["layers"] == [0]
    assert schema["prompt_name"] == "prompt-0"
    assert schema["token_slice"] == [0, _TRACE_TOKEN_COUNT]
    assert schema["hidden_dim_slice"] == [0, _FULL_INNER]
    assert set(z.files) == expected_keys
    assert set(schema["arrays"]) == expected_keys
    for key, (shape, dtype) in (
        _EXPECTED_LAYER0_O_FULL_INNER_COLS0_64_ARRAY_SPECS
        | _EXPECTED_LAYER0_O_FULL_INNER_COLS64_128_ARRAY_SPECS
    ).items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))


def test_layer_trace_o_full_inner_projection_cols128_256_fixtures_schema_shape_dtype():
    z = np.load(_load("layer_trace_o_full_inner_projection_cols128_256_fixtures.npz"))
    schema = json.load(open(_load("fixtures_schema.json")))[
        "files"
    ]["layer_trace_o_full_inner_projection_cols128_256_fixtures.npz"]
    expected_keys = (
        set(_EXPECTED_LAYER0_O_FULL_INNER_COLS128_192_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_O_FULL_INNER_COLS192_256_ARRAY_SPECS)
    )
    assert schema["layers"] == [0]
    assert schema["prompt_name"] == "prompt-0"
    assert schema["token_slice"] == [0, _TRACE_TOKEN_COUNT]
    assert schema["hidden_dim_slice"] == [0, _FULL_INNER]
    assert set(z.files) == expected_keys
    assert set(schema["arrays"]) == expected_keys
    for key, (shape, dtype) in (
        _EXPECTED_LAYER0_O_FULL_INNER_COLS128_192_ARRAY_SPECS
        | _EXPECTED_LAYER0_O_FULL_INNER_COLS192_256_ARRAY_SPECS
    ).items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))


def test_layer_trace_o_full_inner_projection_cols256_384_fixtures_schema_shape_dtype():
    z = np.load(_load("layer_trace_o_full_inner_projection_cols256_384_fixtures.npz"))
    schema = json.load(open(_load("fixtures_schema.json")))[
        "files"
    ]["layer_trace_o_full_inner_projection_cols256_384_fixtures.npz"]
    expected_keys = (
        set(_EXPECTED_LAYER0_O_FULL_INNER_COLS256_320_ARRAY_SPECS)
        | set(_EXPECTED_LAYER0_O_FULL_INNER_COLS320_384_ARRAY_SPECS)
    )
    assert schema["layers"] == [0]
    assert schema["prompt_name"] == "prompt-0"
    assert schema["token_slice"] == [0, _TRACE_TOKEN_COUNT]
    assert schema["hidden_dim_slice"] == [0, _FULL_INNER]
    assert set(z.files) == expected_keys
    assert set(schema["arrays"]) == expected_keys
    for key, (shape, dtype) in (
        _EXPECTED_LAYER0_O_FULL_INNER_COLS256_320_ARRAY_SPECS
        | _EXPECTED_LAYER0_O_FULL_INNER_COLS320_384_ARRAY_SPECS
    ).items():
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        assert schema["arrays"][key]["shape"] == list(shape)
        assert schema["arrays"][key]["dtype"] == str(np.dtype(dtype))

def test_layer_trace_fixtures_schema_documents_attention_head10_cols640_704():
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    slices = schema["files"]["layer_trace_fixtures.npz"]["additional_trace_slices"]
    assert {
        "name": "layer0_attention_head10_tokens0_5_cols640_704",
        "token_slice": [0, 5],
        "query_head": 10,
        "kv_head": 2,
        "context_hidden_dim_slice": [640, 704],
    } in slices


def test_layer_trace_fixtures_schema_documents_attention_head11_cols704_768():
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    slices = schema["files"]["layer_trace_fixtures.npz"]["additional_trace_slices"]
    assert {
        "name": "layer0_attention_head11_tokens0_5_cols704_768",
        "token_slice": [0, 5],
        "query_head": 11,
        "kv_head": 2,
        "context_hidden_dim_slice": [704, 768],
    } in slices


def test_layer_trace_fixtures_are_nontrivial_and_probabilities_normalize():
    z = np.load(_load("layer_trace_fixtures.npz"))
    for layer_index in _EXPECTED_TRACE_LAYERS:
        assert float(np.abs(z[f"layer{layer_index}_hidden_in_fp16"]).sum()) > 0.0
        assert float(np.abs(z[f"layer{layer_index}_final_K_fp16"]).sum()) > 0.0
        assert float(np.abs(z[f"layer{layer_index}_final_V_fp16"]).sum()) > 0.0
        probs = z[f"layer{layer_index}_attention_probs_fp32"]
        assert np.all(np.isfinite(probs))
        assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)

def test_layer0_k_projection_tile_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    a = z["layer0_k_proj_tile_a_fp16"]
    b = z["layer0_k_proj_tile_b_fp16"]
    expected = z["layer0_k_proj_tile_expected_fp32"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _TILE_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    assert np.any(b != np.float16(0.0))
    assert np.array_equal(a.astype(np.float32) @ b.astype(np.float32), expected)



def test_layer0_k_projection_full_inner_cols8_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    a = z["layer0_k_proj_full_inner_cols8_a_fp16"]
    b = z["layer0_k_proj_full_inner_cols8_b_fp16"]
    expected_fp32 = z["layer0_k_proj_full_inner_cols8_expected_fp32"]
    expected_fp16 = z["layer0_k_proj_full_inner_cols8_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    assert np.any(b != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    assert np.array_equal(recomputed_fp32, expected_fp32)
    assert np.array_equal(recomputed_fp32.astype(np.float16), expected_fp16)


def test_layer0_k_projection_full_inner_cols0_16_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    a = z["layer0_k_proj_full_inner_cols0_16_a_fp16"]
    b = z["layer0_k_proj_full_inner_cols0_16_b_fp16"]
    expected_fp32 = z["layer0_k_proj_full_inner_cols0_16_expected_fp32"]
    expected_fp16 = z["layer0_k_proj_full_inner_cols0_16_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    assert np.any(b[:, :_TILE_COLS] != np.float16(0.0))
    assert np.any(b[:, _TILE_COLS:] != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    assert np.array_equal(recomputed_fp32, expected_fp32)
    assert np.array_equal(recomputed_fp32.astype(np.float16), expected_fp16)
    assert np.array_equal(a, z["layer0_k_proj_full_inner_cols8_a_fp16"])
    assert np.array_equal(b[:, :_TILE_COLS], z["layer0_k_proj_full_inner_cols8_b_fp16"])
    assert np.array_equal(expected_fp32[:, :_TILE_COLS], z["layer0_k_proj_full_inner_cols8_expected_fp32"])

def test_layer0_k_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_full_inner_projection_fixtures.npz"))
    trace = np.load(_load("layer_trace_fixtures.npz"))
    a = z["layer0_k_proj_full_inner_cols0_64_a_fp16"]
    b = z["layer0_k_proj_full_inner_cols0_64_b_fp16"]
    expected_fp32 = z["layer0_k_proj_full_inner_cols0_64_expected_fp32"]
    expected_fp16 = z["layer0_k_proj_full_inner_cols0_64_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    for tile in range(8):
        start = tile * _TILE_COLS
        end = start + _TILE_COLS
        assert np.any(b[:, start:end] != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    assert np.array_equal(recomputed_fp32, expected_fp32)
    assert np.array_equal(recomputed_fp32.astype(np.float16), expected_fp16)
    assert np.array_equal(a, trace["layer0_k_proj_full_inner_cols8_a_fp16"])
    assert np.array_equal(b[:, :_TILE_COLS], trace["layer0_k_proj_full_inner_cols8_b_fp16"])
    assert np.array_equal(
        expected_fp32[:, : 2 * _TILE_COLS],
        trace["layer0_k_proj_full_inner_cols0_16_expected_fp32"],
    )


def test_layer0_v_projection_full_inner_cols8_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    a = z["layer0_v_proj_full_inner_cols8_a_fp16"]
    b = z["layer0_v_proj_full_inner_cols8_b_fp16"]
    expected_fp32 = z["layer0_v_proj_full_inner_cols8_expected_fp32"]
    expected_fp16 = z["layer0_v_proj_full_inner_cols8_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    assert np.any(b != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    assert np.array_equal(recomputed_fp32, expected_fp32)
    assert np.array_equal(recomputed_fp32.astype(np.float16), expected_fp16)
    assert np.array_equal(a, z["layer0_k_proj_full_inner_cols8_a_fp16"])


def test_layer0_v_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_full_inner_projection_fixtures.npz"))
    trace = np.load(_load("layer_trace_fixtures.npz"))
    a = z["layer0_v_proj_full_inner_cols0_64_a_fp16"]
    b = z["layer0_v_proj_full_inner_cols0_64_b_fp16"]
    expected_fp32 = z["layer0_v_proj_full_inner_cols0_64_expected_fp32"]
    expected_fp16 = z["layer0_v_proj_full_inner_cols0_64_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    for tile in range(8):
        start = tile * _TILE_COLS
        end = start + _TILE_COLS
        assert np.any(b[:, start:end] != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    assert np.array_equal(recomputed_fp32, expected_fp32)
    assert np.array_equal(recomputed_fp32.astype(np.float16), expected_fp16)
    assert np.array_equal(a, trace["layer0_v_proj_full_inner_cols8_a_fp16"])
    assert np.array_equal(b[:, :_TILE_COLS], trace["layer0_v_proj_full_inner_cols8_b_fp16"])
    assert np.array_equal(expected_fp32[:, :_TILE_COLS], trace["layer0_v_proj_full_inner_cols8_expected_fp32"])


def test_layer0_q_projection_full_inner_cols8_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    a = z["layer0_q_proj_full_inner_cols8_a_fp16"]
    b = z["layer0_q_proj_full_inner_cols8_b_fp16"]
    expected_fp32 = z["layer0_q_proj_full_inner_cols8_expected_fp32"]
    expected_fp16 = z["layer0_q_proj_full_inner_cols8_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    assert np.any(b != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    assert np.array_equal(recomputed_fp32, expected_fp32)
    assert np.array_equal(recomputed_fp32.astype(np.float16), expected_fp16)
    assert np.array_equal(a, z["layer0_k_proj_full_inner_cols8_a_fp16"])

def test_layer0_q_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_q_full_inner_projection_fixtures.npz"))
    trace = np.load(_load("layer_trace_fixtures.npz"))
    a = z["layer0_q_proj_full_inner_cols0_64_a_fp16"]
    b = z["layer0_q_proj_full_inner_cols0_64_b_fp16"]
    expected_fp32 = z["layer0_q_proj_full_inner_cols0_64_expected_fp32"]
    expected_fp16 = z["layer0_q_proj_full_inner_cols0_64_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    for tile in range(8):
        start = tile * _TILE_COLS
        end = start + _TILE_COLS
        assert np.any(b[:, start:end] != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    assert np.array_equal(recomputed_fp32, expected_fp32)
    assert np.array_equal(recomputed_fp32.astype(np.float16), expected_fp16)
    assert np.array_equal(a, trace["layer0_q_proj_full_inner_cols8_a_fp16"])
    assert np.array_equal(b[:, :_TILE_COLS], trace["layer0_q_proj_full_inner_cols8_b_fp16"])
    assert np.array_equal(expected_fp32[:, :_TILE_COLS], trace["layer0_q_proj_full_inner_cols8_expected_fp32"])

def test_layer0_o_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_o_full_inner_projection_fixtures.npz"))
    trace = np.load(_load("layer_trace_fixtures.npz"))
    a = z["layer0_o_proj_full_inner_cols0_64_a_fp16"]
    b = z["layer0_o_proj_full_inner_cols0_64_b_fp16"]
    expected_fp32 = z["layer0_o_proj_full_inner_cols0_64_expected_fp32"]
    expected_fp16 = z["layer0_o_proj_full_inner_cols0_64_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    for tile in range(8):
        start = tile * _TILE_COLS
        end = start + _TILE_COLS
        assert np.any(b[:, start:end] != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(expected_fp32, recomputed_fp32, rtol=0.0, atol=2.5e-7)
    assert np.array_equal(expected_fp32.astype(np.float16), expected_fp16)
    assert np.array_equal(
        expected_fp16[:_TRACE_TOKEN_COUNT, :_TRACE_DIM],
        trace["layer0_o_proj_output_fp16"],
    )

def test_layer0_o_projection_full_inner_cols64_128_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_o_full_inner_projection_fixtures.npz"))
    a = z["layer0_o_proj_full_inner_cols64_128_a_fp16"]
    b = z["layer0_o_proj_full_inner_cols64_128_b_fp16"]
    expected_fp32 = z["layer0_o_proj_full_inner_cols64_128_expected_fp32"]
    expected_fp16 = z["layer0_o_proj_full_inner_cols64_128_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    assert np.array_equal(a, z["layer0_o_proj_full_inner_cols0_64_a_fp16"])
    assert not np.array_equal(b, z["layer0_o_proj_full_inner_cols0_64_b_fp16"])
    for tile in range(8):
        start = tile * _TILE_COLS
        end = start + _TILE_COLS
        assert np.any(b[:, start:end] != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(expected_fp32, recomputed_fp32, rtol=0.0, atol=2.5e-7)
    assert np.array_equal(expected_fp32.astype(np.float16), expected_fp16)


def test_layer0_o_projection_full_inner_cols128_192_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_o_full_inner_projection_cols128_256_fixtures.npz"))
    base_z = np.load(_load("layer_trace_o_full_inner_projection_fixtures.npz"))
    a = z["layer0_o_proj_full_inner_cols128_192_a_fp16"]
    b = z["layer0_o_proj_full_inner_cols128_192_b_fp16"]
    expected_fp32 = z["layer0_o_proj_full_inner_cols128_192_expected_fp32"]
    expected_fp16 = z["layer0_o_proj_full_inner_cols128_192_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    assert np.array_equal(a, base_z["layer0_o_proj_full_inner_cols0_64_a_fp16"])
    assert not np.array_equal(b, base_z["layer0_o_proj_full_inner_cols0_64_b_fp16"])
    assert not np.array_equal(b, base_z["layer0_o_proj_full_inner_cols64_128_b_fp16"])
    for tile in range(8):
        start = tile * _TILE_COLS
        end = start + _TILE_COLS
        assert np.any(b[:, start:end] != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(expected_fp32, recomputed_fp32, rtol=0.0, atol=2.5e-7)
    assert np.array_equal(expected_fp32.astype(np.float16), expected_fp16)


def test_layer0_o_projection_full_inner_cols192_256_fixture_matches_fp32_matmul_oracle():
    z = np.load(_load("layer_trace_o_full_inner_projection_cols128_256_fixtures.npz"))
    base_z = np.load(_load("layer_trace_o_full_inner_projection_fixtures.npz"))
    a = z["layer0_o_proj_full_inner_cols192_256_a_fp16"]
    b = z["layer0_o_proj_full_inner_cols192_256_b_fp16"]
    expected_fp32 = z["layer0_o_proj_full_inner_cols192_256_expected_fp32"]
    expected_fp16 = z["layer0_o_proj_full_inner_cols192_256_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    assert np.array_equal(a, base_z["layer0_o_proj_full_inner_cols0_64_a_fp16"])
    assert not np.array_equal(b, base_z["layer0_o_proj_full_inner_cols0_64_b_fp16"])
    assert not np.array_equal(b, base_z["layer0_o_proj_full_inner_cols64_128_b_fp16"])
    assert not np.array_equal(b, z["layer0_o_proj_full_inner_cols128_192_b_fp16"])
    for tile in range(8):
        start = tile * _TILE_COLS
        end = start + _TILE_COLS
        assert np.any(b[:, start:end] != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(expected_fp32, recomputed_fp32, rtol=0.0, atol=2.5e-7)
    assert np.array_equal(expected_fp32.astype(np.float16), expected_fp16)


@pytest.mark.parametrize(
    ("prefix", "previous_prefix"),
    (
        ("layer0_o_proj_full_inner_cols256_320", "layer0_o_proj_full_inner_cols192_256"),
        ("layer0_o_proj_full_inner_cols320_384", "layer0_o_proj_full_inner_cols256_320"),
    ),
)
def test_layer0_o_projection_full_inner_cols256_384_fixtures_match_fp32_matmul_oracle(
    prefix,
    previous_prefix,
):
    z = np.load(_load("layer_trace_o_full_inner_projection_cols256_384_fixtures.npz"))
    base_z = np.load(_load("layer_trace_o_full_inner_projection_fixtures.npz"))
    prior_z = np.load(_load("layer_trace_o_full_inner_projection_cols128_256_fixtures.npz"))
    a = z[f"{prefix}_a_fp16"]
    b = z[f"{prefix}_b_fp16"]
    expected_fp32 = z[f"{prefix}_expected_fp32"]
    expected_fp16 = z[f"{prefix}_expected_fp16"]

    padded_tail = np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _FULL_INNER), dtype=np.float16)
    assert np.any(a[:_TILE_VALID_ROWS] != np.float16(0.0))
    assert np.array_equal(a[_TILE_VALID_ROWS:], padded_tail)
    assert np.array_equal(a, base_z["layer0_o_proj_full_inner_cols0_64_a_fp16"])
    assert not np.array_equal(b, base_z["layer0_o_proj_full_inner_cols0_64_b_fp16"])
    assert not np.array_equal(b, base_z["layer0_o_proj_full_inner_cols64_128_b_fp16"])
    previous_b_key = f"{previous_prefix}_b_fp16"
    if previous_b_key in z:
        assert not np.array_equal(b, z[previous_b_key])
    else:
        assert not np.array_equal(b, prior_z[previous_b_key])
    for tile in range(8):
        start = tile * _TILE_COLS
        end = start + _TILE_COLS
        assert np.any(b[:, start:end] != np.float16(0.0))
    recomputed_fp32 = np.ascontiguousarray(a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(expected_fp32, recomputed_fp32, rtol=0.0, atol=2.5e-7)
    assert np.array_equal(expected_fp32.astype(np.float16), expected_fp16)

def test_layer_trace_attention_residual_cols0_64_fixtures_schema_shape_dtype():
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    entry = schema["files"]["layer_trace_attention_residual_cols0_64_fixtures.npz"]
    assert entry["kind"] == "npz"
    assert entry["prompt_name"] == "prompt-0"
    assert entry["S"] == 6
    assert entry["n_prefix"] == 5
    assert entry["layers"] == [0]
    assert entry["token_slice"] == [0, 5]
    assert entry["hidden_dim_slice"] == [0, 64]
    arrays = entry["arrays"]
    expected = {
        "layer0_attention_residual_cols0_64_hidden_in_fp16": ([8, 64], "float16"),
        "layer0_attention_residual_cols0_64_o_proj_output_fp16": ([8, 64], "float16"),
        "layer0_attention_residual_cols0_64_expected_fp16": ([8, 64], "float16"),
    }
    assert set(arrays) == set(expected)
    for name, (shape, dtype) in expected.items():
        assert arrays[name]["shape"] == shape
        assert arrays[name]["dtype"] == dtype


def test_layer0_attention_residual_cols0_64_fixture_matches_fp16_add_oracle():
    z = np.load(_load("layer_trace_attention_residual_cols0_64_fixtures.npz"))
    trace = np.load(_load("layer_trace_fixtures.npz"))
    o_proj = np.load(_load("layer_trace_o_full_inner_projection_fixtures.npz"))
    hidden = z["layer0_attention_residual_cols0_64_hidden_in_fp16"]
    projected = z["layer0_attention_residual_cols0_64_o_proj_output_fp16"]
    expected = z["layer0_attention_residual_cols0_64_expected_fp16"]

    assert hidden.shape == (8, 64)
    assert projected.shape == (8, 64)
    assert expected.shape == (8, 64)
    assert np.array_equal((hidden + projected).astype(np.float16), expected)
    assert np.array_equal(projected, o_proj["layer0_o_proj_full_inner_cols0_64_expected_fp16"])
    assert np.array_equal(expected[:_TRACE_TOKEN_COUNT, :_TRACE_DIM], trace["layer0_attention_residual_fp16"])
    assert np.count_nonzero(hidden[5:]) == 0
    assert np.count_nonzero(projected[5:]) == 0
    assert np.count_nonzero(expected[5:]) == 0


def test_layer_trace_post_attention_rmsnorm_cols0_64_fixtures_schema_shape_dtype():
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    entry = schema["files"]["layer_trace_post_attention_rmsnorm_cols0_64_fixtures.npz"]
    assert entry["kind"] == "npz"
    assert entry["prompt_name"] == "prompt-0"
    assert entry["S"] == 6
    assert entry["n_prefix"] == 5
    assert entry["layers"] == [0]
    assert entry["token_slice"] == [0, 5]
    assert entry["input_hidden_dim_slice"] == [0, rf.HIDDEN_SIZE]
    assert entry["output_hidden_dim_slice"] == [0, 64]
    arrays = entry["arrays"]
    expected = {
        "layer0_post_attention_rmsnorm_cols0_64_residual_in_fp16": ([8, rf.HIDDEN_SIZE], "float16"),
        "layer0_post_attention_rmsnorm_cols0_64_weight_fp16": ([rf.HIDDEN_SIZE], "float16"),
        "layer0_post_attention_rmsnorm_cols0_64_expected_fp16": ([8, 64], "float16"),
    }
    assert set(arrays) == set(expected)
    for name, (shape, dtype) in expected.items():
        assert arrays[name]["shape"] == shape
        assert arrays[name]["dtype"] == dtype


def test_layer0_post_attention_rmsnorm_cols0_64_fixture_matches_full_hidden_oracle():
    fixture_path = _load("layer_trace_post_attention_rmsnorm_cols0_64_fixtures.npz")
    with open(fixture_path, "rb") as fh:
        assert rf.digest_bytes(fh.read()) == _POST_ATTENTION_RMSNORM_COLS0_64_FIXTURE_SHA256
    z = np.load(fixture_path)
    trace = np.load(_load("layer_trace_fixtures.npz"))
    residual = z["layer0_post_attention_rmsnorm_cols0_64_residual_in_fp16"]
    weight = z["layer0_post_attention_rmsnorm_cols0_64_weight_fp16"]
    expected = z["layer0_post_attention_rmsnorm_cols0_64_expected_fp16"]

    assert residual.shape == (8, rf.HIDDEN_SIZE)
    assert weight.shape == (rf.HIDDEN_SIZE,)
    assert expected.shape == (8, 64)
    assert (
        rf.digest_bytes(expected.tobytes())
        == _POST_ATTENTION_RMSNORM_COLS0_64_EXPECTED_FP16_SHA256
    )
    assert np.count_nonzero(residual[:_TILE_VALID_ROWS, 64:]) > 0
    assert np.count_nonzero(residual[_TILE_VALID_ROWS:]) == 0
    mean_sq = np.mean(residual.astype(np.float32) * residual.astype(np.float32), axis=-1, keepdims=True)
    recomputed = (residual.astype(np.float32) / np.sqrt(mean_sq + np.float32(1e-5))) * weight.astype(np.float32)
    np.testing.assert_array_equal(recomputed.astype(np.float16)[:, :64], expected)
    assert np.array_equal(expected[:_TRACE_TOKEN_COUNT, :_TRACE_DIM], trace["layer0_post_norm_fp16"])

def test_layer_trace_mlp_full_inner_projection_fixtures_schema_shape_dtype():
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    entry = schema["files"]["layer_trace_mlp_full_inner_projection_fixtures.npz"]
    assert entry["kind"] == "npz"
    assert entry["prompt_name"] == "prompt-0"
    assert entry["S"] == 6
    assert entry["n_prefix"] == 5
    assert entry["layers"] == [0]
    assert entry["token_slice"] == [0, _TRACE_TOKEN_COUNT]
    assert entry["hidden_dim_slice"] == [0, rf.HIDDEN_SIZE]
    assert entry["output_hidden_dim_slice"] == [0, 64]
    arrays = entry["arrays"]
    expected = {
        "layer0_mlp_gate_proj_full_inner_cols0_64_a_fp16": ([8, rf.HIDDEN_SIZE], "float16"),
        "layer0_mlp_gate_proj_full_inner_cols0_64_b_fp16": ([rf.HIDDEN_SIZE, 64], "float16"),
        "layer0_mlp_gate_proj_full_inner_cols0_64_expected_fp32": ([8, 64], "float32"),
        "layer0_mlp_gate_proj_full_inner_cols0_64_expected_fp16": ([8, 64], "float16"),
        "layer0_mlp_up_proj_full_inner_cols0_64_a_fp16": ([8, rf.HIDDEN_SIZE], "float16"),
        "layer0_mlp_up_proj_full_inner_cols0_64_b_fp16": ([rf.HIDDEN_SIZE, 64], "float16"),
        "layer0_mlp_up_proj_full_inner_cols0_64_expected_fp32": ([8, 64], "float32"),
        "layer0_mlp_up_proj_full_inner_cols0_64_expected_fp16": ([8, 64], "float16"),
    }
    assert set(arrays) == set(expected)
    for name, (shape, dtype) in expected.items():
        assert arrays[name]["shape"] == shape
        assert arrays[name]["dtype"] == dtype


def test_layer0_mlp_gate_up_cols0_64_fixtures_match_fp32_matmul_oracle():
    fixture_path = _load("layer_trace_mlp_full_inner_projection_fixtures.npz")
    with open(fixture_path, "rb") as fh:
        assert rf.digest_bytes(fh.read()) == _MLP_FULL_INNER_PROJECTION_FIXTURE_SHA256
    z = np.load(fixture_path)
    trace = np.load(_load("layer_trace_fixtures.npz"))

    for projection, expected_sha in (
        ("gate", _MLP_GATE_PROJ_COLS0_64_EXPECTED_FP16_SHA256),
        ("up", _MLP_UP_PROJ_COLS0_64_EXPECTED_FP16_SHA256),
    ):
        prefix = f"layer0_mlp_{projection}_proj_full_inner_cols0_64"
        a = z[f"{prefix}_a_fp16"]
        b = z[f"{prefix}_b_fp16"]
        expected_fp32 = z[f"{prefix}_expected_fp32"]
        expected_fp16 = z[f"{prefix}_expected_fp16"]

        assert a.shape == (8, rf.HIDDEN_SIZE)
        assert b.shape == (rf.HIDDEN_SIZE, 64)
        assert expected_fp32.shape == (8, 64)
        assert expected_fp16.shape == (8, 64)
        assert rf.digest_bytes(expected_fp16.tobytes()) == expected_sha
        np.testing.assert_allclose(
            a.astype(np.float32) @ b.astype(np.float32),
            expected_fp32,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(expected_fp32.astype(np.float16), expected_fp16)
        trace_key = f"{projection}_proj_fp16"
        assert np.array_equal(
            expected_fp16[:_TRACE_TOKEN_COUNT, :_TRACE_DIM],
            trace[f"layer0_{trace_key}"],
        )
        assert np.count_nonzero(a[:_TILE_VALID_ROWS, 64:]) > 0
        assert np.count_nonzero(a[_TILE_VALID_ROWS:]) == 0
        assert np.count_nonzero(expected_fp16[_TILE_VALID_ROWS:]) == 0


def test_layer0_post_layer_hidden_full_width_fixture_schema_shape_dtype():
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    entry = schema["files"]["layer_trace_layer0_post_layer_hidden_fixtures.npz"]
    assert entry["kind"] == "npz"
    assert entry["prompt_name"] == "prompt-0"
    assert entry["S"] == 6
    assert entry["n_prefix"] == 5
    assert entry["layers"] == [0]
    assert entry["token_slice"] == [0, 5]
    assert entry["hidden_dim_slice"] == [0, rf.HIDDEN_SIZE]
    assert entry["arrays"] == {
        "layer0_post_layer_hidden_fp16": {
            "shape": [8, rf.HIDDEN_SIZE],
            "dtype": "float16",
        }
    }
    assert (
        entry["note"]
        == "layer-0 post-layer hidden state rows0:8 full hidden width after full attention and MLP residual; pads rows beyond prompt prefix"
    )


def test_layer0_post_layer_hidden_full_width_fixture_matches_compact_trace():
    fixture_path = _load("layer_trace_layer0_post_layer_hidden_fixtures.npz")
    with open(fixture_path, "rb") as fh:
        assert rf.digest_bytes(fh.read()) == _LAYER0_POST_LAYER_HIDDEN_FIXTURE_SHA256
    z = np.load(fixture_path)
    trace = np.load(_load("layer_trace_fixtures.npz"))
    hidden = z["layer0_post_layer_hidden_fp16"]
    assert hidden.shape == (8, rf.HIDDEN_SIZE)
    assert hidden.dtype == np.float16
    np.testing.assert_array_equal(
        hidden[:_TRACE_TOKEN_COUNT, :_TRACE_DIM],
        trace["layer0_mlp_residual_out_fp16"],
    )
    assert np.count_nonzero(hidden[:_TILE_VALID_ROWS, _TRACE_DIM:]) > 0
    assert np.count_nonzero(hidden[_TILE_VALID_ROWS:]) == 0

def test_layer0_mlp_activation_cols0_64_fixture_matches_silu_multiply_oracle():
    fixture_path = _load("layer_trace_mlp_activation_cols0_64_fixtures.npz")
    z = np.load(fixture_path)
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    entry = schema["files"]["layer_trace_mlp_activation_cols0_64_fixtures.npz"]
    arrays = entry["arrays"]
    expected_schema = {
        "layer0_mlp_activation_cols0_64_gate_fp16": ([8, 64], "float16"),
        "layer0_mlp_activation_cols0_64_up_fp16": ([8, 64], "float16"),
        "layer0_mlp_activation_cols0_64_expected_fp16": ([8, 64], "float16"),
    }
    assert set(arrays) == set(expected_schema)
    for name, (shape, dtype) in expected_schema.items():
        assert arrays[name]["shape"] == shape
        assert arrays[name]["dtype"] == dtype
        assert z[name].shape == tuple(shape)

    gate = z["layer0_mlp_activation_cols0_64_gate_fp16"]
    up = z["layer0_mlp_activation_cols0_64_up_fp16"]
    expected = z["layer0_mlp_activation_cols0_64_expected_fp16"]
    gate_fp32 = gate.astype(np.float32)
    silu_gate_fp16 = (
        gate_fp32 / (np.float32(1.0) + np.exp(-gate_fp32))
    ).astype(np.float16)
    recomputed = (silu_gate_fp16 * up).astype(np.float16)
    np.testing.assert_array_equal(recomputed, expected)
    assert np.count_nonzero(gate[_TILE_VALID_ROWS:]) == 0
    assert np.count_nonzero(up[_TILE_VALID_ROWS:]) == 0
    assert np.count_nonzero(expected[_TILE_VALID_ROWS:]) == 0

def test_layer0_mlp_activation_full_inner_fixture_schema_shape_dtype():
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    entry = schema["files"]["layer_trace_mlp_activation_full_inner_fixtures.npz"]
    assert entry["kind"] == "npz"
    assert entry["prompt_name"] == "prompt-0"
    assert entry["S"] == 6
    assert entry["n_prefix"] == 5
    assert entry["layers"] == [0]
    assert entry["token_slice"] == [0, 5]
    assert entry["hidden_dim_slice"] == []
    assert entry["inner_dim_slice"] == [0, rf.INTERMEDIATE_SIZE]
    arrays = entry["arrays"]
    expected_schema = {
        "layer0_mlp_activation_full_inner_gate_fp16": ([8, rf.INTERMEDIATE_SIZE], "float16"),
        "layer0_mlp_activation_full_inner_up_fp16": ([8, rf.INTERMEDIATE_SIZE], "float16"),
        "layer0_mlp_activation_full_inner_expected_fp16": ([8, rf.INTERMEDIATE_SIZE], "float16"),
    }
    assert set(arrays) == set(expected_schema)
    for name, (shape, dtype) in expected_schema.items():
        assert arrays[name]["shape"] == shape
        assert arrays[name]["dtype"] == dtype


def test_layer0_mlp_activation_full_inner_fixture_matches_silu_multiply_oracle():
    fixture_path = _load("layer_trace_mlp_activation_full_inner_fixtures.npz")
    with open(fixture_path, "rb") as fh:
        assert rf.digest_bytes(fh.read()) == _MLP_ACTIVATION_FULL_INNER_FIXTURE_SHA256
    z = np.load(fixture_path)
    compact = np.load(_load("layer_trace_mlp_activation_cols0_64_fixtures.npz"))
    trace = np.load(_load("layer_trace_fixtures.npz"))
    gate = z["layer0_mlp_activation_full_inner_gate_fp16"]
    up = z["layer0_mlp_activation_full_inner_up_fp16"]
    expected = z["layer0_mlp_activation_full_inner_expected_fp16"]
    assert gate.shape == (8, rf.INTERMEDIATE_SIZE)
    assert up.shape == (8, rf.INTERMEDIATE_SIZE)
    assert expected.shape == (8, rf.INTERMEDIATE_SIZE)
    gate_fp32 = gate.astype(np.float32)
    silu_gate_fp16 = (
        gate_fp32 / (np.float32(1.0) + np.exp(-gate_fp32))
    ).astype(np.float16)
    np.testing.assert_array_equal((silu_gate_fp16 * up).astype(np.float16), expected)
    np.testing.assert_array_equal(
        gate[:, :64], compact["layer0_mlp_activation_cols0_64_gate_fp16"]
    )
    np.testing.assert_array_equal(
        up[:, :64], compact["layer0_mlp_activation_cols0_64_up_fp16"]
    )
    np.testing.assert_array_equal(
        expected[:, :64],
        compact["layer0_mlp_activation_cols0_64_expected_fp16"],
    )
    np.testing.assert_array_equal(
        expected[:_TRACE_TOKEN_COUNT, :_TRACE_DIM],
        trace["layer0_gated_mlp_fp16"],
    )
    assert np.count_nonzero(expected[:_TILE_VALID_ROWS, 64:]) > 0
    assert np.count_nonzero(gate[_TILE_VALID_ROWS:]) == 0
    assert np.count_nonzero(up[_TILE_VALID_ROWS:]) == 0
    assert np.count_nonzero(expected[_TILE_VALID_ROWS:]) == 0


def test_layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_fixture_matches_partial_fp32_oracle():
    fixture_path = _load("layer_trace_mlp_down_projection_inner_cols0_64_to_cols0_64_fixtures.npz")
    with open(fixture_path, "rb") as fh:
        assert (
            rf.digest_bytes(fh.read())
            == _MLP_DOWN_PROJ_INNER_COLS0_64_TO_COLS0_64_FIXTURE_SHA256
        )
    z = np.load(fixture_path)
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    entry = schema["files"][
        "layer_trace_mlp_down_projection_inner_cols0_64_to_cols0_64_fixtures.npz"
    ]
    arrays = entry["arrays"]
    expected_schema = {
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_activation_fp16": (
            [8, 64],
            "float16",
        ),
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_weight_fp16": (
            [64, 64],
            "float16",
        ),
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_expected_fp32": (
            [8, 64],
            "float32",
        ),
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_expected_fp16": (
            [8, 64],
            "float16",
        ),
    }
    assert set(arrays) == set(expected_schema)
    for name, (shape, dtype) in expected_schema.items():
        assert arrays[name]["shape"] == shape
        assert arrays[name]["dtype"] == dtype
        assert z[name].shape == tuple(shape)

    activation = z[
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_activation_fp16"
    ]
    weight = z["layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_weight_fp16"]
    expected_fp32 = z[
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_expected_fp32"
    ]
    expected_fp16 = z[
        "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_expected_fp16"
    ]
    assert (
        rf.digest_bytes(expected_fp32.tobytes())
        == _MLP_DOWN_PROJ_INNER_COLS0_64_TO_COLS0_64_EXPECTED_FP32_SHA256
    )
    np.testing.assert_allclose(
        activation.astype(np.float32) @ weight.astype(np.float32),
        expected_fp32,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(expected_fp32.astype(np.float16), expected_fp16)
    assert np.count_nonzero(activation[:_TILE_VALID_ROWS]) > 0
    assert np.count_nonzero(weight) > 0
    assert np.count_nonzero(activation[_TILE_VALID_ROWS:]) == 0
    assert np.count_nonzero(expected_fp32[_TILE_VALID_ROWS:]) == 0


@pytest.mark.parametrize(
    (
        "array_prefix",
        "file_stem",
        "output_slice",
        "_fixture_sha256",
        "_expected_fp32_sha256",
        "_matches_compact_trace",
    ),
    _MLP_DOWN_PROJ_FULL_INNER_CASES,
)
def test_layer0_mlp_down_proj_full_inner_chunk_fixtures_schema_shape_dtype(
    array_prefix,
    file_stem,
    output_slice,
    _fixture_sha256,
    _expected_fp32_sha256,
    _matches_compact_trace,
):
    del _fixture_sha256, _expected_fp32_sha256, _matches_compact_trace
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    for chunk_index in range(4):
        entry = schema["files"][f"{file_stem}_chunk{chunk_index}_fixtures.npz"]
        assert entry["kind"] == "npz"
        assert entry["prompt_name"] == "prompt-0"
        assert entry["S"] == 6
        assert entry["n_prefix"] == 5
        assert entry["layers"] == [0]
        assert entry["token_slice"] == [0, 5]
        assert entry["inner_dim_slice"] == [chunk_index * 2048, (chunk_index + 1) * 2048]
        assert entry["output_hidden_dim_slice"] == list(output_slice)
        chunk_prefix = f"{array_prefix}_chunk{chunk_index}"
        expected_schema = {
            f"{chunk_prefix}_activation_fp16": ([8, 2048], "float16"),
            f"{chunk_prefix}_weight_fp16": ([2048, 64], "float16"),
            f"{chunk_prefix}_expected_fp32": ([8, 64], "float32"),
            f"{chunk_prefix}_expected_fp16": ([8, 64], "float16"),
        }
        assert set(entry["arrays"]) == set(expected_schema)
        for name, (shape, dtype) in expected_schema.items():
            assert entry["arrays"][name]["shape"] == shape
            assert entry["arrays"][name]["dtype"] == dtype


@pytest.mark.parametrize(
    (
        "array_prefix",
        "file_stem",
        "output_slice",
        "fixture_sha256",
        "expected_fp32_sha256",
        "matches_compact_trace",
    ),
    _MLP_DOWN_PROJ_FULL_INNER_CASES,
)
def test_layer0_mlp_down_proj_full_inner_fixtures_match_fp32_oracle(
    array_prefix,
    file_stem,
    output_slice,
    fixture_sha256,
    expected_fp32_sha256,
    matches_compact_trace,
):
    schema = json.loads(Path(_load("fixtures_schema.json")).read_text(encoding="utf-8"))
    final_path = _load(f"{file_stem}_fixtures.npz")
    with open(final_path, "rb") as fh:
        assert rf.digest_bytes(fh.read()) == fixture_sha256
    final = np.load(final_path)
    trace = np.load(_load("layer_trace_fixtures.npz"))
    activation_full = np.load(_load("layer_trace_mlp_activation_full_inner_fixtures.npz"))
    activation_expected = activation_full["layer0_mlp_activation_full_inner_expected_fp16"]
    accumulated = np.zeros((8, 64), dtype=np.float32)
    for chunk_index in range(4):
        chunk_path = _load(f"{file_stem}_chunk{chunk_index}_fixtures.npz")
        z = np.load(chunk_path)
        chunk_prefix = f"{array_prefix}_chunk{chunk_index}"
        activation = z[f"{chunk_prefix}_activation_fp16"]
        weight = z[f"{chunk_prefix}_weight_fp16"]
        expected_fp32 = z[f"{chunk_prefix}_expected_fp32"]
        expected_fp16 = z[f"{chunk_prefix}_expected_fp16"]
        start, stop = schema["files"][f"{file_stem}_chunk{chunk_index}_fixtures.npz"][
            "inner_dim_slice"
        ]
        np.testing.assert_array_equal(activation, activation_expected[:, start:stop])
        np.testing.assert_allclose(
            activation.astype(np.float32) @ weight.astype(np.float32),
            expected_fp32,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(expected_fp32.astype(np.float16), expected_fp16)
        accumulated += expected_fp32
    assert schema["files"][f"{file_stem}_fixtures.npz"]["output_hidden_dim_slice"] == list(
        output_slice
    )
    expected_fp32 = final[f"{array_prefix}_expected_fp32"]
    expected_fp16 = final[f"{array_prefix}_expected_fp16"]
    assert rf.digest_bytes(expected_fp32.tobytes()) == expected_fp32_sha256
    np.testing.assert_allclose(accumulated, expected_fp32, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(expected_fp32.astype(np.float16), expected_fp16)
    if matches_compact_trace:
        np.testing.assert_array_equal(
            expected_fp16[:_TRACE_TOKEN_COUNT, :_TRACE_DIM],
            trace["layer0_down_proj_output_fp16"],
        )
    else:
        assert np.count_nonzero(expected_fp32[:_TILE_VALID_ROWS]) > 0
    assert np.count_nonzero(expected_fp32[_TILE_VALID_ROWS:]) == 0



def test_layer0_k_rope_pair_slice_fixture_matches_split_half_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    packed = z["layer0_k_rope_pairs12_20_input_fp16"]
    cos = z["layer0_k_rope_pairs12_20_cos_fp32"]
    sin = z["layer0_k_rope_pairs12_20_sin_fp32"]
    expected = z["layer0_k_rope_pairs12_20_expected_fp16"]

    assert packed.shape == (2, _ROPE_PAIR_COUNT)
    assert np.any(sin != np.float32(0.0))
    left = packed[0].astype(np.float32)
    right = packed[1].astype(np.float32)
    recomputed = np.stack(
        (
            left * cos - right * sin,
            right * cos + left * sin,
        ),
        axis=0,
    ).astype(np.float16)
    assert np.array_equal(recomputed, expected)

def test_layer0_k_rope_tokens0_5_head0_full_head_fixture_matches_split_half_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    packed = z["layer0_k_rope_tokens0_5_head0_full_head_input_fp16"]
    cos = z["layer0_k_rope_tokens0_5_head0_full_head_cos_fp32"]
    sin = z["layer0_k_rope_tokens0_5_head0_full_head_sin_fp32"]
    expected = z["layer0_k_rope_tokens0_5_head0_full_head_expected_fp16"]

    assert packed.shape == (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT)
    assert np.all(sin[0] == np.float32(0.0))
    assert np.any(sin[1:] != np.float32(0.0))
    left = packed[:, 0, :].astype(np.float32)
    right = packed[:, 1, :].astype(np.float32)
    recomputed = np.stack(
        (
            left * cos - right * sin,
            right * cos + left * sin,
        ),
        axis=1,
    ).astype(np.float16)
    assert np.array_equal(recomputed, expected)
    assert np.array_equal(expected[1], z["layer0_k_rope_token1_head0_full_head_expected_fp16"])


def test_layer0_k_rope_token1_head0_full_head_fixture_matches_split_half_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    packed = z["layer0_k_rope_token1_head0_full_head_input_fp16"]
    cos = z["layer0_k_rope_token1_head0_full_head_cos_fp32"]
    sin = z["layer0_k_rope_token1_head0_full_head_sin_fp32"]
    expected = z["layer0_k_rope_token1_head0_full_head_expected_fp16"]

    assert packed.shape == (2, _ROPE_FULL_HEAD_PAIR_COUNT)
    assert np.any(sin != np.float32(0.0))
    left = packed[0].astype(np.float32)
    right = packed[1].astype(np.float32)
    recomputed = np.stack(
        (
            left * cos - right * sin,
            right * cos + left * sin,
        ),
        axis=0,
    ).astype(np.float16)
    assert np.array_equal(recomputed, expected)






def test_layer0_q_rope_token1_head0_full_head_fixture_matches_split_half_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    packed = z["layer0_q_rope_token1_head0_full_head_input_fp16"]
    cos = z["layer0_q_rope_token1_head0_full_head_cos_fp32"]
    sin = z["layer0_q_rope_token1_head0_full_head_sin_fp32"]
    expected = z["layer0_q_rope_token1_head0_full_head_expected_fp16"]

    assert packed.shape == (2, _ROPE_FULL_HEAD_PAIR_COUNT)
    assert np.any(sin != np.float32(0.0))
    left = packed[0].astype(np.float32)
    right = packed[1].astype(np.float32)
    recomputed = np.stack(
        (
            left * cos - right * sin,
            right * cos + left * sin,
        ),
        axis=0,
    ).astype(np.float16)
    assert np.array_equal(recomputed, expected)

def test_layer0_q_rope_tokens0_5_head0_full_head_fixture_matches_split_half_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    packed = z["layer0_q_rope_tokens0_5_head0_full_head_input_fp16"]
    cos = z["layer0_q_rope_tokens0_5_head0_full_head_cos_fp32"]
    sin = z["layer0_q_rope_tokens0_5_head0_full_head_sin_fp32"]
    expected = z["layer0_q_rope_tokens0_5_head0_full_head_expected_fp16"]

    assert packed.shape == (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT)
    assert np.all(sin[0] == np.float32(0.0))
    assert np.any(sin[1:] != np.float32(0.0))
    left = packed[:, 0, :].astype(np.float32)
    right = packed[:, 1, :].astype(np.float32)
    recomputed = np.stack(
        (
            left * cos - right * sin,
            right * cos + left * sin,
        ),
        axis=1,
    ).astype(np.float16)
    assert np.array_equal(recomputed, expected)
    assert np.array_equal(expected[1], z["layer0_q_rope_token1_head0_full_head_expected_fp16"])

def test_layer0_q_rope_tokens0_5_head1_full_head_fixture_matches_split_half_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    packed = z["layer0_q_rope_tokens0_5_head1_full_head_input_fp16"]
    cos = z["layer0_q_rope_tokens0_5_head1_full_head_cos_fp32"]
    sin = z["layer0_q_rope_tokens0_5_head1_full_head_sin_fp32"]
    expected = z["layer0_q_rope_tokens0_5_head1_full_head_expected_fp16"]
    trace_head1 = z["layer0_q_rope_fp16"][0, 1]

    assert packed.shape == (5, 2, _ROPE_FULL_HEAD_PAIR_COUNT)
    assert np.all(sin[0] == np.float32(0.0))
    assert np.any(sin[1:] != np.float32(0.0))
    left = packed[:, 0, :].astype(np.float32)
    right = packed[:, 1, :].astype(np.float32)
    recomputed = np.stack(
        (
            left * cos - right * sin,
            right * cos + left * sin,
        ),
        axis=1,
    ).astype(np.float16)
    assert np.array_equal(recomputed, expected)
    q_full = np.concatenate((expected[:, 0, :], expected[:, 1, :]), axis=1)
    assert np.array_equal(q_full[:_TRACE_TOKEN_COUNT, :_TRACE_DIM], trace_head1)


def test_layer0_attention_score_raw_head0_tokens0_5_fixture_matches_rope_dot_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    q = z["layer0_attention_score_raw_head0_tokens0_5_q_fp16"]
    k_as_b = z["layer0_attention_score_raw_head0_tokens0_5_k_as_b_fp16"]
    expected = z["layer0_attention_score_raw_head0_tokens0_5_expected_fp32"]
    q_rope = z["layer0_q_rope_tokens0_5_head0_full_head_expected_fp16"]
    k_rope = z["layer0_k_rope_tokens0_5_head0_full_head_expected_fp16"]

    q_full = np.concatenate((q_rope[:, 0, :], q_rope[:, 1, :]), axis=1)
    k_full = np.concatenate((k_rope[:, 0, :], k_rope[:, 1, :]), axis=1)
    assert np.array_equal(q[:5], q_full)
    assert np.array_equal(k_as_b[:, :5], k_full.T)
    assert np.all(q[5:] == np.float16(0.0))
    assert np.all(k_as_b[:, 5:] == np.float16(0.0))
    recomputed = q.astype(np.float32) @ k_as_b.astype(np.float32)
    assert np.array_equal(recomputed, expected)


def test_layer0_attention_score_raw_head0_tokens0_5_fixture_matches_scaled_trace_cells():
    z = np.load(_load("layer_trace_fixtures.npz"))
    raw = z["layer0_attention_score_raw_head0_tokens0_5_expected_fp32"]
    scaled = raw * np.float32(1.0 / np.sqrt(rf.HEAD_DIM))
    trace = z["layer0_attention_scores_fp32"][0, 0]

    np.testing.assert_allclose(scaled[0, :1], trace[0, :1], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(scaled[1, :2], trace[1, :2], rtol=0.0, atol=1e-6)
    assert np.all(~np.isfinite(trace[0, 1:]))
    assert np.all(~np.isfinite(trace[1, 2:]))

def test_layer0_attention_scores_head0_tokens0_5_scaled_masked_fixture_matches_trace_and_seeded_accum_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    q_scaled = z["layer0_attention_scores_head0_tokens0_5_scaled_masked_q_scaled_fp16"]
    k_as_b = z["layer0_attention_scores_head0_tokens0_5_scaled_masked_k_as_b_fp16"]
    seed = z["layer0_attention_scores_head0_tokens0_5_scaled_masked_seed_fp32"]
    expected = z["layer0_attention_scores_head0_tokens0_5_scaled_masked_expected_fp32"]
    q_raw = z["layer0_attention_score_raw_head0_tokens0_5_q_fp16"]
    k_raw = z["layer0_attention_score_raw_head0_tokens0_5_k_as_b_fp16"]
    trace = z["layer0_attention_scores_fp32"][0, 0]

    assert np.array_equal(q_scaled, (q_raw.astype(np.float32) * np.float32(0.125)).astype(np.float16))
    assert np.array_equal(k_as_b, k_raw)

    valid_tokens = 5
    seed_expected = np.full((_TILE_ROWS, _TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(valid_tokens):
        seed_expected[row, : row + 1] = np.float32(0.0)
    assert np.array_equal(seed, seed_expected)

    recomputed = q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    assert np.array_equal(recomputed, expected)
    np.testing.assert_allclose(expected[0, :1], trace[0, :1], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(expected[1, :2], trace[1, :2], rtol=0.0, atol=1e-6)
    assert np.all(~np.isfinite(expected[0, 1:]))
    assert np.all(~np.isfinite(expected[1, 2:]))


def test_layer0_attention_probs_head0_tokens0_5_softmax_fixture_matches_scaled_masked_softmax_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    scores = z["layer0_attention_probs_head0_tokens0_5_softmax_input_fp32"]
    expected = z["layer0_attention_probs_head0_tokens0_5_softmax_expected_fp32"]
    row_sums = z["layer0_attention_probs_head0_tokens0_5_softmax_row_sums_fp32"]
    scaled_masked = z["layer0_attention_scores_head0_tokens0_5_scaled_masked_expected_fp32"]

    assert np.array_equal(scores, scaled_masked)
    recomputed = np.zeros_like(scores, dtype=np.float32)
    recomputed_row_sums = np.zeros((_TILE_ROWS,), dtype=np.float32)
    for row in range(_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        recomputed[row] = (exps / denom).astype(np.float32)
        recomputed_row_sums[row] = np.sum(recomputed[row], dtype=np.float32)

    assert np.all(np.isfinite(expected))
    assert np.all(expected >= np.float32(0.0))
    assert np.all(expected <= np.float32(1.0))
    assert np.array_equal(expected, recomputed)
    assert np.array_equal(row_sums, recomputed_row_sums)
    np.testing.assert_allclose(row_sums[:_TILE_VALID_ROWS], np.ones((_TILE_VALID_ROWS,), dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.array_equal(row_sums[_TILE_VALID_ROWS:], np.zeros((_TILE_ROWS - _TILE_VALID_ROWS,), dtype=np.float32))
    assert np.array_equal(expected[_TILE_VALID_ROWS:], np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _TILE_ROWS), dtype=np.float32))
    assert np.all(expected[~np.isfinite(scores)] == np.float32(0.0))


def test_layer0_attention_probs_head0_tokens0_5_softmax_fixture_cross_checks_trace_probability_rows():
    z = np.load(_load("layer_trace_fixtures.npz"))
    expected = z["layer0_attention_probs_head0_tokens0_5_softmax_expected_fp32"]
    trace_probs = z["layer0_attention_probs_fp32"][0, 0]

    np.testing.assert_allclose(expected[:_TRACE_TOKEN_COUNT, :_TRACE_SCORE_SOURCE_TOKENS], trace_probs, rtol=0.0, atol=1e-6)
    assert np.all(expected[:_TRACE_TOKEN_COUNT, _TRACE_SCORE_SOURCE_TOKENS:] == np.float32(0.0))

def test_layer0_attention_scores_head1_tokens0_5_scaled_masked_fixture_matches_gqa_trace_and_seeded_accum_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    q_scaled = z["layer0_attention_scores_head1_tokens0_5_scaled_masked_q_scaled_fp16"]
    k_as_b = z["layer0_attention_scores_head1_tokens0_5_scaled_masked_k_as_b_fp16"]
    seed = z["layer0_attention_scores_head1_tokens0_5_scaled_masked_seed_fp32"]
    expected = z["layer0_attention_scores_head1_tokens0_5_scaled_masked_expected_fp32"]
    q_rope = z["layer0_q_rope_tokens0_5_head1_full_head_expected_fp16"]
    k_rope = z["layer0_k_rope_tokens0_5_head0_full_head_expected_fp16"]
    trace = z["layer0_attention_scores_fp32"][0, 1]

    q_full = np.concatenate((q_rope[:, 0, :], q_rope[:, 1, :]), axis=1)
    k_full = np.concatenate((k_rope[:, 0, :], k_rope[:, 1, :]), axis=1)
    assert np.array_equal(q_scaled[:5], (q_full.astype(np.float32) * np.float32(0.125)).astype(np.float16))
    assert np.array_equal(k_as_b[:, :5], k_full.T)
    assert np.all(q_scaled[5:] == np.float16(0.0))
    assert np.all(k_as_b[:, 5:] == np.float16(0.0))

    seed_expected = np.full((_TILE_ROWS, _TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(_TILE_VALID_ROWS):
        seed_expected[row, : row + 1] = np.float32(0.0)
    assert np.array_equal(seed, seed_expected)

    recomputed = q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    assert np.array_equal(recomputed, expected)
    np.testing.assert_allclose(expected[0, :1], trace[0, :1], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(expected[1, :2], trace[1, :2], rtol=0.0, atol=1e-6)
    assert np.all(~np.isfinite(expected[0, 1:]))
    assert np.all(~np.isfinite(expected[1, 2:]))


def test_layer0_attention_probs_head1_tokens0_5_softmax_fixture_matches_scaled_masked_softmax_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    scores = z["layer0_attention_probs_head1_tokens0_5_softmax_input_fp32"]
    expected = z["layer0_attention_probs_head1_tokens0_5_softmax_expected_fp32"]
    row_sums = z["layer0_attention_probs_head1_tokens0_5_softmax_row_sums_fp32"]
    scaled_masked = z["layer0_attention_scores_head1_tokens0_5_scaled_masked_expected_fp32"]
    trace_probs = z["layer0_attention_probs_fp32"][0, 1]

    assert np.array_equal(scores, scaled_masked)
    recomputed = np.zeros_like(scores, dtype=np.float32)
    recomputed_row_sums = np.zeros((_TILE_ROWS,), dtype=np.float32)
    for row in range(_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        recomputed[row] = (exps / denom).astype(np.float32)
        recomputed_row_sums[row] = np.sum(recomputed[row], dtype=np.float32)

    assert np.all(np.isfinite(expected))
    assert np.all(expected >= np.float32(0.0))
    assert np.all(expected <= np.float32(1.0))
    assert np.array_equal(expected, recomputed)
    assert np.array_equal(row_sums, recomputed_row_sums)
    np.testing.assert_allclose(expected[:_TRACE_TOKEN_COUNT, :_TRACE_SCORE_SOURCE_TOKENS], trace_probs, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(row_sums[:_TILE_VALID_ROWS], np.ones((_TILE_VALID_ROWS,), dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.array_equal(row_sums[_TILE_VALID_ROWS:], np.zeros((_TILE_ROWS - _TILE_VALID_ROWS,), dtype=np.float32))
    assert np.array_equal(expected[_TILE_VALID_ROWS:], np.zeros((_TILE_ROWS - _TILE_VALID_ROWS, _TILE_ROWS), dtype=np.float32))
    assert np.all(expected[~np.isfinite(scores)] == np.float32(0.0))





def test_layer0_attention_context_head0_tokens0_5_cols0_64_fixture_matches_probs_value_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    probs = z["layer0_attention_context_head0_tokens0_5_cols0_64_probs_fp16"]
    v_as_b = z["layer0_attention_context_head0_tokens0_5_cols0_64_v_as_b_fp16"]
    expected = z["layer0_attention_context_head0_tokens0_5_cols0_64_expected_fp32"]
    expected_fp16 = z["layer0_attention_context_head0_tokens0_5_cols0_64_expected_fp16"]
    trace_probs = z["layer0_attention_probs_fp32"][0, 0]
    trace_context = z["layer0_attention_context_fp16"][0, 0, :, :16]

    assert np.array_equal(probs[:2, :5], trace_probs.astype(np.float16))
    assert np.all(probs[:5, 5:] == np.float16(0.0))
    assert np.all(probs[5:] == np.float16(0.0))
    assert np.any(v_as_b[:5] != np.float16(0.0))
    assert np.all(v_as_b[5:] == np.float16(0.0))

    recomputed = probs.astype(np.float32) @ v_as_b.astype(np.float32)
    assert np.array_equal(recomputed, expected)
    assert np.array_equal(expected.astype(np.float16), expected_fp16)
    np.testing.assert_allclose(expected_fp16[:2, :16], trace_context, rtol=0.0, atol=1e-5)
    assert np.all(expected_fp16[5:] == np.float16(0.0))

def test_layer0_attention_context_head1_tokens0_5_cols64_128_fixture_matches_gqa_probs_value_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    probs = z["layer0_attention_context_head1_tokens0_5_cols64_128_probs_fp16"]
    v_as_b = z["layer0_attention_context_head1_tokens0_5_cols64_128_v_as_b_fp16"]
    expected = z["layer0_attention_context_head1_tokens0_5_cols64_128_expected_fp32"]
    expected_fp16 = z["layer0_attention_context_head1_tokens0_5_cols64_128_expected_fp16"]
    softmax = z["layer0_attention_probs_head1_tokens0_5_softmax_expected_fp32"]
    trace_probs = z["layer0_attention_probs_fp32"][0, 1]
    trace_context = z["layer0_attention_context_fp16"][0, 1, :, :16]
    kv_head0_values = z["layer0_v_proj_fp16"][0, 0]
    kv_head1_values = z["layer0_v_proj_fp16"][0, 1]

    assert np.array_equal(probs[:5, :5], softmax[:5, :5].astype(np.float16))
    np.testing.assert_allclose(probs[:_TRACE_TOKEN_COUNT, :5], trace_probs.astype(np.float16), rtol=0.0, atol=2e-5)
    assert np.all(probs[:5, 5:] == np.float16(0.0))
    assert np.all(probs[5:] == np.float16(0.0))
    assert np.array_equal(v_as_b[:_TRACE_TOKEN_COUNT, :_TRACE_DIM], kv_head0_values)
    assert not np.array_equal(v_as_b[:_TRACE_TOKEN_COUNT, :_TRACE_DIM], kv_head1_values)
    assert np.all(v_as_b[5:] == np.float16(0.0))

    recomputed = probs.astype(np.float32) @ v_as_b.astype(np.float32)
    assert np.array_equal(recomputed, expected)
    assert np.array_equal(expected.astype(np.float16), expected_fp16)
    np.testing.assert_allclose(expected_fp16[:2, :16], trace_context, rtol=0.0, atol=2e-5)
    assert np.all(expected_fp16[5:] == np.float16(0.0))


def test_layer0_q_rope_tokens0_5_head2_full_head_fixture_matches_split_half_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    packed = z["layer0_q_rope_tokens0_5_head2_full_head_input_fp16"]
    cos = z["layer0_q_rope_tokens0_5_head2_full_head_cos_fp32"]
    sin = z["layer0_q_rope_tokens0_5_head2_full_head_sin_fp32"]
    expected = z["layer0_q_rope_tokens0_5_head2_full_head_expected_fp16"]

    left = packed[:, 0, :].astype(np.float32)
    right = packed[:, 1, :].astype(np.float32)
    recomputed = np.stack(
        (
            left * cos - right * sin,
            right * cos + left * sin,
        ),
        axis=1,
    ).astype(np.float16)
    assert np.array_equal(recomputed, expected)
    head1_expected = z["layer0_q_rope_tokens0_5_head1_full_head_expected_fp16"]
    assert not np.array_equal(expected, head1_expected)


def test_layer0_attention_scores_head2_tokens0_5_scaled_masked_fixture_matches_gqa_seeded_accum_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    q_scaled = z["layer0_attention_scores_head2_tokens0_5_scaled_masked_q_scaled_fp16"]
    k_as_b = z["layer0_attention_scores_head2_tokens0_5_scaled_masked_k_as_b_fp16"]
    seed = z["layer0_attention_scores_head2_tokens0_5_scaled_masked_seed_fp32"]
    expected = z["layer0_attention_scores_head2_tokens0_5_scaled_masked_expected_fp32"]
    q_rope = z["layer0_q_rope_tokens0_5_head2_full_head_expected_fp16"]
    k_rope = z["layer0_k_rope_tokens0_5_head0_full_head_expected_fp16"]

    q_full = np.concatenate((q_rope[:, 0, :], q_rope[:, 1, :]), axis=1)
    k_full = np.concatenate((k_rope[:, 0, :], k_rope[:, 1, :]), axis=1)
    assert np.array_equal(q_scaled[:5], (q_full.astype(np.float32) * np.float32(0.125)).astype(np.float16))
    assert np.array_equal(k_as_b[:, :5], k_full.T)
    assert np.all(q_scaled[5:] == np.float16(0.0))
    assert np.all(k_as_b[:, 5:] == np.float16(0.0))

    seed_expected = np.full((_TILE_ROWS, _TILE_ROWS), -np.inf, dtype=np.float32)
    for row in range(_TILE_VALID_ROWS):
        seed_expected[row, : row + 1] = np.float32(0.0)
    assert np.array_equal(seed, seed_expected)

    recomputed = q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed
    assert np.array_equal(recomputed, expected)
    assert np.all(~np.isfinite(expected[0, 1:]))
    assert np.all(~np.isfinite(expected[1, 2:]))


def test_layer0_attention_probs_head2_tokens0_5_softmax_fixture_matches_scaled_masked_softmax_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    scores = z["layer0_attention_probs_head2_tokens0_5_softmax_input_fp32"]
    expected = z["layer0_attention_probs_head2_tokens0_5_softmax_expected_fp32"]
    row_sums = z["layer0_attention_probs_head2_tokens0_5_softmax_row_sums_fp32"]
    scaled_masked = z["layer0_attention_scores_head2_tokens0_5_scaled_masked_expected_fp32"]

    assert np.array_equal(scores, scaled_masked)
    recomputed = np.zeros_like(scores, dtype=np.float32)
    recomputed_row_sums = np.zeros((_TILE_ROWS,), dtype=np.float32)
    for row in range(_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if not finite.any():
            continue
        row_max = np.max(scores[row, finite])
        exps = np.zeros((_TILE_ROWS,), dtype=np.float32)
        exps[finite] = np.exp(scores[row, finite] - row_max).astype(np.float32)
        denom = np.sum(exps, dtype=np.float32)
        recomputed[row] = (exps / denom).astype(np.float32)
        recomputed_row_sums[row] = np.sum(recomputed[row], dtype=np.float32)

    assert np.all(np.isfinite(expected))
    assert np.array_equal(expected, recomputed)
    assert np.array_equal(row_sums, recomputed_row_sums)
    np.testing.assert_allclose(row_sums[:_TILE_VALID_ROWS], np.ones((_TILE_VALID_ROWS,), dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.array_equal(row_sums[_TILE_VALID_ROWS:], np.zeros((_TILE_ROWS - _TILE_VALID_ROWS,), dtype=np.float32))
    assert np.all(expected[~np.isfinite(scores)] == np.float32(0.0))


def test_layer0_attention_context_head2_tokens0_5_cols128_192_fixture_matches_gqa_probs_value_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    probs = z["layer0_attention_context_head2_tokens0_5_cols128_192_probs_fp16"]
    v_as_b = z["layer0_attention_context_head2_tokens0_5_cols128_192_v_as_b_fp16"]
    expected = z["layer0_attention_context_head2_tokens0_5_cols128_192_expected_fp32"]
    expected_fp16 = z["layer0_attention_context_head2_tokens0_5_cols128_192_expected_fp16"]
    softmax = z["layer0_attention_probs_head2_tokens0_5_softmax_expected_fp32"]
    kv_head0_values = z["layer0_v_proj_fp16"][0, 0]
    kv_head1_values = z["layer0_v_proj_fp16"][0, 1]

    assert np.array_equal(probs[:5, :5], softmax[:5, :5].astype(np.float16))
    assert np.all(probs[:5, 5:] == np.float16(0.0))
    assert np.all(probs[5:] == np.float16(0.0))
    assert np.array_equal(v_as_b[:_TRACE_TOKEN_COUNT, :_TRACE_DIM], kv_head0_values)
    assert not np.array_equal(v_as_b[:_TRACE_TOKEN_COUNT, :_TRACE_DIM], kv_head1_values)
    assert np.all(v_as_b[5:] == np.float16(0.0))

    recomputed = probs.astype(np.float32) @ v_as_b.astype(np.float32)
    assert np.array_equal(recomputed, expected)
    assert np.array_equal(expected.astype(np.float16), expected_fp16)
    assert np.all(expected_fp16[5:] == np.float16(0.0))


def test_layer0_q_rope_pair_slice_fixture_matches_split_half_oracle():
    z = np.load(_load("layer_trace_fixtures.npz"))
    packed = z["layer0_q_rope_pairs12_20_input_fp16"]
    cos = z["layer0_q_rope_pairs12_20_cos_fp32"]
    sin = z["layer0_q_rope_pairs12_20_sin_fp32"]
    expected = z["layer0_q_rope_pairs12_20_expected_fp16"]

    assert packed.shape == (2, _ROPE_PAIR_COUNT)
    assert np.any(sin != np.float32(0.0))
    left = packed[0].astype(np.float32)
    right = packed[1].astype(np.float32)
    recomputed = np.stack(
        (
            left * cos - right * sin,
            right * cos + left * sin,
        ),
        axis=0,
    ).astype(np.float16)
    assert np.array_equal(recomputed, expected)




# ---------------------------------------------------------------------------
# primitives_fixtures.npz — deterministic intermediate tensors (Lane A2 schema)
# ---------------------------------------------------------------------------
def test_primitives_schema_and_determinism():
    """Recompute the deterministic numpy tensors in-process and require the
    on-disk arrays match exactly — proving determinism without the mlx oracle."""
    prim = rf._make_primitives()
    z = np.load(_load("primitives_fixtures.npz"))
    assert set(z.files) == set(_EXPECTED_PRIMITIVE_KEYS.keys())
    for key in _EXPECTED_PRIMITIVE_KEYS:
        shape, dtype = _EXPECTED_PRIMITIVE_KEYS[key]
        arr = z[key]
        assert arr.shape == shape, f"{key} shape {arr.shape} != {shape}"
        assert arr.dtype == dtype, f"{key} dtype {arr.dtype} != {dtype}"
        # Byte-for-byte identical to the regenerated deterministic tensor.
        assert np.array_equal(arr, prim[key]), f"{key} not deterministic"


def test_primitives_math_is_ground_truth():
    """The committed expected outputs equal the exact Llama math on the inputs
    (fp32-accumulate matmul, RMSNorm eps=1e-05, SiLU, fp16 cast)."""
    z = np.load(_load("primitives_fixtures.npz"))
    # cast: fp32 -> fp16 exact.
    assert np.array_equal(z["cast_in_fp32"].astype(np.float16), z["cast_expected_fp16"])
    # matmul: (8,16)@(16,8) fp32 accumulate -> fp16.
    mm = (z["matmul_a_fp16"].astype(np.float32) @
          z["matmul_b_fp16"].astype(np.float32)).astype(np.float16)
    assert np.array_equal(mm, z["matmul_expected_fp16"])
    # rms_norm over last axis with the Llama weight vector and eps from config.
    x = z["rms_x_fp16"].astype(np.float32)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + rf.RMS_NORM_EPS)
    rms_expected = ((x / rms) * z["rms_weight_fp16"].astype(np.float32)).astype(np.float16)
    assert np.array_equal(rms_expected, z["rms_expected_fp16"])
    # silu: x * sigmoid(x).
    s = z["silu_x_fp16"].astype(np.float32)
    silu_expected = (s * (1.0 / (1.0 + np.exp(-s)))).astype(np.float16)
    assert np.array_equal(silu_expected, z["silu_expected_fp16"])


@pytest.mark.parametrize(
    ("head", "kv_head", "start", "stop"),
    ((13, 3, 832, 896), (14, 3, 896, 960), (15, 3, 960, 1024), (16, 4, 1024, 1088)),
)
def test_layer0_attention_future_head_scores_softmax_context_fixtures_match_gqa_oracles(head, kv_head, start, stop):
    z = np.load(_load("layer_trace_fixtures.npz"))
    q_scaled = z[f"layer0_attention_scores_head{head}_tokens0_5_scaled_masked_q_scaled_fp16"]
    k_as_b = z[f"layer0_attention_scores_head{head}_tokens0_5_scaled_masked_k_as_b_fp16"]
    seed = z[f"layer0_attention_scores_head{head}_tokens0_5_scaled_masked_seed_fp32"]
    scores = z[f"layer0_attention_scores_head{head}_tokens0_5_scaled_masked_expected_fp32"]
    probs = z[f"layer0_attention_probs_head{head}_tokens0_5_softmax_expected_fp32"]
    row_sums = z[f"layer0_attention_probs_head{head}_tokens0_5_softmax_row_sums_fp32"]
    context_probs = z[f"layer0_attention_context_head{head}_tokens0_5_cols{start}_{stop}_probs_fp16"]
    v_as_b = z[f"layer0_attention_context_head{head}_tokens0_5_cols{start}_{stop}_v_as_b_fp16"]
    context = z[f"layer0_attention_context_head{head}_tokens0_5_cols{start}_{stop}_expected_fp32"]

    assert np.array_equal(scores, q_scaled.astype(np.float32) @ k_as_b.astype(np.float32) + seed)
    recomputed = np.zeros_like(scores, dtype=np.float32)
    for row in range(_TILE_ROWS):
        finite = np.isfinite(scores[row])
        if finite.any():
            exps = np.zeros((_TILE_ROWS,), dtype=np.float32)
            exps[finite] = np.exp(scores[row, finite] - np.max(scores[row, finite])).astype(np.float32)
            recomputed[row] = (exps / np.sum(exps, dtype=np.float32)).astype(np.float32)
    np.testing.assert_allclose(probs, recomputed, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(row_sums[:5], np.ones((5,), dtype=np.float32), rtol=0.0, atol=1e-6)
    assert np.all(row_sums[5:] == 0.0)
    np.testing.assert_allclose(context_probs[:5, :5].astype(np.float32), probs[:5, :5].astype(np.float16).astype(np.float32), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(context, context_probs.astype(np.float32) @ v_as_b.astype(np.float32), rtol=0.0, atol=0.0)
