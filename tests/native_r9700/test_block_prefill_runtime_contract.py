"""Hardware-free contracts for configurable multi-row native prefill blocks."""

import json
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_DIR = REPO_ROOT / "native_r9700"
RUNTIME_SOURCE = NATIVE_DIR / "runtime_contract.cpp"
SESSION_SOURCE = NATIVE_DIR / "amdev_session.cpp"
RUNNER_SOURCES = [
    NATIVE_DIR / name
    for name in (
        "amdev_packets.cpp",
        "runtime_contract.cpp",
        "prefill_npz.cpp",
        "vram_layout.cpp",
        "vram_allocator.cpp",
        "dynamic_page_table.cpp",
        "resident_memory.cpp",
        "vram_smoke_asset.cpp",
        "hsa_code_image_asset.cpp",
        "model_weight_binder.cpp",
        "amdev_session.cpp",
        "kernel_catalog.cpp",
        "device_memory.cpp",
        "hardware_lock.cpp",
        "llama_stage_layout.cpp",
        "llama_layer_executor.cpp",
        "kernel_assets.cpp",
        "runtime.cpp",
        "runner.cpp",
    )
]
ALLOWED_BLOCK_TOKENS = (1, 2, 4, 8, 16, 32)


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build_dir = tmp_path_factory.mktemp("block_prefill_runner")
    executable = build_dir / "native_r9700_runner"
    completed = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            *map(str, RUNNER_SOURCES),
            "-I",
            str(NATIVE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not completed.stdout
    assert not completed.stderr
    return executable


def _prefill_command(runner: Path, tmp_path: Path) -> list[str]:
    return [
        str(runner),
        "--native-prefill-proof",
        "--model",
        "synthetic-model",
        "--token-ids-json",
        "[1,2]",
        "--out",
        str(tmp_path / "block-prefill.npz"),
        "--log",
        str(tmp_path / "block-prefill.log"),
    ]


def _run_prefill(
    runner: Path, tmp_path: Path, optional_arguments: list[str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _prefill_command(runner, tmp_path) + optional_arguments,
        capture_output=True,
        text=True,
        check=False,
    )


def _structured_result(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    return json.loads(completed.stdout.splitlines()[-1])


def test_prefill_command_defaults_to_four_token_blocks(runner, tmp_path):
    """Without an explicit diagnostic override, production uses capacity four."""
    completed = _run_prefill(runner, tmp_path, [])

    assert completed.returncode == 1
    assert "failure_stage: layer_weight_table" in completed.stdout
    assert "block_tokens: 4" in completed.stdout
    assert "block_count: 0" in completed.stdout
    assert _structured_result(completed)["block_tokens"] == 4
    assert _structured_result(completed)["block_count"] == 0


@pytest.mark.parametrize("block_tokens", ALLOWED_BLOCK_TOKENS)
def test_runner_accepts_exact_block_token_capacity_ladder(
    runner, tmp_path, block_tokens
):
    """Each allowed capacity reaches request execution and is reported verbatim."""
    completed = _run_prefill(
        runner, tmp_path, ["--block-tokens", str(block_tokens)]
    )

    assert completed.returncode == 1
    assert "failure_stage: layer_weight_table" in completed.stdout
    assert f"block_tokens: {block_tokens}" in completed.stdout
    assert _structured_result(completed)["block_tokens"] == block_tokens


@pytest.mark.parametrize(
    "optional_arguments",
    [
        ["--block-tokens"],
        ["--block-tokens", "0"],
        ["--block-tokens", "3"],
        ["--block-tokens", "129"],
        ["--block-tokens", "-1"],
        ["--block-tokens", "eight"],
        ["--block-tokens=8"],
        ["--block-tokens", "8", "--block-tokens", "4"],
    ],
)
def test_runner_rejects_missing_invalid_equals_and_duplicate_block_flags(
    runner, tmp_path, optional_arguments
):
    """Malformed capacities stop at strict CLI validation before hardware or logs."""
    completed = _run_prefill(runner, tmp_path, optional_arguments)

    assert completed.returncode == 2
    assert "failure_stage: native_prefill_request" in completed.stdout
    assert "[1,2]" not in completed.stdout
    assert not (tmp_path / "block-prefill.log").exists()


@pytest.mark.parametrize(
    "optional_arguments",
    [
        [
            "--block-tokens",
            "8",
            "--gpu-stage-profile",
            "--completion-policy",
            "terminal",
            "--barrier-policy",
            "overlap-kv",
        ],
        [
            "--barrier-policy",
            "full",
            "--completion-policy",
            "per-stage",
            "--gpu-stage-profile",
            "--block-tokens",
            "2",
        ],
    ],
)
def test_block_capacity_composes_with_all_existing_optional_flags_in_any_order(
    runner, tmp_path, optional_arguments
):
    """The four strict optional controls compose without positional coupling."""
    completed = _run_prefill(runner, tmp_path, optional_arguments)

    assert completed.returncode == 1
    assert "failure_stage: layer_weight_table" in completed.stdout
    expected = int(optional_arguments[optional_arguments.index("--block-tokens") + 1])
    assert _structured_result(completed)["block_tokens"] == expected


def test_runtime_builds_requested_capacity_and_one_zero_padded_upload_per_block():
    """The runtime packs live rows into one capacity-sized, zero-padded upload."""
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")

    assert "request.block_tokens, &persistent_dispatch, &detail" in " ".join(
        source.split()
    )
    helper_start = source.index("auto upload_embedding_block")
    helper_end = source.index("};", helper_start) + 2
    helper = source[helper_start:helper_end]
    normalized_helper = " ".join(helper.split())
    assert "persistent_dispatch.block_capacity) * kLlamaEmbeddingRowBytes" in normalized_helper
    assert "offset < block.token_count" in helper
    assert "request.token_ids[block.position + offset]" in helper
    assert "static_cast<size_t>(offset) *" in helper
    assert "kLlamaEmbeddingRowBytes" in helper
    assert helper.count("resident.upload_named(") == 1
    assert "embedding_bytes.size()" in helper
    assert "result->block_tokens = request.block_tokens;" in source
    assert "persistent_dispatch.token_blocks.size()" in source



def test_resident_weight_and_embedding_uploads_retain_exact_fill_validation():
    """Capacity-sized uploads preserve the resident session's exact-fill boundary."""
    source = SESSION_SOURCE.read_text(encoding="utf-8")
    upload_start = source.index("bool ResidentHsaSession::upload_named(")
    upload_end = source.index("\n}", upload_start)
    upload = source[upload_start:upload_end]

    assert "byte_count != state.requested_allocation_byte_counts[buffer_index]" in upload
    assert "must exactly fill its declared weight-window allocation" in upload