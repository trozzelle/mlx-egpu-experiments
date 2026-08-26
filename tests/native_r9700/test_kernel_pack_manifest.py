"""RED contracts for the offline Kernel Pack manifest validator/generator.

The production module is intentionally loaded from its task-set-3 path instead
of imported through a package.  This keeps the failure pointed at the missing
offline owner while these contracts exercise the complete frozen v1 record.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_MODULE = REPO_ROOT / "native_r9700" / "kernel_pack_manifest.py"
UPSTREAM_POLICY = REPO_ROOT / "docs" / "upstream-reference-manifest.yaml"
TARGET = "gfx1201"
IMAGE_BYTES = b"offline-test-image\x00" * 4
IMAGE_SHA256 = hashlib.sha256(IMAGE_BYTES).hexdigest()
PACK_SHA256_PLACEHOLDER = "00" * 32
PINNED_LLVM_REVISION = "8dba93818258d95c46fa2c17e902a8256e4d91b5"
SHA_A = "aa" * 32
SHA_B = "bb" * 32
SHA_C = "cc" * 32
SHA_D = "dd" * 32

EVIDENCE_FIELDS = {
    "record_path",
    "record_kind",
    "evidence_slot",
    "record_id",
    "record_sha256",
    "subject_target",
    "image_sha256",
    "pack_sha256",
    "producer_kind",
    "tool_digest",
    "input_digest",
    "output_digest",
}
EVIDENCE_KINDS = {
    "offline_oracle",
    "offline_review",
    "target_conformance",
    "native_run",
    "benchmark",
}
EVIDENCE_SLOTS = {
    "numpy_oracle",
    "source_review",
    "isa_review",
    "resource_review",
    "layout_proof",
    "scalar_native_projection",
    "conformance",
    "native_run",
    "benchmark",
}
ALLOWED_EVIDENCE_PAIRS = {
    ("offline_oracle", "numpy_oracle"),
    ("offline_review", "source_review"),
    ("offline_review", "isa_review"),
    ("offline_review", "resource_review"),
    ("offline_review", "layout_proof"),
    ("target_conformance", "scalar_native_projection"),
    ("target_conformance", "conformance"),
    ("native_run", "native_run"),
    ("benchmark", "benchmark"),
}
EVIDENCE_PAYLOAD_IDENTITY_FIELDS = (
    "record_id",
    "record_kind",
    "evidence_slot",
    "subject_target",
    "image_sha256",
    "pack_sha256",
    "producer_kind",
    "tool_digest",
    "input_digest",
    "output_digest",
)
RESOURCE_REPORT_FIELDS = (
    "rsrc1",
    "rsrc2",
    "rsrc3",
    "wave_size",
    "sgpr_count",
    "vgpr_count",
    "lds_bytes",
    "private_segment_bytes",
    "metadata_provenance",
)
PHYSICAL_LAYOUT_FIELDS = (
    "source_tensor_layout_version",
    "physical_layout_version",
    "layout_spec_path",
    "layout_spec_sha256",
    "inverse_fixture_path",
    "inverse_fixture_sha256",
    "inverse_n",
    "inverse_k",
    "inverse_source_f16",
    "layout_mapping",
    "layout_strides",
    "alignment_bytes",
    "padding_bytes",
    "swizzle",
    "layout_origin",
    "inverse_fixture_input_digest",
    "inverse_fixture_output_digest",
    "layout_status",
    "failure_stage",
    "exit_status",
    "wrapper_exit_status",
)


def _load_manifest_module() -> ModuleType:
    """Load the offline owner, failing explicitly while it is absent."""

    assert MANIFEST_MODULE.is_file(), (
        "task-set-3 production owner is missing: "
        "native_r9700/kernel_pack_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("native_r9700_kernel_pack_manifest", MANIFEST_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(root: Path, relative: str, data: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256(data)


def _record_file(root: Path, relative: str, payload: MappingLike) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _write(root, relative, data)


class MappingLike(dict[str, Any]):
    """Typing-only alias that keeps fixture construction readable."""


def _evidence_ref(
    root: Path,
    *,
    name: str,
    kind: str,
    slot: str,
    subject_target: str,
    image_sha256: str,
    pack_sha256: str,
    producer_kind: str,
    tool_digest: str,
    input_digest: str,
    output_digest: str,
    payload_fields: MappingLike | None = None,
) -> dict[str, str]:
    relative = f"evidence/{name}.json"
    record_id = f"{name}-record-v1"
    payload: MappingLike = {
        "record_id": record_id,
        "record_kind": kind,
        "evidence_slot": slot,
        "subject_target": subject_target,
        "image_sha256": image_sha256,
        "pack_sha256": pack_sha256,
        "producer_kind": producer_kind,
        "tool_digest": tool_digest,
        "input_digest": input_digest,
        "output_digest": output_digest,
    }
    if payload_fields is not None:
        payload.update(payload_fields)
    record_sha256 = _record_file(root, relative, payload)
    return {
        "record_path": relative,
        "record_kind": kind,
        "evidence_slot": slot,
        "record_id": record_id,
        "record_sha256": record_sha256,
        "subject_target": subject_target,
        "image_sha256": image_sha256,
        "pack_sha256": pack_sha256,
        "producer_kind": producer_kind,
        "tool_digest": tool_digest,
        "input_digest": input_digest,
        "output_digest": output_digest,
    }


def _base_record(root: Path) -> dict[str, Any]:
    """Build a concrete B0 scalar-control record with all frozen fields."""

    source_sha256 = _write(root, "src/demo_kernel.cpp", b"extern \"C\" void demo_kernel() {}\n")
    image_sha256 = _write(root, "images/demo_kernel.image", IMAGE_BYTES)
    assert image_sha256 == IMAGE_SHA256

    oracle = _evidence_ref(
        root,
        name="numpy-oracle",
        kind="offline_oracle",
        slot="numpy_oracle",
        subject_target="",
        image_sha256="",
        pack_sha256="",
        producer_kind="cpu_reference",
        tool_digest="",
        input_digest=SHA_A,
        output_digest=SHA_B,
    )
    source_review = _evidence_ref(
        root,
        name="source-review",
        kind="offline_review",
        slot="source_review",
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=PACK_SHA256_PLACEHOLDER,
        producer_kind="",
        tool_digest=SHA_A,
        input_digest=SHA_B,
        output_digest=SHA_C,
    )
    conformance = _evidence_ref(
        root,
        name="conformance",
        kind="target_conformance",
        slot="conformance",
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=PACK_SHA256_PLACEHOLDER,
        producer_kind="r9700_native",
        tool_digest="",
        input_digest=SHA_A,
        output_digest=SHA_C,
    )
    native_run = _evidence_ref(
        root,
        name="native-run",
        kind="native_run",
        slot="native_run",
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=PACK_SHA256_PLACEHOLDER,
        producer_kind="r9700_native",
        tool_digest="",
        input_digest=SHA_A,
        output_digest=SHA_D,
    )
    resource_review = _evidence_ref(
        root,
        name="resource-review",
        kind="offline_review",
        slot="resource_review",
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=PACK_SHA256_PLACEHOLDER,
        producer_kind="",
        tool_digest=SHA_B,
        input_digest=SHA_C,
        output_digest=SHA_D,
        payload_fields={
            "rsrc1": 1,
            "rsrc2": 2,
            "rsrc3": 3,
            "wave_size": 32,
            "sgpr_count": 8,
            "vgpr_count": 8,
            "lds_bytes": 0,
            "private_segment_bytes": 0,
            "metadata_provenance": "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst",
        },
    )
    isa_review = _evidence_ref(
        root,
        name="isa-review",
        kind="offline_review",
        slot="isa_review",
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=PACK_SHA256_PLACEHOLDER,
        producer_kind="",
        tool_digest=SHA_C,
        input_digest=SHA_A,
        output_digest=SHA_D,
        payload_fields={
            "isa_categories": ["scalar"],
            "unsupported_instructions": [],
        },
    )

    return {
        "schema_version": 1,
        "name": "b0-demo-kernel-pack",
        "version": "1.0.0",
        "target": TARGET,
        "required_features": ["wave32"],
        "provenance": {
            "upstream_repository": "https://github.com/llvm/llvm-project",
            "upstream_revision": PINNED_LLVM_REVISION,
            "upstream_paths": ["llvm/docs/AMDGPUUsage.rst"],
            "local_sources": [
                {"path": "src/demo_kernel.cpp", "sha256": source_sha256}
            ],
            "license_reviews": [
                {
                    "component": "llvm/docs/AMDGPUUsage.rst",
                    "spdx_expression": "Apache-2.0 WITH LLVM-exception",
                    "review_id": "license-review-llvm-amdgpu-usage-v1",
                    "status": "accepted",
                },
                {
                    "component": "src/demo_kernel.cpp",
                    "spdx_expression": "Apache-2.0 WITH LLVM-exception",
                    "review_id": "license-review-demo-source-v1",
                    "status": "accepted",
                },
                {
                    "component": "images/demo_kernel.image",
                    "spdx_expression": "Apache-2.0 WITH LLVM-exception",
                    "review_id": "license-review-demo-image-v1",
                    "status": "accepted",
                },
                {
                    "component": "generated/b0-demo-kernel-pack.cpp",
                    "spdx_expression": "Apache-2.0 WITH LLVM-exception",
                    "review_id": "license-review-demo-generated-v1",
                    "status": "accepted",
                },
            ],
            "modifications": [
                {
                    "component": "generated/b0-demo-kernel-pack.cpp",
                    "summary": "deterministic allocation-free C++ view initializer generated offline",
                }
            ],
        },
        "image": {
            "image_path": "images/demo_kernel.image",
            "image_sha256": image_sha256,
            "image_size": len(IMAGE_BYTES),
            "code_object_version": "amdhsa-v5",
            "build": {
                "toolchain_id": "llvm-amdgpu",
                "toolchain_revision": PINNED_LLVM_REVISION,
                "generator_id": "r9700-kernel-pack-manifest",
                "generator_revision": "0123456789abcdef0123456789abcdef01234567",
                "command_sha256": SHA_A,
            },
        },
        "entries": [
            {
                "symbol": "demo_kernel",
                "descriptor_offset": 8,
                "entry_offset": 24,
                "kernargs": {
                    "bytes": 16,
                    "fields": [
                        {
                            "name": "input",
                            "type": "uint64",
                            "offset": 0,
                            "size": 8,
                            "alignment": 8,
                        },
                        {
                            "name": "count",
                            "type": "uint32",
                            "offset": 8,
                            "size": 4,
                            "alignment": 4,
                        },
                    ],
                    "tail_padding_bytes": 4,
                },
                "resources": {
                    "rsrc1": 1,
                    "rsrc2": 2,
                    "rsrc3": 3,
                    "wave_size": 32,
                    "sgpr_count": 8,
                    "vgpr_count": 8,
                    "lds_bytes": 0,
                    "private_segment_bytes": 0,
                    "metadata_provenance": "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst",
                },
                "geometry": {
                    "cases": [
                        {
                            "shape_family": "b0-demo-fixed-v1",
                            "geometry_rule": "exact-global-v1",
                            "workgroup_x": 1,
                            "workgroup_y": 1,
                            "workgroup_z": 1,
                            "global_x": 1,
                            "global_y": 1,
                            "global_z": 1,
                            "grid_tile_m": 0,
                            "grid_tile_n": 0,
                            "dynamic_lds_allowed": False,
                            "dynamic_lds_max_bytes": 0,
                        }
                    ]
                },
            }
        ],
        "compatibility": {
            "input_dtype": "fp16",
            "weight_dtype": "fp16",
            "output_dtype": "fp16",
            "source_tensor_layout_version": "b0-source-layout-v1",
            "shape_families": [
                {
                    "name": "b0-demo-fixed-v1",
                    "fixed_dimensions": [
                        {"name": "K", "value": 4},
                        {"name": "N", "value": 4},
                    ],
                    "runtime_dimension": None,
                    "tail_policy": "none",
                    "geometry_rule": "exact-global-v1",
                }
            ],
            "weight_packing_version": "source-equivalent-v1",
        },
        "numerics": {
            "input_dtype": "fp16",
            "accumulation_dtype": "fp32",
            "output_dtype": "fp16",
            "cast_points": [
                {"stage": "accumulate", "from_dtype": "fp16", "to_dtype": "fp32"}
            ],
            "finite_value_rule": "finite-input-output-v1",
            "tolerance_policy": "b0-scalar-exact-v1",
            "reference_set_kind": "b0_scalar_control",
            "retained_reference": oracle,
            "numpy_oracle": None,
            "scalar_native_projection": None,
        },
        "evidence": {
            "conformance": conformance,
            "source_review": source_review,
            "native_run": native_run,
            "resource_review": resource_review,
            "isa_review": isa_review,
            "layout_proof": None,
            "benchmark_record": None,
            "benchmark_not_applicable_reason": "correctness-control pack; no promoted benchmark",
        },
    }


def _reseal_pack(module: ModuleType, record: dict[str, Any]) -> str:
    """Bind all non-oracle EvidenceRef pack digests to the canonical identity."""

    digest = module.compute_pack_sha256(record)
    for ref in (
        record["evidence"]["conformance"],
        record["evidence"]["source_review"],
        record["evidence"]["native_run"],
        record["evidence"]["resource_review"],
        record["evidence"]["isa_review"],
        record["evidence"].get("layout_proof"),
        record["evidence"].get("benchmark_record"),
        record["numerics"].get("scalar_native_projection"),
    ):
        if ref is not None:
            ref["pack_sha256"] = digest
    return digest


def _sync_evidence_payloads(root: Path, record: dict[str, Any]) -> None:
    """Refresh ordinary report payload identity fields after pack resealing."""
    binding_fields = (
        "record_id",
        "record_kind",
        "evidence_slot",
        "subject_target",
        "image_sha256",
        "pack_sha256",
        "producer_kind",
        "tool_digest",
        "input_digest",
        "output_digest",
    )
    refs = [
        record["evidence"]["conformance"],
        record["evidence"]["source_review"],
        record["evidence"]["native_run"],
        record["evidence"]["resource_review"],
        record["evidence"]["isa_review"],
        record["evidence"].get("benchmark_record"),
        record["numerics"].get("retained_reference"),
        record["numerics"].get("numpy_oracle"),
        record["numerics"].get("scalar_native_projection"),
    ]
    for reference in refs:
        if reference is None:
            continue
        path = root / reference["record_path"]
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for field in binding_fields:
            payload[field] = reference[field]
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.write_bytes(data)
        reference["record_sha256"] = _sha256(data)


def _isolate_evidence_files(root: Path, record: dict[str, Any], suffix: str) -> None:
    """Give each independently resealed record private evidence file paths."""
    refs = [
        record["evidence"]["conformance"],
        record["evidence"]["source_review"],
        record["evidence"]["native_run"],
        record["evidence"]["resource_review"],
        record["evidence"]["isa_review"],
        record["evidence"].get("layout_proof"),
        record["evidence"].get("benchmark_record"),
        record["numerics"].get("retained_reference"),
        record["numerics"].get("numpy_oracle"),
        record["numerics"].get("scalar_native_projection"),
    ]
    for reference in refs:
        if reference is None:
            continue
        old_path = root / reference["record_path"]
        if not old_path.is_file():
            continue
        relative = Path(reference["record_path"])
        isolated_relative = relative.with_name(f"{relative.stem}-{suffix}{relative.suffix}")
        isolated_path = root / isolated_relative
        isolated_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(old_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["record_id"] = f"{reference['record_id']}-{suffix}"
            isolated_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        else:
            isolated_bytes = old_path.read_bytes()
        isolated_path.write_bytes(isolated_bytes)
        reference["record_path"] = isolated_relative.as_posix()
        reference["record_id"] = f"{reference['record_id']}-{suffix}"
        reference["record_sha256"] = _sha256(isolated_bytes)


def _f2_variant(
    module: ModuleType, root: Path, scalar_record: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Build the frozen F2 bounded-M/WMMA shape and dual reference set."""

    record = copy.deepcopy(scalar_record)
    record["name"] = "f2-linear-gate-up-f16-v1"
    record["required_features"] = ["wave32"]
    record["entries"][0]["symbol"] = "linear_wmma_f16"
    record["entries"][0]["kernargs"] = {
        "bytes": 32,
        "fields": [
            {"name": "a", "type": "uint64", "offset": 0, "size": 8, "alignment": 8},
            {"name": "b", "type": "uint64", "offset": 8, "size": 8, "alignment": 8},
            {"name": "c", "type": "uint64", "offset": 16, "size": 8, "alignment": 8},
            {"name": "m", "type": "uint32", "offset": 24, "size": 4, "alignment": 4},
        ],
        "tail_padding_bytes": 4,
    }
    record["entries"][0]["geometry"]["cases"] = [
        {
            "shape_family": "f2-linear-gate-up-f16-v1",
            "geometry_rule": "f2-wmma-64x64-m-tail-v1",
            "workgroup_x": 128,
            "workgroup_y": 4,
            "workgroup_z": 1,
            "global_x": 0,
            "global_y": 0,
            "global_z": 0,
            "grid_tile_m": 64,
            "grid_tile_n": 64,
            "dynamic_lds_allowed": False,
            "dynamic_lds_max_bytes": 0,
        }
    ]
    record["compatibility"] = {
        "input_dtype": "fp16",
        "weight_dtype": "fp16",
        "output_dtype": "fp16",
        "source_tensor_layout_version": "f16-row-major-nk-source-v1",
        "shape_families": [
            {
                "name": "f2-linear-gate-up-f16-v1",
                "fixed_dimensions": [
                    {"name": "K", "value": 2048},
                    {"name": "N", "value": 8192},
                ],
                "runtime_dimension": {
                    "name": "M",
                    "min_value": 1,
                    "max_value": 128,
                    "full_value": 128,
                },
                "tail_policy": "masked/padded",
                "geometry_rule": "f2-wmma-64x64-m-tail-v1",
            }
        ],
        "weight_packing_version": "f2-wmma-physical-tile-v1",
    }
    record["numerics"] = {
        "input_dtype": "fp16",
        "accumulation_dtype": "fp32",
        "output_dtype": "fp16",
        "cast_points": [
            {"stage": "wmma-accumulate", "from_dtype": "fp16", "to_dtype": "fp32"}
        ],
        "finite_value_rule": "finite-input-output-v1",
        "tolerance_policy": "F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1",
        "reference_set_kind": "f2_wmma_dual",
        "retained_reference": None,
        "numpy_oracle": copy.deepcopy(
            scalar_record["numerics"]["retained_reference"]
        ),
        "scalar_native_projection": _evidence_ref(
            root,
            name="native-projection",
            kind="target_conformance",
            slot="scalar_native_projection",
            subject_target=TARGET,
            image_sha256=IMAGE_SHA256,
            pack_sha256=PACK_SHA256_PLACEHOLDER,
            producer_kind="r9700_native",
            tool_digest="",
            input_digest=SHA_A,
            output_digest=SHA_D,
        ),
    }
    layout_proof = _evidence_ref(
        root,
        name="layout-proof",
        kind="offline_review",
        slot="layout_proof",
        subject_target=TARGET,
        image_sha256=IMAGE_SHA256,
        pack_sha256=PACK_SHA256_PLACEHOLDER,
        producer_kind="",
        tool_digest=SHA_A,
        input_digest=SHA_B,
        output_digest=SHA_C,
    )
    layout_proof["record_path"] = "logs/f2/wmma-physical-layout-proof.json"
    layout_proof["record_id"] = "f2-wmma-physical-layout-proof-v1"
    record["evidence"]["layout_proof"] = layout_proof
    pack_sha256 = _reseal_pack(module, record)
    _sync_evidence_payloads(root, record)
    layout_payload = {
        key: layout_proof[key] for key in EVIDENCE_FIELDS if key != "record_sha256"
    }
    layout_payload.update(
        {
            "record_sha256": "",
            "source_tensor_layout_version": "f16-row-major-nk-source-v1",
            "physical_layout_version": "f2-wmma-physical-tile-v1",
            "layout_spec_path": "build/f2-wmma/f2-wmma-physical-layout-spec.json",
            "layout_spec_sha256": SHA_A,
            "inverse_fixture_path": "build/f2-wmma/f2-wmma-physical-layout-inverse.npz",
            "inverse_fixture_sha256": SHA_B,
            "inverse_n": [0, 0, 0, 15, 16, 16, 8191],
            "inverse_k": [0, 1, 16, 15, 0, 2047, 2047],
            "inverse_source_f16": [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700],
            "layout_mapping": {
                "source_element": "source_weight[n*K+k]",
                "physical_byte_offset": (
                    "((((n // 16) * 128 + (k // 16)) * 512) + "
                    "(((k % 16) * 16 + (n % 16)) * 2))"
                ),
                "b_tile": "tile_n=n//16,tile_k=k//16,row=k%16,col=n%16",
                "lds_byte_offset": "((k % 16) * 16 + (n % 16)) * 2",
            },
            "layout_strides": {
                "source_row_stride_elements": 2048,
                "physical_tile_stride_bytes": 512,
                "lds_tile_stride_bytes": 512,
                "tile_n_count": 512,
                "tile_k_count": 128,
            },
            "alignment_bytes": 16,
            "padding_bytes": 0,
            "swizzle": "none",
            "layout_origin": "pinned_header",
            "inverse_fixture_input_digest": SHA_B,
            "inverse_fixture_output_digest": SHA_C,
            "layout_status": "pass",
            "failure_stage": "none",
            "exit_status": 0,
            "wrapper_exit_status": 0,
        }
    )
    layout_payload["record_sha256"] = _sha256(
        json.dumps(
            {
                key: value
                for key, value in layout_payload.items()
                if key != "record_sha256"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    layout_proof["record_sha256"] = layout_payload["record_sha256"]
    layout_path = root / layout_proof["record_path"]
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_bytes(
        json.dumps(layout_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    record["evidence"]["benchmark_not_applicable_reason"] = (
        "correctness-control F2 record; no promoted benchmark"
    )
    return record, pack_sha256


def _fixture(tmp_path: Path) -> tuple[ModuleType, Path, dict[str, Any], str]:
    module = _load_manifest_module()
    record = _base_record(tmp_path)
    digest = _reseal_pack(module, record)
    _sync_evidence_payloads(tmp_path, record)
    return module, tmp_path, record, digest


def _validate(module: ModuleType, record: dict[str, Any], root: Path) -> Any:
    return module.validate_manifest(record, asset_root=root)


def _expect_rejection(module: ModuleType, callback: Callable[[], Any]) -> None:
    with pytest.raises(module.ManifestError):
        callback()


def _reverse_mapping_order(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _reverse_mapping_order(child)
            for key, child in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mapping_order(child) for child in value]
    return value
def _remove_pack_sha256_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_pack_sha256_fields(child)
            for key, child in value.items()
            if key not in {"pack_sha256", "record_sha256"}
        }
    if isinstance(value, list):
        return [_remove_pack_sha256_fields(child) for child in value]
    return value


def _independent_pack_sha256(record: dict[str, Any]) -> str:
    normalized = copy.deepcopy(record)
    normalized.pop("evidence")
    preimage = {
        "domain": "r9700-kernel-pack-identity-v1",
        "pack": _remove_pack_sha256_fields(normalized),
    }
    canonical = json.dumps(
        preimage,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(canonical)



def _matrix_ref(kind: str, slot: str, *, pack_sha256: str = SHA_D) -> dict[str, str]:
    if kind == "offline_oracle":
        producer_kind = "cpu_reference"
        subject_target = ""
        image_sha256 = ""
        bound_pack_sha256 = ""
        tool_digest = ""
    elif kind == "offline_review":
        producer_kind = ""
        subject_target = TARGET
        image_sha256 = IMAGE_SHA256
        bound_pack_sha256 = pack_sha256
        tool_digest = SHA_A
    elif kind in {"target_conformance", "native_run"}:
        producer_kind = "r9700_native"
        subject_target = TARGET
        image_sha256 = IMAGE_SHA256
        bound_pack_sha256 = pack_sha256
        tool_digest = ""
    else:
        producer_kind = "r9700_native"
        subject_target = TARGET
        image_sha256 = IMAGE_SHA256
        bound_pack_sha256 = pack_sha256
        tool_digest = SHA_A
    return {
        "record_path": f"evidence/{kind}-{slot}.json",
        "record_kind": kind,
        "evidence_slot": slot,
        "record_id": f"{kind}-{slot}-record-v1",
        "record_sha256": SHA_B,
        "subject_target": subject_target,
        "image_sha256": image_sha256,
        "pack_sha256": bound_pack_sha256,
        "producer_kind": producer_kind,
        "tool_digest": tool_digest,
        "input_digest": SHA_C,
        "output_digest": SHA_D,
    }


def test_loader_and_validator_accept_a_complete_owned_manifest(tmp_path: Path) -> None:
    module, root, record, digest = _fixture(tmp_path)
    manifest_path = root / "b0-demo-kernel-pack.pack.json"
    manifest_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    loaded = module.load_manifest(manifest_path)
    assert loaded == record
    _validate(module, loaded, root)
    assert module.compute_pack_sha256(loaded) == digest


def test_loader_rejects_duplicate_json_keys_and_validator_rejects_unknown_keys(tmp_path: Path) -> None:
    module, root, record, _ = _fixture(tmp_path)
    duplicate_path = root / "duplicate.pack.json"
    duplicate_path.write_text(
        '{"schema_version":1,"schema_version":1,"name":"duplicate"}',
        encoding="utf-8",
    )
    _expect_rejection(module, lambda: module.load_manifest(duplicate_path))

    unknown = copy.deepcopy(record)
    unknown["unexpected"] = True
    _expect_rejection(module, lambda: _validate(module, unknown, root))

    nested_unknown = copy.deepcopy(record)
    nested_unknown["provenance"]["unreviewed_extension"] = "reject"
    _expect_rejection(module, lambda: _validate(module, nested_unknown, root))


def test_provenance_requires_immutable_pin_exact_paths_modifications_and_component_licenses(
    tmp_path: Path,
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)
    assert re.fullmatch(r"[0-9a-f]{40}", record["provenance"]["upstream_revision"])
    assert record["provenance"]["upstream_paths"] == ["llvm/docs/AMDGPUUsage.rst"]
    assert record["provenance"]["modifications"]

    for mutation in (
        lambda value: value.update(upstream_revision="main"),
        lambda value: value.update(upstream_paths=["/absolute/upstream/path"]),
        lambda value: value.update(upstream_paths=["../escape"]),
        lambda value: value.pop("modifications"),
        lambda value: value.update(upstream_repository="https://example.invalid/repo.git"),
    ):
        malformed = copy.deepcopy(record)
        mutation(malformed["provenance"])
        _expect_rejection(module, lambda malformed=malformed: _validate(module, malformed, root))

    missing_license = copy.deepcopy(record)
    missing_license["provenance"]["license_reviews"] = missing_license["provenance"]["license_reviews"][1:]
    _expect_rejection(module, lambda: _validate(module, missing_license, root))

    unknown_license = copy.deepcopy(record)
    unknown_license["provenance"]["license_reviews"][0]["status"] = "unknown"
    _expect_rejection(module, lambda: _validate(module, unknown_license, root))

    empty_expression = copy.deepcopy(record)
    empty_expression["provenance"]["license_reviews"][0]["spdx_expression"] = ""
    _expect_rejection(module, lambda: _validate(module, empty_expression, root))


def test_source_and_image_identity_requires_safe_paths_lowercase_hashes_and_exact_bytes(
    tmp_path: Path,
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)

    mutations = []
    wrong_source_digest = copy.deepcopy(record)
    wrong_source_digest["provenance"]["local_sources"][0]["sha256"] = SHA_D
    mutations.append(wrong_source_digest)

    uppercase_image_digest = copy.deepcopy(record)
    uppercase_image_digest["image"]["image_sha256"] = IMAGE_SHA256.upper()
    mutations.append(uppercase_image_digest)

    escaped_image = copy.deepcopy(record)
    escaped_image["image"]["image_path"] = "../outside.image"
    mutations.append(escaped_image)

    absolute_source = copy.deepcopy(record)
    absolute_source["provenance"]["local_sources"][0]["path"] = str(root / "src/demo_kernel.cpp")
    mutations.append(absolute_source)

    wrong_size = copy.deepcopy(record)
    wrong_size["image"]["image_size"] += 1
    mutations.append(wrong_size)

    for malformed in mutations:
        _expect_rejection(module, lambda malformed=malformed: _validate(module, malformed, root))


def test_target_code_object_and_build_identity_are_required_and_concrete(tmp_path: Path) -> None:
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)

    for path, value in (
        (("target",), "gfx1100"),
        (("image", "code_object_version"), ""),
        (("image", "build", "toolchain_revision"), "release"),
        (("image", "build", "generator_revision"), "branch-name"),
        (("image", "build", "command_sha256"), "not-a-digest"),
    ):
        malformed = copy.deepcopy(record)
        node: Any = malformed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        _expect_rejection(module, lambda malformed=malformed: _validate(module, malformed, root))


def test_descriptor_offsets_symbols_kernargs_resources_and_geometry_are_closed(tmp_path: Path) -> None:
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)

    duplicate_symbol = copy.deepcopy(record)
    duplicate_symbol["entries"].append(copy.deepcopy(duplicate_symbol["entries"][0]))
    _expect_rejection(module, lambda: _validate(module, duplicate_symbol, root))

    offset_outside_image = copy.deepcopy(record)
    offset_outside_image["entries"][0]["entry_offset"] = len(IMAGE_BYTES) + 1
    _expect_rejection(module, lambda: _validate(module, offset_outside_image, root))

    misaligned_field = copy.deepcopy(record)
    misaligned_field["entries"][0]["kernargs"]["fields"][1]["offset"] = 10
    _expect_rejection(module, lambda: _validate(module, misaligned_field, root))

    overlapping_field = copy.deepcopy(record)
    overlapping_field["entries"][0]["kernargs"]["fields"][1]["offset"] = 4
    _expect_rejection(module, lambda: _validate(module, overlapping_field, root))

    wrong_tail_padding = copy.deepcopy(record)
    wrong_tail_padding["entries"][0]["kernargs"]["tail_padding_bytes"] = 0
    _expect_rejection(module, lambda: _validate(module, wrong_tail_padding, root))

    unknown_resource_provenance = copy.deepcopy(record)
    unknown_resource_provenance["entries"][0]["resources"]["metadata_provenance"] = "unknown"
    _expect_rejection(module, lambda: _validate(module, unknown_resource_provenance, root))

    arbitrary_geometry = copy.deepcopy(record)
    arbitrary_geometry["entries"][0]["geometry"]["cases"][0]["geometry_rule"] = "m * n"
    _expect_rejection(module, lambda: _validate(module, arbitrary_geometry, root))

    exact_global_with_tile = copy.deepcopy(record)
    exact_global_with_tile["entries"][0]["geometry"]["cases"][0]["grid_tile_m"] = 64
    _expect_rejection(module, lambda: _validate(module, exact_global_with_tile, root))


def test_shapes_packing_dtypes_and_numerics_cannot_disagree(tmp_path: Path) -> None:
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)

    unknown_dtype = copy.deepcopy(record)
    unknown_dtype["compatibility"]["input_dtype"] = "fp80"
    _expect_rejection(module, lambda: _validate(module, unknown_dtype, root))

    duplicate_dimension = copy.deepcopy(record)
    duplicate_dimension["compatibility"]["shape_families"][0]["fixed_dimensions"].append(
        {"name": "K", "value": 8}
    )
    _expect_rejection(module, lambda: _validate(module, duplicate_dimension, root))

    contradictory_geometry = copy.deepcopy(record)
    contradictory_geometry["compatibility"]["shape_families"][0]["geometry_rule"] = "other-rule"
    _expect_rejection(module, lambda: _validate(module, contradictory_geometry, root))

    packing_missing = copy.deepcopy(record)
    del packing_missing["compatibility"]["weight_packing_version"]
    _expect_rejection(module, lambda: _validate(module, packing_missing, root))

    numerical_dtype_mismatch = copy.deepcopy(record)
    numerical_dtype_mismatch["numerics"]["output_dtype"] = "fp32"
    _expect_rejection(module, lambda: _validate(module, numerical_dtype_mismatch, root))

    missing_tolerance = copy.deepcopy(record)
    missing_tolerance["numerics"]["tolerance_policy"] = ""
    _expect_rejection(module, lambda: _validate(module, missing_tolerance, root))

    unknown_reference_set = copy.deepcopy(record)
    unknown_reference_set["numerics"]["reference_set_kind"] = "implicit"
    _expect_rejection(module, lambda: _validate(module, unknown_reference_set, root))


def test_f2_wmma_shape_tail_packing_and_dual_numerical_references_are_exact(
    tmp_path: Path,
) -> None:
    module, root, scalar_record, _ = _fixture(tmp_path)
    record, digest = _f2_variant(module, root, scalar_record)
    _validate(module, record, root)

    family = record["compatibility"]["shape_families"][0]
    assert family["fixed_dimensions"] == [
        {"name": "K", "value": 2048},
        {"name": "N", "value": 8192},
    ]
    assert family["runtime_dimension"] == {
        "name": "M",
        "min_value": 1,
        "max_value": 128,
        "full_value": 128,
    }
    assert family["tail_policy"] == "masked/padded"
    assert family["geometry_rule"] == "f2-wmma-64x64-m-tail-v1"
    assert record["entries"][0]["geometry"]["cases"][0]["workgroup_x"] == 128
    assert record["entries"][0]["geometry"]["cases"][0]["workgroup_y"] == 4
    assert record["entries"][0]["geometry"]["cases"][0]["grid_tile_m"] == 64
    assert record["entries"][0]["geometry"]["cases"][0]["grid_tile_n"] == 64
    assert record["numerics"]["numpy_oracle"]["input_digest"] == SHA_A
    assert record["numerics"]["scalar_native_projection"]["input_digest"] == SHA_A
    assert record["numerics"]["numpy_oracle"]["output_digest"] != record["numerics"]["scalar_native_projection"]["output_digest"]
    assert module.compute_pack_sha256(record) == digest


def test_conditional_reference_sets_and_layout_proof_are_fail_closed(tmp_path: Path) -> None:
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)

    b0_with_numpy_oracle = copy.deepcopy(record)
    b0_with_numpy_oracle["numerics"]["numpy_oracle"] = copy.deepcopy(
        b0_with_numpy_oracle["numerics"]["retained_reference"]
    )
    _expect_rejection(module, lambda: _validate(module, b0_with_numpy_oracle, root))

    b0_without_retained = copy.deepcopy(record)
    b0_without_retained["numerics"]["retained_reference"] = None
    _expect_rejection(module, lambda: _validate(module, b0_without_retained, root))

    distinct_physical_pack = copy.deepcopy(record)
    distinct_physical_pack["compatibility"]["weight_packing_version"] = "f2-wmma-physical-tile-v1"
    _expect_rejection(module, lambda: _validate(module, distinct_physical_pack, root))

    malformed_layout_proof = copy.deepcopy(record)
    malformed_layout_proof["evidence"]["layout_proof"] = copy.deepcopy(
        malformed_layout_proof["evidence"]["resource_review"]
    )
    malformed_layout_proof["evidence"]["layout_proof"]["evidence_slot"] = "resource_review"
    _expect_rejection(module, lambda: _validate(module, malformed_layout_proof, root))


def test_b0_source_review_is_required_and_exactly_bound(tmp_path: Path) -> None:
    module, root, record, digest = _fixture(tmp_path)
    _validate(module, record, root)
    source_review = record["evidence"]["source_review"]
    assert source_review["record_kind"] == "offline_review"
    assert source_review["evidence_slot"] == "source_review"
    assert source_review["subject_target"] == TARGET
    assert source_review["image_sha256"] == IMAGE_SHA256
    assert source_review["pack_sha256"] == digest

    missing = copy.deepcopy(record)
    del missing["evidence"]["source_review"]
    _expect_rejection(module, lambda: _validate(module, missing, root))

    wrong_identity = copy.deepcopy(record)
    wrong_identity["evidence"]["source_review"]["evidence_slot"] = "isa_review"
    _expect_rejection(module, lambda: _validate(module, wrong_identity, root))


def test_evidence_ref_exposes_the_exact_five_kind_nine_slot_matrix(tmp_path: Path) -> None:
    module, root, record, digest = _fixture(tmp_path)
    _validate(module, record, root)
    assert EVIDENCE_FIELDS == {
        "record_path",
        "record_kind",
        "evidence_slot",
        "record_id",
        "record_sha256",
        "subject_target",
        "image_sha256",
        "pack_sha256",
        "producer_kind",
        "tool_digest",
        "input_digest",
        "output_digest",
    }
    assert EVIDENCE_KINDS == {
        "offline_oracle",
        "offline_review",
        "target_conformance",
        "native_run",
        "benchmark",
    }
    assert EVIDENCE_SLOTS == {
        "numpy_oracle",
        "source_review",
        "isa_review",
        "resource_review",
        "layout_proof",
        "scalar_native_projection",
        "conformance",
        "native_run",
        "benchmark",
    }
    assert len(ALLOWED_EVIDENCE_PAIRS) == 9

    for kind in sorted(EVIDENCE_KINDS):
        for slot in sorted(EVIDENCE_SLOTS):
            ref = _matrix_ref(kind, slot, pack_sha256=digest)
            if (kind, slot) in ALLOWED_EVIDENCE_PAIRS:
                module.validate_evidence_ref(
                    ref,
                    subject_target=TARGET,
                    image_sha256=IMAGE_SHA256,
                    pack_sha256=digest,
                )
            else:
                _expect_rejection(
                    module,
                    lambda ref=ref: module.validate_evidence_ref(
                        ref,
                        subject_target=TARGET,
                        image_sha256=IMAGE_SHA256,
                        pack_sha256=digest,
                    ),
                )


def test_evidence_conditional_fields_are_unconditional_and_exact(tmp_path: Path) -> None:
    module, root, record, digest = _fixture(tmp_path)
    _validate(module, record, root)

    for field, bad_value in (
        ("tool_digest", SHA_A),
        ("subject_target", TARGET),
        ("image_sha256", IMAGE_SHA256),
        ("pack_sha256", digest),
    ):
        malformed = copy.deepcopy(record["numerics"]["retained_reference"])
        malformed[field] = bad_value
        _expect_rejection(
            module,
            lambda malformed=malformed: module.validate_evidence_ref(
                malformed,
                subject_target=TARGET,
                image_sha256=IMAGE_SHA256,
                pack_sha256=digest,
            ),
        )

    malformed_native = copy.deepcopy(record["evidence"]["native_run"])
    malformed_native["tool_digest"] = SHA_A
    _expect_rejection(
        module,
        lambda: module.validate_evidence_ref(
            malformed_native,
            subject_target=TARGET,
            image_sha256=IMAGE_SHA256,
            pack_sha256=digest,
        ),
    )

    malformed_review = copy.deepcopy(record["evidence"]["resource_review"])
    malformed_review["producer_kind"] = "r9700_native"
    _expect_rejection(
        module,
        lambda: module.validate_evidence_ref(
            malformed_review,
            subject_target=TARGET,
            image_sha256=IMAGE_SHA256,
            pack_sha256=digest,
        ),
    )


def test_pack_sha256_is_rfc8785_style_canonical_nonrecursive_and_evidence_excluded(
    tmp_path: Path,
) -> None:
    module, root, record, digest = _fixture(tmp_path)
    assert module.compute_pack_sha256(record) == digest
    assert digest == _independent_pack_sha256(record)

    reordered = _reverse_mapping_order(record)
    assert module.compute_pack_sha256(reordered) == digest

    evidence_changed = copy.deepcopy(record)
    evidence_changed["evidence"]["conformance"]["record_id"] = "different-request-record"
    evidence_changed["evidence"]["native_run"]["output_digest"] = SHA_A
    evidence_changed["numerics"]["retained_reference"]["pack_sha256"] = SHA_D
    assert module.compute_pack_sha256(evidence_changed) == digest

    identity_changed = copy.deepcopy(record)
    identity_changed["entries"][0]["resources"]["rsrc1"] = 99
    assert module.compute_pack_sha256(identity_changed) != digest


def test_isa_and_resource_reviews_bind_exact_tool_input_output_digests(
    tmp_path: Path,
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)

    for section, field in (
        ("isa_review", "tool_digest"),
        ("isa_review", "input_digest"),
        ("isa_review", "output_digest"),
        ("resource_review", "tool_digest"),
        ("resource_review", "input_digest"),
        ("resource_review", "output_digest"),
    ):
        malformed = copy.deepcopy(record)
        malformed["evidence"][section][field] = ""
        _expect_rejection(module, lambda malformed=malformed: _validate(module, malformed, root))

    wrong_kind = copy.deepcopy(record)
    wrong_kind["evidence"]["isa_review"]["record_kind"] = "target_conformance"
    wrong_kind["evidence"]["isa_review"]["producer_kind"] = "r9700_native"
    wrong_kind["evidence"]["isa_review"]["tool_digest"] = ""
    _expect_rejection(module, lambda: _validate(module, wrong_kind, root))


def test_pack_sha256_rejects_nonfinite_numbers_and_wrong_bound_evidence_digest(tmp_path: Path) -> None:
    module, root, record, digest = _fixture(tmp_path)

    nonfinite = copy.deepcopy(record)
    nonfinite["entries"][0]["resources"]["lds_bytes"] = math.nan
    _expect_rejection(module, lambda: module.compute_pack_sha256(nonfinite))
    _expect_rejection(module, lambda: _validate(module, nonfinite, root))

    wrong_pack_digest = copy.deepcopy(record)
    wrong_pack_digest["evidence"]["conformance"]["pack_sha256"] = SHA_D
    _expect_rejection(module, lambda: _validate(module, wrong_pack_digest, root))


def test_policy_input_accepts_only_the_pinned_p3_source_record(tmp_path: Path) -> None:
    module, root, record, _ = _fixture(tmp_path)
    assert UPSTREAM_POLICY.is_file()
    module.validate_manifest(record, asset_root=root, policy_path=UPSTREAM_POLICY)

    wrong_pin = copy.deepcopy(record)
    wrong_pin["provenance"]["upstream_revision"] = "1" * 40
    _expect_rejection(
        module,
        lambda: module.validate_manifest(
            wrong_pin,
            asset_root=root,
            policy_path=UPSTREAM_POLICY,
        ),
    )

    wrong_path = copy.deepcopy(record)
    wrong_path["provenance"]["upstream_paths"] = ["llvm/lib/Target/AMDGPU/AMDGPU.td"]
    _expect_rejection(
        module,
        lambda: module.validate_manifest(
            wrong_path,
            asset_root=root,
            policy_path=UPSTREAM_POLICY,
        ),
    )


def test_generated_cpp_initializers_are_reproducible_allocation_free_views(tmp_path: Path) -> None:
    module, root, record, digest = _fixture(tmp_path)
    _validate(module, record, root)

    first = module.generate_cpp_initializers(record)
    second = module.generate_cpp_initializers(copy.deepcopy(record))
    reordered = module.generate_cpp_initializers(_reverse_mapping_order(record))
    assert first == second == reordered
    assert record["name"] in first
    assert digest in first
    assert "KernelPackRecord" in first
    assert "KernelPackEvidence value{};" in first
    assert first.index("KernelPackEvidence value{};") < first.index("value.conformance")
    assert "std::string_view" in first
    assert "KernelPackSpan" in first
    assert "KernelPackOptional" in first

    for forbidden in (
        "std::string ",
        "std::vector",
        "std::map",
        "new ",
        "malloc(",
        "free(",
        "ifstream",
        "fopen(",
        "json",
        "yaml",
        "curl",
        "socket",
        "hipLaunch",
        "cuda",
        "__DATE__",
        "__TIME__",
    ):
        assert forbidden not in first.lower(), f"generated runtime view contains {forbidden!r}"

    assert str(root) not in first
    assert "upstream-reference-manifest" not in first


def test_validation_and_generation_are_offline_and_never_launch_network_or_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, root, record, _ = _fixture(tmp_path)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("offline manifest tooling attempted external/GPU work")

    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    monkeypatch.setattr("ctypes.CDLL", forbidden)

    _validate(module, record, root)
    output = module.generate_cpp_initializers(record)
    assert output


def test_runtime_boundary_is_not_a_yaml_or_json_manifest_parser(tmp_path: Path) -> None:
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)
    generated = module.generate_cpp_initializers(record)
    normalized = generated.lower()
    assert "upstream-reference-manifest.yaml" not in normalized
    assert "yaml-cpp" not in normalized
    assert "nlohmann::json" not in normalized
    assert "ifstream" not in normalized
    assert "std::filesystem" not in normalized


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("name", ""),
        ("version", "1.0"),
        ("required_features", ["wave32", "wave32"]),
        ("entries", []),
        ("image", None),
        ("compatibility", None),
        ("numerics", None),
        ("evidence", None),
    ],
)
def test_required_record_shape_and_identity_fields_reject_malformed_values(
    tmp_path: Path, field: str, value: Any
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    malformed[field] = value
    _expect_rejection(module, lambda: _validate(module, malformed, root))


def test_f2_layout_record_digest_matches_the_consumer_preimage_contract(
    tmp_path: Path,
) -> None:
    """The P3 resolver accepts the F2 producer's non-self-referential digest."""

    module, root, scalar_record, _ = _fixture(tmp_path)
    record, digest = _f2_variant(module, root, scalar_record)
    layout_ref = record["evidence"]["layout_proof"]
    payload = json.loads((root / layout_ref["record_path"]).read_text(encoding="utf-8"))

    assert payload["record_sha256"] == layout_ref["record_sha256"]
    assert payload["pack_sha256"] == digest
    assert payload["record_sha256"] == _sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "record_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _validate(module, record, root)

def _rewrite_evidence_record(
    root: Path, reference: dict[str, str], payload: dict[str, Any]
) -> None:
    """Write a payload and reseal its ordinary or layout-proof digest."""
    if reference["evidence_slot"] == "layout_proof":
        payload = dict(payload)
        payload["record_sha256"] = _sha256(
            json.dumps(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "record_sha256"
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        data = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        reference["record_sha256"] = payload["record_sha256"]
    else:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        reference["record_sha256"] = _sha256(data)
    (root / reference["record_path"]).write_bytes(data)


def _evidence_payload(reference: dict[str, str], **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "record_id": reference["record_id"],
        "record_kind": reference["record_kind"],
        "evidence_slot": reference["evidence_slot"],
        "subject_target": reference["subject_target"],
        "image_sha256": reference["image_sha256"],
        "pack_sha256": reference["pack_sha256"],
        "tool_digest": reference["tool_digest"],
        "input_digest": reference["input_digest"],
        "output_digest": reference["output_digest"],
    }
    payload.update(fields)
    return payload


def _reseal_mutated_record(module: ModuleType, root: Path, record: dict[str, Any]) -> str:
    """Reseal a mutated fixture without leaving stale identity/report bytes."""
    refs = [
        record["evidence"]["conformance"],
        record["evidence"]["source_review"],
        record["evidence"]["native_run"],
        record["evidence"]["resource_review"],
        record["evidence"]["isa_review"],
        record["evidence"].get("layout_proof"),
        record["evidence"].get("benchmark_record"),
        record["numerics"].get("retained_reference"),
        record["numerics"].get("numpy_oracle"),
        record["numerics"].get("scalar_native_projection"),
    ]
    for reference in refs:
        if reference is None or reference["record_kind"] == "offline_oracle":
            continue
        reference["subject_target"] = record["target"]
        reference["image_sha256"] = record["image"]["image_sha256"]

    digest = _reseal_pack(module, record)
    _sync_evidence_payloads(root, record)

    resource_ref = record["evidence"]["resource_review"]
    resource_payload = json.loads(
        (root / resource_ref["record_path"]).read_text(encoding="utf-8")
    )
    for field in RESOURCE_REPORT_FIELDS:
        resource_payload[field] = record["entries"][0]["resources"][field]
    _rewrite_evidence_record(root, resource_ref, resource_payload)

    layout_ref = record["evidence"].get("layout_proof")
    if layout_ref is not None:
        layout_payload = json.loads(
            (root / layout_ref["record_path"]).read_text(encoding="utf-8")
        )
        for field in EVIDENCE_PAYLOAD_IDENTITY_FIELDS:
            layout_payload[field] = layout_ref[field]
        _rewrite_evidence_record(root, layout_ref, layout_payload)
    return digest




def test_isa_and_resource_report_contents_are_bound_to_the_manifest(
    tmp_path: Path,
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    resource_values = record["entries"][0]["resources"]
    resource_report = {
        key: resource_values[key]
        for key in (
            "rsrc1",
            "rsrc2",
            "rsrc3",
            "wave_size",
            "sgpr_count",
            "vgpr_count",
            "lds_bytes",
            "private_segment_bytes",
            "metadata_provenance",
        )
    }
    resource_mismatch = dict(resource_report)
    resource_mismatch["rsrc1"] += 1
    isa_report = {
        "isa_categories": ["scalar"],
        "unsupported_instructions": [],
    }
    isa_mismatch = {
        **isa_report,
        "isa_categories": ["unsupported"],
        "unsupported_instructions": ["not-an-admitted-instruction"],
    }
    cases = (
        ("resource_review", resource_mismatch),
        ("isa_review", isa_mismatch),
        ("resource_review", {**resource_report, "tool_digest": SHA_D}),
    )
    for section, report_fields in cases:
        malformed = copy.deepcopy(record)
        reference = malformed["evidence"][section]
        _rewrite_evidence_record(
            root,
            reference,
            _evidence_payload(reference, **report_fields),
        )
        _expect_rejection(module, lambda malformed=malformed: _validate(module, malformed, root))


@pytest.mark.parametrize(
    ("section", "missing_field"),
    (
        ("resource_review", "rsrc1"),
        ("isa_review", "isa_categories"),
    ),
    ids=("resource-rsrc1", "isa-categories"),
)
def test_resource_and_isa_reports_require_closed_semantic_fields(
    tmp_path: Path, section: str, missing_field: str
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    reference = malformed["evidence"][section]
    payload = json.loads((root / reference["record_path"]).read_text(encoding="utf-8"))
    del payload[missing_field]
    _rewrite_evidence_record(root, reference, payload)
    _expect_rejection(module, lambda: _validate(module, malformed, root))


@pytest.mark.parametrize("section", ("resource_review", "isa_review"))
@pytest.mark.parametrize(
    "missing_field",
    (
        "record_id",
        "record_kind",
        "evidence_slot",
        "subject_target",
        "image_sha256",
        "pack_sha256",
        "tool_digest",
        "input_digest",
        "output_digest",
    ),
)
def test_resource_and_isa_reports_require_every_identity_binding_field(
    tmp_path: Path, section: str, missing_field: str
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    reference = malformed["evidence"][section]
    payload = json.loads((root / reference["record_path"]).read_text(encoding="utf-8"))
    del payload[missing_field]
    _rewrite_evidence_record(root, reference, payload)
    _expect_rejection(module, lambda: _validate(module, malformed, root))


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        (
            "uint32 fixed dimension",
            lambda value: value["compatibility"]["shape_families"][0]["fixed_dimensions"][0].update(
                value=2**32
            ),
        ),
        (
            "uint64 LDS byte count",
            lambda value: value["entries"][0]["resources"].update(lds_bytes=2**64),
        ),
    ],
    ids=("uint32-overflow", "uint64-overflow"),
)
def test_manifest_rejects_values_outside_generated_cpp_integer_widths(
    tmp_path: Path,
    case: str,
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    mutate(malformed)
    try:
        _reseal_pack(module, malformed)
    except module.ManifestError:
        if case != "uint64 LDS byte count":
            raise
        return

    _expect_rejection(module, lambda: _validate(module, malformed, root))
    _expect_rejection(module, lambda: module.generate_cpp_initializers(malformed))


@pytest.mark.parametrize(
    ("path_kind", "spelling"),
    [
        ("source", "src//demo_kernel.cpp"),
        ("source", "src/./demo_kernel.cpp"),
        ("image", "images//demo_kernel.image"),
        ("image", "images/./demo_kernel.image"),
        ("modification", "generated/b0-demo-kernel-pack.cpp/"),
    ],
    ids=(
        "source-repeated-separator",
        "source-dot-component",
        "image-repeated-separator",
        "image-dot-component",
        "modification-trailing-separator",
    ),
)
def test_manifest_rejects_noncanonical_raw_posix_path_spellings(
    tmp_path: Path, path_kind: str, spelling: str
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    if path_kind == "source":
        old_path = malformed["provenance"]["local_sources"][0]["path"]
        malformed["provenance"]["local_sources"][0]["path"] = spelling
    elif path_kind == "image":
        old_path = malformed["image"]["image_path"]
        malformed["image"]["image_path"] = spelling
    else:
        old_path = malformed["provenance"]["modifications"][0]["component"]
        malformed["provenance"]["modifications"][0]["component"] = spelling
    for review in malformed["provenance"]["license_reviews"]:
        if review["component"] == old_path:
            review["component"] = spelling
    _reseal_pack(module, malformed)

    _expect_rejection(module, lambda: _validate(module, malformed, root))


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        (
            "compatibility input uses kernarg uint32",
            lambda value: (
                value["compatibility"].update(input_dtype="uint32"),
                value["numerics"].update(input_dtype="uint32"),
            ),
        ),
        (
            "compatibility weight uses pointer",
            lambda value: value["compatibility"].update(weight_dtype="pointer"),
        ),
        (
            "compatibility output uses kernarg double",
            lambda value: (
                value["compatibility"].update(output_dtype="double"),
                value["numerics"].update(output_dtype="double"),
            ),
        ),
        (
            "numerics accumulation uses kernarg int64",
            lambda value: (
                value["numerics"].update(accumulation_dtype="int64"),
                value["numerics"]["cast_points"][0].update(to_dtype="int64"),
            ),
        ),
    ],
    ids=("tensor-input", "tensor-weight", "tensor-output", "accumulation"),
)
def test_tensor_and_numerical_dtypes_reject_kernarg_only_vocabulary(
    tmp_path: Path,
    case: str,
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    del case
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)
    malformed = copy.deepcopy(record)
    mutate(malformed)
    _reseal_pack(module, malformed)
    _expect_rejection(module, lambda: _validate(module, malformed, root))


def _generated_exported_symbol(generated: str) -> str:
    matches = re.findall(
        r"const KernelPackRecord& ([A-Za-z_][A-Za-z0-9_]*) = ", generated
    )
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("collision_case", ["version", "sanitized-name"])
def test_generated_symbols_are_unique_for_complete_pack_identity(
    tmp_path: Path, collision_case: str
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    _validate(module, record, root)

    first = copy.deepcopy(record)
    second = copy.deepcopy(record)
    if collision_case == "version":
        second["version"] = "1.0.1"
    else:
        first["name"] = "foo-bar"
        second["name"] = "foo_bar"
    _isolate_evidence_files(root, first, "first")
    _isolate_evidence_files(root, second, "second")
    _reseal_pack(module, first)
    _sync_evidence_payloads(root, first)
    _reseal_pack(module, second)
    _sync_evidence_payloads(root, second)
    _validate(module, first, root)
    _validate(module, second, root)

    first_symbol = _generated_exported_symbol(module.generate_cpp_initializers(first))
    second_symbol = _generated_exported_symbol(module.generate_cpp_initializers(second))
    assert first_symbol != second_symbol


def test_control_characters_are_rejected_or_emitted_as_valid_cpp_strings(
    tmp_path: Path,
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    malformed["provenance"]["modifications"][0]["summary"] = (
        "generated offline\x00with control"
    )
    malformed["provenance"]["license_reviews"][0]["review_id"] = "review\x01id"
    malformed["numerics"]["tolerance_policy"] = "b0-control\x02policy"
    malformed["image"]["build"]["toolchain_id"] = "llvm-amdgpu\x03"
    try:
        _reseal_pack(module, malformed)
    except module.ManifestError:
        return

    try:
        _validate(module, malformed, root)
    except module.ManifestError:
        return
    try:
        generated = module.generate_cpp_initializers(malformed)
    except module.ManifestError:
        return
    assert not re.search(r"\\u00(?:00|01|02|03)", generated)
    assert not any(char in generated for char in "\x00\x01\x02\x03")


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        ("wrong target", lambda value: value.update(target="gfx9999")),
        (
            "unsafe source path",
            lambda value: value["provenance"]["local_sources"][0].update(
                path="../outside.cpp"
            ),
        ),
        (
            "image digest drift",
            lambda value: value["image"].update(image_sha256="00" * 32),
        ),
        (
            "unaccepted license",
            lambda value: value["provenance"]["license_reviews"][0].update(
                status="pending"
            ),
        ),
    ],
    ids=("wrong-target", "unsafe-path", "image-digest", "license-status"),
)
def test_generation_refuses_records_without_full_manifest_validation(
    tmp_path: Path,
    case: str,
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    del case
    module, _root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    mutate(malformed)
    with pytest.raises(module.ManifestError):
        module.generate_cpp_initializers(malformed)


@pytest.mark.parametrize(
    "missing_field",
    tuple(
        field
        for field in PHYSICAL_LAYOUT_FIELDS
        if field not in {"inverse_n", "inverse_k", "inverse_source_f16"}
    ),
)
def test_f2_layout_proof_requires_complete_mapping_spec_and_fixture_identity(
    tmp_path: Path, missing_field: str
) -> None:
    module, root, scalar_record, _ = _fixture(tmp_path)
    record, _ = _f2_variant(module, root, scalar_record)
    layout_ref = record["evidence"]["layout_proof"]
    malformed = copy.deepcopy(record)
    payload = json.loads((root / layout_ref["record_path"]).read_text(encoding="utf-8"))
    del payload[missing_field]
    _rewrite_evidence_record(root, malformed["evidence"]["layout_proof"], payload)

    _expect_rejection(module, lambda: _validate(module, malformed, root))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_tensor_layout_version", "wrong-source-layout-v1"),
        ("physical_layout_version", "wrong-physical-layout-v1"),
        ("layout_spec_path", "build/f2-wmma/other-layout-spec.json"),
        ("layout_spec_sha256", SHA_D),
        ("inverse_fixture_path", "build/f2-wmma/other-inverse-fixture.npz"),
        ("inverse_fixture_sha256", SHA_D),
        (
            "layout_mapping",
            {
                "source_element": "source_weight[k*K+n]",
                "physical_byte_offset": "wrong",
                "b_tile": "wrong",
                "lds_byte_offset": "wrong",
            },
        ),
        (
            "layout_strides",
            {
                "source_row_stride_elements": 1,
                "physical_tile_stride_bytes": 1,
                "lds_tile_stride_bytes": 1,
                "tile_n_count": 1,
                "tile_k_count": 1,
            },
        ),
        ("alignment_bytes", 8),
        ("padding_bytes", 1),
        ("swizzle", "xor-v1"),
        ("layout_origin", "unreviewed"),
        ("inverse_fixture_input_digest", SHA_D),
        ("inverse_fixture_output_digest", SHA_D),
        ("layout_status", "fail"),
        ("failure_stage", "inverse-round-trip"),
        ("exit_status", 1),
        ("wrapper_exit_status", 1),
    ],
)
def test_f2_layout_proof_bindings_and_status_are_frozen(
    tmp_path: Path, field: str, value: Any
) -> None:
    module, root, scalar_record, _ = _fixture(tmp_path)
    record, _ = _f2_variant(module, root, scalar_record)
    layout_ref = record["evidence"]["layout_proof"]
    malformed = copy.deepcopy(record)
    payload = json.loads((root / layout_ref["record_path"]).read_text(encoding="utf-8"))
    payload[field] = value
    _rewrite_evidence_record(root, malformed["evidence"]["layout_proof"], payload)

    _expect_rejection(module, lambda: _validate(module, malformed, root))


def test_every_evidence_payload_requires_all_identity_bindings(tmp_path: Path) -> None:
    module, root, base_record, _ = _fixture(tmp_path)
    f2_record, _ = _f2_variant(module, root, base_record)

    benchmark_record = copy.deepcopy(base_record)
    benchmark_record["evidence"]["benchmark_record"] = _evidence_ref(
        root,
        name="benchmark-current-review",
        kind="benchmark",
        slot="benchmark",
        subject_target=TARGET,
        image_sha256=IMAGE_SHA256,
        pack_sha256=PACK_SHA256_PLACEHOLDER,
        producer_kind="r9700_native",
        tool_digest=SHA_A,
        input_digest=SHA_B,
        output_digest=SHA_C,
    )
    benchmark_record["evidence"]["benchmark_not_applicable_reason"] = ""
    _reseal_mutated_record(module, root, benchmark_record)

    cases = (
        ("b0-conformance", base_record, "evidence", "conformance"),
        ("b0-source-review", base_record, "evidence", "source_review"),
        ("b0-native-run", base_record, "evidence", "native_run"),
        ("b0-retained-oracle", base_record, "numerics", "retained_reference"),
        ("f2-layout-proof", f2_record, "evidence", "layout_proof"),
        ("f2-numpy-oracle", f2_record, "numerics", "numpy_oracle"),
        ("f2-native-projection", f2_record, "numerics", "scalar_native_projection"),
        ("benchmark", benchmark_record, "evidence", "benchmark_record"),
    )
    for case_name, source_record, container, section in cases:
        for missing_field in EVIDENCE_PAYLOAD_IDENTITY_FIELDS:
            malformed = copy.deepcopy(source_record)
            suffix = f"{case_name}-{missing_field}"
            _isolate_evidence_files(root, malformed, suffix)
            _reseal_mutated_record(module, root, malformed)
            reference = malformed[container][section]
            payload = json.loads(
                (root / reference["record_path"]).read_text(encoding="utf-8")
            )
            del payload[missing_field]
            _rewrite_evidence_record(root, reference, payload)
            _expect_rejection(
                module,
                lambda malformed=malformed: _validate(module, malformed, root),
            )


def test_evidence_record_ids_are_nonempty_in_references_and_payloads(tmp_path: Path) -> None:
    module, root, base_record, _ = _fixture(tmp_path)
    f2_record, _ = _f2_variant(module, root, base_record)
    benchmark_record = copy.deepcopy(base_record)
    benchmark_record["evidence"]["benchmark_record"] = _evidence_ref(
        root,
        name="benchmark-empty-id",
        kind="benchmark",
        slot="benchmark",
        subject_target=TARGET,
        image_sha256=IMAGE_SHA256,
        pack_sha256=PACK_SHA256_PLACEHOLDER,
        producer_kind="r9700_native",
        tool_digest=SHA_A,
        input_digest=SHA_B,
        output_digest=SHA_C,
    )
    benchmark_record["evidence"]["benchmark_not_applicable_reason"] = ""
    _reseal_mutated_record(module, root, benchmark_record)

    cases = (
        ("b0-conformance", base_record, "evidence", "conformance"),
        ("b0-source-review", base_record, "evidence", "source_review"),
        ("b0-native-run", base_record, "evidence", "native_run"),
        ("b0-retained-oracle", base_record, "numerics", "retained_reference"),
        ("f2-layout-proof", f2_record, "evidence", "layout_proof"),
        ("f2-numpy-oracle", f2_record, "numerics", "numpy_oracle"),
        ("f2-native-projection", f2_record, "numerics", "scalar_native_projection"),
        ("benchmark", benchmark_record, "evidence", "benchmark_record"),
    )
    for case_name, source_record, container, section in cases:
        malformed = copy.deepcopy(source_record)
        _isolate_evidence_files(root, malformed, f"{case_name}-empty-id")
        reference = malformed[container][section]
        reference["record_id"] = ""
        _reseal_mutated_record(module, root, malformed)
        _expect_rejection(module, lambda malformed=malformed: _validate(module, malformed, root))


def test_metadata_provenance_uses_one_runtime_admitted_cited_value(tmp_path: Path) -> None:
    module, root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    malformed["entries"][0]["resources"][
        "metadata_provenance"
    ] = "source_amdgpu_metadata"
    _reseal_mutated_record(module, root, malformed)

    _expect_rejection(module, lambda: _validate(module, malformed, root))


@pytest.mark.parametrize("field", ("rsrc1", "rsrc2", "rsrc3", "sgpr_count", "vgpr_count"))
def test_required_resource_registers_and_counts_are_positive(
    tmp_path: Path, field: str
) -> None:
    module, root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    malformed["entries"][0]["resources"][field] = 0
    _reseal_mutated_record(module, root, malformed)

    _expect_rejection(module, lambda: _validate(module, malformed, root))


def test_image_bytes_must_be_nonempty(tmp_path: Path) -> None:
    module, root, record, _ = _fixture(tmp_path)
    malformed = copy.deepcopy(record)
    empty_image_path = "images/empty-kernel.image"
    (root / empty_image_path).parent.mkdir(parents=True, exist_ok=True)
    (root / empty_image_path).write_bytes(b"")
    old_image_path = malformed["image"]["image_path"]
    malformed["image"].update(
        image_path=empty_image_path,
        image_sha256=_sha256(b""),
        image_size=0,
    )
    for review in malformed["provenance"]["license_reviews"]:
        if review["component"] == old_image_path:
            review["component"] = empty_image_path
    _reseal_mutated_record(module, root, malformed)

    _expect_rejection(module, lambda: module._validate_image(malformed["image"], root))


@pytest.mark.parametrize("field", ("lds_bytes", "private_segment_bytes"))
def test_pack_sha256_rejects_integers_outside_rfc8785_interoperable_range(
    tmp_path: Path, field: str
) -> None:
    module, _root, record, _ = _fixture(tmp_path)
    at_limit = copy.deepcopy(record)
    at_limit["entries"][0]["resources"][field] = (2**53) - 1
    assert module.compute_pack_sha256(at_limit)

    outside = copy.deepcopy(record)
    outside["entries"][0]["resources"][field] = 2**53
    _expect_rejection(module, lambda: module.compute_pack_sha256(outside))
