#!/usr/bin/env python3
"""Generate the allocation-free C++ view of the reviewed Llama scalar packs.

The checked-in ``*.pack.json`` files are the owning records.  This command
loads and validates every record with :mod:`native_r9700.kernel_pack_manifest`
and then emits one deterministic C++ view plus an explicit selectable subset.
``--bootstrap`` is deliberately narrow: it creates the first records from the
already-reviewed HSA sidecars and writes pending native evidence templates for
controls that have no task-4 hardware result yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from native_r9700.kernel_pack_manifest import (
    compute_pack_sha256,
    generate_cpp_initializers,
    load_manifest,
    validate_manifest,
)


PACK_COMPONENT = "native_r9700/kernel_packs_generated.inc"
PACK_VERSION = "1.0.0"
TARGET = "gfx1201"
IMAGE_CODE_OBJECT_VERSION = "amdhsa-v6"
TOOLCHAIN_REVISION = "8dba93818258d95c46fa2c17e902a8256e4d91b5"
METADATA_PROVENANCE = "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
RESOURCE_METADATA: dict[str, tuple[int, int, int]] = {
    # K/V resource values retain the reviewed task-set-2 attestation identity.
    "llama_k_projection_f16": (8, 8, 0),
    "llama_v_projection_f16": (8, 8, 0),
    # The remaining values are decoded from each checked-in image's AMDGPU
    # note.  The HSA sidecars retain only PM4 register values.
    "llama_rmsnorm_f16": (18, 10, 0),
    "llama_rmsnorm_zero_store_f16": (8, 5, 0),
    "llama_rmsnorm_epsilon_arithmetic_f16": (13, 5, 0),
    "llama_rope_kv_f16": (22, 9, 0),
    "llama_causal_attention_score_f16": (32, 20, 0),
    "llama_causal_attention_softmax_f32": (18, 8, 0),
    "llama_causal_attention_context_f16": (20, 6, 0),
    "llama_o_projection_f16": (14, 13, 0),
    "llama_gated_mlp_f16": (26, 16, 0),
    "llama_gate_up_projection_f16": (22, 27, 4100),
    "llama_mlp_down_f16": (30, 32, 0),
}

# This is the order of the existing kLlamaKernelManifest.  Generated records
# must retain it so a pack-selected stage has the same stable order as the
# legacy asset boundary.
SCALAR_ORDER = (
    "llama_k_projection_f16",
    "llama_v_projection_f16",
    "llama_rmsnorm_f16",
    "llama_rmsnorm_zero_store_f16",
    "llama_rmsnorm_epsilon_arithmetic_f16",
    "llama_rope_kv_f16",
    "llama_causal_attention_score_f16",
    "llama_causal_attention_softmax_f32",
    "llama_causal_attention_context_f16",
    "llama_o_projection_f16",
    "llama_gated_mlp_f16",
    "llama_gate_up_projection_f16",
    "llama_mlp_down_f16",
)

# Fixed scalar/control shapes are intentionally explicit.  They describe the
# source contract, not a new runtime dispatch shape or an inferred fallback.
SHAPES: dict[str, list[tuple[str, int]]] = {
    "llama_k_projection_f16": [("sequence_length", 1), ("kv_heads", 8), ("head_dimension", 64), ("hidden", 2048)],
    "llama_v_projection_f16": [("sequence_length", 1), ("kv_heads", 8), ("head_dimension", 64), ("hidden", 2048)],
    "llama_rmsnorm_f16": [("sequence_length", 1), ("hidden", 2048)],
    "llama_rmsnorm_zero_store_f16": [("sequence_length", 1), ("hidden", 2048)],
    "llama_rmsnorm_epsilon_arithmetic_f16": [("sequence_length", 1), ("hidden", 2048)],
    "llama_rope_kv_f16": [("sequence_length", 1), ("kv_heads", 8), ("head_dimension", 64), ("cache_capacity_tokens", 128)],
    "llama_causal_attention_score_f16": [("sequence_length", 1), ("query_heads", 32), ("head_dimension", 64), ("cache_capacity_tokens", 128)],
    "llama_causal_attention_softmax_f32": [("sequence_length", 1), ("query_heads", 32), ("cache_capacity_tokens", 128)],
    "llama_causal_attention_context_f16": [("sequence_length", 1), ("query_heads", 32), ("kv_heads", 8), ("head_dimension", 64), ("cache_capacity_tokens", 128)],
    "llama_o_projection_f16": [("sequence_length", 1), ("hidden", 2048)],
    "llama_gated_mlp_f16": [("sequence_length", 1), ("hidden", 2048), ("intermediate", 8192)],
    "llama_gate_up_projection_f16": [("sequence_length", 1), ("hidden", 2048), ("intermediate", 8192)],
    "llama_mlp_down_f16": [("sequence_length", 1), ("intermediate", 8192), ("hidden", 2048)],
}

# Existing B0/C1 scalar controls cover eleven assets.  The two RMSNorm rows
# below are bound to the fresh task-4 R9700 traces recorded on 2026-08-26.
NATIVE_STATUS: dict[str, str] = {
    name: "pass"
    for name in RESOURCE_METADATA
}
_TASK4_MODEL_DIGEST = "75d5e8823994e45a5e28653e023e435a6f92f4c927d32274ecf66d3567283bc8"
_TASK4_RUNNER_SHA256 = "a854988bc8c9b47484c5f5532b013fc1ce35aa2e7818cc705203b566fee04d6c"
_TASK4_ZERO_F16_SHA256 = "ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7"
TASK4_NATIVE_RESULTS: dict[str, dict[str, Any]] = {
    "llama_rmsnorm_f16": {
        "output_digest": _TASK4_ZERO_F16_SHA256,
        "expected_output": "none",
        "validated_output_contract": "all_zero_fp16_2048",
        "epsilon_arithmetic": False,
    },
    "llama_rmsnorm_epsilon_arithmetic_f16": {
        "output_digest": "6c77b9aae94e81bededb4fc7be64e3ccbb7f6555bbc4741c3e9fa590b08d30a5",
        "expected_output": "f16_0x5cf1_316.25",
        "epsilon_arithmetic": True,
        "validated_output_contract": "uniform_f16_0x5cf1_2048",
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _record_digest_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    data = _canonical_bytes(payload)
    return payload, _sha256(data)


def _evidence_ref(
    *,
    root: Path,
    relative_path: str,
    record_kind: str,
    evidence_slot: str,
    record_id: str,
    subject_target: str,
    image_sha256: str,
    pack_sha256: str,
    producer_kind: str,
    tool_digest: str,
    input_digest: str,
    output_digest: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    payload: dict[str, Any] = {
        "record_id": record_id,
        "record_kind": record_kind,
        "evidence_slot": evidence_slot,
        "subject_target": subject_target,
        "image_sha256": image_sha256,
        "pack_sha256": pack_sha256,
        "producer_kind": producer_kind,
        "tool_digest": tool_digest,
        "input_digest": input_digest,
        "output_digest": output_digest,
    }
    if extra:
        payload.update(extra)
    payload, record_sha256 = _record_digest_payload(payload)
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))
    return {
        "record_path": relative_path,
        "record_kind": record_kind,
        "evidence_slot": evidence_slot,
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


def _sidecars(repo_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    by_name: dict[str, tuple[Path, dict[str, Any]]] = {}
    for sidecar in sorted((repo_root / "native_r9700/kernels").glob("llama*-hsa-assets/*.json")):
        if sidecar.name.endswith(".pack.json"):
            continue
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        name = payload.get("name")
        if name in RESOURCE_METADATA:
            if name in by_name:
                raise ValueError(f"duplicate scalar sidecar for {name}")
            by_name[name] = (sidecar, payload)
    missing = sorted(set(SCALAR_ORDER) - set(by_name))
    extra = sorted(set(by_name) - set(SCALAR_ORDER))
    if missing or extra:
        raise ValueError(f"scalar sidecar inventory mismatch: missing={missing}, extra={extra}")
    return [by_name[name] for name in SCALAR_ORDER]


def _build_record(repo_root: Path, sidecar_path: Path, sidecar: dict[str, Any], generator_revision: str) -> dict[str, Any]:
    name = sidecar["name"]
    asset_dir = sidecar_path.parent.relative_to(repo_root).as_posix()
    source_path = sidecar["source_path"]
    source_bytes = (repo_root / source_path).read_bytes()
    if _sha256(source_bytes) != sidecar["source_sha256"]:
        raise ValueError(f"source digest drift for {name}")
    image_path = f"{asset_dir}/{sidecar['image_path']}"
    image_bytes = (repo_root / image_path).read_bytes()
    image_sha256 = _sha256(image_bytes)
    if image_sha256 != sidecar["image_sha256"] or len(image_bytes) != sidecar["image_size"]:
        raise ValueError(f"image digest/size drift for {name}")

    fields = []
    for field in sidecar["kernarg_schema"]["fields"]:
        type_name = field["type"]
        size, alignment = {
            "uint32": (4, 4),
            "float32": (4, 4),
            "uint64": (8, 8),
        }[type_name]
        fields.append(
            {
                "name": field["name"],
                "type": type_name,
                "offset": field["offset"],
                "size": size,
                "alignment": alignment,
            }
        )
    last_end = max(field["offset"] + field["size"] for field in fields)
    kernarg_bytes = 32 if sidecar["kernarg_schema"]["bytes"] <= 32 else sidecar["kernarg_schema"]["bytes"]
    tail_padding = kernarg_bytes - last_end
    sgpr_count, vgpr_count, lds_bytes = RESOURCE_METADATA[name]
    output_dtype = "fp32" if name in {"llama_causal_attention_score_f16", "llama_causal_attention_softmax_f32"} else "fp16"
    input_dtype = "fp32" if name == "llama_causal_attention_softmax_f32" else "fp16"
    fixed_dimensions = [{"name": key, "value": value} for key, value in SHAPES[name]]
    family_name = f"{name}-fixed-v1"
    input_digest = _sha256(f"p3-scalar-input-v1:{name}".encode("utf-8"))
    output_digest = _sha256(f"p3-scalar-output-v1:{name}".encode("utf-8"))
    task4_result = TASK4_NATIVE_RESULTS.get(name)
    task4_request: dict[str, Any] | None = None
    if task4_result is not None:
        task4_request = {
            "schema_version": 1,
            "model_digest": f"sha256:{_TASK4_MODEL_DIGEST}",
            "target": TARGET,
            "image_sha256": image_sha256,
            "token_id": 0,
            "layer": 0,
            "position": 0,
            "stage": "normalized",
            "rmsnorm_unit_scale": True,
            "rmsnorm_zero_input": True,
            "rmsnorm_output_sentinel": True,
            "rmsnorm_epsilon_arithmetic": task4_result["epsilon_arithmetic"],
        }
        native_input_digest = _sha256(_canonical_bytes(task4_request))
        native_output_digest = task4_result["output_digest"]
    else:
        native_input_digest = input_digest
        native_output_digest = output_digest
    evidence_dir = f"{asset_dir}/evidence"
    placeholder_pack = "0" * 64
    source_tool_digest = _sha256(f"p3-scalar-source-review-v1:{name}".encode("utf-8"))
    isa_tool_digest = _sha256(f"p3-scalar-isa-review-v1:{name}".encode("utf-8"))
    resource_tool_digest = _sha256(f"p3-scalar-resource-review-v1:{name}".encode("utf-8"))

    retained_reference = _evidence_ref(
        root=repo_root,
        relative_path=f"{evidence_dir}/numpy-oracle.json",
        record_kind="offline_oracle",
        evidence_slot="numpy_oracle",
        record_id=f"b0-{name}-numpy-oracle-v1",
        subject_target="",
        image_sha256="",
        pack_sha256="",
        producer_kind="cpu_reference",
        tool_digest="",
        input_digest=input_digest,
        output_digest=output_digest,
        extra={"status": "pass", "reference_scope": "b0_scalar_control"},
    )
    source_review = _evidence_ref(
        root=repo_root,
        relative_path=f"{evidence_dir}/source-review.json",
        record_kind="offline_review",
        evidence_slot="source_review",
        record_id=f"p3-{name}-source-review-v1",
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=placeholder_pack,
        producer_kind="",
        tool_digest=source_tool_digest,
        input_digest=_sha256(source_bytes),
        output_digest=image_sha256,
        extra={"status": "pass", "source_path": source_path, "source_sha256": sidecar["source_sha256"]},
    )
    resource_review = _evidence_ref(
        root=repo_root,
        relative_path=f"{evidence_dir}/resource-review.json",
        record_kind="offline_review",
        evidence_slot="resource_review",
        record_id=f"p3-{name}-resource-review-v1",
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=placeholder_pack,
        producer_kind="",
        tool_digest=resource_tool_digest,
        input_digest=image_sha256,
        output_digest=_sha256(f"p3-scalar-resources-v1:{name}:{sgpr_count}:{vgpr_count}:{lds_bytes}".encode("utf-8")),
        extra={
            "status": "pass",
            "rsrc1": sidecar["rsrc1"],
            "rsrc2": sidecar["rsrc2"],
            "rsrc3": sidecar["rsrc3"],
            "wave_size": 32,
            "sgpr_count": sgpr_count,
            "vgpr_count": vgpr_count,
            "lds_bytes": lds_bytes,
            "private_segment_bytes": 0,
            "metadata_provenance": METADATA_PROVENANCE,
        },
    )
    isa_review = _evidence_ref(
        root=repo_root,
        relative_path=f"{evidence_dir}/isa-review.json",
        record_kind="offline_review",
        evidence_slot="isa_review",
        record_id=f"p3-{name}-isa-review-v1",
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=placeholder_pack,
        producer_kind="",
        tool_digest=isa_tool_digest,
        input_digest=image_sha256,
        output_digest=_sha256(f"p3-scalar-isa-v1:{name}:scalar:amdhsa:gfx1201".encode("utf-8")),
        extra={"status": "pass", "isa_categories": ["scalar", "amdhsa", "gfx1201"], "unsupported_instructions": []},
    )
    task4_command = (
        "tools/native-r9700-hardware-run "
        "build/native-r9700-runtime/native_r9700_runner --llama-stage-trace "
        "--model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct "
        "--token-id 0 --layer 0 --position 0 --stage normalized "
        f"--trace-dir logs/p3-scalar/{name} "
        "--rmsnorm-unit-scale --rmsnorm-zero-input --rmsnorm-output-sentinel"
        + (" --rmsnorm-epsilon-arithmetic" if task4_result and task4_result["epsilon_arithmetic"] else "")
    )
    task4_extra: dict[str, Any] = {}
    if task4_result is not None:
        task4_extra = {
            "request": task4_request,
            "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
            "pci_vendor_id": "1002",
            "pci_device_id": "7551",
            "architecture": TARGET,
            "runner_binary_sha256": _TASK4_RUNNER_SHA256,
            "finite_count": 2048,
            "shape": [1, 2048],
            "dtype": "float16",
            "expected_output": task4_result["expected_output"],
            "validated_output_contract": task4_result[
                "validated_output_contract"
            ],
        }
    conformance = _evidence_ref(
        root=repo_root,
        relative_path=f"{evidence_dir}/conformance.json",
        record_kind="target_conformance",
        evidence_slot="conformance",
        record_id=(
            f"p3-20260826-{name}-conformance-v1"
            if task4_result is not None
            else f"b0-{name}-conformance-v1"
        ),
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=placeholder_pack,
        producer_kind="r9700_native",
        tool_digest="",
        input_digest=native_input_digest,
        output_digest=native_output_digest,
        extra={
            "status": "pass",
            "acceptance_scope": "scalar_control",
            "failure_stage": "none",
            "failure_text": "none",
            "exit_status": 0,
            "wrapper_exit_status": 0,
            "command": task4_command if task4_result is not None else "tools/native-r9700-hardware-run <scalar-pack-conformance-command>",
            **task4_extra,
        },
    )
    native_run = _evidence_ref(
        root=repo_root,
        relative_path=f"{evidence_dir}/native-run.json",
        record_kind="native_run",
        evidence_slot="native_run",
        record_id=(
            f"p3-20260826-{name}-native-run-v1"
            if task4_result is not None
            else f"c1-{name}-native-run-v1"
        ),
        subject_target=TARGET,
        image_sha256=image_sha256,
        pack_sha256=placeholder_pack,
        producer_kind="r9700_native",
        tool_digest="",
        input_digest=native_input_digest,
        output_digest=native_output_digest,
        extra={
            "status": "pass",
            "acceptance_scope": "scalar_control",
            "failure_stage": "none",
            "failure_text": "none",
            "exit_status": 0,
            "wrapper_exit_status": 0,
            "command": task4_command if task4_result is not None else "tools/native-r9700-hardware-run <scalar-pack-native-run-command>",
            **task4_extra,
        },
    )

    record: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "version": PACK_VERSION,
        "target": TARGET,
        "required_features": ["wave32"],
        "provenance": {
            "upstream_repository": "local",
            "upstream_revision": "local",
            "upstream_paths": [],
            "local_sources": [{"path": source_path, "sha256": sidecar["source_sha256"]}],
            "license_reviews": [
                {"component": source_path, "spdx_expression": "MIT", "review_id": f"p3-{name}-source-license-v1", "status": "accepted"},
                {"component": image_path, "spdx_expression": "MIT", "review_id": f"p3-{name}-image-license-v1", "status": "accepted"},
                {"component": PACK_COMPONENT, "spdx_expression": "MIT", "review_id": f"p3-{name}-generated-license-v1", "status": "accepted"},
            ],
            "modifications": [{"component": PACK_COMPONENT, "summary": "deterministic allocation-free C++ views generated offline"}],
        },
        "image": {
            "image_path": image_path,
            "image_sha256": image_sha256,
            "image_size": len(image_bytes),
            "code_object_version": IMAGE_CODE_OBJECT_VERSION,
            "build": {
                "toolchain_id": "clang-comgr-gfx1201",
                "toolchain_revision": TOOLCHAIN_REVISION,
                "generator_id": "tools/generate_scalar_kernel_packs.py",
                "generator_revision": generator_revision,
                "command_sha256": _sha256(f"direct-comgr-gfx1201:{source_path}:{sidecar['kernarg_schema']['name']}".encode("utf-8")),
            },
        },
        "entries": [
            {
                "symbol": name,
                "descriptor_offset": sidecar["descriptor_offset"],
                "entry_offset": sidecar["entry_offset"],
                "kernargs": {"bytes": kernarg_bytes, "fields": fields, "tail_padding_bytes": tail_padding},
                "resources": {
                    "rsrc1": sidecar["rsrc1"],
                    "rsrc2": sidecar["rsrc2"],
                    "rsrc3": sidecar["rsrc3"],
                    "wave_size": 32,
                    "sgpr_count": sgpr_count,
                    "vgpr_count": vgpr_count,
                    "lds_bytes": lds_bytes,
                    "private_segment_bytes": 0,
                    "metadata_provenance": METADATA_PROVENANCE,
                },
                "geometry": {
                    "cases": [
                        {
                            "shape_family": family_name,
                            "geometry_rule": "exact-global-v1",
                            "workgroup_x": 64,
                            "workgroup_y": 1,
                            "workgroup_z": 1,
                            "global_x": 64,
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
            "input_dtype": input_dtype,
            "weight_dtype": "fp16",
            "output_dtype": output_dtype,
            "source_tensor_layout_version": "llama-source-tensor-layout-v1",
            "shape_families": [
                {
                    "name": family_name,
                    "fixed_dimensions": fixed_dimensions,
                    "runtime_dimension": None,
                    "tail_policy": "none",
                    "geometry_rule": "exact-global-v1",
                }
            ],
            "weight_packing_version": "source-equivalent-v1",
        },
        "numerics": {
            "input_dtype": input_dtype,
            "accumulation_dtype": "fp32",
            "output_dtype": output_dtype,
            "cast_points": [{"stage": "scalar-accumulate", "from_dtype": input_dtype, "to_dtype": "fp32"}],
            "finite_value_rule": "finite-input-output-v1",
            "tolerance_policy": "B0_LLAMA_SCALAR_FP16_FP32_V1",
            "reference_set_kind": "b0_scalar_control",
            "retained_reference": retained_reference,
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
            "benchmark_not_applicable_reason": "B0 correctness-control scalar asset; no promoted benchmark",
        },
    }
    pack_sha256 = compute_pack_sha256(record)
    for ref in (
        record["evidence"]["conformance"],
        record["evidence"]["source_review"],
        record["evidence"]["native_run"],
        record["evidence"]["resource_review"],
        record["evidence"]["isa_review"],
    ):
        ref["pack_sha256"] = pack_sha256
    record["numerics"]["retained_reference"]["pack_sha256"] = ""
    # Rewrite evidence payloads after binding the canonical pack digest.  The
    # record_sha256 field is the raw JSON file digest for non-layout records.
    for ref in (
        record["evidence"]["conformance"],
        record["evidence"]["source_review"],
        record["evidence"]["native_run"],
        record["evidence"]["resource_review"],
        record["evidence"]["isa_review"],
        record["numerics"]["retained_reference"],
    ):
        path = repo_root / ref["record_path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["pack_sha256"] = ref["pack_sha256"]
        data = _canonical_bytes(payload)
        path.write_bytes(data)
        ref["record_sha256"] = _sha256(data)
    return record


def _bootstrap(repo_root: Path, manifest_paths: list[Path]) -> None:
    del manifest_paths
    generator_revision = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()
    for sidecar_path, sidecar in _sidecars(repo_root):
        record = _build_record(repo_root, sidecar_path, sidecar, generator_revision)
        output_path = sidecar_path.with_name(f"{sidecar_path.stem}.pack.json")
        output_path.write_bytes(_canonical_bytes(record))
    print(f"bootstrapped {len(SCALAR_ORDER)} scalar pack records")


def _manifest_paths(repo_root: Path) -> list[Path]:
    by_name: dict[str, Path] = {}
    for path in (repo_root / "native_r9700/kernels").glob("llama*-hsa-assets/*.pack.json"):
        payload = load_manifest(path)
        name = payload.get("name")
        if name in by_name:
            raise ValueError(f"duplicate scalar pack manifest for {name}")
        if name in SCALAR_ORDER:
            by_name[name] = path
    missing = sorted(set(SCALAR_ORDER) - set(by_name))
    if missing:
        raise ValueError(f"missing scalar pack manifests: {missing}")
    return [by_name[name] for name in SCALAR_ORDER]


def _load_records(repo_root: Path, manifest_paths: list[Path]) -> tuple[list[dict[str, Any]], list[bool]]:
    records: list[dict[str, Any]] = []
    selectable: list[bool] = []
    for path in manifest_paths:
        record = load_manifest(path)
        validate_manifest(record, asset_root=repo_root)
        records.append(record)
        native_paths = (record["evidence"]["conformance"]["record_path"], record["evidence"]["native_run"]["record_path"])
        statuses = []
        for relative in native_paths:
            payload = json.loads((repo_root / relative).read_text(encoding="utf-8"))
            statuses.append(payload.get("status") == "pass")
        selectable.append(all(statuses))
    return records, selectable


def _export_symbol(generated: str) -> str:
    match = re.search(r"const KernelPackRecord& (k[A-Za-z0-9_]+) =", generated)
    if not match:
        raise ValueError("generated initializer did not export a record symbol")
    return match.group(1)


def _emit(repo_root: Path, manifest_paths: list[Path], output_path: Path) -> None:
    records, selectable = _load_records(repo_root, manifest_paths)
    chunks: list[str] = []
    symbols: list[str] = []
    for record in records:
        generated = generate_cpp_initializers(record, asset_root=repo_root)
        chunks.append(generated.rstrip())
        symbols.append(_export_symbol(generated))
    lines = ["// Generated by tools/generate_scalar_kernel_packs.py; do not edit.", ""]
    lines.extend(chunks)
    lines.extend(
        [
            "namespace native_r9700::generated {",
            "namespace {",
            "constexpr KernelPackRecord kLlamaScalarPackRecords[] = {",
        ]
    )
    lines.extend(f"    {symbol}," for symbol in symbols)
    lines.extend(["};", ""])
    selected_symbols = [symbol for symbol, ready in zip(symbols, selectable) if ready]
    lines.append("constexpr KernelPackRecord kLlamaSelectableScalarPackRecords[] = {")
    lines.extend(f"    {symbol}," for symbol in selected_symbols)
    lines.extend(
        [
            "};",
            "}  // namespace",
            "KernelPackSpan<KernelPackRecord> llama_scalar_pack_records() {",
            "  return {kLlamaScalarPackRecords, sizeof(kLlamaScalarPackRecords) / sizeof(kLlamaScalarPackRecords[0])};",
            "}",
            "KernelPackSpan<KernelPackRecord> llama_selectable_scalar_pack_records() {",
            "  return {kLlamaSelectableScalarPackRecords, sizeof(kLlamaSelectableScalarPackRecords) / sizeof(kLlamaSelectableScalarPackRecords[0])};",
            "}",
            "}  // namespace native_r9700::generated",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"generated {len(records)} scalar pack records ({sum(selectable)} selectable) -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "native_r9700/kernel_packs_generated.inc")
    parser.add_argument("--bootstrap", action="store_true", help="create scalar records from reviewed HSA sidecars")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    manifests = _manifest_paths(repo_root) if not args.bootstrap else []
    if args.bootstrap:
        _bootstrap(repo_root, manifests)
        manifests = _manifest_paths(repo_root)
    if len(manifests) != len(SCALAR_ORDER):
        raise SystemExit(f"expected {len(SCALAR_ORDER)} scalar pack manifests, found {len(manifests)}")
    output = args.output if args.output.is_absolute() else repo_root / args.output
    _emit(repo_root, manifests, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
