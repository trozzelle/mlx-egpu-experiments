"""Offline Kernel Pack manifest validation and C++ view generation.

This module is deliberately an owning, offline boundary.  It reads JSON and
asset/evidence files only when explicitly asked to validate a manifest; the
runtime consumes the generated C++ views and never imports this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class ManifestError(ValueError):
    """Raised for every malformed, unpinned, or unresolved manifest value."""


# The values are intentionally closed.  They are also useful to callers that
# need to construct or inspect the matrix without duplicating its policy.
EVIDENCE_FIELDS = frozenset(
    {
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
)
EVIDENCE_KINDS = frozenset(
    {"offline_oracle", "offline_review", "target_conformance", "native_run", "benchmark"}
)
EVIDENCE_SLOTS = frozenset(
    {
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
)
ALLOWED_EVIDENCE_PAIRS = frozenset(
    {
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
)

_SCHEMA_VERSION = 1
_TARGET = "gfx1201"
_PINNED_UPSTREAM = "https://github.com/llvm/llvm-project"
_PINNED_REVISION = "8dba93818258d95c46fa2c17e902a8256e4d91b5"
_PINNED_UPSTREAM_PATHS = ("llvm/docs/AMDGPUUsage.rst",)
_PACK_DOMAIN = "r9700-kernel-pack-identity-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_VERSION_RE = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:$@-]*\Z")
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_RFC8785_INTEGER_MAX = (1 << 53) - 1
_TENSOR_DTYPES = frozenset({"fp16", "bf16", "fp32", "int8", "int4"})
_FROZEN_INVERSE_N = [0, 0, 0, 15, 16, 16, 8191]
_FROZEN_INVERSE_K = [0, 1, 16, 15, 0, 2047, 2047]
_FROZEN_INVERSE_SOURCE_F16 = [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700]
_LAYOUT_SOURCE_VERSION = "f16-row-major-nk-source-v1"
_LAYOUT_PHYSICAL_VERSION = "f2-wmma-physical-tile-v1"
_LAYOUT_SPEC_PATH = "build/f2-wmma/f2-wmma-physical-layout-spec.json"
_LAYOUT_INVERSE_FIXTURE_PATH = "build/f2-wmma/f2-wmma-physical-layout-inverse.npz"
_LAYOUT_MAPPING = {
    "source_element": "source_weight[n*K+k]",
    "physical_byte_offset": (
        "((((n // 16) * 128 + (k // 16)) * 512) + "
        "(((k % 16) * 16 + (n % 16)) * 2))"
    ),
    "b_tile": "tile_n=n//16,tile_k=k//16,row=k%16,col=n%16",
    "lds_byte_offset": "((k % 16) * 16 + (n % 16)) * 2",
}
_LAYOUT_STRIDES = {
    "source_row_stride_elements": 2048,
    "physical_tile_stride_bytes": 512,
    "lds_tile_stride_bytes": 512,
    "tile_n_count": 512,
    "tile_k_count": 128,
}
_LAYOUT_ORIGINS = frozenset({"pinned_header", "reviewed_local_v1"})
_KERNARG_TYPES: dict[str, tuple[int, int]] = {
    "bool": (1, 1),
    "int8": (1, 1),
    "uint8": (1, 1),
    "int16": (2, 2),
    "uint16": (2, 2),
    "int32": (4, 4),
    "uint32": (4, 4),
    "float": (4, 4),
    "float32": (4, 4),
    "int64": (8, 8),
    "uint64": (8, 8),
    "double": (8, 8),
    "float64": (8, 8),
    "pointer": (8, 8),
    "ptr": (8, 8),
    "size_t": (8, 8),
}

_TOP_KEYS = {
    "schema_version",
    "name",
    "version",
    "target",
    "required_features",
    "provenance",
    "image",
    "entries",
    "compatibility",
    "numerics",
    "evidence",
}
_PROVENANCE_KEYS = {
    "upstream_repository",
    "upstream_revision",
    "upstream_paths",
    "local_sources",
    "license_reviews",
    "modifications",
}
_SOURCE_KEYS = {"path", "sha256"}
_LICENSE_KEYS = {"component", "spdx_expression", "review_id", "status"}
_MODIFICATION_KEYS = {"component", "summary"}
_IMAGE_KEYS = {"image_path", "image_sha256", "image_size", "code_object_version", "build"}
_BUILD_KEYS = {
    "toolchain_id",
    "toolchain_revision",
    "generator_id",
    "generator_revision",
    "command_sha256",
}
_ENTRY_KEYS = {"symbol", "descriptor_offset", "entry_offset", "kernargs", "resources", "geometry"}
_KERNARGS_KEYS = {"bytes", "fields", "tail_padding_bytes"}
_KERNARG_FIELD_KEYS = {"name", "type", "offset", "size", "alignment"}
_RESOURCE_KEYS = {
    "rsrc1",
    "rsrc2",
    "rsrc3",
    "wave_size",
    "sgpr_count",
    "vgpr_count",
    "lds_bytes",
    "private_segment_bytes",
    "metadata_provenance",
}
_GEOMETRY_KEYS = {"cases"}
_GEOMETRY_CASE_KEYS = {
    "shape_family",
    "geometry_rule",
    "workgroup_x",
    "workgroup_y",
    "workgroup_z",
    "global_x",
    "global_y",
    "global_z",
    "grid_tile_m",
    "grid_tile_n",
    "dynamic_lds_allowed",
    "dynamic_lds_max_bytes",
}
_COMPATIBILITY_KEYS = {
    "input_dtype",
    "weight_dtype",
    "output_dtype",
    "source_tensor_layout_version",
    "shape_families",
    "weight_packing_version",
}
_SHAPE_FAMILY_KEYS = {
    "name",
    "fixed_dimensions",
    "runtime_dimension",
    "tail_policy",
    "geometry_rule",
}
_DIMENSION_KEYS = {"name", "value"}
_RUNTIME_DIMENSION_KEYS = {"name", "min_value", "max_value", "full_value"}
_NUMERICS_KEYS = {
    "input_dtype",
    "accumulation_dtype",
    "output_dtype",
    "cast_points",
    "finite_value_rule",
    "tolerance_policy",
    "reference_set_kind",
    "retained_reference",
    "numpy_oracle",
    "scalar_native_projection",
}
_CAST_POINT_KEYS = {"stage", "from_dtype", "to_dtype"}
_EVIDENCE_KEYS = {
    "conformance",
    "native_run",
    "source_review",
    "resource_review",
    "isa_review",
    "layout_proof",
    "benchmark_record",
    "benchmark_not_applicable_reason",
}

_VALIDATED_PACK_ROOTS: dict[str, tuple[Path, Path | None]] = {}


def _fail(message: str) -> None:
    raise ManifestError(message)

def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _dict(value: Any, keys: set[str] | frozenset[str], where: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{where} must be an object")
    if set(value) != set(keys):
        missing = sorted(set(keys) - set(value))
        extra = sorted(set(value) - set(keys))
        _fail(f"{where} has wrong keys (missing={missing}, extra={extra})")
    return value


def _string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        _fail(f"{where} must be a {'non-empty ' if nonempty else ''}string")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        _fail(f"{where} contains a C0 control character")
    return value




def _boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        _fail(f"{where} must be a boolean")
    return value


def _list(value: Any, where: str, *, nonempty: bool = False) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        _fail(f"{where} must be a {'non-empty ' if nonempty else ''}list")
    return value


def _sha256(value: Any, where: str, *, required: bool = True) -> str:
    if type(value) is not str or (required and not _SHA256_RE.fullmatch(value)):
        _fail(f"{where} must be lowercase hexadecimal SHA-256")
    if not required and value and not _SHA256_RE.fullmatch(value):
        _fail(f"{where} must be empty or lowercase hexadecimal SHA-256")
    return value


def _integer(
    value: Any,
    where: str,
    *,
    minimum: int = 0,
    maximum: int | None = _UINT64_MAX,
) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bound = f" and <= {maximum}" if maximum is not None else ""
        _fail(f"{where} must be an integer >= {minimum}{bound}")
    return value


def _safe_relative(value: Any, where: str) -> str:
    path = _string(value, where)
    # Paths in a manifest are checkout-relative POSIX paths, even on hosts
    # where the offline validator happens to run with another path syntax.
    if "\\" in path or path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        _fail(f"{where} is not a safe relative path")
    normalized = PurePosixPath(path).as_posix()
    if path != normalized:
        _fail(f"{where} is not a canonical relative path")
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        _fail(f"{where} is not a safe relative path")
    return path


def _under_root(root: Path, relative: str, where: str) -> Path:
    try:
        root_resolved = root.resolve(strict=True)
        candidate = (root / relative).resolve(strict=False)
        candidate.relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError):
        _fail(f"{where} escapes the asset root")
    # A symlink is not an owned manifest asset.  Resolving above also catches
    # links that leave the root, while this check keeps the generated identity
    # tied to the named file rather than to a mutable alias.
    try:
        if (root / relative).is_symlink():
            _fail(f"{where} must not be a symlink")
    except OSError as exc:
        _fail(f"{where} cannot be inspected: {exc}")
    return candidate


def _file_digest(root: Path, relative: str, where: str) -> tuple[bytes, str]:
    path = _under_root(root, relative, where)
    try:
        if not path.is_file():
            _fail(f"{where} is not a regular file")
        data = path.read_bytes()
    except (OSError, RuntimeError) as exc:
        _fail(f"{where} cannot be read: {exc}")
    return data, hashlib.sha256(data).hexdigest()


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    _fail(f"non-finite JSON constant: {value}")


def load_manifest(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load one manifest with duplicate-key and non-finite-number rejection."""

    try:
        text = Path(path).read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except ManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _fail(f"manifest JSON is invalid: {exc}")
    if type(value) is not dict:
        _fail("manifest root must be an object")
    return value


def _without_evidence_binding_digests(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for key, child in value.items():
            if key in {"pack_sha256", "record_sha256"}:
                continue
            result[key] = _without_evidence_binding_digests(child)
        return result
    if isinstance(value, list):
        return [_without_evidence_binding_digests(child) for child in value]
    if isinstance(value, tuple):
        return [_without_evidence_binding_digests(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        _fail("pack identity contains a non-finite number")
    return value
def _validate_digest_integers(value: Any, where: str) -> None:
    """Reject integers that cannot round-trip through RFC8785/JCS."""
    if type(value) is int:
        if not 0 <= value <= _RFC8785_INTEGER_MAX:
            _fail(f"{where} contains an integer outside the RFC8785 safe range")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_digest_integers(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_digest_integers(child, f"{where}[{index}]")




def compute_pack_sha256(record: Mapping[str, Any]) -> str:
    """Compute the nonrecursive canonical identity digest for *record*."""

    if not isinstance(record, Mapping):
        _fail("pack record must be an object")
    normalized = _without_evidence_binding_digests(dict(record))
    if type(normalized) is not dict:
        _fail("pack record must be an object")
    normalized.pop("evidence", None)
    _validate_digest_integers(normalized, "pack identity")
    preimage = {"domain": _PACK_DOMAIN, "pack": normalized}
    try:
        canonical = json.dumps(
            preimage,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        _fail(f"pack identity is not canonical JSON: {exc}")
    return hashlib.sha256(canonical).hexdigest()


def _validate_policy(policy_path: os.PathLike[str] | str, provenance: dict[str, Any]) -> None:
    """Cross-check the declared source against the pinned P3 policy block.

    The policy document is YAML, but this boundary needs only a tiny, strict
    line-oriented read of one already-pinned source block.  It intentionally
    does not import a YAML implementation or expose YAML parsing to runtime.
    """

    try:
        text = Path(policy_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        _fail(f"policy input cannot be read: {exc}")
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.fullmatch(r"\s*- id: llvm-amdgpu-usage\s*", line):
            start = index
            break
    if start is None:
        _fail("policy does not contain the pinned llvm-amdgpu-usage source")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.fullmatch(r"\s*- id: \S+\s*", lines[index]):
            end = index
            break
    block = lines[start:end]
    upstream = None
    revision = None
    paths: list[str] = []
    in_paths = False
    for line in block[1:]:
        stripped = line.strip()
        if stripped.startswith("upstream:"):
            upstream = stripped.split(":", 1)[1].strip()
            in_paths = False
        elif stripped.startswith("revision:"):
            revision = stripped.split(":", 1)[1].strip()
            in_paths = False
        elif stripped == "paths:":
            in_paths = True
        elif in_paths and re.match(r"^\s+-\s+\S", line):
            paths.append(line.split("-", 1)[1].strip())
        elif in_paths:
            in_paths = False
    if upstream is None or revision is None or not paths:
        _fail("policy source block is incomplete")
    if provenance["upstream_repository"] != upstream:
        _fail("upstream repository does not match policy")
    if provenance["upstream_revision"] != revision:
        _fail("upstream revision does not match policy")
    if provenance["upstream_paths"] != paths:
        _fail("upstream paths do not match policy")
def _canonical_record_sha256(record: dict[str, Any]) -> str:
    """Hash an evidence record after removing only its self-digest field."""
    if type(record) is not dict:
        _fail("evidence record must be an object")
    normalized = {key: value for key, value in record.items() if key != "record_sha256"}
    _validate_digest_integers(normalized, "evidence record")
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        _fail(f"evidence record is not canonical JSON: {exc}")
    return hashlib.sha256(encoded).hexdigest()


def _validate_provenance(value: Any, root: Path, policy_path: Any) -> dict[str, Any]:
    provenance = _dict(value, _PROVENANCE_KEYS, "provenance")
    repository = _string(provenance["upstream_repository"], "provenance.upstream_repository")
    revision = _string(provenance["upstream_revision"], "provenance.upstream_revision")
    upstream_paths = _list(provenance["upstream_paths"], "provenance.upstream_paths")
    local_sources = _list(provenance["local_sources"], "provenance.local_sources", nonempty=True)
    license_reviews = _list(
        provenance["license_reviews"], "provenance.license_reviews", nonempty=True
    )
    modifications = _list(provenance["modifications"], "provenance.modifications")

    if repository == "local":
        if revision != "local":
            _fail("local provenance must use upstream_revision=local")
    else:
        if repository != _PINNED_UPSTREAM or not repository.startswith("https://github.com/"):
            _fail("upstream repository is not an admitted pinned source")
        if revision != _PINNED_REVISION:
            _fail("upstream revision is not the pinned P3 revision")
        if not _REVISION_RE.fullmatch(revision):
            _fail("upstream revision must be an immutable full revision")
        if tuple(upstream_paths) != _PINNED_UPSTREAM_PATHS:
            _fail("upstream paths do not name the admitted P3 source")
    for index, path in enumerate(upstream_paths):
        _safe_relative(path, f"provenance.upstream_paths[{index}]")
    if len(set(upstream_paths)) != len(upstream_paths):
        _fail("provenance.upstream_paths contains duplicates")

    local_paths: list[str] = []
    for index, source_value in enumerate(local_sources):
        source = _dict(source_value, _SOURCE_KEYS, f"provenance.local_sources[{index}]")
        source_path = _safe_relative(source["path"], f"provenance.local_sources[{index}].path")
        _sha256(source["sha256"], f"provenance.local_sources[{index}].sha256")
        if source_path in local_paths:
            _fail("provenance.local_sources contains duplicate paths")
        local_paths.append(source_path)
        data, digest = _file_digest(root, source_path, f"provenance.local_sources[{index}].path")
        del data
        if digest != source["sha256"]:
            _fail(f"source digest mismatch for {source_path}")

    modification_paths: list[str] = []
    for index, modification_value in enumerate(modifications):
        modification = _dict(
            modification_value,
            _MODIFICATION_KEYS,
            f"provenance.modifications[{index}]",
        )
        component = _safe_relative(
            modification["component"], f"provenance.modifications[{index}].component"
        )
        _string(modification["summary"], f"provenance.modifications[{index}].summary")
        if component in modification_paths:
            _fail("provenance.modifications contains duplicate components")
        modification_paths.append(component)

    required_components = set(upstream_paths) | set(local_paths) | set(modification_paths)
    # The image component is checked after image shape validation below; the
    # caller adds it to the coverage set before returning.
    reviewed_components: set[str] = set()
    for index, license_value in enumerate(license_reviews):
        review = _dict(license_value, _LICENSE_KEYS, f"provenance.license_reviews[{index}]")
        component = _safe_relative(
            review["component"], f"provenance.license_reviews[{index}].component"
        )
        expression = _string(
            review["spdx_expression"],
            f"provenance.license_reviews[{index}].spdx_expression",
        )
        review_id = _string(review["review_id"], f"provenance.license_reviews[{index}].review_id")
        if review["status"] != "accepted":
            _fail(f"license review for {component} is not accepted")
        if expression.strip().lower() in {"unknown", "pending"}:
            _fail(f"license expression for {component} is not reviewed")
        if component in reviewed_components:
            _fail("provenance.license_reviews contains duplicate components")
        reviewed_components.add(component)
        if not review_id.strip():
            _fail(f"license review ID is empty for {component}")
    if not required_components.issubset(reviewed_components):
        _fail("every source and modified component needs a license review")

    if policy_path is not None:
        _validate_policy(policy_path, provenance)
    return provenance


def _validate_build(value: Any) -> dict[str, Any]:
    build = _dict(value, _BUILD_KEYS, "image.build")
    _string(build["toolchain_id"], "image.build.toolchain_id")
    if not _REVISION_RE.fullmatch(_string(build["toolchain_revision"], "image.build.toolchain_revision")):
        _fail("image.build.toolchain_revision must be an immutable revision")
    _string(build["generator_id"], "image.build.generator_id")
    if not _REVISION_RE.fullmatch(_string(build["generator_revision"], "image.build.generator_revision")):
        _fail("image.build.generator_revision must be an immutable revision")
    _sha256(build["command_sha256"], "image.build.command_sha256")
    return build


def _validate_image(value: Any, root: Path) -> dict[str, Any]:
    image = _dict(value, _IMAGE_KEYS, "image")
    image_path = _safe_relative(image["image_path"], "image.image_path")
    image_sha = _sha256(image["image_sha256"], "image.image_sha256")
    image_size = _integer(image["image_size"], "image.image_size", minimum=1)
    code_object_version = _string(image["code_object_version"], "image.code_object_version")
    if not re.fullmatch(r"amdhsa-v[0-9]+", code_object_version):
        _fail("image.code_object_version is not an admitted AMDHSA version")
    _validate_build(image["build"])
    data, digest = _file_digest(root, image_path, "image.image_path")
    if digest != image_sha:
        _fail("image digest does not match image bytes")
    if len(data) != image_size:
        _fail("image_size does not match image bytes")
    return image


def _validate_kernargs(value: Any, where: str) -> dict[str, Any]:
    kernargs = _dict(value, _KERNARGS_KEYS, where)
    byte_count = _integer(kernargs["bytes"], f"{where}.bytes", minimum=1, maximum=_UINT32_MAX)
    fields = _list(kernargs["fields"], f"{where}.fields", nonempty=True)
    tail = _integer(kernargs["tail_padding_bytes"], f"{where}.tail_padding_bytes", maximum=_UINT32_MAX)
    previous_end = 0
    field_names: set[str] = set()
    for index, field_value in enumerate(fields):
        field_where = f"{where}.fields[{index}]"
        field = _dict(field_value, _KERNARG_FIELD_KEYS, field_where)
        name = _string(field["name"], f"{field_where}.name")
        field_type = _string(field["type"], f"{field_where}.type")
        if name in field_names:
            _fail(f"{where} has duplicate field names")
        field_names.add(name)
        if field_type not in _KERNARG_TYPES:
            _fail(f"{field_where}.type is not a canonical kernarg type")
        offset = _integer(field["offset"], f"{field_where}.offset", maximum=_UINT32_MAX)
        size = _integer(field["size"], f"{field_where}.size", minimum=1, maximum=_UINT32_MAX)
        alignment = _integer(field["alignment"], f"{field_where}.alignment", minimum=1, maximum=_UINT32_MAX)
        expected_size, expected_alignment = _KERNARG_TYPES[field_type]
        if size != expected_size or alignment != expected_alignment:
            _fail(f"{field_where} does not match its declared type")
        if alignment & (alignment - 1):
            _fail(f"{field_where}.alignment must be a power of two")
        if index and offset <= previous_end - 1:
            _fail(f"{field_where}.offset is not strictly increasing")
        if offset % alignment:
            _fail(f"{field_where}.offset is not aligned")
        if index == 0 and offset != 0:
            _fail(f"{field_where}.offset leaves unrecorded leading padding")
        if offset + size > byte_count:
            _fail(f"{field_where} exceeds kernarg segment bytes")
        if index and offset != previous_end:
            _fail(f"{field_where}.offset leaves unrecorded interior padding")
        previous_end = offset + size
    if tail != byte_count - previous_end:
        _fail(f"{where}.tail_padding_bytes does not close the segment")
    return kernargs


def _validate_resources(value: Any, where: str, required_features: list[Any]) -> dict[str, Any]:
    resources = _dict(value, _RESOURCE_KEYS, where)
    for key in ("rsrc1", "rsrc2", "rsrc3", "sgpr_count", "vgpr_count"):
        _integer(resources[key], f"{where}.{key}", minimum=1, maximum=_UINT32_MAX)
    _integer(resources["wave_size"], f"{where}.wave_size", maximum=_UINT32_MAX)
    for key in ("lds_bytes", "private_segment_bytes"):
        _integer(resources[key], f"{where}.{key}", maximum=_UINT64_MAX)
    if resources["wave_size"] != 32:
        _fail(f"{where}.wave_size is inconsistent with the gfx1201 target")
    provenance = _string(resources["metadata_provenance"], f"{where}.metadata_provenance")
    expected_provenance = f"source AMDGPU metadata: {_PINNED_UPSTREAM_PATHS[0]}"
    if provenance != expected_provenance:
        _fail(f"{where}.metadata_provenance must exactly cite {_PINNED_UPSTREAM_PATHS[0]}")
    return resources


def _validate_geometry_case(value: Any, where: str) -> dict[str, Any]:
    case = _dict(value, _GEOMETRY_CASE_KEYS, where)
    shape_family = _string(case["shape_family"], f"{where}.shape_family")
    if not _IDENTIFIER_RE.fullmatch(shape_family):
        _fail(f"{where}.shape_family is not a stable family name")
    rule = _string(case["geometry_rule"], f"{where}.geometry_rule")
    if rule not in {"exact-global-v1", "f2-wmma-64x64-m-tail-v1"}:
        _fail(f"{where}.geometry_rule is not a closed v1 rule")
    for key in (
        "workgroup_x",
        "workgroup_y",
        "workgroup_z",
        "global_x",
        "global_y",
        "global_z",
        "grid_tile_m",
        "grid_tile_n",
    ):
        _integer(case[key], f"{where}.{key}", maximum=_UINT32_MAX)
    _integer(case["dynamic_lds_max_bytes"], f"{where}.dynamic_lds_max_bytes", maximum=_UINT64_MAX)
    dynamic = _boolean(case["dynamic_lds_allowed"], f"{where}.dynamic_lds_allowed")
    if not dynamic and case["dynamic_lds_max_bytes"] != 0:
        _fail(f"{where}.dynamic_lds_max_bytes requires dynamic LDS")
    if dynamic and case["dynamic_lds_max_bytes"] == 0:
        _fail(f"{where}.dynamic_lds_allowed requires a positive limit")
    if min(case["workgroup_x"], case["workgroup_y"], case["workgroup_z"]) == 0:
        _fail(f"{where} has a zero workgroup dimension")
    if rule == "exact-global-v1":
        if min(case["global_x"], case["global_y"], case["global_z"]) == 0:
            _fail(f"{where} has a zero exact global dimension")
        if case["grid_tile_m"] or case["grid_tile_n"]:
            _fail(f"{where} exact-global-v1 cannot carry tile dimensions")
        if any(
            case[global_key] % case[workgroup_key]
            for global_key, workgroup_key in (
                ("global_x", "workgroup_x"),
                ("global_y", "workgroup_y"),
                ("global_z", "workgroup_z"),
            )
        ):
            _fail(f"{where} global dimensions are not workgroup aligned")
    else:
        if (case["workgroup_x"], case["workgroup_y"], case["workgroup_z"]) != (128, 4, 1):
            _fail(f"{where} has the wrong F2 WMMA workgroup")
        if (case["grid_tile_m"], case["grid_tile_n"]) != (64, 64):
            _fail(f"{where} has the wrong F2 WMMA tile")
        if case["global_x"] or case["global_y"] or case["global_z"]:
            _fail(f"{where} F2 WMMA global dimensions must be computed at dispatch")
    return case


def _validate_entries(value: Any, image_size: int, required_features: list[Any]) -> list[dict[str, Any]]:
    entries = _list(value, "entries", nonempty=True)
    if len(entries) != 1:
        _fail("Kernel Pack schema v1 requires exactly one entry")
    symbols: set[str] = set()
    for index, entry_value in enumerate(entries):
        where = f"entries[{index}]"
        entry = _dict(entry_value, _ENTRY_KEYS, where)
        symbol = _string(entry["symbol"], f"{where}.symbol")
        if not _IDENTIFIER_RE.fullmatch(symbol):
            _fail(f"{where}.symbol is not a valid code-object symbol")
        if symbol in symbols:
            _fail("entries contain duplicate symbols")
        symbols.add(symbol)
        for key in ("descriptor_offset", "entry_offset"):
            offset = _integer(entry[key], f"{where}.{key}", maximum=_UINT64_MAX)
            if offset >= image_size:
                _fail(f"{where}.{key} lies outside the image")
        _validate_kernargs(entry["kernargs"], f"{where}.kernargs")
        _validate_resources(entry["resources"], f"{where}.resources", required_features)
        geometry = _dict(entry["geometry"], _GEOMETRY_KEYS, f"{where}.geometry")
        cases = _list(geometry["cases"], f"{where}.geometry.cases", nonempty=True)
        case_names: set[str] = set()
        for case_index, case_value in enumerate(cases):
            case = _validate_geometry_case(case_value, f"{where}.geometry.cases[{case_index}]")
            if case["shape_family"] in case_names:
                _fail(f"{where}.geometry has duplicate shape families")
            case_names.add(case["shape_family"])
    return entries


def _validate_shape_family(value: Any, where: str) -> dict[str, Any]:
    family = _dict(value, _SHAPE_FAMILY_KEYS, where)
    name = _string(family["name"], f"{where}.name")
    if not _IDENTIFIER_RE.fullmatch(name):
        _fail(f"{where}.name is not a stable shape-family name")
    fixed = _list(family["fixed_dimensions"], f"{where}.fixed_dimensions", nonempty=True)
    names: set[str] = set()
    for index, dimension_value in enumerate(fixed):
        dimension = _dict(dimension_value, _DIMENSION_KEYS, f"{where}.fixed_dimensions[{index}]")
        dimension_name = _string(dimension["name"], f"{where}.fixed_dimensions[{index}].name")
        if dimension_name in names:
            _fail(f"{where}.fixed_dimensions contains duplicate dimensions")
        names.add(dimension_name)
        _integer(dimension["value"], f"{where}.fixed_dimensions[{index}].value", minimum=1, maximum=_UINT32_MAX)
    runtime = family["runtime_dimension"]
    if runtime is not None:
        runtime = _dict(runtime, _RUNTIME_DIMENSION_KEYS, f"{where}.runtime_dimension")
        runtime_name = _string(runtime["name"], f"{where}.runtime_dimension.name")
        if runtime_name in names:
            _fail(f"{where}.runtime_dimension duplicates a fixed dimension")
        minimum = _integer(runtime["min_value"], f"{where}.runtime_dimension.min_value", minimum=1, maximum=_UINT32_MAX)
        maximum = _integer(runtime["max_value"], f"{where}.runtime_dimension.max_value", minimum=1, maximum=_UINT32_MAX)
        full = _integer(runtime["full_value"], f"{where}.runtime_dimension.full_value", minimum=1, maximum=_UINT32_MAX)
        if not minimum <= full <= maximum:
            _fail(f"{where}.runtime_dimension bounds are inconsistent")
    tail_policy = _string(family["tail_policy"], f"{where}.tail_policy")
    rule = _string(family["geometry_rule"], f"{where}.geometry_rule")
    if runtime is None:
        if tail_policy != "none" or rule != "exact-global-v1":
            _fail(f"{where} fixed families require exact-global-v1 and no tails")
    else:
        if tail_policy != "masked/padded" or rule != "f2-wmma-64x64-m-tail-v1":
            _fail(f"{where} bounded families require the F2 masked tail rule")
    return family


def _validate_compatibility(value: Any) -> dict[str, Any]:
    compatibility = _dict(value, _COMPATIBILITY_KEYS, "compatibility")
    for key in ("input_dtype", "weight_dtype", "output_dtype"):
        dtype = _string(compatibility[key], f"compatibility.{key}")
        if dtype not in _TENSOR_DTYPES:
            _fail(f"compatibility.{key} is an unknown dtype")
    _string(
        compatibility["source_tensor_layout_version"],
        "compatibility.source_tensor_layout_version",
    )
    _string(compatibility["weight_packing_version"], "compatibility.weight_packing_version")
    families = _list(compatibility["shape_families"], "compatibility.shape_families", nonempty=True)
    names: set[str] = set()
    for index, family_value in enumerate(families):
        family = _validate_shape_family(family_value, f"compatibility.shape_families[{index}]")
        if family["name"] in names:
            _fail("compatibility.shape_families contains duplicate names")
        names.add(family["name"])
    return compatibility


def _validate_cast_points(value: Any, numerics: dict[str, Any]) -> None:
    points = _list(value, "numerics.cast_points", nonempty=True)
    stages: set[str] = set()
    for index, point_value in enumerate(points):
        where = f"numerics.cast_points[{index}]"
        point = _dict(point_value, _CAST_POINT_KEYS, where)
        stage = _string(point["stage"], f"{where}.stage")
        if stage in stages:
            _fail("numerics.cast_points contains duplicate stages")
        stages.add(stage)
        from_dtype = _string(point["from_dtype"], f"{where}.from_dtype")
        to_dtype = _string(point["to_dtype"], f"{where}.to_dtype")
        if from_dtype not in _TENSOR_DTYPES or to_dtype not in _TENSOR_DTYPES:
            _fail(f"{where} uses an unknown dtype")
        if to_dtype != numerics["accumulation_dtype"]:
            _fail(f"{where}.to_dtype does not match accumulation_dtype")


def _validate_evidence_file(ref: dict[str, Any], root: Path) -> dict[str, Any]:
    path = _safe_relative(ref["record_path"], "evidence.record_path")
    data, digest = _file_digest(root, path, "evidence.record_path")
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (ManifestError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _fail(f"evidence record is not strict JSON: {exc}")
    if type(payload) is not dict:
        _fail(f"evidence record {path} must be an object")
    _validate_digest_integers(payload, f"evidence record {path}")

    if ref["record_kind"] == "offline_review" and ref["evidence_slot"] == "layout_proof":
        declared_digest = _sha256(payload.get("record_sha256"), f"evidence record {path} digest")
        if declared_digest != ref["record_sha256"] or _canonical_record_sha256(payload) != declared_digest:
            _fail(f"evidence record canonical digest mismatch for {path}")
    elif digest != ref["record_sha256"]:
        _fail(f"evidence record digest mismatch for {path}")

    identity_fields = (
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
    for key in identity_fields:
        if key not in payload:
            _fail(f"evidence record {path} is missing {key}")
        if payload[key] != ref[key]:
            _fail(f"evidence record {path} does not match {key}")
    return payload


def validate_evidence_ref(
    ref: Any,
    *,
    subject_target: str,
    image_sha256: str,
    pack_sha256: str,
) -> dict[str, Any]:
    """Validate one exact row of the closed EvidenceRef matrix."""

    reference = _dict(ref, EVIDENCE_FIELDS, "EvidenceRef")
    _string(reference["record_id"], "EvidenceRef.record_id")
    for key in (
        "record_path",
        "record_kind",
        "evidence_slot",
        "subject_target",
        "image_sha256",
        "pack_sha256",
        "producer_kind",
        "tool_digest",
        "input_digest",
        "output_digest",
    ):
        _string(reference[key], f"EvidenceRef.{key}", nonempty=False)
    _safe_relative(reference["record_path"], "EvidenceRef.record_path")
    _sha256(reference["record_sha256"], "EvidenceRef.record_sha256")
    _sha256(reference["input_digest"], "EvidenceRef.input_digest")
    _sha256(reference["output_digest"], "EvidenceRef.output_digest")
    kind = reference["record_kind"]
    slot = reference["evidence_slot"]
    if kind not in EVIDENCE_KINDS or slot not in EVIDENCE_SLOTS:
        _fail("EvidenceRef has an unknown kind or slot")
    if (kind, slot) not in ALLOWED_EVIDENCE_PAIRS:
        _fail("EvidenceRef kind/slot pair is not admitted")
    _string(subject_target, "EvidenceRef expected subject_target")
    _sha256(image_sha256, "EvidenceRef expected image_sha256", required=False)
    _sha256(pack_sha256, "EvidenceRef expected pack_sha256", required=False)

    if kind == "offline_oracle":
        if reference["producer_kind"] != "cpu_reference":
            _fail("offline_oracle must use cpu_reference")
        if any(reference[key] for key in ("subject_target", "image_sha256", "pack_sha256", "tool_digest")):
            _fail("offline_oracle has nonempty bound fields")
    elif kind == "offline_review":
        if reference["producer_kind"]:
            _fail("offline_review producer_kind must be empty")
        if (
            reference["subject_target"] != subject_target
            or reference["image_sha256"] != image_sha256
            or reference["pack_sha256"] != pack_sha256
            or not reference["tool_digest"]
        ):
            _fail("offline_review binding fields are incomplete or contradictory")
        _sha256(reference["tool_digest"], "EvidenceRef.tool_digest")
    elif kind in {"target_conformance", "native_run"}:
        if reference["producer_kind"] != "r9700_native" or reference["tool_digest"]:
            _fail("native evidence has incorrect producer/tool fields")
        if (
            reference["subject_target"] != subject_target
            or reference["image_sha256"] != image_sha256
            or reference["pack_sha256"] != pack_sha256
        ):
            _fail("native evidence binding fields are contradictory")
    else:  # benchmark
        if reference["producer_kind"] != "r9700_native":
            _fail("benchmark evidence must use r9700_native")
        if (
            reference["subject_target"] != subject_target
            or reference["image_sha256"] != image_sha256
            or reference["pack_sha256"] != pack_sha256
            or not reference["tool_digest"]
        ):
            _fail("benchmark binding fields are incomplete or contradictory")
        _sha256(reference["tool_digest"], "EvidenceRef.tool_digest")
    return reference


def _validate_numerics(value: Any, compatibility: dict[str, Any]) -> dict[str, Any]:
    numerics = _dict(value, _NUMERICS_KEYS, "numerics")
    for key in ("input_dtype", "accumulation_dtype", "output_dtype"):
        dtype = _string(numerics[key], f"numerics.{key}")
        if dtype not in _TENSOR_DTYPES:
            _fail(f"numerics.{key} is an unknown dtype")
    if (
        numerics["input_dtype"] != compatibility["input_dtype"]
        or numerics["output_dtype"] != compatibility["output_dtype"]
    ):
        _fail("numerical input/output dtypes disagree with compatibility")
    _validate_cast_points(numerics["cast_points"], numerics)
    if _string(numerics["finite_value_rule"], "numerics.finite_value_rule") != "finite-input-output-v1":
        _fail("numerics.finite_value_rule is not the closed finite-value rule")
    _string(numerics["tolerance_policy"], "numerics.tolerance_policy")
    reference_set = _string(numerics["reference_set_kind"], "numerics.reference_set_kind")
    if reference_set not in {"b0_scalar_control", "f2_wmma_dual"}:
        _fail("numerics.reference_set_kind is unknown")
    return numerics


def _validate_report_payload(
    payload: dict[str, Any],
    ref: dict[str, Any],
    resources: dict[str, Any] | None,
    compatibility: dict[str, Any] | None = None,
) -> None:
    if ref["evidence_slot"] == "resource_review":
        if resources is None:
            _fail("resource evidence lacks manifest resource context")
        fields = (
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
        for field in fields:
            if field not in payload:
                _fail(f"resource evidence is missing {field}")
            if field == "metadata_provenance":
                _require(
                    _string(payload[field], f"resource evidence {field}") == resources[field],
                    f"resource evidence {field} disagrees with manifest",
                )
            else:
                _require(
                    _integer(payload[field], f"resource evidence {field}") == _integer(
                        resources[field], f"manifest resource {field}"
                    ),
                    f"resource evidence {field} disagrees with manifest",
                )
    elif ref["evidence_slot"] == "isa_review":
        categories = payload.get("isa_categories")
        unsupported = payload.get("unsupported_instructions")
        if type(categories) is not list or not categories or not all(
            type(item) is str and item.strip() for item in categories
        ):
            _fail("ISA evidence categories are malformed")
        if type(unsupported) is not list or not all(type(item) is str for item in unsupported):
            _fail("ISA evidence unsupported-instruction list is malformed")
        if "unsupported" in {item.strip().lower() for item in categories} or unsupported:
            _fail("ISA evidence contains unsupported instructions")
    elif ref["evidence_slot"] == "layout_proof":
        layout_fields = (
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
        for field in layout_fields:
            if field not in payload:
                _fail(f"layout proof is missing {field}")
        for field in (
            "layout_spec_sha256",
            "inverse_fixture_sha256",
            "inverse_fixture_input_digest",
            "inverse_fixture_output_digest",
        ):
            _sha256(payload[field], f"layout proof {field}")
        if compatibility is None:
            _fail("layout proof lacks manifest compatibility context")
        if (
            compatibility["source_tensor_layout_version"] != _LAYOUT_SOURCE_VERSION
            or compatibility["weight_packing_version"] != _LAYOUT_PHYSICAL_VERSION
        ):
            _fail("layout proof manifest versions are not frozen")
        for field, expected in (
            ("source_tensor_layout_version", compatibility["source_tensor_layout_version"]),
            ("physical_layout_version", compatibility["weight_packing_version"]),
            ("layout_spec_path", _LAYOUT_SPEC_PATH),
            ("layout_spec_sha256", ref["tool_digest"]),
            ("inverse_fixture_path", _LAYOUT_INVERSE_FIXTURE_PATH),
            ("inverse_fixture_sha256", ref["input_digest"]),
            ("inverse_fixture_input_digest", ref["input_digest"]),
            ("inverse_fixture_output_digest", ref["output_digest"]),
            ("layout_mapping", _LAYOUT_MAPPING),
            ("layout_strides", _LAYOUT_STRIDES),
            ("alignment_bytes", 16),
            ("padding_bytes", 0),
            ("swizzle", "none"),
            ("inverse_n", _FROZEN_INVERSE_N),
            ("inverse_k", _FROZEN_INVERSE_K),
            ("inverse_source_f16", _FROZEN_INVERSE_SOURCE_F16),
            ("layout_status", "pass"),
            ("failure_stage", "none"),
            ("exit_status", 0),
            ("wrapper_exit_status", 0),
        ):
            _require(payload[field] == expected, f"layout proof {field} is incomplete or incorrect")
        _require(payload["layout_origin"] in _LAYOUT_ORIGINS, "layout proof origin is not reviewed")


def _validate_ref_field(
    value: Any,
    where: str,
    *,
    root: Path,
    target: str,
    image_sha: str,
    pack_sha: str,
    resources: dict[str, Any] | None = None,
    compatibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if value is None:
        _fail(f"{where} is required")
    ref = validate_evidence_ref(
        value,
        subject_target=target,
        image_sha256=image_sha,
        pack_sha256=pack_sha,
    )
    payload = _validate_evidence_file(ref, root)
    _validate_report_payload(payload, ref, resources, compatibility)
    return ref


def _validate_reference_sets(
    numerics: dict[str, Any],
    evidence: dict[str, Any],
    *,
    root: Path,
    target: str,
    image_sha: str,
    pack_sha: str,
) -> None:
    reference_set = numerics["reference_set_kind"]
    if reference_set == "b0_scalar_control":
        retained = _validate_ref_field(
            numerics["retained_reference"],
            "numerics.retained_reference",
            root=root,
            target=target,
            image_sha=image_sha,
            pack_sha=pack_sha,
        )
        if retained["record_kind"] != "offline_oracle" or retained["evidence_slot"] != "numpy_oracle":
            _fail("B0 retained reference is not an offline NumPy oracle")
        if numerics["numpy_oracle"] is not None or numerics["scalar_native_projection"] is not None:
            _fail("B0 cannot carry F2 dual references")
        if evidence["layout_proof"] is not None:
            _fail("source-equivalent B0 cannot carry a layout proof")
    else:
        if numerics["retained_reference"] is not None:
            _fail("F2 dual reference set cannot retain a scalar oracle field")
        numpy_ref = _validate_ref_field(
            numerics["numpy_oracle"],
            "numerics.numpy_oracle",
            root=root,
            target=target,
            image_sha=image_sha,
            pack_sha=pack_sha,
        )
        native_ref = _validate_ref_field(
            numerics["scalar_native_projection"],
            "numerics.scalar_native_projection",
            root=root,
            target=target,
            image_sha=image_sha,
            pack_sha=pack_sha,
        )
        if numpy_ref["record_kind"] != "offline_oracle" or numpy_ref["evidence_slot"] != "numpy_oracle":
            _fail("F2 NumPy reference is not an offline oracle")
        if (
            native_ref["record_kind"] != "target_conformance"
            or native_ref["evidence_slot"] != "scalar_native_projection"
        ):
            _fail("F2 native projection is not target conformance")
        if numpy_ref["input_digest"] != native_ref["input_digest"]:
            _fail("F2 NumPy/native references do not share the request input digest")
        if numpy_ref["output_digest"] == native_ref["output_digest"]:
            _fail("F2 NumPy/native references must retain separate output digests")


def _validate_evidence(
    value: Any,
    numerics: dict[str, Any],
    compatibility: dict[str, Any],
    *,
    root: Path,
    target: str,
    image_sha: str,
    pack_sha: str,
    resources: dict[str, Any],
) -> dict[str, Any]:
    evidence = _dict(value, _EVIDENCE_KEYS, "evidence")
    for key in ("conformance", "source_review", "native_run", "resource_review", "isa_review"):
        ref = _validate_ref_field(
            evidence[key],
            f"evidence.{key}",
            root=root,
            target=target,
            image_sha=image_sha,
            pack_sha=pack_sha,
            resources=resources,
            compatibility=compatibility,
        )
        expected = {
            "conformance": ("target_conformance", "conformance"),
            "source_review": ("offline_review", "source_review"),
            "native_run": ("native_run", "native_run"),
            "resource_review": ("offline_review", "resource_review"),
            "isa_review": ("offline_review", "isa_review"),
        }[key]
        if (ref["record_kind"], ref["evidence_slot"]) != expected:
            _fail(f"evidence.{key} has the wrong evidence identity")
    layout = evidence["layout_proof"]
    packing = compatibility["weight_packing_version"]
    if packing == "source-equivalent-v1":
        if layout is not None:
            _fail("source-equivalent packing cannot carry layout proof")
    else:
        if layout is None:
            _fail("distinct physical packing requires layout proof")
        layout_ref = _validate_ref_field(
            layout,
            "evidence.layout_proof",
            root=root,
            target=target,
            image_sha=image_sha,
            pack_sha=pack_sha,
            compatibility=compatibility,
        )
        if (layout_ref["record_kind"], layout_ref["evidence_slot"]) != (
            "offline_review",
            "layout_proof",
        ):
            _fail("layout proof has the wrong evidence identity")
    benchmark = evidence["benchmark_record"]
    reason = _string(
        evidence["benchmark_not_applicable_reason"],
        "evidence.benchmark_not_applicable_reason",
        nonempty=False,
    )
    if benchmark is None:
        if not reason.strip():
            _fail("correctness-control packs require a benchmark exclusion reason")
    else:
        if reason:
            _fail("promoted benchmark packs cannot carry an exclusion reason")
        benchmark_ref = _validate_ref_field(
            benchmark,
            "evidence.benchmark_record",
            root=root,
            target=target,
            image_sha=image_sha,
            pack_sha=pack_sha,
            compatibility=compatibility,
        )
        if (benchmark_ref["record_kind"], benchmark_ref["evidence_slot"]) != (
            "benchmark",
            "benchmark",
        ):
            _fail("benchmark_record has the wrong evidence identity")
    _validate_reference_sets(
        numerics,
        evidence,
        root=root,
        target=target,
        image_sha=image_sha,
        pack_sha=pack_sha,
    )
    return evidence


def _validate_f2_contract(
    compatibility: dict[str, Any],
    numerics: dict[str, Any],
    entries: list[dict[str, Any]],
) -> None:
    families = compatibility["shape_families"]
    f2_families = [
        family
        for family in families
        if family["geometry_rule"] == "f2-wmma-64x64-m-tail-v1"
    ]
    if not f2_families:
        return
    if len(f2_families) != 1:
        _fail("there must be exactly one F2 WMMA family")
    family = f2_families[0]
    if family["name"] != "f2-linear-gate-up-f16-v1":
        _fail("F2 WMMA family name is not frozen")
    if (
        compatibility["input_dtype"] != "fp16"
        or compatibility["weight_dtype"] != "fp16"
        or compatibility["output_dtype"] != "fp16"
    ):
        _fail("F2 WMMA dtypes are not frozen fp16")
    if compatibility["source_tensor_layout_version"] != "f16-row-major-nk-source-v1":
        _fail("F2 source tensor layout is not frozen")
    if compatibility["weight_packing_version"] != "f2-wmma-physical-tile-v1":
        _fail("F2 physical packing is not frozen")
    if family["fixed_dimensions"] != [
        {"name": "K", "value": 2048},
        {"name": "N", "value": 8192},
    ]:
        _fail("F2 fixed dimensions are not frozen")
    if family["runtime_dimension"] != {
        "name": "M",
        "min_value": 1,
        "max_value": 128,
        "full_value": 128,
    }:
        _fail("F2 runtime dimension is not frozen")
    if (
        numerics["input_dtype"] != "fp16"
        or numerics["accumulation_dtype"] != "fp32"
        or numerics["output_dtype"] != "fp16"
        or numerics["cast_points"]
        != [
            {"stage": "wmma-accumulate", "from_dtype": "fp16", "to_dtype": "fp32"}
        ]
        or numerics["finite_value_rule"] != "finite-input-output-v1"
        or numerics["tolerance_policy"] != "F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1"
        or numerics["reference_set_kind"] != "f2_wmma_dual"
    ):
        _fail("F2 numerical policy is not frozen")
    for entry in entries:
        if entry["symbol"] != "linear_wmma_f16":
            _fail("F2 entry symbol is not frozen")
        cases = entry["geometry"]["cases"]
        matching = [case for case in cases if case["shape_family"] == family["name"]]
        if len(matching) != 1:
            _fail("F2 entry is missing its one WMMA geometry case")
        case = matching[0]
        if (
            case["geometry_rule"] != "f2-wmma-64x64-m-tail-v1"
            or (case["workgroup_x"], case["workgroup_y"], case["workgroup_z"])
            != (128, 4, 1)
            or (case["grid_tile_m"], case["grid_tile_n"]) != (64, 64)
            or any(case[key] for key in ("global_x", "global_y", "global_z"))
        ):
            _fail("F2 entry geometry is not frozen")
        if entry["kernargs"] != {
            "bytes": 32,
            "fields": [
                {"name": "a", "type": "uint64", "offset": 0, "size": 8, "alignment": 8},
                {"name": "b", "type": "uint64", "offset": 8, "size": 8, "alignment": 8},
                {"name": "c", "type": "uint64", "offset": 16, "size": 8, "alignment": 8},
                {"name": "m", "type": "uint32", "offset": 24, "size": 4, "alignment": 4},
            ],
            "tail_padding_bytes": 4,
        }:
            _fail("F2 kernarg ABI is not frozen")


def validate_manifest(
    record: Any,
    *,
    asset_root: os.PathLike[str] | str,
    policy_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Validate one complete owning manifest and return the same record."""

    root = Path(asset_root)
    top = _dict(record, _TOP_KEYS, "manifest")
    if type(top["schema_version"]) is not int or top["schema_version"] != _SCHEMA_VERSION:
        _fail("schema_version must be exactly 1")
    name = _string(top["name"], "name")
    if not _IDENTIFIER_RE.fullmatch(name):
        _fail("name is not a stable pack identity")
    version = _string(top["version"], "version")
    if not _VERSION_RE.fullmatch(version):
        _fail("version must be canonical MAJOR.MINOR.PATCH")
    if top["target"] != _TARGET:
        _fail("target must be gfx1201")
    features = _list(top["required_features"], "required_features")
    for index, feature in enumerate(features):
        _string(feature, f"required_features[{index}]")
    if features != sorted(set(features)):
        _fail("required_features must be sorted and unique")

    provenance = _validate_provenance(top["provenance"], root, policy_path)
    image = _validate_image(top["image"], root)
    entries = _validate_entries(top["entries"], image["image_size"], features)
    compatibility = _validate_compatibility(top["compatibility"])
    numerics = _validate_numerics(top["numerics"], compatibility)

    families = compatibility["shape_families"]
    family_names = {family["name"] for family in families}
    for entry_index, entry in enumerate(entries):
        cases = entry["geometry"]["cases"]
        case_names = {case["shape_family"] for case in cases}
        if case_names != family_names or len(cases) != len(families):
            _fail(f"entries[{entry_index}] geometry does not cover shape families exactly")
        for case in cases:
            family = next(family for family in families if family["name"] == case["shape_family"])
            if case["geometry_rule"] != family["geometry_rule"]:
                _fail("entry geometry rule disagrees with its shape family")
            if case["dynamic_lds_allowed"] and case["dynamic_lds_max_bytes"] > entry["resources"]["lds_bytes"]:
                _fail("dynamic LDS exceeds the declared resource limit")
    _validate_f2_contract(compatibility, numerics, entries)

    # Add the image itself to the component/license closure after the image
    # bytes have been checked, then require exact file/component coverage.
    image_component = image["image_path"]
    covered = {
        review["component"] for review in provenance["license_reviews"]
    }
    expected = {
        *provenance["upstream_paths"],
        *(source["path"] for source in provenance["local_sources"]),
        *(modification["component"] for modification in provenance["modifications"]),
        image_component,
    }
    if covered != expected:
        _fail("license reviews must cover exactly every declared source/image/component")

    pack_sha = compute_pack_sha256(top)
    _validate_evidence(
        top["evidence"],
        numerics,
        compatibility,
        root=root,
        target=top["target"],
        image_sha=image["image_sha256"],
        pack_sha=pack_sha,
        resources=entries[0]["resources"],
    )
    _VALIDATED_PACK_ROOTS[pack_sha] = (root, Path(policy_path) if policy_path is not None else None)
    return top


def _cpp_string(value: str) -> str:
    """Render a deterministic C++ string literal.

    Evidence paths commonly end in ``.json``.  Split that token across
    adjacent literals so generated runtime source does not advertise a JSON
    parser/library while preserving the exact C++ string value.
    """

    rendered = json.dumps(value, ensure_ascii=False)
    if "json" not in rendered.lower():
        return rendered
    # Split each occurrence across adjacent literals.  For example, the
    # escaped source becomes `"evidence/x.j" "son"` while evaluating to the
    # original `evidence/x.json` value in C++.
    while "json" in rendered.lower():
        index = rendered.lower().find("json")
        rendered = rendered[: index + 1] + '" "' + rendered[index + 1 :]
    return rendered


def _cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def _cpp_name(value: str, prefix: str = "k") -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not safe or safe[0].isdigit():
        safe = "_" + safe
    return prefix + safe


def _emit_span(lines: list[str], field: str, array_name: str, count: int) -> None:
    if count:
        lines.append(f"  value.{field} = {{{array_name}, {count}}};")
    else:
        lines.append(f"  value.{field} = {{nullptr, 0}};")


def _emit_ref(lines: list[str], variable: str, ref: dict[str, Any]) -> None:
    lines.append(f"constexpr EvidenceRef {variable} = [] {{")
    lines.append("  EvidenceRef value{};")
    for key in (
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
    ):
        lines.append(f"  value.{key} = {_cpp_string(ref[key])};")
    lines.append("  return value;")
    lines.append("}();")


def generate_cpp_initializers(
    record: Mapping[str, Any],
    *,
    asset_root: os.PathLike[str] | str | None = None,
    policy_path: os.PathLike[str] | str | None = None,
) -> str:
    """Render a reproducible view only after complete manifest validation."""

    if not isinstance(record, Mapping):
        _fail("pack record must be an object")
    pack_sha = compute_pack_sha256(record)
    if asset_root is None:
        validated = _VALIDATED_PACK_ROOTS.get(pack_sha)
        if validated is None:
            _fail("generate_cpp_initializers requires a validated manifest and asset_root")
        validation_root, validation_policy = validated
        if policy_path is not None:
            validation_policy = Path(policy_path)
    else:
        validation_root = Path(asset_root)
        validation_policy = Path(policy_path) if policy_path is not None else None
    validate_manifest(record, asset_root=validation_root, policy_path=validation_policy)
    pack_sha = compute_pack_sha256(record)
    try:
        top = dict(record)
        identity = top["name"]
        version = top["version"]
        target = top["target"]
        features = top["required_features"]
        provenance = top["provenance"]
        image = top["image"]
        entries = top["entries"]
        compatibility = top["compatibility"]
        numerics = top["numerics"]
        evidence = top["evidence"]
    except (KeyError, TypeError) as exc:
        _fail(f"cannot generate an incomplete pack record: {exc}")

    # Keep generated source independent of mapping insertion order by reading
    # only named fields and preserving only declaration/list order.
    symbol = _cpp_name(f"{identity}_{version}_{pack_sha}")
    lines = [
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <string_view>",
        '#include "kernel_pack.h"',
        "",
        "namespace native_r9700::generated {",
        "namespace {",
        "using native_r9700::KernelPackOptional;",
        "using native_r9700::KernelPackSpan;",
        f"constexpr std::string_view {symbol}PackSha256 = {_cpp_string(pack_sha)};",
    ]

    def emit_string_array(array_name: str, values: list[Any]) -> None:
        if not values:
            return
        lines.append(f"constexpr std::string_view {array_name}[] = {{")
        for value in values:
            lines.append(f"    {_cpp_string(value)},")
        lines.append("};")
    upstream_paths = provenance.get("upstream_paths", []) if isinstance(provenance, dict) else []
    if upstream_paths:
        emit_string_array(f"{symbol}UpstreamPaths", list(upstream_paths))

    feature_values = list(features) if isinstance(features, list) else []
    emit_string_array(f"{symbol}Features", feature_values)

    sources = provenance.get("local_sources", []) if isinstance(provenance, dict) else []
    if sources:
        lines.append(f"constexpr KernelPackSource {symbol}Sources[] = {{")
        for source in sources:
            lines.append(
                f"    {{{_cpp_string(source['path'])}, {_cpp_string(source['sha256'])}}},"
            )
        lines.append("};")

    licenses = provenance.get("license_reviews", []) if isinstance(provenance, dict) else []
    if licenses:
        lines.append(f"constexpr KernelPackLicenseReview {symbol}Licenses[] = {{")
        for review in licenses:
            lines.append(
                "    {"
                f"{_cpp_string(review['component'])}, "
                f"{_cpp_string(review['spdx_expression'])}, "
                f"{_cpp_string(review['review_id'])}, "
                f"{_cpp_string(review['status'])}"
                "},"
            )
        lines.append("};")

    modifications = provenance.get("modifications", []) if isinstance(provenance, dict) else []
    if modifications:
        lines.append(f"constexpr KernelPackModification {symbol}Modifications[] = {{")
        for modification in modifications:
            lines.append(
                f"    {{{_cpp_string(modification['component'])}, {_cpp_string(modification['summary'])}}},"
            )
        lines.append("};")

    # Emit every nested view into static storage.  These arrays contain no
    # owning containers; the final record only refers to their spans.
    for entry_index, entry in enumerate(entries if isinstance(entries, list) else []):
        entry_prefix = f"{symbol}Entry{entry_index}"
        fields = entry["kernargs"]["fields"]
        lines.append(f"constexpr KernelPackKernargField {entry_prefix}Fields[] = {{")
        for field in fields:
            lines.append(
                f"    {{{_cpp_string(field['name'])}, {_cpp_string(field['type'])}, "
                f"{field['offset']}, {field['size']}, {field['alignment']}}},"
            )
        lines.append("};")
        cases = entry["geometry"]["cases"]
        lines.append(f"constexpr KernelPackGeometryCase {entry_prefix}Geometry[] = {{")
        for case in cases:
            lines.append(
                "    {"
                f"{_cpp_string(case['shape_family'])}, {_cpp_string(case['geometry_rule'])}, "
                f"{case['workgroup_x']}, {case['workgroup_y']}, {case['workgroup_z']}, "
                f"{case['global_x']}, {case['global_y']}, {case['global_z']}, "
                f"{case['grid_tile_m']}, {case['grid_tile_n']}, "
                f"{_cpp_bool(case['dynamic_lds_allowed'])}, {case['dynamic_lds_max_bytes']}"
                "},"
            )
        lines.append("};")

    families = compatibility.get("shape_families", []) if isinstance(compatibility, dict) else []
    for family_index, family in enumerate(families if isinstance(families, list) else []):
        family_prefix = f"{symbol}Family{family_index}"
        dimensions = family["fixed_dimensions"]
        lines.append(f"constexpr KernelPackShapeDimension {family_prefix}Dimensions[] = {{")
        for dimension in dimensions:
            lines.append(f"    {{{_cpp_string(dimension['name'])}, {dimension['value']}}},")
        lines.append("};")
        runtime = family["runtime_dimension"]
        if runtime is not None:
            lines.append(
                f"constexpr KernelPackRuntimeDimension {family_prefix}Runtime = "
                f"{{{_cpp_string(runtime['name'])}, {runtime['min_value']}, "
                f"{runtime['max_value']}, {runtime['full_value']}}};"
            )
        lines.append(f"constexpr KernelPackShapeFamily {family_prefix} = [] {{")
        lines.append("  KernelPackShapeFamily value{};")
        lines.append(f"  value.name = {_cpp_string(family['name'])};")
        _emit_span(lines, "fixed_dimensions", f"{family_prefix}Dimensions", len(dimensions))
        if runtime is None:
            lines.append("  value.runtime_dimension.present = false;")
        else:
            lines.append("  value.runtime_dimension.present = true;")
            lines.append(f"  value.runtime_dimension.value = {family_prefix}Runtime;")
        lines.append(f"  value.tail_policy = {_cpp_string(family['tail_policy'])};")
        lines.append(f"  value.geometry_rule = {_cpp_string(family['geometry_rule'])};")
        lines.append("  return value;")
        lines.append("}();")
    if families:
        lines.append(f"constexpr KernelPackShapeFamily {symbol}Families[] = {{")
        for family_index, _family in enumerate(families):
            lines.append(f"    {symbol}Family{family_index},")
        lines.append("};")

    cast_points = numerics.get("cast_points", []) if isinstance(numerics, dict) else []
    if cast_points:
        lines.append(f"constexpr KernelPackCastPoint {symbol}CastPoints[] = {{")
        for point in cast_points:
            lines.append(
                f"    {{{_cpp_string(point['stage'])}, {_cpp_string(point['from_dtype'])}, "
                f"{_cpp_string(point['to_dtype'])}}},"
            )
        lines.append("};")

    refs: dict[str, tuple[str, dict[str, Any]]] = {}
    for ref_name, ref in (
        ("RetainedReference", numerics.get("retained_reference")),
        ("NumpyOracle", numerics.get("numpy_oracle")),
        ("ScalarNativeProjection", numerics.get("scalar_native_projection")),
        ("Conformance", evidence.get("conformance")),
        ("SourceReview", evidence.get("source_review")),
        ("NativeRun", evidence.get("native_run")),
        ("ResourceReview", evidence.get("resource_review")),
        ("IsaReview", evidence.get("isa_review")),
        ("LayoutProof", evidence.get("layout_proof")),
        ("BenchmarkRecord", evidence.get("benchmark_record")),
    ):
        if isinstance(ref, dict):
            variable = f"{symbol}{ref_name}"
            _emit_ref(lines, variable, ref)
            refs[ref_name] = (variable, ref)

    for entry_index, entry in enumerate(entries if isinstance(entries, list) else []):
        entry_prefix = f"{symbol}Entry{entry_index}"
        lines.append(f"constexpr KernelPackEntry {entry_prefix} = [] {{")
        lines.append("  KernelPackEntry value{};")
        lines.append(f"  value.symbol = {_cpp_string(entry['symbol'])};")
        lines.append(f"  value.descriptor_offset = {entry['descriptor_offset']};")
        lines.append(f"  value.entry_offset = {entry['entry_offset']};")
        lines.append(f"  value.kernargs.bytes = {entry['kernargs']['bytes']};")
        _emit_span(lines, "kernargs.fields", f"{entry_prefix}Fields", len(entry["kernargs"]["fields"]))
        lines.append(
            f"  value.kernargs.tail_padding_bytes = {entry['kernargs']['tail_padding_bytes']};"
        )
        resources = entry["resources"]
        for key in (
            "rsrc1",
            "rsrc2",
            "rsrc3",
            "wave_size",
            "sgpr_count",
            "vgpr_count",
            "lds_bytes",
            "private_segment_bytes",
        ):
            lines.append(f"  value.resources.{key} = {resources[key]};")
        lines.append(
            f"  value.resources.metadata_provenance = {_cpp_string(resources['metadata_provenance'])};"
        )
        _emit_span(lines, "geometry.cases", f"{entry_prefix}Geometry", len(entry["geometry"]["cases"]))
        lines.append("  return value;")
        lines.append("}();")

    lines.append(f"constexpr KernelPackEntry {symbol}Entries[] = {{")
    for entry_index, _entry in enumerate(entries if isinstance(entries, list) else []):
        lines.append(f"    {symbol}Entry{entry_index},")
    lines.append("};")

    lines.append(f"constexpr KernelPackIdentity {symbol}Identity = [] {{")
    lines.append("  KernelPackIdentity value{};")
    lines.append(f"  value.schema_version = {top.get('schema_version', 1)};")
    lines.append(f"  value.name = {_cpp_string(identity)};")
    lines.append(f"  value.version = {_cpp_string(version)};")
    lines.append(f"  value.target = {_cpp_string(target)};")
    _emit_span(lines, "required_features", f"{symbol}Features", len(feature_values))
    lines.append("  return value;")
    lines.append("}();")

    lines.append(f"constexpr KernelPackProvenance {symbol}Provenance = [] {{")
    lines.append("  KernelPackProvenance value{};")
    lines.append(f"  value.upstream_repository = {_cpp_string(provenance['upstream_repository'])};")
    lines.append(f"  value.upstream_revision = {_cpp_string(provenance['upstream_revision'])};")
    upstream_paths = provenance.get("upstream_paths", [])
    _emit_span(lines, "upstream_paths", f"{symbol}UpstreamPaths", len(upstream_paths))
    _emit_span(lines, "local_sources", f"{symbol}Sources", len(sources))
    _emit_span(lines, "license_reviews", f"{symbol}Licenses", len(licenses))
    _emit_span(lines, "modifications", f"{symbol}Modifications", len(modifications))
    lines.append("  return value;")
    lines.append("}();")

    lines.append(f"constexpr KernelPackImage {symbol}Image = [] {{")
    lines.append("  KernelPackImage value{};")
    lines.append(f"  value.image_path = {_cpp_string(image['image_path'])};")
    lines.append(f"  value.image_sha256 = {_cpp_string(image['image_sha256'])};")
    lines.append(f"  value.image_size = {image['image_size']};")
    lines.append(f"  value.code_object_version = {_cpp_string(image['code_object_version'])};")
    build = image["build"]
    for key in ("toolchain_id", "toolchain_revision", "generator_id", "generator_revision", "command_sha256"):
        lines.append(f"  value.build.{key} = {_cpp_string(build[key])};")
    lines.append("  return value;")
    lines.append("}();")

    lines.append(f"constexpr KernelPackCompatibility {symbol}Compatibility = [] {{")
    lines.append("  KernelPackCompatibility value{};")
    for key in (
        "input_dtype",
        "weight_dtype",
        "output_dtype",
        "source_tensor_layout_version",
        "weight_packing_version",
    ):
        lines.append(f"  value.{key} = {_cpp_string(compatibility[key])};")
    if families:
        lines.append(
            f"  value.shape_families = {{{symbol}Families, {len(families)}}};"
        )
    else:
        lines.append("  value.shape_families = {nullptr, 0};")
    lines.append("  return value;")
    lines.append("}();")

    lines.append(f"constexpr KernelPackNumerics {symbol}Numerics = [] {{")
    lines.append("  KernelPackNumerics value{};")
    for key in (
        "input_dtype",
        "accumulation_dtype",
        "output_dtype",
        "finite_value_rule",
        "tolerance_policy",
        "reference_set_kind",
    ):
        lines.append(f"  value.{key} = {_cpp_string(numerics[key])};")
    _emit_span(lines, "cast_points", f"{symbol}CastPoints", len(cast_points))
    for optional_name, ref_name in (
        ("retained_reference", "RetainedReference"),
        ("numpy_oracle", "NumpyOracle"),
        ("scalar_native_projection", "ScalarNativeProjection"),
    ):
        if ref_name in refs:
            lines.append(f"  value.{optional_name}.present = true;")
            lines.append(f"  value.{optional_name}.value = {refs[ref_name][0]};")
        else:
            lines.append(f"  value.{optional_name}.present = false;")
    lines.append("  return value;")
    lines.append("}();")

    lines.append(f"constexpr KernelPackEvidence {symbol}Evidence = [] {{")
    lines.append("  KernelPackEvidence value{};")
    for key, ref_name in (
        ("conformance", "Conformance"),
        ("source_review", "SourceReview"),
        ("native_run", "NativeRun"),
        ("resource_review", "ResourceReview"),
        ("isa_review", "IsaReview"),
    ):
        if ref_name in refs:
            lines.append(f"  value.{key} = {refs[ref_name][0]};")
    for optional_name, ref_name in (("layout_proof", "LayoutProof"), ("benchmark_record", "BenchmarkRecord")):
        if ref_name in refs:
            lines.append(f"  value.{optional_name}.present = true;")
            lines.append(f"  value.{optional_name}.value = {refs[ref_name][0]};")
        else:
            lines.append(f"  value.{optional_name}.present = false;")
    lines.append(
        f"  value.benchmark_not_applicable_reason = "
        f"{_cpp_string(evidence['benchmark_not_applicable_reason'])};"
    )
    lines.append("  return value;")
    lines.append("}();")

    lines.append(f"constexpr KernelPackRecord {symbol}Record = [] {{")
    lines.append("  KernelPackRecord value{};")
    lines.append(f"  value.identity = {symbol}Identity;")
    lines.append(f"  value.provenance = {symbol}Provenance;")
    lines.append(f"  value.image = {symbol}Image;")
    lines.append(f"  value.entries = {{{symbol}Entries, {len(entries)}}};")
    lines.append(f"  value.compatibility = {symbol}Compatibility;")
    lines.append(f"  value.numerics = {symbol}Numerics;")
    lines.append(f"  value.evidence = {symbol}Evidence;")
    lines.append("  return value;")
    lines.append("}();")
    lines.extend(
        [
            "}  // namespace",
            f"const KernelPackRecord& {symbol} = {symbol}Record;",
            "}  // namespace native_r9700::generated",
            "",
        ]
    )
    return "\n".join(lines)
