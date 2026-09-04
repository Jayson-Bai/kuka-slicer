"""Explicit input contract for the conformal-lattice geometry pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping


CONFORMAL_LATTICE_SPEC_V1 = "conformal_lattice_spec_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ConformalLatticeSpec:
    """Validated, serialisable input for the independent geometry pipeline.

    ``source_surface.double_sine`` is used only when ``provider`` is
    ``double_sine``.  It contains the six analytical height-field parameters,
    ``xy_bounds_mm`` (``[xmin, ymin, xmax, ymax]``), and ``samples``
    (``[x_count, y_count]``).  A ``triangle_mesh`` source is supplied later as
    an STL byte stream or an indexed-NPZ byte stream (``vertices`` and
    ``faces``); its SHA-256 is checked before mesh preparation.
    """

    source_surface: dict[str, object]
    parameterization: dict[str, object]
    lattice: dict[str, object]
    fill_field: dict[str, object]
    orientation_field: dict[str, object]
    layer_embedding: dict[str, object]
    quality_limits: dict[str, object]
    random_seed: int
    raw_config: dict[str, object]

    @property
    def source_provider(self) -> str:
        return str(self.source_surface["provider"])

    @property
    def source_sha256(self) -> str:
        return str(self.source_surface["sha256"])

    @property
    def source_file(self) -> str:
        return str(self.source_surface["source_file"])

    def metadata(self) -> dict[str, object]:
        """Return the immutable contract metadata carried into every result."""

        return {
            "format": CONFORMAL_LATTICE_SPEC_V1,
            "random_seed": self.random_seed,
            "source_surface": self.source_surface,
            "parameterization": self.parameterization,
            "lattice": self.lattice,
            "fill_field": self.fill_field,
            "orientation_field": self.orientation_field,
            "layer_embedding": self.layer_embedding,
            "quality_limits": self.quality_limits,
        }


def canonical_json_sha256(value: Mapping[str, object]) -> str:
    """Hash a config deterministically without relying on a file name."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def double_sine_source_sha256(source_surface: Mapping[str, object]) -> str:
    """Return the reproducible source hash for a generated double-sine mesh."""

    value = dict(source_surface)
    value.pop("sha256", None)
    return canonical_json_sha256(value)


def load_conformal_lattice_spec(data: bytes | str | Mapping[str, object]) -> ConformalLatticeSpec:
    """Read and validate a ``conformal_lattice_spec_v1`` JSON document."""

    raw = _decode(data)
    if raw.get("format") != CONFORMAL_LATTICE_SPEC_V1:
        raise ValueError(f"conformal lattice format must be {CONFORMAL_LATTICE_SPEC_V1}")
    if raw.get("units") != "mm":
        raise ValueError("conformal lattice units must be mm")

    source = _object(raw, "source_surface")
    provider = source.get("provider")
    if provider not in ("double_sine", "triangle_mesh"):
        raise ValueError("source_surface.provider must be double_sine or triangle_mesh")
    if source.get("domain") != "outer_boundary_only":
        raise ValueError("source_surface.domain must be outer_boundary_only")
    if not isinstance(source.get("source_file"), str) or not source["source_file"].strip():
        raise ValueError("source_surface.source_file must be a non-empty string")
    source_hash = source.get("sha256")
    if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
        raise ValueError("source_surface.sha256 must be a lowercase SHA-256 hex digest")
    if provider == "double_sine":
        _validate_double_sine_source(source)

    parameterization = _object(raw, "parameterization")
    if parameterization.get("method") != "lscm":
        raise ValueError("parameterization.method must be lscm")
    anchor_strategy = parameterization.get("anchor_strategy")
    if anchor_strategy not in ("farthest_boundary_pair", "user"):
        raise ValueError("parameterization.anchor_strategy must be farthest_boundary_pair or user")
    if parameterization.get("seam_strategy") not in ("none", "user", "auto_cut_graph"):
        raise ValueError("parameterization.seam_strategy must be none, user, or auto_cut_graph")
    if anchor_strategy == "user":
        anchors = parameterization.get("anchors")
        if not isinstance(anchors, list) or len(anchors) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in anchors
        ) or anchors[0] == anchors[1]:
            raise ValueError("parameterization.anchors must contain two distinct non-negative vertex indices")

    lattice = _object(raw, "lattice")
    if lattice.get("family") != "triangular_dual_hex":
        raise ValueError("lattice.family must be triangular_dual_hex")
    _positive(lattice.get("wall_width_mm"), "lattice.wall_width_mm")
    _positive(lattice.get("base_cell_size_mm"), "lattice.base_cell_size_mm")
    if lattice.get("boundary_mode") not in ("clip", "inset", "boundary_frame"):
        raise ValueError("lattice.boundary_mode must be clip, inset, or boundary_frame")
    _vector(lattice.get("phase_origin"), "lattice.phase_origin", length=2)

    fill_field = _object(raw, "fill_field")
    orientation_field = _object(raw, "orientation_field")
    layer_embedding = _object(raw, "layer_embedding")
    quality_limits = _object(raw, "quality_limits")
    fill_mode = fill_field.get("mode")
    if fill_mode not in ("fixed_cell_size", "weighted_composite", "direct_target_fill_ratio"):
        raise ValueError("fill_field.mode is unsupported")
    if not isinstance(fill_field.get("drivers"), list):
        raise ValueError("fill_field.drivers must be an array")
    if fill_mode == "fixed_cell_size" and fill_field["drivers"]:
        raise ValueError("fixed_cell_size must not declare fill_field.drivers")
    if orientation_field.get("mode") not in ("global_axis", "principal_curvature", "user", "stress", "external"):
        raise ValueError("orientation_field.mode is unsupported")
    _finite(orientation_field.get("angle_deg", 0.0), "orientation_field.angle_deg")
    if layer_embedding.get("mode") not in ("target_surface_normal_stack", "symmetric_shape_morphing"):
        raise ValueError("layer_embedding.mode is unsupported")
    if layer_embedding.get("mode") == "symmetric_shape_morphing":
        if layer_embedding.get("transition") != "smoothstep":
            raise ValueError("symmetric_shape_morphing requires transition=smoothstep")
        start_layer = layer_embedding.get("surface_start_layer")
        if not isinstance(start_layer, int) or isinstance(start_layer, bool) or start_layer < 0:
            raise ValueError("symmetric_shape_morphing requires a non-negative integer surface_start_layer")
    seed = raw.get("random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    return ConformalLatticeSpec(
        source_surface=dict(source),
        parameterization=dict(parameterization),
        lattice=dict(lattice),
        fill_field=dict(fill_field),
        orientation_field=dict(orientation_field),
        layer_embedding=dict(layer_embedding),
        quality_limits=dict(quality_limits),
        random_seed=seed,
        raw_config=raw,
    )


def _decode(data: bytes | str | Mapping[str, object]) -> dict[str, object]:
    if isinstance(data, bytes):
        value: Any = json.loads(data.decode("utf-8"))
    elif isinstance(data, str):
        value = json.loads(data)
    else:
        value = dict(data)
    if not isinstance(value, dict):
        raise ValueError("conformal lattice spec must be a JSON object")
    return value


def _object(parent: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _vector(value: object, name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} finite numbers")
    return tuple(_finite(item, name) for item in value)


def _validate_double_sine_source(source: Mapping[str, object]) -> None:
    surface = _object(source, "double_sine")
    for key in ("amplitude_mm", "wavelength_x_mm", "wavelength_y_mm", "phase_x_rad", "phase_y_rad", "z_reference_mm"):
        _finite(surface.get(key), f"source_surface.double_sine.{key}")
    _positive(surface.get("wavelength_x_mm"), "source_surface.double_sine.wavelength_x_mm")
    _positive(surface.get("wavelength_y_mm"), "source_surface.double_sine.wavelength_y_mm")
    bounds = _vector(surface.get("xy_bounds_mm"), "source_surface.double_sine.xy_bounds_mm", length=4)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("source_surface.double_sine.xy_bounds_mm must have positive dimensions")
    samples = surface.get("samples")
    if not isinstance(samples, list) or len(samples) != 2 or any(not isinstance(value, int) or value < 2 for value in samples):
        raise ValueError("source_surface.double_sine.samples must contain two integers >= 2")
