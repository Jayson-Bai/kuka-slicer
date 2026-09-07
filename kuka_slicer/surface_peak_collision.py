"""Offline peak-curvature collision preflight for mapped surface NPZ files.

This module deliberately stays outside slicing and preview rendering.  It
checks only the full-curvature layers recorded by the mapper, resolves the
paired surface JSON and source STL by their immutable hashes, preserves holes
from the STL section, and tests the CAD-derived heater-block triangle mesh.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from shapely import contains_xy

from .slicer import _intersect_mesh_at_z, _solid_geometry_from_contours, _stitch_segments
from .stl_io import load_stl
from .surface_mapper.contracts import SourceNPZ, SurfaceTarget, load_surface_target, read_source_npz


_ASSET_PATH = Path(__file__).resolve().parents[1] / "assets" / "printhead" / "printhead_interference_check.preview.json"
_PATH_KEYS = ("R", "F", "T")
_POSE_GRID_MM = 0.5
_POSE_GRID_DEG = 0.5
_COARSE_SAMPLE_SPACING_MM = 0.5
_REFINED_SAMPLE_SPACING_MM = 0.1
_REFINEMENT_RADIUS_MM = 6.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _surface_digest(target: SurfaceTarget) -> str:
    return hashlib.sha256(
        json.dumps(target.raw_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mapping_meta(source: SourceNPZ) -> dict[str, Any]:
    mapping = source.meta.get("surface_mapping")
    if not isinstance(mapping, dict) or mapping.get("format") != "surface_mapping_v1":
        raise ValueError("NPZ does not include a surface_mapping_v1 record")
    return mapping


def resolve_surface_collision_inputs(npz_path: Path) -> tuple[SourceNPZ, SurfaceTarget, Path, Path]:
    """Resolve the exact JSON/STL pairing named by a mapped NPZ's hashes."""

    npz_path = Path(npz_path).resolve()
    source = read_source_npz(npz_path.read_bytes(), source_name=npz_path.name)
    mapping = _mapping_meta(source)
    expected_surface = str(mapping.get("target_surface_sha256", ""))
    expected_stl = str(mapping.get("target_source_sha256", ""))
    if len(expected_surface) != 64 or len(expected_stl) != 64:
        raise ValueError("NPZ surface_mapping is missing target surface/STL SHA-256 hashes")

    json_matches: list[tuple[SurfaceTarget, Path]] = []
    for candidate in npz_path.parent.glob("*.json"):
        try:
            target = load_surface_target(candidate.read_bytes())
        except ValueError:
            continue
        if _surface_digest(target) == expected_surface:
            json_matches.append((target, candidate))
    if len(json_matches) != 1:
        raise ValueError(f"expected exactly one matching surface JSON beside {npz_path.name}; found {len(json_matches)}")

    # The standard test package keeps the source model under its sibling 01_模型
    # directory.  Bound the search to that package instead of searching disks.
    package_root = npz_path.parent.parent
    stl_matches = [candidate for candidate in package_root.rglob("*.stl") if _sha256(candidate) == expected_stl]
    if len(stl_matches) != 1:
        raise ValueError(f"expected exactly one source STL matching NPZ hash under {package_root}; found {len(stl_matches)}")
    target, json_path = json_matches[0]
    return source, target, json_path, stl_matches[0]


def _heater_mesh(sample_spacing_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(_ASSET_PATH.read_text(encoding="utf-8"))
    components = data.get("model_components")
    if not isinstance(components, list):
        raise ValueError("printhead asset does not provide exact model components")
    heater = next((item for item in components if item.get("name") == "heater_block"), None)
    if not isinstance(heater, dict):
        raise ValueError("printhead asset has no heater_block mesh")
    positions = np.asarray(data["positions"], dtype=np.float64)
    triangles = np.asarray(data["triangles"], dtype=np.int64)
    triangle_indexes = np.asarray(heater["triangle_indices"], dtype=np.int64)
    mesh = positions[triangles[triangle_indexes]]
    bounds = np.vstack((np.min(mesh.reshape(-1, 3), axis=0), np.max(mesh.reshape(-1, 3), axis=0)))
    return mesh, bounds[0], bounds[1], _sample_mesh_surface(mesh, sample_spacing_mm)


def _sample_mesh_surface(mesh: np.ndarray, sample_spacing_mm: float) -> np.ndarray:
    """Adaptively probe real mesh edges/faces; no box proxy is used."""

    samples: list[np.ndarray] = [mesh.reshape(-1, 3), np.mean(mesh, axis=1)]
    edges = np.concatenate((mesh[:, (0, 1)], mesh[:, (1, 2)], mesh[:, (2, 0)]), axis=0)
    for start, end in edges:
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / sample_spacing_mm)))
        if count > 1:
            fractions = np.arange(1, count, dtype=np.float64)[:, None] / count
            samples.append(start + fractions * (end - start))
    joined = np.vstack(samples)
    # Stable sub-millimetre coordinate keys remove shared triangle edges.
    keys = np.round(joined / 1e-5).astype(np.int64)
    _, unique = np.unique(keys, axis=0, return_index=True)
    return joined[np.sort(unique)]


def _rotation_frame(abc_deg: np.ndarray) -> np.ndarray:
    """Tool axes in base coordinates, matching the UI/mapper Z-Y-X convention."""

    a, b, c = np.radians(abc_deg)
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rz = np.array(((ca, -sa, 0.0), (sa, ca, 0.0), (0.0, 0.0, 1.0)))
    ry = np.array(((cb, 0.0, sb), (0.0, 1.0, 0.0), (-sb, 0.0, cb)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cc, -sc), (0.0, sc, cc)))
    flat = np.array(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    return rz @ ry @ rx @ flat


def _inside_closed_mesh(points: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    """Odd/even ray test against the heater's actual triangle mesh."""

    if not len(points):
        return np.zeros(0, dtype=bool)
    direction = np.array((1.0, 0.0, 0.0))
    first, second, third = mesh[:, 0], mesh[:, 1], mesh[:, 2]
    edge1, edge2 = second - first, third - first
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    usable = np.abs(determinant) > 1e-10
    first, edge1, edge2, h, determinant = first[usable], edge1[usable], edge2[usable], h[usable], determinant[usable]
    result = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), 256):
        probe = points[start : start + 256]
        s = probe[:, None, :] - first[None, :, :]
        u = np.einsum("pij,ij->pi", s, h) / determinant
        q = np.cross(s, edge1[None, :, :])
        v = q[:, :, 0] / determinant  # dot(direction, q) where direction=(1,0,0)
        distance = np.einsum("pij,ij->pi", q, edge2) / determinant
        hits = (u > 1e-8) & (v > 1e-8) & (u + v < 1.0 - 1e-8) & (distance > 1e-8)
        result[start : start + len(probe)] = np.count_nonzero(hits, axis=1) % 2 == 1
    return result


def _layer_poses(source: SourceNPZ, layer: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for kind in _PATH_KEYS:
        values = source.arrays.get(f"layer_{layer:04d}_{kind}")
        if values is None or values.shape[-1] != 6:
            continue
        finite = values[np.isfinite(values).all(axis=2)]
        if finite.size:
            rows.append(np.asarray(finite, dtype=np.float64))
    if not rows:
        return np.empty((0, 6), dtype=np.float64)
    poses = np.vstack(rows)
    keys = np.column_stack((np.round(poses[:, :3] / _POSE_GRID_MM), np.round(poses[:, 3:] / _POSE_GRID_DEG))).astype(np.int64)
    _, indexes = np.unique(keys, axis=0, return_index=True)
    return poses[np.sort(indexes)]


def _material_geometry(stl_path: Path, flat_z: float):
    mesh = load_stl(stl_path)
    tolerance = 1e-5
    contours = _stitch_segments(_intersect_mesh_at_z(mesh.triangles, flat_z, tolerance), tolerance)
    return _solid_geometry_from_contours(contours)


def _flat_layer_z(poses: np.ndarray, surface: SurfaceTarget, alpha: float) -> float:
    return float(np.median(poses[:, 2] - alpha * surface.surface.height(poses[:, 0], poses[:, 1])))


def _candidate_pyramid_mask(points_local: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    """Four-sided TCP-to-heater candidate envelope, used only as a broad phase."""

    depth = max(abs(float(bounds_max[0])), 1e-6)
    scale = np.maximum(1.0, -points_local[:, 0] / depth)
    return (
        (points_local[:, 0] <= 1e-5)
        & (points_local[:, 0] >= bounds_min[0] - 1e-5)
        & (points_local[:, 1] >= bounds_min[1] * scale - 1e-5)
        & (points_local[:, 1] <= bounds_max[1] * scale + 1e-5)
        & (points_local[:, 2] >= bounds_min[2] * scale - 1e-5)
        & (points_local[:, 2] <= bounds_max[2] * scale + 1e-5)
    )


def _peak_layers_and_alpha(source: SourceNPZ) -> tuple[list[int], list[float] | dict[str, float]]:
    progression = _mapping_meta(source).get("progression")
    if not isinstance(progression, dict):
        raise ValueError("NPZ surface_mapping has no progression")
    peak_layers = [int(value) for value in progression.get("peak_layers", [])]
    alpha_by_layer = progression.get("alpha_by_layer")
    if not peak_layers or not isinstance(alpha_by_layer, (list, dict)):
        raise ValueError("NPZ surface_mapping has no peak-curvature layers")
    return peak_layers, alpha_by_layer


def _run_mesh_sampling(
    source: SourceNPZ,
    target: SurfaceTarget,
    stl_path: Path,
    *,
    sample_spacing_mm: float,
    focus_tcp_xyzabc: list[float] | None = None,
) -> dict[str, object]:
    """Run one complete or locally focused mesh-sampling pass."""

    peak_layers, alpha_by_layer = _peak_layers_and_alpha(source)
    heater, bounds_min, bounds_max, probes = _heater_mesh(sample_spacing_mm)
    checked = 0
    candidate_points = 0
    minimum_clearance_mm = float("inf")
    minimum_clearance_location: dict[str, object] | None = None
    per_layer: list[dict[str, object]] = []

    for layer in peak_layers:
        poses = _layer_poses(source, layer)
        if focus_tcp_xyzabc is not None:
            center_xy = np.asarray(focus_tcp_xyzabc[:2], dtype=np.float64)
            poses = poses[np.linalg.norm(poses[:, :2] - center_xy, axis=1) <= _REFINEMENT_RADIUS_MM]
        if not len(poses):
            if focus_tcp_xyzabc is None:
                raise ValueError(f"peak layer {layer} has no XYZABC tool path")
            continue
        alpha = float(alpha_by_layer[layer] if isinstance(alpha_by_layer, list) else alpha_by_layer[str(layer)])
        flat_z = _flat_layer_z(poses, target, alpha)
        material = _material_geometry(stl_path, flat_z)
        if material.is_empty:
            raise ValueError(f"source STL has no material at peak layer {layer} (z={flat_z:.4f} mm)")
        layer_candidates = 0
        for pose_index, pose in enumerate(poses):
            frame = _rotation_frame(pose[3:])
            probe_world = pose[:3] + probes @ frame.T
            in_material = contains_xy(material, probe_world[:, 0], probe_world[:, 1])
            if not np.any(in_material):
                checked += 1
                continue
            heater_surface_points = probe_world[in_material]
            surface_points = heater_surface_points.copy()
            surface_points[:, 2] = flat_z + alpha * target.surface.height(surface_points[:, 0], surface_points[:, 1])
            local = (surface_points - pose[:3]) @ frame
            broad = _candidate_pyramid_mask(local, bounds_min, bounds_max)
            local = local[broad]
            # This is the normal-to-build-plane distance between the actual
            # heater triangle samples and the curved material sampled directly
            # beneath them.  It is reported as a sampled clearance, not as a
            # fabricated safety allowance; no extra offset is applied.
            sampled_clearance = np.abs(surface_points[broad, 2] - heater_surface_points[broad, 2])
            if len(sampled_clearance):
                local_minimum_index = int(np.argmin(sampled_clearance))
                local_minimum = float(sampled_clearance[local_minimum_index])
                if local_minimum < minimum_clearance_mm:
                    minimum_clearance_mm = local_minimum
                    minimum_clearance_location = {
                        "layer": layer,
                        "pose_index": pose_index,
                        "tcp_xyzabc": [float(value) for value in pose],
                    }
            layer_candidates += len(local)
            candidate_points += len(local)
            inside_bounds = np.all((local >= bounds_min - 1e-6) & (local <= bounds_max + 1e-6), axis=1)
            inside = _inside_closed_mesh(local[inside_bounds], heater)
            if np.any(inside):
                hit = local[inside_bounds][inside][0]
                return {
                    "passed": False, "peak_layers": peak_layers, "tested_pose_count": checked + 1,
                    "candidate_surface_points": candidate_points,
                    "sampling_pitch_mm": sample_spacing_mm,
                    "minimum_sampled_clearance_mm": 0.0,
                    "collision": {"layer": layer, "pose_index": pose_index, "tcp_xyzabc": [float(v) for v in pose], "tool_local_point_mm": [float(v) for v in hit]},
                }
            checked += 1
        per_layer.append({"layer": layer, "poses": int(len(poses)), "flat_z_mm": flat_z, "candidate_surface_points": layer_candidates})
    return {
        "passed": True, "peak_layers": peak_layers, "tested_pose_count": checked,
        "candidate_surface_points": candidate_points,
        "sampling_pitch_mm": sample_spacing_mm,
        "minimum_sampled_clearance_mm": None if not np.isfinite(minimum_clearance_mm) else minimum_clearance_mm,
        "minimum_sampled_clearance_location": minimum_clearance_location,
        "layers": per_layer,
    }


def check_peak_surface_collision(npz_path: Path) -> dict[str, object]:
    """Run global coarse screening then refine near the closest sampled TCP."""

    source, target, json_path, stl_path = resolve_surface_collision_inputs(npz_path)
    coarse = _run_mesh_sampling(
        source, target, stl_path, sample_spacing_mm=_COARSE_SAMPLE_SPACING_MM,
    )
    report: dict[str, object] = {
        "ok": True,
        "npz": Path(npz_path).name,
        "surface_json": json_path.name,
        "source_stl": stl_path.name,
        "peak_layers": coarse["peak_layers"],
        "coarse": coarse,
        "coarse_sampling_pitch_mm": _COARSE_SAMPLE_SPACING_MM,
        "refinement_sampling_pitch_mm": _REFINED_SAMPLE_SPACING_MM,
        "refinement_radius_mm": _REFINEMENT_RADIUS_MM,
    }
    if not coarse["passed"]:
        report.update(coarse)
        report["refinement"] = {"performed": False, "reason": "coarse_collision"}
        return report

    location = coarse["minimum_sampled_clearance_location"]
    if not isinstance(location, dict):
        raise ValueError("global coarse pass produced no valid heater/material candidate")
    tcp_xyzabc = location.get("tcp_xyzabc")
    if not isinstance(tcp_xyzabc, list) or len(tcp_xyzabc) != 6:
        raise ValueError("global coarse pass did not retain the closest TCP pose")
    refinement = _run_mesh_sampling(
        source, target, stl_path,
        sample_spacing_mm=_REFINED_SAMPLE_SPACING_MM,
        focus_tcp_xyzabc=tcp_xyzabc,
    )
    report.update(refinement)
    report["tested_pose_count"] = int(coarse["tested_pose_count"]) + int(refinement["tested_pose_count"])
    report["candidate_surface_points"] = int(coarse["candidate_surface_points"]) + int(refinement["candidate_surface_points"])
    report["refinement"] = {"performed": True, **refinement}
    return report
