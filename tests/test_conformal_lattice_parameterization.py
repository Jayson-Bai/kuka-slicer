from __future__ import annotations

import math

import numpy as np
import pytest

from kuka_slicer.conformal_lattice import (
    build_double_sine_surface_domain,
    evaluate_conformal_quality,
    farthest_boundary_anchors,
    parameterize_lscm,
    parameterize_spec_lscm,
    prepare_surface_mesh_domain,
)
from kuka_slicer.conformal_lattice.contracts import double_sine_source_sha256, load_conformal_lattice_spec
from kuka_slicer.conformal_lattice.distortion import require_valid_uv


def _grid_domain(surface, *, columns: int = 5, rows: int = 4):
    vertices = np.asarray(
        [[float(column), float(row), float(surface(column, row))] for row in range(rows) for column in range(columns)],
        dtype=float,
    )
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns
            upper_right = upper_left + 1
            faces.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    return prepare_surface_mesh_domain(vertices, np.asarray(faces, dtype=np.int64))


def _cylinder_domain(*, columns: int = 9, rows: int = 4):
    radius = 10.0
    vertices = []
    for row in range(rows):
        for column in range(columns):
            theta = math.pi * column / (columns - 1)
            vertices.append([radius * math.cos(theta), radius * math.sin(theta), float(row)])
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns
            upper_right = upper_left + 1
            faces.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    return prepare_surface_mesh_domain(np.asarray(vertices), np.asarray(faces, dtype=np.int64))


def test_lscm_parameterizes_a_planar_patch_without_distortion_or_flips():
    domain = _grid_domain(lambda _x, _y: 0.0)

    result = parameterize_lscm(domain, max_conformal_ratio=1.0 + 1e-9)

    assert result.anchor_strategy == "farthest_boundary_pair"
    assert result.anchor_vertices == farthest_boundary_anchors(domain)
    assert result.quality.flipped_face_count == 0
    assert result.quality.degenerate_uv_face_count == 0
    assert result.quality.overlapping_uv_face_pairs == ()
    assert result.quality.summary["conformal_ratio_max"] == pytest.approx(1.0, abs=1e-9)
    assert result.quality.summary["angle_error_deg_max"] == pytest.approx(0.0, abs=1e-7)
    metadata = result.metadata()
    assert metadata["anchors"][0]["xyz"] == pytest.approx(domain.vertices[result.anchor_vertices[0]])


def test_lscm_parameterizes_an_open_cylinder_as_a_nearly_isometric_patch():
    result = parameterize_lscm(_cylinder_domain(), max_conformal_ratio=1.0 + 1e-7)

    assert result.quality.flipped_face_count == 0
    assert result.quality.summary["conformal_ratio_max"] == pytest.approx(1.0, abs=1e-7)
    assert result.quality.summary["angle_error_deg_max"] == pytest.approx(0.0, abs=1e-6)


def test_lscm_accepts_user_boundary_anchors_and_reports_an_approximate_double_sine_map():
    domain = _grid_domain(lambda x, y: 0.25 * math.sin(x / 2.0) * math.sin(y / 2.0), columns=7, rows=6)
    boundary = domain.boundary_loops[0]

    result = parameterize_lscm(domain, anchors=(int(boundary[0]), int(boundary[len(boundary) // 2])))

    assert result.anchor_strategy == "user"
    assert result.quality.flipped_face_count == 0
    assert result.quality.degenerate_uv_face_count == 0
    assert result.quality.overlapping_uv_face_pairs == ()
    assert result.quality.summary["conformal_ratio_max"] >= 1.0
    assert result.solver["backend"] == "scipy.sparse.linalg.lsmr"


def test_lscm_parameterizes_the_contract_double_sine_surface_without_topology_failure():
    source = {
        "provider": "double_sine",
        "source_file": "generated://gate-2-double-sine",
        "sha256": "0" * 64,
        "domain": "outer_boundary_only",
        "double_sine": {
            "amplitude_mm": 0.8,
            "wavelength_x_mm": 20.0,
            "wavelength_y_mm": 30.0,
            "phase_x_rad": 0.0,
            "phase_y_rad": 0.0,
            "z_reference_mm": 0.0,
            "xy_bounds_mm": [0.0, 0.0, 20.0, 30.0],
            "samples": [7, 6],
        },
    }
    source["sha256"] = double_sine_source_sha256(source)
    spec = load_conformal_lattice_spec(
        {
            "format": "conformal_lattice_spec_v1",
            "units": "mm",
            "source_surface": source,
            "parameterization": {"method": "lscm", "anchor_strategy": "farthest_boundary_pair", "seam_strategy": "none"},
            "lattice": {"family": "triangular_dual_hex", "wall_width_mm": 2.0, "base_cell_size_mm": 5.0, "boundary_mode": "clip", "phase_origin": [0.0, 0.0]},
            "fill_field": {"mode": "weighted_composite", "minimum": None, "maximum": None, "smoothing_length_mm": None, "drivers": []},
            "orientation_field": {"mode": "global_axis", "angle_deg": 0.0, "constraints": []},
            "layer_embedding": {"mode": "target_surface_normal_stack"},
            "quality_limits": {},
            "random_seed": 0,
        }
    )

    result = parameterize_spec_lscm(spec, build_double_sine_surface_domain(spec))

    assert result.quality.flipped_face_count == 0
    assert result.quality.degenerate_uv_face_count == 0
    assert result.quality.overlapping_uv_face_pairs == ()
    assert math.isfinite(float(result.quality.summary["conformal_ratio_max"]))


def test_spec_declares_user_anchors_and_quality_limit_for_lscm():
    domain = _grid_domain(lambda _x, _y: 0.0)
    boundary = domain.boundary_loops[0]
    spec = load_conformal_lattice_spec(
        {
            "format": "conformal_lattice_spec_v1",
            "units": "mm",
            "source_surface": {
                "provider": "triangle_mesh",
                "source_file": "patch.npz",
                "sha256": "0" * 64,
                "domain": "outer_boundary_only",
            },
            "parameterization": {"method": "lscm", "anchor_strategy": "user", "anchors": [int(boundary[0]), int(boundary[len(boundary) // 2])], "seam_strategy": "none"},
            "lattice": {"family": "triangular_dual_hex", "wall_width_mm": 2.0, "base_cell_size_mm": 5.0, "boundary_mode": "clip", "phase_origin": [0.0, 0.0]},
            "fill_field": {"mode": "weighted_composite", "minimum": None, "maximum": None, "smoothing_length_mm": None, "drivers": []},
            "orientation_field": {"mode": "global_axis", "angle_deg": 0.0, "constraints": []},
            "layer_embedding": {"mode": "target_surface_normal_stack"},
            "quality_limits": {"max_conformal_ratio": 1.0 + 1e-9},
            "random_seed": 0,
        }
    )

    result = parameterize_spec_lscm(spec, domain)

    assert result.anchor_strategy == "user"
    assert "scipy_version" in result.solver


def test_lscm_rejects_closed_surface_without_an_explicit_seam():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    domain = prepare_surface_mesh_domain(vertices, faces)

    with pytest.raises(ValueError, match="exactly one boundary loop"):
        parameterize_lscm(domain)


def test_quality_rejects_nonadjacent_uv_overlap_as_a_hard_failure():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]], dtype=float)
    domain = prepare_surface_mesh_domain(vertices, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64))
    overlapping_uv = np.asarray([[0, 0], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]], dtype=float)

    quality = evaluate_conformal_quality(domain, overlapping_uv)

    assert quality.overlapping_uv_face_pairs == ((0, 1),)
    with pytest.raises(ValueError, match="overlaps"):
        require_valid_uv(quality)
