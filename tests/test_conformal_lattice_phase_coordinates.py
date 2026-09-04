from __future__ import annotations

import math

import numpy as np
import pytest

from kuka_slicer.conformal_lattice import (
    PhaseAnchor,
    PhaseQuality,
    build_orientation_field,
    compose_design_fields,
    parameterize_lscm,
    prepare_surface_mesh_domain,
    require_valid_phase,
    solve_phase_coordinates,
)
from kuka_slicer.conformal_lattice.phase_coordinates import _target_phase_gradients


def _grid_domain(*, columns: int = 5, rows: int = 4):
    vertices = np.asarray([[float(column), float(row), 0.0] for row in range(rows) for column in range(columns)])
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns
            upper_right = upper_left + 1
            faces.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    return prepare_surface_mesh_domain(vertices, np.asarray(faces, dtype=np.int64))


def _inputs(*, eta: np.ndarray | None = None):
    domain = _grid_domain()
    parameterization = parameterize_lscm(domain)
    fields = compose_design_fields(
        domain,
        wall_width_mm=0.5,
        eta_min=0.1,
        eta_max=0.8,
        direct_target_fill_ratio=eta if eta is not None else np.full(len(domain.vertices), 0.3),
        smoothing_length_mm=0.0,
    )
    orientation = build_orientation_field(domain, mode="global_axis", global_axis_xyz=np.asarray([1.0, 0.0, 0.0]))
    return domain, parameterization, fields, orientation


def test_phase_coordinates_fit_constant_target_and_keep_explicit_anchor():
    domain, parameterization, fields, orientation = _inputs()
    phase = solve_phase_coordinates(
        domain,
        parameterization,
        fields,
        orientation,
        phase_anchor=PhaseAnchor(vertex=0, phi_p=2.0, phi_q=-3.0),
    )

    assert phase.phi_p[0] == pytest.approx(2.0)
    assert phase.phi_q[0] == pytest.approx(-3.0)
    assert phase.quality.summary["rms_residual_per_uv_area"] < 1e-8
    assert phase.quality.flipped_phase_triangle_count == 0
    assert phase.quality.degenerate_phase_triangle_count == 0
    assert not phase.quality.overlapping_phase_face_pairs
    assert np.all(phase.quality.jacobian_det_per_face > 0.0)
    assert np.all(phase.cell_size_uv_per_vertex > 0.0)


def test_higher_fill_ratio_increases_target_phase_gradient_magnitude():
    domain, parameterization, fields, orientation = _inputs(eta=np.linspace(0.2, 0.6, 20))
    scale = np.sqrt(np.mean(parameterization.quality.area_scale_per_face))
    cell_uv = fields.target_cell_size_mm / scale
    p, q = _target_phase_gradients(domain, parameterization.uv, orientation, cell_uv)
    cell_by_face = np.asarray([np.mean(fields.target_cell_size_mm[face]) for face in domain.faces])
    low = int(np.argmax(cell_by_face))
    high = int(np.argmin(cell_by_face))

    assert np.linalg.norm(p[high]) > np.linalg.norm(p[low])
    assert np.linalg.norm(q[high]) > np.linalg.norm(q[low])
    assert math.isclose(np.linalg.norm(p[high]), 1.0 / np.mean(cell_uv[domain.faces[high]]), rel_tol=1e-10)


def test_nonintegrable_variable_density_is_solved_with_an_explicit_residual():
    domain, parameterization, fields, orientation = _inputs(eta=np.linspace(0.2, 0.6, 20))
    phase = solve_phase_coordinates(domain, parameterization, fields, orientation)

    assert phase.quality.summary["rms_residual_per_uv_area"] > 1e-3
    assert phase.quality.summary["max_residual"] >= phase.quality.summary["rms_residual_per_uv_area"]
    assert phase.quality.flipped_phase_triangle_count == 0


def test_invalid_phase_quality_is_a_hard_failure_with_remediation():
    quality = PhaseQuality(
        residual_per_face=np.asarray([0.0]),
        jacobian_det_per_face=np.asarray([-1.0]),
        flipped_phase_triangle_count=1,
        degenerate_phase_triangle_count=0,
        overlapping_phase_face_pairs=(),
        summary={},
    )

    with pytest.raises(ValueError, match="reduce density gradient"):
        require_valid_phase(quality)
