from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from kuka_slicer.conformal_lattice import (
    ExternalScalarMetadata,
    FieldComponent,
    build_orientation_field,
    compose_design_fields,
    constant_driver,
    curvature_driver,
    external_scalar_driver,
    prepare_surface_mesh_domain,
    roi_driver,
)
from kuka_slicer.conformal_lattice.scalar_fields import edge_length_graph


def _grid_domain(surface=lambda _x, _y: 0.0, *, columns: int = 5, rows: int = 4):
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


def _cylinder_domain(*, columns: int = 9, rows: int = 5):
    vertices = []
    for row in range(rows):
        for column in range(columns):
            theta = math.pi * column / (columns - 1)
            vertices.append([10.0 * math.cos(theta), 10.0 * math.sin(theta), float(row)])
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns
            upper_right = upper_left + 1
            faces.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    return prepare_surface_mesh_domain(np.asarray(vertices), np.asarray(faces, dtype=np.int64))


def test_constant_roi_curvature_and_external_drivers_share_normalised_field_contract():
    domain = _grid_domain(lambda x, y: 0.2 * math.sin(x) * math.sin(y), columns=6, rows=6)
    constant = constant_driver(len(domain.vertices), 0.25)
    roi = roi_driver(domain, seed_vertices=[0], radius_mm=2.0)
    curvature = curvature_driver(domain, mode="mean_abs")
    face_values = np.linspace(2.0, 8.0, len(domain.faces))
    metadata = ExternalScalarMetadata(
        source="fea/example.csv",
        units="MPa",
        sha256=hashlib.sha256(face_values.tobytes()).hexdigest(),
    )
    external = external_scalar_driver(domain, face_values, location="face", metadata=metadata)

    assert constant == pytest.approx(np.full(len(domain.vertices), 0.25))
    assert roi.values[0] == pytest.approx(1.0)
    assert np.all((roi.values >= 0.0) & (roi.values <= 1.0))
    assert np.all((curvature.values >= 0.0) & (curvature.values <= 1.0))
    assert external.metadata["units"] == "MPa"
    assert external.values.shape == (len(domain.vertices),)


def test_design_fields_keep_locked_targets_and_limit_adjacent_cell_scale_change():
    domain = _grid_domain(columns=5, rows=4)
    direct_eta = np.linspace(0.12, 0.72, len(domain.vertices))
    locks = np.zeros(len(domain.vertices), dtype=bool)
    locks[0] = True
    result = compose_design_fields(
        domain,
        wall_width_mm=2.0,
        eta_min=0.1,
        eta_max=0.8,
        direct_target_fill_ratio=direct_eta,
        smoothing_length_mm=1.0,
        max_log_size_gradient=0.15,
        locked_vertices=locks,
    )

    assert result.target_fill_ratio[0] == pytest.approx(direct_eta[0])
    assert np.all((result.target_fill_ratio >= 0.1) & (result.target_fill_ratio <= 0.8))
    graph = edge_length_graph(domain).tocoo()
    for left, right, length in zip(graph.row, graph.col, graph.data):
        if left < right:
            delta = abs(math.log(result.target_cell_size_mm[right]) - math.log(result.target_cell_size_mm[left]))
            assert delta <= 0.15 * length + 1e-10
    assert result.report["fill_ratio_note"].startswith("target value")
    assert "target_fill_ratio" in result.heatmap_payload()


def test_weighted_composite_uses_one_interface_and_does_not_treat_driver_as_fill_ratio():
    domain = _grid_domain()
    result = compose_design_fields(
        domain,
        wall_width_mm=2.0,
        eta_min=0.2,
        eta_max=0.6,
        components=[FieldComponent("constant", constant_driver(len(domain.vertices), 0.5), weight=0.0)],
        baseline_logit=0.0,
    )

    assert result.density_driver == pytest.approx(np.zeros(len(domain.vertices)))
    assert result.target_fill_ratio == pytest.approx(np.full(len(domain.vertices), 0.4))
    assert result.report["combination"]["mode"] == "weighted_composite"


def test_global_axis_orientation_is_tangent_unit_rosy6_and_has_serialisable_heatmap():
    domain = _grid_domain()
    field = build_orientation_field(domain, mode="global_axis", global_axis_xyz=np.asarray([1.0, 0.0, 0.0]), smoothing_iterations=3)

    assert np.allclose(np.sum(field.tangent_vectors_xyz * field.vertex_normals_xyz, axis=1), 0.0, atol=1e-10)
    assert np.allclose(np.abs(field.rosy6), 1.0)
    assert field.rosy6 == pytest.approx(np.ones(len(domain.vertices), dtype=np.complex128))
    assert field.heatmap_payload()["report"]["mode"] == "global_axis"


def test_principal_curvature_orientation_is_explicit_about_undefined_directions_and_singularities():
    planar = build_orientation_field(_grid_domain(), mode="principal_curvature")
    cylinder = build_orientation_field(_cylinder_domain(), mode="principal_curvature", smoothing_iterations=2)

    assert planar.principal_direction_defined is not None
    assert not np.any(planar.principal_direction_defined)
    assert any(item.reason == "principal_direction_undefined" for item in planar.singularities)
    assert cylinder.principal_direction_defined is not None
    assert np.any(cylinder.principal_direction_defined)
    assert np.allclose(np.sum(cylinder.tangent_vectors_xyz * cylinder.vertex_normals_xyz, axis=1), 0.0, atol=1e-9)
    assert np.allclose(np.abs(cylinder.rosy6), 1.0)


def test_external_field_requires_provenance_and_locked_gradient_conflicts_are_not_silently_changed():
    domain = _grid_domain(columns=3, rows=2)
    with pytest.raises(ValueError, match="sha256"):
        ExternalScalarMetadata(source="x", units="mm", sha256="not-a-hash")
    locked = np.ones(len(domain.vertices), dtype=bool)
    direct_eta = np.linspace(0.1, 0.8, len(domain.vertices))
    with pytest.raises(ValueError, match="locked vertices violate"):
        compose_design_fields(
            domain,
            wall_width_mm=2.0,
            eta_min=0.1,
            eta_max=0.8,
            direct_target_fill_ratio=direct_eta,
            max_log_size_gradient=0.01,
            locked_vertices=locked,
        )
