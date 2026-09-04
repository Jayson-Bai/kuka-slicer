from __future__ import annotations

import numpy as np
import pytest

from kuka_slicer.conformal_lattice import (
    conformal_lattice_preview_payload,
    conformal_lattice_preview_html,
    embed_lattice_layers,
    generate_conformal_lattice_geometry,
    validate_realized_fill_ratio,
)
from tests.test_conformal_lattice_lattice_generator import _inputs
from kuka_slicer.conformal_lattice.layer_embedding import _symmetric_alphas
from kuka_slicer.surface_mapper.progression import LayerProgression


def test_normal_stack_preserves_node_and_edge_topology_at_each_explicit_offset():
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="inset")
    stack = embed_lattice_layers(domain, orientation, geometry, layer_offsets_mm=(-1.0, 0.0, 1.0))

    assert stack.mode == "target_surface_normal_stack"
    assert stack.node_positions_xyz.shape == (3, len(geometry.lattice_nodes_xyz), 3)
    np.testing.assert_allclose(stack.node_positions_xyz[1], geometry.lattice_nodes_xyz)
    assert stack.lattice_edges is geometry.lattice_edges
    assert np.allclose(np.linalg.norm(stack.node_positions_xyz[2] - stack.node_positions_xyz[1], axis=1), 1.0)
    assert stack.report["topology_shared_across_layers"] is True


def test_symmetric_compatibility_mode_requires_flat_reference_and_labels_nonconformal_intermediates():
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="inset")
    flat = geometry.lattice_nodes_xyz + np.asarray([0.0, 0.0, 5.0])
    stack = embed_lattice_layers(
        domain,
        orientation,
        geometry,
        mode="symmetric_shape_morphing",
        symmetric_layer_count=4,
        flat_reference_nodes_xyz=flat,
    )

    np.testing.assert_allclose(stack.node_positions_xyz[0], flat)
    np.testing.assert_allclose(stack.node_positions_xyz[-1], flat)
    np.testing.assert_allclose(stack.node_positions_xyz[1], geometry.lattice_nodes_xyz)
    np.testing.assert_allclose(stack.node_positions_xyz[2], geometry.lattice_nodes_xyz)
    assert stack.report["strict_conformal_claim"].startswith("not_claimed")


@pytest.mark.parametrize(
    ("layer_count", "surface_start_layer"),
    [(1, 0), (2, 0), (3, 0), (4, 0), (9, 2), (10, 3)],
)
def test_symmetric_alphas_are_exactly_the_old_layer_progression(layer_count, surface_start_layer):
    progression = LayerProgression(surface_start_layer, layer_count - 1)

    actual = _symmetric_alphas(layer_count, surface_start_layer=surface_start_layer)
    expected = np.asarray([progression.alpha(index) for index in range(layer_count)])

    np.testing.assert_allclose(actual, expected)
    assert np.flatnonzero(np.isclose(actual, 1.0)).tolist() == list(progression.peak_layers)


def test_symmetric_compatibility_mode_respects_the_requested_start_and_return_layers():
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="inset")
    flat = geometry.lattice_nodes_xyz + np.asarray([0.0, 0.0, 5.0])
    stack = embed_lattice_layers(
        domain,
        orientation,
        geometry,
        mode="symmetric_shape_morphing",
        symmetric_layer_count=10,
        surface_start_layer=3,
        flat_reference_nodes_xyz=flat,
    )

    expected = np.asarray([LayerProgression(3, 9).alpha(index) for index in range(10)])
    np.testing.assert_allclose(stack.layer_offsets_mm, expected)
    np.testing.assert_allclose(stack.node_positions_xyz[3], flat)
    assert stack.report["surface_start_layer"] == 3
    assert stack.report["surface_return_layer"] == 6
    assert stack.report["peak_layer_indices"] == [4, 5]


@pytest.mark.parametrize(
    ("layer_count", "surface_start_layer"),
    [(3, 2), (4, -1), (0, 0)],
)
def test_symmetric_alphas_reject_invalid_legacy_progression_inputs(layer_count, surface_start_layer):
    with pytest.raises(ValueError):
        _symmetric_alphas(layer_count, surface_start_layer=surface_start_layer)


def test_read_only_preview_contains_required_geometry_and_quality_views_without_mutating_geometry():
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="inset")
    validation = validate_realized_fill_ratio(domain, parameterization, fields, phase, geometry, samples_per_triangle_side=5)
    before = geometry.lattice_nodes_xyz.copy()
    payload = conformal_lattice_preview_payload(
        domain, parameterization, fields, orientation, phase, geometry, fill_validation=validation
    )

    assert payload["read_only"] is True
    assert {"surface_3d", "uv", "conformal_distortion", "fill_ratio", "orientation", "defects", "phase", "layers"} <= set(payload)
    assert payload["fill_ratio"]["actual"]["report"]["evaluated_cell_count"] == len(geometry.cell_valence)
    np.testing.assert_allclose(geometry.lattice_nodes_xyz, before)


def test_preview_html_is_independent_read_only_canvas_inspector_with_all_gate7_views():
    html = conformal_lattice_preview_html()

    assert "/api/preview" in html
    assert "三维格栅" in html
    assert "UV 参数域" in html
    assert "共形畸变" in html
    assert "填充率" in html
    assert "方向场" in html
    assert "拓扑缺陷" in html
    assert "层间结构" in html
    assert "只读" in html
