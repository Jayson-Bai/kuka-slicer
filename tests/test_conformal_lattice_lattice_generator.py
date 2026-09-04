from __future__ import annotations

import json

import numpy as np

from kuka_slicer.conformal_lattice import (
    build_orientation_field,
    compose_design_fields,
    generate_conformal_lattice_geometry,
    parameterize_lscm,
    prepare_surface_mesh_domain,
    solve_phase_coordinates,
)


def _inputs(*, columns: int = 8, rows: int = 7):
    vertices = np.asarray([[float(column), float(row), 0.0] for row in range(rows) for column in range(columns)])
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = lower_left + columns
            upper_right = upper_left + 1
            faces.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    domain = prepare_surface_mesh_domain(vertices, np.asarray(faces, dtype=np.int64))
    parameterization = parameterize_lscm(domain)
    fields = compose_design_fields(
        domain,
        wall_width_mm=0.5,
        eta_min=0.1,
        eta_max=0.8,
        direct_target_fill_ratio=np.full(len(domain.vertices), 0.3),
    )
    orientation = build_orientation_field(domain, mode="global_axis", global_axis_xyz=np.asarray([1.0, 0.0, 0.0]))
    phase = solve_phase_coordinates(domain, parameterization, fields, orientation)
    return domain, parameterization, fields, orientation, phase


def test_clip_generates_dual_hex_edges_with_auditable_barycentric_surface_mapping():
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="clip")

    assert len(geometry.lattice_edges) > 0
    assert len(geometry.cell_valence) > 0
    assert np.any(geometry.cell_is_boundary)
    assert np.allclose(np.sum(geometry.barycentric_weights_per_node, axis=1), 1.0)
    for node, face_id, barycentric in zip(
        geometry.lattice_nodes_xyz,
        geometry.source_triangle_id_per_node,
        geometry.barycentric_weights_per_node,
    ):
        np.testing.assert_allclose(node, barycentric @ domain.vertices[domain.faces[face_id]], atol=1e-9)
    parent_counts = np.bincount(geometry.lattice_edge_parent_id)
    assert np.max(parent_counts) >= 2
    assert geometry.report["boundary_clipped_cell_count"] > 0


def test_inset_keeps_only_complete_hexagons_and_npz_carries_geometry_contract(tmp_path):
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(
        domain,
        parameterization,
        fields,
        orientation,
        phase,
        boundary_mode="inset",
        phase_origin=(0.15, -0.2),
        random_seed=12,
        config_metadata={"source": "test"},
    )

    assert len(geometry.cell_valence) > 0
    assert np.all(geometry.cell_valence == 6)
    assert not np.any(geometry.cell_is_boundary)
    output = geometry.save_npz(
        tmp_path / "conformal_lattice_geometry_v1.npz",
        domain=domain,
        parameterization=parameterization,
        design_fields=fields,
        orientation=orientation,
        phase=phase,
    )
    with np.load(output) as saved:
        assert {"lattice_nodes_phase", "lattice_nodes_uv", "lattice_nodes_xyz", "lattice_edges", "cell_offsets", "cell_node_indices", "source_triangle_id_per_node", "barycentric_weights_per_node", "mapping_residual_per_node"} <= set(saved.files)
        metadata = json.loads(str(saved["meta"]))
        assert metadata["format"] == "conformal_lattice_geometry_v1"
        assert metadata["random_seed"] == 12
