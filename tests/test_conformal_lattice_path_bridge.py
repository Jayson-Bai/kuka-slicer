from __future__ import annotations

import json

import numpy as np
import pytest

from kuka_slicer.conformal_lattice import (
    ExtrusionVolumeModel,
    build_conformal_lattice_path_graph,
    embed_lattice_layers,
    generate_conformal_lattice_geometry,
    write_conformal_lattice_external_npz,
)
from tests.test_conformal_lattice_lattice_generator import _inputs


def test_path_bridge_uses_stable_structure_ids_and_actual_3d_edge_lengths():
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="inset")
    model = ExtrusionVolumeModel(bead_cross_section_area_mm2=0.24, e_volume_per_unit_mm3=0.08)
    graph = build_conformal_lattice_path_graph(geometry, model)

    expected_ids = (geometry.lattice_edge_parent_id.astype(np.int64) << 32) | geometry.lattice_edge_segment_id
    np.testing.assert_array_equal(graph.edge_ids, np.sort(expected_ids))
    assert len(np.unique(graph.edge_ids)) == len(graph.edge_ids)
    assert graph.layer_node_positions_xyz.shape[0] == 1
    assert np.all(graph.edge_length_mm > 0.0)
    np.testing.assert_allclose(
        graph.cumulative_extrusion_e[..., 1],
        graph.edge_length_mm * 0.24 / 0.08,
    )
    assert graph.metadata["quality"]["all_edge_paths_are_two_points"] is True
    assert graph.metadata["solver_seed"] == 0
    assert graph.report["legacy_surface_mapper_used"] is False
    assert graph.report["legacy_honeycomb_pathing_used"] is False


def test_path_bridge_exports_existing_npz_contract_with_edge_path_provenance(tmp_path):
    domain, parameterization, fields, orientation, phase = _inputs()
    geometry = generate_conformal_lattice_geometry(domain, parameterization, fields, orientation, phase, boundary_mode="inset")
    stack = embed_lattice_layers(domain, orientation, geometry, layer_offsets_mm=(0.0, 0.4))
    graph = build_conformal_lattice_path_graph(
        geometry,
        ExtrusionVolumeModel(bead_cross_section_area_mm2=0.2, e_volume_per_unit_mm3=0.1),
        layer_embedding=stack,
        config_metadata={"test_case": "edge_paths"},
    )
    output = write_conformal_lattice_external_npz(graph, tmp_path / "conformal_paths.npz")

    with np.load(output, allow_pickle=False) as archive:
        assert {"meta", "layer_0000_R", "layer_0000_R_E", "layer_0000_F", "layer_0000_T", "layer_0001_R", "layer_0001_R_E", "layer_0001_F", "layer_0001_T"} <= set(archive.files)
        assert archive["layer_0000_R"].shape == (len(graph.edge_ids), 2, 3)
        np.testing.assert_allclose(archive["layer_0000_R_E"][:, 0], 0.0)
        np.testing.assert_allclose(archive["layer_0000_R_E"][:, 1], graph.cumulative_extrusion_e[0, :, 1])
        metadata = json.loads(str(archive["meta"]))
    assert metadata["format"] == "external_layer_paths_v1"
    bridge = metadata["conformal_lattice_path_bridge"]
    assert bridge["edge_ids_by_layer"]["0"] == graph.edge_ids.tolist()
    assert bridge["trail_partition_status"].startswith("not_planned")
    assert bridge["core_handoff"].endswith("final XYZABC")
    assert metadata["extrusion_compensation"]["requires_xy_preservation"] is False


def test_path_bridge_rejects_an_implicit_or_invalid_e_volume_conversion():
    with pytest.raises(ValueError, match="e_volume_per_unit_mm3"):
        ExtrusionVolumeModel(bead_cross_section_area_mm2=0.2, e_volume_per_unit_mm3=0.0)
