from __future__ import annotations

import json

import numpy as np
import pytest

from kuka_slicer.conformal_lattice import (
    ExtrusionVolumeModel,
    load_conformal_lattice_spec,
    run_conformal_lattice_pipeline,
    write_conformal_lattice_outputs,
)
from kuka_slicer.conformal_lattice.contracts import double_sine_source_sha256
from kuka_slicer.conformal_lattice.fiber_reinforcement import (
    MixedWallFiberSettings,
    apply_mixed_wall_fiber_strategy,
    reserve_fiber_layer_interfaces,
)
from kuka_slicer.surface_preview.server import conformal_lattice_config_payload


def _spec(*, surface_start_layer: int = 0):
    source = {
        "provider": "double_sine",
        "source_file": "generated://pipeline-test-double-sine",
        "domain": "outer_boundary_only",
        "reference_stl": {
            "file_name": "honeycomb.stl",
            "sha256": "a" * 64,
            "build_axis": "z",
            "xy_bounds_mm": [0.0, 0.0, 8.0, 7.0],
        },
        "double_sine": {
            "type": "double_sine_product",
            "amplitude_mm": 0.2,
            "wavelength_x_mm": 30.0,
            "wavelength_y_mm": 35.0,
            "phase_x_rad": 0.0,
            "phase_y_rad": 0.0,
            "z_reference_mm": 0.0,
            "xy_bounds_mm": [0.0, 0.0, 8.0, 7.0],
            "samples": [8, 7],
        },
    }
    source["sha256"] = double_sine_source_sha256(source)
    return load_conformal_lattice_spec(
        {
            "format": "conformal_lattice_spec_v1",
            "units": "mm",
            "source_surface": source,
            "parameterization": {
                "method": "lscm",
                "anchor_strategy": "farthest_boundary_pair",
                "seam_strategy": "none",
            },
            "lattice": {
                "family": "triangular_dual_hex",
                "wall_width_mm": 0.5,
                "base_cell_size_mm": 2.0,
                "boundary_mode": "inset",
                "phase_origin": [0.0, 0.0],
            },
            "fill_field": {"mode": "fixed_cell_size", "drivers": []},
            "orientation_field": {"mode": "global_axis", "angle_deg": 0.0, "constraints": []},
            "layer_embedding": {
                "mode": "symmetric_shape_morphing",
                "transition": "smoothstep",
                "surface_start_layer": surface_start_layer,
            },
            "quality_limits": {},
            "random_seed": 0,
        }
    )


@pytest.mark.parametrize("cell_size_mm", [3.0, 4.0])
def test_pipeline_aligns_a_y_directed_wall_to_the_automatic_length_midplane(cell_size_mm):
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["20"],
            "part_width_mm": ["20"],
            "part_height_mm": ["2"],
            "amplitude_mm": ["0"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": [str(cell_size_mm)],
            "surface_start_layer": ["0"],
            "samples_x": ["9"],
            "samples_y": ["9"],
            "align_load_line": ["true"],
            "phase_origin_x_mm": ["9"],
            "phase_origin_y_mm": ["-4"],
        }
    )

    run = run_conformal_lattice_pipeline(config)
    centre_index = np.flatnonzero(
        np.isclose(run.domain.vertices[:, 0], 10.0) & np.isclose(run.domain.vertices[:, 1], 10.0)
    )
    assert len(centre_index) == 1
    target_phase = np.asarray(
        [run.phase.phi_p[centre_index[0]], run.phase.phi_q[centre_index[0]]], dtype=np.float64
    )
    alignment = run.geometry.metadata["load_line_alignment"]

    assert alignment["load_center_xy_mm"] == [10.0, 10.0]
    assert np.asarray(run.geometry.metadata["phase_origin"]) == pytest.approx(target_phase - [0.5, 0.0])
    assert np.asarray(alignment["resolved_phase_origin"]) == pytest.approx(target_phase - [0.5, 0.0])
    assert _phase_point_lies_on_an_edge(target_phase, run.geometry.lattice_nodes_phase, run.geometry.lattice_edges)


def _phase_point_lies_on_an_edge(point, nodes, edges) -> bool:
    endpoints = nodes[edges]
    starts = endpoints[:, 0, :]
    directions = endpoints[:, 1, :] - starts
    offsets = point - starts
    cross = directions[:, 0] * offsets[:, 1] - directions[:, 1] * offsets[:, 0]
    dot = np.einsum("ij,ij->i", offsets, point - endpoints[:, 1, :])
    return bool(np.any((np.abs(cross) < 1e-8) & (dot <= 1e-8)))


def test_pipeline_skips_expensive_fill_diagnostic_for_production_path_export(tmp_path):
    run = run_conformal_lattice_pipeline(
        _spec(),
        logical_layer_count=4,
        extrusion=ExtrusionVolumeModel(
            bead_cross_section_area_mm2=0.2,
            e_volume_per_unit_mm3=0.1,
            preview_line_width_mm=0.6,
        ),
        fill_samples_per_triangle_side=3,
    )

    assert run.design_fields.target_cell_size_mm == pytest.approx(
        np.full(len(run.domain.vertices), 2.0)
    )
    assert run.layer_embedding.report["alpha_by_layer"] == [0.0, 1.0, 1.0, 0.0]
    assert run.path_graph is not None
    assert run.report["path_npz_export_available"] is True
    assert run.fill_validation is None
    assert run.report["fill_ratio"]["status"] == "skipped"
    assert run.preview_payload()["read_only"] is True
    main_preview = run.main_preview_payload(planning_line_width_mm=0.6)
    assert main_preview["preview_source"] == "conformal_lattice_external_source_job"
    assert main_preview["geometry_mode"] == "surface_3d"
    assert main_preview["line_widths"]["resin"] == pytest.approx(0.6)
    assert main_preview["conformal_lattice"]["uses_existing_main_canvas"] is True
    assert len(main_preview["layers"]) == 4
    assert all(0 < len(layer["resin_paths"]) < len(run.path_graph.edge_ids) for layer in main_preview["layers"])

    outputs = write_conformal_lattice_outputs(run, tmp_path)
    assert set(outputs) == {"geometry", "paths"}
    with np.load(outputs["geometry"], allow_pickle=False) as archive:
        assert archive["target_cell_size_mm_per_vertex"] == pytest.approx(2.0)
        geometry_metadata = json.loads(str(archive["meta"]))
    assert geometry_metadata["fill_ratio_validation"] is None
    with np.load(outputs["paths"], allow_pickle=False) as archive:
        metadata = json.loads(str(archive["meta"]))
    assert metadata["format"] == "external_layer_paths_v1"
    assert metadata["conformal_lattice_path_bridge"]["trail_partition_status"] == "planned_from_conformal_structural_graph"
    from kuka_slicer.ui_server import _preview_payload_from_source_npz

    loaded_preview = _preview_payload_from_source_npz(outputs["paths"].read_bytes(), outputs["paths"].name)
    assert loaded_preview["preview_source"] == "conformal_lattice_external_source_npz"
    assert loaded_preview["line_widths"]["resin"] == pytest.approx(0.6)
    assert loaded_preview["conformal_lattice"]["uses_existing_main_canvas"] is True


def test_auto_boundary_phase_policy_moves_the_lattice_without_changing_legacy_specs():
    automatic = _spec()
    automatic.lattice["boundary_phase_policy"] = "auto_avoid_outer_boundary_coincidence"

    automatic_run = run_conformal_lattice_pipeline(automatic, logical_layer_count=4)
    automatic_report = automatic_run.geometry.metadata["config"]["boundary_phase"]

    assert automatic_report["policy"] == "auto_avoid_outer_boundary_coincidence"
    assert automatic_report["requested_phase_origin"] == [0.0, 0.0]
    assert automatic_report["effective_phase_origin"] != [0.0, 0.0]
    assert automatic_run.geometry.metadata["phase_origin"] == pytest.approx(
        automatic_report["effective_phase_origin"]
    )

    legacy_run = run_conformal_lattice_pipeline(_spec(), logical_layer_count=4)
    legacy_report = legacy_run.geometry.metadata["config"]["boundary_phase"]
    assert legacy_report == {
        "policy": "manual",
        "requested_phase_origin": [0.0, 0.0],
        "effective_phase_origin": [0.0, 0.0],
    }


@pytest.mark.parametrize("cell_size_mm", [3.0, 4.0])
def test_length_midplane_alignment_places_a_y_directed_wall_at_the_part_centre(cell_size_mm):
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["20"],
            "part_width_mm": ["20"],
            "part_height_mm": ["2"],
            "amplitude_mm": ["0"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": [str(cell_size_mm)],
            "surface_start_layer": ["0"],
            "samples_x": ["9"],
            "samples_y": ["9"],
            "align_load_line": ["true"],
        }
    )

    run = run_conformal_lattice_pipeline(config)
    centre_index = np.flatnonzero(
        np.isclose(run.domain.vertices[:, 0], 10.0) & np.isclose(run.domain.vertices[:, 1], 10.0)
    )
    assert len(centre_index) == 1
    target_phase = np.asarray(
        [run.phase.phi_p[centre_index[0]], run.phase.phi_q[centre_index[0]]], dtype=np.float64
    )
    alignment = run.geometry.metadata["config"]["load_line_alignment"]

    assert alignment["load_center_xy_mm"] == [10.0, 10.0]
    assert np.asarray(run.geometry.metadata["phase_origin"]) == pytest.approx(target_phase - [0.5, 0.0])
    assert _phase_point_lies_on_an_edge(target_phase, run.geometry.lattice_nodes_phase, run.geometry.lattice_edges)


def test_pipeline_applies_explicit_x_y_honeycomb_feature_targets_independently():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["20"],
            "part_width_mm": ["20"],
            "part_height_mm": ["2"],
            "amplitude_mm": ["0"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["4"],
            "surface_start_layer": ["0"],
            "samples_x": ["9"],
            "samples_y": ["9"],
            "honeycomb_align_x": ["true"],
            "honeycomb_align_x_mm": ["10"],
            "honeycomb_align_y": ["true"],
            "honeycomb_align_y_mm": ["10"],
        }
    )

    run = run_conformal_lattice_pipeline(config)
    alignment = run.geometry.metadata["config"]["load_line_alignment"]
    target_index = np.flatnonzero(
        np.isclose(run.domain.vertices[:, 0], 10.0) & np.isclose(run.domain.vertices[:, 1], 10.0)
    )
    assert len(target_index) == 1
    target_phase = np.asarray(
        [run.phase.phi_p[target_index[0]], run.phase.phi_q[target_index[0]]], dtype=np.float64
    )

    assert alignment["mode"] == "honeycomb_center_features"
    assert alignment["align_x"] is True
    assert alignment["align_y"] is True
    assert alignment["target_xy_mm"] == [10.0, 10.0]
    assert np.asarray(run.geometry.metadata["phase_origin"]) == pytest.approx(
        target_phase - [0.5, 0.4330127018922193]
    )


def test_pipeline_interprets_base_cell_size_as_the_true_hexagon_edge_length():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["40"],
            "part_width_mm": ["40"],
            "part_height_mm": ["1"],
            "amplitude_mm": ["0"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["5"],
            "surface_start_layer": ["0"],
            "samples_x": ["17"],
            "samples_y": ["17"],
            "boundary_mode": ["inset"],
        }
    )

    run = run_conformal_lattice_pipeline(config)
    geometry = run.geometry
    side_lengths = []
    for start, end, is_boundary in zip(
        geometry.cell_offsets[:-1], geometry.cell_offsets[1:], geometry.cell_is_boundary
    ):
        if is_boundary or end - start != 6:
            continue
        points = geometry.lattice_nodes_xyz[geometry.cell_node_indices[start:end], :2]
        side_lengths.extend(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))

    assert side_lengths
    assert np.median(side_lengths) == pytest.approx(5.0, abs=1e-6)


def _phase_point_lies_on_an_edge(point, nodes, edges) -> bool:
    endpoints = nodes[edges]
    starts = endpoints[:, 0, :]
    directions = endpoints[:, 1, :] - starts
    offsets = point - starts
    cross = directions[:, 0] * offsets[:, 1] - directions[:, 1] * offsets[:, 0]
    dot = np.einsum("ij,ij->i", offsets, point - endpoints[:, 1, :])
    return bool(np.any((np.abs(cross) < 1e-8) & (dot <= 1e-8)))


def test_pipeline_runs_actual_fill_diagnostic_only_when_explicitly_requested():
    run = run_conformal_lattice_pipeline(
        _spec(),
        logical_layer_count=4,
        validate_fill_ratio=True,
        fill_samples_per_triangle_side=3,
    )

    assert run.fill_validation is not None
    assert run.report["fill_ratio"]["evaluated_cell_count"] > 0


def test_pipeline_keeps_path_export_disabled_without_process_e_conversion(tmp_path):
    run = run_conformal_lattice_pipeline(_spec(), logical_layer_count=4, fill_samples_per_triangle_side=3)

    assert run.path_graph is None
    assert run.report["path_npz_export_available"] is False
    assert "bead_cross_section_area_mm2" in run.report["path_npz_requirement"]
    outputs = write_conformal_lattice_outputs(run, tmp_path)
    assert set(outputs) == {"geometry"}
    assert not (tmp_path / "external_layer_paths_v1.npz").exists()
    with pytest.raises(ValueError, match="path graph"):
        run.main_preview_payload(planning_line_width_mm=0.6)


@pytest.mark.parametrize(
    ("logical_layer_count", "surface_start_layer", "error"),
    [(0, 0, "logical_layer_count"), (4, 3, "surface_start_layer")],
)
def test_pipeline_rejects_invalid_logical_layer_progression(logical_layer_count, surface_start_layer, error):
    with pytest.raises(ValueError, match=error):
        run_conformal_lattice_pipeline(_spec(surface_start_layer=surface_start_layer), logical_layer_count=logical_layer_count)


def test_rectangular_physical_part_derives_monotonic_layer_centres_and_requires_the_matching_count():
    spec = conformal_lattice_config_payload(
        {
            "part_length_mm": ["8"], "part_width_mm": ["7"], "part_height_mm": ["10"],
            "samples_x": ["8"], "samples_y": ["7"],
            "wall_width_mm": ["2"], "base_cell_size_mm": ["5"], "surface_start_layer": ["1"],
        }
    )

    run = run_conformal_lattice_pipeline(
        spec,
        physical_layer_height_mm=2.0,
        extrusion=ExtrusionVolumeModel(0.2, 0.1),
    )

    assert run.layer_embedding.report["base_z_by_layer_mm"] == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0])
    assert run.layer_embedding.report["surface_start_layer"] == 1
    assert run.path_graph is not None
    assert run.path_graph.outer_boundary_paths_xyz is not None
    assert run.path_graph.outer_boundary_paths_xyz.shape == (5, 27, 3)
    np.testing.assert_allclose(run.path_graph.outer_boundary_paths_xyz[:, 0], run.path_graph.outer_boundary_paths_xyz[:, -1])
    job = run.path_graph.to_external_source_job()
    assert job.meta["path_roles"]["R"]["0"][0] == "conformal_outer_boundary"
    assert job.material_paths[0].paths[0].shape == (27, 6)
    assert np.allclose(job.material_paths[0].paths[0][:, 3:], 0.0)
    assert np.linalg.norm(job.material_paths[2].paths[0][:, 3:]) > 1e-3
    with pytest.raises(ValueError, match="logical_layer_count"):
        run_conformal_lattice_pipeline(spec, logical_layer_count=4, physical_layer_height_mm=2.0)


def test_symmetric_grips_are_split_on_every_layer_and_reuse_x_one_stroke_zigzag():
    spec = conformal_lattice_config_payload(
        {
            "part_length_mm": ["40"],
            "part_width_mm": ["20"],
            "part_height_mm": ["4"],
            "specimen_variant": ["tensile"],
            "grip_end_length_mm": ["8"],
            "amplitude_mm": ["1.5"],
            "wavelength_x_mm": ["24"],
            "wavelength_y_mm": ["30"],
            "surface_start_layer": ["0"],
            "samples_x": ["21"],
            "samples_y": ["11"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["5"],
        }
    )

    run = run_conformal_lattice_pipeline(
        spec,
        physical_layer_height_mm=1.0,
        extrusion=ExtrusionVolumeModel(bead_cross_section_area_mm2=0.2, e_volume_per_unit_mm3=0.1),
    )
    assert run.domain.report["global_xy_bounds_mm"] == [0.0, 0.0, 40.0, 20.0]
    assert run.domain.report["generated_xy_bounds_mm"] == [8.0, 0.0, 32.0, 20.0]
    assert np.any(np.isclose(run.geometry.lattice_nodes_xyz[:, 0], 8.0))
    assert np.any(np.isclose(run.geometry.lattice_nodes_xyz[:, 0], 32.0))
    assert run.geometry.metadata["config"]["partition_connection"]["mode"] == "shared_separator_wall_endpoints"
    job = run.path_graph.to_external_source_job()  # type: ignore[union-attr]
    expected_prefix = [
        "conformal_outer_boundary",
        "conformal_partition_wall",
        "conformal_partition_wall",
        "conformal_grip_zigzag_x_one_stroke",
        "conformal_grip_zigzag_x_one_stroke",
    ]
    for layer, group in enumerate(job.material_paths):
        roles = job.meta["path_roles"]["R"][str(layer)]
        assert roles[:5] == expected_prefix
        assert "conformal_honeycomb_macro_partition" in roles[5:]
        assert roles.count("conformal_grip_zigzag_x_one_stroke") == 2
        left = group.paths[3]
        right = group.paths[4]
        assert np.max(left[:, 0]) <= 6.0 + 1e-6
        assert np.min(right[:, 0]) >= 34.0 - 1e-6
        left_delta = np.diff(left[:, :2], axis=0)
        assert np.median(np.abs(left_delta[:, 0])) > np.median(np.abs(left_delta[:, 1]))
        left_e = group.extrusion[3]
        assert np.diff(left_e) == pytest.approx(np.linalg.norm(np.diff(left[:, :3], axis=0), axis=1) * 2.0)


def test_bending_variant_uses_the_full_rectangle_without_grip_or_partition_paths():
    spec = conformal_lattice_config_payload(
        {
            "part_length_mm": ["40"],
            "part_width_mm": ["20"],
            "part_height_mm": ["4"],
            "specimen_variant": ["bending"],
            "grip_end_length_mm": ["8"],
            "amplitude_mm": ["1.5"],
            "wavelength_x_mm": ["24"],
            "wavelength_y_mm": ["30"],
            "surface_start_layer": ["0"],
            "samples_x": ["21"],
            "samples_y": ["11"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["5"],
        }
    )

    run = run_conformal_lattice_pipeline(
        spec,
        physical_layer_height_mm=1.0,
        extrusion=ExtrusionVolumeModel(bead_cross_section_area_mm2=0.2, e_volume_per_unit_mm3=0.1),
    )

    assert "symmetric_grip_end_length_mm" not in spec["part"]
    assert run.domain.report["generated_xy_bounds_mm"] == [0.0, 0.0, 40.0, 20.0]
    assert run.continuous_course_plan is not None
    assert run.continuous_course_plan.course_bounds_mm == pytest.approx((0.0, 0.0, 40.0, 20.0))
    job = run.path_graph.to_external_source_job()  # type: ignore[union-attr]
    for layer, roles in job.meta["path_roles"]["R"].items():
        assert roles.count("conformal_outer_boundary") == 1, layer
        assert "conformal_partition_wall" not in roles
        assert "conformal_grip_zigzag_x_one_stroke" not in roles


def test_design_json_without_process_fiber_settings_leaves_the_resin_graph_unchanged():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["50"],
            "part_height_mm": ["10"],
            "grip_end_length_mm": ["25"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["10"],
            "orientation_angle_deg": ["90"],
            "honeycomb_align_x": ["false"],
            "honeycomb_align_y": ["false"],
            "surface_parameter_mode": ["tensile_centered_wave_count"],
            "amplitude_mm": ["1.5"],
            "wave_count_x": ["1.5"],
            "wave_count_y": ["1.5"],
            "surface_start_layer": ["3"],
            "surface_start_layer_semantics": ["first_nonzero_curvature_physical"],
        }
    )
    run = run_conformal_lattice_pipeline(
        config,
        physical_layer_height_mm=0.5,
        extrusion=ExtrusionVolumeModel(
            bead_cross_section_area_mm2=1.0,
            e_volume_per_unit_mm3=1.0,
            preview_line_width_mm=2.0,
        ),
    )
    assert run.path_graph is not None
    source_job = run.path_graph.to_external_source_job()
    before_last_resin = next(
        group.paths[0].copy()
        for group in source_job.material_paths
        if group.material == "R" and group.layer_index == 19
    )

    result = reserve_fiber_layer_interfaces(
        spec=run.spec,
        layer_embedding=run.layer_embedding,
    )

    fiber_groups = [group for group in source_job.material_paths if group.material == "F"]
    assert result.resin_layer_indices == ()
    assert result.reserved is False
    assert result.enabled is False
    assert result.total_path_count == 0
    assert result.nominal_final_height_mm == pytest.approx(10.0)
    assert fiber_groups == []
    assert result.report["path_generation"] == "disabled_pending_replacement"
    assert result.report["automatic_resin_z_raise"] is False
    assert result.report["core_material_paths"] == "none"
    after_last_resin = next(
        group.paths[0]
        for group in source_job.material_paths
        if group.material == "R" and group.layer_index == 19
    )
    np.testing.assert_allclose(after_last_resin, before_last_resin)


def test_main_ui_mixed_wall_strategy_emits_fiber_and_raises_later_resin_layers():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["50"],
            "part_height_mm": ["10"],
            "grip_end_length_mm": ["25"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["10"],
            "orientation_angle_deg": ["90"],
            "honeycomb_align_x": ["false"],
            "honeycomb_align_y": ["false"],
            "surface_parameter_mode": ["tensile_centered_wave_count"],
            "amplitude_mm": ["1.5"],
            "wave_count_x": ["1.5"],
            "wave_count_y": ["1.5"],
            "surface_start_layer": ["3"],
            "surface_start_layer_semantics": ["first_nonzero_curvature_physical"],
        }
    )
    run = run_conformal_lattice_pipeline(
        config,
        physical_layer_height_mm=0.5,
        extrusion=ExtrusionVolumeModel(
            bead_cross_section_area_mm2=1.0,
            e_volume_per_unit_mm3=1.0,
            preview_line_width_mm=2.0,
        ),
    )
    assert run.path_graph is not None
    source_job = run.path_graph.to_external_source_job()
    before_later_resin = next(
        group.paths[0].copy()
        for group in source_job.material_paths
        if group.material == "R" and group.layer_index == 3
    )

    result = apply_mixed_wall_fiber_strategy(
        source_job=source_job,
        graph=run.path_graph,
        settings=MixedWallFiberSettings(True, 2, 19),
        fiber_layer_height_mm=0.1,
        fiber_e_per_mm=1.0,
    )

    fiber_groups = [group for group in source_job.material_paths if group.material == "F"]
    after_later_resin = next(
        group.paths[0]
        for group in source_job.material_paths
        if group.material == "R" and group.layer_index == 3
    )
    assert result.enabled is True
    assert result.report["primary_axis_source"] == "automatic_longest_planar_span"
    assert result.total_path_count > 0
    assert fiber_groups
    assert all(path.shape[1] == 6 for group in fiber_groups for path in group.paths)
    assert np.all(after_later_resin[:, 2] > before_later_resin[:, 2])
