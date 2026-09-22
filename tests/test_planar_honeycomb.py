from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kuka_slicer.conformal_lattice.contracts import load_conformal_lattice_spec
from kuka_slicer.conformal_lattice.brim import ConformalBrimSettings, apply_conformal_brim
from kuka_slicer.conformal_lattice.path_bridge import ExtrusionVolumeModel
from kuka_slicer.conformal_lattice.pipeline import run_conformal_lattice_pipeline
from kuka_slicer.surface_preview.server import (
    conformal_lattice_config_payload,
    planar_lattice_config_payload,
    surface_preview_html,
)
from kuka_slicer.ui_server import (
    _SlicerUiHandler,
    _conformal_spec_ui_summary,
    _index_html,
)


def _planar_config() -> dict[str, object]:
    return planar_lattice_config_payload(
        {
            "part_length_mm": ["10"],
            "part_width_mm": ["8"],
            "part_height_mm": ["1"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["3"],
            # Deliberately invalid curve-only values prove that the flat export
            # neither validates nor serialises the double-sine controls.
            "wavelength_x_mm": ["0"],
            "samples_x": ["1"],
        }
    )


def test_planar_export_keeps_lattice_and_part_but_omits_double_sine_parameters():
    config = _planar_config()

    assert config["format"] == "conformal_lattice_spec_v1"
    assert config["source_surface"]["provider"] == "planar"
    assert config["source_surface"]["xy_bounds_mm"] == [0.0, 0.0, 10.0, 8.0]
    assert "double_sine" not in json.dumps(config)
    assert config["layer_embedding"] == {"mode": "planar_stack"}
    assert config["part"] == {
        "boundary": "rectangle",
        "length_mm": 10.0,
        "width_mm": 8.0,
        "final_height_mm": 1.0,
    }
    assert config["lattice"]["base_cell_size_mm"] == 3.0
    spec = load_conformal_lattice_spec(config)
    assert spec.source_provider == "planar"


def test_planar_pipeline_reuses_lattice_paths_with_flat_layers_and_flat_orientation():
    run = run_conformal_lattice_pipeline(
        _planar_config(),
        physical_layer_height_mm=0.25,
        extrusion=ExtrusionVolumeModel(0.5, 0.5),
    )

    assert run.domain.report["provider"] == "planar"
    assert run.layer_embedding.mode == "planar_stack"
    assert run.layer_embedding.report["base_z_by_layer_mm"] == [0.125, 0.375, 0.625, 0.875]
    for layer_index, points in enumerate(run.layer_embedding.node_positions_xyz):
        np.testing.assert_allclose(points[:, 2], 0.125 + 0.25 * layer_index)
    assert run.path_graph is not None
    source_job = run.path_graph.to_external_base_source_job()
    deposited = [path for group in source_job.material_paths for path in group.paths]
    assert deposited
    assert all(np.array_equal(path[:, 3:], np.zeros_like(path[:, 3:])) for path in deposited)
    assert len(run.continuous_course_paths_by_layer) == 4


def test_planar_conformal_brim_is_a_calibrated_first_layer_one_stroke():
    run = run_conformal_lattice_pipeline(
        _planar_config(),
        physical_layer_height_mm=0.25,
        extrusion=ExtrusionVolumeModel(0.5, 0.5, preview_line_width_mm=2.0),
    )
    assert run.path_graph is not None
    source_job = run.path_graph.to_external_base_source_job()

    report = apply_conformal_brim(
        source_job,
        ConformalBrimSettings(
            enabled=True,
            width_mm=3.0,
            brim_type="outer_only",
            separation_mm=0.2,
            one_stroke=True,
            line_width_mm=2.0,
        ),
    )

    first_layer = source_job.material_paths[0]
    roles = source_job.meta["path_roles"]["R"]["0"]
    assert report["path_count"] == 1
    assert report["one_stroke_applied"] is True
    assert report["one_stroke_strategy"] == "conformal_outer_spiral"
    assert roles[:2] == ["brim", "conformal_outer_boundary"]
    brim = first_layer.paths[0]
    assert brim.shape[1] == 6
    assert np.allclose(brim[:, 3:], 0.0)
    assert first_layer.extrusion is not None
    assert np.all(np.diff(first_layer.extrusion[0]) >= 0.0)
    assert float(np.min(brim[:, 0])) < 0.0
    assert float(np.max(brim[:, 0])) > 10.0
    assert float(np.min(brim[:, 1])) < 0.0
    assert float(np.max(brim[:, 1])) > 8.0


def test_double_sine_conformal_brim_emits_finite_boundary_pose_field():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["20"],
            "part_width_mm": ["10"],
            "part_height_mm": ["2"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["4"],
            "amplitude_mm": ["1.0"],
            "wavelength_x_mm": ["20"],
            "wavelength_y_mm": ["10"],
            "surface_start_layer": ["0"],
            "samples_x": ["16"],
            "samples_y": ["12"],
        }
    )
    run = run_conformal_lattice_pipeline(
        config,
        physical_layer_height_mm=0.5,
        extrusion=ExtrusionVolumeModel(1.0, 1.0, preview_line_width_mm=2.0),
    )
    assert run.path_graph is not None
    source_job = run.path_graph.to_external_base_source_job()

    report = apply_conformal_brim(
        source_job,
        ConformalBrimSettings(True, 2.0, "outer_only", 0.0, True, 2.0),
    )

    assert report["path_count"] == 1
    brim = source_job.material_paths[0].paths[0]
    assert brim.shape[1] == 6
    assert np.isfinite(brim).all()


def test_planar_export_keeps_the_same_continuous_course_xy_paths_as_zero_curvature():
    planar = run_conformal_lattice_pipeline(
        _planar_config(),
        physical_layer_height_mm=0.25,
    )
    conformal = run_conformal_lattice_pipeline(
        conformal_lattice_config_payload(
            {
                "part_length_mm": ["10"],
                "part_width_mm": ["8"],
                "part_height_mm": ["1"],
                "wall_width_mm": ["2"],
                "base_cell_size_mm": ["3"],
                "amplitude_mm": ["0"],
                "surface_start_layer": ["0"],
                "samples_x": ["8"],
                "samples_y": ["8"],
            }
        ),
        physical_layer_height_mm=0.25,
    )

    assert len(planar.continuous_course_paths_by_layer) == len(
        conformal.continuous_course_paths_by_layer
    )
    for planar_layer, conformal_layer in zip(
        planar.continuous_course_paths_by_layer,
        conformal.continuous_course_paths_by_layer,
    ):
        assert len(planar_layer) == len(conformal_layer)
        for (planar_path, _), (conformal_path, _) in zip(planar_layer, conformal_layer):
            np.testing.assert_allclose(planar_path[:, :2], conformal_path[:, :2])


def test_designer_and_main_ui_expose_one_flat_json_route_with_shared_core_controls():
    designer = surface_preview_html()
    main = _index_html()
    summary = _conformal_spec_ui_summary(
        json.dumps(_planar_config()).encode("utf-8"),
        "flat.json",
    )

    assert 'id="exportPlanarConfig"' in designer
    assert "导出平面蜂窝结构 JSON" in designer
    assert "/api/export-planar-lattice-config" in designer
    assert summary["design_kind"] == "planar_honeycomb"
    assert summary["design_label"] == "平面蜂窝"
    assert "曲面或平面蜂窝结构 JSON" in main
    assert "formData.append('conformal_spec'" in main
    assert "appendCurrentCoreSettings(formData)" in main
    assert "conformal_fiber_enabled" in main
    assert "formData.append('prusa_brim_enabled'" in main
    assert "formData.append('prusa_start_x_mm', document.getElementById('prusaStartX').value)" in main
    assert "formData.append('prusa_start_y_mm', document.getElementById('prusaStartY').value)" in main
    assert "fetch('/conformal-slice'" in main


def test_main_ui_processes_planar_json_through_core_with_optional_fiber(tmp_path: Path):
    config = _planar_config()
    handler = object.__new__(_SlicerUiHandler)
    handler.server_output_dir = tmp_path

    result = handler._handle_conformal_slice(
        "",
        request_data=(
            {
                "core_resin_layer_height": ["0.25"],
                "conformal_fiber_enabled": ["true"],
            },
            {"conformal_spec": ("planar_honeycomb.json", json.dumps(config).encode("utf-8"))},
        ),
    )

    assert result["design_kind"] == "planar_honeycomb"
    assert result["effective_infill_pattern"] == "平面蜂窝连续路径"
    assert result["fiber_reinforcement"]["enabled"] is True
    assert result["fiber_reinforcement"]["layer_interface_source"] == (
        "flat_resin_interlayer_policy_v1"
    )
    assert result["fiber_reinforcement"]["resin_layer_indices"] == [0, 1, 2]
    assert any(layer["fiber_paths"] for layer in result["preview"]["layers"])

    job_dir = tmp_path / result["download_url"].split("/")[-2]
    with np.load(job_dir / "planar_honeycomb_core.npz", allow_pickle=False) as core:
        np.testing.assert_array_equal(core["a"], 0.0)
        np.testing.assert_array_equal(core["b"], 0.0)
        np.testing.assert_array_equal(core["c"], 0.0)


def test_main_ui_planar_json_honors_disabled_fiber_toggle(tmp_path: Path):
    config = _planar_config()
    handler = object.__new__(_SlicerUiHandler)
    handler.server_output_dir = tmp_path

    result = handler._handle_conformal_slice(
        "",
        request_data=(
            {
                "core_resin_layer_height": ["0.5"],
                "conformal_fiber_enabled": ["false"],
                "prusa_brim_enabled": ["true"],
                "prusa_brim_width": ["3"],
                "prusa_brim_type": ["outer_only"],
                "prusa_brim_separation": ["0.2"],
                "prusa_brim_one_stroke": ["true"],
            },
            {"conformal_spec": ("planar_no_fiber.json", json.dumps(config).encode("utf-8"))},
        ),
    )

    assert result["design_kind"] == "planar_honeycomb"
    assert result["fiber_reinforcement"]["enabled"] is False
    assert result["fiber_reinforcement"]["reason"] == "disabled_in_main_ui"
    assert all(not layer["fiber_paths"] for layer in result["preview"]["layers"])
    assert result["brim"]["path_count"] == 1
    assert result["brim"]["one_stroke_applied"] is True
    bounds = result["preview"]["bounds"]
    assert float(bounds["max_x"]) - float(bounds["min_x"]) > 10.0
    assert float(bounds["max_y"]) - float(bounds["min_y"]) > 8.0
