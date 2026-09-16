from pathlib import Path
from types import SimpleNamespace
import importlib
import inspect
import json
import zipfile

import numpy as np
import pytest

from kuka_slicer.slicer import SliceConfig
from kuka_slicer.ui_server import (
    _FINAL_CORE_PREVIEW_XYZ_TOLERANCE_MM,
    _SlicerUiHandler,
    _core_output_download_path,
    _core_preview_overlay_from_commands,
    _core_preview_xy_offset,
    _ensure_offline_planner_import_paths,
    _index_html,
    _load_core_print_params,
    _parse_core_process_params,
    _planning_mesh_for_gcode_source,
    _preview_payload_from_final_core_npz,
    _use_native_prusa_gcode_for_core,
    merge_fiber_paths_into_job,
)
from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths
from kuka_slicer.conformal_lattice.fiber_reinforcement import (
    ContinuousCourseFiberSettings,
    MixedWallFiberSettings,
    _primary_wall_mask,
    apply_continuous_course_fiber_strategy,
    apply_mixed_wall_fiber_strategy,
    derive_symmetric_curvature_fiber_interfaces,
)
from kuka_slicer.conformal_lattice.path_bridge import ExtrusionVolumeModel
from kuka_slicer.conformal_lattice.pipeline import run_conformal_lattice_pipeline
from kuka_slicer.surface_preview.server import conformal_lattice_config_payload


def test_conformal_design_json_generates_core_output_without_source_npz_round_trip(tmp_path: Path):
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["10"],
            "part_width_mm": ["8"],
            "part_height_mm": ["1"],
            "layer_height_mm": ["0.5"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["3"],
            "surface_start_layer": ["0"],
            "samples_x": ["8"],
            "samples_y": ["8"],
        }
    )
    handler = object.__new__(_SlicerUiHandler)
    handler.server_output_dir = tmp_path
    progress: list[int] = []

    result = handler._handle_conformal_slice(
        "",
        request_data=(
            {"core_resin_layer_height": ["0.25"]},
            {"conformal_spec": ("small_design.json", json.dumps(config).encode("utf-8"))},
        ),
        progress_callback=lambda value, _message: progress.append(value),
    )

    assert result["layers"] == 4
    assert result["effective_infill_pattern"] == "共形蜂窝连续路径"
    assert result["infill_pattern_execution"] == {"applied": True, "mode": "continuous_course_network_v1"}
    assert result["fiber_reinforcement"]["enabled"] is True
    assert result["fiber_reinforcement"]["reserved"] is False
    assert result["fiber_reinforcement"]["total_fiber_path_count"] > 0
    assert result["fiber_reinforcement"]["automatic_resin_z_raise"] is True
    assert any(layer["fiber_paths"] for layer in result["preview"]["layers"])
    assert result["preview"]["preview_source"] == "final_core_npz"
    assert result["preview"]["tool_orientation"]["available"] is True
    assert result["core_runtime"]["source"] == "workspace"
    assert result["core_runtime"]["cubic_sampler_fast_path"] is True
    assert result["workflow_timing"]["core_export_s"] >= 0.0
    assert result["workflow_timing"]["preview_s"] >= 0.0
    assert result["workflow_timing"]["total_s"] >= result["workflow_timing"]["core_export_s"]
    job_dir = tmp_path / result["download_url"].split("/")[-2]
    assert not (job_dir / "external_layer_paths_v1.npz").exists()
    assert not (job_dir / "conformal_lattice_geometry_v1.npz").exists()
    with np.load(job_dir / "conformal_lattice_core.npz", allow_pickle=False) as core:
        assert np.linalg.norm(np.column_stack((core["a"], core["b"], core["c"]))) > 1e-3
        assert "max_tcp_orientation_speed_deg_s" not in core.files
        manifest = json.loads(str(core["core_injection_manifest"].item()))
        assert manifest["format"] == "core_npz_local_injection_v2"
        assert manifest["injection_state"] == "base"
        assert manifest["offset_frame"] == "calibrated_flat_print_reference"
        assert manifest["offset_application"] == "per_sample_pose_rotated"
        assert manifest["base_parameters"]["tool_offset"] == [0.0, 0.0, 0.0]
        assert manifest["base_parameters"]["resin_z_print_compensation_mm"] == 0.0
        assert "max_tcp_orientation_speed_deg_s" not in json.dumps(manifest)
    assert progress[-1] == 97


def test_main_ui_continuous_course_fiber_strategy_reaches_final_core_output(tmp_path: Path):
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["30"],
            "part_width_mm": ["20"],
            "part_height_mm": ["2"],
            "grip_end_length_mm": ["5"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["6"],
            "orientation_angle_deg": ["90"],
            "honeycomb_align_x": ["false"],
            "honeycomb_align_y": ["false"],
            "surface_start_layer": ["1"],
            "samples_x": ["12"],
            "samples_y": ["12"],
        }
    )
    handler = object.__new__(_SlicerUiHandler)
    handler.server_output_dir = tmp_path

    result = handler._handle_conformal_slice(
        "",
        request_data=(
            {
                "core_resin_layer_height": ["0.5"],
                "conformal_fiber_enabled": ["true"],
            },
            {"conformal_spec": ("mixed_wall.json", json.dumps(config).encode("utf-8"))},
        ),
    )

    report = result["fiber_reinforcement"]
    assert report["enabled"] is True
    assert report["mode"] == "continuous_course_network_v1"
    assert report["course_semantics"] == "every rectangle-clipped continuous-course fragment is an independent resin/F path with normal Core travel/cut boundaries"
    assert report["layer_interface_source"] == "design_json_symmetric_nonzero_curvature"
    assert report["after_resin_physical_layers"] == [1, 4]
    assert report["total_fiber_path_count"] > 0
    assert report["automatic_resin_z_raise"] is True
    job_dir = tmp_path / result["download_url"].split("/")[-2]
    with np.load(job_dir / "conformal_lattice_core.npz", allow_pickle=False) as core:
        assert core["x"].size > 0
    assert any(layer["fiber_paths"] for layer in result["preview"]["layers"])
    assert sum(
        len(layer["fiber_cut_events"])
        for layer in result["preview"]["layers"]
    ) == report["total_fiber_path_count"]


def test_production_continuous_courses_replace_only_legacy_honeycomb_and_keep_grips():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["50"],
            "part_height_mm": ["10"],
            "grip_end_length_mm": ["25"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["6"],
            "surface_start_layer": ["1"],
            "samples_x": ["48"],
            "samples_y": ["24"],
        }
    )
    run = run_conformal_lattice_pipeline(
        config,
        physical_layer_height_mm=0.5,
        extrusion=ExtrusionVolumeModel(1.0, 1.0),
    )
    assert run.path_graph is not None
    assert run.continuous_course_plan is not None
    assert len(run.continuous_course_plan.paths_xy) == 6
    assert run.continuous_course_plan.course_bounds_mm == pytest.approx((25.0, 0.0, 125.0, 50.0))
    assert run.continuous_course_plan.row_pitch_mm - np.sqrt(3.0) * 6.0 == pytest.approx(4.0)
    assert all(np.all(np.diff(path[:, 0]) >= -1e-9) for path in run.continuous_course_plan.paths_xy)
    source_job = run.path_graph.to_external_base_source_job()
    first, last = derive_symmetric_curvature_fiber_interfaces(run.layer_embedding)
    result = apply_continuous_course_fiber_strategy(
        source_job=source_job,
        graph=run.path_graph,
        course_paths_by_layer=run.continuous_course_paths_by_layer,
        settings=ContinuousCourseFiberSettings(True, first, last),
        fiber_layer_height_mm=0.2,
        fiber_e_per_mm=1.0,
    )

    assert result.paths_per_layer == 6
    first_layer_travel = next(group for group in source_job.travel_paths if group.layer_index == 0)
    resin = next(group for group in source_job.material_paths if group.material == "R" and group.layer_index == 0)
    # Every independent deposition path now receives an explicit route.  The
    # router follows the working-region perimeter/partition boundary rather
    # than leaving Core to create a chord through the honeycomb pores.
    assert len(first_layer_travel.paths) == len(resin.paths) - 1
    for route in first_layer_travel.paths:
        for start, end in zip(route, route[1:]):
            midpoint = (start[:2] + end[:2]) * 0.5
            assert not (25.0 + 1e-7 < midpoint[0] < 125.0 - 1e-7 and 1e-7 < midpoint[1] < 50.0 - 1e-7)
    roles = source_job.meta["path_roles"]["R"]["0"]
    assert "conformal_honeycomb_macro_partition" not in roles
    assert roles.count("conformal_continuous_course_fragment") == 6
    assert "conformal_outer_boundary" in roles
    assert roles.count("conformal_partition_wall") == 2
    assert roles.count("conformal_grip_zigzag_x_one_stroke") == 2
    resin = next(group for group in source_job.material_paths if group.material == "R" and group.layer_index == 0)
    courses = [path for path, role in zip(resin.paths, roles) if role == "conformal_continuous_course_fragment"]
    assert len(courses) == 6
    # Serpentine ordering may reverse independent courses to avoid a diagonal
    # inter-course Travel; their geometric endpoints remain the same sides.
    assert all({round(float(path[0, 0]), 6), round(float(path[-1, 0]), 6)} == {25.0, 125.0} for path in courses)
    assert all(abs(np.diff(path[:, 0])).max() > 0.0 for path in courses)
    selected_layer = result.resin_layer_indices[0]
    selected_roles = source_job.meta["path_roles"]["R"][str(selected_layer)]
    selected_resin = next(group for group in source_job.material_paths if group.material == "R" and group.layer_index == selected_layer)
    selected_courses = [path for path, role in zip(selected_resin.paths, selected_roles) if role == "conformal_continuous_course_fragment"]
    fiber = next(group for group in source_job.material_paths if group.material == "F" and group.layer_index == selected_layer)
    assert len(fiber.paths) == len(selected_courses) == 6
    for resin_path, fiber_path in zip(selected_courses, fiber.paths):
        np.testing.assert_allclose(resin_path[:, :2], fiber_path[:, :2])


def test_boundary_clipped_continuous_courses_are_retained_as_independent_paths():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["160"],
            "part_width_mm": ["60"],
            "part_height_mm": ["10"],
            "grip_end_length_mm": ["25"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["10"],
            "surface_start_layer": ["1"],
            "samples_x": ["48"],
            "samples_y": ["24"],
        }
    )
    run = run_conformal_lattice_pipeline(
        config,
        physical_layer_height_mm=0.5,
        extrusion=ExtrusionVolumeModel(1.0, 1.0),
    )
    assert run.path_graph is not None
    assert run.continuous_course_plan is not None
    # Four through-routes plus three independently clipped fragments on each
    # of the upper/lower rectangle boundaries.  Pure diagonal corner grazes
    # are excluded because they have no horizontal pore-channel support.
    assert len(run.continuous_course_plan.paths_xy) == 10
    assert all(np.all(path[:, 0] >= 25.0 - 1e-9) and np.all(path[:, 0] <= 135.0 + 1e-9) for path in run.continuous_course_plan.paths_xy)
    assert sum(np.isclose(path[:, 1], 0.0).any() for path in run.continuous_course_plan.paths_xy) == 3
    assert sum(np.isclose(path[:, 1], 60.0).any() for path in run.continuous_course_plan.paths_xy) == 3

    source_job = run.path_graph.to_external_base_source_job()
    first, last = derive_symmetric_curvature_fiber_interfaces(run.layer_embedding)
    result = apply_continuous_course_fiber_strategy(
        source_job=source_job,
        graph=run.path_graph,
        course_paths_by_layer=run.continuous_course_paths_by_layer,
        settings=ContinuousCourseFiberSettings(True, first, last),
        fiber_layer_height_mm=0.2,
        fiber_e_per_mm=1.0,
    )
    assert result.paths_per_layer == 10
    roles = source_job.meta["path_roles"]["R"]["0"]
    assert roles.count("conformal_continuous_course_fragment") == 10
    resin = next(group for group in source_job.material_paths if group.material == "R" and group.layer_index == 0)
    fiber = next(group for group in source_job.material_paths if group.material == "F" and group.layer_index == result.resin_layer_indices[0])
    resin_fragments = [path for path, role in zip(resin.paths, roles) if role == "conformal_continuous_course_fragment"]
    assert len(resin_fragments) == len(fiber.paths) == 10
    for resin_path, fiber_path in zip(resin_fragments, fiber.paths):
        np.testing.assert_allclose(resin_path[:, :2], fiber_path[:, :2])


def test_x_fiber_strategy_accepts_default_vertical_wall_honeycomb():
    """Default 0° honeycomb uses inclined X-progressing zigzag chains."""

    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["30"],
            "part_width_mm": ["20"],
            "part_height_mm": ["2"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["6"],
            "orientation_angle_deg": ["0"],
            "honeycomb_align_x": ["false"],
            "honeycomb_align_y": ["false"],
            "surface_start_layer": ["1"],
            "samples_x": ["12"],
            "samples_y": ["12"],
        }
    )
    run = run_conformal_lattice_pipeline(
        config,
        physical_layer_height_mm=0.5,
        extrusion=ExtrusionVolumeModel(1.0, 1.0),
    )
    assert run.path_graph is not None
    mask, selection = _primary_wall_mask(
        np.asarray(run.path_graph.layer_node_positions_xyz[0]),
        np.asarray(run.path_graph.edge_node_ids),
        0,
    )

    assert selection == "axis_progressing_zigzag_wall_chain"
    assert int(np.count_nonzero(mask)) > 0


def test_mixed_wall_chain_strategy_shares_complete_courses_between_resin_and_fiber():
    """Mapped triangle segments must never become independent F/R courses."""

    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["50"],
            "part_height_mm": ["10"],
            "grip_end_length_mm": ["25"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["6"],
            "orientation_angle_deg": ["0"],
            "surface_start_layer": ["1"],
            "samples_x": ["48"],
            "samples_y": ["24"],
        }
    )
    run = run_conformal_lattice_pipeline(
        config,
        physical_layer_height_mm=0.5,
        extrusion=ExtrusionVolumeModel(1.0, 1.0),
    )
    assert run.path_graph is not None
    source_job = run.path_graph.to_external_source_job()
    first, last = derive_symmetric_curvature_fiber_interfaces(run.layer_embedding)
    result = apply_mixed_wall_fiber_strategy(
        source_job=source_job,
        graph=run.path_graph,
        settings=MixedWallFiberSettings(True, first, last),
        fiber_layer_height_mm=0.2,
        fiber_e_per_mm=1.0,
    )

    report = result.report
    assert report["mode"] == "uniform_mixed_wall_chain_v2"
    assert report["paths_per_fiber_layer"] < len(run.path_graph.edge_node_ids)
    assert report["resin_honeycomb_paths_per_layer"] == report["paths_per_fiber_layer"]
    selected_layer = result.resin_layer_indices[0]
    resin_group = next(group for group in source_job.material_paths if group.layer_index == selected_layer and group.material == "R")
    fiber_group = next(group for group in source_job.material_paths if group.layer_index == selected_layer and group.material == "F")
    resin_chain_paths = resin_group.paths[-len(fiber_group.paths):]
    assert len(fiber_group.paths) == report["paths_per_fiber_layer"]
    assert max(len(path) for path in fiber_group.paths) > 2
    for resin_path, fiber_path in zip(resin_chain_paths, fiber_group.paths):
        np.testing.assert_allclose(resin_path[:, :2], fiber_path[:, :2])
    assert source_job.travel_paths == []


def test_conformal_debug_export_is_opt_in_and_never_becomes_core_input(tmp_path: Path):
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["10"],
            "part_width_mm": ["8"],
            "part_height_mm": ["1"],
            "layer_height_mm": ["0.5"],
            "wall_width_mm": ["2"],
            "base_cell_size_mm": ["3"],
            "surface_start_layer": ["0"],
            "samples_x": ["8"],
            "samples_y": ["8"],
        }
    )
    handler = object.__new__(_SlicerUiHandler)
    handler.server_output_dir = tmp_path

    result = handler._handle_conformal_slice(
        "",
        request_data=(
            {"core_resin_layer_height": ["0.25"], "conformal_debug_export": ["true"]},
            {"conformal_spec": ("small_design.json", json.dumps(config).encode("utf-8"))},
        ),
    )

    job_dir = tmp_path / result["download_url"].split("/")[-2]
    assert result["debug_filename"] == "conformal_lattice_debug.zip"
    assert result["debug_download_url"].endswith("/conformal_lattice_debug.zip")
    with zipfile.ZipFile(job_dir / result["debug_filename"]) as archive:
        assert set(archive.namelist()) == {
            "conformal_lattice_geometry_v1.npz",
            "external_layer_paths_v1.npz",
        }

    # The debug source NPZ remains a reference artifact only.  Its legacy
    # serialized route must nevertheless produce byte-identical trajectory
    # arrays to the production in-memory SourceJob route.
    _ensure_offline_planner_import_paths()
    process_params_module = importlib.import_module("external_npz_preprocessor.process_params")
    export_runner = importlib.import_module("external_npz_preprocessor.export_runner")
    legacy_core = job_dir / "legacy_debug_reference_core.npz"
    export_runner.convert_external_npz(
        job_dir / "external_layer_paths_v1.npz",
        legacy_core,
        _parse_core_process_params(
            {"core_resin_layer_height": ["0.25"], "conformal_debug_export": ["true"]},
            process_params_module,
        ),
        chunk_size=5_000_000,
    )
    with (
        np.load(job_dir / "conformal_lattice_core.npz", allow_pickle=False) as direct,
        np.load(legacy_core, allow_pickle=False) as serialized,
    ):
        assert direct.files == serialized.files
        for key in direct.files:
            assert direct[key].dtype == serialized[key].dtype
            assert direct[key].shape == serialized[key].shape
            assert direct[key].tobytes() == serialized[key].tobytes()


def test_core_download_keeps_single_part_npz_as_npz(tmp_path: Path):
    output = tmp_path / "part_core.npz"
    output.write_bytes(b"npz")

    assert _core_output_download_path(output) == output


def test_prusa_brim_one_stroke_uses_native_gcode_for_final_core_input():
    native_gcode = b"G1 X1 Y1 E1"
    standard_prusa = SliceConfig(slicing_kernel="prusa")
    one_stroke_prusa = SliceConfig(slicing_kernel="prusa", brim_one_stroke=True)

    assert _use_native_prusa_gcode_for_core(standard_prusa, native_gcode)
    assert _use_native_prusa_gcode_for_core(one_stroke_prusa, native_gcode)


def test_gcode_planning_mesh_uses_the_resolved_build_axis():
    from kuka_slicer.stl_io import Mesh

    raw = Mesh(np.asarray([[
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
    ]]))
    planning = _planning_mesh_for_gcode_source(raw, SliceConfig(build_axis="y"))

    np.testing.assert_array_equal(planning.triangles, raw.triangles[:, :, [0, 2, 1]])


def test_fiber_interpath_travels_are_included_in_the_source_preview_timeline():
    job = ExternalSourceJob(
        material_paths=[
            MaterialPaths(0, "R", [np.asarray([[0.0, 0.0, 0.2], [1.0, 0.0, 0.2]])])
        ],
        meta={"motion_order": {"0": [{"kind": "deposit", "index": 0}]}},
    )
    fibers = {
        0: [
            [[0.0, 2.0, 0.3], [1.0, 2.0, 0.3]],
            [[3.0, 2.0, 0.3], [4.0, 2.0, 0.3]],
        ]
    }
    connector = np.asarray([[1.0, 2.0, 0.3], [1.0, 1.0, 0.3], [3.0, 2.0, 0.3]])

    merge_fiber_paths_into_job(job, fibers, {0: [connector]})

    assert job.travel_paths[0].paths == [connector]
    assert job.meta["motion_order"]["0"][-3:] == [
        {"kind": "fiber_deposit", "index": 0},
        {"kind": "fiber_travel", "index": 0},
        {"kind": "fiber_deposit", "index": 1},
    ]


def test_core_download_bundles_split_parts_and_sidecars(tmp_path: Path):
    output = tmp_path / "part_core.npz"
    (tmp_path / "part_core_part0000.npz").write_bytes(b"part0")
    (tmp_path / "part_core_part0001.npz").write_bytes(b"part1")
    (tmp_path / "part_core.offset.json").write_text("{}", encoding="utf-8")
    (tmp_path / "part_core.timing.json").write_text("{}", encoding="utf-8")

    package = _core_output_download_path(output)

    assert package.suffix == ".zip"
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == {
            "part_core_part0000.npz",
            "part_core_part0001.npz",
            "part_core.offset.json",
            "part_core.timing.json",
        }
        assert all(entry.compress_type == zipfile.ZIP_STORED for entry in archive.infolist())


def test_ui_uses_pre_core_source_preview_and_exposes_core_export_progress():
    html = _index_html()
    handler_source = inspect.getsource(importlib.import_module("kuka_slicer.ui_server")._SlicerUiHandler._handle_slice)
    core_defaults = _load_core_print_params()

    assert "previewData = result.preview" in html
    assert "hasOrderedFiber" in html
    assert "points.length >= 1" in html
    assert "path.length === 1" in html
    assert "送入 Core 前的源 NPZ" in html
    assert "_preview_payload_from_core_source_job" in handler_source
    assert "_planning_mesh_for_gcode_source(mesh, config)" in handler_source
    assert "_preview_payload_from_final_core_npz(core_npz_path, config)" not in handler_source
    assert "commands_callback=capture_core_preview" not in handler_source
    assert 'id="exportProgressBar"' in html
    assert '<div class="exportActionRow">' in html
    assert 'id="exportProgress" class="exportProgress"' in html
    assert '.exportProgress.visible { display: grid; }' in html
    assert html.index('id="exportProgress"') < html.index("<main>")
    assert 'id="exportProgressState"' in html
    assert "String(job.message || '等待处理任务')" in html
    assert "阶段进度不等同于剩余时间估算" in html
    assert "exportElapsedEl.textContent = '总用时 '" in html
    assert 'Core：仓库源码 · 三次样条优化已启用' in html
    assert "slice-status?job_id=" in html
    assert 'id="coreDt" type="number" min="0.0001" step="0.0001" value="0.004"' in html
    assert 'id="coreMaxTcpOrientationSpeed"' in html
    assert "core_max_tcp_orientation_speed" in html
    assert "25 °/s 是离线 Core 的工程默认值" in html
    assert 'id="conformalFiberEnabled" type="checkbox" checked' in html
    assert '启用连续纤维路径' in html
    assert 'id="conformalFiberDoubleWallAxis"' not in html
    assert "conformal_fiber_double_wall_axis" not in html
    assert 'id="showFiberCutEvents" type="checkbox" checked' in html
    assert '纤维剪切点（CUT）' in html
    assert "path_id: rawEntry.path_id" in html
    assert "kuka.conformalContinuousCourseFiber.v2" in html
    assert "const firstFiberLayerPosition = layers.findIndex" in html
    assert "layerSlider.value = firstFiberLayerPosition >= 0 ? firstFiberLayerPosition : 0" in html
    assert "containsFiber ? ' · 含纤维' : ''" in html
    assert 'id="showCoreTravelPaths"' in html
    assert 'id="showPrimeline"' in html
    assert 'id="prusaRaftAutoContact"' in html
    assert 'id="prusaRaftContactLayerHeight"' in html
    assert 'id="prusaRaftContactLayerHeight" type="number" min="0.1" max="2" step="0.05" value="0.75"' in html
    assert 'id="prusaRaftContactDensity"' in html
    assert 'id="prusaRaftContactExtrusionWidth"' in html
    assert "drawOriginMarker" in html
    assert "previewData?.core_overlay?.sequence" in html
    assert ".filter((entry) => entry.role !== 'layer_lift')" in html
    assert 'id="coreResinFan"' not in html
    assert 'id="coreFiberFan"' not in html
    assert "coreMaterialColumns" in html
    assert "coreTravelPanel" in html
    assert 'id="coreNpzPreviewButton"' in html
    assert 'id="conformalDebugExportButton"' in html
    assert 'aria-pressed="false">调试导出：关' in html
    assert "formData.append('conformal_debug_export'" in html
    assert "'corePrimelineLength', 'coreDt', 'coreMaxTcpOrientationSpeed'" in html
    assert 'id="conformalDebugDownload"' in html
    assert "/choose-core-npz-preview" in html
    assert "applyFinalCorePreview" in html
    assert 'id="paths"' not in html
    assert 'id="executedKernel"' not in html
    assert 'id="executedPlanningLineWidth"' not in html
    fiber_start = html.index('id="coreFiberStartAccel"')
    fiber_speed = html.index("<h5>打印速度</h5>", html.index("<h4>纤维</h4>"))
    assert fiber_start < fiber_speed
    assert f'"feed_mm_s": {core_defaults.fiber.feed_mm_s}' in html
    assert f'"temperature_c": {core_defaults.resin.temperature_c}' in html
    assert f'"prime_length_mm": {core_defaults.resin.prime_length_mm}' in html
    # The page reflects the active checked-in process preset rather than a
    # stale hard-coded placement.
    assert f'"start_x_mm": {_load_core_print_params().start_x_mm}' in html
    assert 'fetch(\'/ui-settings\'' in html
    assert "const adjustValue = (targetInput, direction) =>" in html
    assert 'className = \'magnitudeInputWrap\'' in html
    assert 'className = \'magnitudeSpinButton\'' in html
    assert "targetInput.dispatchEvent(new Event('input'" in html
    assert '.processBand .actions {\n      display: flex;' in html
    assert 'grid-column: 1 / -1;' in html
    assert 'flex-wrap: nowrap;' in html


def test_ui_exposes_live_core_worker_warmup_progress():
    html = _index_html()

    assert 'id="coreWarmup"' in html
    assert 'id="coreWarmupBar"' in html
    assert 'aria-label="Core worker 预热进度"' in html
    assert "fetch('/core-warmup-status'" in html
    assert "window.setTimeout(pollCoreWarmup, 250)" in html


def test_ui_local_core_preview_uses_final_npz_trajectory(tmp_path: Path, monkeypatch):
    output = tmp_path / "ordinary_core.npz"
    np.savez_compressed(
        output,
        x=np.asarray([1.0, 2.0]),
        y=np.asarray([3.0, 4.0]),
        z=np.asarray([0.5, 0.5]),
        tool_id=np.asarray([2, 2]),
        move_type=np.asarray([1, 1]),
        event_flag=np.asarray([0, 0]),
        layer_index=np.asarray([0, 0]),
        path_id=np.asarray([1, 1]),
        path_end_flag=np.asarray([0, 1]),
        move_type_vocab_keys=np.asarray(["PRINT"]),
        move_type_vocab_vals=np.asarray([1]),
    )
    monkeypatch.setattr("kuka_slicer.ui_server._choose_final_core_npz_file", lambda _: output)
    handler = object.__new__(_SlicerUiHandler)
    captured: dict[str, object] = {}
    handler._send_json = captured.update
    _SlicerUiHandler.core_preview_last_directory = None
    _SlicerUiHandler.core_preview_picker_state_path = None

    handler._choose_core_npz_preview()

    assert captured["ok"] is True
    assert captured["file_name"] == output.name
    assert captured["preview"]["preview_source"] == "final_core_npz"
    assert captured["preview"]["layers"][0]["resin_paths"][0]["points"] == [
        [1.0, 3.0, 0.5],
        [2.0, 4.0, 0.5],
    ]


def test_final_core_preview_loads_all_parts_when_selecting_one_part(tmp_path: Path):
    for index, x in enumerate((1.0, 2.0)):
        np.savez_compressed(
            tmp_path / f"ordinary_core_part{index:03d}.npz",
            x=np.asarray([x, x + 0.1]),
            y=np.asarray([0.0, 0.0]),
            z=np.asarray([0.5, 0.5]),
            tool_id=np.asarray([2, 2]),
            move_type=np.asarray([1, 1]),
            event_flag=np.asarray([0, 0]),
            layer_index=np.asarray([index, index]),
            path_id=np.asarray([1, 1]),
            path_end_flag=np.asarray([0, 1]),
            move_type_vocab_keys=np.asarray(["PRINT"]),
            move_type_vocab_vals=np.asarray([1]),
        )

    preview = _preview_payload_from_final_core_npz(
        tmp_path / "ordinary_core_part000.npz",
        SliceConfig(line_width=2.0),
    )

    assert [layer["index"] for layer in preview["layers"]] == [0, 1]


def test_final_core_preview_uses_final_rows_and_bounds_each_path(tmp_path: Path):
    output = tmp_path / "final_core.npz"
    dense_resin = np.column_stack((
        np.linspace(10.0, 20.0, 2_000),
        np.sin(np.linspace(0.0, 2.0, 2_000)),
        np.full(2_000, 0.4),
    ))
    travel = np.asarray([[20.0, 0.0, 0.4], [22.0, 2.0, 0.4]])
    event = np.asarray([[999.0, 999.0, 99.0]])
    points = np.vstack((dense_resin, travel, event))
    np.savez_compressed(
        output,
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        tool_id=np.asarray([2] * len(dense_resin) + [2, 2, 2]),
        move_type=np.asarray([1] * len(dense_resin) + [0, 0, 0]),
        event_flag=np.asarray([0] * (len(dense_resin) + 2) + [1]),
        layer_index=np.zeros(len(points), dtype=np.uint32),
        path_id=np.asarray([7] * len(dense_resin) + [8, 8, 0]),
        path_end_flag=np.asarray([0] * (len(dense_resin) - 1) + [1, 0, 1, 0]),
        move_type_vocab_keys=np.asarray(["TRAVEL", "PRINT"]),
        move_type_vocab_vals=np.asarray([0, 1]),
    )

    preview = _preview_payload_from_final_core_npz(output, SliceConfig(line_width=2.0))
    layer = preview["layers"][0]
    resin = layer["resin_paths"][0]["points"]

    assert preview["preview_source"] == "final_core_npz"
    assert 2 < len(resin) < len(dense_resin)
    assert resin[0] == dense_resin[0].tolist()
    assert resin[-1] == dense_resin[-1].tolist()
    final_resin_points = {tuple(point) for point in dense_resin.tolist()}
    assert all(tuple(point) in final_resin_points for point in resin)
    simplified = np.asarray(resin)
    distances = []
    for point in dense_resin:
        segment_distances = []
        for start, end in zip(simplified, simplified[1:]):
            chord = end - start
            denominator = float(np.dot(chord, chord))
            fraction = 0.0 if denominator <= 1e-24 else float(
                np.clip(np.dot(point - start, chord) / denominator, 0.0, 1.0)
            )
            segment_distances.append(np.linalg.norm(point - (start + fraction * chord)))
        distances.append(min(segment_distances))
    assert max(distances) <= _FINAL_CORE_PREVIEW_XYZ_TOLERANCE_MM + 1e-12
    assert layer["travel_paths"] == [travel.tolist()]
    assert preview["bounds"]["max_x"] == 22.0
    assert preview["bounds"]["max_y"] == pytest.approx(2.0)


def test_final_core_preview_exposes_real_fiber_cut_event_coordinates(tmp_path: Path):
    output = tmp_path / "fiber_cut_core.npz"
    points = np.asarray([
        [1.0, 2.0, 0.6],
        [3.0, 4.0, 0.7],
        [3.0, 4.0, 0.7],
    ])
    np.savez_compressed(
        output,
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        tool_id=np.asarray([1, 1, 1]),
        move_type=np.asarray([1, 1, 0]),
        event_flag=np.asarray([0, 0, 1], dtype=np.uint8),
        event_type=np.asarray([0, 0, 8], dtype=np.uint8),
        preview_layer_index=np.asarray([2, 2, 2], dtype=np.uint32),
        path_id=np.asarray([17, 17, 17], dtype=np.uint32),
        path_end_flag=np.asarray([0, 1, 0], dtype=np.uint8),
        move_type_vocab_keys=np.asarray(["TRAVEL", "PRINT"]),
        move_type_vocab_vals=np.asarray([0, 1], dtype=np.uint8),
        event_type_vocab_keys=np.asarray(["", "cut"]),
        event_type_vocab_vals=np.asarray([0, 8], dtype=np.uint8),
    )

    preview = _preview_payload_from_final_core_npz(output, SliceConfig(line_width=2.0))
    layer = preview["layers"][0]

    assert layer["motion_paths"][0]["path_id"] == 17
    assert layer["fiber_cut_events"] == [{"point": [3.0, 4.0, 0.7], "path_id": 17}]
    assert preview["bounds"]["max_x"] == 3.0
    assert preview["bounds"]["max_y"] == 4.0


def test_final_core_preview_decimates_dense_resin_without_changing_npz(tmp_path: Path):
    output = tmp_path / "dense_final_core.npz"
    count = 16_003
    points = np.column_stack((
        np.arange(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
        np.full(count, 0.5),
    ))
    extrusion = np.linspace(2.0, 4.0, count)
    np.savez_compressed(
        output,
        x=points[:, 0], y=points[:, 1], z=points[:, 2], e=extrusion,
        tool_id=np.full(count, 2),
        move_type=np.full(count, 1),
        event_flag=np.zeros(count, dtype=np.uint8),
        layer_index=np.zeros(count, dtype=np.uint32),
        path_id=np.full(count, 7),
        path_end_flag=np.r_[np.zeros(count - 1, dtype=np.uint8), 1],
        move_type_vocab_keys=np.asarray(["PRINT"]),
        move_type_vocab_vals=np.asarray([1]),
    )

    preview = _preview_payload_from_final_core_npz(output, SliceConfig(line_width=2.0))

    resin_paths = preview["layers"][0]["resin_paths"]
    assert [len(path["points"]) for path in resin_paths] == [2]
    assert resin_paths[0]["points"] == points[[0, -1]].tolist()
    assert resin_paths[0]["extrusion"] == extrusion[[0, -1]].tolist()
    assert preview["preview_sampling"]["source_npz_row_count"] == count
    assert preview["preview_sampling"]["source_displayable_point_count"] == count
    assert preview["preview_sampling"]["display_point_count"] == 2
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["x"], points[:, 0])
        np.testing.assert_array_equal(data["e"], extrusion)


def test_final_core_preview_joins_adjacent_travel_paths_without_rewriting_points(tmp_path: Path):
    output = tmp_path / "adjacent_travel.npz"
    points = np.asarray([
        [0.0, 0.0, 0.5], [1.0, 0.0, 0.5],
        [1.0, 0.0, 0.5], [1.25, 0.5, 0.5], [2.0, 1.0, 0.5],
    ])
    a = np.asarray([0.0, 0.0, 5.0, 10.0, 15.0])
    b = np.asarray([0.0, 0.0, 1.0, 2.0, 3.0])
    c = np.asarray([0.0, 0.0, -1.0, -2.0, -3.0])
    np.savez_compressed(
        output,
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        a=a, b=b, c=c,
        tool_id=np.asarray([2, 2, 2, 2, 2]),
        move_type=np.asarray([1, 1, 0, 0, 0]),
        event_flag=np.zeros(len(points), dtype=np.uint8),
        layer_index=np.zeros(len(points), dtype=np.uint32),
        path_id=np.asarray([1, 1, 2, 2, 3]),
        path_end_flag=np.asarray([0, 1, 0, 1, 1]),
        move_type_vocab_keys=np.asarray(["TRAVEL", "PRINT"]),
        move_type_vocab_vals=np.asarray([0, 1]),
    )

    preview = _preview_payload_from_final_core_npz(output, SliceConfig(line_width=2.0))
    layer = preview["layers"][0]
    expected_travel = np.column_stack((points[2:], a[2:], b[2:], c[2:])).tolist()

    assert layer["travel_paths"] == [expected_travel]
    assert layer["motion_paths"][1]["points"] == expected_travel
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["x"], points[:, 0])
        np.testing.assert_array_equal(data["y"], points[:, 1])
        np.testing.assert_array_equal(data["z"], points[:, 2])
        np.testing.assert_array_equal(data["a"], a)
        np.testing.assert_array_equal(data["b"], b)
        np.testing.assert_array_equal(data["c"], c)


def test_final_core_preview_preserves_a_sharp_orientation_transition(tmp_path: Path):
    output = tmp_path / "orientation_corner.npz"
    points = np.column_stack((
        np.arange(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        np.full(6, 0.5),
    ))
    b = np.asarray([0.0, 0.0, 0.0, 30.0, 30.0, 30.0])
    np.savez_compressed(
        output,
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        a=np.zeros(6), b=b, c=np.zeros(6),
        tool_id=np.full(6, 1),
        move_type=np.full(6, 1),
        event_flag=np.zeros(6, dtype=np.uint8),
        layer_index=np.zeros(6, dtype=np.uint32),
        path_id=np.full(6, 7),
        path_end_flag=np.r_[np.zeros(5, dtype=np.uint8), 1],
        move_type_vocab_keys=np.asarray(["PRINT"]),
        move_type_vocab_vals=np.asarray([1]),
    )

    preview = _preview_payload_from_final_core_npz(
        output, SliceConfig(line_width=2.0)
    )

    fiber = preview["layers"][0]["fiber_paths"][0]
    assert [2.0, 0.0, 0.5, 0.0, 0.0, 0.0] in fiber
    assert [3.0, 0.0, 0.5, 0.0, 30.0, 0.0] in fiber
    assert len(fiber) < len(points)
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["b"], b)


def test_final_core_preview_omits_stationary_print_process_rows_only(tmp_path: Path):
    output = tmp_path / "stationary_process.npz"
    spatial_points = np.asarray([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
    stationary_process_points = np.asarray([
        [1.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
    ])
    points = np.vstack((spatial_points, stationary_process_points))
    process_e = np.asarray([0.0, 0.5, 0.5, 10.0, 20.0])
    np.savez_compressed(
        output,
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        e=process_e,
        seq=np.arange(len(points), dtype=np.int64),
        tool_id=np.full(len(points), 2),
        move_type=np.full(len(points), 1),
        event_flag=np.zeros(len(points), dtype=np.uint8),
        layer_index=np.zeros(len(points), dtype=np.uint32),
        path_id=np.asarray([1, 1, 2, 2, 2]),
        path_end_flag=np.asarray([0, 1, 0, 0, 1]),
        move_type_vocab_keys=np.asarray(["TRAVEL", "PRINT"]),
        move_type_vocab_vals=np.asarray([0, 1]),
    )

    preview = _preview_payload_from_final_core_npz(output, SliceConfig(line_width=2.0))

    layer = preview["layers"][0]
    assert layer["resin_paths"] == [{
        "role": "final_resin",
        "points": spatial_points.tolist(),
        "extrusion": [0.0, 0.5],
    }]
    assert layer["motion_paths"][0]["extrusion"] == [0.0, 0.5]
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["e"], process_e)
        np.testing.assert_array_equal(data["seq"], np.arange(len(points)))


def test_final_core_preview_preserves_zero_e_connector_profile(tmp_path: Path):
    output = tmp_path / "zero_e_connector.npz"
    points = np.asarray([
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
    ])
    extrusion = np.asarray([5.0, 5.0, 5.4])
    np.savez_compressed(
        output,
        x=points[:, 0], y=points[:, 1], z=points[:, 2], e=extrusion,
        tool_id=np.full(len(points), 2),
        move_type=np.full(len(points), 1),
        event_flag=np.zeros(len(points), dtype=np.uint8),
        layer_index=np.zeros(len(points), dtype=np.uint32),
        path_id=np.full(len(points), 1),
        path_end_flag=np.asarray([0, 0, 1]),
        move_type_vocab_keys=np.asarray(["PRINT"]),
        move_type_vocab_vals=np.asarray([1]),
    )

    preview = _preview_payload_from_final_core_npz(output, SliceConfig(line_width=2.0))

    resin = preview["layers"][0]["resin_paths"][0]
    assert resin["points"] == points.tolist()
    assert resin["extrusion"] == extrusion.tolist()
    # The browser uses the equal first two E values to render this segment as
    # the gray-blue dashed, zero-extrusion connector rather than deposition.
    assert resin["extrusion"][1] - resin["extrusion"][0] == 0.0


def test_core_cooling_is_always_enabled_without_ui_switches():
    _ensure_offline_planner_import_paths()
    module = importlib.import_module("external_npz_preprocessor.process_params")

    params = _parse_core_process_params(
        {"core_resin_fan": ["false"], "core_fiber_fan": ["false"]},
        module,
    )

    assert params.resin.fan_enabled is True
    assert params.fiber.fan_enabled is True


def test_main_ui_parses_tcp_orientation_speed_into_core_params():
    _ensure_offline_planner_import_paths()
    module = importlib.import_module("external_npz_preprocessor.process_params")

    params = _parse_core_process_params(
        {"core_max_tcp_orientation_speed": ["37.5"]},
        module,
    )

    assert params.max_tcp_orientation_speed_deg_s == pytest.approx(37.5)


def test_offline_slicer_always_exports_a_zero_offset_base_contract():
    _ensure_offline_planner_import_paths()
    module = importlib.import_module("external_npz_preprocessor.process_params")

    params = _parse_core_process_params(
        {
            "core_fiber_offset_x": ["9.0"],
            "core_fiber_offset_y": ["8.0"],
            "core_fiber_offset_z": ["7.0"],
            "core_resin_z_comp": ["-30.5"],
        },
        module,
    )

    assert params.export.fiber_x_print_compensation_mm == 0.0
    assert params.export.fiber_y_print_compensation_mm == 0.0
    assert params.export.fiber_z_print_compensation_mm == 0.0
    assert params.export.resin_z_print_compensation_mm == 0.0

    html = _index_html()
    assert "机器喷头偏置由上位机现场注入" in html
    assert 'id="coreFiberOffsetX"' not in html
    assert 'id="coreFiberOffsetY"' not in html
    assert 'id="coreFiberOffsetZ"' not in html
    assert 'id="coreResinZComp"' not in html


def test_core_placement_uses_integrated_prusa_start_xy():
    _ensure_offline_planner_import_paths()
    module = importlib.import_module("external_npz_preprocessor.process_params")

    params = _parse_core_process_params(
        {
            "prusa_start_x_mm": ["10"],
            "prusa_start_y_mm": ["10"],
        },
        module,
    )

    assert params.start_x_mm == 10.0
    assert params.start_y_mm == 10.0


def test_core_preview_overlay_collapses_multi_segment_prusa_travel_for_ordering():
    point = lambda x: SimpleNamespace(x=x, y=0.0, z=0.5)
    commands = [
        SimpleNamespace(raw="external_npz_prusa_travel", type="TRAVEL", layer=0),
        SimpleNamespace(raw="external_npz_prusa_travel", type="TRAVEL", layer=0),
        SimpleNamespace(raw="external_npz_polyline", type="PRINT", subtype="RESIN_PRINT", layer=0),
        SimpleNamespace(
            raw="external_npz_travel",
            type="TRAVEL",
            layer=0,
            start_pos=point(10.0),
            pos=point(20.0),
        ),
        SimpleNamespace(raw="external_npz_polyline", type="PRINT", subtype="FIBER_PRINT", layer=0),
        SimpleNamespace(
            raw="external_npz_travel",
            type="TRAVEL",
            layer=0,
            start_pos=point(30.0),
            pos=point(40.0),
        ),
    ]

    sequence = _core_preview_overlay_from_commands(commands)["sequence"]

    assert [(item["role"], item["anchor"]) for item in sequence] == [
        ("core_travel", 2),
        ("core_travel", 3),
    ]


def test_core_preview_overlay_maps_core_coordinates_back_to_prusa_frame():
    point = lambda x, y: SimpleNamespace(x=x, y=y, z=0.5)
    commands = [
        SimpleNamespace(
            raw="external_npz_layer_lift",
            type="TRAVEL",
            layer=2,
            start_pos=point(92.605, 178.221),
            pos=point(92.605, 178.221),
        )
    ]

    overlay = _core_preview_overlay_from_commands(commands, xy_offset=(30.0, 10.0))

    assert overlay["layer_lift_paths"][0]["points"] == [
        [122.605, 188.221, 0.5],
        [122.605, 188.221, 0.5],
    ]


def test_core_preview_xy_offset_uses_material_frame_minimum():
    job = SimpleNamespace(
        material_paths=[
            SimpleNamespace(paths=[[[30.0, 10.0, 0.5], [40.0, 20.0, 0.5]]])
        ]
    )
    core_params = SimpleNamespace(start_x_mm=0.0, start_y_mm=0.0)

    assert _core_preview_xy_offset(job, core_params) == (30.0, 10.0)
