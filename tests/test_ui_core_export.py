from pathlib import Path
from types import SimpleNamespace
import importlib
import inspect
import zipfile

import numpy as np
import pytest

from kuka_slicer.slicer import SliceConfig
from kuka_slicer.ui_server import (
    _core_output_download_path,
    _core_preview_overlay_from_commands,
    _core_preview_xy_offset,
    _ensure_offline_planner_import_paths,
    _index_html,
    _load_core_print_params,
    _parse_core_process_params,
    _preview_payload_from_final_core_npz,
    _use_native_prusa_gcode_for_core,
    merge_fiber_paths_into_job,
)
from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths


def test_core_download_keeps_single_part_npz_as_npz(tmp_path: Path):
    output = tmp_path / "part_core.npz"
    output.write_bytes(b"npz")

    assert _core_output_download_path(output) == output


def test_prusa_brim_one_stroke_uses_adapter_job_for_final_core_input():
    native_gcode = b"G1 X1 Y1 E1"
    standard_prusa = SliceConfig(slicing_kernel="prusa")
    one_stroke_prusa = SliceConfig(slicing_kernel="prusa", brim_one_stroke=True)

    assert _use_native_prusa_gcode_for_core(standard_prusa, native_gcode)
    assert not _use_native_prusa_gcode_for_core(one_stroke_prusa, native_gcode)


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
    assert "preview = _preview_payload(" in handler_source
    assert "_preview_payload_from_final_core_npz(core_npz_path, config)" not in handler_source
    assert "commands_callback=capture_core_preview" not in handler_source
    assert 'id="exportProgressBar"' in html
    assert '<div class="exportActionRow">' in html
    assert 'id="exportProgress" class="exportProgress"' in html
    assert '.exportProgress.visible { display: inline-flex; }' in html
    assert "slice-status?job_id=" in html
    assert 'id="coreDt" type="number" min="0.0001" step="0.0001" value="0.004"' in html
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
    assert len(resin) == len(dense_resin)
    assert resin[0] == dense_resin[0].tolist()
    assert resin[-1] == dense_resin[-1].tolist()
    final_resin_points = {tuple(point) for point in dense_resin.tolist()}
    assert all(tuple(point) in final_resin_points for point in resin)
    assert layer["travel_paths"] == [travel.tolist()]
    assert preview["bounds"]["max_x"] == 22.0
    assert preview["bounds"]["max_y"] == pytest.approx(2.0)


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
    }]
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["e"], process_e)
        np.testing.assert_array_equal(data["seq"], np.arange(len(points)))


def test_core_cooling_is_always_enabled_without_ui_switches():
    _ensure_offline_planner_import_paths()
    module = importlib.import_module("external_npz_preprocessor.process_params")

    params = _parse_core_process_params(
        {"core_resin_fan": ["false"], "core_fiber_fan": ["false"]},
        module,
    )

    assert params.resin.fan_enabled is True
    assert params.fiber.fan_enabled is True


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
