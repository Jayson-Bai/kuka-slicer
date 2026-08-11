from pathlib import Path
from types import SimpleNamespace
import importlib
import zipfile

from kuka_slicer.ui_server import (
    _core_output_download_path,
    _core_preview_overlay_from_commands,
    _core_preview_xy_offset,
    _ensure_offline_planner_import_paths,
    _index_html,
    _parse_core_process_params,
)


def test_core_download_keeps_single_part_npz_as_npz(tmp_path: Path):
    output = tmp_path / "part_core.npz"
    output.write_bytes(b"npz")

    assert _core_output_download_path(output) == output


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


def test_ui_keeps_prusa_preview_and_exposes_core_export_progress():
    html = _index_html()

    assert "previewData = result.preview" in html
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
    assert '"feed_mm_s": 15.0' in html
    assert '"temperature_c": 240.0' in html
    assert '"prime_length_mm": 21.0' in html
    # The page loads the checked-in process preset, whose Prusa placement is
    # intentionally 20 mm from the X origin rather than the code fallback.
    assert '"start_x_mm": 20.0' in html
    assert 'fetch(\'/ui-settings\'' in html
    assert "const adjustValue = (targetInput, direction) =>" in html
    assert 'className = \'magnitudeInputWrap\'' in html
    assert 'className = \'magnitudeSpinButton\'' in html
    assert "targetInput.dispatchEvent(new Event('input'" in html
    assert '.processBand .actions {\n      display: flex;' in html
    assert 'grid-column: 1 / -1;' in html
    assert 'flex-wrap: nowrap;' in html


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
