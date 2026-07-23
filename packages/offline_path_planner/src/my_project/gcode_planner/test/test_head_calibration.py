from datetime import datetime
import json

from path_processing_core.head_calibration import (
    DEFAULT_HEAD_CALIBRATION,
    DEFAULT_DATA_ROOT,
    DEFAULT_HEAD_CALIBRATION_PATH,
    default_data_root,
    HeadCalibration,
    calibration_relative_offsets,
    load_head_calibration,
    save_head_calibration,
)


def test_load_head_calibration_returns_defaults_when_file_missing(tmp_path):
    path = tmp_path / "head_offsets.json"

    cal = load_head_calibration(path)

    assert cal.resin_z_print_compensation_mm == 0.0
    assert cal.fiber_x_print_compensation_mm == 0.0
    assert cal.fiber_y_print_compensation_mm == 0.0
    assert cal.fiber_z_print_compensation_mm == 0.0
    assert DEFAULT_HEAD_CALIBRATION["resin"]["z_print_compensation_mm"] == 0.0
    assert DEFAULT_HEAD_CALIBRATION["fiber"]["z_offset_mm"] == 0.0


def test_default_head_calibration_path_uses_workspace_data_dir():
    assert DEFAULT_HEAD_CALIBRATION_PATH == (
        DEFAULT_DATA_ROOT / "head_calibration_offsets" / "head_offsets.json"
    )


def test_default_data_root_is_stable_when_cwd_changes(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    package_dir = workspace / "src" / "my_project" / "gcode_planner"
    package_dir.mkdir(parents=True)
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()

    monkeypatch.chdir(other_cwd)

    assert default_data_root(start_paths=(package_dir,)) == workspace / "data"


def test_load_head_calibration_returns_defaults_for_malformed_json(tmp_path):
    path = tmp_path / "head_offsets.json"
    path.write_text("{not valid json", encoding="utf-8")

    assert load_head_calibration(path) == HeadCalibration()


def test_load_head_calibration_preserves_present_fields_and_defaults_missing_fields(
    tmp_path,
):
    path = tmp_path / "head_offsets.json"
    path.write_text(
        json.dumps(
            {
                "resin": {"z_print_compensation_mm": "-2.5"},
                "fiber": {"x_print_compensation_mm": 3.5},
            }
        ),
        encoding="utf-8",
    )

    cal = load_head_calibration(path)

    assert cal == HeadCalibration(
        resin_z_print_compensation_mm=-2.5,
        fiber_x_print_compensation_mm=3.5,
        fiber_y_print_compensation_mm=0.0,
        fiber_z_print_compensation_mm=0.0,
    )


def test_load_head_calibration_accepts_legacy_fiber_z_print_compensation_key(tmp_path):
    path = tmp_path / "head_offsets.json"
    path.write_text(
        json.dumps(
            {
                "resin": {"z_print_compensation_mm": -2.5},
                "fiber": {"z_print_compensation_mm": -7.5},
            }
        ),
        encoding="utf-8",
    )

    cal = load_head_calibration(path)

    assert cal.fiber_z_print_compensation_mm == -7.5


def test_save_head_calibration_overwrites_current_file_with_timestamp(tmp_path):
    path = tmp_path / "head_offsets.json"
    save_head_calibration(
        HeadCalibration(
            resin_z_print_compensation_mm=-20.0,
            fiber_x_print_compensation_mm=5.0,
            fiber_y_print_compensation_mm=4.0,
            fiber_z_print_compensation_mm=-25.0,
        ),
        path,
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert datetime.fromisoformat(data["updated_at"])
    assert data["resin"] == {"z_print_compensation_mm": -20.0}
    assert data["fiber"] == {
        "x_print_compensation_mm": 5.0,
        "y_print_compensation_mm": 4.0,
        "z_offset_mm": -25.0,
    }

    save_head_calibration(HeadCalibration(resin_z_print_compensation_mm=-1.0), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["resin"]["z_print_compensation_mm"] == -1.0
    assert data["fiber"]["x_print_compensation_mm"] == 0.0


def test_save_head_calibration_round_trips_through_load_head_calibration(tmp_path):
    path = tmp_path / "head_offsets.json"
    expected = HeadCalibration(
        resin_z_print_compensation_mm=-20.0,
        fiber_x_print_compensation_mm=5.0,
        fiber_y_print_compensation_mm=4.0,
        fiber_z_print_compensation_mm=-25.0,
    )

    save_head_calibration(expected, path)

    assert load_head_calibration(path) == expected


def test_calibration_relative_offsets_stack_resin_z_with_direct_fiber_head_offset():
    cal = HeadCalibration(
        resin_z_print_compensation_mm=-20.0,
        fiber_x_print_compensation_mm=3.0,
        fiber_y_print_compensation_mm=2.0,
        fiber_z_print_compensation_mm=4.0,
    )

    assert calibration_relative_offsets(cal, from_tool="resin", to_tool="fiber") == (
        3.0,
        2.0,
        -16.0,
    )
    assert calibration_relative_offsets(cal, from_tool="fiber", to_tool="resin") == (
        -3.0,
        -2.0,
        16.0,
    )
