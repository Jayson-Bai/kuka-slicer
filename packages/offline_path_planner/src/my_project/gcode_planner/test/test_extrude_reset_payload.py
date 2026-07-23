import json

import numpy as np

import path_processing_core.npz_exporter as npz_exporter
from gcode_planner.cli import _default_output_path
from path_processing_core.npz_exporter import export_npz
from path_processing_core.types import (
    ExtrudeWait,
    MCommand,
    MoveCommand,
    Position,
    ResetECommand,
    ToolChangeCommand,
)


def test_extrude_reset_payload_uses_current_tool_id(tmp_path):
    out = tmp_path / "reset_payload.npz"
    parsed = [
        ResetECommand(
            type="RESET_E",
            val=0.0,
            line=10,
            layer=0,
            subtype="Custom",
            raw="G92 E0",
        )
    ]

    export_npz(parsed, str(out))

    data = np.load(out)
    payload = data["payload"][0].decode("utf-8").rstrip("\x00")
    event_type_vocab = {
        int(value): key.decode("utf-8").rstrip("\x00")
        for key, value in zip(
            data["event_type_vocab_keys"],
            data["event_type_vocab_vals"],
        )
    }

    assert event_type_vocab[int(data["event_type"][0])] == "extrude_reset"
    assert payload == "2"


def test_extrude_wait_exports_stationary_e_change(tmp_path):
    out = tmp_path / "extrude_wait.npz"
    parsed = [
        MoveCommand(
            type="TRAVEL",
            cmd="G0",
            start_pos=Position(1.0, 2.0, 3.0, 0.0, 0.0, 0.0),
            pos=Position(1.0, 2.0, 3.0, 0.0, 0.0, 0.0),
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="TRAVEL",
            raw="G0 X1 Y2 Z3",
            is_pure_state_change=False,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.5,
            delta_e=1.0,
            feedrate=120.0,
            line=2,
            raw="G1 E1 F120",
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=0.1,
        default_feed_mm_s=10.0,
        enable_extrude_wait=True,
    )

    data = np.load(out)
    assert data["x"][-1] == data["x"][-2]
    assert data["y"][-1] == data["y"][-2]
    assert data["z"][-1] == data["z"][-2]
    assert data["e"][-1] > data["e"][-2]
    assert np.isclose(data["e"][-1], 1.0)


def test_npz_export_includes_layer_progress_metadata(tmp_path):
    out = tmp_path / "layer_progress.npz"
    parsed = [
        MoveCommand(
            type="TRAVEL",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="TRAVEL",
            raw="G1 X1 F600",
            is_pure_state_change=False,
        ),
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(2.0, 0.0, 0.2, 0.0, 0.0, 0.0),
            e_val=1.0,
            delta_e=1.0,
            feedrate=600.0,
            line=2,
            layer=2,
            subtype="Perimeter",
            raw="G1 X2 Z0.2 E1 F600",
            is_pure_state_change=False,
        ),
    ]

    export_npz(parsed, str(out), dt=0.1, default_feed_mm_s=10.0)

    data = np.load(out)
    assert "layer_index" in data.files
    assert "total_layers" in data.files
    assert data["layer_index"][0] == 0
    assert data["layer_index"][-1] == 2
    assert data["total_layers"][0] == 3
    assert data["total_layers"][-1] == 3


def test_npz_export_includes_path_metadata_for_print_segments(tmp_path):
    out = tmp_path / "path_metadata.npz"
    parsed = [
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.5,
            delta_e=0.5,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="Perimeter",
            raw="G1 X1 E0.5 F600",
            is_pure_state_change=False,
        ),
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=1.0,
            delta_e=0.5,
            feedrate=600.0,
            line=2,
            layer=0,
            subtype="Perimeter",
            raw="G1 X2 E1.0 F600",
            is_pure_state_change=False,
        ),
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(2.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(3.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            e_val=1.5,
            delta_e=0.5,
            feedrate=600.0,
            line=3,
            layer=0,
            subtype="Infill",
            raw="G1 X3 Y1 E1.5 F600",
            is_pure_state_change=False,
        ),
    ]

    export_npz(parsed, str(out), dt=0.1, default_feed_mm_s=10.0)

    data = np.load(out)
    assert "path_id" in data.files
    assert "path_end_flag" in data.files
    assert "move_type" in data.files
    path_ids = data["path_id"]
    end_flags = data["path_end_flag"]
    assert path_ids[0] > 0
    assert len(set(path_ids.tolist())) == 2
    first_path = path_ids[0]
    first_path_indices = np.where(path_ids == first_path)[0]
    assert end_flags[first_path_indices[-1]] == 1
    assert np.all(end_flags[first_path_indices[:-1]] == 0)
    assert end_flags[-1] == 1
    for path_id in sorted(set(path_ids.tolist())):
        indices = np.where(path_ids == path_id)[0]
        assert int(np.sum(end_flags[indices])) == 1
        assert end_flags[indices[-1]] == 1


def test_path_end_flag_marks_only_final_row_after_segment_extrude_wait(tmp_path):
    out = tmp_path / "single_safe_pause_point.npz"
    parsed = [
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.5,
            delta_e=0.5,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="Perimeter",
            raw="G1 X1 E0.5 F600",
            is_pure_state_change=False,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.2,
            delta_e=-0.2,
            feedrate=120.0,
            line=2,
            layer=0,
            subtype="Perimeter",
            raw="G1 E0.3 F120",
        ),
    ]

    export_npz(parsed, str(out), dt=0.1, default_feed_mm_s=10.0)

    data = np.load(out)
    path_ids = data["path_id"]
    end_flags = data["path_end_flag"]
    nonzero_ids = sorted(set(int(v) for v in path_ids if int(v) > 0))
    assert nonzero_ids == [int(path_ids[0])]
    indices = np.where(path_ids == nonzero_ids[0])[0]
    assert int(np.sum(end_flags[indices])) == 1
    assert end_flags[indices[-1]] == 1


def _decoded_src_lines(data):
    return [item.decode("utf-8").rstrip("\x00") for item in data["src_line"]]


def _decoded_event_type_vocab(data):
    return {
        int(value): key.decode("utf-8").rstrip("\x00")
        for key, value in zip(
            data["event_type_vocab_keys"],
            data["event_type_vocab_vals"],
        )
    }


def _move(start, end, e_start, e_end, line, subtype="WALL-OUTER"):
    return MoveCommand(
        type="PRINT",
        cmd="G1",
        start_pos=Position(start[0], start[1], start[2], 0.0, 0.0, 0.0),
        pos=Position(end[0], end[1], end[2], 0.0, 0.0, 0.0),
        e_val=e_end,
        delta_e=e_end - e_start,
        feedrate=600.0,
        line=line,
        layer=0,
        subtype=subtype,
        raw=f"G1 X{end[0]} Y{end[1]} E{e_end}",
        is_pure_state_change=False,
    )


def test_wall_outline_export_preserves_original_polyline_without_spline_range(tmp_path):
    out = tmp_path / "wall_outline.npz"
    parsed = [
        _move((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), 0.0, 2.0, 1),
        _move((20.0, 0.0, 0.0), (40.0, 0.0, 0.0), 2.0, 4.0, 2),
        _move((40.0, 0.0, 0.0), (40.15, 0.01, 0.0), 4.0, 4.02, 3),
        _move((40.15, 0.01, 0.0), (40.28, 0.02, 0.0), 4.02, 4.04, 4),
        _move((40.28, 0.02, 0.0), (40.4, 0.02, 0.0), 4.04, 4.06, 5),
        _move((40.4, 0.02, 0.0), (40.4, 10.0, 0.0), 4.06, 5.0, 6),
    ]

    export_npz(parsed, str(out), dt=0.1, default_feed_mm_s=10.0)

    data = np.load(out)
    src_lines = set(_decoded_src_lines(data))
    assert len(src_lines) < len(parsed)
    assert len(data["x"]) < 130
    assert np.min(data["x"]) >= -1e-4
    assert np.max(data["x"]) <= 40.4001
    assert np.min(data["y"]) >= -1e-4
    assert np.max(data["y"]) <= 10.0001


def test_wall_outline_many_short_segments_use_single_polyline_time_profile(tmp_path):
    out = tmp_path / "wall_polyline.npz"
    points = [
        (10.0, 0.0, 0.0),
        (9.8, 2.0, 0.0),
        (9.2, 3.9, 0.0),
        (8.3, 5.6, 0.0),
        (7.1, 7.1, 0.0),
        (5.6, 8.3, 0.0),
        (3.9, 9.2, 0.0),
        (2.0, 9.8, 0.0),
        (0.0, 10.0, 0.0),
    ]
    parsed = []
    start = (10.0, -2.0, 0.0)
    e_start = 0.0
    for index, end in enumerate(points, start=1):
        e_end = e_start + 0.2
        parsed.append(_move(start, end, e_start, e_end, index))
        start = end
        e_start = e_end

    export_npz(parsed, str(out), dt=0.1, default_feed_mm_s=10.0)

    data = np.load(out)
    assert len(data["x"]) < 120
    assert np.min(data["x"]) >= -1e-4
    assert np.max(data["x"]) <= 10.0001
    assert np.min(data["y"]) >= -2.0001
    assert np.max(data["y"]) <= 10.0001
    assert np.isclose(data["e"][-1], e_start)


def test_short_print_cluster_uses_single_polyline_time_profile(tmp_path):
    out = tmp_path / "short_cluster_polyline.npz"
    parsed = [
        _move((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), 0.0, 0.08, 1, subtype="SKIN"),
        _move((0.2, 0.0, 0.0), (0.4, 0.0, 0.0), 0.08, 0.16, 2, subtype="SKIN"),
        _move((0.4, 0.0, 0.0), (0.6, 0.0, 0.0), 0.16, 0.24, 3, subtype="SKIN"),
        _move((0.6, 0.0, 0.0), (0.8, 0.0, 0.0), 0.24, 0.32, 4, subtype="SKIN"),
        _move((0.8, 0.0, 0.0), (12.0, 0.0, 0.0), 0.32, 4.8, 5, subtype="SKIN"),
    ]

    export_npz(parsed, str(out), dt=0.1, default_feed_mm_s=10.0)

    data = np.load(out)
    src_lines = _decoded_src_lines(data)
    assert len(data["x"]) < 125
    assert src_lines.count("1-4") > 1
    assert np.isclose(data["e"][-1], 4.8)


def test_cut_event_lifts_with_matched_fiber_feed_and_preserves_next_prime(tmp_path):
    out = tmp_path / "cut_lift_sequence.npz"
    parsed = [
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=5.0,
            delta_e=5.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="FIBER",
            raw="fiber_print",
        ),
        MCommand(
            type="M_COMMAND",
            code="CUT",
            params={"P": 1.0},
            line=2,
            layer=0,
            subtype="FIBER",
            raw="external_npz_cut",
            tool=1,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.5,
            delta_e=-2.0,
            feedrate=300.0,
            line=3,
            layer=0,
            subtype="FIBER",
            raw="external_npz_retract",
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.5,
            delta_e=2.0,
            feedrate=300.0,
            line=4,
            layer=0,
            subtype="FIBER",
            raw="external_npz_prime",
        ),
        MoveCommand(
            type="TRAVEL",
            cmd="G0",
            start_pos=Position(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(20.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=5.0,
            delta_e=0.0,
            feedrate=600.0,
            line=5,
            layer=0,
            subtype="TRAVEL",
            raw="next_path_travel",
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.5,
            delta_e=3.0,
            feedrate=300.0,
            line=6,
            layer=0,
            subtype="FIBER",
            raw="next_path_prime",
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=1.0,
        default_feed_mm_s=10.0,
        enable_extrude_wait=True,
        cut_lift_mm=20.0,
        cut_wait_s=5.0,
        initial_tool_id=1,
    )

    data = np.load(out)
    src_lines = _decoded_src_lines(data)
    event_vocab = _decoded_event_type_vocab(data)
    event_types = [event_vocab[int(value)] for value in data["event_type"]]

    cut_idx = event_types.index("cut")
    assert np.isclose(data["x"][cut_idx], 10.0)
    assert np.isclose(data["z"][cut_idx], 0.0)
    assert np.isclose(data["e"][cut_idx], 5.0)

    lift_idx = [idx for idx, src in enumerate(src_lines) if src == "2" and idx > cut_idx]
    assert lift_idx
    assert np.isclose(np.max(data["z"][lift_idx]), 20.0)
    assert np.isclose(np.max(data["e"][lift_idx]), 25.0)

    high_hold_idx = [
        idx
        for idx in lift_idx
        if np.isclose(data["z"][idx], 20.0) and np.isclose(data["e"][idx], 25.0)
    ]
    assert len(high_hold_idx) >= 3

    safety_retract_idx = [
        idx
        for idx in lift_idx
        if idx > high_hold_idx[-1]
        and np.isclose(data["z"][idx], 20.0)
        and data["e"][idx] < 25.0
    ]
    assert safety_retract_idx
    assert np.isclose(data["e"][safety_retract_idx[-1]], 5.0)

    post_retract_idx = src_lines.index("3")
    post_prime_idx = src_lines.index("4")
    assert post_retract_idx > safety_retract_idx[-1]
    assert post_prime_idx > post_retract_idx
    assert np.isclose(data["e"][post_retract_idx], 3.0)
    assert np.isclose(data["e"][post_prime_idx], 5.0)

    travel_idx = [idx for idx, src in enumerate(src_lines) if src == "5"]
    assert travel_idx
    assert np.isclose(data["z"][travel_idx[0]], 20.0)
    assert np.isclose(data["z"][travel_idx[-1]], 0.0)
    assert np.isclose(data["e"][travel_idx[0]], 5.0)
    assert np.isclose(data["e"][travel_idx[-1]], 5.0)
    next_prime_idx = src_lines.index("6")
    assert next_prime_idx > travel_idx[-1]
    assert np.isclose(data["x"][next_prime_idx], data["x"][travel_idx[-1]])
    assert np.isclose(data["z"][next_prime_idx], data["z"][travel_idx[-1]])
    assert np.isclose(data["e"][next_prime_idx], 8.0)


def test_prime_extrude_wait_runs_after_previous_travel_by_default(tmp_path):
    out = tmp_path / "prime_after_travel.npz"
    parsed = [
        MoveCommand(
            type="TRAVEL",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="TRAVEL",
            raw="G1 X10 F600",
            is_pure_state_change=False,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=1.0,
            delta_e=4.0,
            feedrate=240.0,
            line=2,
            raw="G1 E4 F240",
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=1.0,
        default_feed_mm_s=10.0,
        enable_extrude_wait=True,
    )

    data = np.load(out)
    src_lines = _decoded_src_lines(data)
    travel_idx = [idx for idx, src in enumerate(src_lines) if src == "1"]
    extrude_wait_idx = [idx for idx, src in enumerate(src_lines) if src == "2"]

    assert extrude_wait_idx
    assert extrude_wait_idx[0] > travel_idx[-1]
    assert np.isclose(data["e"][travel_idx[0]], 0.0)
    assert np.isclose(data["e"][travel_idx[-1]], 0.0)
    assert np.isclose(data["x"][extrude_wait_idx[0]], data["x"][travel_idx[-1]])
    assert np.isclose(data["z"][extrude_wait_idx[0]], data["z"][travel_idx[-1]])
    assert np.isclose(data["e"][extrude_wait_idx[-1]], 4.0)


def test_retract_wait_overlaps_previous_travel_before_reset_by_default(
        tmp_path):
    out = tmp_path / "retract_overlap.npz"
    parsed = [
        MoveCommand(
            type="TRAVEL",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=10.0,
            delta_e=0.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="TRAVEL",
            raw="G1 X10 F600",
            is_pure_state_change=False,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.5,
            delta_e=-3.0,
            feedrate=360.0,
            line=2,
            raw="G1 E7 F360",
        ),
        ResetECommand(
            type="RESET_E",
            val=0.0,
            line=3,
            layer=0,
            subtype="TRAVEL",
            raw="G92 E0",
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=1.0,
        default_feed_mm_s=10.0,
        enable_extrude_wait=True,
    )

    data = np.load(out)
    src_lines = _decoded_src_lines(data)
    travel_idx = [idx for idx, src in enumerate(src_lines) if src == "1"]
    event_type_vocab = _decoded_event_type_vocab(data)
    reset_idx = src_lines.index("3")

    assert "2" not in src_lines
    assert np.isclose(data["e"][travel_idx[0]], 10.0)
    assert np.isclose(data["e"][travel_idx[-1]], 7.0)
    assert (
        event_type_vocab[int(data["event_type"][reset_idx])]
        == "extrude_reset"
    )
    assert np.isclose(data["e"][reset_idx], 7.0)


def test_retract_across_reset_keeps_existing_extrude_wait_rows(tmp_path):
    out = tmp_path / "cross_reset_retract.npz"
    parsed = [
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=10.0,
            delta_e=10.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="Perimeter",
            raw="G1 X10 E10 F600",
            is_pure_state_change=False,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.5,
            delta_e=-3.0,
            feedrate=360.0,
            line=2,
            raw="G1 E7 F360",
        ),
        ResetECommand(
            type="RESET_E",
            val=0.0,
            line=3,
            layer=0,
            subtype="Perimeter",
            raw="G92 E0",
        ),
        MoveCommand(
            type="TRAVEL",
            cmd="G1",
            start_pos=Position(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(12.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=4,
            layer=0,
            subtype="TRAVEL",
            raw="G1 X12 F600",
            is_pure_state_change=False,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=1.0,
            delta_e=4.0,
            feedrate=240.0,
            line=5,
            raw="G1 E4 F240",
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=1.0,
        default_feed_mm_s=10.0,
        enable_extrude_wait=True,
    )

    data = np.load(out)
    src_lines = _decoded_src_lines(data)
    retract_idx = src_lines.index("2")
    travel_idx = [idx for idx, src in enumerate(src_lines) if src == "4"]
    prime_idx = [idx for idx, src in enumerate(src_lines) if src == "5"]

    assert "2" in src_lines
    assert prime_idx
    assert prime_idx[0] > travel_idx[-1]
    assert np.isclose(data["e"][retract_idx], 7.0)
    assert np.isclose(data["e"][travel_idx[0]], 0.0)
    assert np.isclose(data["e"][travel_idx[-1]], 0.0)
    assert np.isclose(data["x"][prime_idx[0]], data["x"][travel_idx[-1]])
    assert np.isclose(data["z"][prime_idx[0]], data["z"][travel_idx[-1]])
    assert np.isclose(data["e"][prime_idx[-1]], 4.0)


def test_travel_extrude_overlap_can_be_disabled(tmp_path):
    out = tmp_path / "prime_overlap_disabled.npz"
    parsed = [
        MoveCommand(
            type="TRAVEL",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="TRAVEL",
            raw="G1 X10 F600",
            is_pure_state_change=False,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=1.0,
            delta_e=4.0,
            feedrate=240.0,
            line=2,
            raw="G1 E4 F240",
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=1.0,
        default_feed_mm_s=10.0,
        enable_extrude_wait=True,
        enable_travel_extrude_overlap=False,
    )

    data = np.load(out)
    src_lines = _decoded_src_lines(data)
    travel_idx = [idx for idx, src in enumerate(src_lines) if src == "1"]
    extrude_wait_idx = [idx for idx, src in enumerate(src_lines) if src == "2"]

    assert extrude_wait_idx
    assert np.isclose(data["e"][travel_idx[-1]], 0.0)
    assert np.isclose(data["e"][extrude_wait_idx[-1]], 4.0)


def test_cli_enables_extrude_wait_for_formal_exports():
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] /
        "gcode_planner" /
        "cli.py").read_text(
        encoding="utf-8")
    export_call = src.split(
        "    stats = export_npz(", 1)[1].split(
        "    t3 = time.perf_counter()", 1)[0]

    assert "enable_extrude_wait=True" in export_call
    assert (
        "resin_z_print_compensation_mm=args.resin_z_print_compensation_mm"
        in export_call
    )


def test_export_npz_safely_lifts_before_fiber_tool_offset_and_then_changes_tool(tmp_path):
    out = tmp_path / "safe_fiber_offset.npz"
    parsed = [
        ToolChangeCommand(
            type="TOOL_CHANGE",
            tool=0,
            line=1,
            layer=0,
            subtype="TRAVEL",
        )
    ]

    export_npz(
        parsed,
        str(out),
        dt=0.1,
        default_feed_mm_s=10.0,
        tool_offset=(5.0, 4.0, -25.0),
        resin_z_print_compensation_mm=-20.0,
    )

    data = np.load(out)
    event_vocab = _decoded_event_type_vocab(data)
    event_types = [event_vocab[int(value)] for value in data["event_type"]]
    tool_change_idx = event_types.index("tool_change_cf")

    assert np.any(np.isclose(data["z"], -20.0, atol=1e-4))
    non_event = data["event_flag"] == 0
    safe_lift_idx = np.where(
        non_event
        & np.isclose(data["x"], 0.0, atol=1e-4)
        & np.isclose(data["y"], 0.0, atol=1e-4)
        & np.isclose(data["z"], 0.0, atol=1e-4)
    )[0]
    offset_idx = np.where(
        non_event
        & np.isclose(data["x"], 5.0, atol=1e-4)
        & np.isclose(data["y"], 4.0, atol=1e-4)
        & np.isclose(data["z"], -25.0, atol=1e-4)
    )[0]

    assert len(safe_lift_idx) > 0
    assert len(offset_idx) > 0
    assert safe_lift_idx[-1] < offset_idx[-1] < tool_change_idx


def test_export_npz_defaults_to_resin_tool_without_initial_tool_change(tmp_path):
    out = tmp_path / "default_resin_tool.npz"
    parsed = [
        ResetECommand(
            type="RESET_E",
            val=0.0,
            line=1,
            layer=0,
            subtype="RESIN_PRINT",
            raw="G92 E0",
        )
    ]

    export_npz(parsed, str(out), dt=0.1, default_feed_mm_s=10.0)

    data = np.load(out)
    event_vocab = _decoded_event_type_vocab(data)
    event_types = [event_vocab[int(value)] for value in data["event_type"]]
    non_empty_events = [event for event in event_types if event]

    assert non_empty_events == ["extrude_reset"]
    assert int(data["tool_id"][0]) == 2


def test_export_npz_records_resin_z_compensation_sidecar(tmp_path):
    import json

    out = tmp_path / "z_comp.npz"
    parsed = [
        ToolChangeCommand(
            type="TOOL_CHANGE",
            tool=1,
            line=0,
            layer=0,
            subtype="TRAVEL",
        ),
        MoveCommand(
            type="TRAVEL",
            cmd="G0",
            start_pos=Position(10.0, 20.0, 5.0, 1.0, 2.0, 3.0),
            pos=Position(10.0, 20.0, 5.0, 1.0, 2.0, 3.0),
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="TRAVEL",
            raw="G0 X10 Y20 Z5",
            is_pure_state_change=False,
        ),
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(10.0, 20.0, 5.0, 1.0, 2.0, 3.0),
            pos=Position(20.0, 20.0, 5.0, 1.0, 2.0, 3.0),
            e_val=1.0,
            delta_e=1.0,
            feedrate=600.0,
            line=2,
            layer=0,
            subtype="Perimeter",
            raw="G1 X20 E1",
            is_pure_state_change=False,
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=0.1,
        default_feed_mm_s=10.0,
        resin_z_print_compensation_mm=-2.0,
    )

    data = np.load(out)
    assert data["move_type"][0] == 0
    assert np.isclose(data["x"][0], 0.0)
    assert np.isclose(data["y"][0], 0.0)
    assert np.isclose(data["z"][0], 0.0)
    before_extrude = data["e"] <= 1e-9
    assert np.any(np.isclose(data["z"][before_extrude], -2.0, atol=1e-4))
    assert np.any(np.isclose(data["z"][before_extrude], 3.0, atol=1e-4))

    sidecar = json.loads(
        out.with_suffix(".offset.json").read_text(encoding="utf-8")
    )
    assert sidecar["resin_z_print_compensation_mm"] == -2.0


def test_export_npz_applies_external_start_travel_before_resin_z_compensation(tmp_path):
    out = tmp_path / "external_start_before_z_comp.npz"
    parsed = [
        MCommand(
            type="M_COMMAND",
            code="M104",
            params={"S": 180.0},
            line=1,
            layer=0,
            subtype="TRAVEL",
            raw="M104 S180",
            tool=1,
        ),
        MoveCommand(
            type="TRAVEL",
            cmd="G0",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(50.0, 50.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=2,
            layer=0,
            subtype="TRAVEL",
            raw="external_npz_start_xy_travel",
            is_pure_state_change=False,
        ),
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(50.0, 50.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(60.0, 50.0, 0.0, 0.0, 0.0, 0.0),
            e_val=1.0,
            delta_e=1.0,
            feedrate=600.0,
            line=3,
            layer=0,
            subtype="RESIN_PRINT",
            raw="G1 X60 Y50 E1",
            is_pure_state_change=False,
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=0.1,
        default_feed_mm_s=10.0,
        resin_z_print_compensation_mm=-2.0,
    )

    data = np.load(out)
    start_travel_idx = np.where(
        (data["event_flag"] == 0)
        & (data["move_type"] == 0)
        & np.isclose(data["x"], 50.0, atol=1e-4)
        & np.isclose(data["y"], 50.0, atol=1e-4)
        & np.isclose(data["z"], 0.0, atol=1e-4)
    )[0]
    resin_comp_idx = np.where(
        (data["event_flag"] == 0)
        & (data["move_type"] == 0)
        & np.isclose(data["x"], 50.0, atol=1e-4)
        & np.isclose(data["y"], 50.0, atol=1e-4)
        & np.isclose(data["z"], -2.0, atol=1e-4)
    )[0]

    assert len(start_travel_idx) > 0
    assert len(resin_comp_idx) > 0
    assert int(start_travel_idx[-1]) < int(resin_comp_idx[0])


def test_flat_export_generates_layer_preview_without_manifest(tmp_path):
    out = tmp_path / "flat_preview.npz"
    parsed = [
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=1.0,
            delta_e=1.0,
            feedrate=600.0,
            line=1,
            layer=1,
            subtype="Perimeter",
            raw="G1 X10 E1",
            is_pure_state_change=False,
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=0.1,
        default_feed_mm_s=10.0,
        split_by_layer_type=False,
        plot_layer_xy=True,
        plot_stride=1,
    )

    assert out.exists()
    manifest_path = tmp_path / "flat_preview" / "flat_preview_manifest.json"
    assert not manifest_path.exists()
    preview_path = (
        tmp_path / "flat_preview" / "layer_previews" / "layer_0001.png"
    )
    assert preview_path.exists()


def test_flat_preview_breaks_lines_across_travel_and_prime(tmp_path, monkeypatch):
    out = tmp_path / "flat_preview_break.npz"
    captured = {}

    def capture_plot(layer_points, base_root):
        captured["layer_points"] = layer_points
        captured["base_root"] = base_root

    monkeypatch.setattr(npz_exporter, "_plot_flat_layer_previews", capture_plot)

    parsed = [
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=1.0,
            delta_e=1.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="SKIN",
            raw="G1 X1 E1",
            is_pure_state_change=False,
        ),
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.5,
            delta_e=-0.5,
            feedrate=2400.0,
            line=2,
            layer=0,
            subtype="SKIN",
            raw="G1 E0.5",
            is_pure_state_change=True,
        ),
        MoveCommand(
            type="TRAVEL",
            cmd="G0",
            start_pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(100.0, 100.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.5,
            delta_e=0.0,
            feedrate=600.0,
            line=3,
            layer=0,
            subtype="TRAVEL",
            raw="G0 X100 Y100",
            is_pure_state_change=False,
        ),
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(100.0, 100.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(100.0, 100.0, 0.0, 0.0, 0.0, 0.0),
            e_val=1.5,
            delta_e=1.0,
            feedrate=900.0,
            line=4,
            layer=0,
            subtype="WALL-INNER",
            raw="G1 E1.5",
            is_pure_state_change=True,
        ),
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(100.0, 100.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(101.0, 100.0, 0.0, 0.0, 0.0, 0.0),
            e_val=2.5,
            delta_e=1.0,
            feedrate=600.0,
            line=5,
            layer=0,
            subtype="WALL-INNER",
            raw="G1 X101 E2.5",
            is_pure_state_change=False,
        ),
    ]

    export_npz(
        parsed,
        str(out),
        dt=0.1,
        default_feed_mm_s=10.0,
        split_by_layer_type=False,
        plot_layer_xy=True,
        plot_stride=1,
        enable_extrude_wait=False,
    )

    xs, ys = captured["layer_points"][0]
    points = list(zip(xs, ys))
    assert any(np.isnan(x) and np.isnan(y) for x, y in points)
    assert not any(
        np.isclose(x1, 1.0)
        and np.isclose(y1, 0.0)
        and np.isclose(x2, 100.0)
        and np.isclose(y2, 100.0)
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    )


def test_default_output_path_places_npz_inside_named_output_directory(tmp_path):
    gcode = tmp_path / "100_10_cylinder_624.gcode"
    output_dir = tmp_path / "output_npz"

    assert _default_output_path(str(gcode), str(output_dir)) == str(
        output_dir / "100_10_cylinder_624" / "100_10_cylinder_624.npz"
    )


def test_nested_flat_export_uses_parent_directory_for_previews(tmp_path, monkeypatch):
    out = tmp_path / "output_npz" / "sample" / "sample.npz"
    captured = {}

    def capture_plot(layer_points, base_root):
        captured["base_root"] = base_root

    monkeypatch.setattr(npz_exporter, "_plot_flat_layer_previews", capture_plot)

    parsed = [
        MoveCommand(
            type="PRINT",
            cmd="G1",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=1.0,
            delta_e=1.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="WALL",
            raw="G1 X1 E1",
            is_pure_state_change=False,
        )
    ]

    export_npz(
        parsed,
        str(out),
        dt=0.1,
        default_feed_mm_s=10.0,
        plot_layer_xy=True,
    )

    assert out.exists()
    assert (out.parent / "sample.offset.json").exists()
    assert captured["base_root"] == str(out.parent)



def _commands_with_travel_print_and_event():
    return [
        MoveCommand(
            type="TRAVEL",
            cmd="G0",
            start_pos=Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            pos=Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=1,
            layer=0,
            subtype="TRAVEL",
            raw="G0 X1 F600",
            is_pure_state_change=False,
        ),
        _move((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), 0.0, 1.0, 2),
        ResetECommand(
            type="RESET_E",
            val=0.0,
            line=3,
            layer=0,
            subtype="Perimeter",
            raw="G92 E0",
        ),
    ]


def test_export_npz_writes_rsi_timing_array_and_sidecar(tmp_path):
    out = tmp_path / "timed.npz"
    stats = export_npz(_commands_with_travel_print_and_event(), str(out), dt=0.1)

    with np.load(out) as data:
        assert "planned_time_s" in data
        assert len(data["planned_time_s"]) == len(data["seq"])
        assert np.all(np.isfinite(data["planned_time_s"]))
        event_rows = data["event_flag"] == 1
        assert np.any(event_rows)
        assert all(
            data["planned_time_s"][i] == data["planned_time_s"][i - 1]
            for i in np.flatnonzero(event_rows)
        )

    metadata = json.loads(
        (tmp_path / "timed.timing.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "rsi_print_timing"
    assert metadata["total_planned_time_s"] >= 0.0
    assert metadata["event_rows_ignored"] >= 1
    assert metadata["segments"]
    assert {"t_acc_s", "t_flat_s", "t_dec_s"}.issubset(
        metadata["segments"][0])
    assert stats["timing_sidecar"] == str(tmp_path / "timed.timing.json")
