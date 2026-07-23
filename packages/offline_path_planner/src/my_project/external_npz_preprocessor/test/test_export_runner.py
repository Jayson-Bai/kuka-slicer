from pathlib import Path

from external_npz_preprocessor.export_runner import (
    default_output_npz_dir,
    default_output_path_for_source,
    default_source_npz_template_dir,
    resolve_output_path,
)


def test_empty_output_path_defaults_to_named_data_output_directory(tmp_path):
    source = Path("/tmp/example/source_file.npz")

    assert resolve_output_path(source, "", data_root=tmp_path) == (
        tmp_path / "output_npz" / "source_file" / "source_file.npz"
    )


def test_default_preprocessor_data_directories_are_under_data_root(tmp_path):
    assert default_source_npz_template_dir(tmp_path) == (
        tmp_path / "external_npz_preprocessor" / "source_npz_templates"
    )
    assert default_output_npz_dir(tmp_path) == tmp_path / "output_npz"
    assert default_output_path_for_source("part_a.npz", data_root=tmp_path) == (
        tmp_path / "output_npz" / "part_a" / "part_a.npz"
    )


def test_convert_uses_shared_head_calibration_offsets(tmp_path, monkeypatch):
    import json
    import numpy as np

    import external_npz_preprocessor.export_runner as runner
    from external_npz_preprocessor.process_params import ProcessParams

    source = tmp_path / "source.npz"
    resin_paths = np.array([[[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]]], dtype=np.float32)
    np.savez(source, meta=np.array(json.dumps({"format": "external_layer_paths_v1"})), layer_0000_R=resin_paths)
    calibration_path = tmp_path / "head_offsets.json"
    calibration_path.write_text(
        json.dumps(
            {
                "resin": {"z_print_compensation_mm": 1.25},
                "fiber": {
                    "x_print_compensation_mm": 2.0,
                    "y_print_compensation_mm": -3.0,
                    "z_offset_mm": 4.0,
                },
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_export_npz(commands, output_path, **kwargs):
        captured["kwargs"] = kwargs
        return {"rows": 0, "parts": 0, "total_s": 0.0}

    monkeypatch.setattr(runner, "export_npz", fake_export_npz)

    runner.convert_external_npz(
        source,
        tmp_path / "out.npz",
        ProcessParams(
            travel_feed_mm_s=7.0,
            first_layer_travel_feed_mm_s=2.0,
        ),
        calibration_path=calibration_path,
        cut_lift_mm=22.0,
        cut_wait_s=11.0,
    )

    assert captured["kwargs"]["tool_offset"] == (2.0, -3.0, 4.0)
    assert captured["kwargs"]["resin_z_print_compensation_mm"] == 1.25
    assert captured["kwargs"]["cut_lift_mm"] == 22.0
    assert captured["kwargs"]["cut_wait_s"] == 11.0
    assert captured["kwargs"]["default_feed_mm_s"] == 7.0
    assert "fiber_retract_length_mm" not in captured["kwargs"]
    assert captured["kwargs"]["external_npz_cut_absolute_e"] is True


def test_exporter_uses_curve_start_acceleration_without_changing_default(tmp_path, monkeypatch):
    import path_processing_core.npz_exporter as exporter
    from path_processing_core.polynomial_interpolator import InterpolatedPoint
    from path_processing_core.types import GlobalCurveCommand, Position

    captured = []

    def fake_sample_global_curve_iter(curve, **kwargs):
        captured.append((curve.raw, kwargs.get("t_acc")))
        yield InterpolatedPoint(
            t=0.0,
            pos=curve.start_pos,
            e=curve.e_val,
            extrude_speed=0.0,
            feedrate_mm_min=curve.feedrate,
            cmd_type=curve.type,
            line=curve.line,
            raw=curve.raw,
        )

    monkeypatch.setattr(
        exporter,
        "sample_global_curve_iter",
        fake_sample_global_curve_iter,
    )
    start = Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    end = Position(10.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    commands = [
        GlobalCurveCommand(
            type="PRINT",
            cmd="POLYLINE",
            start_pos=start,
            control_points=[end],
            e_val=1.0,
            delta_e=1.0,
            feedrate=600.0,
            line=1,
            raw="default_curve",
        ),
        GlobalCurveCommand(
            type="PRINT",
            cmd="POLYLINE",
            start_pos=start,
            control_points=[end],
            e_val=2.0,
            delta_e=1.0,
            feedrate=600.0,
            line=2,
            raw="fiber_curve",
            time_acc_s=4.5,
        ),
    ]

    exporter.export_npz(commands, str(tmp_path / "out.npz"), dt=0.004)

    assert captured == [("default_curve", None), ("fiber_curve", 4.5)]


def _decoded_src_lines(data):
    return [raw.decode("utf-8").rstrip("\x00") for raw in data["src_line"]]


def _decoded_event_types(data):
    vocab = {
        int(value): key.decode("utf-8").rstrip("\x00")
        for key, value in zip(data["event_type_vocab_keys"], data["event_type_vocab_vals"])
    }
    return [vocab[int(value)] for value in data["event_type"]]


def _next_src_line_group(src_lines, after_index):
    start = next(
        idx
        for idx in range(after_index + 1, len(src_lines))
        if src_lines[idx] != src_lines[after_index]
    )
    source = src_lines[start]
    end = next(
        (
            idx
            for idx in range(start + 1, len(src_lines))
            if src_lines[idx] != source
        ),
        len(src_lines),
    )
    return list(range(start, end))


def test_final_resin_layer_end_travel_is_last_runtime_trajectory_before_auto_abort(
    tmp_path,
):
    import json
    import numpy as np

    from external_npz_preprocessor.converter import source_job_to_parsed_commands
    from external_npz_preprocessor.export_runner import convert_external_npz
    from external_npz_preprocessor.process_params import ProcessParams
    from external_npz_preprocessor.source_npz import load_source_npz
    from path_processing_core.types import MoveCommand

    source = tmp_path / "final_resin_layer.npz"
    np.savez(
        source,
        meta=np.array(json.dumps({"format": "external_layer_paths_v1"})),
        layer_0000_R=np.array(
            [[[0.0, 0.0, 0.5], [10.0, 0.0, 0.5]]],
            dtype=np.float32,
        ),
    )
    calibration_path = tmp_path / "head_offsets.json"
    calibration_path.write_text(
        json.dumps(
            {
                "resin": {"z_print_compensation_mm": 0.0},
                "fiber": {
                    "x_print_compensation_mm": 0.0,
                    "y_print_compensation_mm": 0.0,
                    "z_offset_mm": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    params = ProcessParams()
    commands = source_job_to_parsed_commands(
        load_source_npz(source, default_abc=params.default_abc),
        params,
    )
    layer_end_travel = next(
        cmd
        for cmd in commands
        if isinstance(cmd, MoveCommand)
        and cmd.raw == "external_npz_resin_layer_end_travel"
    )

    output = tmp_path / "out.npz"
    convert_external_npz(
        source,
        output,
        params,
        calibration_path=calibration_path,
    )

    with np.load(output) as data:
        src_lines = _decoded_src_lines(data)
        travel_rows = [
            index
            for index, src_line in enumerate(src_lines)
            if src_line == str(layer_end_travel.line)
        ]

        assert travel_rows
        assert travel_rows[-1] == len(data["seq"]) - 1
        assert np.isclose(data["x"][travel_rows[-1]], 30.0)
        assert np.isclose(data["y"][travel_rows[-1]], 0.0)
        assert np.allclose(data["z"][travel_rows], 0.5)
        assert np.allclose(data["e"][travel_rows], 0.0)


def test_resin_layer_end_travel_is_exported_before_tool_change_safe_lift(tmp_path):
    import json
    import numpy as np

    from external_npz_preprocessor.converter import source_job_to_parsed_commands
    from external_npz_preprocessor.export_runner import convert_external_npz
    from external_npz_preprocessor.process_params import ProcessParams
    from external_npz_preprocessor.source_npz import load_source_npz
    from path_processing_core.types import MoveCommand, ToolChangeCommand

    source = tmp_path / "source.npz"
    np.savez(
        source,
        meta=np.array(json.dumps({"format": "external_layer_paths_v1"})),
        layer_0000_R=np.array(
            [[[0.0, 0.0, 0.5], [10.0, 0.0, 0.5]]],
            dtype=np.float32,
        ),
        layer_0000_F=np.array(
            [[[2.0, 1.0, 0.6], [8.0, 1.0, 0.6]]],
            dtype=np.float32,
        ),
    )
    calibration_path = tmp_path / "head_offsets.json"
    calibration_path.write_text(
        json.dumps(
            {
                "resin": {"z_print_compensation_mm": 0.0},
                "fiber": {
                    "x_print_compensation_mm": 1.0,
                    "y_print_compensation_mm": 0.0,
                    "z_offset_mm": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    params = ProcessParams()
    commands = source_job_to_parsed_commands(
        load_source_npz(source, default_abc=params.default_abc),
        params,
    )
    layer_end_travel = next(
        cmd
        for cmd in commands
        if isinstance(cmd, MoveCommand)
        and cmd.raw == "external_npz_resin_layer_end_travel"
    )
    travel_command_index = commands.index(layer_end_travel)
    fiber_tool_change = next(
        cmd
        for cmd in commands[travel_command_index + 1:]
        if isinstance(cmd, ToolChangeCommand) and cmd.tool == 0
    )

    output = tmp_path / "out.npz"
    convert_external_npz(
        source,
        output,
        params,
        calibration_path=calibration_path,
    )

    with np.load(output) as data:
        src_lines = _decoded_src_lines(data)
        events = _decoded_event_types(data)
        travel_rows = [
            index
            for index, src_line in enumerate(src_lines)
            if src_line == str(layer_end_travel.line)
        ]
        tool_change_rows = [
            index
            for index, src_line in enumerate(src_lines)
            if src_line == str(fiber_tool_change.line)
        ]

        assert travel_rows
        assert tool_change_rows
        assert max(travel_rows) < min(tool_change_rows)
        assert np.isclose(data["x"][travel_rows[-1]], 30.0)
        assert np.isclose(data["y"][travel_rows[-1]], 0.0)
        assert np.allclose(data["z"][travel_rows], 0.5)
        assert np.allclose(data["e"][travel_rows], 0.0)
        assert np.isclose(np.max(data["z"][tool_change_rows]), 20.5)
        assert events.index("tool_change_cf") in tool_change_rows


def test_external_npz_reset_anchor_starts_at_zero_without_changing_ordinary_holds(
    tmp_path,
):
    import numpy as np

    from path_processing_core.npz_exporter import export_npz
    from path_processing_core.types import (
        ExtrudeWait,
        MoveCommand,
        Position,
        ResetECommand,
    )

    dt = 0.004
    hold_position = Position(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    def export_after_reset(raw, filename):
        commands = [
            MoveCommand(
                type="TRAVEL",
                cmd="G0",
                start_pos=hold_position,
                pos=hold_position,
                e_val=0.0,
                delta_e=0.0,
                feedrate=600.0,
                line=1,
                layer=0,
                subtype="TRAVEL",
                raw="establish_hold_position",
                is_pure_state_change=False,
            ),
            ExtrudeWait(
                type="EXTRUDE_WAIT",
                wait_sec=dt,
                delta_e=5.0,
                feedrate=600.0,
                line=2,
                layer=0,
                subtype="TRAVEL",
                raw="accumulate_e",
            ),
            ResetECommand(
                type="RESET_E",
                val=0.0,
                line=3,
                layer=0,
                subtype="TRAVEL",
                raw="external_npz_path_reset",
            ),
            ExtrudeWait(
                type="EXTRUDE_WAIT",
                wait_sec=dt,
                delta_e=0.0,
                feedrate=600.0,
                line=4,
                layer=0,
                subtype="TRAVEL",
                raw=raw,
            ),
        ]
        output = tmp_path / filename
        export_npz(commands, str(output), dt=dt, enable_extrude_wait=True)
        return output

    anchor_output = export_after_reset("external_npz_reset_anchor", "anchor.npz")
    with np.load(anchor_output) as anchor_data:
        anchor_src_lines = _decoded_src_lines(anchor_data)
        anchor_event_types = _decoded_event_types(anchor_data)
        reset_idx = anchor_event_types.index("extrude_reset")
        anchor_idx = [
            idx for idx, source in enumerate(anchor_src_lines) if source == "4"
        ]

        assert np.isclose(anchor_data["e"][reset_idx], 5.0)
        assert len(anchor_idx) == 1
        assert np.isclose(anchor_data["e"][anchor_idx[0]], 0.0)
        for field in ("x", "y", "z", "a", "b", "c"):
            assert np.isclose(
                anchor_data[field][anchor_idx[0]],
                anchor_data[field][reset_idx],
            )

    near_match_output = export_after_reset(
        "external_npz_reset_anchor_suffix",
        "near_match.npz",
    )
    with np.load(near_match_output) as near_match_data:
        near_match_src_lines = _decoded_src_lines(near_match_data)
        near_match_idx = [
            idx for idx, source in enumerate(near_match_src_lines) if source == "4"
        ]

        assert len(near_match_idx) == 1
        assert np.isclose(near_match_data["e"][near_match_idx[0]], 5.0)


def test_external_npz_prime_settle_exports_125_stationary_rows(tmp_path):
    import numpy as np

    from path_processing_core.npz_exporter import export_npz
    from path_processing_core.types import ExtrudeWait, MoveCommand, Position

    dt = 0.004
    hold_position = Position(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    commands = [
        MoveCommand(
            type="TRAVEL",
            cmd="G0",
            start_pos=hold_position,
            pos=hold_position,
            e_val=0.0,
            delta_e=0.0,
            feedrate=600.0,
            line=10,
            layer=0,
            subtype="TRAVEL",
            raw="establish_hold_position",
            is_pure_state_change=False,
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=dt,
            delta_e=2.0,
            feedrate=600.0,
            line=11,
            layer=0,
            subtype="TRAVEL",
            raw="external_npz_prime",
        ),
        ExtrudeWait(
            type="EXTRUDE_WAIT",
            wait_sec=0.5,
            delta_e=0.0,
            feedrate=600.0,
            line=12,
            layer=0,
            subtype="TRAVEL",
            raw="external_npz_prime_settle",
        ),
    ]
    output = tmp_path / "prime_settle.npz"

    export_npz(commands, str(output), dt=dt, enable_extrude_wait=True)

    with np.load(output) as data:
        settle_idx = [
            idx for idx, source in enumerate(_decoded_src_lines(data)) if source == "12"
        ]
        assert len(settle_idx) == 125
        for field, expected in zip(
            ("x", "y", "z"),
            (hold_position.x, hold_position.y, hold_position.z),
        ):
            assert np.allclose(data[field][settle_idx], expected)
        assert np.allclose(data["e"][settle_idx], 2.0)


def test_external_cut_uses_timed_resets_to_isolate_lift_and_retract(tmp_path):
    import numpy as np

    from path_processing_core.npz_exporter import export_npz
    from path_processing_core.types import (
        MCommand,
        MoveCommand,
        Position,
        ResetECommand,
    )

    commands = [
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
        ResetECommand(
            type="RESET_E",
            val=0.0,
            line=3,
            layer=0,
            subtype="FIBER",
            raw="external_npz_path_reset",
        ),
    ]
    output = tmp_path / "timed_cut_resets.npz"

    export_npz(
        commands,
        str(output),
        dt=1.0,
        default_feed_mm_s=10.0,
        enable_extrude_wait=True,
        cut_lift_mm=12.5,
        cut_wait_s=15.0,
        initial_tool_id=1,
        external_npz_cut_absolute_e=True,
    )

    with np.load(output) as data:
        events = _decoded_event_types(data)
        cut_idx = events.index("cut")
        reset_idx = [idx for idx, event in enumerate(events) if event == "extrude_reset"]
        assert len(reset_idx) == 4

        pre_cut_reset, high_reset, final_cut_reset, path_reset = reset_idx
        assert pre_cut_reset < cut_idx < high_reset < final_cut_reset < path_reset
        assert np.isclose(data["e"][pre_cut_reset], 5.0)
        assert np.isclose(data["e"][cut_idx - 1], 0.0)
        assert np.isclose(data["e"][cut_idx], 0.0)

        lift_and_settle = list(range(cut_idx + 1, high_reset))
        assert np.isclose(np.max(data["z"][lift_and_settle]), 12.5)
        assert np.isclose(np.max(data["e"][lift_and_settle]), 12.5)
        assert np.isclose(data["e"][lift_and_settle[-1]], 12.5)
        assert sum(np.isclose(data["e"][idx], 12.5) for idx in lift_and_settle) >= 3

        high_anchor = high_reset + 1
        assert np.isclose(data["e"][high_reset], 12.5)
        assert np.isclose(data["e"][high_anchor], 0.0)
        retract_and_settle = list(range(high_anchor + 1, final_cut_reset))
        assert np.allclose(data["z"][retract_and_settle], 12.5)
        assert np.isclose(np.min(data["e"][retract_and_settle]), -12.5)
        assert np.isclose(data["e"][retract_and_settle[-1]], -12.5)
        assert sum(np.isclose(data["e"][idx], -12.5) for idx in retract_and_settle) >= 3

        final_anchor = final_cut_reset + 1
        assert np.isclose(data["e"][final_cut_reset], -12.5)
        assert np.isclose(data["e"][final_anchor], 0.0)
        trailing_wait = list(range(final_anchor + 1, path_reset))
        assert trailing_wait
        assert np.allclose(data["z"][trailing_wait], 12.5)
        assert np.allclose(data["e"][trailing_wait], 0.0)


def test_fiber_cut_and_ui_actions_use_independent_absolute_e_boundaries(tmp_path):
    import json
    import numpy as np

    from external_npz_preprocessor.export_runner import convert_external_npz
    from external_npz_preprocessor.process_params import FiberProcessParams, ProcessParams

    source = tmp_path / "two_fiber_paths.npz"
    fiber_paths = np.array(
        [
            [[0.0, 0.0, 0.6], [10.0, 0.0, 0.6]],
            [[30.0, 0.0, 0.6], [40.0, 0.0, 0.6]],
        ],
        dtype=np.float32,
    )
    np.savez(
        source,
        meta=np.array(json.dumps({"format": "external_layer_paths_v1"})),
        layer_0000_F=fiber_paths,
    )
    calibration_path = tmp_path / "head_offsets.json"
    calibration_path.write_text(
        json.dumps(
            {
                "resin": {"z_print_compensation_mm": 0.0},
                "fiber": {
                    "x_print_compensation_mm": 0.0,
                    "y_print_compensation_mm": 0.0,
                    "z_offset_mm": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.npz"
    params = ProcessParams(
        dt=1.0,
        fiber=FiberProcessParams(
            extrusion_scale=1.0,
            feed_mm_s=10.0,
            retract_length_mm=4.0,
            prime_length_mm=6.0,
            retract_speed_mm_s=4.0,
            prime_speed_mm_s=6.0,
        ),
        travel_feed_mm_s=10.0,
    )

    convert_external_npz(
        source,
        out,
        params,
        calibration_path=calibration_path,
        cut_lift_mm=20.0,
        cut_wait_s=0.0,
    )

    data = np.load(out)
    src_lines = _decoded_src_lines(data)
    event_types = _decoded_event_types(data)
    cut_idx = event_types.index("cut")
    cut_e = float(data["e"][cut_idx])
    cut_z = float(data["z"][cut_idx])

    pre_cut_reset_idx = cut_idx - 2
    pre_cut_anchor_idx = cut_idx - 1
    assert event_types[pre_cut_reset_idx] == "extrude_reset"
    assert data["e"][pre_cut_reset_idx] > 0.0
    assert event_types[pre_cut_anchor_idx] == ""
    assert np.isclose(data["e"][pre_cut_anchor_idx], 0.0)
    assert np.isclose(cut_e, 0.0)
    for field in ("x", "y", "z", "a", "b", "c"):
        assert np.isclose(data[field][pre_cut_anchor_idx], data[field][cut_idx])

    high_reset_idx = next(
        idx
        for idx in range(cut_idx + 1, len(event_types))
        if event_types[idx] == "extrude_reset"
    )
    lift_idx = list(range(cut_idx + 1, high_reset_idx))
    assert lift_idx
    assert np.isclose(np.max(data["z"][lift_idx]), cut_z + 20.0)
    assert np.isclose(np.max(data["e"][lift_idx]), 20.0)
    assert np.isclose(data["z"][lift_idx[-1]], cut_z + 20.0)
    assert np.isclose(data["e"][lift_idx[-1]], 20.0)
    assert np.isclose(data["e"][high_reset_idx], 20.0)
    assert np.isclose(data["z"][high_reset_idx], cut_z + 20.0)

    high_anchor_idx = high_reset_idx + 1
    assert np.isclose(data["e"][high_anchor_idx], 0.0)
    for field in ("x", "y", "z", "a", "b", "c"):
        assert np.isclose(data[field][high_anchor_idx], data[field][high_reset_idx])

    final_cut_reset_idx = next(
        idx
        for idx in range(high_anchor_idx + 1, len(event_types))
        if event_types[idx] == "extrude_reset"
    )
    retract_idx = list(range(high_anchor_idx + 1, final_cut_reset_idx))
    assert retract_idx
    assert np.allclose(data["z"][retract_idx], cut_z + 20.0)
    assert np.isclose(np.min(data["e"][retract_idx]), -20.0)
    assert np.isclose(data["e"][retract_idx[-1]], -20.0)
    assert np.isclose(data["e"][final_cut_reset_idx], -20.0)

    final_cut_anchor_idx = final_cut_reset_idx + 1
    assert np.isclose(data["e"][final_cut_anchor_idx], 0.0)
    path_reset_idx = next(
        idx
        for idx in range(final_cut_anchor_idx + 1, len(event_types))
        if event_types[idx] == "extrude_reset"
    )
    assert np.isclose(data["e"][path_reset_idx], 0.0)
    path_anchor_idx = path_reset_idx + 1
    assert np.isclose(data["e"][path_anchor_idx], 0.0)

    travel_idx = _next_src_line_group(src_lines, path_anchor_idx)
    assert travel_idx
    assert np.allclose(data["e"][travel_idx], 0.0)
    assert np.isclose(data["z"][travel_idx[0]], cut_z + 20.0)
    assert np.isclose(data["z"][travel_idx[-1]], cut_z)
    assert np.isclose(data["x"][travel_idx[0]], 10.0)
    assert np.isclose(data["x"][travel_idx[-1]], 30.0)
    assert np.allclose(data["y"][travel_idx], 0.0)

    second_cut_idx = event_types.index("cut", cut_idx + 1)
    second_pre_cut_reset_idx = second_cut_idx - 2
    second_pre_cut_anchor_idx = second_cut_idx - 1
    second_print_idx = list(range(travel_idx[-1] + 1, second_pre_cut_reset_idx))
    assert second_print_idx
    assert not any(event_types[idx] for idx in second_print_idx)
    assert np.min(data["e"][second_print_idx]) >= 0.0
    assert np.max(data["e"][second_print_idx]) > 0.0
    assert event_types[second_pre_cut_reset_idx] == "extrude_reset"
    assert data["e"][second_pre_cut_reset_idx] > 0.0
    assert np.isclose(data["e"][second_pre_cut_anchor_idx], 0.0)

    second_cut_z = float(data["z"][second_cut_idx])
    second_high_reset_idx = next(
        idx
        for idx in range(second_cut_idx + 1, len(event_types))
        if event_types[idx] == "extrude_reset"
    )
    second_lift_idx = list(range(second_cut_idx + 1, second_high_reset_idx))
    assert second_lift_idx
    assert np.isclose(
        np.max(data["z"][second_lift_idx]),
        second_cut_z + 20.0,
    )
    assert np.isclose(np.max(data["e"][second_lift_idx]), 20.0)
    assert np.isclose(data["e"][second_high_reset_idx], 20.0)

    second_high_anchor_idx = second_high_reset_idx + 1
    assert np.isclose(data["e"][second_high_anchor_idx], 0.0)
    second_final_cut_reset_idx = next(
        idx
        for idx in range(second_high_anchor_idx + 1, len(event_types))
        if event_types[idx] == "extrude_reset"
    )
    second_retract_idx = list(
        range(second_high_anchor_idx + 1, second_final_cut_reset_idx)
    )
    assert second_retract_idx
    assert np.allclose(data["z"][second_retract_idx], second_cut_z + 20.0)
    assert np.isclose(np.min(data["e"][second_retract_idx]), -20.0)
    assert np.isclose(data["e"][second_final_cut_reset_idx], -20.0)

    second_final_cut_anchor_idx = second_final_cut_reset_idx + 1
    assert np.isclose(data["e"][second_final_cut_anchor_idx], 0.0)
    layer_retract_reset_idx = next(
        idx
        for idx in range(second_final_cut_anchor_idx + 1, len(event_types))
        if event_types[idx] == "extrude_reset"
    )
    assert np.isclose(data["e"][layer_retract_reset_idx], 0.0)

    layer_retract_anchor_idx = _next_src_line_group(
        src_lines, layer_retract_reset_idx
    )
    assert len(layer_retract_anchor_idx) == 1
    assert np.isclose(data["e"][layer_retract_anchor_idx[0]], 0.0)
    ui_retract_idx = _next_src_line_group(
        src_lines, layer_retract_anchor_idx[-1]
    )
    assert np.isclose(data["e"][ui_retract_idx[-1]], -4.0)

    path_reset_idx = next(
        idx
        for idx in range(layer_retract_reset_idx + 1, len(event_types))
        if event_types[idx] == "extrude_reset"
    )
    assert np.isclose(data["e"][path_reset_idx], -4.0)
    path_anchor_idx = _next_src_line_group(src_lines, path_reset_idx)
    assert len(path_anchor_idx) == 1
    assert np.isclose(data["e"][path_anchor_idx[0]], 0.0)


def test_convert_writes_startup_events_and_tool_reset_order_to_npz(tmp_path):
    import json
    import numpy as np

    from external_npz_preprocessor.process_params import ProcessParams

    source = tmp_path / "source.npz"
    resin_paths = np.array([[[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]]], dtype=np.float32)
    fiber_paths = np.array([[[2.0, 0.0, 0.6], [4.0, 0.0, 0.6]]], dtype=np.float32)
    np.savez(
        source,
        meta=np.array(json.dumps({"format": "external_layer_paths_v1"})),
        layer_0000_R=resin_paths,
        layer_0000_F=fiber_paths,
    )
    calibration_path = tmp_path / "head_offsets.json"
    calibration_path.write_text(
        json.dumps(
            {
                "resin": {"z_print_compensation_mm": 0.0},
                "fiber": {
                    "x_print_compensation_mm": 0.0,
                    "y_print_compensation_mm": 0.0,
                    "z_offset_mm": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.npz"

    from external_npz_preprocessor.export_runner import convert_external_npz

    convert_external_npz(source, out, ProcessParams(), calibration_path=calibration_path)

    with np.load(out) as data:
        events = _decoded_event_types(data)
        non_empty_events = [event for event in events if event]

        assert non_empty_events == [
            "fan_resin",
            "fan_cf",
            "heat_resin",
            "heat_cf",
            "extrude_reset",
            "extrude_reset",
            "extrude_reset",
            "tool_change_cf",
            "extrude_reset",
            "extrude_reset",
            "extrude_reset",
            "extrude_reset",
            "extrude_reset",
            "cut",
            "extrude_reset",
            "extrude_reset",
            "extrude_reset",
            "extrude_reset",
        ]

        cut_row_idx = events.index("cut")
        assert data["payload"][cut_row_idx].decode("utf-8").rstrip("\x00") == "1"


def test_convert_skips_fiber_startup_events_when_source_has_no_fiber_paths(tmp_path):
    import json
    import numpy as np

    from external_npz_preprocessor.export_runner import convert_external_npz
    from external_npz_preprocessor.process_params import ProcessParams

    source = tmp_path / "resin_only_source.npz"
    resin_paths = np.array(
        [[[0.0, 0.0, 0.5], [10.0, 0.0, 0.5]]],
        dtype=np.float32,
    )
    np.savez(
        source,
        meta=np.array(json.dumps({"format": "external_layer_paths_v1"})),
        layer_0000_R=resin_paths,
    )
    calibration_path = tmp_path / "head_offsets.json"
    calibration_path.write_text(
        json.dumps(
            {
                "resin": {"z_print_compensation_mm": 0.0},
                "fiber": {
                    "x_print_compensation_mm": 0.0,
                    "y_print_compensation_mm": 0.0,
                    "z_offset_mm": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.npz"

    convert_external_npz(source, out, ProcessParams(), calibration_path=calibration_path)

    data = np.load(out)
    event_vocab = {
        int(value): key.decode("utf-8").rstrip("\x00")
        for key, value in zip(data["event_type_vocab_keys"], data["event_type_vocab_vals"])
    }
    events = [event_vocab[int(value)] for value in data["event_type"]]
    non_empty_events = [event for event in events if event]

    assert "fan_resin" in non_empty_events
    assert "heat_resin" in non_empty_events
    assert "fan_cf" not in non_empty_events
    assert "heat_cf" not in non_empty_events
