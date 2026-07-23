import math
import numpy as np

from gcode_planner.gcode_parser import parse_gcode_lines
from path_processing_core.head_calibration import HeadCalibration
from path_processing_core.npz_exporter import export_npz
from gcode_planner.print_test_generator import (
    EXTRUSION_PER_MM3,
    FIBER_GCODE_TOOL,
    FIBER_TOOL_ID,
    RESIN_GCODE_TOOL,
    RESIN_TOOL_ID,
    expand_test_values,
    generate_composite_test_matrix_gcode,
    generate_head_test_matrix_gcode,
    generate_pose_adjust_gcode,
    generate_test_matrix_gcode,
    generate_test_line_gcode,
    generate_z_adjust_gcode,
)
from path_processing_core.types import ExtrudeWait, MoveCommand, ResetECommand, ToolChangeCommand


def _moves(lines):
    return [cmd for cmd in parse_gcode_lines(lines) if isinstance(cmd, MoveCommand)]


def test_default_resin_extrusion_constant_matches_1p75mm_filament():
    expected = 1.0 / (math.pi * (1.75 / 2.0) ** 2)
    assert math.isclose(EXTRUSION_PER_MM3, expected, rel_tol=1e-6)


def test_pose_adjust_gcode_moves_xyzabc_without_extrusion():
    lines = generate_pose_adjust_gcode(
        start_pose=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        target_pose=(7.0, 8.0, 9.0, 4.0, 5.0, 6.0),
        speed_mm_s=5.0,
    )

    moves = _moves(lines)

    assert len(moves) == 1
    assert moves[0].start_pos.x == 1.0
    assert moves[0].pos.x == 7.0
    assert moves[0].pos.y == 8.0
    assert moves[0].pos.z == 9.0
    assert moves[0].delta_e == 0.0


def test_z_adjust_gcode_starts_from_current_rsi_correction_and_uses_resin_tool():
    lines = generate_z_adjust_gcode(
        start_pose=(0.0, 0.0, 1.5, 0.0, 0.0, 0.0),
        target_z=1.6,
        speed_mm_s=5.0,
    )
    parsed = parse_gcode_lines(lines)
    moves = [cmd for cmd in parsed if isinstance(cmd, MoveCommand)]
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]
    resets = [cmd for cmd in parsed if isinstance(cmd, ResetECommand)]

    assert tools == []
    assert resets == []
    assert moves[0].start_pos.z == 1.5
    assert moves[0].pos.z == 1.6
    assert moves[0].feedrate == 300.0
    assert moves[0].delta_e == 0.0


def test_test_line_gcode_uses_fixed_resin_line_width_and_finish_lift():
    lines = generate_test_line_gcode(
        start_pose=(1.0, 2.0, 0.4, 0.0, 0.0, 0.0),
        layer_height_mm=0.5,
        speed_mm_s=10.0,
        line_length_mm=200.0,
        finish_lift_mm=10.0,
    )
    parsed = parse_gcode_lines(lines)
    moves = [cmd for cmd in parsed if isinstance(cmd, MoveCommand)]
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]

    assert RESIN_TOOL_ID == 2
    assert tools == []
    print_move = next(cmd for cmd in moves if cmd.type == "PRINT")
    assert print_move.start_pos.x == 1.0
    assert print_move.pos.x == 201.0
    assert math.isclose(print_move.delta_e, 200.0 * 2.0 * 0.5 * EXTRUSION_PER_MM3, abs_tol=1e-6)
    lift_move = moves[-1]
    assert lift_move.type == "TRAVEL"
    assert lift_move.pos.z == 10.4
    assert lift_move.delta_e == 0.0


def test_test_line_gcode_adds_prime_and_retract_as_filament_lengths():
    lines = generate_test_line_gcode(
        start_pose=(1.0, 2.0, 0.4, 0.0, 0.0, 0.0),
        layer_height_mm=0.5,
        speed_mm_s=10.0,
        line_length_mm=200.0,
        finish_lift_mm=10.0,
        prime_length_mm=5.0,
        retract_length_mm=3.0,
        prime_speed_mm_s=2.0,
        retract_speed_mm_s=8.0,
    )

    parsed = parse_gcode_lines(lines)
    waits = [cmd for cmd in parsed if isinstance(cmd, ExtrudeWait)]
    print_move = next(
        cmd for cmd in parsed if isinstance(cmd, MoveCommand) and cmd.type == "PRINT"
    )

    assert len(waits) == 2
    assert math.isclose(waits[0].delta_e, 5.0)
    assert math.isclose(waits[0].feedrate, 2.0 * 60.0)
    assert math.isclose(print_move.delta_e, 200.0 * 2.0 * 0.5 * EXTRUSION_PER_MM3, abs_tol=1e-6)
    assert math.isclose(waits[1].delta_e, -3.0)
    assert math.isclose(waits[1].feedrate, 8.0 * 60.0)


def test_expand_test_values_accepts_single_value_and_inclusive_range():
    assert expand_test_values("0.5") == [0.5]
    assert expand_test_values("0.5-0.7") == [0.5, 0.6, 0.7]


def test_expand_test_values_preserves_small_positive_values():
    assert expand_test_values("0.05") == [0.05]
    assert expand_test_values("0.05-0.05") == [0.05]


def test_expand_test_values_rejects_descending_range():
    try:
        expand_test_values("1.0-0.8")
    except ValueError as exc:
        assert "不能小于" in str(exc)
    else:
        raise AssertionError("expected descending range to be rejected")


def test_test_matrix_gcode_generates_full_combinations_with_y_spacing_and_scaled_e():
    lines = generate_test_matrix_gcode(
        start_pose=(1.0, 2.0, 0.4, 0.0, 0.0, 0.0),
        layer_heights_mm=[0.5, 0.6],
        extrusion_scales=[0.8, 1.0],
        speed_mm_s=10.0,
        line_length_mm=300.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
    )

    moves = _moves(lines)
    print_moves = [cmd for cmd in moves if cmd.type == "PRINT"]
    travel_moves = [cmd for cmd in moves if cmd.type == "TRAVEL"]

    assert len(print_moves) == 4
    assert [(cmd.start_pos.x, cmd.start_pos.y) for cmd in print_moves] == [
        (1.0, 2.0),
        (301.0, 12.0),
        (1.0, 22.0),
        (301.0, 32.0),
    ]
    assert [cmd.pos.x for cmd in print_moves] == [301.0, 1.0, 301.0, 1.0]
    assert [cmd.pos.z for cmd in print_moves] == [0.9, 0.9, 1.0, 1.0]

    expected_deltas = [
        300.0 * 2.0 * 0.5 * EXTRUSION_PER_MM3 * 0.8,
        300.0 * 2.0 * 0.5 * EXTRUSION_PER_MM3 * 1.0,
        300.0 * 2.0 * 0.6 * EXTRUSION_PER_MM3 * 0.8,
        300.0 * 2.0 * 0.6 * EXTRUSION_PER_MM3 * 1.0,
    ]
    for move, expected_delta in zip(print_moves, expected_deltas):
        assert math.isclose(move.delta_e, expected_delta, abs_tol=1e-6)

    assert any(
        cmd.start_pos.x == 301.0
        and cmd.start_pos.y == 2.0
        and cmd.start_pos.z == 10.9
        and cmd.pos.x == 301.0
        and cmd.pos.y == 12.0
        and cmd.pos.z == 10.9
        and cmd.delta_e == 0.0
        for cmd in travel_moves
    )
    assert any(
        cmd.start_pos.x == 301.0
        and cmd.start_pos.y == 12.0
        and cmd.start_pos.z == 10.9
        and cmd.pos.x == 301.0
        and cmd.pos.y == 12.0
        and cmd.pos.z == 0.9
        and cmd.delta_e == 0.0
        for cmd in travel_moves
    )
    assert moves[-1].type == "TRAVEL"
    assert moves[-1].pos.x == 1.0
    assert moves[-1].pos.z == 11.0


def test_fiber_matrix_uses_fiber_tool_and_serpentine_geometry():
    lines = generate_head_test_matrix_gcode(
        start_pose=(1.0, 2.0, 0.4, 0.0, 0.0, 0.0),
        tool="fiber",
        layer_heights_mm=[0.5],
        extrusion_scales=[0.8, 1.0],
        speed_mm_s=10.0,
        line_length_mm=300.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        prime_length_mm=5.0,
        retract_length_mm=3.0,
        prime_speed_mm_s=2.0,
        retract_speed_mm_s=8.0,
    )

    parsed = parse_gcode_lines(lines)
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]
    waits = [cmd for cmd in parsed if isinstance(cmd, ExtrudeWait)]
    print_moves = [cmd for cmd in _moves(lines) if cmd.type == "PRINT"]

    assert FIBER_TOOL_ID == 1
    assert FIBER_GCODE_TOOL == 0
    assert tools[0].tool == FIBER_GCODE_TOOL
    assert "T0" in lines
    assert [(cmd.start_pos.x, cmd.start_pos.y) for cmd in print_moves] == [
        (1.0, 2.0),
        (301.0, 12.0),
    ]
    assert [cmd.pos.x for cmd in print_moves] == [301.0, 1.0]
    assert [cmd.delta_e for cmd in print_moves] == [240.0, 300.0]

    waits_before_first_print = [cmd for cmd in waits if cmd.line < print_moves[0].line]
    waits_before_second_print = [
        cmd for cmd in waits if print_moves[0].line < cmd.line < print_moves[1].line
    ]
    assert [cmd.delta_e for cmd in waits_before_first_print] == [-3.0, 5.0]
    assert [cmd.delta_e for cmd in waits_before_second_print] == [-3.0, 5.0]


def test_composite_fiber_segment_extrudes_by_path_length_only():
    lines = generate_composite_test_matrix_gcode(
        start_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        resin_layer_heights_mm=[0.5],
        resin_extrusion_scales=[1.0],
        fiber_layer_heights_mm=[0.05],
        fiber_extrusion_scales=[1.0],
        speed_mm_s=10.0,
        line_length_mm=300.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        calibration=HeadCalibration(),
        tool_change_safe_lift_mm=10.0,
    )

    parsed = parse_gcode_lines(lines)
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]
    moves = [cmd for cmd in parsed if isinstance(cmd, MoveCommand)]
    fiber_tool_line = next(cmd.line for cmd in tools if cmd.tool == FIBER_GCODE_TOOL)
    first_fiber_print = next(
        cmd for cmd in moves if cmd.type == "PRINT" and cmd.line > fiber_tool_line
    )

    assert math.isclose(first_fiber_print.delta_e, 300.0)


def test_resin_matrix_shifts_y_between_lines_and_lifts_only_after_matrix():
    lines = generate_head_test_matrix_gcode(
        start_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tool="resin",
        layer_heights_mm=[0.5],
        extrusion_scales=[1.0, 1.0],
        speed_mm_s=10.0,
        line_length_mm=100.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        prime_length_mm=18.0,
        retract_length_mm=15.0,
        prime_speed_mm_s=15.0,
        retract_speed_mm_s=30.0,
    )

    moves = _moves(lines)
    resin_prints = [cmd for cmd in moves if cmd.type == "PRINT"]
    resin_travels = [cmd for cmd in moves if cmd.type == "TRAVEL"]

    assert len(resin_prints) == 2
    between_resin_lines = [
        cmd
        for cmd in resin_travels
        if resin_prints[0].line < cmd.line < resin_prints[1].line
    ]
    assert any(
        cmd.start_pos.z == cmd.pos.z == 0.5
        and cmd.start_pos.x == cmd.pos.x == 100.0
        and cmd.start_pos.y == 0.0
        and cmd.pos.y == 10.0
        for cmd in between_resin_lines
    )
    assert not any(
        cmd.start_pos.z > 0.5 or cmd.pos.z > 0.5
        for cmd in between_resin_lines
    )

    final_y_shift = next(
        cmd
        for cmd in resin_travels
        if cmd.line > resin_prints[-1].line
        and cmd.start_pos.z == cmd.pos.z == 0.5
        and cmd.start_pos.y == 10.0
        and cmd.pos.y == 20.0
    )
    final_lift = next(
        cmd for cmd in resin_travels if cmd.line > final_y_shift.line and cmd.pos.z == 10.5
    )
    assert final_lift.start_pos.x == final_y_shift.pos.x
    assert final_lift.start_pos.y == final_y_shift.pos.y


def test_composite_matrix_inserts_safe_lift_compensation_and_tool_change():
    lines = generate_composite_test_matrix_gcode(
        start_pose=(5.0, 4.0, -45.0, 0.0, 0.0, 0.0),
        resin_layer_heights_mm=[0.5],
        resin_extrusion_scales=[1.0],
        fiber_layer_heights_mm=[0.6],
        fiber_extrusion_scales=[1.1],
        speed_mm_s=10.0,
        line_length_mm=300.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        prime_length_mm=0.0,
        retract_length_mm=0.0,
        prime_speed_mm_s=2.0,
        retract_speed_mm_s=8.0,
        fiber_prime_length_mm=0.0,
        fiber_retract_length_mm=0.0,
        fiber_prime_speed_mm_s=2.0,
        fiber_retract_speed_mm_s=8.0,
        calibration=HeadCalibration(
            resin_z_print_compensation_mm=-20.0,
            fiber_x_print_compensation_mm=5.0,
            fiber_y_print_compensation_mm=4.0,
            fiber_z_print_compensation_mm=-25.0,
        ),
        tool_change_safe_lift_mm=10.0,
    )

    text = "\n".join(lines)
    parsed = parse_gcode_lines(lines)
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]
    moves = [cmd for cmd in parsed if isinstance(cmd, MoveCommand)]

    assert ";TOOL_CHANGE_SAFE_LIFT:10.000000" in text
    assert ";TOOL_CHANGE_COMPENSATION:-5.000000,-4.000000,25.000000" in text
    assert ";TOOL_CHANGE_COMPENSATION:5.000000,4.000000,-25.000000" in text
    assert "T0" in lines
    assert [cmd.tool for cmd in tools] == [RESIN_GCODE_TOOL, FIBER_GCODE_TOOL]
    fiber_tool_line = next(cmd.line for cmd in tools if cmd.tool == FIBER_GCODE_TOOL)
    resin_to_fiber_compensation_line = lines.index(
        ";TOOL_CHANGE_COMPENSATION:5.000000,4.000000,-25.000000"
    )
    assert lines.index("T0") < resin_to_fiber_compensation_line

    moves_after_fiber_tool = [cmd for cmd in moves if cmd.line > fiber_tool_line]
    resin_to_fiber_reposition = next(
        cmd
        for cmd in moves_after_fiber_tool
        if cmd.type == "TRAVEL"
        and cmd.pos.x == 5.0
        and cmd.pos.y == 4.0
        and cmd.pos.z == -34.5
    )
    assert resin_to_fiber_reposition.start_pos.x == 0.0
    assert resin_to_fiber_reposition.start_pos.y == 0.0
    assert resin_to_fiber_reposition.start_pos.z == -9.5

    first_fiber_print = next(cmd for cmd in moves_after_fiber_tool if cmd.type == "PRINT")
    descent_to_print_z = next(
        cmd
        for cmd in moves_after_fiber_tool
        if cmd.type == "TRAVEL"
        and cmd.line < first_fiber_print.line
        and cmd.start_pos.x == cmd.pos.x == first_fiber_print.start_pos.x
        and cmd.start_pos.y == cmd.pos.y == first_fiber_print.start_pos.y
        and cmd.start_pos.z > cmd.pos.z
    )
    assert descent_to_print_z.pos.z == first_fiber_print.start_pos.z


def test_composite_from_confirmed_fiber_pose_applies_resin_z_once():
    calibration = HeadCalibration(
        resin_z_print_compensation_mm=-20.0,
        fiber_x_print_compensation_mm=3.0,
        fiber_y_print_compensation_mm=2.0,
        fiber_z_print_compensation_mm=4.0,
    )
    start_pose = (3.0, 2.0, -16.0, 0.0, 0.0, 0.0)

    lines = generate_composite_test_matrix_gcode(
        start_pose=start_pose,
        resin_layer_heights_mm=[0.5],
        resin_extrusion_scales=[1.0],
        fiber_layer_heights_mm=[0.05],
        fiber_extrusion_scales=[1.0],
        speed_mm_s=10.0,
        line_length_mm=100.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        calibration=calibration,
        tool_change_safe_lift_mm=10.0,
    )

    text = "\n".join(lines)
    parsed = parse_gcode_lines(lines)
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]
    moves = [cmd for cmd in parsed if isinstance(cmd, MoveCommand)]
    fiber_tool_line = next(cmd.line for cmd in tools if cmd.tool == FIBER_GCODE_TOOL)
    first_resin_print = next(
        cmd for cmd in moves if cmd.type == "PRINT" and cmd.line < fiber_tool_line
    )
    first_fiber_print = next(
        cmd for cmd in moves if cmd.type == "PRINT" and cmd.line > fiber_tool_line
    )

    assert ";TOOL_CHANGE_COMPENSATION:-3.000000,-2.000000,-4.000000" in text
    assert ";TOOL_CHANGE_COMPENSATION:3.000000,2.000000,4.000000" in text
    assert math.isclose(first_resin_print.start_pos.z, -19.5)
    assert math.isclose(first_fiber_print.start_pos.z, -15.45)


def test_composite_resin_lines_shift_y_without_intermediate_z_lift():
    lines = generate_composite_test_matrix_gcode(
        start_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        resin_layer_heights_mm=[0.5],
        resin_extrusion_scales=[1.0, 1.0],
        fiber_layer_heights_mm=[0.05],
        fiber_extrusion_scales=[1.0],
        speed_mm_s=10.0,
        line_length_mm=100.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        prime_length_mm=18.0,
        retract_length_mm=15.0,
        prime_speed_mm_s=15.0,
        retract_speed_mm_s=30.0,
        fiber_prime_length_mm=12.0,
        fiber_retract_length_mm=10.0,
        fiber_prime_speed_mm_s=5.0,
        fiber_retract_speed_mm_s=5.0,
        calibration=HeadCalibration(),
        tool_change_safe_lift_mm=10.0,
    )

    parsed = parse_gcode_lines(lines)
    moves = [cmd for cmd in parsed if isinstance(cmd, MoveCommand)]
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]
    fiber_tool_line = next(cmd.line for cmd in tools if cmd.tool == FIBER_GCODE_TOOL)
    resin_prints = [cmd for cmd in moves if cmd.type == "PRINT" and cmd.line < fiber_tool_line]
    resin_travels = [cmd for cmd in moves if cmd.type == "TRAVEL" and cmd.line < fiber_tool_line]

    assert len(resin_prints) == 2
    between_resin_lines = [
        cmd for cmd in resin_travels
        if resin_prints[0].line < cmd.line < resin_prints[1].line
    ]
    assert any(
        cmd.start_pos.z == cmd.pos.z == 0.5
        and cmd.start_pos.x == cmd.pos.x == 100.0
        and cmd.start_pos.y == 0.0
        and cmd.pos.y == 10.0
        for cmd in between_resin_lines
    )
    assert not any(
        cmd.start_pos.z > 0.5 or cmd.pos.z > 0.5
        for cmd in between_resin_lines
    )

    after_last_resin = [
        cmd for cmd in resin_travels if cmd.line > resin_prints[-1].line
    ]
    final_y_shift = next(
        cmd for cmd in after_last_resin
        if cmd.start_pos.z == cmd.pos.z == 0.5
        and cmd.start_pos.y == 10.0
        and cmd.pos.y == 20.0
    )
    final_lift = next(
        cmd for cmd in after_last_resin
        if cmd.line > final_y_shift.line and cmd.pos.z == 10.5
    )
    assert final_lift.start_pos.x == final_y_shift.pos.x
    assert final_lift.start_pos.y == final_y_shift.pos.y


def test_composite_tool_change_to_fiber_stays_safe_and_uses_fiber_layer_height():
    lines = generate_composite_test_matrix_gcode(
        start_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        resin_layer_heights_mm=[0.5],
        resin_extrusion_scales=[1.0, 1.0],
        fiber_layer_heights_mm=[0.05],
        fiber_extrusion_scales=[1.0],
        speed_mm_s=10.0,
        line_length_mm=100.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        calibration=HeadCalibration(
            resin_z_print_compensation_mm=-20.0,
            fiber_x_print_compensation_mm=3.0,
            fiber_y_print_compensation_mm=2.0,
            fiber_z_print_compensation_mm=4.0,
        ),
        tool_change_safe_lift_mm=10.0,
    )

    parsed = parse_gcode_lines(lines)
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]
    moves = [cmd for cmd in parsed if isinstance(cmd, MoveCommand)]
    fiber_tool_line = next(cmd.line for cmd in tools if cmd.tool == FIBER_GCODE_TOOL)
    first_fiber_print = next(
        cmd for cmd in moves if cmd.type == "PRINT" and cmd.line > fiber_tool_line
    )
    assert math.isclose(first_fiber_print.start_pos.z, 0.55)

    compensation_move = next(
        cmd
        for cmd in moves
        if cmd.line > fiber_tool_line
        and cmd.type == "TRAVEL"
        and math.isclose(cmd.pos.x, 0.0)
        and math.isclose(cmd.pos.y, 0.0)
        and math.isclose(cmd.pos.z, 10.5)
    )
    assert math.isclose(compensation_move.pos.z, 10.5)

    descent_to_print_z = next(
        cmd
        for cmd in moves
        if cmd.line > compensation_move.line
        and cmd.line < first_fiber_print.line
        and cmd.type == "TRAVEL"
        and math.isclose(cmd.start_pos.z, compensation_move.pos.z)
        and math.isclose(cmd.pos.z, first_fiber_print.start_pos.z)
    )
    assert descent_to_print_z.pos.x == first_fiber_print.start_pos.x
    assert descent_to_print_z.pos.y == first_fiber_print.start_pos.y


def test_composite_fiber_starts_at_first_resin_line_with_resin_height_added():
    lines = generate_composite_test_matrix_gcode(
        start_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        resin_layer_heights_mm=[0.5],
        resin_extrusion_scales=[1.0],
        fiber_layer_heights_mm=[0.05],
        fiber_extrusion_scales=[1.0],
        speed_mm_s=10.0,
        line_length_mm=100.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        prime_length_mm=18.0,
        retract_length_mm=15.0,
        prime_speed_mm_s=15.0,
        retract_speed_mm_s=30.0,
        fiber_prime_length_mm=12.0,
        fiber_retract_length_mm=10.0,
        fiber_prime_speed_mm_s=5.0,
        fiber_retract_speed_mm_s=5.0,
        calibration=HeadCalibration(),
        tool_change_safe_lift_mm=10.0,
    )

    parsed = parse_gcode_lines(lines)
    tools = [cmd for cmd in parsed if isinstance(cmd, ToolChangeCommand)]
    moves = [cmd for cmd in parsed if isinstance(cmd, MoveCommand)]
    fiber_tool_line = next(cmd.line for cmd in tools if cmd.tool == FIBER_GCODE_TOOL)
    fiber_print = next(
        cmd for cmd in moves if cmd.type == "PRINT" and cmd.line > fiber_tool_line
    )

    assert fiber_print.start_pos.x == 0.0
    assert fiber_print.start_pos.y == 0.0
    assert math.isclose(fiber_print.start_pos.z, 0.55)
    assert math.isclose(fiber_print.pos.z, 0.55)


def test_composite_npz_switches_from_initial_fiber_tool_after_safe_lift(tmp_path):
    calibration = HeadCalibration(
        resin_z_print_compensation_mm=-20.0,
        fiber_x_print_compensation_mm=3.0,
        fiber_y_print_compensation_mm=2.0,
        fiber_z_print_compensation_mm=4.0,
    )
    lines = generate_composite_test_matrix_gcode(
        start_pose=(3.0, 2.0, -16.0, 0.0, 0.0, 0.0),
        resin_layer_heights_mm=[0.5],
        resin_extrusion_scales=[1.0],
        fiber_layer_heights_mm=[0.05],
        fiber_extrusion_scales=[1.0],
        speed_mm_s=10.0,
        line_length_mm=10.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        calibration=calibration,
        tool_change_safe_lift_mm=10.0,
    )
    out = tmp_path / "composite_initial_fiber.npz"

    export_npz(
        parse_gcode_lines(lines),
        str(out),
        dt=0.004,
        default_feed_mm_s=10.0,
        enable_extrude_wait=True,
        initial_tool_id=FIBER_TOOL_ID,
    )

    data = np.load(out)
    event_type_vocab = {
        int(value): key.decode("utf-8").rstrip("\x00")
        for key, value in zip(
            data["event_type_vocab_keys"],
            data["event_type_vocab_vals"],
        )
    }
    event_types = [event_type_vocab[int(v)] for v in data["event_type"]]
    resin_switch_idx = event_types.index("tool_change_resin")
    first_resin_print_idx = next(
        idx for idx, value in enumerate(data["move_type"])
        if value == 1 and idx > resin_switch_idx
    )

    assert np.isclose(data["z"][resin_switch_idx], -6.0)
    assert data["z"][first_resin_print_idx] < data["z"][resin_switch_idx]


def test_composite_npz_starts_with_resin_tool_change_before_reset(tmp_path):
    start_pose = (-0.34, -1.24, -24.45, 0.0, 0.0, 0.0)
    lines = generate_composite_test_matrix_gcode(
        start_pose=start_pose,
        resin_layer_heights_mm=[0.5],
        resin_extrusion_scales=[1.0],
        fiber_layer_heights_mm=[0.05],
        fiber_extrusion_scales=[1.0],
        speed_mm_s=10.0,
        line_length_mm=10.0,
        y_spacing_mm=10.0,
        finish_lift_mm=10.0,
        prime_length_mm=0.1,
        retract_length_mm=0.1,
        prime_speed_mm_s=5.0,
        retract_speed_mm_s=5.0,
        fiber_prime_length_mm=0.1,
        fiber_retract_length_mm=0.1,
        fiber_prime_speed_mm_s=5.0,
        fiber_retract_speed_mm_s=5.0,
        calibration=HeadCalibration(),
        tool_change_safe_lift_mm=10.0,
    )
    out = tmp_path / "composite_matrix.npz"

    export_npz(
        parse_gcode_lines(lines),
        str(out),
        dt=0.004,
        default_feed_mm_s=10.0,
        enable_extrude_wait=True,
        initial_tool_id=FIBER_TOOL_ID,
    )

    data = np.load(out)
    event_type_vocab = {
        int(value): key.decode("utf-8").rstrip("\x00")
        for key, value in zip(
            data["event_type_vocab_keys"],
            data["event_type_vocab_vals"],
        )
    }
    event_types = [event_type_vocab[int(v)] for v in data["event_type"]]
    non_empty_events = [event for event in event_types if event]

    assert non_empty_events[:2] == ["tool_change_resin", "extrude_reset"]


def test_test_matrix_gcode_rejects_more_than_45_lines():
    try:
        generate_test_matrix_gcode(
            start_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            layer_heights_mm=[0.1, 0.2, 0.3, 0.4, 0.5],
            extrusion_scales=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            speed_mm_s=10.0,
        )
    except ValueError as exc:
        assert "45" in str(exc)
    else:
        raise AssertionError("expected matrix line limit to be enforced")


def test_test_matrix_gcode_adds_prime_and_retract_as_filament_lengths():
    lines = generate_test_matrix_gcode(
        start_pose=(1.0, 2.0, 0.4, 0.0, 0.0, 0.0),
        layer_heights_mm=[0.5],
        extrusion_scales=[1.0],
        speed_mm_s=10.0,
        line_length_mm=300.0,
        prime_length_mm=5.0,
        retract_length_mm=3.0,
        prime_speed_mm_s=2.0,
        retract_speed_mm_s=8.0,
        finish_lift_mm=10.0,
    )

    parsed = parse_gcode_lines(lines)
    waits = [cmd for cmd in parsed if isinstance(cmd, ExtrudeWait)]
    print_move = next(
        cmd for cmd in parsed if isinstance(cmd, MoveCommand) and cmd.type == "PRINT"
    )

    assert len(waits) == 2
    assert math.isclose(waits[0].delta_e, 5.0)
    assert math.isclose(waits[0].feedrate, 2.0 * 60.0)
    assert math.isclose(print_move.delta_e, 300.0 * 2.0 * 0.5 * EXTRUSION_PER_MM3, abs_tol=1e-6)
    assert math.isclose(waits[1].delta_e, -3.0)
    assert math.isclose(waits[1].feedrate, 8.0 * 60.0)
