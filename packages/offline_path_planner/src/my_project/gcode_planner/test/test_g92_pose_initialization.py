from gcode_planner.gcode_parser import parse_gcode_lines
from gcode_planner.primeline import insert_resin_primeline
from path_processing_core.types import ExtrudeWait, MoveCommand


def test_g92_xyzabc_initializes_pose_without_motion():
    parsed = parse_gcode_lines([
        "G90",
        "M82",
        "G92 X1.0 Y2.0 Z3.0 A4.0 B5.0 C6.0 E0",
        "G1 Z4.0 F60",
    ])

    moves = [cmd for cmd in parsed if isinstance(cmd, MoveCommand)]

    assert len(moves) == 1
    assert moves[0].start_pos.x == 1.0
    assert moves[0].start_pos.y == 2.0
    assert moves[0].start_pos.z == 3.0
    assert moves[0].start_pos.a == 4.0
    assert moves[0].start_pos.b == 5.0
    assert moves[0].start_pos.c == 6.0
    assert moves[0].pos.z == 4.0


def test_insert_resin_primeline_before_first_gcode_print_path():
    parsed = parse_gcode_lines([
        "G90",
        "M82",
        "T1",
        "G92 X10 Y20 Z0.5 A1 B2 C3 E0",
        "G1 X10 Y20 Z0.5 F600",
        "G1 E18 F900",
        "G1 X30 Y20 E20 F600",
        "G1 E5 F1800",
    ])

    out = insert_resin_primeline(parsed, length_mm=100.0, y_offset_mm=10.0)

    prints = [cmd for cmd in out if isinstance(cmd, MoveCommand) and cmd.type == "PRINT"]
    waits = [cmd for cmd in out if isinstance(cmd, ExtrudeWait)]
    assert len(prints) == 2
    primeline = prints[0]
    assert primeline.raw == "gcode_primeline"
    assert primeline.start_pos.x == 10.0
    assert primeline.start_pos.y == 10.0
    assert primeline.start_pos.z == 0.5
    assert primeline.pos.x == 110.0
    assert primeline.pos.y == 10.0
    assert primeline.delta_e == 10.0
    assert primeline.e_val == 28.0
    assert [wait.delta_e for wait in waits] == [18.0, -15.0, 18.0, -15.0]
    travels = [cmd for cmd in out if isinstance(cmd, MoveCommand) and cmd.type == "TRAVEL"]
    assert any(cmd.raw == "gcode_primeline_return_travel" for cmd in travels)
    assert prints[1].start_pos.x == 10.0
    assert prints[1].start_pos.y == 20.0
    assert prints[1].e_val == 33.0
