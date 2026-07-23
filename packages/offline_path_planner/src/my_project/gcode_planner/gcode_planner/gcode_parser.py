#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from typing import List

try:
    import rclpy
    from rclpy.node import Node
except ModuleNotFoundError:
    rclpy = None

    class Node:
        """Import placeholder used when the optional ROS 2 runtime is absent."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "The ROS node entry point requires rclpy. "
                "Use gcode_planner.cli for standalone offline conversion."
            )

from path_processing_core.types import (
    Position,
    MoveCommand,
    ExtrudeWait,
    ResetECommand,
    ToolChangeCommand,
    MCommand,
    ParsedCommandList,
)


class MachineState:
    """解析内部状态，纯 Python，无 ROS 依赖."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0

        self.e = 0.0
        self.f = 100.0  # 默认进给速度 mm/min

        self.is_absolute_coord = True  # G90
        self.is_absolute_e = True  # M82
        self.current_tool = 0  # T0
        self.current_layer = 0
        self.current_subtype = "UNKNOWN"


# ------------------- 纯函数：文件读取与解析 -------------------

def load_gcode_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]


def parse_gcode_lines(lines: List[str]) -> ParsedCommandList:
    state = MachineState()
    parsed: ParsedCommandList = []
    handlers = _get_command_handlers(parsed)

    for line_idx, line in enumerate(lines):
        raw_line = line
        stripped = raw_line.strip()
        if stripped.startswith(";"):
            _handle_comment(state, stripped)
            continue
        if ";" in raw_line:
            raw_line = raw_line.split(";", 1)[0]
        clean_line = raw_line.strip()
        if not clean_line:
            continue

        parts = clean_line.split()
        cmd_type = parts[0].upper()

        params = {}
        for part in parts[1:]:
            match = re.match(r"([A-Z])([-+]?\d*\.?\d+)", part.upper())
            if match:
                key, val = match.groups()
                params[key] = float(val)

        handler = handlers.get(cmd_type)
        if handler:
            handler(state, params, line_idx, cmd_type, clean_line)
        elif cmd_type.startswith("T"):
            _handle_tool_change(parsed, state, cmd_type, line_idx, clean_line)
        elif cmd_type.startswith("M"):
            _handle_m_command(parsed, state, cmd_type, params, line_idx, clean_line)
        else:
            continue

    return parsed


# ------------------- ROS 封装（可选） -------------------

class GCodeParser(Node):
    def __init__(self):
        super().__init__("gcode_parser")
        self.declare_parameter("data_root", default_data_root())
        self.declare_parameter("input_gcode_dir", "")
        self.declare_parameter("output_npz_dir", "")
        self.declare_parameter("gcode_file_path", "")

        data_root = self.get_parameter("data_root").get_parameter_value().string_value
        input_gcode_dir = self.get_parameter("input_gcode_dir").get_parameter_value().string_value
        output_npz_dir = self.get_parameter("output_npz_dir").get_parameter_value().string_value
        gcode_file_path = self.get_parameter("gcode_file_path").get_parameter_value().string_value

        if not input_gcode_dir:
            input_gcode_dir = os.path.join(data_root, "input_gcode")
        if not output_npz_dir:
            output_npz_dir = os.path.join(data_root, "output_npz")
        if not gcode_file_path:
            gcode_file_path = _select_default_gcode_file(input_gcode_dir)

        self.gcode_file_path = gcode_file_path
        self.input_gcode_dir = input_gcode_dir
        self.output_npz_dir = output_npz_dir

        if not self.gcode_file_path:
            self.get_logger().error("No GCode file path resolved.")
            rclpy.shutdown()
            return

        try:
            lines = load_gcode_lines(self.gcode_file_path)
            self.parsed_commands = parse_gcode_lines(lines)
            self.get_logger().info(
                f"Parsed {len(self.parsed_commands)} commands from {self.gcode_file_path}")
        except Exception as e:
            self.get_logger().error(f"Error reading/parsing GCode file: {e}")
            self.parsed_commands = []


# ------------------- 解析处理函数 -------------------

def _get_command_handlers(parsed: ParsedCommandList):
    return {
        "G90": lambda s, p, i, c, r=None: _handle_g90(s),
        "G91": lambda s, p, i, c, r=None: _handle_g91(s),
        "M82": lambda s, p, i, c, r=None: _handle_m82(s),
        "M83": lambda s, p, i, c, r=None: _handle_m83(s),
        "G92": lambda s, p, i, c, r=None: _handle_g92(parsed, s, p, i, r),
        "G0": lambda s, p, i, c, r=None: _handle_move(parsed, s, p, i, c, r),
        "G1": lambda s, p, i, c, r=None: _handle_move(parsed, s, p, i, c, r),
        "G21": lambda s, p, i, c, r=None: None,
        "G28": lambda s, p, i, c, r=None: None,
        "G29": lambda s, p, i, c, r=None: None,
    }


def _handle_g90(state: MachineState):
    state.is_absolute_coord = True


def _handle_g91(state: MachineState):
    state.is_absolute_coord = False


def _handle_m82(state: MachineState):
    state.is_absolute_e = True


def _handle_m83(state: MachineState):
    state.is_absolute_e = False


def _handle_g92(parsed: ParsedCommandList, state: MachineState, params, line_idx, raw_line=None):
    if "X" in params:
        state.x = params["X"]
    if "Y" in params:
        state.y = params["Y"]
    if "Z" in params:
        state.z = params["Z"]
    if "A" in params:
        state.a = params["A"]
    if "B" in params:
        state.b = params["B"]
    if "C" in params:
        state.c = params["C"]
    if "E" in params:
        new_e = params["E"]
        state.e = new_e
        parsed.append(
            ResetECommand(
                type="RESET_E",
                val=new_e,
                line=line_idx + 1,
                layer=state.current_layer,
                subtype=state.current_subtype or "UNKNOWN",
                raw=raw_line,
                pose=Position(state.x, state.y, state.z, state.a, state.b, state.c),
            )
        )


def _handle_move(
        parsed: ParsedCommandList,
        state: MachineState,
        params,
        line_idx,
        cmd_type,
        raw_line=None):
    prev_pos = (state.x, state.y, state.z, state.a, state.b, state.c)
    if state.is_absolute_coord:
        target_x = params.get("X", state.x)
        target_y = params.get("Y", state.y)
        target_z = params.get("Z", state.z)
        target_a = params.get("A", state.a)
        target_b = params.get("B", state.b)
        target_c = params.get("C", state.c)
    else:
        target_x = state.x + params.get("X", 0.0)
        target_y = state.y + params.get("Y", 0.0)
        target_z = state.z + params.get("Z", 0.0)
        target_a = state.a + params.get("A", 0.0)
        target_b = state.b + params.get("B", 0.0)
        target_c = state.c + params.get("C", 0.0)

    target_f = params.get("F", state.f)

    if state.is_absolute_e:
        target_e = params.get("E", state.e)
        delta_e = target_e - state.e
        state.e = target_e
    else:
        delta_e = params.get("E", 0.0)
        state.e += delta_e

    state.x, state.y, state.z = target_x, target_y, target_z
    state.a, state.b, state.c = target_a, target_b, target_c
    state.f = target_f

    position_unchanged = (
        abs(target_x - prev_pos[0]) < 1e-6
        and abs(target_y - prev_pos[1]) < 1e-6
        and abs(target_z - prev_pos[2]) < 1e-6
        and abs(target_a - prev_pos[3]) < 1e-6
        and abs(target_b - prev_pos[4]) < 1e-6
        and abs(target_c - prev_pos[5]) < 1e-6
    )

    if position_unchanged and abs(delta_e) > 0:
        wait_sec = 0.0
        if target_f > 0:
            wait_sec = abs(delta_e) / target_f * 60.0
            parsed.append(
                ExtrudeWait(
                    type="EXTRUDE_WAIT",
                    wait_sec=wait_sec,
                    delta_e=delta_e,
                    feedrate=target_f,
                    line=line_idx + 1,
                    layer=state.current_layer,
                    subtype=state.current_subtype or "UNKNOWN",
                    raw=raw_line,
                )
            )
            return

    move_pos = Position(
        x=state.x,
        y=state.y,
        z=state.z,
        a=state.a,
        b=state.b,
        c=state.c,
    )
    move_start_pos = Position(
        x=prev_pos[0],
        y=prev_pos[1],
        z=prev_pos[2],
        a=prev_pos[3],
        b=prev_pos[4],
        c=prev_pos[5],
    )

    if cmd_type == "G0":
        move_type = "TRAVEL"
    else:
        move_type = "PRINT" if abs(delta_e) > 1e-9 else "TRAVEL"

    if move_type == "TRAVEL":
        subtype = "TRAVEL"
    else:
        subtype = state.current_subtype or "UNKNOWN"

    move_cmd = MoveCommand(
        type=move_type,
        cmd=cmd_type,
        start_pos=move_start_pos,
        pos=move_pos,
        e_val=state.e,
        delta_e=delta_e,
        feedrate=state.f,
        line=line_idx + 1,
        layer=state.current_layer,
        subtype=subtype,
        raw=raw_line,
        is_pure_state_change=position_unchanged,
    )

    parsed.append(move_cmd)


def _handle_tool_change(
        parsed: ParsedCommandList,
        state: MachineState,
        cmd_type,
        line_idx,
        raw_line=None):
    try:
        tool_idx = int(cmd_type[1:])
    except (ValueError, IndexError):
        return
    state.current_tool = tool_idx
    parsed.append(
        ToolChangeCommand(
            type="TOOL_CHANGE",
            tool=tool_idx,
            line=line_idx + 1,
            layer=state.current_layer,
            subtype=state.current_subtype or "UNKNOWN",
            raw=raw_line,
        )
    )


def _handle_m_command(
        parsed: ParsedCommandList,
        state: MachineState,
        cmd_type,
        params,
        line_idx=None,
        raw_line=None):
    cmd = MCommand(
        type="M_COMMAND",
        code=cmd_type,
        params=params,
        line=line_idx + 1 if line_idx is not None else None,
        layer=state.current_layer,
        subtype=state.current_subtype or "UNKNOWN",
        raw=raw_line,
    )
    if "T" in params:
        try:
            cmd.tool = int(params["T"])
        except (ValueError, TypeError):
            cmd.tool = None
    parsed.append(cmd)


def _handle_comment(state: MachineState, line: str):
    # Marlin-style comments
    if line.startswith(";LAYER_CHANGE"):
        state.current_layer += 1
        return
    if line.startswith(";LAYER:"):
        try:
            state.current_layer = int(line.split(":", 1)[1].strip())
        except Exception:
            pass
        return
    if line.startswith(";TYPE:"):
        state.current_subtype = line.split(":", 1)[1].strip()


def _select_default_gcode_file(input_dir):
    if not os.path.isdir(input_dir):
        return ""
    try:
        candidates = sorted(
            name for name in os.listdir(input_dir) if name.lower().endswith(".gcode")
        )
    except Exception:
        return ""
    if not candidates:
        return ""
    return os.path.join(input_dir, candidates[0])


def default_data_root() -> str:
    """默认 data 目录（相对当前文件向上推到 kuka_ram_ws/data."""
    here = os.path.abspath(__file__)
    candidate = os.path.abspath(os.path.join(here, "../../../../../data"))
    return candidate


def main(args=None):
    if rclpy is None:
        raise RuntimeError(
            "The ROS node entry point requires rclpy. "
            "Use kuka-offline-gcode for standalone offline conversion."
        )
    rclpy.init(args=args)
    gcode_parser = GCodeParser()
    try:
        rclpy.spin(gcode_parser)
    except KeyboardInterrupt:
        pass
    finally:
        gcode_parser.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
