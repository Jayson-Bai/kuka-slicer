#!/usr/bin/env python3
# Roughly estimate NPZ row count without writing files.

from __future__ import annotations

import argparse
import os
import sys

# Allow running from repo without installing.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src", "my_project", "gcode_planner")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from gcode_planner.gcode_parser import load_gcode_lines, parse_gcode_lines
from gcode_planner.types import (
    MoveCommand,
    GlobalCurveCommand,
    ToolChangeCommand,
    MCommand,
    ResetECommand,
)
from gcode_planner.bspline_approximation import GlobalSplinePlanner
from gcode_planner.polynomial_interpolator import sample_global_curve_iter


def estimate_rows(
    gcode_path: str,
    dt: float = 0.004,
    corner_angle_deg: float = 10.0,
    corner_retreat_ratio: float = 0.2,
    density: int = 0,
    degree: int = 3,
    default_feed_mm_s: float = 10.0,
) -> int:
    lines = load_gcode_lines(gcode_path)
    parsed = parse_gcode_lines(lines)

    planner = GlobalSplinePlanner()
    buffer: list[MoveCommand] = []
    current_type: str | None = None
    last_feedrate_mm_min = None
    count = 0

    def _make_gc(move: MoveCommand) -> GlobalCurveCommand:
        return GlobalCurveCommand(
            type=move.type,
            cmd="SPLINE",
            start_pos=move.start_pos,
            control_points=[move.pos, move.pos, move.pos],
            e_val=move.e_val,
            delta_e=move.delta_e,
            feedrate=move.feedrate,
            line=move.line,
            raw=move.raw or "fallback_linear",
            constraints=[],
            original_moves=[move],
        )

    def _append_sample(gc: GlobalCurveCommand) -> None:
        nonlocal count, last_feedrate_mm_min
        feed_mm_min = gc.feedrate if (gc.feedrate is not None and gc.feedrate > 0) else last_feedrate_mm_min
        if feed_mm_min is None or feed_mm_min <= 0:
            target_velocity = default_feed_mm_s
        else:
            target_velocity = feed_mm_min / 60.0
        if gc.feedrate is not None and gc.feedrate > 0:
            last_feedrate_mm_min = gc.feedrate
        for _ in sample_global_curve_iter(gc, dt=dt, target_velocity=target_velocity):
            count += 1

    def flush_moves() -> None:
        nonlocal buffer, current_type
        if not buffer:
            return
        if len(buffer) == 1:
            gc_list = [_make_gc(buffer[0])]
        elif len(buffer) == 2:
            gc_list = [_make_gc(buffer[0]), _make_gc(buffer[1])]
        else:
            gc = planner.fit_global_curve(
                buffer,
                corner_angle_deg=corner_angle_deg,
                corner_retreat_ratio=corner_retreat_ratio,
                density=density,
                degree=degree,
            )
            if gc is None:
                raise ValueError(f"B样条拟合失败（段类型: {current_type}, 段长度: {len(buffer)})")
            gc_list = [gc]

        for gc in gc_list:
            _append_sample(gc)
        buffer = []
        current_type = None

    for cmd in parsed:
        if isinstance(cmd, (ToolChangeCommand, MCommand, ResetECommand)):
            flush_moves()
            # 事件行占 1
            count += 1
            continue

        if isinstance(cmd, MoveCommand):
            if cmd.is_pure_state_change:
                continue
            if current_type is None:
                current_type = cmd.type
            if cmd.type != current_type:
                flush_moves()
                current_type = cmd.type
            buffer.append(cmd)
            continue

        if isinstance(cmd, GlobalCurveCommand):
            flush_moves()
            _append_sample(cmd)
            continue

    flush_moves()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate NPZ rows without writing files.")
    parser.add_argument("--gcode", required=True, help="GCode 文件路径")
    parser.add_argument("--dt", type=float, default=0.004, help="采样周期秒，默认 4ms")
    parser.add_argument("--corner-angle-deg", type=float, default=10.0, help="角点判定夹角阈值（度）")
    parser.add_argument("--corner-retreat-ratio", type=float, default=0.2, help="角点回退比例（0-0.49）")
    parser.add_argument("--density", type=int, default=0, help="数据点加密密度（>=0）")
    parser.add_argument("--degree", type=int, default=3, help="B样条阶次（默认3）")
    parser.add_argument("--default-feed-mm-s", type=float, default=10.0, help="无有效 F 时的默认速度 (mm/s)")
    args = parser.parse_args()

    total = estimate_rows(
        args.gcode,
        dt=args.dt,
        corner_angle_deg=args.corner_angle_deg,
        corner_retreat_ratio=args.corner_retreat_ratio,
        density=args.density,
        degree=args.degree,
        default_feed_mm_s=args.default_feed_mm_s,
    )
    print(f"[info] estimated_rows={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
