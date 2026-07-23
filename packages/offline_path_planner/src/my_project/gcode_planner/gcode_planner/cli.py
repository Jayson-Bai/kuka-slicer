#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线流水线 CLI：读取 GCode -> 解析 -> B 样条拟合/采样 -> 导出 npz 分片."""

import argparse
import os
import sys
import time

from .gcode_parser import (
    load_gcode_lines,
    parse_gcode_lines,
    _select_default_gcode_file,
    default_data_root,
)
from path_processing_core.npz_exporter import export_npz
from .primeline import insert_resin_primeline


def _default_output_path(gcode_path: str, output_dir: str) -> str:
    base = os.path.splitext(os.path.basename(gcode_path))[0]
    return os.path.join(output_dir, base, base + ".npz")


def main(argv=None):
    parser = argparse.ArgumentParser(description="GCode -> NPZ 离线导出（分片）")
    parser.add_argument("--gcode", type=str, help="GCode 文件路径，缺省则从 input_gcode_dir 挑首个 .gcode")
    parser.add_argument("--data-root", type=str, default=default_data_root(), help="默认数据根目录")
    parser.add_argument(
        "--input-gcode-dir",
        type=str,
        default="",
        help="GCode 输入目录（未指定 gcode 时使用）")
    parser.add_argument("--output-dir", type=str, default="", help="npz 输出目录")
    parser.add_argument("--out", type=str, default="", help="输出 npz 文件路径（优先级最高）")
    parser.add_argument("--dt", type=float, default=0.004, help="采样周期秒，默认 4ms")
    parser.add_argument(
        "--default-feed-mm-s",
        type=float,
        default=10.0,
        help="无有效 F 时的默认速度 (mm/s)")
    parser.add_argument("--corner-angle-deg", type=float, default=10.0, help="角点判定夹角阈值（度）")
    parser.add_argument("--corner-retreat-ratio", type=float, default=0.2, help="角点回退比例（0-0.49）")
    parser.add_argument("--density", type=int, default=0, help="数据点加密密度（>=0）")
    parser.add_argument("--degree", type=int, default=3, help="B样条阶次（默认3）")
    parser.add_argument(
        "--max-fit-points-per-segment",
        type=int,
        default=20000,
        help="单段拟合点数上限，防止 density 过高导致内存/CPU 爆炸")
    parser.add_argument("--export-sleep-ms", type=int, default=0, help="导出节流：休眠毫秒数，默认 0")
    parser.add_argument("--export-yield-every", type=int, default=0, help="导出节流：每处理 N 条触发休眠，默认 0")
    parser.add_argument("--split-by-layer-type", action="store_true", help="按层+打印子类型分别导出 npz")
    parser.add_argument(
        "--plot-layer-xy",
        action="store_true",
        help="导出后按层生成 XY 路径图（仅对 split-by-layer-type 生效）")
    parser.add_argument("--plot-stride", type=int, default=5, help="绘图抽样步长，默认 5")
    parser.add_argument("--tool-offset-x", type=float, default=0.0,
                        help="Tool 1 X offset from Tool 2 (mm)")
    parser.add_argument("--tool-offset-y", type=float, default=0.0,
                        help="Tool 1 Y offset from Tool 2 (mm)")
    parser.add_argument("--tool-offset-z", type=float, default=0.0,
                        help="Tool 1 Z offset from Tool 2 (mm)")
    parser.add_argument(
        "--resin-z-print-compensation-mm",
        type=float,
        default=0.0,
        help="Resin Z print compensation prepended as a Z travel before formal printing (mm)")
    parser.add_argument("--cut-lift-mm", type=float, default=20.0, help="Cut lift distance (mm)")
    parser.add_argument("--cut-wait-s", type=float, default=15.0, help="Cut wait time from event trigger (s)")
    args = parser.parse_args(argv)

    input_dir = args.input_gcode_dir or os.path.join(args.data_root, "input_gcode")
    output_dir = args.output_dir or os.path.join(args.data_root, "output_npz")

    gcode_path = args.gcode or _select_default_gcode_file(input_dir)
    if not gcode_path or not os.path.exists(gcode_path):
        print(f"[错误] 未找到 GCode 文件: {gcode_path or '(empty)'}", file=sys.stderr)
        return 1

    if args.out:
        output_path = args.out
        out_dir = os.path.dirname(output_path) or "."
    else:
        out_dir = output_dir
        output_path = _default_output_path(gcode_path, out_dir)

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    lines = load_gcode_lines(gcode_path)
    t1 = time.perf_counter()
    parsed = insert_resin_primeline(parse_gcode_lines(lines))
    t2 = time.perf_counter()
    stats = export_npz(
        parsed,
        output_path,
        dt=args.dt,
        chunk_size=5000000,
        default_feed_mm_s=args.default_feed_mm_s,
        corner_angle_deg=args.corner_angle_deg,
        corner_retreat_ratio=args.corner_retreat_ratio,
        density=args.density,
        degree=args.degree,
        max_fit_points_per_segment=args.max_fit_points_per_segment,
        export_sleep_ms=args.export_sleep_ms,
        export_yield_every=args.export_yield_every,
        split_by_layer_type=args.split_by_layer_type,
        plot_layer_xy=args.plot_layer_xy,
        plot_stride=args.plot_stride,
        tool_offset=(args.tool_offset_x, args.tool_offset_y, args.tool_offset_z),
        enable_extrude_wait=True,
        resin_z_print_compensation_mm=args.resin_z_print_compensation_mm,
        cut_lift_mm=args.cut_lift_mm,
        cut_wait_s=args.cut_wait_s,
    )
    t3 = time.perf_counter()
    print("[信息] 导出完成: %s (npz, chunk<=5000000)" % output_path)
    print(
        (
            "[信息] 耗时统计: 读取GCode=%.3fs, 解析=%.3fs, "
            "导出总计=%.3fs (拟合=%.3fs, 采样=%.3fs, 写入=%.3fs, "
            "清单=%.3fs, 绘图=%.3fs, 行数=%d, 分片=%d)"
        )
        % (
            t1 - t0,
            t2 - t1,
            t3 - t2,
            stats.get("fit_s", 0.0),
            stats.get("sample_s", 0.0),
            stats.get("write_s", 0.0),
            stats.get("manifest_s", 0.0),
            stats.get("plot_s", 0.0),
            stats.get("rows", 0),
            stats.get("parts", 0),
        )
    )
    print(
        (
            "[信息] 拟合细分: 生成点=%.3fs, 密度加密=%.3fs, "
            "准备数据=%.3fs, 参数化=%.3fs, 节点生成=%.3fs, "
            "最小二乘=%.3fs, 后处理=%.3fs"
        )
        % (
            stats.get("fit_gen_points_s", 0.0),
            stats.get("fit_density_s", 0.0),
            stats.get("fit_prepare_data_s", 0.0),
            stats.get("fit_param_s", 0.0),
            stats.get("fit_knot_s", 0.0),
            stats.get("fit_lsq_s", 0.0),
            stats.get("fit_post_ctrl_s", 0.0),
        )
    )
    print(
        "[信息] 最小二乘细分: 基函数矩阵=%.3fs, Qk构造=%.3fs, 法方程构造=%.3fs, 求解=%.3fs, 合计=%.3fs"
        % (
            stats.get("fit_lsq_basis_build_s", 0.0),
            stats.get("fit_lsq_qk_build_s", 0.0),
            stats.get("fit_lsq_normal_mat_s", 0.0),
            stats.get("fit_lsq_solve_s", 0.0),
            stats.get("fit_lsq_total_s", 0.0),
        )
    )
    print(
        "[信息] 采样细分: 弧长映射=%.3fs, u查找=%.3fs, deBoor=%.3fs, 姿态=%.3fs, 挤出=%.3fs"
        % (
            stats.get("sample_arc_map_s", 0.0),
            stats.get("sample_lookup_s", 0.0),
            stats.get("sample_deboor_s", 0.0),
            stats.get("sample_pose_s", 0.0),
            stats.get("sample_extrude_s", 0.0),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
