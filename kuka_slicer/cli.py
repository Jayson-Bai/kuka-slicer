from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .external_npz import ExternalSourceJob, MaterialPaths, write_external_source_npz
from .slicer import (
    DEFAULT_FIBER_LAYER_HEIGHT_MM,
    DEFAULT_FIBER_LINE_WIDTH_MM,
    DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
    DEFAULT_RESIN_INFILL_OVERLAP_PERCENT,
    DEFAULT_RESIN_LAYER_HEIGHT_MM,
    DEFAULT_RESIN_LINE_WIDTH_MM,
    DEFAULT_PRUSA_CONTINUITY_SMOOTHING_ANGLE_DEGREES,
    PySLMConfig,
    SliceConfig,
    normalize_job_xy_origin,
    recommended_pyslm_strategy_defaults,
    slice_mesh_to_job,
)
from .stl_io import load_stl
from .ui_server import run_ui_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuka-slicer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    slice_parser = subparsers.add_parser("slice", help="slice STL into external source NPZ")
    slice_parser.add_argument("input_stl", type=Path)
    slice_parser.add_argument("output_npz", type=Path)
    slice_parser.add_argument(
        "--layer-height",
        type=float,
        help=(
            f"layer height in mm; defaults by material "
            f"(R={DEFAULT_RESIN_LAYER_HEIGHT_MM}, F={DEFAULT_FIBER_LAYER_HEIGHT_MM})"
        ),
    )
    slice_parser.add_argument(
        "--line-width",
        type=float,
        help=(
            f"nominal process line width in mm; defaults by material "
            f"(R={DEFAULT_RESIN_LINE_WIDTH_MM}, F={DEFAULT_FIBER_LINE_WIDTH_MM})"
        ),
    )
    slice_parser.add_argument(
        "--planning-line-width",
        type=float,
        default=None,
        help=(
            "measured flattened resin width used only for Prusa toolpath spacing, "
            "overlap, and deposited-width checks; defaults to --line-width and "
            "does not change the NPZ nominal line width or extrusion multiplier; "
            "strict measured-width mode safely executes grid, triangles, and "
            "gyroid as documented single-axis layer schedules"
        ),
    )
    slice_parser.add_argument("--z-min", type=float)
    slice_parser.add_argument("--z-max", type=float)
    slice_parser.add_argument("--material", choices=["R", "F"], default="R")
    slice_parser.add_argument(
        "--build-axis",
        choices=["x", "y", "z"],
        default="z",
        help="source STL axis used as the layer-height/build direction",
    )
    slice_parser.add_argument(
        "--slicing-kernel",
        choices=["legacy", "pyslm"],
        default="legacy",
        help="toolpath kernel; pyslm is independent and requires optional dependencies",
    )
    slice_parser.add_argument(
        "--pyslm-hatcher",
        choices=["basic", "stripe", "island", "basic_island"],
        default="basic",
        help="PySLM native hatcher strategy",
    )
    slice_parser.add_argument("--pyslm-hatch-angle", type=float)
    slice_parser.add_argument("--pyslm-layer-angle-increment", type=float, default=0.0)
    slice_parser.add_argument("--pyslm-hatch-distance", type=float)
    slice_parser.add_argument("--pyslm-contour-offset", type=float)
    slice_parser.add_argument("--pyslm-spot-compensation", type=float)
    slice_parser.add_argument("--pyslm-volume-offset-hatch", type=float)
    slice_parser.add_argument("--pyslm-num-outer-contours", type=int)
    slice_parser.add_argument("--pyslm-num-inner-contours", type=int)
    slice_parser.add_argument(
        "--pyslm-scan-contour-first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    slice_parser.add_argument(
        "--pyslm-hatch-sort",
        choices=["none", "alternate", "unidirectional", "linear", "directional"],
        default="none",
    )
    slice_parser.add_argument("--pyslm-stripe-width", type=float)
    slice_parser.add_argument("--pyslm-stripe-overlap", type=float)
    slice_parser.add_argument("--pyslm-stripe-offset", type=float)
    slice_parser.add_argument("--pyslm-island-width", type=float)
    slice_parser.add_argument("--pyslm-island-overlap", type=float)
    slice_parser.add_argument("--pyslm-island-offset", type=float)
    slice_parser.add_argument(
        "--pyslm-fix-polygons",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    slice_parser.add_argument("--pyslm-simplification-factor", type=float)
    slice_parser.add_argument(
        "--pyslm-simplification-preserve-topology",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    slice_parser.add_argument(
        "--pyslm-simplification-mode",
        choices=["absolute", "bound"],
        default="absolute",
    )
    slice_parser.add_argument("--curve", choices=["flat", "sinusoidal"], default="flat")
    slice_parser.add_argument("--curve-amplitude", type=float, default=0.0)
    slice_parser.add_argument("--curve-period", type=float, default=50.0)
    slice_parser.add_argument(
        "--infill-pattern",
        choices=[
            "none",
            "rectilinear",
            "aligned_rectilinear",
            "line",
            "grid",
            "triangles",
            "gyroid",
            "concentric",
            "zigzag",
            "isotropic",
        ],
        default="rectilinear",
        help="resin fill pattern",
    )
    slice_parser.add_argument(
        "--infill-density",
        type=float,
        default=DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
        help="resin infill density percent in the range [0, 100]",
    )
    slice_parser.add_argument(
        "--infill-overlap",
        type=float,
        default=DEFAULT_RESIN_INFILL_OVERLAP_PERCENT,
        help="resin path overlap percent used for infill spacing and wall overlap",
    )
    slice_parser.add_argument(
        "--triangle-path-optimization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reorder and reverse legacy triangle paths to reduce travel",
    )
    slice_parser.add_argument(
        "--zigzag-path-optimization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reorder, reverse, and merge legacy zigzag paths",
    )
    slice_parser.add_argument("--perimeter-count", type=int, default=2)
    slice_parser.add_argument(
        "--smoothing-angle",
        type=float,
        default=DEFAULT_PRUSA_CONTINUITY_SMOOTHING_ANGLE_DEGREES,
        help="corner angle threshold in degrees for resin path smoothing",
    )
    slice_parser.add_argument(
        "--smoothing-radius-factor",
        type=float,
        default=0.35,
        help="resin smoothing radius as a fraction of line width",
    )

    template_parser = subparsers.add_parser(
        "make-template", help="write the documented two-layer R/F template"
    )
    template_parser.add_argument("output_npz", type=Path)

    ui_parser = subparsers.add_parser("ui", help="start the local slicer web UI")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8765)
    ui_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    args = parser.parse_args(argv)
    if args.command == "slice":
        return _slice_command(args)
    if args.command == "make-template":
        return _template_command(args)
    if args.command == "ui":
        run_ui_server(args.host, args.port, args.output_dir)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


def _slice_command(args: argparse.Namespace) -> int:
    mesh = load_stl(args.input_stl)
    material_defaults = {
        "R": (DEFAULT_RESIN_LAYER_HEIGHT_MM, DEFAULT_RESIN_LINE_WIDTH_MM),
        "F": (DEFAULT_FIBER_LAYER_HEIGHT_MM, DEFAULT_FIBER_LINE_WIDTH_MM),
    }
    layer_height = (
        material_defaults[args.material][0]
        if args.layer_height is None
        else args.layer_height
    )
    line_width = (
        material_defaults[args.material][1]
        if args.line_width is None
        else args.line_width
    )
    pyslm_strategy_defaults = recommended_pyslm_strategy_defaults(layer_height, line_width)
    config = SliceConfig(
        layer_height=layer_height,
        line_width=line_width,
        planning_line_width=args.planning_line_width,
        material=args.material,
        slicing_kernel=args.slicing_kernel,
        pyslm=PySLMConfig(
            hatcher=args.pyslm_hatcher,
            hatch_angle=args.pyslm_hatch_angle,
            layer_angle_increment=args.pyslm_layer_angle_increment,
            hatch_distance=args.pyslm_hatch_distance,
            contour_offset=args.pyslm_contour_offset,
            spot_compensation=args.pyslm_spot_compensation,
            volume_offset_hatch=args.pyslm_volume_offset_hatch,
            num_outer_contours=args.pyslm_num_outer_contours,
            num_inner_contours=args.pyslm_num_inner_contours,
            scan_contour_first=args.pyslm_scan_contour_first,
            hatch_sort=args.pyslm_hatch_sort,
            stripe_width=(
                pyslm_strategy_defaults.width
                if args.pyslm_stripe_width is None
                else args.pyslm_stripe_width
            ),
            stripe_overlap=(
                pyslm_strategy_defaults.overlap
                if args.pyslm_stripe_overlap is None
                else args.pyslm_stripe_overlap
            ),
            stripe_offset=(
                pyslm_strategy_defaults.offset
                if args.pyslm_stripe_offset is None
                else args.pyslm_stripe_offset
            ),
            island_width=(
                pyslm_strategy_defaults.width
                if args.pyslm_island_width is None
                else args.pyslm_island_width
            ),
            island_overlap=(
                pyslm_strategy_defaults.overlap
                if args.pyslm_island_overlap is None
                else args.pyslm_island_overlap
            ),
            island_offset=(
                pyslm_strategy_defaults.offset
                if args.pyslm_island_offset is None
                else args.pyslm_island_offset
            ),
            fix_polygons=args.pyslm_fix_polygons,
            simplification_factor=args.pyslm_simplification_factor,
            simplification_preserve_topology=args.pyslm_simplification_preserve_topology,
            simplification_mode=args.pyslm_simplification_mode,
        ),
        build_axis=args.build_axis,
        z_min=args.z_min,
        z_max=args.z_max,
        curve_mode=args.curve,
        curve_amplitude=args.curve_amplitude,
        curve_period=args.curve_period,
        infill_pattern=args.infill_pattern,
        infill_density=args.infill_density,
        infill_overlap=args.infill_overlap,
        triangle_path_optimization=args.triangle_path_optimization,
        zigzag_path_optimization=args.zigzag_path_optimization,
        perimeter_count=args.perimeter_count,
        smoothing_angle=args.smoothing_angle,
        smoothing_radius_factor=args.smoothing_radius_factor,
    )
    job = slice_mesh_to_job(mesh, config)
    normalize_job_xy_origin(job)
    write_external_source_npz(job, args.output_npz)
    path_count = sum(len(group.paths) for group in job.material_paths)
    print(f"wrote {args.output_npz} with {len(job.material_paths)} layer/material groups and {path_count} paths")
    return 0


def _template_command(args: argparse.Namespace) -> int:
    job = ExternalSourceJob(
        material_paths=[
            MaterialPaths(0, "R", [_path([[0, 0, 0.5], [30, 0, 0.5], [30, 20, 0.5]]), _path([[5, 5, 0.5], [25, 5, 0.5]])]),
            MaterialPaths(0, "F", [_path([[2, 2, 0.6], [28, 18, 0.6]])]),
            MaterialPaths(1, "R", [_path([[0, 0, 1.0], [30, 0, 1.0], [30, 20, 1.0]])]),
            MaterialPaths(1, "F", [_path([[2, 18, 1.1], [28, 2, 1.1]])]),
        ],
        meta={
            "format": "external_layer_paths_v1",
            "unit": "mm",
            "point_columns": ["x", "y", "z"],
            "materials": {"R": "resin", "F": "fiber"},
        },
    )
    write_external_source_npz(job, args.output_npz)
    print(f"wrote template {args.output_npz}")
    return 0


def _path(points: list[list[float]]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)
