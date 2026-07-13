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
    SliceConfig,
    normalize_job_xy_origin,
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
            f"path line width in mm; defaults by material "
            f"(R={DEFAULT_RESIN_LINE_WIDTH_MM}, F={DEFAULT_FIBER_LINE_WIDTH_MM})"
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
    slice_parser.add_argument("--curve", choices=["flat", "sinusoidal"], default="flat")
    slice_parser.add_argument("--curve-amplitude", type=float, default=0.0)
    slice_parser.add_argument("--curve-period", type=float, default=50.0)
    slice_parser.add_argument(
        "--infill-pattern",
        choices=[
            "contour",
            "contour_offset",
            "lines_x",
            "lines_y",
            "grid",
            "triangles",
            "gyroid",
            "diagonal",
            "alternating_diagonal",
        ],
        default="lines_x",
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
    config = SliceConfig(
        layer_height=args.layer_height,
        line_width=args.line_width,
        material=args.material,
        build_axis=args.build_axis,
        z_min=args.z_min,
        z_max=args.z_max,
        curve_mode=args.curve,
        curve_amplitude=args.curve_amplitude,
        curve_period=args.curve_period,
        infill_pattern=args.infill_pattern,
        infill_density=args.infill_density,
        infill_overlap=args.infill_overlap,
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
