"""Measure Core density fitting against one mapped curved source NPZ."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="mapped external source NPZ")
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--densities", type=int, nargs="+", default=[0, 1, 2, 3])
    return parser.parse_args()


def _print_path_polylines(commands) -> dict[str, np.ndarray]:
    from path_processing_core.types import MoveCommand

    paths: dict[str, np.ndarray] = {}
    index = 0
    while index < len(commands):
        command = commands[index]
        if not isinstance(command, MoveCommand) or command.type != "PRINT":
            index += 1
            continue
        moves = [command]
        index += 1
        while index < len(commands):
            candidate = commands[index]
            if not (
                isinstance(candidate, MoveCommand)
                and candidate.type == "PRINT"
                and candidate.layer == command.layer
                and candidate.subtype == command.subtype
            ):
                break
            moves.append(candidate)
            index += 1
        source_line = (
            str(moves[0].line)
            if len(moves) == 1
            else f"{moves[0].line}-{moves[-1].line}"
        )
        paths[source_line] = np.asarray(
            [
                [moves[0].start_pos.x, moves[0].start_pos.y, moves[0].start_pos.z],
                *[[move.pos.x, move.pos.y, move.pos.z] for move in moves],
            ],
            dtype=np.float64,
        )
    return paths


def _point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    vector_norms = np.sum(vectors * vectors, axis=1)
    values: list[np.ndarray] = []
    for chunk in np.array_split(points, max(1, len(points) // 2048)):
        offsets = chunk[:, None, :] - starts[None, :, :]
        t = np.clip(
            np.sum(offsets * vectors[None, :, :], axis=2) / vector_norms[None, :],
            0.0,
            1.0,
        )
        nearest = starts[None, :, :] + t[:, :, None] * vectors[None, :, :]
        values.append(np.sqrt(np.sum((chunk[:, None, :] - nearest) ** 2, axis=2)).min(axis=1))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def _output_parts(output: Path) -> list[Path]:
    if output.is_file():
        return [output]
    return sorted(output.parent.glob(f"{output.stem}_part*.npz"))


def _output_size_bytes(output: Path) -> int:
    return sum(part.stat().st_size for part in _output_parts(output))


def _measure_output(output: Path, path_polylines: dict[str, np.ndarray]) -> dict[str, float | int]:
    source_line_parts = []
    xyz_parts = []
    for part in _output_parts(output):
        with np.load(part, allow_pickle=False) as data:
            source_line_parts.append(np.char.decode(data["src_line"], "utf-8"))
            xyz_parts.append(
                np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64)
            )
    if not xyz_parts:
        raise FileNotFoundError(f"Core did not create an NPZ output for {output}")
    source_lines = np.concatenate(source_line_parts)
    xyz = np.concatenate(xyz_parts)
    errors: list[np.ndarray] = []
    matched = 0
    for source_line, polyline in path_polylines.items():
        indices = np.flatnonzero(source_lines == source_line)
        if indices.size:
            errors.append(_point_to_polyline_distances(xyz[indices], polyline))
            matched += int(indices.size)
    combined = np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)
    return {
        "rows": int(len(xyz)),
        "matched_print_rows": matched,
        "mean_error_mm": float(combined.mean()),
        "p95_error_mm": float(np.quantile(combined, 0.95)),
        "max_error_mm": float(combined.max()),
    }


def main() -> int:
    args = _parse_args()
    from external_npz_preprocessor.converter import source_job_to_parsed_commands
    from external_npz_preprocessor.export_runner import convert_source_job
    from external_npz_preprocessor.param_config import load_print_params
    from external_npz_preprocessor.source_npz import load_source_npz

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = load_print_params(args.params)
    job = load_source_npz(args.source, default_abc=params.default_abc)
    path_polylines = _print_path_polylines(source_job_to_parsed_commands(job, params))
    results = []
    for density in args.densities:
        output = args.output_dir / f"density_{density}.npz"
        started = time.perf_counter()
        stats = convert_source_job(
            job,
            source_path=args.source,
            output_path=output,
            params=replace(params, density=density),
            chunk_size=2_000_000,
        )
        metrics = _measure_output(output, path_polylines)
        results.append(
            {
                "density": density,
                "export_wall_s": round(time.perf_counter() - started, 3),
                "core_total_s": round(float(stats.get("total_s", 0.0)), 3),
                "file_mib": round(_output_size_bytes(output) / 1024 / 1024, 2),
                **metrics,
            }
        )
        (args.output_dir / "results.json").write_text(
            json.dumps({"source": str(args.source), "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({"source": str(args.source), "path_count": len(path_polylines), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
