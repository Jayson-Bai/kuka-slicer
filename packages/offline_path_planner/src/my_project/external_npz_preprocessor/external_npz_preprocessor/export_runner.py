"""Run the full external NPZ to system NPZ conversion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from path_processing_core.head_calibration import (
    DEFAULT_DATA_ROOT,
    DEFAULT_HEAD_CALIBRATION_PATH,
    load_head_calibration,
)
from path_processing_core.npz_exporter import export_npz

from .converter import source_job_to_parsed_commands
from .process_params import ProcessParams
from .source_gcode import load_source_gcode, with_fiber_paths
from .source_npz import SourceJob, load_source_npz


_SURFACE_MAPPED_DEFAULT_DENSITY = 4


def default_source_npz_template_dir(data_root: str | Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    return root / "external_npz_preprocessor" / "source_npz_templates"


def default_output_npz_dir(data_root: str | Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    return root / "output_npz"


def default_output_path_for_source(
    source_path: str | Path, data_root: str | Path | None = None
) -> Path:
    source = Path(source_path).expanduser()
    return default_output_npz_dir(data_root) / source.stem / f"{source.stem}.npz"


def ensure_default_data_dirs(data_root: str | Path | None = None) -> None:
    default_source_npz_template_dir(data_root).mkdir(parents=True, exist_ok=True)
    default_output_npz_dir(data_root).mkdir(parents=True, exist_ok=True)


def resolve_output_path(
    source_path: str | Path, output_path: str | Path | None, data_root: str | Path | None = None
) -> Path:
    if output_path is None or not str(output_path).strip():
        return default_output_path_for_source(source_path, data_root=data_root)
    return Path(output_path).expanduser()


def load_shared_export_offsets(
    calibration_path: str | Path = DEFAULT_HEAD_CALIBRATION_PATH,
) -> tuple[tuple[float, float, float], float]:
    calibration = load_head_calibration(calibration_path)
    tool_offset = (
        float(calibration.fiber_x_print_compensation_mm),
        float(calibration.fiber_y_print_compensation_mm),
        float(calibration.fiber_z_print_compensation_mm),
    )
    resin_z = float(calibration.resin_z_print_compensation_mm)
    return tool_offset, resin_z


def convert_external_npz(
    source_path: str | Path,
    output_path: str | Path | None,
    params: ProcessParams,
    progress_callback=None,
    calibration_path: str | Path = DEFAULT_HEAD_CALIBRATION_PATH,
    cut_lift_mm: float | None = None,
    cut_wait_s: float | None = None,
    chunk_size: int = 100000,
    commands_callback=None,
) -> dict:
    job = load_source_npz(source_path, default_abc=params.default_abc)
    return convert_source_job(
        job,
        source_path=source_path,
        output_path=output_path,
        params=params,
        progress_callback=progress_callback,
        calibration_path=calibration_path,
        cut_lift_mm=cut_lift_mm,
        cut_wait_s=cut_wait_s,
        chunk_size=chunk_size,
        commands_callback=commands_callback,
    )


def convert_gcode(
    source_path: str | Path,
    output_path: str | Path | None,
    params: ProcessParams,
    progress_callback=None,
    calibration_path: str | Path = DEFAULT_HEAD_CALIBRATION_PATH,
    cut_lift_mm: float | None = None,
    cut_wait_s: float | None = None,
    chunk_size: int = 100000,
    commands_callback=None,
    fiber_paths_by_layer=None,
) -> dict:
    """Convert native Prusa G-code through the external-NPZ Core pipeline."""

    job = load_source_gcode(source_path, default_abc=params.default_abc)
    if fiber_paths_by_layer:
        job = with_fiber_paths(
            job,
            fiber_paths_by_layer,
            default_abc=params.default_abc,
        )
    return convert_source_job(
        job,
        source_path=source_path,
        output_path=output_path,
        params=params,
        progress_callback=progress_callback,
        calibration_path=calibration_path,
        cut_lift_mm=cut_lift_mm,
        cut_wait_s=cut_wait_s,
        chunk_size=chunk_size,
        commands_callback=commands_callback,
    )


def convert_source_job(
    job: SourceJob,
    *,
    source_path: str | Path,
    output_path: str | Path | None,
    params: ProcessParams,
    progress_callback=None,
    calibration_path: str | Path = DEFAULT_HEAD_CALIBRATION_PATH,
    cut_lift_mm: float | None = None,
    cut_wait_s: float | None = None,
    chunk_size: int = 100000,
    commands_callback=None,
) -> dict:
    """Export one normalized source job through the sole Core consumer path."""

    if job.meta.get("surface_mapping") is not None and int(params.density) == 0:
        # A mapped path needs enough fitting samples to follow its Z curvature.
        # A non-zero caller value is deliberate and always takes precedence.
        params = replace(params, density=_SURFACE_MAPPED_DEFAULT_DENSITY)
    resolved_output = resolve_output_path(source_path, output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    commands = source_job_to_parsed_commands(job, params)
    if commands_callback is not None:
        commands_callback(commands)
    export_params = params.export
    file_tool_offset, file_resin_z = load_shared_export_offsets(calibration_path)
    tool_offset = (
        (
            float(export_params.fiber_x_print_compensation_mm),
            float(export_params.fiber_y_print_compensation_mm),
            float(export_params.fiber_z_print_compensation_mm),
        )
        if all(
            value is not None
            for value in (
                export_params.fiber_x_print_compensation_mm,
                export_params.fiber_y_print_compensation_mm,
                export_params.fiber_z_print_compensation_mm,
            )
        )
        else file_tool_offset
    )
    resin_z_print_compensation_mm = (
        float(export_params.resin_z_print_compensation_mm)
        if export_params.resin_z_print_compensation_mm is not None
        else file_resin_z
    )
    export_kwargs = {
        "dt": params.dt,
        "chunk_size": chunk_size,
        "default_feed_mm_s": params.travel_feed_mm_s,
        "corner_angle_deg": params.corner_angle_deg,
        "corner_retreat_ratio": params.corner_retreat_ratio,
        "density": params.density,
        "degree": params.degree,
        "max_fit_points_per_segment": params.max_fit_points_per_segment,
        "progress_callback": progress_callback,
        "enable_extrude_wait": export_params.enable_extrude_wait,
        "enable_travel_extrude_overlap": export_params.enable_travel_extrude_overlap,
        "tool_offset": tool_offset,
        "resin_z_print_compensation_mm": resin_z_print_compensation_mm,
        "initial_tool_id": export_params.initial_tool_id,
        "tool_change_safe_lift_mm": export_params.tool_change_safe_lift_mm,
        "cut_lift_mm": (
            export_params.cut_lift_mm if cut_lift_mm is None else float(cut_lift_mm)
        ),
        "cut_wait_s": (
            export_params.cut_wait_s if cut_wait_s is None else float(cut_wait_s)
        ),
        "external_npz_cut_absolute_e": export_params.external_npz_cut_absolute_e,
    }
    if export_params.fiber_retract_length_mm is not None:
        export_kwargs["fiber_retract_length_mm"] = export_params.fiber_retract_length_mm
    return export_npz(
        commands,
        str(resolved_output),
        **export_kwargs,
    )
