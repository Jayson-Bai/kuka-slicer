"""Run the full external NPZ to system NPZ conversion."""

from __future__ import annotations

from pathlib import Path

from path_processing_core.head_calibration import (
    DEFAULT_DATA_ROOT,
    DEFAULT_HEAD_CALIBRATION_PATH,
    load_head_calibration,
)
from path_processing_core.npz_exporter import export_npz

from .converter import source_job_to_parsed_commands
from .process_params import ProcessParams
from .source_npz import load_source_npz


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
    cut_lift_mm: float = 20.0,
    cut_wait_s: float = 15.0,
) -> dict:
    resolved_output = resolve_output_path(source_path, output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    job = load_source_npz(source_path, default_abc=params.default_abc)
    commands = source_job_to_parsed_commands(job, params)
    tool_offset, resin_z_print_compensation_mm = load_shared_export_offsets(calibration_path)
    return export_npz(
        commands,
        str(resolved_output),
        dt=params.dt,
        default_feed_mm_s=params.travel_feed_mm_s,
        corner_angle_deg=params.corner_angle_deg,
        corner_retreat_ratio=params.corner_retreat_ratio,
        density=params.density,
        degree=params.degree,
        max_fit_points_per_segment=params.max_fit_points_per_segment,
        progress_callback=progress_callback,
        enable_extrude_wait=True,
        tool_offset=tool_offset,
        resin_z_print_compensation_mm=resin_z_print_compensation_mm,
        cut_lift_mm=cut_lift_mm,
        cut_wait_s=cut_wait_s,
        external_npz_cut_absolute_e=True,
    )
