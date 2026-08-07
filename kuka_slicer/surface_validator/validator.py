"""Validate a planar NPZ, its mapped counterpart, and a surface definition.

This module intentionally has no path-editing operations.  Its only output is
evidence about whether the existing ``curved.npz`` can move to the Core
interface-validation stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Literal

import numpy as np

from ..surface_mapper.contracts import SourceNPZ, SurfaceTarget
from ..surface_mapper.sampling import SurfaceSamplingConfig, resample_material_paths


Status = Literal["pass", "warning", "fail"]
_PATH_KEY = re.compile(r"^layer_(\d{4,})_([RFT])$")
_E_KEY = re.compile(r"^(layer_\d{4,}_[RF])_E$")
_XY_DECIMALS = 6


@dataclass(frozen=True, slots=True)
class ValidatorLimits:
    """Explicit, conservative defaults for checks without material calibration."""

    max_slope: float = 0.5
    max_layer_gap_ratio: float = 1.5
    max_segment_fraction_of_wavelength: float = 0.125
    numeric_tolerance_mm: float = 1e-8
    extrusion_relative_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.max_slope <= 0 or self.max_layer_gap_ratio <= 0:
            raise ValueError("slope and layer-gap limits must be positive")
        if not 0 < self.max_segment_fraction_of_wavelength <= 1:
            raise ValueError("max_segment_fraction_of_wavelength must be in (0, 1]")
        if self.numeric_tolerance_mm < 0 or self.extrusion_relative_tolerance < 0:
            raise ValueError("validation tolerances must be non-negative")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One independently reviewable validation conclusion."""

    name: str
    status: Status
    summary: str
    details: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class SurfaceValidationReport:
    """Machine-readable outcome of the read-only surface validation stage."""

    checks: tuple[CheckResult, ...]
    limits: ValidatorLimits
    flat_z_bounds_mm: tuple[float, float]
    curved_z_bounds_mm: tuple[float, float]

    @property
    def status(self) -> Status:
        if any(check.status == "fail" for check in self.checks):
            return "fail"
        if any(check.status == "warning" for check in self.checks):
            return "warning"
        return "pass"

    def payload(self) -> dict[str, Any]:
        return {
            "format": "surface_validation_report_v1",
            "overall_status": self.status,
            "decision": {
                "fail": "不得进入 Core",
                "warning": "允许导出报告，但需人工确认或工艺试验",
                "pass": "可进入 Core 端到端验证",
            }[self.status],
            "limits": asdict(self.limits),
            "geometry": {
                "flat_z_bounds_mm": list(self.flat_z_bounds_mm),
                "curved_z_bounds_mm": list(self.curved_z_bounds_mm),
                "actual_max_height_mm": self.curved_z_bounds_mm[1] - self.curved_z_bounds_mm[0],
            },
            "checks": [check.payload() for check in self.checks],
        }


def validate_surface_job(
    flat_source: SourceNPZ,
    curved_source: SourceNPZ,
    target: SurfaceTarget,
    limits: ValidatorLimits | None = None,
) -> SurfaceValidationReport:
    """Validate the mapped job without mutating either source object."""

    active_limits = limits or ValidatorLimits()
    paired_flat_source = _flat_reference_for_mapped_job(flat_source, curved_source)
    checks = (
        _check_path_contract(paired_flat_source, curved_source, active_limits),
        _check_surface_identity(flat_source, curved_source, target, active_limits),
        _check_mapping_metadata(curved_source),
        _check_mapped_geometry(paired_flat_source, curved_source, target, active_limits),
        _check_layer_geometry(paired_flat_source, curved_source, active_limits),
        _check_sampling_density(curved_source, target, active_limits),
        _check_extrusion(paired_flat_source, curved_source, active_limits),
        _check_travel_risk(curved_source, active_limits),
        _check_topology_evidence(paired_flat_source, curved_source, active_limits),
    )
    return SurfaceValidationReport(
        checks=checks,
        limits=active_limits,
        flat_z_bounds_mm=flat_source.z_bounds_mm,
        curved_z_bounds_mm=curved_source.z_bounds_mm,
    )


def _flat_reference_for_mapped_job(
    flat_source: SourceNPZ, curved_source: SourceNPZ
) -> SourceNPZ:
    """Recreate the mapper's planar R/F grid before pairwise validation.

    The validator remains read-only: it only derives an in-memory reference
    when the curved job explicitly records surface-material resampling.  Older
    mapped files retain their original flat grid unchanged.
    """

    mapping = curved_source.meta.get("surface_mapping")
    sampling = mapping.get("sampling") if isinstance(mapping, dict) else None
    if not isinstance(sampling, dict):
        return flat_source
    if sampling.get("format") != "surface_material_resampling_v1":
        return flat_source
    max_segment = sampling.get("max_segment_length_mm")
    if max_segment is None:
        return flat_source
    try:
        config = SurfaceSamplingConfig(max_segment_length_mm=float(max_segment))
    except (TypeError, ValueError):
        return flat_source
    return resample_material_paths(flat_source, config).source


def _check_path_contract(
    flat_source: SourceNPZ, curved_source: SourceNPZ, limits: ValidatorLimits
) -> CheckResult:
    flat_keys = set(flat_source.path_keys)
    curved_keys = set(curved_source.path_keys)
    errors: list[str] = []
    for key in sorted(flat_keys | curved_keys):
        flat = flat_source.arrays.get(key)
        curved = curved_source.arrays.get(key)
        if flat is None or curved is None:
            errors.append(f"{key} 缺失")
            continue
        if flat.shape[:2] != curved.shape[:2]:
            errors.append(f"{key} 形状不同")
            continue
        if not np.array_equal(np.isfinite(flat[..., 0]), np.isfinite(curved[..., 0])):
            errors.append(f"{key} 的 NaN 填充不同")
            continue
        if not np.allclose(flat[..., :2], curved[..., :2], equal_nan=True, atol=limits.numeric_tolerance_mm, rtol=0.0):
            errors.append(f"{key} 的 XY 被修改")
        valid = np.isfinite(curved[..., 0])
        if not np.all(np.isfinite(curved[..., :3][valid])):
            errors.append(f"{key} 存在非有限 XYZ")
        elif np.any(curved[..., 2][valid] < -limits.numeric_tolerance_mm):
            errors.append(f"{key} 存在负 Z")
    if errors:
        return CheckResult("路径契约与 Z 安全", "fail", "路径键、XY、填充或 Z 安全性被破坏", {"errors": errors})
    return CheckResult(
        "路径契约与 Z 安全",
        "pass",
        "R/F/T 分组、逻辑层键、路径形状、XY、NaN 填充与非负有限 Z 均保持一致",
        {"path_key_count": len(flat_keys), "curved_z_bounds_mm": list(curved_source.z_bounds_mm)},
    )


def _check_surface_identity(
    flat_source: SourceNPZ, curved_source: SourceNPZ, target: SurfaceTarget, limits: ValidatorLimits
) -> CheckResult:
    flat_hash = _source_hash(flat_source)
    curved_hash = _source_hash(curved_source)
    target_hash = target.source_sha256
    hashes = {value for value in (flat_hash, curved_hash, target_hash) if value}
    if len(hashes) > 1:
        return CheckResult(
            "曲面与源模型身份",
            "fail",
            "flat.npz、curved.npz 与曲面 JSON 指向不同的源模型",
            {"flat_source_sha256": flat_hash, "curved_source_sha256": curved_hash, "target_source_sha256": target_hash},
        )
    x_min, y_min, x_max, y_max = curved_source.xy_bounds_mm
    domain_errors = [
        name
        for name, actual, bound in (
            ("x_min", x_min, 0.0),
            ("y_min", y_min, 0.0),
            ("x_max", x_max, target.width_mm),
            ("y_max", y_max, target.height_mm),
        )
        if (actual < bound - limits.numeric_tolerance_mm if name.endswith("min") else actual > bound + limits.numeric_tolerance_mm)
    ]
    if domain_errors and not hashes:
        return CheckResult(
            "曲面与源模型身份",
            "fail",
            "没有可用模型哈希，且曲线路径超出曲面 JSON 的 XY 域",
            {"outside_bounds": domain_errors, "curved_xy_bounds_mm": list(curved_source.xy_bounds_mm)},
        )
    if not hashes:
        return CheckResult(
            "曲面与源模型身份",
            "warning",
            "缺少可核验的源 STL 哈希；仅完成 XY 域检查",
            {"curved_xy_bounds_mm": list(curved_source.xy_bounds_mm), "target_domain_mm": [target.width_mm, target.height_mm]},
        )
    return CheckResult(
        "曲面与源模型身份",
        "pass",
        "源模型身份一致；曲面 JSON 与路径属于同一模型",
        {"source_sha256": next(iter(hashes)), "curved_xy_bounds_mm": list(curved_source.xy_bounds_mm)},
    )


def _check_mapping_metadata(curved_source: SourceNPZ) -> CheckResult:
    mapping = curved_source.meta.get("surface_mapping")
    if not isinstance(mapping, dict):
        return CheckResult("映射元数据", "fail", "curved.npz 缺少 surface_mapping 元数据", {})
    progression = mapping.get("progression")
    if mapping.get("format") != "surface_mapping_v1" or not isinstance(progression, dict):
        return CheckResult("映射元数据", "fail", "surface_mapping 元数据格式不完整", {"mapping": mapping})
    if progression.get("basis") != "logical_layer_index":
        return CheckResult("映射元数据", "fail", "映射没有声明逻辑层语义", {"basis": progression.get("basis")})
    alpha = progression.get("alpha_by_layer")
    if not isinstance(alpha, dict):
        return CheckResult("映射元数据", "fail", "映射元数据缺少各逻辑层 alpha", {})
    missing = [str(layer) for layer in curved_source.layer_indices if str(layer) not in alpha]
    if missing:
        return CheckResult("映射元数据", "fail", "部分逻辑层缺少 alpha 元数据", {"missing_layers": missing})
    return CheckResult("映射元数据", "pass", "曲面映射记录完整且使用逻辑层语义", {"layer_count": len(alpha)})


def _check_mapped_geometry(
    flat_source: SourceNPZ, curved_source: SourceNPZ, target: SurfaceTarget, limits: ValidatorLimits
) -> CheckResult:
    mapping = curved_source.meta.get("surface_mapping")
    if not isinstance(mapping, dict) or not isinstance(mapping.get("progression"), dict):
        return CheckResult("逐点 Z 映射", "fail", "无法从元数据验证逐点 Z 映射", {})
    alpha_by_layer = mapping["progression"].get("alpha_by_layer")
    if not isinstance(alpha_by_layer, dict):
        return CheckResult("逐点 Z 映射", "fail", "无法从元数据验证逐点 Z 映射", {})
    max_error = 0.0
    checked_points = 0
    for key in flat_source.path_keys:
        match = _PATH_KEY.match(key)
        assert match is not None
        alpha = alpha_by_layer.get(str(int(match.group(1))))
        if not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
            return CheckResult("逐点 Z 映射", "fail", f"逻辑层 {int(match.group(1))} 的 alpha 无效", {})
        flat = flat_source.arrays[key]
        curved = curved_source.arrays[key]
        valid = np.isfinite(flat[..., 0])
        expected = flat[..., 2][valid] + float(alpha) * target.surface.height(flat[..., 0][valid], flat[..., 1][valid])
        actual = curved[..., 2][valid]
        if actual.size:
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            checked_points += int(actual.size)
    if max_error > limits.numeric_tolerance_mm:
        return CheckResult(
            "逐点 Z 映射",
            "fail",
            "曲面路径与记录的双正弦映射公式不一致",
            {"checked_point_count": checked_points, "max_abs_error_mm": max_error},
        )
    return CheckResult(
        "逐点 Z 映射",
        "pass",
        "所有路径点均满足 z_flat + alpha(logical_layer) × H(x,y)",
        {"checked_point_count": checked_points, "max_abs_error_mm": max_error},
    )


def _check_layer_geometry(
    flat_source: SourceNPZ, curved_source: SourceNPZ, limits: ValidatorLimits
) -> CheckResult:
    per_layer = _layer_xy_z(curved_source)
    flat_per_layer = _layer_xy_z(flat_source)
    inversion_count = 0
    enlarged_gap_count = 0
    compared_count = 0
    max_gap_ratio = 0.0
    for lower, upper in zip(curved_source.layer_indices, curved_source.layer_indices[1:]):
        for role in ("R", "F"):
            lower_points = per_layer.get((lower, role), {})
            upper_points = per_layer.get((upper, role), {})
            flat_lower = flat_per_layer.get((lower, role), {})
            flat_upper = flat_per_layer.get((upper, role), {})
            for xy in lower_points.keys() & upper_points.keys() & flat_lower.keys() & flat_upper.keys():
                curved_gap = upper_points[xy] - lower_points[xy]
                flat_gap = flat_upper[xy] - flat_lower[xy]
                compared_count += 1
                if curved_gap < -limits.numeric_tolerance_mm:
                    inversion_count += 1
                if flat_gap > limits.numeric_tolerance_mm:
                    ratio = curved_gap / flat_gap
                    max_gap_ratio = max(max_gap_ratio, ratio)
                    if ratio > limits.max_layer_gap_ratio:
                        enlarged_gap_count += 1
    if not compared_count:
        return CheckResult(
            "层间几何",
            "warning",
            "相邻逻辑层没有可直接配对的同 XY 材料点，无法自动评估局部层间间隙",
            {"compared_point_count": 0},
        )
    if inversion_count:
        return CheckResult(
            "层间几何",
            "fail",
            "同 XY 的相邻逻辑层出现反向穿插",
            {"compared_point_count": compared_count, "inversion_count": inversion_count, "max_gap_ratio": max_gap_ratio},
        )
    if enlarged_gap_count:
        return CheckResult(
            "层间几何",
            "warning",
            "局部层间间隙超过平面基准的配置倍率，需要工艺确认",
            {
                "compared_point_count": compared_count,
                "max_gap_ratio": max_gap_ratio,
                "limit": limits.max_layer_gap_ratio,
                "exceeding_point_count": enlarged_gap_count,
            },
        )
    return CheckResult(
        "层间几何",
        "pass",
        "可配对的相邻逻辑层未出现反向穿插，局部间隙未超过配置倍率",
        {"compared_point_count": compared_count, "max_gap_ratio": max_gap_ratio, "limit": limits.max_layer_gap_ratio},
    )


def _check_sampling_density(
    curved_source: SourceNPZ, target: SurfaceTarget, limits: ValidatorLimits
) -> CheckResult:
    max_slope = 0.0
    max_segment_xy = 0.0
    max_dz = 0.0
    segment_count = 0
    for key in curved_source.path_keys:
        match = _PATH_KEY.match(key)
        assert match is not None
        if match.group(2) == "T":
            continue
        array = curved_source.arrays[key]
        for path in array:
            count = int(np.isfinite(path[:, 0]).sum())
            if count < 2:
                continue
            delta = np.diff(path[:count, :3], axis=0)
            xy_length = np.linalg.norm(delta[:, :2], axis=1)
            dz = np.abs(delta[:, 2])
            nonzero = xy_length > limits.numeric_tolerance_mm
            if np.any(nonzero):
                max_slope = max(max_slope, float(np.max(dz[nonzero] / xy_length[nonzero])))
                max_segment_xy = max(max_segment_xy, float(np.max(xy_length[nonzero])))
            max_dz = max(max_dz, float(np.max(dz)))
            segment_count += len(delta)
    shortest_wavelength = min(target.surface.wavelength_x_mm, target.surface.wavelength_y_mm)
    permitted_segment = shortest_wavelength * limits.max_segment_fraction_of_wavelength
    warnings: list[str] = []
    if max_slope > limits.max_slope:
        warnings.append("局部路径坡度超过配置阈值")
    if max_segment_xy > permitted_segment:
        warnings.append("XY 采样间距不足以表达最短波长")
    status: Status = "warning" if warnings else "pass"
    summary = "；".join(warnings) if warnings else "局部坡度与曲面采样间距均在配置阈值内"
    return CheckResult(
        "坡度、dz 与采样密度",
        status,
        summary,
        {
            "segment_count": segment_count,
            "max_path_slope": max_slope,
            "max_adjacent_dz_mm": max_dz,
            "max_xy_segment_mm": max_segment_xy,
            "max_slope_limit": limits.max_slope,
            "sampling_segment_limit_mm": permitted_segment,
            "shortest_wavelength_mm": shortest_wavelength,
        },
    )


def _check_extrusion(
    flat_source: SourceNPZ, curved_source: SourceNPZ, limits: ValidatorLimits
) -> CheckResult:
    flat_keys = {key for key in flat_source.arrays if _E_KEY.match(key)}
    curved_keys = {key for key in curved_source.arrays if _E_KEY.match(key)}
    if flat_keys != curved_keys:
        return CheckResult("E 弧长补偿", "fail", "flat 与 curved 的 _E 键集合不一致", {"flat_keys": sorted(flat_keys), "curved_keys": sorted(curved_keys)})
    if not flat_keys:
        return CheckResult("E 弧长补偿", "pass", "输入没有显式 _E 数组；验证器未虚构 E", {"replaced_array_count": 0})
    errors: list[str] = []
    ratios: list[float] = []
    positive_count = 0
    for key in sorted(flat_keys):
        path_key = key[:-2]
        flat_path = flat_source.arrays[path_key]
        curved_path = curved_source.arrays[path_key]
        flat_e = flat_source.arrays[key]
        curved_e = curved_source.arrays[key]
        if flat_e.shape != flat_path.shape[:2] or curved_e.shape != curved_path.shape[:2]:
            errors.append(f"{key} 形状不匹配")
            continue
        if not np.array_equal(np.isfinite(flat_e), np.isfinite(flat_path[..., 0])) or not np.array_equal(np.isfinite(curved_e), np.isfinite(curved_path[..., 0])):
            errors.append(f"{key} 的 NaN 填充不匹配")
            continue
        for path_index, path in enumerate(flat_path):
            count = int(np.isfinite(path[:, 0]).sum())
            if not count:
                continue
            if not math.isclose(float(flat_e[path_index, 0]), float(curved_e[path_index, 0]), abs_tol=limits.numeric_tolerance_mm):
                errors.append(f"{key} 路径 {path_index} 的首个累计 E 被改写")
            for point_index in range(1, count):
                flat_delta_e = float(flat_e[path_index, point_index] - flat_e[path_index, point_index - 1])
                curved_delta_e = float(curved_e[path_index, point_index] - curved_e[path_index, point_index - 1])
                if flat_delta_e <= 0:
                    expected = flat_delta_e
                    ratio = None
                else:
                    flat_length = float(np.linalg.norm(flat_path[path_index, point_index, :3] - flat_path[path_index, point_index - 1, :3]))
                    curved_length = float(np.linalg.norm(curved_path[path_index, point_index, :3] - curved_path[path_index, point_index - 1, :3]))
                    if flat_length <= limits.numeric_tolerance_mm:
                        expected = flat_delta_e
                        ratio = 1.0 if curved_length <= limits.numeric_tolerance_mm else math.inf
                    else:
                        ratio = curved_length / flat_length
                        expected = flat_delta_e * ratio
                    ratios.append(ratio)
                    positive_count += 1
                if not math.isclose(curved_delta_e, expected, rel_tol=limits.extrusion_relative_tolerance, abs_tol=limits.numeric_tolerance_mm):
                    errors.append(f"{key} 路径 {path_index} 点 {point_index} 的 ΔE 不符合弧长补偿")
    if errors:
        return CheckResult("E 弧长补偿", "fail", "显式 E 与对应三维弧长或路径网格不一致", {"errors": errors[:20], "error_count": len(errors)})
    finite_ratios = [ratio for ratio in ratios if math.isfinite(ratio)]
    return CheckResult(
        "E 弧长补偿",
        "pass",
        "已有 _E 均按正 ΔE 的三维弧长比重算，零/负 ΔE 保持原语义",
        {
            "replaced_array_count": len(flat_keys),
            "positive_segment_count": positive_count,
            "mean_length_ratio": float(np.mean(finite_ratios)) if finite_ratios else None,
            "max_length_ratio": float(np.max(finite_ratios)) if finite_ratios else None,
        },
    )


def _check_travel_risk(curved_source: SourceNPZ, limits: ValidatorLimits) -> CheckResult:
    printed_z = _printed_vertex_z(curved_source)
    travel_points = 0
    potential_collisions = 0
    for key in curved_source.path_keys:
        match = _PATH_KEY.match(key)
        assert match is not None
        if match.group(2) != "T":
            continue
        path = curved_source.arrays[key]
        valid = path[np.isfinite(path[..., 0])]
        travel_points += len(valid)
        for point in valid:
            deposited_z = printed_z.get(_xy_key(point[:2]))
            if deposited_z is not None and point[2] <= deposited_z + limits.numeric_tolerance_mm:
                potential_collisions += 1
    if not travel_points:
        return CheckResult("T 空走风险", "pass", "没有 T 空走路径需要检查", {"travel_point_count": 0})
    return CheckResult(
        "T 空走风险",
        "warning",
        "检测到 T 空走路径；尚无自动避障/安全抬升策略，必须人工确认",
        {
            "travel_point_count": travel_points,
            "vertex_level_potential_collision_count": potential_collisions,
            "method": "same-XY deposited-vertex clearance screen; not a complete swept-volume collision proof",
        },
    )


def _check_topology_evidence(
    flat_source: SourceNPZ, curved_source: SourceNPZ, limits: ValidatorLimits
) -> CheckResult:
    closed_paths = 0
    changed_closure = 0
    for key in flat_source.path_keys:
        flat = flat_source.arrays[key]
        curved = curved_source.arrays[key]
        for path_index, path in enumerate(flat):
            count = int(np.isfinite(path[:, 0]).sum())
            if count < 3 or not np.allclose(path[0, :2], path[count - 1, :2], atol=limits.numeric_tolerance_mm, rtol=0.0):
                continue
            closed_paths += 1
            if not np.allclose(curved[path_index, 0, :2], curved[path_index, count - 1, :2], atol=limits.numeric_tolerance_mm, rtol=0.0):
                changed_closure += 1
    if changed_closure:
        return CheckResult("XY 拓扑与边界", "fail", "原本闭合的路径在曲面后失去 XY 闭合", {"changed_closure_count": changed_closure})
    return CheckResult(
        "XY 拓扑与边界",
        "warning",
        "XY 未变且已识别闭合路径保持闭合；NPZ 未标注外轮廓和蜂窝孔洞，无法自动证明孔洞开口",
        {"verified_closed_path_count": closed_paths, "unassessed": ["outer_perimeter_role", "honeycomb_opening_semantics"]},
    )


def _source_hash(source: SourceNPZ) -> str:
    model = source.meta.get("source_model")
    return str(model.get("sha256", "")) if isinstance(model, dict) else ""


def _xy_key(point: np.ndarray) -> tuple[float, float]:
    return tuple(np.round(np.asarray(point, dtype=np.float64), _XY_DECIMALS))  # type: ignore[return-value]


def _layer_xy_z(source: SourceNPZ) -> dict[tuple[int, str], dict[tuple[float, float], float]]:
    result: dict[tuple[int, str], dict[tuple[float, float], list[float]]] = {}
    for key in source.path_keys:
        match = _PATH_KEY.match(key)
        assert match is not None
        role = match.group(2)
        if role == "T":
            continue
        bucket = result.setdefault((int(match.group(1)), role), {})
        points = source.arrays[key][np.isfinite(source.arrays[key][..., 0])]
        for point in points:
            bucket.setdefault(_xy_key(point[:2]), []).append(float(point[2]))
    return {key: {xy: float(np.median(values)) for xy, values in bucket.items()} for key, bucket in result.items()}


def _printed_vertex_z(source: SourceNPZ) -> dict[tuple[float, float], float]:
    result: dict[tuple[float, float], float] = {}
    for key in source.path_keys:
        match = _PATH_KEY.match(key)
        assert match is not None
        if match.group(2) == "T":
            continue
        points = source.arrays[key][np.isfinite(source.arrays[key][..., 0])]
        for point in points:
            xy = _xy_key(point[:2])
            result[xy] = max(result.get(xy, -math.inf), float(point[2]))
    return result
