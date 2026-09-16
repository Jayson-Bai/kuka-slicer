"""
时间参数化器（七阶 S 曲线 + 球面插值）.

- 几何由 bspline_approximation 生成的 GlobalCurveCommand 提供，只负责沿弧长采样。
- 目标匀速暂写死为 10 mm/s，入口/出口速度、加速度、jerk 全 0，对称加速/减速时间固定 2 s。
- 若弧长太短无法形成匀速段，则自动降低峰值速度（等价于把位移缩放进同样的时间，保证不超速且 jerk 连续）。
- 挤出量 E 按 4 ms 采样的路程比例分配，保持绝对挤出。
"""

from dataclasses import dataclass
from typing import List, Optional
import bisect
import math
import time

from .types import Position, GlobalCurveCommand, validate_source_e_profile
from .kuka_orientation import (
    kuka_abc_to_quaternion,
    quaternion_from_rotation_vector,
    quaternion_inverse,
    quaternion_multiply,
    rotation_vector_from_quaternion,
    quaternion_slerp,
    quaternion_to_kuka_abc,
)


_MAX_ORIENTATION_STEP_DEG = 0.1
# A density-6 fit can contain more than one thousand control points.  Ten
# uniform parameter samples per point were insufficient around tight fitted
# turns: the arc map understated local Cartesian distance, so a nominal
# 20 mm/s sample could exceed the RSI step limit.  This refines the *internal*
# arc-length estimate only; it neither inserts trajectory rows nor changes the
# user-selected feedrate/time profile.
_ARC_LENGTH_MAP_SAMPLES_PER_CONTROL_POINT = 24
_ARC_LENGTH_MAP_MIN_SAMPLES = 800
_TRAVEL_WAYPOINT_MIN_RAMP_S = 0.08


# -------------------------- 基础工具 --------------------------

@dataclass
class InterpolatedPoint:
    t: float
    pos: Position
    e: float               # 绝对挤出量
    extrude_speed: float   # dE/dt (mm/s) 供调试
    feedrate_mm_min: float
    cmd_type: str
    line: Optional[int]
    raw: Optional[str]


# -------------------------- B 样条评估（仅用于已生成的控制点） --------------------------

def _make_open_uniform_knots(n_ctrl: int, degree: int = 3) -> List[float]:
    knots: List[float] = []
    for i in range(n_ctrl + degree + 1):
        if i <= degree:
            knots.append(0.0)
        elif i >= n_ctrl:
            knots.append(n_ctrl - degree)
        else:
            knots.append(i - degree)
    return knots


def _find_span(u: float, knots: List[float], degree: int, n_ctrl: int) -> int:
    if abs(u - knots[n_ctrl]) < 1e-9:
        return n_ctrl - 1
    low = degree
    high = n_ctrl
    mid = (low + high) // 2
    while not (knots[mid] <= u < knots[mid + 1]):
        if u < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _find_span_monotonic(
    u: float,
    knots: List[float],
    degree: int,
    n_ctrl: int,
        start_span: int) -> int:
    if abs(u - knots[n_ctrl]) < 1e-9:
        return n_ctrl - 1
    span = max(degree, min(start_span, n_ctrl - 1))
    while span + 1 < n_ctrl and u >= knots[span + 1]:
        span += 1
    while span > degree and u < knots[span]:
        span -= 1
    return span


def _basis_funs(span: int, u: float, degree: int, knots: List[float]) -> List[float]:
    if degree == 3:
        return _basis_funs_cubic(span, u, knots)

    values = [0.0] * (degree + 1)
    values[0] = 1.0
    left = [0.0] * (degree + 1)
    right = [0.0] * (degree + 1)

    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = 0.0 if abs(denom) < 1e-12 else values[r] / denom
            values[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        values[j] = saved
    return values


def _basis_funs_cubic(span: int, u: float, knots: List[float]) -> List[float]:
    """Unrolled cubic Cox-de Boor basis with the generic operation order."""

    values = [1.0, 0.0, 0.0, 0.0]
    left1 = u - knots[span]
    right1 = knots[span + 1] - u

    denominator = right1 + left1
    temporary = 0.0 if abs(denominator) < 1e-12 else values[0] / denominator
    values[0] = 0.0 + right1 * temporary
    saved = left1 * temporary
    values[1] = saved

    left2 = u - knots[span - 1]
    right2 = knots[span + 2] - u
    saved = 0.0
    denominator = right1 + left2
    temporary = 0.0 if abs(denominator) < 1e-12 else values[0] / denominator
    values[0] = saved + right1 * temporary
    saved = left2 * temporary
    denominator = right2 + left1
    temporary = 0.0 if abs(denominator) < 1e-12 else values[1] / denominator
    values[1] = saved + right2 * temporary
    saved = left1 * temporary
    values[2] = saved

    left3 = u - knots[span - 2]
    right3 = knots[span + 3] - u
    saved = 0.0
    denominator = right1 + left3
    temporary = 0.0 if abs(denominator) < 1e-12 else values[0] / denominator
    values[0] = saved + right1 * temporary
    saved = left3 * temporary
    denominator = right2 + left2
    temporary = 0.0 if abs(denominator) < 1e-12 else values[1] / denominator
    values[1] = saved + right2 * temporary
    saved = left2 * temporary
    denominator = right3 + left1
    temporary = 0.0 if abs(denominator) < 1e-12 else values[2] / denominator
    values[2] = saved + right3 * temporary
    saved = left1 * temporary
    values[3] = saved
    return values


def _split_ctrl_components(ctrl: List[Position]):
    return (
        [p.x for p in ctrl],
        [p.y for p in ctrl],
        [p.z for p in ctrl],
        [p.a for p in ctrl],
        [p.b for p in ctrl],
        [p.c for p in ctrl],
    )


def _eval_bspline_point(
    u: float,
    degree: int,
    knots: List[float],
    ctrl_xyzabc,
    n_ctrl: int,
    start_span: int,
):
    span = _find_span_monotonic(u, knots, degree, n_ctrl, start_span)
    coeffs = _basis_funs(span, u, degree, knots)
    start = span - degree
    xs, ys, zs, aa, bb, cc = ctrl_xyzabc

    x = y = z = a = b = c = 0.0
    for offset, coeff in enumerate(coeffs):
        idx = start + offset
        x += coeff * xs[idx]
        y += coeff * ys[idx]
        z += coeff * zs[idx]
        a += coeff * aa[idx]
        b += coeff * bb[idx]
        c += coeff * cc[idx]

    return Position(x=x, y=y, z=z, a=a, b=b, c=c), span


def _sample_kuka_orientation(
    normalized_u: float, parameters, quaternions, tangents=None
):
    """Evaluate a local, interpolating KUKA quaternion curve."""

    if normalized_u <= parameters[0]:
        return quaternions[0]
    if normalized_u >= parameters[-1]:
        return quaternions[-1]
    right = bisect.bisect_right(parameters, normalized_u)
    left = max(0, right - 1)
    right = min(len(parameters) - 1, right)
    span = parameters[right] - parameters[left]
    local = 0.0 if span <= 1e-12 else (normalized_u - parameters[left]) / span
    if not tangents:
        return quaternion_slerp(quaternions[left], quaternions[right], local)
    return _quaternion_squad(
        quaternions[left], quaternions[right],
        tangents[left], tangents[right], local,
    )


def _quaternion_squad(start, end, start_tangent, end_tangent, ratio):
    endpoints = quaternion_slerp(start, end, ratio)
    controls = quaternion_slerp(start_tangent, end_tangent, ratio)
    return quaternion_slerp(
        endpoints, controls, 2.0 * ratio * (1.0 - ratio)
    )


def _build_orientation_tangents(quaternions):
    """Build local SQUAD controls so angular velocity does not jump at knots."""
    if len(quaternions) < 3:
        return list(quaternions)
    tangents = [quaternions[0]]
    for previous, current, following in zip(
        quaternions, quaternions[1:], quaternions[2:]
    ):
        inverse = quaternion_inverse(current)
        before = rotation_vector_from_quaternion(
            quaternion_multiply(inverse, previous)
        )
        after = rotation_vector_from_quaternion(
            quaternion_multiply(inverse, following)
        )
        tangent_delta = tuple(
            -0.25 * (left + right) for left, right in zip(before, after)
        )
        tangents.append(
            quaternion_multiply(
                current, quaternion_from_rotation_vector(tangent_delta)
            )
        )
    tangents.append(quaternions[-1])
    return tangents


def _orientation_refinement_parameters(parameters):
    if not parameters:
        return None
    refined = {float(value) for value in parameters}
    for left, right in zip(parameters, parameters[1:]):
        span = float(right) - float(left)
        for fraction in (0.25, 0.5, 0.75):
            refined.add(float(left) + span * fraction)
    return sorted(refined)


def _sample_source_e_profile(normalized_u: float, parameters, values) -> float:
    """Evaluate the opt-in source E profile with monotonic linear segments."""
    if normalized_u <= parameters[0]:
        return values[0]
    if normalized_u >= parameters[-1]:
        return values[-1]
    right = bisect.bisect_right(parameters, normalized_u)
    left = max(0, right - 1)
    right = min(len(parameters) - 1, right)
    span = parameters[right] - parameters[left]
    local = 0.0 if span <= 1e-12 else (normalized_u - parameters[left]) / span
    return values[left] + (values[right] - values[left]) * local


def _build_arc_length_map(
    ctrl: List[Position],
    degree: int = 3,
    samples: int = 400,
    extra_normalized_parameters=None,
):
    knots = _make_open_uniform_knots(len(ctrl), degree)
    u_min = knots[degree]
    u_max = knots[len(ctrl)]
    ctrl_xyzabc = _split_ctrl_components(ctrl)
    n_ctrl = len(ctrl)

    u_list: List[float] = []
    len_list: List[float] = []

    normalized_parameters = {i / samples for i in range(samples + 1)}
    if extra_normalized_parameters:
        normalized_parameters.update(
            max(0.0, min(1.0, float(value)))
            for value in extra_normalized_parameters
        )

    prev_pos, span = _eval_bspline_point(u_min, degree, knots, ctrl_xyzabc, n_ctrl, degree)
    u_list.append(u_min)
    len_list.append(0.0)

    current_len = 0.0
    for normalized_u in sorted(normalized_parameters)[1:]:
        u = u_min + (u_max - u_min) * normalized_u
        curr_pos, span = _eval_bspline_point(u, degree, knots, ctrl_xyzabc, n_ctrl, span)
        dist = math.sqrt(
            (curr_pos.x - prev_pos.x) ** 2
            + (curr_pos.y - prev_pos.y) ** 2
            + (curr_pos.z - prev_pos.z) ** 2
        )
        current_len += dist
        u_list.append(u)
        len_list.append(current_len)
        prev_pos = curr_pos

    total_length = len_list[-1]
    return u_list, len_list, total_length, knots


def _arc_length_map_sample_count(control_point_count: int) -> int:
    """Return a conservative uniform budget for a fitted spline's arc map."""

    return max(
        _ARC_LENGTH_MAP_MIN_SAMPLES,
        max(2, int(control_point_count)) * _ARC_LENGTH_MAP_SAMPLES_PER_CONTROL_POINT,
    )


def _quaternion_angle_deg(left, right) -> float:
    dot = min(1.0, max(-1.0, abs(sum(a * b for a, b in zip(left, right)))))
    return math.degrees(2.0 * math.acos(dot))


def _validate_max_angular_speed(max_angular_speed_deg_s: Optional[float]) -> Optional[float]:
    if max_angular_speed_deg_s is None:
        return None
    value = float(max_angular_speed_deg_s)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("max_angular_speed_deg_s must be finite and > 0")
    return value


def _pose_timing_increment(
    distance_mm: float,
    angle_deg: float,
    target_velocity: float,
    max_angular_speed_deg_s: Optional[float],
) -> float:
    """Return a smooth combined XYZ/orientation timing metric in millimetres."""
    if max_angular_speed_deg_s is None or angle_deg <= 0.0:
        return float(distance_mm)
    rotation_equivalent_mm = (
        float(target_velocity) * float(angle_deg) / max_angular_speed_deg_s
    )
    return math.hypot(float(distance_mm), rotation_equivalent_mm)


def _smoothed_pose_timing_increments(
    distances_mm,
    angles_deg,
    target_velocity: float,
    max_angular_speed_deg_s: Optional[float],
    t_acc: float,
    t_dec: float,
):
    """Build a look-ahead timing metric without local feedrate jumps.

    The raw orientation limit can rise sharply where surface curvature tends
    to zero.  A forward/backward reachable-speed envelope prevents Core from
    immediately accelerating back to the requested XYZ feedrate at that
    point.  The acceleration scale is derived from the curve's existing
    seventh-order ramp times, so no second UI motion model is introduced.
    """
    distances = [max(0.0, float(value)) for value in distances_mm]
    angles = [max(0.0, float(value)) for value in angles_deg]
    if len(distances) != len(angles):
        raise ValueError("pose timing distances and angles must have equal lengths")
    if max_angular_speed_deg_s is None:
        return distances

    caps = []
    for distance, angle in zip(distances, angles):
        if distance <= 1e-12:
            caps.append(float(target_velocity))
            continue
        equivalent = target_velocity * angle / max_angular_speed_deg_s
        caps.append(
            target_velocity * distance / math.hypot(distance, equivalent)
        )

    if t_dec > 0.0:
        deceleration = target_velocity / t_dec
        for index in range(len(caps) - 2, -1, -1):
            reachable = math.sqrt(
                caps[index + 1] ** 2
                + 2.0 * deceleration * distances[index]
            )
            caps[index] = min(caps[index], reachable)
    if t_acc > 0.0:
        acceleration = target_velocity / t_acc
        for index in range(1, len(caps)):
            reachable = math.sqrt(
                caps[index - 1] ** 2
                + 2.0 * acceleration * distances[index - 1]
            )
            caps[index] = min(caps[index], reachable)

    increments = []
    for distance, angle, cap in zip(distances, angles, caps):
        required_time = 0.0
        if distance > 1e-12:
            required_time = distance / max(cap, 1e-12)
        if angle > 1e-12:
            required_time = max(
                required_time, angle / max_angular_speed_deg_s
            )
        increments.append(target_velocity * required_time)
    return increments


def _interpolate_mapped_value(
    u: float,
    u_list: List[float],
    values: List[float],
    index: int,
) -> float:
    index = max(0, min(index, len(u_list) - 2))
    u0 = u_list[index]
    u1 = u_list[index + 1]
    if abs(u1 - u0) <= 1e-12:
        return float(values[index])
    ratio = max(0.0, min(1.0, (u - u0) / (u1 - u0)))
    return float(values[index]) + (float(values[index + 1]) - float(values[index])) * ratio


def _is_linear_fallback_curve(curve: GlobalCurveCommand) -> bool:
    if curve.raw != "fallback_linear":
        return False
    if len(curve.control_points) != 3:
        return False
    p0 = curve.control_points[0]
    return all(
        abs(cp.x - p0.x) < 1e-12
        and abs(cp.y - p0.y) < 1e-12
        and abs(cp.z - p0.z) < 1e-12
        and abs(cp.a - p0.a) < 1e-12
        and abs(cp.b - p0.b) < 1e-12
        and abs(cp.c - p0.c) < 1e-12
        for cp in curve.control_points[1:]
    )


def _lookup_u_from_s(
        s_norm: float,
        u_list: List[float],
        len_list: List[float],
        total_length: float) -> float:
    """给定归一化弧长 s_norm (0~1)，返回对应的 B 样条参数 u."""
    target_len = s_norm * total_length
    if target_len <= 1e-9:
        return u_list[0]
    if target_len >= total_length - 1e-9:
        return u_list[-1]

    idx = bisect.bisect_right(len_list, target_len)
    if idx == 0:
        return u_list[0]
    if idx >= len(len_list):
        return u_list[-1]

    l0 = len_list[idx - 1]
    l1 = len_list[idx]
    u0 = u_list[idx - 1]
    u1 = u_list[idx]
    if abs(l1 - l0) < 1e-12:
        return u0
    ratio = (target_len - l0) / (l1 - l0)
    return u0 + ratio * (u1 - u0)


def _lookup_u_from_target_len_monotonic(
    target_len: float,
    u_list: List[float],
    len_list: List[float],
    total_length: float,
    start_idx: int,
):
    """单调递增弧长的快速查找，返回 (u, idx)."""
    if target_len <= 1e-9:
        return u_list[0], 0
    if target_len >= total_length - 1e-9:
        return u_list[-1], max(0, len(len_list) - 2)

    idx = max(0, min(start_idx, len(len_list) - 2))
    while idx + 1 < len(len_list) and len_list[idx + 1] < target_len:
        idx += 1
    while idx > 0 and len_list[idx] > target_len:
        idx -= 1

    l0 = len_list[idx]
    l1 = len_list[idx + 1]
    u0 = u_list[idx]
    u1 = u_list[idx + 1]
    if abs(l1 - l0) < 1e-12:
        return u0, idx
    ratio = (target_len - l0) / (l1 - l0)
    return u0 + ratio * (u1 - u0), idx


# -------------------------- 七阶 S 曲线剖面 --------------------------

def _sept_poly_base(tau: float) -> float:
    """归一化 0->1 七阶位置曲线，v/a/jerk 在两端为 0."""
    return (35.0 * tau**4) - (84.0 * tau**5) + (70.0 * tau**6) - (20.0 * tau**7)


def _three_stage_sept_poly(t: float, total: float, t_acc: float, t_dec: float) -> float:
    """
    三段式七阶 S 曲线：加速-匀速-减速，输出归一化路程 s(t)∈[0,1].

    若无匀速段，则退化为对称 S 曲线（总时长 t_acc+t_dec）。
    """
    if t <= 0.0:
        return 0.0
    if t >= total:
        return 1.0

    t_flat = total - t_acc - t_dec
    if t_flat < 0:
        # 时间不足形成匀速段，保持对称 S 曲线
        return t / total

    k = 2.1875  # 基函数导数峰值
    denom = t_flat + (t_acc + t_dec) / k
    if denom <= 0:
        return t / total
    v_flat = 1.0 / denom  # 归一化匀速速度

    if t < t_acc:
        tau = 0.5 * (t / t_acc)
        return (2.0 * v_flat * t_acc / k) * _sept_poly_base(tau)
    elif t < (total - t_dec):
        s_acc = v_flat * t_acc / k
        return s_acc + v_flat * (t - t_acc)
    else:
        t_rem = total - t
        tau = 0.5 * (t_rem / t_dec)
        s_rem = (2.0 * v_flat * t_dec / k) * _sept_poly_base(tau)
        return 1.0 - s_rem


def _compute_time_profile(length: float, target_v: float, t_acc: float, t_dec: float):
    """
    根据弧长和目标匀速计算总时间与匀速时间.

    若长度过短，则匀速段为 0，总时间=t_acc+t_dec。
    """
    if length <= 0.0 or target_v <= 0.0:
        return 0.0, 0.0
    k = 2.1875
    nominal_time = length / target_v
    effective_acc_dec = (t_acc + t_dec) / k
    t_flat = nominal_time - effective_acc_dec
    if t_flat < 0:
        t_flat = 0.0
    total_time = t_acc + t_flat + t_dec
    return total_time, t_flat


def _orientation_only_samples(
    curve: GlobalCurveCommand,
    ctrl: List[Position],
    dt: float,
    max_angular_speed_deg_s: Optional[float],
):
    """Sample an in-place KUKA rotation with the seventh-order S curve.

    Core previously collapsed a zero-XYZ move to a single row even when its
    ABC changed.  That makes a layer-dependent surface normal an RSI jump.
    """

    start = curve.start_pos
    end = ctrl[-1]
    start_q = kuka_abc_to_quaternion(start.a, start.b, start.c)
    end_q = kuka_abc_to_quaternion(end.a, end.b, end.c)
    angle_deg = _quaternion_angle_deg(start_q, end_q)
    if angle_deg <= 1e-9:
        yield InterpolatedPoint(
            t=0.0, pos=start, e=curve.e_val, extrude_speed=0.0,
            feedrate_mm_min=curve.feedrate, cmd_type=curve.type,
            line=curve.line, raw=curve.raw,
        )
        return

    # The largest derivative of the seventh-order base curve is 2.1875.
    # Account for it so every emitted RSI frame remains below the angle step.
    max_step_deg = (
        _MAX_ORIENTATION_STEP_DEG
        if max_angular_speed_deg_s is None
        else max_angular_speed_deg_s * dt
    )
    steps = max(1, int(math.ceil(angle_deg * 2.1875 / max_step_deg)))
    start_e = curve.e_val - curve.delta_e
    previous_e = start_e
    previous_abc = (start.a, start.b, start.c)
    for index in range(steps + 1):
        ratio = _sept_poly_base(index / steps)
        q = quaternion_slerp(start_q, end_q, ratio)
        a, b, c = quaternion_to_kuka_abc(q, near_deg=previous_abc)
        current_e = start_e + curve.delta_e * ratio
        delta_e = current_e - previous_e
        yield InterpolatedPoint(
            t=index * dt,
            pos=Position(start.x, start.y, start.z, a, b, c),
            e=current_e,
            extrude_speed=delta_e / dt if dt > 0.0 else 0.0,
            feedrate_mm_min=0.0,
            cmd_type=curve.type,
            line=curve.line,
            raw=curve.raw,
        )
        previous_e = current_e
        previous_abc = (a, b, c)


def _travel_waypoint_ramp_s(
    ramp_limit_s: float,
    timing_length: float,
    target_velocity: float,
) -> float:
    """Return a bounded per-edge ramp for an exact waypoint stop."""
    if ramp_limit_s <= 0.0 or target_velocity <= 0.0:
        return 0.0
    proportional_ramp = 0.5 * timing_length / target_velocity
    return min(
        float(ramp_limit_s),
        max(_TRAVEL_WAYPOINT_MIN_RAMP_S, proportional_ramp),
    )


def _sample_travel_polyline_with_waypoint_stops(
    curve: GlobalCurveCommand,
    points: List[Position],
    seg_lengths: List[float],
    seg_angles: List[float],
    e_profile: Optional[List[float]],
    *,
    dt: float,
    target_velocity: float,
    t_acc: float,
    t_dec: float,
    max_angular_speed_deg_s: Optional[float],
):
    """Sample every preserved travel edge with a zero-speed waypoint stop.

    Each edge gets an independent seventh-order start/stop profile. Repeating
    the shared waypoint as the next edge's first frame creates one explicit
    stationary RSI frame, so XYZ and KUKA ABC both restart from zero discrete
    velocity without changing the obstacle-avoiding polyline geometry.
    """
    total_length = sum(seg_lengths)
    curve_start_e = curve.e_val - curve.delta_e
    current_e = curve_start_e
    cumulative_length = 0.0
    elapsed_t = 0.0
    previous_abc = (points[0].a, points[0].b, points[0].c)

    for seg_idx, (start, end, seg_len, seg_angle) in enumerate(
        zip(points, points[1:], seg_lengths, seg_angles)
    ):
        timing_length = _pose_timing_increment(
            seg_len,
            seg_angle,
            target_velocity,
            max_angular_speed_deg_s,
        )
        if timing_length <= 1e-12:
            cumulative_length += seg_len
            continue

        seg_t_acc = _travel_waypoint_ramp_s(
            t_acc, timing_length, target_velocity
        )
        seg_t_dec = _travel_waypoint_ramp_s(
            t_dec, timing_length, target_velocity
        )
        total_time, _ = _compute_time_profile(
            timing_length, target_velocity, seg_t_acc, seg_t_dec
        )
        if total_time <= 0.0:
            cumulative_length += seg_len
            continue
        num_steps = int(math.ceil(total_time / dt))
        corrected_total_time = num_steps * dt

        if e_profile is None:
            segment_start_e = (
                curve_start_e
                if total_length <= 1e-12
                else curve_start_e
                + curve.delta_e * (cumulative_length / total_length)
            )
            segment_end_e = (
                curve.e_val
                if total_length <= 1e-12
                else curve_start_e
                + curve.delta_e * ((cumulative_length + seg_len) / total_length)
            )
        else:
            segment_start_e = e_profile[seg_idx]
            segment_end_e = e_profile[seg_idx + 1]

        start_q = kuka_abc_to_quaternion(start.a, start.b, start.c)
        end_q = kuka_abc_to_quaternion(end.a, end.b, end.c)
        previous_local = 0.0

        for step in range(num_steps + 1):
            local_t = step * dt
            local = _three_stage_sept_poly(
                local_t, corrected_total_time, seg_t_acc, seg_t_dec
            )
            local = max(0.0, min(1.0, local))
            if step == num_steps:
                local = 1.0

            pos = Position(
                x=start.x + (end.x - start.x) * local,
                y=start.y + (end.y - start.y) * local,
                z=start.z + (end.z - start.z) * local,
                a=start.a,
                b=start.b,
                c=start.c,
            )
            q = quaternion_slerp(start_q, end_q, local)
            pos.a, pos.b, pos.c = quaternion_to_kuka_abc(
                q, near_deg=previous_abc
            )
            previous_abc = (pos.a, pos.b, pos.c)

            target_e = segment_start_e + (
                segment_end_e - segment_start_e
            ) * local
            delta_e = target_e - current_e
            current_e = target_e
            delta_s = (local - previous_local) * seg_len
            previous_local = local

            yield InterpolatedPoint(
                t=elapsed_t + local_t,
                pos=pos,
                e=current_e,
                extrude_speed=delta_e / dt if dt > 0.0 else 0.0,
                feedrate_mm_min=(delta_s / dt * 60.0) if dt > 0.0 else 0.0,
                cmd_type=curve.type,
                line=curve.line,
                raw=curve.raw,
            )

        cumulative_length += seg_len
        # The next edge starts with the same pose on the following RSI frame.
        elapsed_t += corrected_total_time + dt


# -------------------------- 采样主逻辑 --------------------------

def sample_global_curve_iter(
    curve: GlobalCurveCommand,
    dt: float = 0.004,
    target_velocity: float = 10.0,  # mm/s
    t_acc: float = 2.0,
    t_dec: float = 2.0,
    max_angular_speed_deg_s: Optional[float] = None,
    profile: Optional[dict] = None,
):
    """
    对一条全局 B 样条进行时间参数化并采样（生成器）.

    - 入口/出口 v/a/jerk 均为 0，对称加/减速时间固定。
    - 匀速段无法满足时自动退化为对称 S 曲线（速度整体下降，不超速）。
    - 挤出按弧长比例分配，保持绝对挤出量不变。
    """
    if curve is None:
        return

    max_angular_speed_deg_s = _validate_max_angular_speed(
        max_angular_speed_deg_s
    )

    ctrl = [curve.start_pos] + curve.control_points
    degree = 3
    n_ctrl = len(ctrl)
    ctrl_xyzabc = _split_ctrl_components(ctrl)

    if profile is not None:
        profile.setdefault("sample_arc_map_s", 0.0)
        profile.setdefault("sample_lookup_s", 0.0)
        profile.setdefault("sample_deboor_s", 0.0)
        profile.setdefault("sample_pose_s", 0.0)
        profile.setdefault("sample_extrude_s", 0.0)

    if all(
        abs(point.x - curve.start_pos.x) <= 1e-9
        and abs(point.y - curve.start_pos.y) <= 1e-9
        and abs(point.z - curve.start_pos.z) <= 1e-9
        for point in ctrl[1:]
    ):
        yield from _orientation_only_samples(
            curve, ctrl, dt, max_angular_speed_deg_s
        )
        return

    if (curve.cmd or "").upper() == "POLYLINE":
        points = [curve.start_pos] + list(curve.control_points)
        e_profile = _polyline_e_profile(curve, len(points))
        seg_lengths = []
        seg_angles = []
        total_length = 0.0
        for start, end in zip(points, points[1:]):
            length = math.sqrt(
                (end.x - start.x) ** 2
                + (end.y - start.y) ** 2
                + (end.z - start.z) ** 2
            )
            seg_lengths.append(length)
            seg_angles.append(
                _quaternion_angle_deg(
                    kuka_abc_to_quaternion(start.a, start.b, start.c),
                    kuka_abc_to_quaternion(end.a, end.b, end.c),
                )
            )
            total_length += length
        if total_length <= 1e-9:
            yield InterpolatedPoint(
                t=0.0,
                pos=curve.start_pos,
                e=curve.e_val,
                extrude_speed=0.0,
                feedrate_mm_min=curve.feedrate,
                cmd_type=curve.type,
                line=curve.line,
                raw=curve.raw,
            )
            return

        if curve.type == "TRAVEL" and len(points) > 2:
            yield from _sample_travel_polyline_with_waypoint_stops(
                curve,
                points,
                seg_lengths,
                seg_angles,
                e_profile,
                dt=dt,
                target_velocity=target_velocity,
                t_acc=t_acc,
                t_dec=t_dec,
                max_angular_speed_deg_s=max_angular_speed_deg_s,
            )
            return

        timing_seg_lengths = _smoothed_pose_timing_increments(
            seg_lengths,
            seg_angles,
            target_velocity,
            max_angular_speed_deg_s,
            t_acc,
            t_dec,
        )
        total_timing_length = sum(timing_seg_lengths)
        total_time, _ = _compute_time_profile(
            total_timing_length, target_velocity, t_acc, t_dec
        )
        if total_time <= 0.0:
            return
        num_steps = int(math.ceil(total_time / dt))
        corrected_total_time = num_steps * dt
        start_e = curve.e_val - curve.delta_e
        current_e = start_e
        prev_s = 0.0
        seg_idx = 0
        seg_start_s = 0.0
        seg_start_timing = 0.0
        previous_abc = (points[0].a, points[0].b, points[0].c)

        for i in range(num_steps + 1):
            t = i * dt
            s_norm = _three_stage_sept_poly(t, corrected_total_time, t_acc, t_dec)
            s_norm_clamped = max(0.0, min(1.0, s_norm))
            if i == num_steps:
                s_norm_clamped = 1.0

            curr_timing = s_norm_clamped * total_timing_length
            while (
                seg_idx < len(timing_seg_lengths) - 1
                and curr_timing > seg_start_timing + timing_seg_lengths[seg_idx]
            ):
                seg_start_s += seg_lengths[seg_idx]
                seg_start_timing += timing_seg_lengths[seg_idx]
                seg_idx += 1

            seg_len = seg_lengths[seg_idx]
            timing_seg_len = timing_seg_lengths[seg_idx]
            local = (
                0.0
                if timing_seg_len <= 1e-12
                else (curr_timing - seg_start_timing) / timing_seg_len
            )
            local = max(0.0, min(1.0, local))
            curr_s = seg_start_s + local * seg_len
            start = points[seg_idx]
            end = points[seg_idx + 1]
            pos = Position(
                x=start.x + (end.x - start.x) * local,
                y=start.y + (end.y - start.y) * local,
                z=start.z + (end.z - start.z) * local,
                a=start.a,
                b=start.b,
                c=start.c,
            )
            q = quaternion_slerp(
                kuka_abc_to_quaternion(start.a, start.b, start.c),
                kuka_abc_to_quaternion(end.a, end.b, end.c),
                local,
            )
            pos.a, pos.b, pos.c = quaternion_to_kuka_abc(q, near_deg=previous_abc)
            previous_abc = (pos.a, pos.b, pos.c)

            delta_s = curr_s - prev_s
            if e_profile is None:
                delta_e = curve.delta_e * (delta_s / total_length)
                current_e += delta_e
            else:
                target_e = e_profile[seg_idx] + (
                    e_profile[seg_idx + 1] - e_profile[seg_idx]
                ) * local
                delta_e = target_e - current_e
                current_e = target_e
            prev_s = curr_s
            feed_mm_s = delta_s / dt if dt > 0 else 0.0
            feed_mm_min = feed_mm_s * 60.0
            extrude_speed = delta_e / dt if dt > 0 else 0.0

            yield InterpolatedPoint(
                t=t,
                pos=pos,
                e=current_e,
                extrude_speed=extrude_speed,
                feedrate_mm_min=feed_mm_min,
                cmd_type=curve.type,
                line=curve.line,
                raw=curve.raw,
            )
        return

    if _is_linear_fallback_curve(curve):
        end_pos = curve.control_points[-1]
        dx = end_pos.x - curve.start_pos.x
        dy = end_pos.y - curve.start_pos.y
        dz = end_pos.z - curve.start_pos.z
        total_length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if total_length <= 1e-9:
            yield InterpolatedPoint(
                t=0.0,
                pos=curve.start_pos,
                e=curve.e_val,
                extrude_speed=0.0,
                feedrate_mm_min=curve.feedrate,
                cmd_type=curve.type,
                line=curve.line,
                raw=curve.raw,
            )
            return

        angle_deg = _quaternion_angle_deg(
            kuka_abc_to_quaternion(
                curve.start_pos.a, curve.start_pos.b, curve.start_pos.c
            ),
            kuka_abc_to_quaternion(end_pos.a, end_pos.b, end_pos.c),
        )
        timing_length = _pose_timing_increment(
            total_length,
            angle_deg,
            target_velocity,
            max_angular_speed_deg_s,
        )
        total_time, _ = _compute_time_profile(
            timing_length, target_velocity, t_acc, t_dec
        )
        if total_time <= 0.0:
            return
        num_steps = int(math.ceil(total_time / dt))
        corrected_total_time = num_steps * dt
        start_e = curve.e_val - curve.delta_e
        current_e = start_e
        prev_s = 0.0

        same_orientation = (
            abs(curve.start_pos.a - end_pos.a) < 1e-9
            and abs(curve.start_pos.b - end_pos.b) < 1e-9
            and abs(curve.start_pos.c - end_pos.c) < 1e-9
        )

        start_q = end_q = None
        previous_abc = (curve.start_pos.a, curve.start_pos.b, curve.start_pos.c)
        if not same_orientation:
            start_q = kuka_abc_to_quaternion(
                curve.start_pos.a, curve.start_pos.b, curve.start_pos.c,
            )
            end_q = kuka_abc_to_quaternion(end_pos.a, end_pos.b, end_pos.c)

        for i in range(num_steps + 1):
            t = i * dt
            s_norm = _three_stage_sept_poly(t, corrected_total_time, t_acc, t_dec)
            s_norm_clamped = max(0.0, min(1.0, s_norm))
            if i == num_steps:
                s_norm_clamped = 1.0

            pos = Position(
                x=curve.start_pos.x + dx * s_norm_clamped,
                y=curve.start_pos.y + dy * s_norm_clamped,
                z=curve.start_pos.z + dz * s_norm_clamped,
                a=curve.start_pos.a,
                b=curve.start_pos.b,
                c=curve.start_pos.c,
            )

            if same_orientation:
                pos.a = curve.start_pos.a
                pos.b = curve.start_pos.b
                pos.c = curve.start_pos.c
            else:
                q = quaternion_slerp(start_q, end_q, s_norm_clamped)
                pos.a, pos.b, pos.c = quaternion_to_kuka_abc(q, near_deg=previous_abc)
            previous_abc = (pos.a, pos.b, pos.c)

            curr_s = s_norm_clamped * total_length
            delta_s = curr_s - prev_s
            delta_e = curve.delta_e * (delta_s / total_length)
            current_e += delta_e
            prev_s = curr_s
            feed_mm_s = delta_s / dt if dt > 0 else 0.0
            feed_mm_min = feed_mm_s * 60.0
            extrude_speed = delta_e / dt if dt > 0 else 0.0

            yield InterpolatedPoint(
                t=t,
                pos=pos,
                e=current_e,
                extrude_speed=extrude_speed,
                feedrate_mm_min=feed_mm_min,
                cmd_type=curve.type,
                line=curve.line,
                raw=curve.raw,
            )
        return

    # 构建弧长映射
    t0 = time.perf_counter()
    u_list, len_list, total_length, knots = _build_arc_length_map(
        ctrl,
        degree=degree,
        samples=_arc_length_map_sample_count(len(ctrl)),
        extra_normalized_parameters=_orientation_refinement_parameters(
            curve.orientation_parameters
        ),
    )
    if profile is not None:
        profile["sample_arc_map_s"] += time.perf_counter() - t0
    if total_length <= 1e-9:
        # 退化：零长度，直接返回终点
        yield InterpolatedPoint(
            t=0.0,
            pos=curve.start_pos,
            e=curve.e_val,
            extrude_speed=0.0,
            feedrate_mm_min=curve.feedrate,
            cmd_type=curve.type,
            line=curve.line,
            raw=curve.raw,
        )
        return

    # 时间规划：在 XYZ 弧长之外加入四元数角距离。高曲率区域因此自动
    # 获得更多时间，同时仍由同一条七阶进度曲线驱动 XYZ、ABC 与 E。
    orientation_parameters = curve.orientation_parameters
    orientation_quaternions = curve.orientation_quaternions
    orientation_tangents = (
        _build_orientation_tangents(orientation_quaternions)
        if orientation_parameters and orientation_quaternions
        else None
    )
    end_pos = ctrl[-1]
    start_q = kuka_abc_to_quaternion(
        curve.start_pos.a, curve.start_pos.b, curve.start_pos.c,
    )
    end_q = kuka_abc_to_quaternion(end_pos.a, end_pos.b, end_pos.c)
    constant_orientation = (
        abs(curve.start_pos.a - end_pos.a) < 1e-9
        and abs(curve.start_pos.b - end_pos.b) < 1e-9
        and abs(curve.start_pos.c - end_pos.c) < 1e-9
        and not (orientation_parameters and orientation_quaternions)
    )
    u_min = knots[degree]
    u_max = knots[n_ctrl]
    u_span = u_max - u_min

    timing_distances = []
    timing_angles = []
    previous_q = None
    for index, u_value in enumerate(u_list):
        normalized_u = (u_value - u_min) / u_span if u_span > 1e-12 else 0.0
        normalized_s = len_list[index] / total_length
        if orientation_parameters and orientation_quaternions:
            current_q = _sample_kuka_orientation(
                normalized_u,
                orientation_parameters,
                orientation_quaternions,
                orientation_tangents,
            )
        elif constant_orientation:
            current_q = start_q
        else:
            current_q = quaternion_slerp(start_q, end_q, normalized_s)
        if previous_q is not None:
            timing_distances.append(
                len_list[index] - len_list[index - 1]
            )
            timing_angles.append(
                _quaternion_angle_deg(previous_q, current_q)
            )
        previous_q = current_q

    timing_increments = _smoothed_pose_timing_increments(
        timing_distances,
        timing_angles,
        target_velocity,
        max_angular_speed_deg_s,
        t_acc,
        t_dec,
    )
    timing_len_list = [0.0]
    for increment in timing_increments:
        timing_len_list.append(timing_len_list[-1] + increment)
    total_timing_length = timing_len_list[-1]
    total_time, t_flat = _compute_time_profile(
        total_timing_length, target_velocity, t_acc, t_dec
    )
    if total_time <= 0.0:
        return

    num_steps = int(math.ceil(total_time / dt))
    corrected_total_time = num_steps * dt
    start_e = curve.e_val - curve.delta_e
    current_e = start_e
    source_e_profile = validate_source_e_profile(
        curve.source_e_parameters,
        curve.source_e_values,
        start_e=start_e,
        end_e=curve.e_val,
    )

    # 姿态：Planner 曲线使用位置参数轴上的 KUKA 四元数样本；旧调用者
    # 保留端点 SLERP 回退。
    fixed_a = curve.start_pos.a
    fixed_b = curve.start_pos.b
    fixed_c = curve.start_pos.c
    previous_abc = (fixed_a, fixed_b, fixed_c)

    prev_s = 0.0
    lookup_idx = 0
    span = degree
    for i in range(num_steps + 1):
        t = i * dt
        s_norm = _three_stage_sept_poly(t, corrected_total_time, t_acc, t_dec)
        s_norm_clamped = max(0.0, min(1.0, s_norm))
        if i == num_steps:
            s_norm_clamped = 1.0  # 确保最后一点落在终点

        curr_timing_length = s_norm_clamped * total_timing_length
        t_lookup0 = time.perf_counter()
        u, lookup_idx = _lookup_u_from_target_len_monotonic(
            curr_timing_length,
            u_list,
            timing_len_list,
            total_timing_length,
            lookup_idx,
        )
        curr_s = _interpolate_mapped_value(
            u, u_list, len_list, lookup_idx
        )
        if profile is not None:
            profile["sample_lookup_s"] += time.perf_counter() - t_lookup0

        t_deboor0 = time.perf_counter()
        p, span = _eval_bspline_point(u, degree, knots, ctrl_xyzabc, n_ctrl, span)
        if profile is not None:
            profile["sample_deboor_s"] += time.perf_counter() - t_deboor0

        # 姿态插值
        t_pose0 = time.perf_counter()
        normalized_u = (u - u_min) / u_span if u_span > 1e-12 else 0.0
        normalized_u = max(0.0, min(1.0, normalized_u))
        if orientation_parameters and orientation_quaternions:
            q = _sample_kuka_orientation(
                normalized_u,
                orientation_parameters,
                orientation_quaternions,
                orientation_tangents,
            )
            p.a, p.b, p.c = quaternion_to_kuka_abc(q, near_deg=previous_abc)
        elif constant_orientation:
            p.a = fixed_a
            p.b = fixed_b
            p.c = fixed_c
        else:
            spatial_ratio = curr_s / total_length
            q = quaternion_slerp(start_q, end_q, spatial_ratio)
            p.a, p.b, p.c = quaternion_to_kuka_abc(q, near_deg=previous_abc)
        previous_abc = (p.a, p.b, p.c)
        if profile is not None:
            profile["sample_pose_s"] += time.perf_counter() - t_pose0

        # 挤出分配：显式源 E 走参数化分段线性；缺失时保留旧弧长比例。
        t_extrude0 = time.perf_counter()
        delta_s = curr_s - prev_s
        if source_e_profile is None:
            delta_e = curve.delta_e * (delta_s / total_length)
            current_e += delta_e
        else:
            if i == 0:
                target_e = source_e_profile[1][0]
            elif i == num_steps:
                target_e = source_e_profile[1][-1]
            else:
                target_e = _sample_source_e_profile(
                    normalized_u,
                    source_e_profile[0],
                    source_e_profile[1],
                )
            delta_e = target_e - current_e
            if delta_e < -1e-9:
                raise ValueError("source E profile produced a negative sampled extrusion increment")
            if delta_e < 0.0:
                delta_e = 0.0
                target_e = current_e
            current_e = target_e
        prev_s = curr_s

        # 速度估计：用前一帧差分
        feed_mm_s = delta_s / dt if dt > 0 else 0.0
        feed_mm_min = feed_mm_s * 60.0
        extrude_speed = delta_e / dt if dt > 0 else 0.0
        if profile is not None:
            profile["sample_extrude_s"] += time.perf_counter() - t_extrude0

        yield InterpolatedPoint(
            t=t,
            pos=p,
            e=current_e,
            extrude_speed=extrude_speed,
            feedrate_mm_min=feed_mm_min,
            cmd_type=curve.type,
            line=curve.line,
            raw=curve.raw,
        )


def _polyline_e_profile(
    curve: GlobalCurveCommand,
    point_count: int,
) -> Optional[List[float]]:
    profile = getattr(curve, "e_profile", None)
    if profile is None:
        return None
    values = [float(value) for value in profile]
    if len(values) != point_count:
        raise ValueError(
            "polyline E profile must contain one absolute E value per control point"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("polyline E profile must contain only finite values")
    if any(right < left - 1e-6 for left, right in zip(values, values[1:])):
        raise ValueError("polyline E profile must be non-decreasing")
    start_e = float(curve.e_val - curve.delta_e)
    if abs(values[0] - start_e) > 1e-5 or abs(values[-1] - float(curve.e_val)) > 1e-5:
        raise ValueError("polyline E profile endpoints must match curve E state")
    return values


def sample_global_curve(
    curve: GlobalCurveCommand,
    dt: float = 0.004,
    target_velocity: float = 10.0,  # mm/s
    t_acc: float = 2.0,
    t_dec: float = 2.0,
    max_angular_speed_deg_s: Optional[float] = None,
    profile: Optional[dict] = None,
) -> List[InterpolatedPoint]:
    """对一条全局 B 样条进行时间参数化并采样（列表版，兼容旧调用）."""
    return list(
        sample_global_curve_iter(
            curve,
            dt=dt,
            target_velocity=target_velocity,
            t_acc=t_acc,
            t_dec=t_dec,
            max_angular_speed_deg_s=max_angular_speed_deg_s,
            profile=profile))


__all__ = [
    "InterpolatedPoint",
    "sample_global_curve",
    "sample_global_curve_iter",
]
