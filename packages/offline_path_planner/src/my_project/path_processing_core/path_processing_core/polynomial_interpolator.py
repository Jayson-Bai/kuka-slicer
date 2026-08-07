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

from .types import Position, GlobalCurveCommand
from .kuka_orientation import (
    kuka_abc_to_quaternion,
    quaternion_from_rotation_vector,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_to_kuka_abc,
)


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


def _eval_bspline_vector(
    u: float, degree: int, knots: List[float], controls, span: int
) -> tuple[float, float, float]:
    coeffs = _basis_funs(span, u, degree, knots)
    start = span - degree
    return tuple(
        sum(coeff * controls[start + offset][axis] for offset, coeff in enumerate(coeffs))
        for axis in range(3)
    )


def _build_arc_length_map(ctrl: List[Position], degree: int = 3, samples: int = 400):
    knots = _make_open_uniform_knots(len(ctrl), degree)
    u_min = knots[degree]
    u_max = knots[len(ctrl)]
    ctrl_xyzabc = _split_ctrl_components(ctrl)
    n_ctrl = len(ctrl)

    u_list: List[float] = []
    len_list: List[float] = []

    prev_pos, span = _eval_bspline_point(u_min, degree, knots, ctrl_xyzabc, n_ctrl, degree)
    u_list.append(u_min)
    len_list.append(0.0)

    current_len = 0.0
    for i in range(1, samples + 1):
        u = u_min + (u_max - u_min) * i / samples
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


# -------------------------- 采样主逻辑 --------------------------

def sample_global_curve_iter(
    curve: GlobalCurveCommand,
    dt: float = 0.004,
    target_velocity: float = 10.0,  # mm/s
    t_acc: float = 2.0,
    t_dec: float = 2.0,
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

    if (curve.cmd or "").upper() == "POLYLINE":
        points = [curve.start_pos] + list(curve.control_points)
        e_profile = _polyline_e_profile(curve, len(points))
        seg_lengths = []
        total_length = 0.0
        for start, end in zip(points, points[1:]):
            length = math.sqrt(
                (end.x - start.x) ** 2
                + (end.y - start.y) ** 2
                + (end.z - start.z) ** 2
            )
            seg_lengths.append(length)
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

        total_time, _ = _compute_time_profile(total_length, target_velocity, t_acc, t_dec)
        if total_time <= 0.0:
            return
        num_steps = int(math.ceil(total_time / dt))
        corrected_total_time = num_steps * dt
        start_e = curve.e_val - curve.delta_e
        current_e = start_e
        prev_s = 0.0
        seg_idx = 0
        seg_start_s = 0.0
        previous_abc = (points[0].a, points[0].b, points[0].c)

        for i in range(num_steps + 1):
            t = i * dt
            s_norm = _three_stage_sept_poly(t, corrected_total_time, t_acc, t_dec)
            s_norm_clamped = max(0.0, min(1.0, s_norm))
            if i == num_steps:
                s_norm_clamped = 1.0

            curr_s = s_norm_clamped * total_length
            while (
                seg_idx < len(seg_lengths) - 1
                and curr_s > seg_start_s + seg_lengths[seg_idx]
            ):
                seg_start_s += seg_lengths[seg_idx]
                seg_idx += 1

            seg_len = seg_lengths[seg_idx]
            local = 0.0 if seg_len <= 1e-9 else (curr_s - seg_start_s) / seg_len
            local = max(0.0, min(1.0, local))
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

        total_time, _ = _compute_time_profile(total_length, target_velocity, t_acc, t_dec)
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
        ctrl, degree=degree, samples=max(400, len(ctrl) * 10))
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

    # 时间规划
    total_time, t_flat = _compute_time_profile(total_length, target_velocity, t_acc, t_dec)
    if total_time <= 0.0:
        return

    num_steps = int(math.ceil(total_time / dt))
    corrected_total_time = num_steps * dt
    start_e = curve.e_val - curve.delta_e
    current_e = start_e

    # 姿态：仅用起点/终点做 slerp
    # Planner-generated curves carry KUKA quaternion-log controls.  Curves
    # created by older callers retain a KUKA quaternion endpoint fallback.
    end_pos = ctrl[-1]
    start_q = kuka_abc_to_quaternion(
        curve.start_pos.a, curve.start_pos.b, curve.start_pos.c,
    )
    end_q = kuka_abc_to_quaternion(end_pos.a, end_pos.b, end_pos.c)
    constant_orientation = (
        abs(curve.start_pos.a - end_pos.a) < 1e-9
        and abs(curve.start_pos.b - end_pos.b) < 1e-9
        and abs(curve.start_pos.c - end_pos.c) < 1e-9
    )
    fixed_a = curve.start_pos.a
    fixed_b = curve.start_pos.b
    fixed_c = curve.start_pos.c
    orientation_controls = curve.orientation_control_vectors
    orientation_reference = curve.orientation_reference_quaternion
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

        curr_s = s_norm_clamped * total_length
        t_lookup0 = time.perf_counter()
        u, lookup_idx = _lookup_u_from_target_len_monotonic(
            curr_s, u_list, len_list, total_length, lookup_idx
        )
        if profile is not None:
            profile["sample_lookup_s"] += time.perf_counter() - t_lookup0

        t_deboor0 = time.perf_counter()
        p, span = _eval_bspline_point(u, degree, knots, ctrl_xyzabc, n_ctrl, span)
        if profile is not None:
            profile["sample_deboor_s"] += time.perf_counter() - t_deboor0

        # 姿态插值
        t_pose0 = time.perf_counter()
        if orientation_reference is not None and orientation_controls:
            vector = _eval_bspline_vector(u, degree, knots, orientation_controls, span)
            q = quaternion_multiply(orientation_reference, quaternion_from_rotation_vector(vector))
            p.a, p.b, p.c = quaternion_to_kuka_abc(q, near_deg=previous_abc)
        elif constant_orientation:
            p.a = fixed_a
            p.b = fixed_b
            p.c = fixed_c
        else:
            q = quaternion_slerp(start_q, end_q, s_norm_clamped)
            p.a, p.b, p.c = quaternion_to_kuka_abc(q, near_deg=previous_abc)
        previous_abc = (p.a, p.b, p.c)
        if profile is not None:
            profile["sample_pose_s"] += time.perf_counter() - t_pose0

        # 挤出分配（按弧长比例）
        t_extrude0 = time.perf_counter()
        delta_s = curr_s - prev_s
        delta_e = curve.delta_e * (delta_s / total_length)
        current_e += delta_e
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
            profile=profile))


__all__ = [
    "InterpolatedPoint",
    "sample_global_curve",
    "sample_global_curve_iter",
]
