"""
全局 B 样条拟合规划器 (Global B-Spline Planner):.

- 接收连续的 MoveCommand 序列。
- 使用 BSpline 外部库进行向心参数化与最小二乘拟合。
- 结合了回退点/加密逻辑（提供数据约束）和 B 样条逼近（提供平滑性）。
"""

import math
import time
from typing import List, Optional, Tuple

from .bspline import parameter_selection as ps
from .bspline import bspline_curve as bc

from .types import MoveCommand, Position, GlobalCurveCommand


def compute_angle_deg(v1: tuple, v2: tuple) -> float:
    """返回 v1 与 v2 的夹角（度），向量长度过短时返回 0."""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_theta))


def _distance_xyz(p1: Position, p2: Position) -> float:
    """欧氏距离，仅使用 XYZ 分量."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dz = p2.z - p1.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _generate_fitting_points(
    moves: List[MoveCommand],
    angle_threshold_deg: float,
    corner_retreat_ratio: float,
) -> List[Position]:
    """
    基于原始折线顶点生成拟合点序列.

    - 首/尾点保留
    - 对夹角超过阈值的顶点，在两侧按比例回退插入两个点
    - 直线处不额外加点
    """
    if not moves:
        return []

    # 去除首尾重合的重复点（相邻零长度）
    pts: List[Position] = [moves[0].start_pos]
    for m in moves:
        pts.append(m.pos)
    unique_pts: List[Position] = [pts[0]]
    for p in pts[1:]:
        if _distance_xyz(unique_pts[-1], p) < 1e-9:
            continue
        unique_pts.append(p)

    if len(unique_pts) <= 2:
        return unique_pts

    densified: List[Position] = [unique_pts[0]]

    # 限制回退比例，避免跨越整个线段
    retreat_ratio = max(0.0, min(corner_retreat_ratio, 0.49))

    for i in range(1, len(unique_pts) - 1):
        prev_pt = unique_pts[i - 1]
        curr_pt = unique_pts[i]
        next_pt = unique_pts[i + 1]

        v0 = (curr_pt.x - prev_pt.x, curr_pt.y - prev_pt.y, curr_pt.z - prev_pt.z)
        v1 = (next_pt.x - curr_pt.x, next_pt.y - curr_pt.y, next_pt.z - curr_pt.z)

        len0 = math.sqrt(sum(c * c for c in v0))
        len1 = math.sqrt(sum(c * c for c in v1))

        if len0 < 1e-9 or len1 < 1e-9:
            densified.append(curr_pt)
            continue

        # compute_angle_deg 可接受 3 维向量
        ang = compute_angle_deg(v0, v1)

        if ang >= angle_threshold_deg:
            retreat0 = len0 * retreat_ratio
            retreat1 = len1 * retreat_ratio

            u0 = (v0[0] / len0, v0[1] / len0, v0[2] / len0)
            u1 = (v1[0] / len1, v1[1] / len1, v1[2] / len1)

            # 前侧回退点
            densified.append(
                Position(
                    x=curr_pt.x - u0[0] * retreat0,
                    y=curr_pt.y - u0[1] * retreat0,
                    z=curr_pt.z - u0[2] * retreat0,
                    a=curr_pt.a,
                    b=curr_pt.b,
                    c=curr_pt.c,
                )
            )

            densified.append(curr_pt)

            # 后侧回退点
            densified.append(
                Position(
                    x=curr_pt.x + u1[0] * retreat1,
                    y=curr_pt.y + u1[1] * retreat1,
                    z=curr_pt.z + u1[2] * retreat1,
                    a=curr_pt.a,
                    b=curr_pt.b,
                    c=curr_pt.c,
                )
            )
        else:
            densified.append(curr_pt)

    densified.append(unique_pts[-1])
    return densified


def _subdivide_points(points: List[Position]) -> List[Position]:
    """
    对点序列进行一次中点加密.

    在每两个相邻点之间插入它们的中点（线性插值）。
    """
    if len(points) < 2:
        return points

    new_points = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        new_points.append(p1)

        # 计算中点
        mid_pt = Position(
            x=(p1.x + p2.x) * 0.5,
            y=(p1.y + p2.y) * 0.5,
            z=(p1.z + p2.z) * 0.5,
            a=(p1.a + p2.a) * 0.5,
            b=(p1.b + p2.b) * 0.5,
            c=(p1.c + p2.c) * 0.5
        )
        new_points.append(mid_pt)

    new_points.append(points[-1])
    return new_points


class GlobalSplinePlanner:
    def __init__(self):
        self.last_fit_profile = self._new_fit_profile()

    @staticmethod
    def _new_fit_profile():
        return {
            "fit_gen_points_s": 0.0,
            "fit_density_s": 0.0,
            "fit_prepare_data_s": 0.0,
            "fit_param_s": 0.0,
            "fit_knot_s": 0.0,
            "fit_lsq_s": 0.0,
            "fit_post_ctrl_s": 0.0,
            "fit_lsq_basis_build_s": 0.0,
            "fit_lsq_qk_build_s": 0.0,
            "fit_lsq_normal_mat_s": 0.0,
            "fit_lsq_solve_s": 0.0,
            "fit_lsq_total_s": 0.0,
        }

    def fit_global_curve(
        self,
        moves: List[MoveCommand],
        corner_angle_deg: float = 10.0,     # 添加控制点的夹角阈值
        corner_retreat_ratio: float = 0.2,  # 角点处添加的控制点位置回退比例
        density: int = 0,  # 数据点加密密度
        degree: int = 3,  # B 样条阶次
        max_fit_points: int = 20000,  # 单段拟合点数上限，防止密度放大拖垮内存/CPU
    ) -> Optional[GlobalCurveCommand]:
        """
        - 对给定的同类型 MoveCommand 序列进行处理.

        - 生成拟合数据点（包含回退点和加密点）。
        - 使用 BSpline 库进行 Centripetal 参数化和 Averaging Knots 生成。
        - 执行最小二乘逼近。
        """
        self.last_fit_profile = self._new_fit_profile()
        profile = self.last_fit_profile

        if not moves:
            return None

        # 1. 生成拟合点（包含回退点逻辑）
        t0 = time.perf_counter()
        fit_points = _generate_fitting_points(
            moves=moves,
            angle_threshold_deg=corner_angle_deg,
            corner_retreat_ratio=corner_retreat_ratio,
        )
        profile["fit_gen_points_s"] += time.perf_counter() - t0

        # 2. 根据密度参数递归加密数据点
        t0 = time.perf_counter()
        max_fit_points = max(2, int(max_fit_points))
        for _ in range(max(0, int(density))):
            projected_points = len(fit_points) * 2 - 1
            if projected_points > max_fit_points:
                break
            fit_points = _subdivide_points(fit_points)
        profile["fit_density_s"] += time.perf_counter() - t0

        n_points = len(fit_points)
        # 至少需要 degree + 1 个点才能进行 B 样条拟合 (或者至少2个点)
        if n_points < 2:
            return None

        # 3. 控制点数量设为数据点数量的 1/5（向上取整），保持逼近（H < N, H > degree）
        calc_n_ctrl = math.ceil(n_points / 5)
        calc_n_ctrl = max(degree + 1, calc_n_ctrl)
        # 逼近要求控制点数量小于数据点数量
        if calc_n_ctrl >= n_points:
            calc_n_ctrl = n_points - 1
        if calc_n_ctrl <= degree:
            # 数据点不足以进行逼近
            return None

        # 4. 准备 BSpline 库所需的数据格式 [[x...], [y...], ...]
        t0 = time.perf_counter()
        D_X = [p.x for p in fit_points]
        D_Y = [p.y for p in fit_points]
        D_Z = [p.z for p in fit_points]
        D_A = [p.a for p in fit_points]
        D_B = [p.b for p in fit_points]
        D_C = [p.c for p in fit_points]
        D = [D_X, D_Y, D_Z, D_A, D_B, D_C]
        D_N = n_points
        profile["fit_prepare_data_s"] += time.perf_counter() - t0

        # 5. 调用 BSpline 库算法
        try:
            # 计算参数 (Centripetal)
            t0 = time.perf_counter()
            param = ps.centripetal(D_N, D)
            profile["fit_param_s"] += time.perf_counter() - t0

            # 逼近模式（不再退化到插值）
            # 生成适合控制点数量的均匀节点向量：节点数量 = H + k + 1
            t0 = time.perf_counter()
            knot = [0.0] * (degree + 1)
            if calc_n_ctrl > degree + 1:
                num_internal = calc_n_ctrl - (degree + 1)
                step = 1.0 / (num_internal + 1)
                for i in range(num_internal):
                    knot.append(step * (i + 1))
            knot.extend([1.0] * (degree + 1))
            profile["fit_knot_s"] += time.perf_counter() - t0

            lsq_profile = {
                "lsq_basis_build_s": 0.0,
                "lsq_qk_build_s": 0.0,
                "lsq_normal_mat_s": 0.0,
                "lsq_solve_s": 0.0,
                "lsq_total_s": 0.0,
            }
            t0 = time.perf_counter()
            ctrl_raw = bc.curve_approximation(
                D, D_N, calc_n_ctrl, degree, param, knot, profile=lsq_profile)
            profile["fit_lsq_s"] += time.perf_counter() - t0
            profile["fit_lsq_basis_build_s"] += lsq_profile["lsq_basis_build_s"]
            profile["fit_lsq_qk_build_s"] += lsq_profile["lsq_qk_build_s"]
            profile["fit_lsq_normal_mat_s"] += lsq_profile["lsq_normal_mat_s"]
            profile["fit_lsq_solve_s"] += lsq_profile["lsq_solve_s"]
            profile["fit_lsq_total_s"] += lsq_profile["lsq_total_s"]

            if not ctrl_raw or len(ctrl_raw[0]) == 0:
                return None

            # 6. 转换回 Position 列表
            t0 = time.perf_counter()
            res_ctrl_points = []
            n_res = len(ctrl_raw[0])
            for i in range(n_res):
                res_ctrl_points.append(Position(
                    x=ctrl_raw[0][i],
                    y=ctrl_raw[1][i],
                    z=ctrl_raw[2][i],
                    a=ctrl_raw[3][i],
                    b=ctrl_raw[4][i],
                    c=ctrl_raw[5][i]
                ))
            profile["fit_post_ctrl_s"] += time.perf_counter() - t0

        except Exception as e:
            # 容错处理
            print(f"B样条拟合失败：{e}")
            return None

        # 挤出量处理
        total_delta_e = sum(m.delta_e for m in moves)
        final_e = moves[-1].e_val
        constraints: List[Tuple[float, float]] = []

        return GlobalCurveCommand(
            # 保留原始段类型并标注拟合
            type=f"{moves[0].type}_FIT",
            cmd="SPLINE",
            start_pos=res_ctrl_points[0],
            control_points=res_ctrl_points[1:],
            e_val=final_e,
            delta_e=total_delta_e,
            feedrate=moves[0].feedrate,
            line=moves[0].line,
            raw="GLOBAL_BSPLINE_LIB",
            constraints=constraints,
            original_moves=moves
        )
