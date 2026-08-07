"""KUKA 姿态约定下的曲面法向工具姿态。"""

from __future__ import annotations

import numpy as np


# 在已标定的平面打印参考姿态中：+X_TOOL 朝下，+Y_TOOL 与工件 +Y 一致。
# KUKA 的 ABC=0 是相对该平面参考姿态的增量，而不是世界坐标的绝对姿态。
_FLAT_TOOL_BASIS = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)


def kuka_abc_for_surface(
    dz_dx: np.ndarray, dz_dy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return relative KUKA ``A(Z)-B(Y)-C(X)`` angles in degrees.

    The mapped surface is ``z(x, y)``.  The nozzle work axis is KUKA's
    ``+X_TOOL`` and must point into the workpiece, i.e. opposite the upward
    surface normal.  ``+Y_TOOL`` uses the projected workpiece +Y axis, giving
    a deterministic minimum-twist roll convention.
    """

    dz_dx = np.asarray(dz_dx, dtype=np.float64)
    dz_dy = np.asarray(dz_dy, dtype=np.float64)
    normal = np.stack((-dz_dx, -dz_dy, np.ones_like(dz_dx)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    tool_x = -normal

    reference_y = np.zeros_like(tool_x)
    reference_y[..., 1] = 1.0
    tool_y = reference_y - np.sum(reference_y * tool_x, axis=-1, keepdims=True) * tool_x
    near_parallel = np.linalg.norm(tool_y, axis=-1) < 1e-9
    if np.any(near_parallel):
        fallback_y = np.zeros_like(tool_x)
        fallback_y[..., 0] = 1.0
        tool_y[near_parallel] = fallback_y[near_parallel] - (
            np.sum(fallback_y[near_parallel] * tool_x[near_parallel], axis=-1, keepdims=True)
            * tool_x[near_parallel]
        )
    tool_y /= np.linalg.norm(tool_y, axis=-1, keepdims=True)
    tool_z = np.cross(tool_x, tool_y)

    desired = np.stack((tool_x, tool_y, tool_z), axis=-1)
    relative = desired @ _FLAT_TOOL_BASIS.T
    b = np.arcsin(np.clip(-relative[..., 2, 0], -1.0, 1.0))
    a = np.arctan2(relative[..., 1, 0], relative[..., 0, 0])
    c = np.arctan2(relative[..., 2, 1], relative[..., 2, 2])
    return np.degrees(a), np.degrees(b), np.degrees(c)
