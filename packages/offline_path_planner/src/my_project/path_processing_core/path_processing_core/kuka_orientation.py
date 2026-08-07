"""Quaternion helpers for KUKA's A(Z)-B(Y)-C(X) orientation convention."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple


Quaternion = Tuple[float, float, float, float]


def kuka_abc_to_quaternion(a_deg: float, b_deg: float, c_deg: float) -> Quaternion:
    """Convert KUKA ABC degrees (Rz(A) @ Ry(B) @ Rx(C)) to a quaternion."""

    a = math.radians(a_deg) * 0.5
    b = math.radians(b_deg) * 0.5
    c = math.radians(c_deg) * 0.5
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    return _normalize((ca * cb * cc + sa * sb * sc,
                       ca * cb * sc - sa * sb * cc,
                       ca * sb * cc + sa * cb * sc,
                       sa * cb * cc - ca * sb * sc))


def quaternion_to_kuka_abc(
    q: Quaternion, near_deg: Optional[Tuple[float, float, float]] = None
) -> Tuple[float, float, float]:
    """Convert a quaternion to KUKA ABC and unwrap A/C near the prior sample."""

    w, x, y, z = _normalize(q)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    b = math.asin(max(-1.0, min(1.0, -r20)))
    a = math.atan2(r10, r00)
    c = math.atan2(r21, r22)
    result = tuple(math.degrees(value) for value in (a, b, c))
    if near_deg is None:
        return result
    return (_nearest_equivalent(result[0], near_deg[0]), result[1], _nearest_equivalent(result[2], near_deg[2]))


def quaternion_multiply(lhs: Quaternion, rhs: Quaternion) -> Quaternion:
    w0, x0, y0, z0 = lhs
    w1, x1, y1, z1 = rhs
    return _normalize((
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ))


def quaternion_inverse(q: Quaternion) -> Quaternion:
    w, x, y, z = _normalize(q)
    return w, -x, -y, -z


def quaternion_slerp(q0: Quaternion, q1: Quaternion, t: float) -> Quaternion:
    w0, x0, y0, z0 = _normalize(q0)
    w1, x1, y1, z1 = _normalize(q1)
    dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
    if dot < 0.0:
        w1, x1, y1, z1, dot = -w1, -x1, -y1, -z1, -dot
    if dot > 0.9995:
        return _normalize((w0 + t * (w1 - w0), x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0)))
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    s0 = math.sin((1.0 - t) * theta) / sin_theta
    s1 = math.sin(t * theta) / sin_theta
    return _normalize((s0 * w0 + s1 * w1, s0 * x0 + s1 * x1, s0 * y0 + s1 * y1, s0 * z0 + s1 * z1))


def rotation_vector_from_quaternion(q: Quaternion) -> Tuple[float, float, float]:
    """Quaternion logarithm as a radians rotation vector in the shortest branch."""

    w, x, y, z = _normalize(q)
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return 0.0, 0.0, 0.0
    angle = 2.0 * math.atan2(sin_half, w)
    factor = angle / sin_half
    return x * factor, y * factor, z * factor


def quaternion_from_rotation_vector(vector: Iterable[float]) -> Quaternion:
    x, y, z = (float(value) for value in vector)
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    factor = math.sin(angle * 0.5) / angle
    return math.cos(angle * 0.5), x * factor, y * factor, z * factor


def _normalize(q: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-15:
        raise ValueError("zero-length quaternion is not a valid KUKA orientation")
    return tuple(value / norm for value in q)  # type: ignore[return-value]


def _nearest_equivalent(value: float, near: float) -> float:
    return value + 360.0 * round((near - value) / 360.0)
