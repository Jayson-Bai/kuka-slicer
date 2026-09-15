import math

import numpy as np
import pytest

from path_processing_core.bspline_approximation import GlobalSplinePlanner, _generate_fitting_points
from path_processing_core.polynomial_interpolator import (
    _build_orientation_tangents,
    _sample_kuka_orientation,
    sample_global_curve_iter,
)
from path_processing_core.types import MoveCommand, Position


def _move(index, start, end):
    return MoveCommand(
        type="PRINT",
        cmd="G1",
        start_pos=start,
        pos=end,
        e_val=float(index + 1),
        delta_e=1.0,
        feedrate=600.0,
        line=index + 1,
    )


def _max_quaternion_angular_speed(samples, dt):
    from path_processing_core.kuka_orientation import kuka_abc_to_quaternion

    quaternions = [
        kuka_abc_to_quaternion(sample.pos.a, sample.pos.b, sample.pos.c)
        for sample in samples
    ]
    steps = []
    for left, right in zip(quaternions, quaternions[1:]):
        dot = min(1.0, max(-1.0, abs(sum(a * b for a, b in zip(left, right)))))
        steps.append(math.degrees(2.0 * math.acos(dot)) / dt)
    return max(steps, default=0.0)


def _max_angular_speed_change(samples, dt):
    from path_processing_core.kuka_orientation import kuka_abc_to_quaternion

    quaternions = [
        kuka_abc_to_quaternion(sample.pos.a, sample.pos.b, sample.pos.c)
        for sample in samples
    ]
    speeds = [
        _quaternion_distance_deg(left, right) / dt
        for left, right in zip(quaternions, quaternions[1:])
    ]
    return max(
        (abs(right - left) for left, right in zip(speeds, speeds[1:])),
        default=0.0,
    )


def _quaternion_distance_deg(left, right):
    dot = min(1.0, max(-1.0, abs(sum(a * b for a, b in zip(left, right)))))
    return math.degrees(2.0 * math.acos(dot))


def test_global_bspline_keeps_middle_kuka_orientation_when_endpoints_match():
    points = [
        Position(float(index), 0.0, 0.5, 0.0, 20.0 * math.sin(math.pi * index / 8.0), 0.0)
        for index in range(9)
    ]
    curve = GlobalSplinePlanner().fit_global_curve(
        [_move(index, points[index], points[index + 1]) for index in range(8)],
        density=1,
    )

    assert curve is not None
    assert curve.orientation_parameters is not None
    assert curve.orientation_quaternions is not None

    samples = list(sample_global_curve_iter(curve, dt=0.1, target_velocity=10.0, t_acc=0.0, t_dec=0.0))

    assert max(sample.pos.b for sample in samples) > 12.0
    assert samples[0].pos.b == pytest.approx(0.0, abs=1e-8)
    assert samples[-1].pos.b == pytest.approx(0.0, abs=1e-8)


def test_local_quaternion_curve_has_continuous_speed_at_orientation_knots():
    from path_processing_core.kuka_orientation import kuka_abc_to_quaternion

    parameters = [0.0, 0.5, 1.0]
    quaternions = [
        kuka_abc_to_quaternion(0.0, value, 0.0)
        for value in (0.0, 30.0, 10.0)
    ]
    tangents = _build_orientation_tangents(quaternions)
    epsilon = 1e-5
    center = _sample_kuka_orientation(
        0.5, parameters, quaternions, tangents
    )
    before = _sample_kuka_orientation(
        0.5 - epsilon, parameters, quaternions, tangents
    )
    after = _sample_kuka_orientation(
        0.5 + epsilon, parameters, quaternions, tangents
    )

    left_speed = _quaternion_distance_deg(before, center) / epsilon
    right_speed = _quaternion_distance_deg(center, after) / epsilon
    assert left_speed == pytest.approx(right_speed, rel=2e-3)


def test_corner_retreat_points_slerp_kuka_orientation_instead_of_copying_the_corner():
    points = [
        Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        Position(10.0, 0.0, 0.5, 0.0, 20.0, 0.0),
        Position(10.0, 10.0, 0.5, 0.0, 0.0, 0.0),
    ]
    fitted = _generate_fitting_points(
        [_move(index, points[index], points[index + 1]) for index in range(2)],
        angle_threshold_deg=10.0,
        corner_retreat_ratio=0.2,
    )

    assert fitted[1].b == pytest.approx(16.0)
    assert fitted[3].b == pytest.approx(16.0)


def test_zero_xyz_kuka_orientation_change_is_expanded_to_continuous_rsi_samples():
    start = Position(10.0, 20.0, 0.5, 0.0, 0.0, 0.0)
    end = Position(10.0, 20.0, 0.5, 0.0, 2.0, 0.0)
    # A two-point move cannot be globally fitted, so exercise Core's existing
    # linear fallback representation of this in-place rotation.
    from path_processing_core.types import GlobalCurveCommand
    curve = GlobalCurveCommand(
        type="TRAVEL", cmd="SPLINE", start_pos=start,
        control_points=[end, end, end], e_val=0.0, delta_e=0.0,
        feedrate=600.0, line=1,
    )
    samples = list(sample_global_curve_iter(curve, dt=0.004))

    assert len(samples) > 2
    assert samples[0].pos.b == pytest.approx(0.0)
    assert samples[-1].pos.b == pytest.approx(2.0)
    assert max(abs(right.pos.b - left.pos.b) for left, right in zip(samples, samples[1:])) < 0.101


def test_orientation_only_move_obeys_requested_tcp_orientation_speed():
    from path_processing_core.types import GlobalCurveCommand

    start = Position(10.0, 20.0, 0.5, 179.0, 0.0, 0.0)
    end = Position(10.0, 20.0, 0.5, -179.0, 0.0, 0.0)
    curve = GlobalCurveCommand(
        type="TRAVEL", cmd="SPLINE", start_pos=start,
        control_points=[end, end, end], e_val=0.0, delta_e=0.0,
        feedrate=600.0, line=1,
    )

    samples = list(
        sample_global_curve_iter(
            curve,
            dt=0.004,
            max_angular_speed_deg_s=10.0,
        )
    )

    assert _max_quaternion_angular_speed(samples, 0.004) <= 10.0 + 1e-5
    assert len(samples) < 200


def test_polyline_curvature_retimes_xyz_abc_and_e_together():
    from path_processing_core.types import GlobalCurveCommand

    curve = GlobalCurveCommand(
        type="PRINT",
        cmd="POLYLINE",
        start_pos=Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        control_points=[
            Position(5.0, 0.0, 0.5, 0.0, 45.0, 0.0),
            Position(10.0, 0.0, 0.5, 0.0, 90.0, 0.0),
        ],
        e_val=10.0,
        delta_e=10.0,
        feedrate=1200.0,
        line=1,
        e_profile=[0.0, 5.0, 10.0],
    )

    unrestricted = list(
        sample_global_curve_iter(
            curve, dt=0.004, target_velocity=20.0, t_acc=0.0, t_dec=0.0,
        )
    )
    limited = list(
        sample_global_curve_iter(
            curve,
            dt=0.004,
            target_velocity=20.0,
            t_acc=0.0,
            t_dec=0.0,
            max_angular_speed_deg_s=25.0,
        )
    )

    assert len(limited) > len(unrestricted)
    assert _max_quaternion_angular_speed(limited, 0.004) <= 25.0 + 1e-4
    assert limited[-1].pos.x == pytest.approx(10.0)
    assert limited[-1].pos.b == pytest.approx(90.0)
    assert limited[-1].e == pytest.approx(10.0)


def test_bspline_surface_orientation_retimes_only_when_limit_requires_it():
    points = [
        Position(float(index), 0.0, 0.5, 0.0, 30.0 * math.sin(math.pi * index / 8.0), 0.0)
        for index in range(9)
    ]
    curve = GlobalSplinePlanner().fit_global_curve(
        [_move(index, points[index], points[index + 1]) for index in range(8)],
        density=1,
    )

    limited = list(
        sample_global_curve_iter(
            curve,
            dt=0.004,
            target_velocity=20.0,
            t_acc=0.2,
            t_dec=0.2,
            max_angular_speed_deg_s=30.0,
        )
    )

    assert _max_quaternion_angular_speed(limited, 0.004) <= 30.0 + 0.05
    assert limited[-1].e == pytest.approx(curve.e_val)


def test_curvature_speed_limit_uses_existing_ramp_as_lookahead_envelope():
    points = [
        Position(float(index), 0.0, 0.5, 0.0, 30.0 * math.sin(math.pi * index / 8.0), 0.0)
        for index in range(9)
    ]
    curve = GlobalSplinePlanner().fit_global_curve(
        [_move(index, points[index], points[index + 1]) for index in range(8)],
        density=1,
    )
    abrupt = list(
        sample_global_curve_iter(
            curve,
            dt=0.004,
            target_velocity=20.0,
            t_acc=0.0,
            t_dec=0.0,
            max_angular_speed_deg_s=25.0,
        )
    )
    smoothed = list(
        sample_global_curve_iter(
            curve,
            dt=0.004,
            target_velocity=20.0,
            t_acc=0.2,
            t_dec=0.2,
            max_angular_speed_deg_s=25.0,
        )
    )

    assert _max_angular_speed_change(smoothed, 0.004) < _max_angular_speed_change(
        abrupt, 0.004
    )
    assert _max_quaternion_angular_speed(smoothed, 0.004) <= 25.0 + 0.05


def test_exporter_aligns_abc_between_adjacent_moves_in_the_same_buffer(tmp_path):
    from path_processing_core.npz_exporter import export_npz

    p0 = Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    p1 = Position(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    discontinuous_start = Position(0.0, 0.0, 1.0, 0.0, 10.0, 0.0)
    p2 = Position(5.0, 0.0, 1.0, 0.0, 10.0, 0.0)
    output = tmp_path / "core.npz"

    export_npz(
        [_move(0, p0, p1), _move(1, discontinuous_start, p2)],
        str(output), dt=0.02,
    )

    with np.load(output, allow_pickle=False) as data:
        b = data["b"]
    assert np.max(np.abs(np.diff(b))) < 1.0
