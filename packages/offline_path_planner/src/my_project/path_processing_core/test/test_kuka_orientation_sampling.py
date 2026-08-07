import math

import pytest

from path_processing_core.bspline_approximation import GlobalSplinePlanner
from path_processing_core.polynomial_interpolator import sample_global_curve_iter
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
    assert curve.orientation_reference_quaternion is not None
    assert curve.orientation_control_vectors is not None

    samples = list(sample_global_curve_iter(curve, dt=0.1, target_velocity=10.0, t_acc=0.0, t_dec=0.0))

    assert max(sample.pos.b for sample in samples) > 12.0
    assert samples[0].pos.b == pytest.approx(0.0, abs=1e-8)
    assert samples[-1].pos.b == pytest.approx(0.0, abs=1e-8)
