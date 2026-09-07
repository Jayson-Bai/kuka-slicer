import math

import numpy as np
import pytest

from path_processing_core.bspline_approximation import GlobalSplinePlanner, _generate_fitting_points
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
    assert curve.orientation_parameters is not None
    assert curve.orientation_quaternions is not None

    samples = list(sample_global_curve_iter(curve, dt=0.1, target_velocity=10.0, t_acc=0.0, t_dec=0.0))

    assert max(sample.pos.b for sample in samples) > 12.0
    assert samples[0].pos.b == pytest.approx(0.0, abs=1e-8)
    assert samples[-1].pos.b == pytest.approx(0.0, abs=1e-8)


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
