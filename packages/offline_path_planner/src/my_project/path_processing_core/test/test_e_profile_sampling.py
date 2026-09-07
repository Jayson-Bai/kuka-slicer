import pytest

from path_processing_core.polynomial_interpolator import sample_global_curve_iter
from path_processing_core.types import GlobalCurveCommand, Position


def test_polyline_sampler_follows_piecewise_e_profile_instead_of_total_arc_length():
    curve = GlobalCurveCommand(
        type="PRINT",
        cmd="POLYLINE",
        start_pos=Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        control_points=[
            Position(1.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            Position(2.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        ],
        e_val=10.0,
        delta_e=10.0,
        feedrate=60.0,
        line=1,
        e_profile=[0.0, 2.0, 10.0],
    )

    samples = list(
        sample_global_curve_iter(
            curve,
            dt=1.0,
            target_velocity=1.0,
            t_acc=0.0,
            t_dec=0.0,
        )
    )

    assert [sample.e for sample in samples] == pytest.approx([0.0, 2.0, 10.0])


def test_polyline_sampler_preserves_source_e_at_four_ms_distance_samples():
    curve = GlobalCurveCommand(
        type="PRINT",
        cmd="POLYLINE",
        start_pos=Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        control_points=[
            Position(1.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            Position(3.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        ],
        e_val=5.0,
        delta_e=5.0,
        feedrate=45_000.0,
        line=1,
        e_profile=[0.0, 1.0, 5.0],
    )

    samples = list(
        sample_global_curve_iter(
            curve,
            dt=0.004,
            target_velocity=250.0,
            t_acc=0.0,
            t_dec=0.0,
        )
    )

    assert [sample.e for sample in samples] == pytest.approx([0.0, 1.0, 3.0, 5.0])
    assert [sample.extrude_speed for sample in samples] == pytest.approx(
        [0.0, 250.0, 500.0, 500.0]
    )
