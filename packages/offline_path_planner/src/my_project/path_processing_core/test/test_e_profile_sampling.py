import pytest

from path_processing_core.polynomial_interpolator import sample_global_curve_iter
from path_processing_core.types import (
    GlobalCurveCommand,
    Position,
    validate_source_e_profile,
)


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


def test_absent_source_e_profile_keeps_the_legacy_arc_length_distribution():
    curve = GlobalCurveCommand(
        type="PRINT_FIT",
        cmd="SPLINE",
        start_pos=Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        control_points=[
            Position(1.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            Position(2.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            Position(3.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        ],
        e_val=6.0,
        delta_e=6.0,
        feedrate=60.0,
        line=1,
    )

    samples = list(sample_global_curve_iter(curve, dt=0.05, target_velocity=20.0, t_acc=0.0, t_dec=0.0))

    assert samples[0].e == pytest.approx(0.0)
    assert samples[-1].e == pytest.approx(6.0)
    assert any(0.0 < sample.e < 6.0 for sample in samples[1:-1])


@pytest.mark.parametrize(
    ("parameters", "values", "message"),
    [
        ([0.0], [0.0], "at least 2"),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 2.0], "strictly increasing"),
        ([0.0, 0.5, 1.0], [0.0, 2.0, 1.0], "monotonic"),
        ([0.1, 1.0], [0.0, 2.0], "start at 0"),
    ],
)
def test_source_e_profile_contract_rejects_invalid_samples(parameters, values, message):
    with pytest.raises(ValueError, match=message):
        validate_source_e_profile(parameters, values, start_e=0.0, end_e=2.0)


@pytest.mark.xfail(
    strict=True,
    reason="B 样条尚未保留共形源路径的逐段 E；C4 完成后移除此基线标记。",
)
def test_bspline_sampler_keeps_a_zero_e_connector_as_a_constant_e_platform():
    """锁定宏路径 [正 E, 零 E, 正 E] 进入 B 样条后的当前缺陷。"""
    curve = GlobalCurveCommand(
        type="PRINT_FIT",
        cmd="SPLINE",
        start_pos=Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        control_points=[
            Position(1.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            Position(2.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            Position(3.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        ],
        e_val=7.0,
        delta_e=7.0,
        feedrate=60.0,
        line=1,
        source_e_parameters=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        source_e_values=[0.0, 3.0, 3.0, 7.0],
    )

    samples = list(
        sample_global_curve_iter(
            curve,
            dt=0.05,
            target_velocity=20.0,
            t_acc=0.0,
            t_dec=0.0,
        )
    )

    platform_samples = [
        sample.e
        for sample in samples
        if 1.0 / 3.0 <= sample.t / samples[-1].t <= 2.0 / 3.0
    ]
    assert len(platform_samples) >= 2
    assert platform_samples == pytest.approx([3.0] * len(platform_samples), abs=1e-9)
