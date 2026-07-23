import math

import pytest

from external_npz_preprocessor.process_params import (
    FiberProcessParams,
    ProcessParams,
    RESIN_FILAMENT_DIAMETER_MM,
    RESIN_FILAMENT_LENGTH_PER_MM3,
    RESIN_FIXED_BEAD_WIDTH_MM,
    ResinProcessParams,
)


def test_resin_e_per_mm_uses_fixed_line_width_layer_height_scale_and_filament_area():
    params = ProcessParams(
        resin=ResinProcessParams(
            layer_height_mm=0.4,
            extrusion_scale=1.5,
            feed_mm_s=8.0,
        ),
        fiber=FiberProcessParams(extrusion_scale=0.8, feed_mm_s=5.0),
    )

    assert params.resin.e_per_mm() == pytest.approx(
        2.0 * 0.4 * 1.5 / (math.pi * (1.75 / 2.0) ** 2)
    )


def test_resin_override_takes_precedence_over_derived_e_per_mm():
    params = ResinProcessParams(
        layer_height_mm=0.4,
        extrusion_scale=1.5,
        feed_mm_s=8.0,
        e_per_mm_override=0.25,
    )

    assert params.e_per_mm() == 0.25


def test_default_process_params_match_current_material_setup():
    params = ProcessParams()

    assert RESIN_FIXED_BEAD_WIDTH_MM == 2.0
    assert RESIN_FILAMENT_DIAMETER_MM == 1.75
    assert RESIN_FILAMENT_LENGTH_PER_MM3 == pytest.approx(
        1.0 / (math.pi * (1.75 / 2.0) ** 2)
    )
    assert not hasattr(params.resin, "bead_width_mm")
    assert params.resin.layer_height_mm == 0.5
    assert params.resin.extrusion_scale == 1.0
    assert params.resin.feed_mm_s == 10.0
    assert params.resin.first_layer_feed_mm_s == 10.0
    assert params.resin.temperature_c == 250.0
    assert params.resin.fan_enabled is True
    assert params.fiber.layer_height_mm == 0.1
    assert params.fiber.extrusion_scale == 1.0
    assert params.fiber.feed_mm_s == 10.0
    assert params.fiber.first_layer_feed_mm_s == 10.0
    assert params.fiber.temperature_c == 250.0
    assert params.fiber.fan_enabled is True
    assert params.fiber.e_per_mm() == 1.0
    assert params.resin.prime_length_mm == 18.0
    assert params.resin.prime_speed_mm_s == 15.0
    assert params.resin.retract_length_mm == 15.0
    assert params.resin.retract_speed_mm_s == 30.0
    assert params.fiber.prime_length_mm == 12.0
    assert params.fiber.prime_speed_mm_s == 5.0
    assert params.fiber.retract_length_mm == 10.0
    assert params.fiber.retract_speed_mm_s == 5.0
    assert params.fiber.start_accel_s == 2.0
    assert params.travel_feed_mm_s == 10.0
    assert params.first_layer_travel_feed_mm_s == 10.0
    assert params.prime_settle_s == pytest.approx(0.5)
    assert params.start_x_mm == 0.0
    assert params.start_y_mm == 0.0
    assert params.corner_angle_deg == 45.0
    assert params.corner_retreat_ratio == 0.65
    assert params.spline_max_error_mm == 0.1
    assert params.spline_max_angle_deg == 45.0
    assert params.source_merge_distance_mm == 0.04
    assert params.corner_retreat_max_mm == 0.4
    assert params.corner_blend_segments == 8


def test_resin_line_width_is_not_part_of_user_configurable_model():
    params = ProcessParams()

    assert not hasattr(params.resin, "bead_width_mm")
    assert params.resin.e_per_mm() == (
        RESIN_FIXED_BEAD_WIDTH_MM
        * params.resin.layer_height_mm
        * params.resin.extrusion_scale
        * RESIN_FILAMENT_LENGTH_PER_MM3
    )


def test_fiber_geometry_fields_are_not_part_of_active_process_model():
    params = ProcessParams()

    assert not hasattr(params.fiber, "diameter_mm")
    assert not hasattr(params.fiber, "line_width_mm")
    assert params.fiber.e_per_mm() == params.fiber.extrusion_scale


def test_primeline_process_params_defaults_are_stable():
    params = ProcessParams()
    assert params.primeline_x_mm == 0.0
    assert params.primeline_y_mm == -10.0
    assert params.primeline_length_mm == 100.0


def test_prime_settle_s_accepts_zero():
    params = ProcessParams(prime_settle_s=0.0)

    assert params.prime_settle_s == pytest.approx(0.0)


def test_prime_settle_s_rejects_negative_values():
    with pytest.raises(ValueError, match=r"^prime_settle_s must be >= 0$"):
        ProcessParams(prime_settle_s=-0.001)


def test_process_params_preserves_pre_prime_settle_positional_dt_slot():
    params = ProcessParams(
        ResinProcessParams(),
        FiberProcessParams(),
        11.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        -7.0,
        80.0,
        0.125,
    )

    assert params.dt == pytest.approx(0.125)
    assert params.prime_settle_s == pytest.approx(0.5)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="positive-infinity"),
        pytest.param(-math.inf, id="negative-infinity"),
    ],
)
def test_prime_settle_s_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match=r"^prime_settle_s must be >= 0$"):
        ProcessParams(prime_settle_s=value)
