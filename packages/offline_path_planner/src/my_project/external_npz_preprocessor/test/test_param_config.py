from pathlib import Path
import math

import pytest

from external_npz_preprocessor.param_config import (
    default_print_params_path,
    load_print_params,
    save_print_params,
)
from external_npz_preprocessor.process_params import FiberProcessParams, ProcessParams, ResinProcessParams


def test_save_and_load_print_params_round_trip(tmp_path):
    path = tmp_path / "print_params.json"
    params = ProcessParams(
        resin=ResinProcessParams(
            layer_height_mm=0.6,
            extrusion_scale=1.2,
            feed_mm_s=11.0,
            temperature_c=245.0,
            fan_enabled=False,
            prime_length_mm=19.0,
            prime_speed_mm_s=16.0,
            retract_length_mm=14.0,
            retract_speed_mm_s=28.0,
            first_layer_feed_mm_s=4.0,
        ),
        fiber=FiberProcessParams(
            layer_height_mm=0.12,
            extrusion_scale=1.3,
            feed_mm_s=9.0,
            temperature_c=255.0,
            fan_enabled=False,
            prime_length_mm=13.0,
            prime_speed_mm_s=6.0,
            retract_length_mm=9.0,
            retract_speed_mm_s=4.0,
            start_accel_s=4.5,
            first_layer_feed_mm_s=3.0,
        ),
        travel_feed_mm_s=12.0,
        first_layer_travel_feed_mm_s=8.0,
        default_a=1.0,
        default_b=2.0,
        default_c=3.0,
        start_x_mm=50.0,
        start_y_mm=60.0,
        primeline_x_mm=3.0,
        primeline_y_mm=-12.0,
        primeline_length_mm=80.0,
        prime_settle_s=0.75,
    )

    save_print_params(params, path)
    loaded = load_print_params(path)

    assert loaded == params


def test_default_print_params_path_uses_data_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "ignored_config"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "ignored_appdata"))

    assert default_print_params_path(data_root=tmp_path) == (
        tmp_path / "external_npz_preprocessor" / "print_params.json"
    )


def test_legacy_fiber_geometry_fields_are_ignored_on_load(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"params":{"fiber":{"diameter_mm":0.6,"line_width_mm":1.0,"layer_height_mm":0.11}}}',
        encoding="utf-8",
    )

    params = load_print_params(path)

    assert params.fiber.layer_height_mm == 0.11
    assert not hasattr(params.fiber, "diameter_mm")
    assert not hasattr(params.fiber, "line_width_mm")


def test_legacy_resin_bead_width_is_ignored_on_load(tmp_path):
    path = tmp_path / "legacy_resin.json"
    path.write_text(
        '{"params":{"resin":{"bead_width_mm":99.0,"layer_height_mm":0.4,"extrusion_scale":2.0}}}',
        encoding="utf-8",
    )

    params = load_print_params(path)

    assert not hasattr(params.resin, "bead_width_mm")
    assert params.resin.layer_height_mm == 0.4
    assert params.resin.e_per_mm() == pytest.approx(
        2.0 * 0.4 * 2.0 / (math.pi * (1.75 / 2.0) ** 2)
    )


def test_legacy_spline_defaults_are_migrated_to_current_safe_defaults(tmp_path):
    path = tmp_path / "legacy_spline.json"
    path.write_text(
        '{"params":{"corner_angle_deg":10.0,"corner_retreat_ratio":0.2}}',
        encoding="utf-8",
    )

    params = load_print_params(path)

    assert params.corner_angle_deg == 45.0
    assert params.corner_retreat_ratio == 0.65
    assert params.spline_max_error_mm == 0.1
    assert params.spline_max_angle_deg == 45.0
    assert params.source_merge_distance_mm == 0.04
    assert params.corner_retreat_max_mm == 0.4
    assert params.corner_blend_segments == 8


def test_previous_external_npz_corner_retreat_default_is_migrated(tmp_path):
    path = tmp_path / "previous_spline.json"
    path.write_text(
        '{"params":{"corner_angle_deg":45.0,"corner_retreat_ratio":0.25,"spline_max_error_mm":0.1}}',
        encoding="utf-8",
    )

    params = load_print_params(path)

    assert params.corner_angle_deg == 45.0
    assert params.corner_retreat_ratio == 0.65
    assert params.spline_max_error_mm == 0.1
    assert params.corner_blend_segments == 8


def test_legacy_print_params_without_prime_settle_s_uses_default(tmp_path):
    path = tmp_path / "legacy_prime_settle.json"
    path.write_text(
        '{"params":{"fiber":{"prime_length_mm":8.0},"travel_feed_mm_s":12.5,'
        '"start_x_mm":30.0,"start_y_mm":40.0}}',
        encoding="utf-8",
    )

    params = load_print_params(path)

    assert params.fiber.prime_length_mm == pytest.approx(8.0)
    assert params.travel_feed_mm_s == pytest.approx(12.5)
    assert params.start_x_mm == pytest.approx(30.0)
    assert params.start_y_mm == pytest.approx(40.0)
    assert params.prime_settle_s == pytest.approx(0.5)


def test_legacy_print_speeds_become_first_layer_defaults(tmp_path):
    path = tmp_path / "legacy_first_layer_speeds.json"
    path.write_text(
        '{"params":{"resin":{"feed_mm_s":7.5},"fiber":{"feed_mm_s":6.5},'
        '"travel_feed_mm_s":9.5}}',
        encoding="utf-8",
    )

    params = load_print_params(path)

    assert params.resin.feed_mm_s == pytest.approx(7.5)
    assert params.resin.first_layer_feed_mm_s == pytest.approx(7.5)
    assert params.fiber.feed_mm_s == pytest.approx(6.5)
    assert params.fiber.first_layer_feed_mm_s == pytest.approx(6.5)
    assert params.travel_feed_mm_s == pytest.approx(9.5)
    assert params.first_layer_travel_feed_mm_s == pytest.approx(9.5)


def test_negative_prime_settle_s_in_json_is_rejected(tmp_path):
    path = tmp_path / "negative_prime_settle.json"
    path.write_text(
        '{"params":{"prime_settle_s":-0.001}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"^prime_settle_s must be >= 0$"):
        load_print_params(path)
