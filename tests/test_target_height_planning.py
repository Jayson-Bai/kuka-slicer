"""User-entered height is the nearest final part height, for all four groups."""
import json

import numpy as np
import pytest

from kuka_slicer.conformal_lattice.contracts import load_conformal_lattice_spec
from kuka_slicer.conformal_lattice.pipeline import _physical_layer_schedule
from kuka_slicer.surface_preview.server import (
    conformal_lattice_config_payload,
    planar_lattice_config_payload,
    surface_payload,
)
from kuka_slicer.ui_server import _SlicerUiHandler


def config_for(curved, target=8):
    build = conformal_lattice_config_payload if curved else planar_lattice_config_payload
    return build({k: [str(v)] for k, v in dict(
        part_length_mm=150, part_width_mm=60, part_height_mm=target,
        specimen_variant="bending", base_cell_size_mm=10,
        continuous_course_phase="bending_zigzag_midline",
        amplitude_mm=1.5, surface_parameter_mode="tensile_centered_wave_count",
        wave_count_x=2.5, wave_count_y=0.5, samples_x=49, samples_y=49,
        surface_start_layer=3, surface_start_layer_semantics="first_nonzero_curvature_physical",
    ).items()})


@pytest.mark.parametrize("curved", [False, True])
@pytest.mark.parametrize("fiber", [False, True])
def test_eight_mm_target_reaches_core_for_all_four_groups(tmp_path, curved, fiber):
    config = config_for(curved)
    # Old and even malformed informational plans cannot override the target.
    config["fiber_aware_resin_only_height_plan"] = {"fiber_enabled_reference_height_mm": 99}
    config["target_height_reference_plan"] = {"resin_only": {"planned_final_height_mm": 99}}
    handler = object.__new__(_SlicerUiHandler)
    handler.server_output_dir = tmp_path
    result = handler._handle_conformal_slice("", request_data=(
        {"core_resin_layer_height": ["0.5"], "core_fiber_layer_height": ["0.1"],
         "conformal_fiber_enabled": [str(fiber).lower()]},
        {"conformal_spec": ("target_8mm.json", json.dumps(config).encode("utf-8"))},
    ))
    expected_count, expected_height = (14, 8.2) if fiber else (16, 8.0)
    assert result["layers"] == expected_count
    plan = result["height_plan"]
    assert plan["target_final_height_mm"] == 8
    assert plan["planned_final_height_mm"] == pytest.approx(expected_height)
    assert plan["height_error_mm"] == pytest.approx(expected_height - 8)
    assert plan["fiber_layer_count"] == (12 if fiber else 0)
    assert result["nominal_final_height_mm"] == pytest.approx(expected_height)
    filename = "conformal_lattice_core.npz" if curved else "planar_honeycomb_core.npz"
    directory = tmp_path / result["download_url"].split("/")[-2]
    with np.load(directory / filename, allow_pickle=False) as output:
        resin_print = (output["event_flag"] == 0) & (output["tool_id"] == 2) & (output["move_type"] == 1)
        # Core stores bead centres. The top extent adds half a resin layer.
        assert float(output["z"][resin_print].max()) + 0.25 == pytest.approx(expected_height, abs=2e-3)


@pytest.mark.parametrize("curved", [False, True])
@pytest.mark.parametrize("target,count", [(8.1, 16), (8.25, 16), (8.26, 17), (8.49, 17)])
def test_resin_rounding_uses_complete_layers_and_lower_ties(curved, target, count):
    spec = load_conformal_lattice_spec(config_for(curved, target))
    actual, z, plan = _physical_layer_schedule(spec, None, physical_layer_height_mm=.5)
    assert actual == count
    assert np.diff(z) == pytest.approx([.5] * (count - 1))
    assert z[-1] + .25 == pytest.approx(count * .5)
    assert plan["planned_final_height_mm"] == count * .5


@pytest.mark.parametrize("curved", [False, True])
def test_fiber_ties_and_active_preset_replanning(curved):
    spec = load_conformal_lattice_spec(config_for(curved, 7.9))
    count, _, plan = _physical_layer_schedule(spec, None, physical_layer_height_mm=.5,
        fiber_layer_height_mm=.1, plan_for_continuous_fiber=True)
    # 13*.5+11*.1=7.6 and 14*.5+12*.1=8.2 are equally distant.
    assert count == 13
    assert plan["planned_final_height_mm"] == pytest.approx(7.6)
    spec = load_conformal_lattice_spec(config_for(curved))
    count, _, plan = _physical_layer_schedule(spec, None, physical_layer_height_mm=.4,
        fiber_layer_height_mm=.1, plan_for_continuous_fiber=True)
    assert count == 16
    assert plan["planned_final_height_mm"] == pytest.approx(7.8)


def test_nonintegral_target_preview_has_no_partial_top_layer():
    result = surface_payload({"part_height_mm": ["8.25"], "surface_start_layer": ["3"],
        "surface_start_layer_semantics": ["first_nonzero_curvature_physical"]})
    stack = result["solid_stack"]
    assert stack["target_final_height_mm"] == 8.25
    assert stack["final_height_mm"] == 8
    assert len(stack["layers"]) == 16
    assert stack["layers"][-1]["base_z_mm"] == pytest.approx(7.75)
