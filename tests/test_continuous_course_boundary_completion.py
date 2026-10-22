"""Boundary half-cells survive designer, material adaptation and final Core export."""
import json

import numpy as np
import pytest

from kuka_slicer.conformal_lattice.continuous_course import build_continuous_course_plan
from kuka_slicer.conformal_lattice.contracts import load_conformal_lattice_spec
from kuka_slicer.surface_preview.server import planar_lattice_config_payload
from kuka_slicer.ui_server import _SlicerUiHandler


def _config():
    return planar_lattice_config_payload({
        "part_length_mm": ["150"], "part_width_mm": ["60"],
        "part_height_mm": ["1.5"], "wall_width_mm": ["2"],
        "base_cell_size_mm": ["8"], "specimen_variant": ["bending"],
        "continuous_course_phase": ["centered"],
    })


def _boundary_fragments(plan):
    return [p for p in plan.paths_xy if np.isclose(p[0, 1], p[-1, 1])
            and (np.isclose(p[0, 1], 0) or np.isclose(p[0, 1], 60))]


def test_8mm_boundary_half_cells_survive_clipping_without_losing_through_courses():
    plan = build_continuous_course_plan(load_conformal_lattice_spec(_config()))
    fragments = _boundary_fragments(plan)
    assert len(plan.paths_xy) == 18
    assert len(fragments) == 12
    assert sum(np.isclose(p[0, 1], 0) for p in fragments) == 6
    assert sum(np.isclose(p[0, 1], 60) for p in fragments) == 6
    assert sum(np.isclose(p[0, 0], 0) and np.isclose(p[-1, 0], 150)
               for p in plan.paths_xy) == 6
    lengths = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in fragments]
    assert lengths == pytest.approx([14.270925303824825] * 12)
    for p in plan.paths_xy:
        assert np.isfinite(p).all()
        assert np.all(p >= [0, 0]) and np.all(p <= [150, 60])
        assert np.all(np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-8)
    for p in fragments:
        assert p.shape == (4, 2)
        assert p[1, 1] == pytest.approx(p[2, 1])
        assert min(p[1, 1], 60 - p[1, 1]) == pytest.approx(2.2153903091734737)


def test_8mm_resin_and_fiber_boundary_paths_reach_core_npz_and_preview(tmp_path):
    config = _config()
    plan = build_continuous_course_plan(load_conformal_lattice_spec(config))
    handler = object.__new__(_SlicerUiHandler)
    handler.server_output_dir = tmp_path
    result = handler._handle_conformal_slice("", request_data=({
        "core_resin_layer_height": ["0.5"],
        "conformal_fiber_enabled": ["true"],
        "core_primeline_enabled": ["false"],
        "prusa_brim_enabled": ["false"],
        "core_part_position_x": ["75"], "core_part_position_y": ["30"],
        "core_cut_wait": ["0.004"], "core_cut_lift": ["1"],
        "core_tool_safe_lift": ["1"],
    }, {"conformal_spec": ("boundary_8mm.json", json.dumps(config).encode())}))

    report = result["fiber_reinforcement"]
    assert result["layers"] == 3
    assert report["resin_honeycomb_paths_per_layer"] == 18
    assert report["paths_per_fiber_layer"] == 18
    assert report["resin_layer_indices"] == [1]
    assert report["total_fiber_path_count"] == 18
    preview = result["preview"]
    assert preview["preview_source"] == "final_core_npz"
    layer = next(layer for layer in preview["layers"] if layer["index"] == 1)
    assert len(layer["fiber_paths"]) == 18
    assert len(layer["fiber_cut_events"]) == 18

    job_dir = tmp_path / result["download_url"].split("/")[-2]
    with np.load(job_dir / "planar_honeycomb_core.npz", allow_pickle=False) as core:
        # Existing scalar-column protocol, tool IDs and embedded injection
        # catalog remain consumable; each short fiber stroke still gets a cut.
        assert len({len(core[k]) for k in ("seq", "x", "y", "z", "e", "tool_id",
                                          "path_id", "path_end_flag", "layer_index")}) == 1
        manifest = json.loads(core["core_injection_manifest"].item())
        cuts = [b for b in manifest["blocks"] if b["kind"] == "cut"]
        assert len(cuts) == 18
        assert all(b["layer"] == 1 and b["tool_id"] == 1 for b in cuts)
        moving_print = (
            (core["layer_index"] == 1) & np.isin(core["move_type"], [1, 3])
            & (core["event_flag"] == 0) & (core["core_injection_role"] == 0)
            & np.r_[False, np.diff(core["e"]) > 0]
            & np.r_[False, np.hypot(np.diff(core["x"]), np.diff(core["y"])) > 1e-7]
        )
        for tool, z in ((2, .75), (1, .85)):
            selected = moving_print & (core["tool_id"] == tool)
            np.testing.assert_allclose(core["z"][selected], z, atol=1e-6)
            xy = np.c_[core["x"], core["y"]]
            boundary_ids = set()
            for p in _boundary_fragments(plan):
                midpoint = (p[1] + p[2]) * .5
                near = selected & (np.linalg.norm(xy - midpoint, axis=1) < .1)
                assert near.any(), (tool, midpoint)
                boundary_ids.update(core["path_id"][near].tolist())
            assert len(boundary_ids) == 12
