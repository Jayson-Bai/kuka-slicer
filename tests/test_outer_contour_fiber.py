"""Designer JSON -> independent perimeter fiber -> unchanged Core contract."""

import copy
import json
from html.parser import HTMLParser

import numpy as np
import pytest

from kuka_slicer.conformal_lattice.contracts import load_conformal_lattice_spec
from kuka_slicer.conformal_lattice.fiber_reinforcement import (
    ContinuousCourseFiberSettings,
    apply_continuous_course_fiber_strategy,
)
from kuka_slicer.conformal_lattice.path_bridge import ExtrusionVolumeModel
from kuka_slicer.conformal_lattice.pipeline import run_conformal_lattice_pipeline
from kuka_slicer.external_npz import external_source_job_to_core_source_job
from kuka_slicer.surface_preview.server import (
    conformal_lattice_config_payload,
    planar_lattice_config_payload,
    surface_preview_html,
)
from kuka_slicer.ui_server import (
    _SlicerUiHandler,
    _conformal_spec_ui_summary,
    _ensure_offline_planner_import_paths,
)


def _config(planar=False, grip=False, enabled=True):
    factory = planar_lattice_config_payload if planar else conformal_lattice_config_payload
    params = {
        "part_length_mm": ["30"], "part_width_mm": ["20"],
        "part_height_mm": ["3"], "wall_width_mm": ["2"],
        "base_cell_size_mm": ["4"], "surface_start_layer": ["1"],
        "amplitude_mm": ["0.4"], "samples_x": ["12"], "samples_y": ["8"],
        "specimen_variant": ["tensile" if grip else "bending"],
        "grip_end_length_mm": ["5"],
    }
    if enabled is not None:
        params["print_outer_contour_fiber"] = [str(enabled).lower()]
    return factory(params)


@pytest.mark.parametrize("planar", [False, True])
def test_optional_json_flag_round_trips_and_rejects_non_boolean(planar):
    default = _config(planar, enabled=None)
    assert default["lattice"]["print_outer_contour_fiber"] is False
    legacy = copy.deepcopy(default)
    del legacy["lattice"]["print_outer_contour_fiber"]
    assert load_conformal_lattice_spec(legacy).lattice.get("print_outer_contour_fiber", False) is False
    for enabled in (False, True):
        encoded = json.dumps(_config(planar, enabled=enabled)).encode()
        assert load_conformal_lattice_spec(encoded).lattice["print_outer_contour_fiber"] is enabled
        assert _conformal_spec_ui_summary(encoded, "design.json")["lattice"]["print_outer_contour_fiber"] is enabled
    for invalid in ("false", "true", 0, 1, None, []):
        default["lattice"]["print_outer_contour_fiber"] = invalid
        with pytest.raises(ValueError, match="print_outer_contour_fiber must be a boolean"):
            load_conformal_lattice_spec(default)


def test_designer_control_is_unchecked_labeled_and_part_of_saved_export_inputs():
    class Inputs(HTMLParser):
        def __init__(self):
            super().__init__()
            self.elements = []

        def handle_starttag(self, tag, attrs):
            self.elements.append((tag, dict(attrs)))

    html = surface_preview_html()
    parser = Inputs()
    parser.feed(html)
    control = next(attrs for tag, attrs in parser.elements
                   if tag == "input" and attrs.get("id") == "print_outer_contour_fiber")
    assert control["type"] == "checkbox" and "checked" not in control
    assert control["aria-describedby"] == "outerContourFiberHint"
    assert any(tag == "label" and attrs.get("for") == "print_outer_contour_fiber"
               for tag, attrs in parser.elements)
    export_ids = next(line for line in html.splitlines() if "const conformalDesignIds =" in line)
    assert "'print_outer_contour_fiber'" in export_ids


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("grip", [False, True])
def test_outline_adds_one_closed_path_per_existing_fiber_layer_without_changing_other_paths(planar, grip):
    run = run_conformal_lattice_pipeline(
        _config(planar, grip), physical_layer_height_mm=.5,
        fiber_layer_height_mm=.1, plan_for_continuous_fiber=True,
        extrusion=ExtrusionVolumeModel(1., 1.), validate_fill_ratio=False,
    )
    plan = run.layer_embedding.report["physical_stack_plan"]
    selected = tuple(plan["fiber_layer_indices"])
    base = run.path_graph.to_external_base_source_job()
    base.meta["physical_stack_plan"] = plan
    jobs, reports = [], []
    for enabled, outline in ((True, False), (True, True), (False, True)):
        job = copy.deepcopy(base)
        report = apply_continuous_course_fiber_strategy(
            source_job=job, graph=run.path_graph,
            course_paths_by_layer=run.continuous_course_paths_by_layer,
            settings=ContinuousCourseFiberSettings(
                enabled, selected[0] + 1, selected[-1] + 1,
                resin_z_preplanned_for_fiber=True, planned_resin_layer_indices=selected,
                print_outer_contour_fiber=outline,
            ), fiber_layer_height_mm=.1, fiber_e_per_mm=1.7,
        )
        jobs.append(job)
        reports.append(report)
    before, after, resin_only = jobs
    assert not any(group.material == "F" for group in resin_only.material_paths)
    assert reports[0].resin_layer_indices == reports[1].resin_layer_indices == selected
    assert reports[1].paths_per_layer == reports[0].paths_per_layer + 1
    assert reports[1].total_path_count == reports[0].total_path_count + len(selected)
    assert reports[1].nominal_final_height_mm == reports[0].nominal_final_height_mm
    added_length = 0.
    for original in before.material_paths:
        group = next(g for g in after.material_paths
                     if (g.material, g.layer_index) == (original.material, original.layer_index))
        offset = int(group.material == "F")
        assert len(group.paths) == len(original.paths) + offset
        for old, new in zip(original.paths, group.paths[offset:]):
            np.testing.assert_array_equal(new, old)
        for old, new in zip(original.extrusion, group.extrusion[offset:]):
            np.testing.assert_array_equal(new, old)
        if not offset:
            continue
        layer = str(group.layer_index)
        resin = next(g for g in after.material_paths if g.material == "R" and g.layer_index == group.layer_index)
        roles = after.meta["path_roles"]["R"][layer]
        expected = resin.paths[roles.index("conformal_outer_boundary")].copy()
        expected[:, 2] += .1
        outline = group.paths[0]
        np.testing.assert_array_equal(outline, expected)
        np.testing.assert_array_equal(outline[0], outline[-1])
        np.testing.assert_allclose(np.ptp(outline[:, :2], axis=0), [30., 20.])
        lengths = np.linalg.norm(np.diff(outline[:, :3], axis=0), axis=1)
        added_length += lengths.sum()
        assert group.extrusion[0][0] == 0.
        np.testing.assert_allclose(np.diff(group.extrusion[0]), lengths * 1.7)
        assert after.meta["path_roles"]["F"][layer][0] == "conformal_outer_boundary"
        records = after.meta["motion_order"][layer]
        assert [r["index"] for r in records if r["kind"] == "fiber_deposit"] == list(range(len(group.paths)))
        connectors = after.meta["fiber_travel_path_indexes"][layer]
        assert len(connectors) == len(group.paths) - 1
        travels = next(g.paths for g in after.travel_paths if g.layer_index == group.layer_index)
        for index, connector in enumerate(connectors):
            np.testing.assert_array_equal(travels[connector][0], group.paths[index][-1])
            # The existing boundary router deduplicates coincident waypoints.
            np.testing.assert_allclose(travels[connector][-1], group.paths[index + 1][0], rtol=0., atol=1e-12)
    assert reports[1].report["total_fiber_length_mm"] - reports[0].report["total_fiber_length_mm"] == pytest.approx(added_length)
    _ensure_offline_planner_import_paths()
    from external_npz_preprocessor.converter import source_job_to_parsed_commands
    from external_npz_preprocessor.process_params import ProcessParams

    core = external_source_job_to_core_source_job(after)
    commands = source_job_to_parsed_commands(core, ProcessParams(primeline_enabled=False))
    fiber_commands = [command for command in commands
                      if ":F:" in (getattr(command, "source_path_id", None) or "")]
    assert len({command.source_path_id for command in fiber_commands}) == reports[1].total_path_count
    assert all(command.subtype == "CONTINUOUS_SOURCE_PRINT" for command in fiber_commands)
    for layer in core.layers:
        if layer.index not in selected:
            assert not layer.fiber_paths
            continue
        original = next(g for g in after.material_paths if g.material == "F" and g.layer_index == layer.index)
        assert len(layer.fiber_paths) == len(original.paths)
        for converted, points, extrusion in zip(layer.fiber_paths, original.paths, original.extrusion):
            np.testing.assert_array_equal(converted.points, points)
            np.testing.assert_array_equal(converted.extrusion, extrusion)


@pytest.mark.parametrize("enabled", [False, True])
def test_designer_json_import_reaches_final_core_with_separate_fiber_cut_events(tmp_path, enabled):
    handler = object.__new__(_SlicerUiHandler)
    handler.server_output_dir = tmp_path
    result = handler._handle_conformal_slice("", request_data=(
        {"core_resin_layer_height": ["0.5"], "conformal_fiber_enabled": ["true"]},
        {"conformal_spec": ("outer-fiber.json", json.dumps(_config(grip=True, enabled=enabled)).encode())},
    ))
    report = result["fiber_reinforcement"]
    assert report["print_outer_contour_fiber"] is enabled
    assert report["outer_contour_fiber_path_count"] == int(enabled) * len(report["resin_layer_indices"])
    assert report["paths_per_fiber_layer"] == report["resin_honeycomb_paths_per_layer"] + int(enabled)
    job_dir = tmp_path / result["download_url"].split("/")[-2]
    with np.load(job_dir / "conformal_lattice_core.npz", allow_pickle=False) as core:
        assert core["x"].size > 0
        assert np.any(core["tool_id"] == 1)
    layers = result["preview"]["layers"]
    assert sum(len(layer["fiber_cut_events"]) for layer in layers) == report["total_fiber_path_count"]
    assert [layer["index"] for layer in layers if layer["fiber_paths"]] == report["resin_layer_indices"]
