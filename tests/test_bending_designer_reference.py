"""The bending baseline is an explicit phase, not a cosmetic fixture overlay."""
import json
import math
import shutil
import subprocess

import numpy as np
import pytest

from kuka_slicer.conformal_lattice.continuous_course import build_continuous_course_plan
from kuka_slicer.conformal_lattice.contracts import load_conformal_lattice_spec
from kuka_slicer.surface_preview.server import (
    conformal_lattice_config_payload,
    planar_lattice_config_payload,
    surface_preview_html,
)


SPAN = 3 * (30 + 4 / math.sqrt(3))


def baseline(**overrides):
    values = dict(
        part_length_mm=150, part_width_mm=60, part_height_mm=8,
        specimen_variant="bending", base_cell_size_mm=10,
        surface_parameter_mode="tensile_centered_wave_count",
        amplitude_mm=1.5, wave_count_x=2.5, wave_count_y=0.5,
        surface_start_layer=3,
        surface_start_layer_semantics="first_nonzero_curvature_physical",
        continuous_course_phase="bending_zigzag_midline", bending_span_mm=SPAN,
    )
    values.update(overrides)
    return {key: [str(value)] for key, value in values.items()}


def contacts(plan, x):
    result = []
    for path in plan.paths_xy:
        for start, end in zip(path[:-1], path[1:]):
            if min(start[0], end[0]) < x < max(start[0], end[0]):
                fraction = (x - start[0]) / (end[0] - start[0])
                result.append((start[1] + fraction * (end[1] - start[1]), fraction))
    return np.asarray(sorted(result))


def test_reference_contacts_and_planar_control_match():
    curved = load_conformal_lattice_spec(conformal_lattice_config_payload(baseline()))
    flat = load_conformal_lattice_spec(planar_lattice_config_payload(baseline()))
    plan = build_continuous_course_plan(curved)
    assert plan.lattice_anchor_mm == pytest.approx((75 - SPAN / 12, 30))
    at_load = contacts(plan, 75)
    assert at_load.shape == (6, 2)
    # The two boundary diagonals have been shortened by Y clipping; their
    # contact is the parent-edge midpoint, not the clipped-fragment midpoint.
    assert at_load[1:-1, 1] == pytest.approx([0.5] * 4)
    assert at_load[:, 0] == pytest.approx([3.349364905, 14.009618943, 24.669872981, 35.330127019, 45.990381057, 56.650635095])
    assert at_load[:, 0] == pytest.approx(60 - at_load[::-1, 0])
    assert np.diff(at_load[:, 0]) == pytest.approx([plan.row_pitch_mm / 2] * 5)
    for x in ((150 - SPAN) / 2, (150 + SPAN) / 2):
        np.testing.assert_allclose(contacts(plan, x)[:, 0], at_load[:, 0])
    flat_plan = build_continuous_course_plan(flat)
    for left, right in zip(plan.paths_xy, flat_plan.paths_xy, strict=True):
        np.testing.assert_allclose(left, right)
    assert curved.part["bending_fixture"]["span_mm"] == pytest.approx(SPAN)


def test_legacy_json_keeps_its_phase_and_span_does_not_move_paths():
    config = conformal_lattice_config_payload(baseline(continuous_course_phase="centered"))
    config["lattice"].pop("continuous_course_phase")
    plan = build_continuous_course_plan(load_conformal_lattice_spec(config))
    assert plan.lattice_anchor_mm == (75, 30)
    first = build_continuous_course_plan(load_conformal_lattice_spec(conformal_lattice_config_payload(baseline())))
    second = build_continuous_course_plan(load_conformal_lattice_spec(conformal_lattice_config_payload(baseline(bending_span_mm=100))))
    for left, right in zip(first.paths_xy, second.paths_xy, strict=True):
        np.testing.assert_allclose(left, right)


@pytest.mark.parametrize("overrides,match", [
    ({"continuous_course_phase": "typo"}, "continuous_course_phase"),
    ({"bending_span_mm": 150}, "span_mm"),
    ({"bending_span_mm": -1}, "bending_span_mm"),
])
def test_invalid_reference_settings_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        conformal_lattice_config_payload(baseline(**overrides))


def test_preview_matches_exported_continuous_geometry():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the actual preview function")
    script = r"""
    const source = require('fs').readFileSync(0, 'utf8');
    const begin = source.indexOf('    function continuousCoursePreview()');
    const end = source.indexOf('    function offsetContinuousCourse', begin);
    const document = { getElementById: (id) => ({ value:
      id === 'specimen_variant' ? 'bending' : 'bending_zigzag_midline' }) };
    const values = { base_cell_size_mm: 10, wall_width_mm: 2 };
    const positiveNumber = (id) => values[id] ?? null;
    const honeycombActiveXBounds = () => [0, 150];
    const payload = { coordinate_system: { xy_bounds_mm: [0, 0, 150, 60] } };
    let continuousCoursePreviewCache = null;
    eval(source.slice(begin, end));
    const plan = continuousCoursePreview();
    console.log(JSON.stringify({ anchor: plan.latticeAnchorMm,
      paths: plan.courses.flatMap((c) => c.fragments) }));
    """
    result = subprocess.run([node, "-e", script], input=surface_preview_html(), text=True, encoding="utf-8", capture_output=True, check=True)
    preview = json.loads(result.stdout)
    plan = build_continuous_course_plan(load_conformal_lattice_spec(conformal_lattice_config_payload(baseline())))
    assert preview["anchor"] == pytest.approx(plan.lattice_anchor_mm)
    # Both implementations emit the same clipped polylines in course order.
    for browser_path, exported_path in zip(preview["paths"], plan.paths_xy, strict=True):
        np.testing.assert_allclose(browser_path, exported_path, atol=1e-7)


def test_reference_defaults_and_state_persistence_are_exposed():
    html = surface_preview_html()
    assert 'value="bending" selected' in html
    assert "wave_count_x: 2.5, wave_count_y: 0.5" in html
    assert "part_width_mm: 60, part_height_mm: 8" in html
    assert "'continuous_course_phase',\n      'bending_span_preview_mm'," in html
    assert "query.set('bending_span_mm'" in html
