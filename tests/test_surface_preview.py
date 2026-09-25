from __future__ import annotations

import math
import json
import shutil
import struct
import subprocess
import textwrap

import numpy as np
import pytest

from kuka_slicer.surface_preview.model import DoubleSineSurface
from kuka_slicer.conformal_lattice.contracts import load_conformal_lattice_spec
from kuka_slicer.surface_preview.server import (
    _load_designer_state,
    _save_designer_state,
    conformal_lattice_config_payload,
    graded_surface_config_payload,
    surface_payload,
    surface_preview_html,
)
from kuka_slicer.surface_preview.stl_domain import stl_projection_domain_from_bytes


def _box_stl_bytes(width: float = 10.0, height: float = 8.0, depth: float = 2.0) -> bytes:
    vertices = [
        (0.0, 0.0, 0.0), (width, 0.0, 0.0), (width, height, 0.0), (0.0, height, 0.0),
        (0.0, 0.0, depth), (width, 0.0, depth), (width, height, depth), (0.0, height, depth),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    data = bytearray(b"surface preview test".ljust(80, b" "))
    data.extend(struct.pack("<I", len(faces)))
    for face in faces:
        values = [0.0, 0.0, 0.0, *(coordinate for index in face for coordinate in vertices[index])]
        data.extend(struct.pack("<12fH", *values, 0))
    return bytes(data)


def test_continuous_course_preview_anchors_nine_mm_cells_and_routes_fibre_through_grips():
    """Exercise the actual preview geometry for the user's 150 x 50 tensile case."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the embedded preview geometry")

    preview_html = surface_preview_html()
    script = textwrap.dedent(
        """
        const source = require('fs').readFileSync(0, 'utf8');
        const begin = source.indexOf('    function continuousCoursePreview()');
        const end = source.indexOf('    function offsetContinuousCourse', begin);
        if (begin < 0 || end < 0) throw new Error('continuous-course preview seam not found');
        const buildPreview = (edgeLength) => new Function('edgeLength', `
          const values = { base_cell_size_mm: edgeLength, wall_width_mm: 2 };
          const positiveNumber = (id) => values[id] ?? null;
          const honeycombActiveXBounds = () => [25, 125];
          const payload = { coordinate_system: { xy_bounds_mm: [0, 0, 150, 50] } };
          let continuousCoursePreviewCache = null;
          ${source.slice(begin, end)}
          return continuousCoursePreview();
        `)(edgeLength);
        const preview = buildPreview(9);
        const resizedPreview = buildPreview(11);
        const centralPore = preview.pores.some((pore) => {
          const centreX = pore.reduce((sum, point) => sum + point[0], 0) / pore.length;
          const centreY = pore.reduce((sum, point) => sum + point[1], 0) / pore.length;
          return Math.abs(centreX - 75) < 1e-8 && Math.abs(centreY - 25) < 1e-8;
        });
        const poreCentres = preview.pores.map((pore) => [
          pore.reduce((sum, point) => sum + point[0], 0) / pore.length,
          pore.reduce((sum, point) => sum + point[1], 0) / pore.length,
        ]);
        const columnsByRow = new Map();
        poreCentres.forEach(([x, y]) => {
          const key = y.toFixed(6);
          if (!columnsByRow.has(key)) columnsByRow.set(key, []);
          columnsByRow.get(key).push(x.toFixed(6));
        });
        const phaseRows = [...columnsByRow.entries()]
          .sort((first, second) => Number(first[0]) - Number(second[0]))
          .map(([, row]) => row);
        const diagonalColumnOffset = 9.0 * 1.5 + 2.0 / Math.sqrt(3.0);
        const columnPitch = 2.0 * diagonalColumnOffset;
        const phaseOffsets = [...new Set(phaseRows.map((row) => {
          const offset = (Number(row[0]) - Number(phaseRows[0][0]) + columnPitch) % columnPitch;
          return offset.toFixed(6);
        }))].map(Number).sort((first, second) => first - second);
        const twoPhasePoreColumns = phaseOffsets.length === 2
          && Math.abs(phaseOffsets[0]) < 1e-6
          && Math.abs(phaseOffsets[1] - diagonalColumnOffset) < 1e-6;
        const rowCentres = [...columnsByRow.keys()].map(Number).sort((first, second) => first - second);
        const interleavedRowStep = Math.min(...rowCentres.slice(1).map((centre, index) => centre - rowCentres[index]));
        const hexHalfHeight = Math.sqrt(3.0) * 9.0 * 0.5;
        const horizontalOpeningMm = preview.rowPitch - 2.0 * hexHalfHeight;
        const diagonalOpeningMm = Math.sqrt(3.0) * 0.5 * (phaseOffsets[1] - 9.0 * 1.5)
          + 0.5 * (interleavedRowStep - hexHalfHeight);
        const uniformPoreEdges = preview.pores.every((pore) => pore.every((point, index) => {
          const next = pore[(index + 1) % pore.length];
          return Math.abs(Math.hypot(next[0] - point[0], next[1] - point[1]) - 9.0) < 1e-8;
        }));
        const cross = (a, b, c) => (
          (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        );
        const pointInside = (point, polygon) => {
          let inside = false;
          polygon.forEach((start, index) => {
            const end = polygon[(index + 1) % polygon.length];
            if ((start[1] > point[1]) !== (end[1] > point[1])) {
              const crossingX = (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0];
              if (point[0] < crossingX) inside = !inside;
            }
          });
          return inside;
        };
        const pointToSegmentDistance = (point, start, end) => {
          const dx = end[0] - start[0];
          const dy = end[1] - start[1];
          const t = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)));
          return Math.hypot(point[0] - start[0] - dx * t, point[1] - start[1] - dy * t);
        };
        let minimumClearance = Infinity;
        let pointInsidePore = false;
        const plannedSegments = new Set();
        let duplicateSegment = false;
        preview.courses.forEach((course) => course.points.slice(1).forEach((point, pointIndex) => {
          const startPoint = course.points[pointIndex];
          for (let sample = 0; sample <= 20; sample += 1) {
            const t = sample / 20;
            const sampledPoint = [
              startPoint[0] + (point[0] - startPoint[0]) * t,
              startPoint[1] + (point[1] - startPoint[1]) * t,
            ];
            if (sampledPoint[0] < 0 || sampledPoint[0] > 150 || sampledPoint[1] < 0 || sampledPoint[1] > 50) continue;
            preview.pores.forEach((pore) => {
              pointInsidePore ||= pointInside(sampledPoint, pore);
              pore.forEach((edgeStart, index) => {
                minimumClearance = Math.min(minimumClearance, pointToSegmentDistance(sampledPoint, edgeStart, pore[(index + 1) % pore.length]));
              });
            });
          }
        }));
        const properIntersection = (a, b, c, d) => (
          cross(a, b, c) * cross(a, b, d) < -1e-9
          && cross(c, d, a) * cross(c, d, b) < -1e-9
        );
        let pathsCross = false;
        preview.courses.forEach((firstCourse, firstIndex) => {
          preview.courses.slice(firstIndex + 1).forEach((secondCourse) => {
            firstCourse.points.slice(1).forEach((firstEnd, firstSegmentIndex) => {
              secondCourse.points.slice(1).forEach((secondEnd, secondSegmentIndex) => {
                pathsCross ||= properIntersection(
                  firstCourse.points[firstSegmentIndex], firstEnd,
                  secondCourse.points[secondSegmentIndex], secondEnd,
                );
              });
            });
          });
        });
        preview.courses.forEach((course) => course.points.slice(1).forEach((point, index) => {
          const start = course.points[index];
          const first = `${start[0].toFixed(6)},${start[1].toFixed(6)}`;
          const second = `${point[0].toFixed(6)},${point[1].toFixed(6)}`;
          const key = first < second ? `${first}|${second}` : `${second}|${first}`;
          duplicateSegment ||= plannedSegments.has(key);
          plannedSegments.add(key);
        }));
        const inclinedSupports = new Set();
        let repeatedInclinedSupport = false;
        preview.courses.forEach((course) => course.points.slice(1).forEach((end, index) => {
          const start = course.points[index];
          const dx = end[0] - start[0];
          const dy = end[1] - start[1];
          if (dx <= 1e-7 || Math.abs(Math.abs(dy / dx) - Math.sqrt(3.0)) > 1e-7) return;
          const sign = dy >= 0 ? 1 : -1;
          const key = `${sign}:${(start[1] - sign * Math.sqrt(3.0) * start[0]).toFixed(6)}`;
          repeatedInclinedSupport ||= inclinedSupports.has(key);
          inclinedSupports.add(key);
        }));
        console.log(JSON.stringify({
          anchor: preview.latticeAnchorMm,
          centralPore,
          twoPhasePoreColumns,
          uniformPoreEdges,
          courseCount: preview.courses.length,
          spansBothPartSides: preview.courses.every((course) => course.points[0][0] <= 0 && course.points.at(-1)[0] >= 150),
          courseClipBounds: preview.courseClipBounds,
          monotoneX: preview.courses.every((course) => course.points.slice(1).every((point, index) => point[0] >= course.points[index][0] - 1e-9)),
          minimumSegmentLength: Math.min(...preview.courses.flatMap((course) => course.points.slice(1).map((point, index) => Math.hypot(point[0] - course.points[index][0], point[1] - course.points[index][1])))),
          openingLaneSpacing: [0, 2, 4].map((index) => preview.courses[index + 1].points[0][1] - preview.courses[index].points[0][1]),
          horizontalOpeningMm,
          diagonalOpeningMm,
          horizontalWallClearanceMm: (horizontalOpeningMm - 2.0) * 0.5,
          openingLanes: preview.courses.map((course) => course.opening?.lane),
          minimumClearance,
          pointInsidePore,
          duplicateSegment,
          repeatedInclinedSupport,
          pathsCross,
          resizedAnchor: resizedPreview.latticeAnchorMm,
          parentSpan: [preview.parentBounds[2] - preview.parentBounds[0], preview.parentBounds[3] - preview.parentBounds[1]],
        }));
        """
    )
    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        input=preview_html,
    )
    geometry = json.loads(result.stdout)

    assert geometry["anchor"] == pytest.approx([75.0, 25.0])
    # The redesigned parent is centred as a whole; the centre is the middle
    # horizontal opening rather than a pore centre so that a 50 mm specimen
    # contains three complete 4 mm openings.
    assert geometry["centralPore"] is False
    # Main and interleaved pores use two stable X phases whose separation is
    # the analytic 2 mm diagonal channel.
    assert geometry["twoPhasePoreColumns"] is True
    assert geometry["uniformPoreEdges"] is True
    assert geometry["courseCount"] == 6
    assert geometry["spansBothPartSides"] is True
    assert geometry["courseClipBounds"] == pytest.approx([25.0, 0.0, 125.0, 50.0])
    assert geometry["monotoneX"] is True
    assert geometry["minimumSegmentLength"] >= 2.0 - 1e-7
    assert geometry["openingLaneSpacing"] == pytest.approx([2.0, 2.0, 2.0])
    assert geometry["horizontalOpeningMm"] == pytest.approx(4.0)
    assert geometry["horizontalWallClearanceMm"] == pytest.approx(1.0)
    assert geometry["diagonalOpeningMm"] == pytest.approx(2.0)
    assert geometry["openingLanes"] == ["lower", "upper"] * 3
    assert geometry["pointInsidePore"] is False
    assert geometry["minimumClearance"] >= 1.0 - 1e-7
    assert geometry["duplicateSegment"] is False
    assert geometry["repeatedInclinedSupport"] is False
    assert geometry["pathsCross"] is False
    assert geometry["resizedAnchor"] == pytest.approx([75.0, 25.0])
    assert geometry["parentSpan"] == pytest.approx([300.0, 300.0])


def test_designer_state_persists_across_preview_server_restarts(tmp_path):
    state_path = tmp_path / "KukaSlicer" / "conformal_designer_state_v1.json"
    state = {
        "part_length_mm": "150",
        "part_width_mm": "50",
        "grip_end_length_mm": "25",
        "wave_count_x": "1.5",
        "inspection_enabled": False,
    }

    _save_designer_state(state_path, state)

    assert _load_designer_state(state_path) == state


def test_double_sine_surface_matches_the_documented_height_field():
    surface = DoubleSineSurface(
        amplitude_mm=2.0,
        wavelength_x_mm=20.0,
        wavelength_y_mm=40.0,
        phase_x_rad=0.0,
        phase_y_rad=0.0,
        z_reference_mm=1.5,
    )

    assert surface.height(5.0, 10.0) == pytest.approx(3.5)
    assert surface.height(0.0, 10.0) == pytest.approx(1.5)


def test_double_sine_surface_reports_analytical_maximum_slope():
    surface = DoubleSineSurface(amplitude_mm=1.0, wavelength_x_mm=20.0, wavelength_y_mm=40.0)
    grid = surface.sample_grid(width_mm=20.0, height_mm=40.0, samples=41)

    assert grid.max_slope == pytest.approx(math.pi / 10.0)
    assert grid.summary()["z_range_mm"] == pytest.approx(2.0)


def test_surface_grid_uses_a_centred_mm_domain():
    grid = DoubleSineSurface().sample_grid(width_mm=60.0, height_mm=20.0, samples=3)

    assert np.allclose(grid.x[0], [-30.0, 0.0, 30.0])
    assert np.allclose(grid.y[:, 0], [-10.0, 0.0, 10.0])


def test_surface_payload_contains_surface_grid_and_diagnostics():
    payload = surface_payload(
        {
            "amplitude_mm": ["1.2"],
            "wavelength_x_mm": ["30"],
            "wavelength_y_mm": ["50"],
            "width_mm": ["90"],
            "height_mm": ["60"],
            "samples": ["9"],
        }
    )

    assert payload["surface"]["type"] == "double_sine_product"
    assert payload["domain"] == {
        "width_mm": 90.0,
        "height_mm": 60.0,
        "samples": 9,
        "mode": "rectangle",
        "projection": None,
    }
    assert len(payload["grid"]["z"]) == 9
    assert len(payload["grid"]["z"][0]) == 9
    assert payload["statistics"]["z_range_mm"] > 0.0


def test_conformal_rectangle_preview_uses_the_exported_lower_left_origin():
    payload = surface_payload(
        {
            "width_mm": ["90"],
            "height_mm": ["60"],
            "samples": ["9"],
        },
        rectangle_origin_lower_left=True,
    )

    assert payload["domain"]["mode"] == "rectangle"
    assert payload["grid"]["x"][0][0] == pytest.approx(0.0)
    assert payload["grid"]["x"][0][-1] == pytest.approx(90.0)
    assert payload["grid"]["y"][0][0] == pytest.approx(0.0)
    assert payload["grid"]["y"][-1][0] == pytest.approx(60.0)


def test_conformal_preview_api_samples_the_center_and_reports_its_coordinate_contract():
    payload = surface_payload(
        {
            "width_mm": ["150"],
            "height_mm": ["100"],
            "part_height_mm": ["10"],
            "surface_start_layer": ["3"],
            "amplitude_mm": ["1.5"],
            "wavelength_x_mm": ["100"],
            "wavelength_y_mm": ["200"],
            "phase_x_pi": ["1"],
            "phase_y_pi": ["0"],
            "check_x_mm": ["75"],
            "check_y_mm": ["50"],
            "samples": ["49"],
        },
        rectangle_origin_lower_left=True,
    )

    assert payload["preview_version"] == "surface_preview_v2"
    assert payload["export_version"] == "conformal_lattice_spec_v1"
    assert payload["coordinate_system"] == {
        "origin_label": "rectangle_lower_left",
        "xy_bounds_mm": [0.0, 0.0, 150.0, 100.0],
    }
    assert payload["grid"]["x"][24][24] == pytest.approx(75.0)
    assert payload["grid"]["y"][24][24] == pytest.approx(50.0)
    assert payload["grid"]["z"][24][24] == pytest.approx(1.5)
    assert payload["statistics"]["z_min_mm"] == pytest.approx(-1.5)
    assert payload["statistics"]["z_max_mm"] == pytest.approx(1.5)
    assert payload["inspection_point"]["height_mm"] == pytest.approx(1.5)
    assert payload["inspection_point"]["slope"] == pytest.approx(0.0, abs=1e-12)


def test_disabled_inspection_point_does_not_constrain_a_smaller_rectangle():
    payload = surface_payload(
        {
            "width_mm": ["40"],
            "height_mm": ["30"],
            "part_height_mm": ["10"],
            "inspection_enabled": ["false"],
            # Stale bending coordinates must be ignored when inspection is off.
            "check_x_mm": ["75"],
            "check_y_mm": ["50"],
        },
        rectangle_origin_lower_left=True,
    )

    assert payload["inspection_point"] is None
    assert payload["solid_stack"] is not None
    assert payload["solid_stack"]["section_y_mm"] == pytest.approx(15.0)


def test_tensile_wave_count_strategy_scales_with_the_rectangle_and_locks_its_own_phase_rule():
    payload = surface_payload(
        {
            "width_mm": ["180"],
            "height_mm": ["80"],
            "amplitude_mm": ["1.5"],
            "surface_parameter_mode": ["tensile_centered_wave_count"],
            "wave_count_x": ["1.5"],
            "wave_count_y": ["0.5"],
            "inspection_enabled": ["false"],
            "samples": ["9"],
        },
        rectangle_origin_lower_left=True,
    )

    assert payload["surface_parameterization"] == {
        "mode": "tensile_centered_wave_count",
        "wave_count_x": 1.5,
        "wave_count_y": 0.5,
        "phase_policy": "gauge_center_positive_peak_and_boundary_zero",
    }
    surface = payload["surface"]
    assert surface["wavelength_x_mm"] == pytest.approx(120.0)
    assert surface["wavelength_y_mm"] == pytest.approx(160.0)
    assert surface["phase_x_rad"] == pytest.approx(math.pi)
    assert surface["phase_y_rad"] == pytest.approx(0.0)
    assert payload["grid"]["z"][4][4] == pytest.approx(1.5)
    assert payload["grid"]["z"][4][0] == pytest.approx(0.0, abs=1e-12)


def test_tensile_wave_count_strategy_rejects_non_half_integer_wave_counts():
    with pytest.raises(ValueError, match="wave_count_x must be a positive half-integer"):
        surface_payload(
            {
                "surface_parameter_mode": ["tensile_centered_wave_count"],
                "wave_count_x": ["1.2"],
                "wave_count_y": ["1.5"],
            }
        )


def test_conformal_solid_stack_reuses_the_symmetric_smoothstep_layer_progression():
    payload = surface_payload(
        {
            "width_mm": ["150"],
            "height_mm": ["100"],
            "part_height_mm": ["10"],
            "surface_start_layer": ["3"],
            "z_reference_mm": ["0.25"],
            "check_x_mm": ["75"],
            "check_y_mm": ["50"],
            "samples": ["49"],
        },
        rectangle_origin_lower_left=True,
    )

    stack = payload["solid_stack"]
    assert stack is not None
    assert stack["reference_layer_height_mm"] == pytest.approx(0.5)
    assert stack["surface_start_layer"] == 3
    assert stack["surface_return_layer"] == 16
    assert stack["peak_layer_indices"] == [9, 10]
    assert stack["representative_peak_layer_index"] == 9
    assert len(stack["layers"]) == 20
    assert stack["layers"][0]["alpha"] == pytest.approx(0.0)
    assert stack["layers"][-1]["alpha"] == pytest.approx(0.0)
    assert stack["layers"][9]["alpha"] == pytest.approx(1.0)
    assert stack["layers"][10]["alpha"] == pytest.approx(1.0)
    assert stack["layers"][0]["xz_points"][0][1] == pytest.approx(0.5)
    assert stack["layers"][9]["base_z_mm"] == pytest.approx(4.75)


def test_designer_first_nonzero_physical_layer_semantics_map_to_the_legacy_progression_boundary():
    payload = surface_payload(
        {
            "width_mm": ["150"],
            "height_mm": ["50"],
            "part_height_mm": ["10"],
            "surface_start_layer": ["3"],
            "surface_start_layer_semantics": ["first_nonzero_curvature_physical"],
        },
        rectangle_origin_lower_left=True,
    )

    stack = payload["solid_stack"]
    assert stack is not None
    assert stack["surface_start_layer"] == 1
    assert stack["first_nonzero_curvature_layer_physical"] == 3
    assert stack["layers"][1]["alpha"] == pytest.approx(0.0)
    assert stack["layers"][2]["alpha"] > 0.0
    assert stack["layers"][18]["alpha"] == pytest.approx(0.0)

    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["50"],
            "part_height_mm": ["10"],
            "surface_start_layer": ["3"],
            "surface_start_layer_semantics": ["first_nonzero_curvature_physical"],
        }
    )
    assert config["layer_embedding"]["surface_start_layer"] == 1
    assert config["layer_embedding"]["first_nonzero_curvature_layer_physical"] == 3


def test_surface_payload_converts_designer_pi_multiples_to_internal_radians():
    payload = surface_payload(
        {
            "phase_x_pi": ["1"],
            "phase_y_pi": ["0.5"],
        }
    )

    assert payload["surface"]["phase_x_rad"] == pytest.approx(math.pi)
    assert payload["surface"]["phase_y_rad"] == pytest.approx(math.pi / 2.0)


def test_surface_payload_keeps_legacy_radian_query_compatibility():
    payload = surface_payload(
        {
            "phase_x_rad": [str(math.pi / 4.0)],
            "phase_y_rad": [str(-math.pi / 2.0)],
        }
    )

    assert payload["surface"]["phase_x_rad"] == pytest.approx(math.pi / 4.0)
    assert payload["surface"]["phase_y_rad"] == pytest.approx(-math.pi / 2.0)


@pytest.mark.parametrize(
    "params, error",
    [
        ({"wavelength_x_mm": ["0"]}, "wavelength_x_mm must be positive"),
        ({"samples": ["7"]}, "samples must be in the range"),
        ({"amplitude_mm": ["nan"]}, "amplitude_mm must be finite"),
    ],
)
def test_surface_payload_rejects_invalid_input(params, error):
    with pytest.raises(ValueError, match=error):
        surface_payload(params)


def test_surface_preview_html_has_an_independent_surface_api_and_controls():
    html = surface_preview_html()

    assert 'fetch(`/api/surface?' in html
    assert 'id="surface_parameter_mode"' in html
    assert 'id="specimen_variant"' in html
    assert 'value="bending">弯曲版：全长蜂窝工作段' in html
    assert 'value="tensile_centered_wave_count" selected' in html
    assert 'id="wave_count_x"' in html
    assert 'id="wave_count_y"' in html
    assert 'id="applyTensilePreset"' in html
    assert 'id="amplitude_mm"' in html
    assert 'id="wavelength_x_mm"' in html
    assert 'id="phase_x_pi"' in html
    assert 'id="phase_y_pi"' in html
    assert 'aria-describedby="phasePiHint"' in html
    assert '输入 π 的倍数：1 表示 π，0.5 表示 π/2，1.5 表示 3π/2' in html
    assert 'id="phase_x_rad"' not in html
    assert 'surface.phase_x_rad' in html
    assert '矩形左下角 (0, 0)' in html
    assert 'id="check_x_mm"' in html
    assert 'id="check_y_mm"' in html
    assert 'id="inspection_enabled" type="checkbox"' in html
    assert 'function syncInspectionPointControls()' in html
    assert 'id="previewMode"' in html
    assert '实体层叠 / XZ 剖面' in html
    assert 'id="surfaceZScale"' in html
    assert 'id="sectionZScale"' in html
    assert '视觉 Z 放大只影响画布' in html
    assert 'const centeredX = x - (bounds[0] + bounds[2]) * 0.5;' in html
    assert 'const uniformScale = Math.min(' in html
    assert '连续路径蜂窝：' in html
    assert 'function drawLatticePreview' in html
    assert 'function continuousCoursePreview' in html
    assert 'id="exportConformalConfig">导出连续路径 JSON</button>' in html
    assert 'id="exportConformalConfig" disabled' not in html
    assert '连续路径目前仅在本页生成和验证' not in html
    assert 'id="latticeLengthSummary"' in html
    assert 'function updateLatticeLengthSummary' in html
    assert '当前平面连续路径：' in html
    assert '一个完整黄色孔洞中心固定在蜂窝工作区中心' in html
    assert 'function appendProjectedClippedPore' in html
    assert '红线为 2 mm 连续纤维的中心线预览' in html
    assert '夹持分界树脂带内侧截断' in html
    assert '红色为孔间通道中心线，裁断后每段独立制造' in html
    assert 'function offsetContinuousCourse' in html
    assert '导出连续路径 JSON' in html
    assert 'function drawFiberDoubleWallPreview' not in html
    assert 'function appendProjectedFiberTrack' not in html
    assert 'function drawSurfaceReferenceFrame' in html
    assert 'function drawSurfaceGuideMesh' in html
    assert 'function physicalPreviewLayer' in html
    assert 'function physicalLayerZ' in html
    assert 'const baseZ = 0;' in html
    assert 'Z=0 基准面' in html
    assert 'α=1 完整曲率层（物理 Z）' in html
    assert 'function surfaceLighting' in html
    assert "const designerStateKey = 'kuka-slicer.conformal-designer-state.v1';" in html
    assert 'function saveDesignerState()' in html
    assert 'function restoreDesignerState()' in html
    assert "fetch('/api/designer-state'" in html
    assert 'function restorePersistentDesignerState()' in html
    assert 'id="samples_x" type="number" min="2" max="512" step="1" value="49"' in html
    assert 'id="samples_y" type="number" min="2" max="512" step="1" value="49"' in html
    assert 'id="samples" type="number" min="8" max="120" step="1" value="49"' in html
    assert '${{' not in html
    assert 'id="canvas"' in html
    assert 'id="exportConformalConfig"' in html
    assert 'id="part_length_mm"' in html
    assert 'id="part_width_mm"' in html
    assert 'id="part_height_mm"' in html
    assert 'id="grip_end_length_mm"' in html
    assert 'function honeycombActiveXBounds()' in html
    assert '曲面仍按完整零件 X/Y 范围计算' in html
    assert 'id="layer_height_mm"' not in html
    assert 'id="wall_width_mm"' in html
    assert 'id="base_cell_size_mm"' in html
    assert 'id="align_load_line" type="checkbox"' in html
    assert 'id="align_load_line" type="checkbox" checked' not in html
    assert 'syncLoadLineAlignmentControls' in html
    assert 'id="honeycomb_align_x" type="checkbox" checked' not in html
    assert 'id="honeycomb_align_y" type="checkbox" checked' not in html
    assert 'id="honeycomb_align_x_mm"' in html
    assert 'id="honeycomb_align_y_mm"' in html
    assert 'centreHoneycombAlignment' in html
    assert '一个完整黄色孔洞中心固定在蜂窝工作区中心' in html
    assert 'id="phase_origin_x_mm"' not in html
    assert '自动避让' in html
    assert 'id="surface_start_layer"' in html
    assert '首个非零曲率层（物理层）' in html
    assert '连续纤维层接口（路径待定义）' not in html
    assert 'id="fiber_first_after_resin_layer"' not in html
    assert 'id="fiber_last_after_resin_layer"' not in html
    assert 'id="fiber_reinforcement_enabled"' not in html
    assert 'id="fiber_symmetric_path_count"' not in html
    assert 'id="fiber_grip_anchor_length_mm"' not in html
    assert "surface_start_layer_semantics', 'first_nonzero_curvature_physical'" in html
    assert 'id="samples_x"' in html
    assert 'id="samples_y"' in html
    assert "updateConformalDesignSummary" in html
    assert '蜂窝网格共形设计器' in html
    assert '外边界固定为矩形' in html
    assert 'id="stlFile"' not in html
    assert '/api/stl-domain' not in html
    assert '/api/export-conformal-lattice-config' in html
    assert "canvas.addEventListener('lostpointercapture', endDrag)" in html


def test_stl_projection_domain_uses_stl_xy_min_as_the_local_origin():
    domain = stl_projection_domain_from_bytes(
        _box_stl_bytes(), file_name="honeycomb.stl", build_axis="z"
    )

    assert domain.width_mm == pytest.approx(10.0)
    assert domain.height_mm == pytest.approx(8.0)
    preview = domain.preview_payload()
    assert preview["file_name"] == "honeycomb.stl"
    outer = np.asarray(preview["polygons"][0]["outer"])
    assert np.min(outer[:, 0]) == pytest.approx(0.0)
    assert np.min(outer[:, 1]) == pytest.approx(0.0)

    payload = surface_payload({"samples": ["8"]}, domain)
    assert payload["domain"]["mode"] == "stl_projection"
    assert payload["grid"]["x"][0][0] == pytest.approx(0.0)
    assert payload["grid"]["y"][0][0] == pytest.approx(0.0)
    assert len(payload["grid"]["material_mask"]) == 7
    assert len(payload["grid"]["material_mask"][0]) == 7
    assert all(all(row) for row in payload["grid"]["material_mask"])

    compact = surface_payload({"samples": ["8"]}, domain, include_projection_geometry=False)
    assert "polygons" not in compact["domain"]["projection"]


def test_exported_surface_config_binds_the_surface_to_the_imported_stl_domain():
    domain = stl_projection_domain_from_bytes(_box_stl_bytes(), file_name="part.stl")
    config = graded_surface_config_payload({"amplitude_mm": ["1.5"]}, domain)

    assert config["format"] == "graded_surface_v1"
    assert config["coordinate_system"]["origin"] == "stl_xy_min"
    assert config["domain"]["source"]["file_name"] == "part.stl"
    assert config["domain"]["source"]["sha256"] == domain.sha256
    assert config["surface"]["amplitude_mm"] == pytest.approx(1.5)
    assert "progression" not in config
    assert "printability" not in config


def test_legacy_surface_export_ignores_conformal_design_inputs():
    domain = stl_projection_domain_from_bytes(_box_stl_bytes(), file_name="part.stl")

    config = graded_surface_config_payload(
        {
            "amplitude_mm": ["1.5"],
            "wall_width_mm": ["2.0"],
            "base_cell_size_mm": ["5.0"],
            "surface_start_layer": ["3"],
            "samples_x": ["48"],
            "samples_y": ["48"],
        },
        domain,
    )

    assert config["format"] == "graded_surface_v1"
    assert config["surface"]["amplitude_mm"] == pytest.approx(1.5)
    assert "lattice" not in config
    assert "layer_embedding" not in config


def test_conformal_lattice_export_binds_double_sine_to_a_rectangular_physical_part_without_stl():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["100"],
            "part_height_mm": ["10"],
            "layer_height_mm": ["0.25"],
            "amplitude_mm": ["1.5"],
            "wavelength_x_mm": ["30"],
            "wavelength_y_mm": ["40"],
            "phase_x_pi": ["1"],
            "phase_y_pi": ["0.5"],
            "wall_width_mm": ["2.0"],
            "base_cell_size_mm": ["5.0"],
            "surface_start_layer": ["3"],
            "samples_x": ["31"],
            "samples_y": ["29"],
            "boundary_mode": ["inset"],
            "phase_origin_x_mm": ["1.25"],
            "phase_origin_y_mm": ["-0.5"],
            "orientation_angle_deg": ["30"],
            "align_load_line": ["false"],
            "random_seed": ["17"],
            "samples": ["8"],
        },
    )

    assert config["format"] == "conformal_lattice_spec_v1"
    source = config["source_surface"]
    assert source["provider"] == "double_sine"
    assert source["domain"] == "outer_boundary_only"
    assert "reference_stl" not in source
    assert source["double_sine"]["xy_bounds_mm"] == [0.0, 0.0, 150.0, 100.0]
    assert source["double_sine"]["samples"] == [31, 29]
    assert source["double_sine"]["phase_x_rad"] == pytest.approx(math.pi)
    assert source["double_sine"]["phase_y_rad"] == pytest.approx(math.pi / 2.0)
    assert config["part"] == {"boundary": "rectangle", "length_mm": 150.0, "width_mm": 100.0, "final_height_mm": 10.0}
    assert config["manufacturing"] == {"layer_height_mm": 0.5, "nominal_bead_width_mm": 2.0}
    assert config["lattice"]["wall_width_mm"] == pytest.approx(2.0)
    assert config["lattice"]["wall_bead_count"] == 1
    assert config["lattice"]["base_cell_size_mm"] == pytest.approx(5.0)
    assert config["lattice"]["boundary_mode"] == "inset"
    assert config["lattice"]["phase_origin"] == [1.25, -0.5]
    assert config["lattice"]["boundary_phase_policy"] == "auto_avoid_outer_boundary_coincidence"
    assert config["lattice"]["load_line_alignment"] == {
        "enabled": False,
        "axis": "x",
        "position": "part_length_midplane",
        "feature": "wall",
    }
    assert config["fill_field"] == {"mode": "fixed_cell_size", "drivers": []}
    assert config["orientation_field"]["angle_deg"] == pytest.approx(30.0)
    assert config["layer_embedding"]["surface_start_layer"] == 3
    assert config["random_seed"] == 17


def test_conformal_lattice_export_resolves_tensile_wave_counts_without_an_inspection_point():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["180"],
            "part_width_mm": ["80"],
            "part_height_mm": ["10"],
            "surface_parameter_mode": ["tensile_centered_wave_count"],
            "amplitude_mm": ["1.5"],
            "wave_count_x": ["1.5"],
            "wave_count_y": ["0.5"],
            "inspection_enabled": ["false"],
            "check_x_mm": ["999"],
            "check_y_mm": ["999"],
        }
    )

    surface = config["source_surface"]["double_sine"]
    assert surface["wavelength_x_mm"] == pytest.approx(120.0)
    assert surface["wavelength_y_mm"] == pytest.approx(160.0)
    assert surface["phase_x_rad"] == pytest.approx(math.pi)
    assert surface["phase_y_rad"] == pytest.approx(0.0)
    assert config["lattice"]["load_line_alignment"]["enabled"] is False


def test_conformal_lattice_export_keeps_the_surface_global_and_records_a_geometry_only_grip_range():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["50"],
            "part_height_mm": ["10"],
            "specimen_variant": ["tensile"],
            "grip_end_length_mm": ["25"],
            "surface_parameter_mode": ["tensile_centered_wave_count"],
            "wave_count_x": ["1.5"],
            "wave_count_y": ["1.5"],
        }
    )

    assert config["source_surface"]["double_sine"]["xy_bounds_mm"] == [0.0, 0.0, 150.0, 50.0]
    assert config["source_surface"]["double_sine"]["wavelength_x_mm"] == pytest.approx(100.0)
    assert config["part"]["specimen_variant"] == "tensile"
    assert config["part"]["symmetric_grip_end_length_mm"] == 25.0
    assert "active_region" not in config["lattice"]


def test_bending_variant_exports_a_full_rectangle_without_a_grip_field():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["50"],
            "part_height_mm": ["10"],
            "specimen_variant": ["bending"],
            # A stale hidden browser value must not leak into a bending JSON.
            "grip_end_length_mm": ["25"],
            "surface_parameter_mode": ["manual_wavelength_phase"],
            "wavelength_x_mm": ["100"],
            "wavelength_y_mm": ["40"],
            "phase_x_pi": ["1"],
            "phase_y_pi": ["0.5"],
        }
    )

    assert config["part"] == {
        "boundary": "rectangle",
        "length_mm": 150.0,
        "width_mm": 50.0,
        "final_height_mm": 10.0,
        "specimen_variant": "bending",
    }
    assert config["source_surface"]["double_sine"]["wavelength_x_mm"] == pytest.approx(100.0)
    assert config["source_surface"]["double_sine"]["xy_bounds_mm"] == [0.0, 0.0, 150.0, 50.0]


def test_contract_rejects_a_grip_field_on_an_explicit_bending_design():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["50"],
            "part_height_mm": ["10"],
            "specimen_variant": ["bending"],
        }
    )
    config["part"]["symmetric_grip_end_length_mm"] = 25.0

    with pytest.raises(ValueError, match="bending part.specimen_variant must not define"):
        load_conformal_lattice_spec(config)


def test_conformal_lattice_export_disables_bending_only_load_line_alignment_by_default():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["180"],
            "part_width_mm": ["80"],
            "part_height_mm": ["10"],
        }
    )

    assert config["lattice"]["load_line_alignment"] == {
        "enabled": False,
        "axis": "x",
        "position": "part_length_midplane",
        "feature": "wall",
    }
    assert config["orientation_field"]["angle_deg"] == 0.0


def test_conformal_lattice_export_records_independent_honeycomb_x_y_feature_alignment():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["50"],
            "part_height_mm": ["10"],
            "orientation_angle_deg": ["30"],
            "honeycomb_align_x": ["true"],
            "honeycomb_align_x_mm": ["72.5"],
            "honeycomb_align_y": ["false"],
            "honeycomb_align_y_mm": ["25"],
        }
    )

    assert config["orientation_field"]["angle_deg"] == 0.0
    assert config["lattice"]["honeycomb_feature_alignment"] == {
        "align_x": True,
        "align_y": False,
        "target_x_mm": 72.5,
        "target_y_mm": 25.0,
        "x_feature": "y_directed_wall",
        "y_feature": "inclined_edge_zigzag_centerline",
        "scope": "center_features_only",
    }


def test_conformal_lattice_export_defaults_to_49_by_49_source_sampling():
    config = conformal_lattice_config_payload(
        {
            "part_length_mm": ["150"],
            "part_width_mm": ["100"],
            "part_height_mm": ["10"],
            "surface_start_layer": ["3"],
        }
    )

    assert config["source_surface"]["double_sine"]["samples"] == [49, 49]


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"wall_width_mm": ["4"], "base_cell_size_mm": ["4"]}, "nominal fill ratio"),
        ({"surface_start_layer": ["-1"]}, "surface_start_layer"),
        ({"samples_x": ["1"]}, "samples_x"),
        ({"samples_y": ["513"]}, "samples_y"),
        ({"part_length_mm": ["150"], "grip_end_length_mm": ["75"]}, "positive honeycomb working length"),
        ({"grip_end_length_mm": ["-0.1"]}, "non-negative"),
        ({"specimen_variant": ["torsion"]}, "specimen_variant must be tensile or bending"),
        ({"specimen_variant": ["tensile"]}, "requires grip_end_length_mm greater than zero"),
    ],
)
def test_conformal_lattice_export_rejects_invalid_design_inputs(params, error):
    domain = stl_projection_domain_from_bytes(_box_stl_bytes(), file_name="honeycomb.stl")

    with pytest.raises(ValueError, match=error):
        conformal_lattice_config_payload(params)


def test_legacy_export_requires_its_stl_but_conformal_export_does_not():
    domain = stl_projection_domain_from_bytes(
        _box_stl_bytes(), file_name="part.stl", build_axis="x"
    )

    legacy = graded_surface_config_payload({}, domain)
    assert legacy["coordinate_system"]["source_build_axis"] == "x"
    conformal = conformal_lattice_config_payload({"part_length_mm": ["10"], "part_width_mm": ["8"], "part_height_mm": ["2"], "surface_start_layer": ["0"]})
    assert conformal["part"]["boundary"] == "rectangle"


def test_conformal_export_rejects_non_integral_2mm_wall_width():
    with pytest.raises(ValueError, match="integer multiple"):
        conformal_lattice_config_payload({"wall_width_mm": ["3"]})
