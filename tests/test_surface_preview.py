from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from kuka_slicer.surface_preview.model import DoubleSineSurface
from kuka_slicer.surface_preview.server import (
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
    assert 'id="amplitude_mm"' in html
    assert 'id="wavelength_x_mm"' in html
    assert 'id="canvas"' in html
    assert 'id="importStl"' in html
    assert 'id="exportConfig"' in html
    assert 'id="exportConformalConfig"' in html
    assert 'id="wall_width_mm"' in html
    assert 'id="base_cell_size_mm"' in html
    assert 'id="surface_start_layer"' in html
    assert 'id="samples_x"' in html
    assert 'id="samples_y"' in html
    assert "updateConformalDesignSummary" in html
    assert "不会写入共形格栅参数" in html
    assert '/api/stl-domain' in html
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
            "layer_height_mm": ["0.5"],
            "amplitude_mm": ["1.5"],
            "wavelength_x_mm": ["30"],
            "wavelength_y_mm": ["40"],
            "wall_width_mm": ["2.0"],
            "base_cell_size_mm": ["5.0"],
            "surface_start_layer": ["3"],
            "samples_x": ["31"],
            "samples_y": ["29"],
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
    assert config["part"] == {"boundary": "rectangle", "length_mm": 150.0, "width_mm": 100.0, "final_height_mm": 10.0}
    assert config["manufacturing"] == {"layer_height_mm": 0.5, "nominal_bead_width_mm": 2.0}
    assert config["lattice"]["wall_width_mm"] == pytest.approx(2.0)
    assert config["lattice"]["wall_bead_count"] == 1
    assert config["lattice"]["base_cell_size_mm"] == pytest.approx(5.0)
    assert config["fill_field"] == {"mode": "fixed_cell_size", "drivers": []}
    assert config["layer_embedding"]["surface_start_layer"] == 3


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"wall_width_mm": ["4"], "base_cell_size_mm": ["4"]}, "nominal fill ratio"),
        ({"surface_start_layer": ["-1"]}, "surface_start_layer"),
        ({"samples_x": ["1"]}, "samples_x"),
        ({"samples_y": ["513"]}, "samples_y"),
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
