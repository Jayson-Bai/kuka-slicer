from __future__ import annotations

import importlib

import numpy as np
import pytest
from shapely.geometry import Polygon

def test_bridge_info_reports_a_stable_native_availability_contract() -> None:
    bridge = importlib.import_module("kuka_slicer.prusa_bridge")

    info = bridge.bridge_info()

    assert set(info) == {"available", "reason", "native_version"}
    assert isinstance(info["available"], bool)
    assert isinstance(info["reason"], str)
    assert info["native_version"] is None or isinstance(info["native_version"], str)


def test_require_native_explains_missing_compiled_extension() -> None:
    bridge = importlib.import_module("kuka_slicer.prusa_bridge")
    if bridge.bridge_info()["available"]:
        pytest.skip("compiled Prusa bridge is available")

    with pytest.raises(bridge.PrusaBridgeUnavailable, match="not available"):
        bridge.require_native()


def test_native_bridge_slices_a_unit_cube_to_one_expolygon() -> None:
    bridge = importlib.import_module("kuka_slicer.prusa_bridge")
    native = bridge.require_native()
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )

    layers = native.slice_expolygons(vertices, faces, [0.5])

    assert len(layers) == 1
    assert len(layers[0]) == 1
    expolygon = layers[0][0]
    assert expolygon["holes"] == []
    assert Polygon(expolygon["outer"]).area == pytest.approx(1.0)


def test_native_bridge_preserves_holes_when_stl_vertices_are_unshared() -> None:
    """STL triangles often repeat vertices; the bridge must rebuild topology."""

    bridge = importlib.import_module("kuka_slicer.prusa_bridge")
    native = bridge.require_native()
    triangles = _hollow_box_triangles()
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(vertices.shape[0], dtype=np.int32).reshape(-1, 3)

    layer = native.slice_expolygons(vertices, faces, [0.5])[0]

    assert len(layer) == 1
    assert len(layer[0]["holes"]) == 1
    assert Polygon(layer[0]["holes"][0]).area == pytest.approx(16.0)


def test_native_bridge_emits_print_and_travel_paths_with_extrusion() -> None:
    """The full Prusa path kernel owns both deposited and travel geometry."""

    bridge = importlib.import_module("kuka_slicer.prusa_bridge")
    native = bridge.require_native()
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [8.0, 8.0, 0.0], [0.0, 8.0, 0.0],
            [0.0, 0.0, 1.0], [8.0, 0.0, 1.0], [8.0, 8.0, 1.0], [0.0, 8.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )

    result = native.slice_print_paths(
        vertices,
        faces,
        layer_height=0.5,
        line_width=0.8,
        perimeter_count=2,
        infill_density=100.0,
        infill_pattern="rectilinear",
        fill_angle_schedule=[0.0],
        perimeter_infill_overlap=2.0,
    )

    assert result["layers"]
    assert result["layers"][0]["paths"]
    assert result["layers"][0]["extrusion"]
    assert result["layers"][0]["travel"]
    assert result["layers"][0]["motions"]
    assert {motion["kind"] for motion in result["layers"][0]["motions"]} >= {
        "deposit",
        "travel",
    }
    for extrusion in result["layers"][0]["extrusion"]:
        assert extrusion[-1] >= extrusion[0]


def test_native_bridge_emits_raft_as_a_distinct_deposition_role() -> None:
    bridge = importlib.import_module("kuka_slicer.prusa_bridge")
    native = bridge.require_native()
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [8.0, 8.0, 0.0], [0.0, 8.0, 0.0],
            [0.0, 0.0, 1.0], [8.0, 0.0, 1.0], [8.0, 8.0, 1.0], [0.0, 8.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )

    result = native.slice_print_paths(
        vertices,
        faces,
        layer_height=0.5,
        line_width=0.8,
        perimeter_count=2,
        infill_density=100.0,
        infill_pattern="rectilinear",
        raft_layers=3,
        raft_expansion=3.0,
        raft_first_layer_density=80.0,
        raft_first_layer_expansion=3.0,
        raft_contact_distance=0.25,
    )

    assert result["layers"]
    assert any("raft" in layer["roles"] for layer in result["layers"])


def test_native_bridge_accepts_advanced_geometry_and_path_controls() -> None:
    """The public native bridge exposes the advanced Prusa controls."""

    bridge = importlib.import_module("kuka_slicer.prusa_bridge")
    native = bridge.require_native()
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [8.0, 8.0, 0.0], [0.0, 8.0, 0.0],
            [0.0, 0.0, 1.0], [8.0, 0.0, 1.0], [8.0, 8.0, 1.0], [0.0, 8.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )

    result = native.slice_print_paths(
        vertices,
        faces,
        layer_height=0.5,
        line_width=0.8,
        perimeter_count=2,
        infill_density=100.0,
        infill_pattern="rectilinear",
        perimeter_generator="classic",
        gap_fill_enabled=False,
        infill_anchor=1.0,
        infill_anchor_max=2.0,
        external_perimeter_width=0.8,
        perimeter_width=0.8,
        infill_width=0.8,
        xy_size_compensation=0.05,
        elephant_foot_compensation=0.02,
        avoid_crossing_max_detour=4.0,
        seam_position="rear",
    )

    assert result["layers"]


def _hollow_box_triangles() -> np.ndarray:
    """Return a 10×10×1 rectangular tube with a 4×4 through-hole."""

    outer = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    inner = np.asarray([[3, 3], [7, 3], [7, 7], [3, 7]], dtype=np.float32)
    bottom_outer = np.column_stack((outer, np.zeros(4, dtype=np.float32)))
    top_outer = np.column_stack((outer, np.ones(4, dtype=np.float32)))
    bottom_inner = np.column_stack((inner, np.zeros(4, dtype=np.float32)))
    top_inner = np.column_stack((inner, np.ones(4, dtype=np.float32)))

    triangles: list[np.ndarray] = []

    def add_quad(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> None:
        triangles.extend((np.asarray((a, b, c), dtype=np.float32), np.asarray((a, c, d), dtype=np.float32)))

    for index in range(4):
        following = (index + 1) % 4
        # Top and bottom annular faces.
        add_quad(top_outer[index], top_outer[following], top_inner[following], top_inner[index])
        add_quad(bottom_outer[following], bottom_outer[index], bottom_inner[index], bottom_inner[following])
        # Outer and inner walls have opposite normals.
        add_quad(bottom_outer[index], bottom_outer[following], top_outer[following], top_outer[index])
        add_quad(bottom_inner[following], bottom_inner[index], top_inner[index], top_inner[following])

    return np.asarray(triangles, dtype=np.float32)
