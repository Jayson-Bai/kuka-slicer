from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, box

from external_npz_preprocessor.source_npz import LayerPaths, MaterialPath, SourceJob, TravelPath
from kuka_slicer.gcode_legacy_postprocess import apply_legacy_resin_optimization
from kuka_slicer.slicer import SliceConfig
from kuka_slicer.stl_io import Mesh


def _mesh() -> Mesh:
    return Mesh(np.asarray([[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]]))


def _path(first, last, start_e):
    return MaterialPath(
        "R",
        0,
        np.asarray([first, last], dtype=np.float64),
        np.asarray([start_e, start_e + 1.0], dtype=np.float64),
    )


def _terminal_loop_path() -> MaterialPath:
    """A 0.213-mm-radius, 300-degree terminal Prusa-like loop."""

    xy = np.asarray([
        [0.0, 0.0], [3.0, 0.0], [3.1845, 0.1065], [3.1845, 0.3195],
        [3.0, 0.426], [2.8155, 0.3195], [2.8155, 0.1065],
    ])
    points = np.column_stack((xy, np.full(len(xy), 0.5), np.zeros((len(xy), 3))))
    return MaterialPath("R", 0, points, np.linspace(0.0, 6.0, len(points)))


def test_gcode_source_uses_legacy_infill_connector_after_parse(monkeypatch):
    source = SourceJob(
        meta={
            "path_roles": {"R": {"0": ["infill", "infill", "infill"]}},
            "motion_order": {"0": [
                {"kind": "deposit", "index": 0},
                {"kind": "travel", "index": 0},
                {"kind": "deposit", "index": 1},
                {"kind": "travel", "index": 1},
                {"kind": "deposit", "index": 2},
            ]},
        },
        layers=[LayerPaths(0, [
            _path([0, 0, 0.5, 0, 0, 0], [4, 0, 0.5, 0, 0, 0], 0.0),
            _path([4, 1, 0.5, 0, 0, 0], [0, 1, 0.5, 0, 0, 0], 1.0),
            _path([0, 2, 0.5, 0, 0, 0], [4, 2, 0.5, 0, 0, 0], 2.0),
        ])],
    )
    monkeypatch.setattr(
        "kuka_slicer.gcode_legacy_postprocess.solid_geometry_at_z",
        lambda *_args: box(-1, -1, 5, 3),
    )

    optimized = apply_legacy_resin_optimization(
        source,
        _mesh(),
        SliceConfig(slicing_kernel="prusa", infill_pattern="zigzag_horizontal", infill_density=100.0),
    )

    layer = optimized.layers[0]
    assert len(layer.resin_paths) < 3
    assert optimized.meta["resin_source"] == "prusa_gcode_legacy_postprocess"
    assert optimized.meta["motion_order"]["0"] == [{"kind": "deposit", "index": 0}]
    assert np.all(np.diff(layer.resin_paths[0].extrusion) >= 0.0)


def test_gcode_source_rebuilds_legacy_travel_around_a_hole(monkeypatch):
    source = SourceJob(
        meta={
            "path_roles": {"R": {"0": ["infill", "infill"]}},
            "motion_order": {"0": [
                {"kind": "deposit", "index": 0},
                {"kind": "travel", "index": 0},
                {"kind": "deposit", "index": 1},
            ]},
        },
        layers=[LayerPaths(0, [
            _path([1, 5, 0.5, 0, 0, 0], [3, 5, 0.5, 0, 0, 0], 0.0),
            _path([7, 5, 0.5, 0, 0, 0], [9, 5, 0.5, 0, 0, 0], 1.0),
        ])],
    )
    solid = box(0, 0, 10, 10).difference(box(4, 3, 6, 7))
    monkeypatch.setattr(
        "kuka_slicer.gcode_legacy_postprocess.solid_geometry_at_z",
        lambda *_args: solid,
    )

    optimized = apply_legacy_resin_optimization(
        source,
        _mesh(),
        SliceConfig(slicing_kernel="prusa", infill_pattern="triangles", infill_density=100.0),
    )

    route = optimized.layers[0].travel_paths[-1].points
    assert len(route) > 2
    assert solid.covers(LineString(route[:, :2]))


def test_gcode_source_trims_only_infill_terminal_loop_and_rebuilds_travel(monkeypatch):
    loop_path = _terminal_loop_path()
    next_path = _path([8, 0, 0.5, 0, 0, 0], [9, 0, 0.5, 0, 0, 0], 7.0)
    source = SourceJob(
        meta={
            "path_roles": {"R": {"0": ["infill", "outer_contour"]}},
            "motion_order": {"0": [
                {"kind": "deposit", "index": 0},
                {"kind": "travel", "index": 0},
                {"kind": "deposit", "index": 1},
            ]},
        },
        layers=[LayerPaths(0, [loop_path, next_path], [], [
            TravelPath(0, np.asarray([
                [2.8155, 0.1065, 0.5, 0, 0, 0], [8.0, 0.0, 0.5, 0, 0, 0],
            ])),
        ])],
    )
    monkeypatch.setattr(
        "kuka_slicer.gcode_legacy_postprocess.solid_geometry_at_z",
        lambda *_args: box(-1, -1, 10, 2),
    )

    optimized = apply_legacy_resin_optimization(
        source, _mesh(), SliceConfig(slicing_kernel="prusa", infill_pattern="triangles")
    )

    layer = optimized.layers[0]
    trimmed = layer.resin_paths[0]
    assert np.array_equal(trimmed.points, loop_path.points[:2])
    assert np.array_equal(trimmed.extrusion, loop_path.extrusion[:2])
    assert np.array_equal(layer.resin_paths[1].points, next_path.points)
    assert optimized.meta["path_roles"]["R"]["0"] == ["infill", "outer_contour"]
    assert optimized.meta["prusa_terminal_infill_loop_trim_layers"] == [0]
    assert optimized.meta["motion_order"]["0"] == [
        {"kind": "deposit", "index": 0},
        {"kind": "travel", "index": 0},
        {"kind": "deposit", "index": 1},
    ]
    route = layer.travel_paths[0].points
    assert np.allclose(route[0, :3], trimmed.points[-1, :3])
    assert np.allclose(route[-1, :3], next_path.points[0, :3])
    assert np.all(np.diff(trimmed.extrusion) >= 0.0)


def test_gcode_source_terminal_loop_travel_routes_around_hole(monkeypatch):
    loop_path = _terminal_loop_path()
    next_path = _path([7, 0, 0.5, 0, 0, 0], [8, 0, 0.5, 0, 0, 0], 7.0)
    source = SourceJob(
        meta={
            "path_roles": {"R": {"0": ["infill", "outer_contour"]}},
            "motion_order": {"0": [
                {"kind": "deposit", "index": 0},
                {"kind": "travel", "index": 0},
                {"kind": "deposit", "index": 1},
            ]},
        },
        layers=[LayerPaths(0, [loop_path, next_path], [], [
            TravelPath(0, np.asarray([
                [2.8155, 0.1065, 0.5, 0, 0, 0], [7.0, 0.0, 0.5, 0, 0, 0],
            ])),
        ])],
    )
    solid = box(-1, -2, 10, 2).difference(box(4, -1, 6, 1))
    monkeypatch.setattr(
        "kuka_slicer.gcode_legacy_postprocess.solid_geometry_at_z",
        lambda *_args: solid,
    )

    optimized = apply_legacy_resin_optimization(
        source, _mesh(), SliceConfig(slicing_kernel="prusa", infill_pattern="triangles")
    )

    route = optimized.layers[0].travel_paths[0].points
    assert len(route) > 2
    assert solid.covers(LineString(route[:, :2]))


def test_gcode_source_applies_brim_one_stroke_after_parse(monkeypatch):
    source = SourceJob(
        meta={
            "path_roles": {"R": {"0": ["brim", "brim"]}},
            "motion_order": {"0": [
                {"kind": "deposit", "index": 0},
                {"kind": "travel", "index": 0},
                {"kind": "deposit", "index": 1},
            ]},
        },
        layers=[LayerPaths(0, [
            _path([0, 0, 0.5, 0, 0, 0], [2, 0, 0.5, 0, 0, 0], 0.0),
            _path([2, 1, 0.5, 0, 0, 0], [0, 1, 0.5, 0, 0, 0], 1.0),
        ])],
    )
    monkeypatch.setattr(
        "kuka_slicer.gcode_legacy_postprocess._connect_brim_paths_one_stroke",
        lambda *_args: [np.asarray([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=np.float64)],
    )

    optimized = apply_legacy_resin_optimization(
        source,
        _mesh(),
        SliceConfig(slicing_kernel="prusa", brim_one_stroke=True),
    )

    layer = optimized.layers[0]
    assert len(layer.resin_paths) == 1
    assert optimized.meta["path_roles"]["R"]["0"] == ["brim"]
    assert optimized.meta["motion_order"]["0"] == [{"kind": "deposit", "index": 0}]
    assert not layer.travel_paths
    assert np.all(np.diff(layer.resin_paths[0].extrusion) >= 0.0)
