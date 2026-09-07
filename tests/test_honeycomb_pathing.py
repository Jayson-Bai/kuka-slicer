from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from kuka_slicer.external_npz import write_external_source_npz
from kuka_slicer.honeycomb_pathing import HoneycombPathingConfig, HoleSafeTravelRouter
from kuka_slicer.honeycomb_pathing.planner import (
    _Edge,
    _endpoint_taper_plan,
    _extrusion_profiles,
    _insert_endpoint_taper_points,
    _minimum_trail_cover,
)
from kuka_slicer.slicer import SliceConfig, slice_mesh_to_job
from kuka_slicer.stl_io import Mesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ROOT = REPOSITORY_ROOT / "packages" / "offline_path_planner"
OFFLINE_SOURCE_ROOTS = (
    OFFLINE_ROOT / "src" / "my_project" / "path_processing_core",
    OFFLINE_ROOT / "src" / "my_project" / "gcode_planner",
    OFFLINE_ROOT / "src" / "my_project" / "external_npz_preprocessor",
)


def test_minimum_trail_cover_uses_each_y_graph_edge_once() -> None:
    center = (0.0, 0.0)
    edges = [
        _Edge(center, (1.0, 0.0), 1.0),
        _Edge(center, (-0.5, 1.0), 1.0),
        _Edge(center, (-0.5, -1.0), 1.0),
    ]

    trails = _minimum_trail_cover(edges)

    # A degree-three junction and three leaves give four odd vertices, so two
    # non-repeating strokes are the mathematical minimum.
    assert len(trails) == 2
    used = Counter()
    for trail in trails:
        for first, second in zip(trail, trail[1:]):
            used[tuple(sorted((first, second)))] += 1
    assert used == Counter({tuple(sorted((edge.a, edge.b))): 1 for edge in edges})


def test_hole_safe_router_detours_around_an_internal_void() -> None:
    """A derived endpoint route must never shortcut through a honeycomb hole."""

    solid = Polygon(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        holes=[[(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)]],
    )
    start = (0.0, 5.0)
    end = (10.0, 5.0)
    graph = {
        start: [((0.0, 8.0), 3.0)],
        (0.0, 8.0): [(start, 3.0), ((10.0, 8.0), 10.0)],
        (10.0, 8.0): [((0.0, 8.0), 10.0), (end, 3.0)],
        end: [((10.0, 8.0), 3.0)],
    }

    route = HoleSafeTravelRouter(solid, graph, spacing_mm=1.0).route(start, end)

    assert route is not None
    assert LineString(route).length > LineString((start, end)).length
    assert LineString(route).intersection(Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])).length <= 1e-8


def test_repeated_endpoint_flow_is_tapered_without_changing_unshared_flow() -> None:
    paths = [
        np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]),
        np.asarray([[10.0, 0.0, 0.0], [14.0, 0.0, 0.0]]),
    ]
    plan, meta = _endpoint_taper_plan(paths, tolerance_mm=1e-4, line_width_mm=2.0)
    tapered_paths = _insert_endpoint_taper_points(paths, plan)
    profiles = _extrusion_profiles(tapered_paths, 1.0, taper_plan=plan)

    assert meta["maximum_endpoint_visits"] == 3
    assert meta["shared_endpoint_count"] == 1
    assert plan[0][0] == pytest.approx(1.0 / 3.0)
    assert tapered_paths[0].shape[0] == 6
    # The three shared starts reduce flow only near the joint; the unrelated
    # path keeps its nominal length/E relationship.
    assert profiles[0][-1] == pytest.approx(2.0 + 2.0 * (1.0 + 1.0 / 3.0) / 2.0)
    assert profiles[3][-1] - profiles[3][0] == pytest.approx(4.0)


def test_honeycomb_slice_exports_to_core_without_retracing_or_crossing_holes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise the public Prusa adapter seam with a real honeycomb section."""

    class NativePrusa:
        def slice_print_paths(self, vertices, faces, **kwargs):
            min_x, min_y = np.min(vertices[:, :2], axis=0)

            def frame(z: float):
                return [
                    [min_x, min_y, z],
                    [min_x + 20.0, min_y, z],
                    [min_x + 20.0, min_y + 18.0, z],
                    [min_x, min_y + 18.0, z],
                    [min_x, min_y, z],
                ]

            def layer(z: float):
                first_wall = [[min_x + 0.5, min_y + 2.0, z], [min_x + 3.0, min_y + 2.0, z]]
                second_wall = [[min_x + 16.0, min_y + 2.0, z], [min_x + 19.0, min_y + 2.0, z]]
                return {
                    "z": z,
                    "paths": [frame(z), first_wall, second_wall],
                    "extrusion": [
                        [0.0, 6.0, 12.0, 18.0, 24.0],
                        [0.0, 2.5],
                        [0.0, 3.0],
                    ],
                    "roles": ["outer_contour", "outer_contour", "outer_contour"],
                    "travel": [
                        [[min_x, min_y, z], [min_x, min_y, z]],
                        [[min_x, min_y, z], [min_x + 0.5, min_y + 2.0, z]],
                        [[min_x + 3.0, min_y + 2.0, z], [min_x + 16.0, min_y + 2.0, z]],
                    ],
                    "motions": [
                        {"kind": "deposit", "index": 0},
                        {"kind": "travel", "index": 1},
                        {"kind": "deposit", "index": 1},
                        {"kind": "travel", "index": 2},
                        {"kind": "deposit", "index": 2},
                    ],
                }

            return {"gcode": "G90\nM82\n", "layers": [layer(1.0), layer(2.0)]}

    monkeypatch.setattr(
        "kuka_slicer.prusa_backend.require_native", lambda: NativePrusa()
    )
    hole_rings = _honeycomb_hole_rings()
    job = slice_mesh_to_job(
        _honeycomb_mesh(hole_rings),
        SliceConfig(
            slicing_kernel="prusa",
            material="R",
            layer_height=1.0,
            first_layer_height=1.0,
            line_width=2.0,
            infill_pattern="none",
            honeycomb_pathing=HoneycombPathingConfig(
                enabled=True,
                topology="macro_partition_zero_e",
            ),
        ),
    )

    planner_meta = job.meta["honeycomb_centerline_pathing"]
    assert job.native_gcode is None
    assert len(job.material_paths) == 2
    assert planner_meta["topology"] == "stl_honeycomb_macro_partition_zero_e"
    assert planner_meta["macro_partition_count"] >= 1
    assert planner_meta["layer_path_count"] == planner_meta["macro_partition_count"] + 1
    assert planner_meta["travel_count"] == planner_meta["macro_partition_count"] - 1
    assert planner_meta["intra_partition_zero_e_max_turn_degrees"] <= 90.0
    assert planner_meta["wall_edge_count"] > 0
    assert planner_meta["minimum_trail_count"] >= 1
    assert planner_meta["repeated_wall_edge_count"] == 0
    assert planner_meta["repeated_wall_length_mm_per_layer"] == 0.0
    assert planner_meta["intra_partition_zero_e_connector_count"] > 0

    holes = unary_union([Polygon(ring) for ring in hole_rings])
    hole_interior = holes.buffer(-1e-4)
    for group in job.material_paths:
        assert len(group.paths) == planner_meta["layer_path_count"]
        assert group.extrusion is not None
        assert len(group.extrusion) == planner_meta["layer_path_count"]
        for path, values in zip(group.paths, group.extrusion):
            assert values.shape[0] == path.shape[0]
            assert np.all(np.diff(values) >= 0.0)
        assert any(
            np.any(
                (np.linalg.norm(np.diff(macro_path[:, :2], axis=0), axis=1) > 1e-8)
                & np.isclose(np.diff(macro_e), 0.0, atol=1e-9)
            )
            for macro_path, macro_e in zip(group.paths[1:], group.extrusion[1:])
        )
        assert all(
            LineString(segment).intersection(hole_interior).length <= 1e-8
            for macro_path in group.paths[1:]
            for segment in zip(macro_path[:-1, :2], macro_path[1:, :2])
        )
        all_points = np.vstack(group.paths)
        assert np.min(all_points[:, :2], axis=0) == pytest.approx([0.0, 0.0])
        assert np.max(all_points[:, :2], axis=0) == pytest.approx([20.0, 18.0])
        assert job.meta["path_roles"]["R"][str(group.layer_index)] == [
            "outer_contour", *("honeycomb_wall" for _ in group.paths[1:])
        ]
    assert len(job.travel_paths) == len(job.material_paths)
    assert all(
        len(group.paths) == planner_meta["travel_count"]
        for group in job.travel_paths
    )

    source_path = tmp_path / "honeycomb_source.npz"
    system_path = tmp_path / "honeycomb_system.npz"
    write_external_source_npz(job, source_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in OFFLINE_SOURCE_ROOTS
    )
    conversion = subprocess.run(
        [
            sys.executable,
            "-m",
            "external_npz_preprocessor.cli",
            "--source",
            str(source_path),
            "--out",
            str(system_path),
            "--dt",
            "0.02",
        ],
        cwd=OFFLINE_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert conversion.returncode == 0, conversion.stderr
    assert system_path.is_file()


def _honeycomb_mesh(hole_rings: list[list[tuple[float, float]]]) -> Mesh:
    outer_ring = [(0.0, 0.0), (20.0, 0.0), (20.0, 18.0), (0.0, 18.0)]
    triangles = []
    for ring in [outer_ring, *hole_rings]:
        for first, second in zip(ring, [*ring[1:], ring[0]]):
            lower_first = [*first, 0.0]
            lower_second = [*second, 0.0]
            upper_first = [*first, 3.0]
            upper_second = [*second, 3.0]
            triangles.extend(
                (
                    [lower_first, lower_second, upper_second],
                    [lower_first, upper_second, upper_first],
                )
            )
    return Mesh(np.asarray(triangles, dtype=np.float32))


def _honeycomb_hole_rings() -> list[list[tuple[float, float]]]:
    return [
        _hexagon((5.0, 5.0), 2.0),
        _hexagon((11.0, 5.0), 2.0),
        _hexagon((8.0, 10.2), 2.0),
        _hexagon((14.0, 10.2), 2.0),
    ]


def _hexagon(center: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    return [
        (
            center[0] + radius * np.cos(np.pi / 3.0 * index),
            center[1] + radius * np.sin(np.pi / 3.0 * index),
        )
        for index in range(6)
    ]
