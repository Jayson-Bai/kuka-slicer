from __future__ import annotations

import numpy as np
from shapely.geometry import box

from kuka_slicer.prusa_backend import slice_mesh_to_job_with_prusa
from kuka_slicer.slicer import PrusaGeometryConfig, PrusaRaftConfig, SliceConfig
from kuka_slicer.stl_io import Mesh


def test_isotropic_schedule_is_forwarded_as_explicit_prusa_fill_angles(monkeypatch):
    received: dict[str, object] = {}

    class Native:
        def slice_print_paths(self, vertices, faces, **kwargs):
            received.update(kwargs)
            return {"layers": []}

    monkeypatch.setattr("kuka_slicer.prusa_backend.require_native", lambda: Native())
    mesh = Mesh(_cube_triangles(10.0))

    job = slice_mesh_to_job_with_prusa(
        mesh,
        SliceConfig(
            slicing_kernel="prusa",
            layer_height=0.5,
            first_layer_height=0.5,
            line_width=2.0,
            infill_pattern="isotropic",
            infill_density=100.0,
            contour_infill_overlap=2.0,
        ),
    )

    assert received["infill_pattern"] == "rectilinear"
    assert received["fill_angle_schedule"] == [45.0, 0.0, -45.0, 90.0]
    assert received["perimeter_infill_overlap"] == 2.0
    assert job.meta["slicing"]["prusa_fill_angle_schedule"] == [45.0, 0.0, -45.0, 90.0]


def test_prusa_backend_preserves_native_deposit_and_travel_motion_order(monkeypatch):
    class Native:
        def slice_print_paths(self, vertices, faces, **kwargs):
            return {
                "layers": [
                    {
                        "z": 10.5,
                        "paths": [
                            [[10.0, 10.0, 10.5], [12.0, 10.0, 10.5]],
                            [[14.0, 10.0, 10.5], [16.0, 10.0, 10.5]],
                        ],
                        "extrusion": [[0.0, 1.0], [1.0, 2.0]],
                        "roles": ["outer_contour", "infill"],
                        "travel": [[[12.0, 10.0, 10.5], [14.0, 10.0, 10.5]]],
                        "motions": [
                            {"kind": "deposit", "index": 0},
                            {"kind": "travel", "index": 0},
                            {"kind": "deposit", "index": 1},
                        ],
                    }
                ]
            }

    monkeypatch.setattr("kuka_slicer.prusa_backend.require_native", lambda: Native())
    job = slice_mesh_to_job_with_prusa(
        Mesh(_cube_triangles(10.0)),
        SliceConfig(
            slicing_kernel="prusa",
            layer_height=0.5,
            first_layer_height=0.5,
            line_width=2.0,
            infill_pattern="zigzag_horizontal",
        ),
    )

    assert job.meta["motion_order"] == {
        "0": [
            {"kind": "deposit", "index": 0},
            {"kind": "travel", "index": 0},
            {"kind": "deposit", "index": 1},
        ]
    }


def test_prusa_backend_forwards_native_detachable_raft_configuration(monkeypatch):
    received: dict[str, object] = {}

    class Native:
        def slice_print_paths(self, vertices, faces, **kwargs):
            received.update(kwargs)
            return {"layers": []}

    monkeypatch.setattr("kuka_slicer.prusa_backend.require_native", lambda: Native())
    raft = PrusaRaftConfig(
        layer_count=3,
        expansion=3.0,
        first_layer_density=80.0,
        first_layer_expansion=3.0,
        contact_distance=0.25,
    )
    job = slice_mesh_to_job_with_prusa(
        Mesh(_cube_triangles(10.0)),
        SliceConfig(
            slicing_kernel="prusa",
            layer_height=0.5,
            first_layer_height=0.5,
            line_width=2.0,
            prusa_raft=raft,
        ),
    )

    assert received["raft_layers"] == 3
    assert received["raft_expansion"] == 3.0
    assert received["raft_first_layer_density"] == 80.0
    assert received["raft_first_layer_expansion"] == 3.0
    assert received["raft_contact_distance"] == 0.25
    assert received["raft_contact_layer_height"] == 0.0
    assert received["raft_contact_density"] == 0.0
    assert received["raft_contact_extrusion_width"] == 0.0
    assert job.meta["slicing"]["prusa_raft"] == {
        "layer_count": 3,
        "expansion": 3.0,
        "first_layer_density": 80.0,
        "first_layer_expansion": 3.0,
        "contact_distance": 0.25,
        "contact_auto": True,
        "contact_layer_height": 0.75,
        "contact_density": 100.0,
        "contact_extrusion_width": 1.5,
    }


def test_prusa_backend_keeps_native_gcode_outside_legacy_npz_metadata(monkeypatch):
    class Native:
        def slice_print_paths(self, vertices, faces, **kwargs):
            return {
                "gcode": "G90\nM82\n",
                "layers": [],
            }

    monkeypatch.setattr("kuka_slicer.prusa_backend.require_native", lambda: Native())
    job = slice_mesh_to_job_with_prusa(
        Mesh(_cube_triangles(10.0)),
        SliceConfig(
            slicing_kernel="prusa",
            layer_height=0.5,
            first_layer_height=0.5,
            line_width=2.0,
        ),
    )

    assert job.native_gcode == "G90\nM82\n"
    assert job.native_gcode_translation_mm == (-10.0, -10.0, 0.0)
    assert "gcode" not in job.meta


def test_prusa_backend_can_merge_brim_with_project_connector(monkeypatch):
    class Native:
        def slice_print_paths(self, vertices, faces, **kwargs):
            return {
                "layers": [
                    {
                        "z": 0.5,
                        "paths": [
                            [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]],
                            [[2.0, 0.0, 0.5], [3.0, 0.0, 0.5]],
                            [[4.0, 0.0, 0.5], [5.0, 0.0, 0.5]],
                        ],
                        "extrusion": [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
                        "roles": ["brim", "brim", "brim"],
                        "travel": [
                            [[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
                            [[3.0, 0.0, 0.5], [4.0, 0.0, 0.5]],
                        ],
                        "motions": [
                            {"kind": "deposit", "index": 0},
                            {"kind": "travel", "index": 0},
                            {"kind": "deposit", "index": 1},
                            {"kind": "travel", "index": 1},
                            {"kind": "deposit", "index": 2},
                        ],
                    }
                ]
            }

    monkeypatch.setattr("kuka_slicer.prusa_backend.require_native", lambda: Native())
    monkeypatch.setattr(
        "kuka_slicer.prusa_backend._connect_brim_paths_one_stroke",
        lambda paths, line_width, tolerance: [
            np.vstack(paths).astype(np.float32)
        ],
    )
    job = slice_mesh_to_job_with_prusa(
        Mesh(_cube_triangles(10.0)),
        SliceConfig(
            slicing_kernel="prusa",
            layer_height=0.5,
            first_layer_height=0.5,
            line_width=2.0,
            brim_enabled=True,
            brim_one_stroke=True,
        ),
    )

    group = job.material_paths[0]
    assert len(group.paths) == 1
    assert len(group.extrusion or []) == 1
    assert job.meta["slicing"]["prusa_brim"]["one_stroke"] is True
    assert job.travel_paths == []
    assert job.meta["motion_order"] == {"0": [{"kind": "deposit", "index": 0}]}


def test_prusa_backend_chains_rectilinear_infill_without_touching_contours(monkeypatch):
    class Native:
        def slice_print_paths(self, vertices, faces, **kwargs):
            return {
                "layers": [
                    {
                        "z": 0.5,
                        "paths": [
                            [[0.0, 3.0, 0.5], [3.0, 3.0, 0.5]],
                            [[0.0, 0.0, 0.5], [3.0, 0.0, 0.5]],
                            [[3.0, 1.0, 0.5], [0.0, 1.0, 0.5]],
                            [[0.0, 2.0, 0.5], [3.0, 2.0, 0.5]],
                        ],
                        "extrusion": [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                        "roles": ["outer_contour", "infill", "infill", "infill"],
                        "travel": [
                            [[3.0, 3.0, 0.5], [0.0, 0.0, 0.5]],
                            [[3.0, 0.0, 0.5], [3.0, 1.0, 0.5]],
                            [[0.0, 1.0, 0.5], [0.0, 2.0, 0.5]],
                        ],
                        "motions": [
                            {"kind": "deposit", "index": 0},
                            {"kind": "travel", "index": 0},
                            {"kind": "deposit", "index": 1},
                            {"kind": "travel", "index": 1},
                            {"kind": "deposit", "index": 2},
                            {"kind": "travel", "index": 2},
                            {"kind": "deposit", "index": 3},
                        ],
                    }
                ]
            }

    monkeypatch.setattr("kuka_slicer.prusa_backend.require_native", lambda: Native())
    monkeypatch.setattr(
        "kuka_slicer.prusa_backend.solid_geometry_at_z",
        lambda mesh, z, tolerance: box(-20.0, -20.0, 10.0, 10.0),
    )
    monkeypatch.setattr(
        "kuka_slicer.prusa_backend._connect_zigzag_infill_paths",
        lambda paths, *args, **kwargs: [np.vstack(paths)],
    )

    job = slice_mesh_to_job_with_prusa(
        Mesh(_cube_triangles(10.0)),
        SliceConfig(
            slicing_kernel="prusa",
            layer_height=0.5,
            first_layer_height=0.5,
            line_width=2.0,
            infill_pattern="isotropic",
        ),
    )

    group = job.material_paths[0]
    assert len(group.paths) == 2
    assert job.meta["path_roles"]["R"]["0"] == ["outer_contour", "infill"]
    assert len(job.travel_paths[0].paths) == 1
    assert job.meta["motion_order"] == {
        "0": [
            {"kind": "deposit", "index": 0},
            {"kind": "travel", "index": 0},
            {"kind": "deposit", "index": 1},
        ]
    }


def test_prusa_backend_does_not_apply_one_stroke_to_non_rectilinear_infill(monkeypatch):
    class Native:
        def slice_print_paths(self, vertices, faces, **kwargs):
            return {
                "layers": [
                    {
                        "z": 0.5,
                        "paths": [
                            [[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
                            [[0.0, 1.0, 0.5], [2.0, 1.0, 0.5]],
                        ],
                        "extrusion": [[0.0, 1.0], [1.0, 2.0]],
                        "roles": ["infill", "infill"],
                    }
                ]
            }

    monkeypatch.setattr("kuka_slicer.prusa_backend.require_native", lambda: Native())
    monkeypatch.setattr(
        "kuka_slicer.prusa_backend._connect_zigzag_infill_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    job = slice_mesh_to_job_with_prusa(
        Mesh(_cube_triangles(10.0)),
        SliceConfig(
            slicing_kernel="prusa",
            layer_height=0.5,
            first_layer_height=0.5,
            line_width=2.0,
            infill_pattern="grid",
        ),
    )

    assert len(job.material_paths[0].paths) == 2


def test_prusa_backend_forwards_native_geometry_and_path_controls(monkeypatch):
    received: dict[str, object] = {}

    class Native:
        def slice_print_paths(self, vertices, faces, **kwargs):
            received.update(kwargs)
            return {"layers": []}

    monkeypatch.setattr("kuka_slicer.prusa_backend.require_native", lambda: Native())
    geometry = PrusaGeometryConfig(
        perimeter_generator="classic",
        gap_fill_enabled=False,
        infill_anchor=2.0,
        infill_anchor_max=6.0,
        external_perimeter_width=1.8,
        perimeter_width=1.9,
        infill_width=2.1,
        xy_size_compensation=0.15,
        elephant_foot_compensation=0.05,
        avoid_crossing_max_detour=12.0,
        seam_position="rear",
    )
    job = slice_mesh_to_job_with_prusa(
        Mesh(_cube_triangles(10.0)),
        SliceConfig(
            slicing_kernel="prusa",
            layer_height=0.5,
            first_layer_height=0.5,
            line_width=2.0,
            prusa_geometry=geometry,
        ),
    )

    assert received["perimeter_generator"] == "classic"
    assert received["gap_fill_enabled"] is False
    assert received["infill_anchor"] == 2.0
    assert received["infill_anchor_max"] == 6.0
    assert received["external_perimeter_width"] == 1.8
    assert received["perimeter_width"] == 1.9
    assert received["infill_width"] == 2.1
    assert received["xy_size_compensation"] == 0.15
    assert received["elephant_foot_compensation"] == 0.05
    assert received["avoid_crossing_max_detour"] == 12.0
    assert received["seam_position"] == "rear"
    assert job.meta["slicing"]["prusa_geometry"] == geometry.to_metadata()


def _cube_triangles(size: float) -> np.ndarray:
    vertices = np.asarray(
        [
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
            [0, 0, size], [size, 0, size], [size, size, size], [0, size, size],
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
    return vertices[faces]
