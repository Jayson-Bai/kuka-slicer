"""STL section repair must preserve holes and never close real mesh gaps."""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from kuka_slicer.fiber_travel import plan_fiber_interpath_travels
from kuka_slicer.honeycomb_pathing import planner
from kuka_slicer.slicer import SliceConfig
from kuka_slicer.stl_io import Mesh


def _walls(ring):
    triangles = []
    for a, b in zip(ring, ring[1:] + ring[:1]):
        triangles.extend((
            [[*a, 0.], [*b, 0.], [*b, 1.]],
            [[*a, 0.], [*b, 1.], [*a, 1.]],
        ))
    return triangles


OUTER = [(0., 0.), (10., 0.), (10., 10.), (0., 10.)]
HOLE = [(4., 4.), (6., 4.), (6., 6.), (4., 6.)]


def test_normal_section_keeps_legacy_branch(monkeypatch):
    def forbid_reconstruction(*args):
        pytest.fail("Trustworthy sections must keep the legacy result")

    monkeypatch.setattr(planner, "_reconstruct_section_from_mesh_edges", forbid_reconstruction)
    solid = planner.solid_geometry_at_z(Mesh(_walls(OUTER) + _walls(HOLE)), .5, 1e-5)
    assert solid.equals(Polygon(OUTER, [HOLE]))


@pytest.mark.parametrize("z", [0., .000001, .4, .5, .999999, 1.])
def test_shared_edge_reconstruction_keeps_sub_tolerance_features_and_real_holes(z):
    # The short first edge is essential to close the outer ring. The narrow
    # hole must not be collapsed by snapping nearby, unrelated mesh edges.
    outer = [(0., 0.), (.0002, 0.), *OUTER[1:]]
    narrow_hole = [(4., 4.), (4.0002, 4.), (4.0002, 6.), (4., 6.)]
    mesh = Mesh(_walls(outer) + _walls(narrow_hole))
    expected = Polygon(outer, [np.asarray(narrow_hole, dtype=np.float32)])
    solid = planner.solid_geometry_at_z(mesh, z, .0005)
    assert solid.is_valid
    assert len(solid.interiors) == 1
    assert solid.symmetric_difference(expected).area < 1e-10


def test_reconstruction_preserves_islands_inside_holes_and_disjoint_material():
    island = [(4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5)]
    separate = [(12., 0.), (13., 0.), (13., 1.), (12., 1.)]
    mesh = Mesh(_walls(OUTER) + _walls(HOLE) + _walls(island) + _walls(separate))
    expected = unary_union([Polygon(OUTER, [HOLE]), Polygon(island), Polygon(separate)])
    solid = planner.solid_geometry_at_z(mesh, .4, .0005)
    assert solid.is_valid and solid.equals(expected)


def test_fallback_does_not_depend_on_triangle_order_or_winding():
    outer = [(0., 0.), (.0002, 0.), *OUTER[1:]]
    triangles = np.asarray(_walls(outer) + _walls(HOLE))
    rng = np.random.default_rng(12)
    for _ in range(4):
        rng.shuffle(triangles)
        triangles = triangles[:, ::-1, :]
        assert planner.solid_geometry_at_z(Mesh(triangles), .4, .0005).equals(Polygon(OUTER, [HOLE]))


@pytest.mark.parametrize("defect", ["open", "near_gap", "duplicate", "self_intersection"])
def test_ambiguous_sections_fail_with_height_instead_of_guessing(defect):
    if defect == "self_intersection":
        triangles = _walls([(0., 0.), (10., 10.), (0., 10.), (10., 0.)])
    else:
        triangles = _walls(OUTER)
        if defect == "open":
            triangles = triangles[2:]
        elif defect == "near_gap":
            # A real open seam, even smaller than the planning tolerance,
            # must not be welded by the shared-edge reconstruction.
            triangles[0][0][0] = .0002
            triangles[1][0][0] = .0002
            triangles[1][2][0] = .0002
        else:
            triangles += triangles[:1]
    with pytest.raises(ValueError, match=r"reliable STL section at Z=0.4 mm"):
        planner.solid_geometry_at_z(Mesh(triangles), .4, .0005)


def test_repaired_section_travel_is_checked_against_independent_hole_geometry():
    outer = [(0., 0.), (.0002, 0.), *OUTER[1:]]
    mesh = Mesh(_walls(outer) + _walls(HOLE))
    paths = {2: [[[1., 5., .6], [3., 5., .6]], [[7., 5., .6], [9., 5., .6]]]}
    connector = plan_fiber_interpath_travels(
        mesh, SliceConfig(tolerance=.0005), paths, reference_z_by_layer={2: .5},
    )[2][0]
    assert connector[0].tolist() == paths[2][0][-1]
    assert connector[-1].tolist() == paths[2][1][0]
    assert np.all(connector[:, 2] == .6)
    assert LineString(connector[:, :2]).intersection(Polygon(HOLE).buffer(-1e-4)).length <= 1e-8
