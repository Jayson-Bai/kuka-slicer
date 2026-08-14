import numpy as np
from shapely.geometry import LineString, Polygon

from kuka_slicer.fiber_travel import plan_fiber_interpath_travels
from kuka_slicer.slicer import SliceConfig
from kuka_slicer.stl_io import Mesh


def _vertical_ring_triangles(ring, z0=0.0, z1=1.0):
    triangles = []
    for first, second in zip(ring, ring[1:] + ring[:1]):
        lower_first = [first[0], first[1], z0]
        lower_second = [second[0], second[1], z0]
        upper_first = [first[0], first[1], z1]
        upper_second = [second[0], second[1], z1]
        triangles.extend(
            ([lower_first, lower_second, upper_second], [lower_first, upper_second, upper_first])
        )
    return triangles


def test_fiber_interpath_travel_uses_shortest_direct_route_unless_a_hole_blocks_it():
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    hole = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
    mesh = Mesh(np.asarray(_vertical_ring_triangles(outer) + _vertical_ring_triangles(hole)))
    paths = {
        0: [
            [[1.0, 5.0, 0.5], [3.0, 5.0, 0.5]],
            [[7.0, 5.0, 0.5], [9.0, 5.0, 0.5]],
        ]
    }

    connector = plan_fiber_interpath_travels(mesh, SliceConfig(), paths)[0][0]

    assert len(connector) > 2
    assert connector[0].tolist() == [3.0, 5.0, 0.5]
    assert connector[-1].tolist() == [7.0, 5.0, 0.5]
    # The shortest valid route may touch/run along the hole boundary, but it
    # must never cross the void interior.
    assert not LineString(connector[:, :2]).crosses(Polygon(hole))


def test_fiber_interpath_travel_uses_unshifted_stl_section_but_keeps_physical_z():
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    hole = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
    mesh = Mesh(np.asarray(_vertical_ring_triangles(outer) + _vertical_ring_triangles(hole)))
    paths = {
        7: [
            [[1.0, 5.0, 1.3], [3.0, 5.0, 1.3]],
            [[7.0, 5.0, 1.3], [9.0, 5.0, 1.3]],
        ]
    }

    connector = plan_fiber_interpath_travels(
        mesh,
        SliceConfig(),
        paths,
        reference_z_by_layer={7: 0.5},
    )[7][0]

    assert np.all(connector[:, 2] == 1.3)
    assert not LineString(connector[:, :2]).crosses(Polygon(hole))
