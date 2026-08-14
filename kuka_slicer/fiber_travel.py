"""Hole-safe travel planning for externally supplied fiber paths.

The fiber JSON is deliberately kept as a deposition-only input format.  This
module adds only the non-depositing connectors required between consecutive
fiber paths, after their final UI placement has been resolved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from shapely.geometry import LineString

from .honeycomb_pathing import HoleSafeTravelRouter, solid_geometry_at_z
from .slicer import SliceConfig, orient_mesh_for_build_axis
from .stl_io import Mesh


def plan_fiber_interpath_travels(
    mesh: Mesh,
    config: SliceConfig,
    fiber_paths_by_layer: Mapping[int, Sequence[Sequence[Sequence[float]]]],
    *,
    reference_z_by_layer: Mapping[int | str, float] | None = None,
) -> dict[int, list[np.ndarray]]:
    """Return shortest no-hole connectors between consecutive fiber paths.

    A direct segment is retained whenever it is clear of internal holes.  For
    blocked endpoints, :class:`HoleSafeTravelRouter` supplies a polyline that
    stays outside those holes.  Routes are evaluated at every fiber layer so
    tapered models cannot accidentally reuse a path through a later void.  A
    fiber course can be physically raised by previously inserted fiber; the
    optional reference map keeps the hole check at its original STL section.
    """

    planned: dict[int, list[np.ndarray]] = {}
    oriented_mesh = orient_mesh_for_build_axis(mesh, config.build_axis)
    for layer_index in sorted(fiber_paths_by_layer):
        paths = [np.asarray(path, dtype=np.float64) for path in fiber_paths_by_layer[layer_index]]
        if len(paths) < 2:
            continue
        if any(path.ndim != 2 or path.shape[0] < 1 or path.shape[1] < 3 for path in paths):
            raise ValueError(f"fiber layer {layer_index} contains an invalid path")
        z = float(paths[0][0, 2])
        references = reference_z_by_layer or {}
        section_z = float(references.get(layer_index, references.get(str(layer_index), z)))
        solid = solid_geometry_at_z(oriented_mesh, section_z, float(config.tolerance))
        router = HoleSafeTravelRouter(
            solid,
            {},
            spacing_mm=max(0.5, min(2.0, float(config.line_width))),
        )
        connectors: list[np.ndarray] = []
        for path_before, path_after in zip(paths, paths[1:]):
            route = router.route(
                (float(path_before[-1, 0]), float(path_before[-1, 1])),
                (float(path_after[0, 0]), float(path_after[0, 1])),
            )
            if route is None:
                raise ValueError(
                    f"cannot find a hole-safe fiber travel on layer {layer_index}"
                )
            if not router.allows(LineString(route)):
                raise RuntimeError("fiber travel router returned a route through a hole")
            connector = np.asarray(
                [[x, y, z] for x, y in route],
                dtype=np.float64,
            )
            connectors.append(connector)
        planned[int(layer_index)] = connectors
    return planned
