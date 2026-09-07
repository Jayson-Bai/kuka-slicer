"""STL-backed XY domain extraction for the standalone surface preview."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from shapely import contains_xy
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from ..slicer import mesh_xy_projection
from ..stl_io import load_stl_bytes

BuildAxis = Literal["x", "y", "z"]


def _coordinate_pairs(coordinates) -> list[list[float]]:
    return [[float(x), float(y)] for x, y, *_ in coordinates]


def _polygons(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        result: list[Polygon] = []
        for item in geometry.geoms:
            result.extend(_polygons(item))
        return result
    return []


@dataclass(frozen=True, slots=True)
class STLProjectionDomain:
    """A local XY material domain whose origin is the STL XY minimum."""

    file_name: str
    sha256: str
    build_axis: BuildAxis
    projection_layer_height_mm: float
    triangle_count: int
    source_xy_bounds_mm: tuple[float, float, float, float]
    geometry: object

    @property
    def width_mm(self) -> float:
        return float(self.source_xy_bounds_mm[2] - self.source_xy_bounds_mm[0])

    @property
    def height_mm(self) -> float:
        return float(self.source_xy_bounds_mm[3] - self.source_xy_bounds_mm[1])

    def preview_payload(self, *, include_polygons: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "file_name": self.file_name,
            "sha256": self.sha256,
            "build_axis": self.build_axis,
            "triangle_count": self.triangle_count,
            "projection_layer_height_mm": self.projection_layer_height_mm,
            "source_xy_bounds_mm": list(self.source_xy_bounds_mm),
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "material_area_mm2": float(self.geometry.area),
        }
        if include_polygons:
            polygons = []
            for polygon in _polygons(self.geometry):
                polygons.append(
                    {
                        "outer": _coordinate_pairs(polygon.exterior.coords),
                        "holes": [_coordinate_pairs(ring.coords) for ring in polygon.interiors],
                    }
                )
            payload["polygons"] = polygons
        return payload

    def material_mask(self, x_mm, y_mm) -> np.ndarray:
        """Return a vectorised material-domain mask for sampled XY positions."""

        return np.asarray(contains_xy(self.geometry, x_mm, y_mm), dtype=bool)


def stl_projection_domain_from_bytes(
    data: bytes,
    *,
    file_name: str,
    build_axis: BuildAxis = "z",
    projection_layer_height_mm: float = 0.5,
) -> STLProjectionDomain:
    """Load an STL and turn its printable sections into a local XY domain."""

    if build_axis not in ("x", "y", "z"):
        raise ValueError("build_axis must be x, y, or z")
    mesh = load_stl_bytes(data)
    geometry = mesh_xy_projection(
        mesh,
        build_axis=build_axis,
        layer_height=float(projection_layer_height_mm),
    )
    if geometry.is_empty or geometry.area <= 0.0:
        raise ValueError("STL has no printable XY projection for the selected build axis")
    x_min, y_min, x_max, y_max = (float(value) for value in geometry.bounds)
    local_geometry = affinity.translate(geometry, xoff=-x_min, yoff=-y_min)
    return STLProjectionDomain(
        file_name=Path(file_name).name or "model.stl",
        sha256=hashlib.sha256(data).hexdigest(),
        build_axis=build_axis,
        projection_layer_height_mm=float(projection_layer_height_mm),
        triangle_count=int(mesh.triangles.shape[0]),
        source_xy_bounds_mm=(x_min, y_min, x_max, y_max),
        geometry=local_geometry,
    )
