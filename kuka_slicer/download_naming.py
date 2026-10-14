"""User-facing honeycomb names; independent of geometry and Core export data."""

from __future__ import annotations

from collections.abc import Mapping


def honeycomb_download_filename(
    source_surface: Mapping[str, object],
    lattice: Mapping[str, object],
    *,
    material: str | None = None,
    extension: str,
    include_outer_contour: bool,
) -> str:
    """Name an already validated design/artifact without modifying its fields.

    Wave counts are recovered from the exported domain and wavelengths so
    manual wavelength designs and re-imported JSON use the same convention.
    """
    if material not in {None, "纤维", "树脂"}:
        raise ValueError("unsupported download material label")
    if extension not in {".json", ".npz", ".zip"}:
        raise ValueError("unsupported download extension")
    provider = source_surface["provider"]
    shape = "平面" if provider == "planar" else "曲面"
    if provider == "double_sine":
        surface = source_surface["double_sine"]
        x0, y0, x1, y1 = surface["xy_bounds_mm"]
        a = float(surface["amplitude_mm"])
        nx = (float(x1) - float(x0)) / float(surface["wavelength_x_mm"])
        ny = (float(y1) - float(y0)) / float(surface["wavelength_y_mm"])
        shape += f"（{a:.12g}-{nx:.12g}-{ny:.12g}）"
    cell = float(lattice["base_cell_size_mm"])
    material_part = f"-{material}" if material else ""
    outer = "（含外轮廓）" if include_outer_contour else ""
    return f"{shape}-{cell:.12g}mm{material_part}{outer}{extension}"
