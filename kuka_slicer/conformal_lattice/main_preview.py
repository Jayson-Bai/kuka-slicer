"""Thin conformal-path adapter for the established main slicer Canvas."""

from __future__ import annotations

import math

from .path_bridge import ConformalLatticePathGraph


def main_preview_payload_from_conformal_path_graph(
    graph: ConformalLatticePathGraph,
    *,
    planning_line_width_mm: float,
) -> dict[str, object]:
    """Adapt already-planned conformal edges to the legacy main preview schema.

    Design wall width is intentionally not used as a visual line width.  The
    process-side planning width must be explicit, just as it is for the old
    mapped-source preview.
    """

    if not math.isfinite(planning_line_width_mm) or planning_line_width_mm <= 0.0:
        raise ValueError("planning_line_width_mm must be positive and finite")

    # Local imports keep the conformal package independent of the UI at import
    # time, and preserve the existing UI module as the Canvas-schema authority.
    from ..slicer import SliceConfig
    from ..ui_server import _preview_payload

    preview = _preview_payload(
        None,
        SliceConfig(line_width=float(planning_line_width_mm)),
        graph.to_external_source_job(),
    )
    preview["preview_source"] = "conformal_lattice_external_source_job"
    preview["conformal_lattice"] = {
        "edge_count_per_layer": int(len(graph.edge_ids)),
        "path_order": "ascending stable edge ID; one two-point path per structural edge",
        "uses_existing_main_canvas": True,
        "planning_line_width_mm": float(planning_line_width_mm),
    }
    return preview
