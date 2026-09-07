"""Read-only payloads for an independent conformal-lattice preview surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .layer_embedding import LayerEmbedding
from .lattice_generator import ConformalLatticeGeometry
from .mesh_domain import SurfaceMeshDomain
from .orientation_field import OrientationField
from .parameterization import LSCMParameterization
from .phase_coordinates import PhaseCoordinates
from .scalar_fields import DesignFieldResult

if TYPE_CHECKING:
    from .fill_ratio_validation import FillRatioValidation


def conformal_lattice_preview_payload(
    domain: SurfaceMeshDomain,
    parameterization: LSCMParameterization,
    design_fields: DesignFieldResult,
    orientation: OrientationField,
    phase: PhaseCoordinates,
    geometry: ConformalLatticeGeometry,
    *,
    fill_validation: "FillRatioValidation | None" = None,
    layer_embedding: LayerEmbedding | None = None,
) -> dict[str, object]:
    """Return serialisable views only; this function never changes geometry."""

    fill = {
        "target_fill_ratio_per_vertex": design_fields.target_fill_ratio.tolist(),
        "target_cell_size_mm_per_vertex": design_fields.target_cell_size_mm.tolist(),
        "actual": None if fill_validation is None else fill_validation.heatmap_payload(),
    }
    return {
        "surface_3d": {"vertices_xyz": domain.vertices.tolist(), "faces": domain.faces.tolist(), "lattice_nodes_xyz": geometry.lattice_nodes_xyz.tolist(), "lattice_edges": geometry.lattice_edges.tolist()},
        "uv": {"surface_uv": parameterization.uv.tolist(), "faces": domain.faces.tolist(), "lattice_nodes_uv": geometry.lattice_nodes_uv.tolist(), "lattice_edges": geometry.lattice_edges.tolist()},
        "conformal_distortion": parameterization.quality.summary | {"ratio_per_face": parameterization.quality.conformal_ratio_per_face.tolist(), "angle_error_deg_per_face": parameterization.quality.angle_error_deg_per_face.tolist()},
        "fill_ratio": fill,
        "orientation": orientation.heatmap_payload(),
        "defects": {"cell_offsets": geometry.cell_offsets.tolist(), "cell_node_indices": geometry.cell_node_indices.tolist(), "cell_valence": geometry.cell_valence.tolist(), "cell_defect_code": geometry.cell_defect_code.tolist(), "cell_is_boundary": geometry.cell_is_boundary.tolist(), "report": geometry.report},
        "phase": phase.heatmap_payload(),
        "layers": None if layer_embedding is None else layer_embedding.preview_payload(),
        "read_only": True,
    }
