"""Independent conformal-lattice structure geometry pipeline.

This package deliberately does not import or alter ``surface_mapper``.  Its
first gate establishes a validated triangular surface domain; lattice and path
generation are added by later gates.
"""

from .contracts import ConformalLatticeSpec, load_conformal_lattice_spec
from .distortion import ConformalQuality, evaluate_conformal_quality, nonadjacent_triangle_overlap_pairs
from .mesh_domain import (
    SurfaceMeshDomain,
    build_double_sine_surface_domain,
    cut_mesh_along_edges,
    load_triangle_mesh_domain,
    prepare_surface_mesh_domain,
)
from .parameterization import (
    LSCMParameterization,
    farthest_boundary_anchors,
    parameterize_lscm,
    parameterize_spec_lscm,
)
from .orientation_field import OrientationField, build_orientation_field
from .lattice_generator import ConformalLatticeGeometry, generate_conformal_lattice_geometry
from .layer_embedding import LayerEmbedding, embed_lattice_layers
from .path_bridge import (
    ConformalLatticePathGraph,
    ExtrusionVolumeModel,
    build_conformal_lattice_path_graph,
    write_conformal_lattice_external_npz,
)
from .preview import conformal_lattice_preview_payload
from .server import conformal_lattice_preview_html, run_conformal_lattice_preview_server
from .fill_ratio_validation import CellScaleCorrection, FillRatioValidation, derive_fill_ratio_cell_size_override, validate_realized_fill_ratio
from .phase_coordinates import (
    PhaseAnchor,
    PhaseCoordinates,
    PhaseQuality,
    evaluate_phase_quality,
    require_valid_phase,
    solve_phase_coordinates,
)
from .scalar_fields import (
    DesignFieldResult,
    ExternalScalarMetadata,
    FieldComponent,
    compose_design_fields,
    constant_driver,
    curvature_driver,
    external_scalar_driver,
    roi_driver,
)

__all__ = [
    "ConformalLatticeSpec",
    "CellScaleCorrection",
    "ConformalLatticeGeometry",
    "ConformalLatticePathGraph",
    "ConformalQuality",
    "DesignFieldResult",
    "ExternalScalarMetadata",
    "ExtrusionVolumeModel",
    "FieldComponent",
    "FillRatioValidation",
    "LSCMParameterization",
    "LayerEmbedding",
    "OrientationField",
    "PhaseAnchor",
    "PhaseCoordinates",
    "PhaseQuality",
    "SurfaceMeshDomain",
    "build_double_sine_surface_domain",
    "build_orientation_field",
    "build_conformal_lattice_path_graph",
    "compose_design_fields",
    "conformal_lattice_preview_payload",
    "conformal_lattice_preview_html",
    "constant_driver",
    "curvature_driver",
    "cut_mesh_along_edges",
    "derive_fill_ratio_cell_size_override",
    "evaluate_conformal_quality",
    "evaluate_phase_quality",
    "embed_lattice_layers",
    "external_scalar_driver",
    "farthest_boundary_anchors",
    "generate_conformal_lattice_geometry",
    "load_conformal_lattice_spec",
    "load_triangle_mesh_domain",
    "prepare_surface_mesh_domain",
    "parameterize_lscm",
    "parameterize_spec_lscm",
    "nonadjacent_triangle_overlap_pairs",
    "require_valid_phase",
    "run_conformal_lattice_preview_server",
    "roi_driver",
    "solve_phase_coordinates",
    "validate_realized_fill_ratio",
    "write_conformal_lattice_external_npz",
]
