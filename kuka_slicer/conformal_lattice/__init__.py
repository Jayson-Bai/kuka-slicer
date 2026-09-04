"""Independent conformal-lattice structure geometry pipeline.

This package deliberately does not import or alter ``surface_mapper``.  Its
first gate establishes a validated triangular surface domain; lattice and path
generation are added by later gates.
"""

from .contracts import ConformalLatticeSpec, load_conformal_lattice_spec
from .distortion import ConformalQuality, evaluate_conformal_quality
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
    "ConformalQuality",
    "DesignFieldResult",
    "ExternalScalarMetadata",
    "FieldComponent",
    "LSCMParameterization",
    "OrientationField",
    "SurfaceMeshDomain",
    "build_double_sine_surface_domain",
    "build_orientation_field",
    "compose_design_fields",
    "constant_driver",
    "curvature_driver",
    "cut_mesh_along_edges",
    "evaluate_conformal_quality",
    "external_scalar_driver",
    "farthest_boundary_anchors",
    "load_conformal_lattice_spec",
    "load_triangle_mesh_domain",
    "prepare_surface_mesh_domain",
    "parameterize_lscm",
    "parameterize_spec_lscm",
    "roi_driver",
]
