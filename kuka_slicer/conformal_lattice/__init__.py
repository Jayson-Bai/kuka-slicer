"""Independent conformal-lattice structure geometry pipeline.

This package deliberately does not import or alter ``surface_mapper``.  Its
first gate establishes a validated triangular surface domain; lattice and path
generation are added by later gates.
"""

from .contracts import ConformalLatticeSpec, load_conformal_lattice_spec
from .mesh_domain import (
    SurfaceMeshDomain,
    build_double_sine_surface_domain,
    cut_mesh_along_edges,
    load_triangle_mesh_domain,
    prepare_surface_mesh_domain,
)

__all__ = [
    "ConformalLatticeSpec",
    "SurfaceMeshDomain",
    "build_double_sine_surface_domain",
    "cut_mesh_along_edges",
    "load_conformal_lattice_spec",
    "load_triangle_mesh_domain",
    "prepare_surface_mesh_domain",
]
