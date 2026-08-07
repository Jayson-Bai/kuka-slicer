"""Independent planar-to-surface path mapping tools.

The mapper consumes the portable surface definition exported by
``surface_preview`` and an ``external_layer_paths_v1`` source NPZ.  It owns
layer progression and Z mapping, while leaving slicing and Core conversion
unchanged.
"""

from .contracts import SurfaceTarget, SourceNPZ, load_surface_target, read_source_npz
from .mapper import MappingResult, SurfaceMappingPlan, map_source_job
from .progression import LayerProgression
from .server import run_surface_mapper_server

__all__ = [
    "LayerProgression",
    "MappingResult",
    "SourceNPZ",
    "SurfaceMappingPlan",
    "SurfaceTarget",
    "load_surface_target",
    "map_source_job",
    "read_source_npz",
    "run_surface_mapper_server",
]
