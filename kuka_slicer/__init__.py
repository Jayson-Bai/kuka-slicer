"""STL to external source NPZ conversion tools."""

from .external_npz import ExternalSourceJob, MaterialPaths, write_external_source_npz
from .slicer import SliceConfig, slice_mesh_to_job
from .stl_io import Mesh, load_stl

__all__ = [
    "ExternalSourceJob",
    "MaterialPaths",
    "Mesh",
    "SliceConfig",
    "load_stl",
    "slice_mesh_to_job",
    "write_external_source_npz",
]

