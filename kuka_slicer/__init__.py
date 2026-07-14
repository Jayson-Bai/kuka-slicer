"""STL to external source NPZ conversion tools."""

from .external_npz import ExternalSourceJob, MaterialPaths, write_external_source_npz
from .slicer import (
    PySLMConfig,
    PySLMStrategyDefaults,
    SliceConfig,
    recommended_pyslm_strategy_defaults,
    slice_mesh_to_job,
)
from .stl_io import Mesh, load_stl

__all__ = [
    "ExternalSourceJob",
    "MaterialPaths",
    "Mesh",
    "PySLMConfig",
    "PySLMStrategyDefaults",
    "SliceConfig",
    "load_stl",
    "recommended_pyslm_strategy_defaults",
    "slice_mesh_to_job",
    "write_external_source_npz",
]
