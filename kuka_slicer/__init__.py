"""STL to external source NPZ conversion tools."""

from .external_npz import (
    DEFAULT_EXPORT_CHORD_TOLERANCE_MM,
    ExternalSourceJob,
    MaterialPaths,
    simplify_job_paths_for_export,
    simplify_path_for_export,
    write_external_source_npz,
)
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
    "DEFAULT_EXPORT_CHORD_TOLERANCE_MM",
    "MaterialPaths",
    "Mesh",
    "PySLMConfig",
    "PySLMStrategyDefaults",
    "SliceConfig",
    "load_stl",
    "recommended_pyslm_strategy_defaults",
    "slice_mesh_to_job",
    "simplify_job_paths_for_export",
    "simplify_path_for_export",
    "write_external_source_npz",
]
