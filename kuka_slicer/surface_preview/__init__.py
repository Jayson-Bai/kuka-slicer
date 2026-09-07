"""Standalone tools for inspecting parameterised deposition surfaces.

This package deliberately has no dependency on the slicer or the integrated
web UI.  It can therefore be embedded by a later UI integration, run through
its own local server, or removed without changing slicing behaviour.
"""

from .model import DoubleSineSurface, SurfaceGrid
from .server import run_surface_preview_server

__all__ = ["DoubleSineSurface", "SurfaceGrid", "run_surface_preview_server"]
