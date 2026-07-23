import sys

from path_processing_core import bspline_approximation as _core_module

sys.modules[__name__] = _core_module
