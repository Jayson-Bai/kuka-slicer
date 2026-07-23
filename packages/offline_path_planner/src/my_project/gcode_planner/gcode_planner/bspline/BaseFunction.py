import importlib
import sys

_core_module = importlib.import_module("path_processing_core.bspline.BaseFunction")

sys.modules[__name__] = _core_module
