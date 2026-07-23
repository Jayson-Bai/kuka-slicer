import sys

from path_processing_core import npz_exporter as _core_module

sys.modules[__name__] = _core_module
