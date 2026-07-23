import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for package in ("path_processing_core", "gcode_planner"):
    path = ROOT / package
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
