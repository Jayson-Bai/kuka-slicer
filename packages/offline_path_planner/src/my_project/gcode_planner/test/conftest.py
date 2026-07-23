import sys
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[2]
for package in ("gcode_planner", "path_processing_core"):
    path = ROOT / package
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


if "rclpy" not in sys.modules:
    rclpy_stub = types.ModuleType("rclpy")
    rclpy_stub.init = lambda *args, **kwargs: None
    rclpy_stub.spin = lambda *args, **kwargs: None
    rclpy_stub.shutdown = lambda *args, **kwargs: None
    node_stub = types.ModuleType("rclpy.node")

    class _Node:
        def __init__(self, *args, **kwargs):
            pass

    node_stub.Node = _Node
    rclpy_stub.node = node_stub
    sys.modules["rclpy"] = rclpy_stub
    sys.modules["rclpy.node"] = node_stub
