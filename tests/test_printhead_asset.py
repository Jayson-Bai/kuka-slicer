from __future__ import annotations

import json
from pathlib import Path
import struct



ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "printhead"


def test_printhead_glb_is_tcp_relative_and_uses_real_dimensions():
    metadata = json.loads(
        (ASSET_DIR / "printhead_interference_check.metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["tcp_cad_mm"] == [-68.2042631, -17.23385, 4.5505142]
    assert metadata["cad_to_tool_axes"] == {
        "x_tool": "-y_cad",
        "y_tool": "+z_cad",
        "z_tool": "-x_cad",
    }
    assert metadata["model_bounds"]["size_mm"] == [74.5, 50.0, 40.5]
    assert "collision_proxy" not in metadata
    assert "collision_proxies" not in metadata
    assert "clearance_cone" not in metadata
    components = metadata["model_components"]
    assert [component["name"] for component in components] == ["upper_housing", "heater_block", "nozzle"]
    heater = components[1]
    assert heater["local_bounds_mm"] == {
        "minimum_mm": [-16.9999955, -7.0000002, -11.0000031],
        "maximum_mm": [-5.0, 8.9999958, 4.9999969],
    }
    assert heater["triangle_count"] == len(heater["triangle_indices"]) == 500
    assert metadata["triangle_count"] == 4870
    assert metadata["source_stl_sha256"] == "6f6f662a3da5e705b6720e5204d6b0b8e244e21aa06f21bd13705602798da50f"
    assert metadata["source_cad_sha256"] == "d76518e6e2aa005d197b5a905149cd3361ed5926b1681fe3cf516999998539e6"

    glb = (ASSET_DIR / "printhead_interference_check.glb").read_bytes()
    magic, version, declared_length = struct.unpack_from("<4sII", glb)
    assert magic == b"glTF"
    assert version == 2
    assert declared_length == len(glb)


def test_printhead_canvas_mesh_keeps_exact_heater_component_contract():
    preview = json.loads(
        (ASSET_DIR / "printhead_interference_check.preview.json").read_text(encoding="utf-8")
    )

    assert preview["format"] == "kuka_printhead_preview_v3"
    assert preview["units"] == "millimeter"
    assert len(preview["positions"]) == 2384
    assert len(preview["triangles"]) == 4870
    assert preview["model_bounds"]["minimum_mm"] == [-74.5, -25.0000042, -28.5000031]
    assert preview["model_bounds"]["maximum_mm"] == [-0.0, 24.9999958, 11.9999969]
    assert preview["model_components"] == preview["metadata"]["model_components"]
    assert "clearance_cone" not in preview
    assert "collision_boxes" not in preview
    assert max(max(triangle) for triangle in preview["triangles"]) < len(preview["positions"])
