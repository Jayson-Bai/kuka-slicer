from __future__ import annotations

import json
from pathlib import Path
import struct



ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "printhead"


def test_printhead_glb_is_tcp_relative_and_uses_real_dimensions():
    metadata = json.loads(
        (ASSET_DIR / "printhead_interference_check.metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["tcp_cad_mm"] == [37.9767978, -109.3782, -29.0]
    assert metadata["cad_to_tool_axes"] == {
        "x_tool": "-y_cad",
        "y_tool": "+z_cad",
        "z_tool": "-x_cad",
    }
    assert metadata["model_bounds"]["size_mm"] == [74.50005, 50.0, 40.5]
    assert "collision_proxy" not in metadata
    assert "collision_proxies" not in metadata
    assert "clearance_cone" not in metadata
    components = metadata["model_components"]
    assert [component["name"] for component in components] == ["upper_housing", "heater_block", "nozzle"]
    heater = components[1]
    assert heater["local_bounds_mm"] == {
        "minimum_mm": [-19.50004, -11.0, -9.0000022],
        "maximum_mm": [-7.5, 5.0, 6.9999978],
    }
    assert heater["triangle_count"] == len(heater["triangle_indices"]) == 500
    assert metadata["triangle_count"] == 4612

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
    assert len(preview["positions"]) == 2256
    assert len(preview["triangles"]) == 4612
    assert preview["model_bounds"]["minimum_mm"] == [-74.50005, -25.0, -28.5000022]
    assert preview["model_bounds"]["maximum_mm"] == [-0.0, 25.0, 11.9999978]
    assert preview["model_components"] == preview["metadata"]["model_components"]
    assert "clearance_cone" not in preview
    assert "collision_boxes" not in preview
    assert max(max(triangle) for triangle in preview["triangles"]) < len(preview["positions"])
