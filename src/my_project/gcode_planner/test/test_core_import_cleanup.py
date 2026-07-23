from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_gcode_production_code_uses_path_processing_core_for_shared_exports():
    files = [
        ROOT / "gcode_planner" / "gcode_planner" / "gcode_parser.py",
        ROOT / "gcode_planner" / "gcode_planner" / "cli.py",
        ROOT / "gcode_planner" / "gcode_planner" / "print_test_generator.py",
        ROOT / "my_project_ui" / "my_project_ui" / "ui_panel.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    assert "from path_processing_core.types import" in combined
    assert "from path_processing_core.npz_exporter import export_npz" in combined
    assert "from path_processing_core.head_calibration import" in combined
    assert "from gcode_planner.npz_exporter import export_npz" not in combined
    assert "from gcode_planner.head_calibration import" not in combined
