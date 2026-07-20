from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_sources_do_not_embed_machine_specific_absolute_paths():
    # The negative look-behind keeps URL schemes such as ``https://`` from
    # being mistaken for a Windows drive path.
    drive_path = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
    machine_home = re.compile(r"/(?:home|Users)/[^/\s]+/")

    checked_files = [
        PROJECT_ROOT / "pyproject.toml",
        *sorted((PROJECT_ROOT / "kuka_slicer").glob("*.py")),
    ]
    violations: list[str] = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        if drive_path.search(text) or machine_home.search(text):
            violations.append(str(path.relative_to(PROJECT_ROOT)))

    assert not violations, f"machine-specific absolute paths found in: {violations}"


def test_release_metadata_uses_portable_relative_project_files():
    metadata = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'readme = "README.md"' in metadata
    assert 'kuka-slicer = "kuka_slicer.cli:main"' in metadata
    assert (PROJECT_ROOT / "README.md").is_file()
    assert (PROJECT_ROOT / "LICENSE").is_file()
    assert (PROJECT_ROOT / "MANIFEST.in").is_file()
