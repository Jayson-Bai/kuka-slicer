import ast
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from kuka_slicer.external_npz import (
    SOURCE_NPZ_CONTRACT_ID,
    ExternalSourceJob,
    MaterialPaths,
    write_external_source_npz,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ROOT = REPOSITORY_ROOT / "packages" / "offline_path_planner"
OFFLINE_SOURCE_ROOTS = (
    OFFLINE_ROOT / "src" / "my_project" / "path_processing_core",
    OFFLINE_ROOT / "src" / "my_project" / "gcode_planner",
    OFFLINE_ROOT / "src" / "my_project" / "external_npz_preprocessor",
)
OFFLINE_NAMESPACES = {
    "path_processing_core",
    "gcode_planner",
    "external_npz_preprocessor",
}


def _top_level_imports(source_root):
    imports = set()
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.partition(".")[0])
    return imports


def _planner_environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in OFFLINE_SOURCE_ROOTS)
    return env


def test_production_packages_have_no_cross_imports():
    slicer_imports = _top_level_imports(REPOSITORY_ROOT / "kuka_slicer")
    assert slicer_imports.isdisjoint(OFFLINE_NAMESPACES)

    for source_root in OFFLINE_SOURCE_ROOTS:
        assert "kuka_slicer" not in _top_level_imports(source_root)


def test_distribution_metadata_keeps_installations_independent():
    root_config = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    offline_config = tomllib.loads(
        (OFFLINE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert root_config["project"]["name"] == "kuka-slicer"
    assert root_config["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "kuka_slicer*"
    ]
    assert offline_config["project"]["name"] == "kuka-offline-planner"
    assert all(
        not dependency.lower().startswith("kuka-slicer")
        for dependency in offline_config["project"]["dependencies"]
    )


def test_slicer_source_npz_runs_through_offline_planner_contract(tmp_path):
    source_path = tmp_path / "source.npz"
    output_path = tmp_path / "system.npz"
    path = np.asarray(
        [[0.0, 0.0, 0.5], [2.0, 0.0, 0.5], [2.0, 2.0, 0.5]],
        dtype=np.float32,
    )
    write_external_source_npz(
        ExternalSourceJob(
            material_paths=[MaterialPaths(0, "R", [path])],
        ),
        source_path,
    )

    with np.load(source_path, allow_pickle=False) as source:
        assert SOURCE_NPZ_CONTRACT_ID in str(source["meta"])

    conversion = subprocess.run(
        [
            sys.executable,
            "-m",
            "external_npz_preprocessor.cli",
            "--source",
            str(source_path),
            "--out",
            str(output_path),
            "--dt",
            "0.02",
        ],
        cwd=OFFLINE_ROOT,
        env=_planner_environment(),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert conversion.returncode == 0, conversion.stderr
    assert output_path.is_file()

    validation = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from path_processing_core.npz_contract import "
                "validate_system_npz_contract; "
                "print(validate_system_npz_contract(sys.argv[1]))"
            ),
            str(output_path),
        ],
        cwd=OFFLINE_ROOT,
        env=_planner_environment(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr
    assert validation.stdout.strip() == "1"
