import numpy as np
import pytest

pytest.importorskip("pyslm")
pytest.importorskip("trimesh")

from kuka_slicer.slicer import PySLMConfig, SliceConfig, slice_mesh_to_job
from kuka_slicer.stl_io import Mesh


def _cube_triangles(size: float) -> np.ndarray:
    s = float(size)
    vertices = np.asarray(
        [
            [0, 0, 0],
            [s, 0, 0],
            [s, s, 0],
            [0, s, 0],
            [0, 0, s],
            [s, 0, s],
            [s, s, s],
            [0, s, s],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices[faces]


@pytest.mark.parametrize("hatcher", ("basic", "stripe", "island", "basic_island"))
def test_native_pyslm_hatchers_preserve_project_output_contract(hatcher: str):
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(
        layer_height=5.0,
        line_width=2.0,
        slicing_kernel="pyslm",
        infill_pattern="rectilinear",
        pyslm=PySLMConfig(hatcher=hatcher, hatch_sort="alternate"),  # type: ignore[arg-type]
    )

    job = slice_mesh_to_job(mesh, config)

    assert job.material_paths
    assert job.meta["slicing"]["slicing_kernel"] == "pyslm"
    assert job.meta["slicing"]["pyslm"]["hatcher"] == hatcher
    assert job.meta["slicing"]["pyslm"]["native_patterns"]
    assert all(path.shape[1] == 3 for group in job.material_paths for path in group.paths)
