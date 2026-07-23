import json

import numpy as np
import pytest

from external_npz_preprocessor.source_npz import load_source_npz
from external_npz_preprocessor.source_template import write_two_layer_template_npz


def test_loads_layer_material_paths_and_normalizes_nx3_to_xyzabc(tmp_path):
    source = tmp_path / "source_xyz.npz"
    resin_paths = np.array([
        [[1.0, 2.0, 0.5], [3.0, 4.0, 0.6]],
    ], dtype=np.float32)
    fiber_paths = np.full((1, 2, 6), np.nan, dtype=np.float32)
    fiber_paths[0, 0] = [5.0, 6.0, 0.7, 10.0, 20.0, 30.0]
    fiber_paths[0, 1] = [7.0, 8.0, 0.8, 11.0, 21.0, 31.0]
    np.savez(
        source,
        meta=np.array(json.dumps({"format": "external_layer_paths_v1"})),
        layer_0000_R=resin_paths,
        layer_0000_F=fiber_paths,
    )

    job = load_source_npz(source, default_abc=(1.0, 2.0, 3.0))

    assert job.meta["format"] == "external_layer_paths_v1"
    assert [layer.index for layer in job.layers] == [0]
    assert len(job.layers[0].resin_paths) == 1
    assert len(job.layers[0].fiber_paths) == 1
    assert job.layers[0].resin_paths[0].points.shape == (2, 6)
    np.testing.assert_allclose(
        job.layers[0].resin_paths[0].points[0],
        [1.0, 2.0, 0.5, 1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        job.layers[0].fiber_paths[0].points[1],
        [7.0, 8.0, 0.8, 11.0, 21.0, 31.0],
    )


def test_rejects_source_paths_without_z_columns(tmp_path):
    source = tmp_path / "bad_xy.npz"
    resin_paths = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    np.savez(source, layer_0000_R=resin_paths)

    with pytest.raises(ValueError, match="Nx3 or Nx6"):
        load_source_npz(source)


def test_two_layer_template_contains_numeric_xyz_resin_and_fiber_paths_per_layer(tmp_path):
    source = write_two_layer_template_npz(tmp_path / "template.npz")

    with np.load(source, allow_pickle=False) as raw:
        assert raw["layer_0000_R"].dtype == np.float32
        assert raw["layer_0000_R"].ndim == 3
        assert raw["layer_0000_R"].shape[2] == 3

    job = load_source_npz(source)

    assert job.meta["format"] == "external_layer_paths_v1"
    assert job.meta["template"] == "two_layer_resin_fiber"
    assert job.meta["point_columns"] == ["x", "y", "z"]
    assert [layer.index for layer in job.layers] == [0, 1]
    assert all(layer.resin_paths for layer in job.layers)
    assert all(layer.fiber_paths for layer in job.layers)
    assert job.layers[0].resin_paths[0].points.shape[1] == 6
    assert job.layers[1].fiber_paths[0].points.shape[1] == 6
