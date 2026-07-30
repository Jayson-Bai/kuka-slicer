import json

import numpy as np
import pytest

from external_npz_preprocessor.source_npz import (
    SOURCE_NPZ_CONTRACT_ID,
    load_source_npz,
)
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
        meta=np.array(json.dumps({"format": SOURCE_NPZ_CONTRACT_ID})),
        layer_0000_R=resin_paths,
        layer_0000_F=fiber_paths,
    )

    job = load_source_npz(source, default_abc=(1.0, 2.0, 3.0))

    assert job.meta["format"] == SOURCE_NPZ_CONTRACT_ID
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


def test_loads_optional_cumulative_resin_e_with_xyz_padding(tmp_path):
    source = tmp_path / "source_with_e.npz"
    resin_paths = np.full((2, 3, 3), np.nan, dtype=np.float32)
    resin_paths[0, :3] = [
        [1.0, 2.0, 0.5],
        [3.0, 2.0, 0.5],
        [3.0, 4.0, 0.5],
    ]
    resin_paths[1, :2] = [
        [5.0, 6.0, 0.5],
        [7.0, 6.0, 0.5],
    ]
    resin_e = np.full((2, 3), np.nan, dtype=np.float32)
    resin_e[0, :3] = [100.0, 101.5, 104.0]
    resin_e[1, :2] = [104.0, 108.5]
    np.savez(
        source,
        meta=np.array(json.dumps({"format": SOURCE_NPZ_CONTRACT_ID})),
        layer_0000_R=resin_paths,
        layer_0000_R_E=resin_e,
    )

    job = load_source_npz(source)

    assert len(job.layers[0].resin_paths) == 2
    np.testing.assert_allclose(
        job.layers[0].resin_paths[0].extrusion,
        [100.0, 101.5, 104.0],
    )
    np.testing.assert_allclose(
        job.layers[0].resin_paths[1].extrusion,
        [104.0, 108.5],
    )


def test_loads_optional_layer_travel_paths(tmp_path):
    source = tmp_path / "source_with_travel.npz"
    travel_paths = np.full((2, 3, 3), np.nan, dtype=np.float32)
    travel_paths[0, :2] = [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]]
    travel_paths[1, :3] = [[2.0, 0.0, 0.5], [2.0, 1.0, 0.5], [3.0, 1.0, 0.5]]
    resin_paths = np.asarray([[[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]]], dtype=np.float32)
    np.savez(
        source,
        meta=np.array(json.dumps({
            "format": SOURCE_NPZ_CONTRACT_ID,
            "motion_order": {"0": [
                {"kind": "travel", "index": 0},
                {"kind": "deposit", "index": 0},
            ]},
        })),
        layer_0000_R=resin_paths,
        layer_0000_T=travel_paths,
    )

    job = load_source_npz(source)

    assert len(job.layers[0].travel_paths) == 2
    np.testing.assert_allclose(
        job.layers[0].travel_paths[1].points[-1],
        [3.0, 1.0, 0.5, 0.0, 0.0, 0.0],
    )
    assert job.meta["motion_order"]["0"][0]["kind"] == "travel"


def test_preserves_float64_source_precision(tmp_path):
    source = tmp_path / "float64_source.npz"
    resin_paths = np.asarray(
        [[[0.1234567890123, 2.0, 0.5], [1.0, 2.0000000000007, 0.5]]],
        dtype=np.float64,
    )
    np.savez(
        source,
        meta=np.array(json.dumps({
            "format": SOURCE_NPZ_CONTRACT_ID,
            "precision": "float64",
        })),
        layer_0000_R=resin_paths,
    )

    job = load_source_npz(source)

    points = job.layers[0].resin_paths[0].points
    assert points.dtype == np.float64
    np.testing.assert_array_equal(points[:, :3], resin_paths[0])


def test_rejects_source_paths_without_z_columns(tmp_path):
    source = tmp_path / "bad_xy.npz"
    resin_paths = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    np.savez(source, layer_0000_R=resin_paths)

    with pytest.raises(ValueError, match="Nx3 or Nx6"):
        load_source_npz(source)


def test_two_layer_template_contains_numeric_xyz_resin_and_fiber_paths_per_layer(tmp_path):
    source = write_two_layer_template_npz(tmp_path / "template.npz")

    with np.load(source, allow_pickle=False) as raw:
        assert raw["layer_0000_R"].dtype == np.float64
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


def test_rejects_explicit_unknown_source_contract(tmp_path):
    source = tmp_path / "future_source.npz"
    resin_paths = np.asarray(
        [[[1.0, 2.0, 0.5], [3.0, 4.0, 0.5]]],
        dtype=np.float32,
    )
    np.savez(
        source,
        meta=np.array(json.dumps({"format": "external_layer_paths_v999"})),
        layer_0000_R=resin_paths,
    )

    with pytest.raises(ValueError, match="unsupported source NPZ format"):
        load_source_npz(source)
