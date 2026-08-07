import json

import numpy as np

from kuka_slicer.external_npz import (
    DEFAULT_EXPORT_CHORD_TOLERANCE_MM,
    ExternalSourceJob,
    MaterialPaths,
    TravelPaths,
    paths_to_padded_array,
    simplify_path_for_export,
    write_external_source_npz,
)


def test_paths_to_padded_array_uses_full_nan_rows():
    paths = [
        np.asarray([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5]], dtype=np.float32),
        np.asarray([[2, 2, 0.5], [3, 2, 0.5]], dtype=np.float32),
    ]

    result = paths_to_padded_array(paths)

    assert result.shape == (2, 3, 3)
    assert result.dtype == np.float64
    assert np.isnan(result[1, 2]).all()
    assert not np.isnan(result[0]).any()


def test_write_external_source_npz(tmp_path):
    output = tmp_path / "source.npz"
    job = ExternalSourceJob(
        material_paths=[
            MaterialPaths(
                0,
                "R",
                [np.asarray([[0, 0, 0.5], [1, 0, 0.5]], dtype=np.float32)],
                extrusion=[np.asarray([0.0, 1.25], dtype=np.float32)],
            )
        ],
        travel_paths=[
            TravelPaths(
                0,
                [np.asarray([[1, 0, 0.5], [2, 1, 0.5]], dtype=np.float32)],
            )
        ],
    )

    write_external_source_npz(job, output)

    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "meta",
            "layer_0000_R",
            "layer_0000_R_E",
            "layer_0000_F",
            "layer_0000_T",
        }
        meta = json.loads(str(archive["meta"]))
        assert meta["format"] == "external_layer_paths_v1"
        assert meta["precision"] == "float64"
        assert meta["path_sampling"]["simplification"] == "none"
        assert meta["layer_semantics"] == {
            "format": "logical_layer_v1",
            "layer_key": "logical_deposition_layer",
            "z_coordinate": "per_point_trajectory",
            "ordering": "ascending_layer_key_then_source_path_order",
            "reconstruct_layers_from_z": False,
        }
        assert archive["layer_0000_R"].shape == (1, 2, 3)
        assert archive["layer_0000_R"].dtype == np.float64
        assert np.allclose(archive["layer_0000_R_E"], [[0.0, 1.25]])
        assert archive["layer_0000_R_E"].dtype == np.float64
        assert archive["layer_0000_F"].shape == (0, 0, 3)
        assert archive["layer_0000_T"].shape == (1, 2, 3)
        assert archive["layer_0000_T"].dtype == np.float64


def test_write_source_preserves_float64_points_and_point_count(tmp_path):
    output = tmp_path / "high_precision_source.npz"
    path = np.asarray(
        [[0.1234567890123, 0.0, 0.5], [0.5, 0.0000000000007, 0.5], [1.0, 0.0, 0.5]],
        dtype=np.float64,
    )
    job = ExternalSourceJob(
        material_paths=[MaterialPaths(0, "R", [path])],
    )

    write_external_source_npz(job, output)

    with np.load(output, allow_pickle=False) as archive:
        written = archive["layer_0000_R"][0]
        assert written.dtype == np.float64
        assert written.shape == path.shape
        assert np.array_equal(written, path)


def test_export_sampling_keeps_only_straight_endpoints():
    path = np.asarray(
        [[float(index), 0.0, 0.5] for index in range(101)],
        dtype=np.float32,
    )

    simplified = simplify_path_for_export(path)

    assert simplified.shape == (2, 3)
    assert np.array_equal(simplified[0], path[0])
    assert np.array_equal(simplified[-1], path[-1])


def test_export_sampling_preserves_arc_shape_with_sparse_points():
    angles = np.linspace(0.0, np.pi * 0.5, 181)
    path = np.column_stack(
        (
            10.0 * np.cos(angles),
            10.0 * np.sin(angles),
            np.full_like(angles, 0.5),
        )
    ).astype(np.float32)

    simplified = simplify_path_for_export(path)

    assert 3 < simplified.shape[0] < 20
    radii = np.linalg.norm(simplified[:, :2], axis=1)
    assert np.allclose(radii, 10.0, atol=DEFAULT_EXPORT_CHORD_TOLERANCE_MM)
    assert np.array_equal(simplified[0], path[0])
    assert np.array_equal(simplified[-1], path[-1])


def test_export_sampling_preserves_closed_contours():
    angles = np.linspace(0.0, np.pi * 2.0, 361)
    path = np.column_stack(
        (
            5.0 * np.cos(angles),
            5.0 * np.sin(angles),
            np.full_like(angles, 0.5),
        )
    ).astype(np.float32)

    simplified = simplify_path_for_export(path)

    assert 8 <= simplified.shape[0] < 40
    assert np.array_equal(simplified[0], simplified[-1])
    assert np.allclose(
        np.linalg.norm(simplified[:-1, :2], axis=1),
        5.0,
        atol=DEFAULT_EXPORT_CHORD_TOLERANCE_MM,
    )
