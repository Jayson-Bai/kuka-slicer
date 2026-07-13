import json

import numpy as np

from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths, paths_to_padded_array, write_external_source_npz


def test_paths_to_padded_array_uses_full_nan_rows():
    paths = [
        np.asarray([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5]], dtype=np.float32),
        np.asarray([[2, 2, 0.5], [3, 2, 0.5]], dtype=np.float32),
    ]

    result = paths_to_padded_array(paths)

    assert result.shape == (2, 3, 3)
    assert result.dtype == np.float32
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
            )
        ]
    )

    write_external_source_npz(job, output)

    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {"meta", "layer_0000_R"}
        meta = json.loads(str(archive["meta"]))
        assert meta["format"] == "external_layer_paths_v1"
        assert archive["layer_0000_R"].shape == (1, 2, 3)

