import numpy as np
import pytest

from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths, write_external_source_npz
from kuka_slicer.extrusion_compensator import compensate_extrusion
from kuka_slicer.surface_mapper.contracts import read_source_npz


def _source(tmp_path):
    output = tmp_path / "flat.npz"
    write_external_source_npz(
        ExternalSourceJob(
            material_paths=[
                MaterialPaths(
                    0,
                    "R",
                    [
                        np.asarray(
                            [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            [[4.0, 0.0, 0.5], [5.0, 0.0, 0.5]],
                            dtype=np.float64,
                        ),
                    ],
                    extrusion=[
                        np.asarray([4.0, 5.0, 7.0]),
                        np.asarray([10.0, 10.0]),
                    ],
                )
            ]
        ),
        output,
    )
    return read_source_npz(output.read_bytes())


def _curved_from(flat):
    arrays = {key: value.copy() for key, value in flat.arrays.items()}
    path = arrays["layer_0000_R"]
    for path_index in range(path.shape[0]):
        point_count = int(np.isfinite(path[path_index, :, 0]).sum())
        path[path_index, :point_count, 2] += np.arange(point_count, dtype=np.float64)
    return type(flat)(arrays=arrays, meta={"surface_mapping": {}}, source_name="curved.npz")


def test_compensator_replaces_existing_cumulative_e_by_3d_arc_length(tmp_path):
    flat = _source(tmp_path)
    result = compensate_extrusion(flat, _curved_from(flat))

    compensated = result.source.arrays["layer_0000_R_E"]
    assert compensated.shape == flat.arrays["layer_0000_R_E"].shape
    assert compensated[0, :3] == pytest.approx([4.0, 4.0 + np.sqrt(2.0), 4.0 + 3.0 * np.sqrt(2.0)])
    # The second path has a zero E increment and is intentionally not scaled.
    assert compensated[1, :2] == pytest.approx([10.0, 10.0])
    assert np.isnan(compensated[1, 2])
    assert result.replaced_arrays == ("layer_0000_R_E",)
    assert result.positive_segment_count == 2
    assert result.source.meta["surface_mapping"]["extrusion"] == "arc_length_ratio_compensated"
    assert result.source.meta["extrusion_compensation"]["format"] == "arc_length_ratio_v1"


def test_compensator_keeps_no_e_inputs_without_creating_an_e_array(tmp_path):
    output = tmp_path / "without_e.npz"
    write_external_source_npz(
        ExternalSourceJob(
            material_paths=[
                MaterialPaths(0, "R", [np.asarray([[0, 0, 0.5], [1, 0, 0.5]])])
            ]
        ),
        output,
    )
    flat = read_source_npz(output.read_bytes())
    result = compensate_extrusion(flat, _curved_from(flat))

    assert not any(key.endswith("_E") for key in result.source.arrays)
    assert result.replaced_arrays == ()


def test_compensator_rejects_a_pair_with_changed_xy_or_padding(tmp_path):
    flat = _source(tmp_path)
    curved = _curved_from(flat)
    curved.arrays["layer_0000_R"][0, 1, 0] += 0.1

    with pytest.raises(ValueError, match="differ in XY"):
        compensate_extrusion(flat, curved)


def test_compensator_rejects_a_pair_with_different_e_array_keys(tmp_path):
    flat = _source(tmp_path)
    curved = _curved_from(flat)
    del curved.arrays["layer_0000_R_E"]

    with pytest.raises(ValueError, match="same extrusion array keys"):
        compensate_extrusion(flat, curved)
