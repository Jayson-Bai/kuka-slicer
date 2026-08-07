import json

import numpy as np

from external_npz_preprocessor.export_runner import (
    convert_external_npz,
    convert_gcode,
)
from external_npz_preprocessor.process_params import ProcessParams, ResinProcessParams
from external_npz_preprocessor.source_gcode import load_source_gcode


_PRUSA_GCODE = """\
G90
M82
G92 X0 Y0 Z0 E0
G1 Z0.5 F900
;TYPE:External perimeter
G1 X1 Y0 E0.1
G1 X2 Y0 E0.2
G0 X3 Y0
;TYPE:Internal infill
G1 X4 Y0 E0.3
M83
G1 X5 Y0 E0.1
M82
G92 E0
G1 Z1.0
G0 X0 Y1
G1 X1 Y1 E0.2
"""


def _write_equivalent_npz(path, *, include_fiber=False):
    arrays = {}
    if include_fiber:
        arrays["layer_0000_F"] = np.array(
            [[[0.0, 2.0, 0.6], [1.0, 2.0, 0.6]]], dtype=np.float64
        )
        arrays["layer_0001_F"] = np.array(
            [[[0.0, 3.0, 1.1], [1.0, 3.0, 1.1]]], dtype=np.float64
        )
    np.savez(
        path,
        meta=np.array(
            json.dumps(
                {
                    "format": "external_layer_paths_v1",
                    "path_roles": {
                        "R": {"0": ["outer_contour", "infill"], "1": ["infill"]}
                    },
                    "motion_order": {
                        "0": [
                            {"kind": "deposit", "index": 0},
                            {"kind": "travel", "index": 0},
                            {"kind": "deposit", "index": 1},
                        ],
                        "1": [
                            {"kind": "travel", "index": 0},
                            {"kind": "deposit", "index": 0},
                        ],
                    },
                }
            )
        ),
        layer_0000_R=np.array(
            [
                [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [2.0, 0.0, 0.5]],
                [[3.0, 0.0, 0.5], [4.0, 0.0, 0.5], [5.0, 0.0, 0.5]],
            ],
            dtype=np.float64,
        ),
        layer_0000_R_E=np.array([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]], dtype=np.float64),
        layer_0000_T=np.array([[[2.0, 0.0, 0.5], [3.0, 0.0, 0.5]]], dtype=np.float64),
        layer_0001_R=np.array([[[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float64),
        layer_0001_R_E=np.array([[0.4, 0.6]], dtype=np.float64),
        layer_0001_T=np.array([[[5.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], dtype=np.float64),
        **arrays,
    )


def _calibration(path):
    path.write_text(
        json.dumps({"resin": {"z_print_compensation_mm": 0.0}, "fiber": {}}),
        encoding="utf-8",
    )


def _params():
    return ProcessParams(
        primeline_enabled=False,
        dt=0.05,
        resin=ResinProcessParams(
            feed_mm_s=10.0,
            first_layer_feed_mm_s=10.0,
            prime_length_mm=0.0,
            retract_length_mm=0.0,
        ),
    )


def test_prusa_gcode_adapter_matches_native_source_path_grouping_and_e(tmp_path):
    gcode = tmp_path / "native.gcode"
    gcode.write_text(_PRUSA_GCODE, encoding="utf-8")

    job = load_source_gcode(gcode)

    assert [layer.index for layer in job.layers] == [0, 1]
    assert [len(layer.resin_paths) for layer in job.layers] == [2, 1]
    assert [len(layer.travel_paths) for layer in job.layers] == [1, 1]
    np.testing.assert_allclose(job.layers[0].resin_paths[0].extrusion, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(job.layers[0].resin_paths[1].extrusion, [0.2, 0.3, 0.4])
    np.testing.assert_allclose(job.layers[1].resin_paths[0].extrusion, [0.4, 0.6])
    assert job.meta["motion_order"] == {
        "0": [
            {"kind": "deposit", "index": 0},
            {"kind": "travel", "index": 0},
            {"kind": "deposit", "index": 1},
        ],
        "1": [
            {"kind": "travel", "index": 0},
            {"kind": "deposit", "index": 0},
        ],
    }


def test_gcode_and_equivalent_external_npz_produce_byte_identical_core_npz(tmp_path):
    gcode = tmp_path / "native.gcode"
    source_npz = tmp_path / "source.npz"
    output_gcode = tmp_path / "gcode_core.npz"
    output_gcode_repeat = tmp_path / "gcode_core_repeat.npz"
    output_npz = tmp_path / "source_core.npz"
    calibration = tmp_path / "calibration.json"
    gcode.write_text(_PRUSA_GCODE, encoding="utf-8")
    _write_equivalent_npz(source_npz)
    _calibration(calibration)

    convert_gcode(gcode, output_gcode, _params(), calibration_path=calibration)
    convert_gcode(gcode, output_gcode_repeat, _params(), calibration_path=calibration)
    convert_external_npz(source_npz, output_npz, _params(), calibration_path=calibration)

    assert output_gcode.read_bytes() == output_gcode_repeat.read_bytes()
    assert output_gcode.read_bytes() == output_npz.read_bytes()


def test_gcode_with_expanded_fiber_paths_matches_external_npz_byte_for_byte(tmp_path):
    gcode = tmp_path / "native.gcode"
    source_npz = tmp_path / "source_with_fiber.npz"
    output_gcode = tmp_path / "gcode_fiber_core.npz"
    output_npz = tmp_path / "source_fiber_core.npz"
    calibration = tmp_path / "calibration.json"
    fiber_paths_by_layer = {
        0: [np.array([[0.0, 2.0, 0.6], [1.0, 2.0, 0.6]])],
        1: [np.array([[0.0, 3.0, 1.1], [1.0, 3.0, 1.1]])],
    }
    gcode.write_text(_PRUSA_GCODE, encoding="utf-8")
    _write_equivalent_npz(source_npz, include_fiber=True)
    _calibration(calibration)

    convert_gcode(
        gcode,
        output_gcode,
        _params(),
        calibration_path=calibration,
        fiber_paths_by_layer=fiber_paths_by_layer,
    )
    convert_external_npz(source_npz, output_npz, _params(), calibration_path=calibration)

    assert output_gcode.read_bytes() == output_npz.read_bytes()
