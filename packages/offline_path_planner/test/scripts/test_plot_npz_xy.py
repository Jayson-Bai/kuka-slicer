from pathlib import Path
import importlib.util

import numpy as np


def _load_plot_npz_xy_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "plot_npz_xy.py"
    )
    spec = importlib.util.spec_from_file_location("plot_npz_xy", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_xy_defaults_to_positive_extrusion_print_segments(tmp_path):
    module = _load_plot_npz_xy_module()
    npz_path = tmp_path / "preview.npz"
    np.savez(
        npz_path,
        x=np.array([0.0, 10.0, 11.0, 12.0, 20.0, 21.0], dtype=np.float32),
        y=np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0], dtype=np.float32),
        e=np.array([0.0, 0.0, 1.0, 0.5, 0.5, 2.0], dtype=np.float32),
        move_type=np.array([0, 0, 1, 1, 0, 3], dtype=np.uint8),
        move_type_vocab_keys=np.array(
            [b"TRAVEL", b"PRINT", b"TRAVEL_FIT", b"PRINT_FIT"],
            dtype="S32",
        ),
        move_type_vocab_vals=np.array([0, 1, 2, 3], dtype=np.uint8),
    )

    xy = module.load_xy([npz_path])

    expected = np.array(
        [
            [10.0, 0.0],
            [11.0, 0.0],
            [np.nan, np.nan],
            [20.0, 5.0],
            [21.0, 5.0],
        ],
        dtype=np.float32,
    )
    assert xy.shape == expected.shape
    np.testing.assert_allclose(xy, expected, equal_nan=True)


def test_load_xy_can_include_all_points_for_diagnostics(tmp_path):
    module = _load_plot_npz_xy_module()
    npz_path = tmp_path / "preview.npz"
    np.savez(
        npz_path,
        x=np.array([0.0, 10.0, 11.0], dtype=np.float32),
        y=np.array([0.0, 1.0, 1.0], dtype=np.float32),
    )

    xy = module.load_xy([npz_path], include_travel=True)

    np.testing.assert_allclose(
        xy,
        np.array([[0.0, 0.0], [10.0, 1.0], [11.0, 1.0]], dtype=np.float32),
    )
