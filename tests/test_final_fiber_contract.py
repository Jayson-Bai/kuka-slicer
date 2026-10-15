"""Regressions for JSON fiber geometry and final-NPZ-only UI previews."""
import numpy as np
import pytest

from kuka_slicer.slicer import SliceConfig
from kuka_slicer.ui_server import (
    _ensure_offline_planner_import_paths,
    _preview_payload_from_final_core_npz,
)


@pytest.mark.parametrize("density", [0, 6])
@pytest.mark.parametrize("offset", [(0., 0.), (50., 30.)])
def test_fiber_export_preserves_source_polyline_without_fitting(tmp_path, monkeypatch, density, offset):
    _ensure_offline_planner_import_paths()
    from path_processing_core import npz_exporter
    from path_processing_core.types import MoveCommand, Position

    # Unevenly spaced points and a tight return exposed the old whole-stroke fit.
    xy = np.array([[0., 0.], [.002, 0.], [2., 0.], [2., .01],
                   [2., 2.], [1., 2.], [1., .5]]) + offset
    poses = [Position(x, y, 1., 0., 0., 0.) for x, y in xy]
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    commands = [MoveCommand(
        "PRINT", "G1", a, b, float(e), float(de), 1200., i,
        subtype="FIBER_PRINT", source_path_id="json:0",
    ) for i, (a, b, e, de) in enumerate(zip(poses, poses[1:], np.cumsum(lengths), lengths))]
    captured = []
    sample = npz_exporter.sample_global_curve_iter

    def record(curve, **kwargs):
        captured.append(curve)
        return sample(curve, **kwargs)

    def forbid_fit(*args, **kwargs):
        pytest.fail("Fiber must never enter the spline fitter")

    monkeypatch.setattr(npz_exporter.GlobalSplinePlanner, "fit_global_curve", forbid_fit)
    monkeypatch.setattr(npz_exporter, "sample_global_curve_iter", record)
    output = tmp_path / "fiber.npz"
    npz_exporter.export_npz(commands, str(output), initial_tool_id=1, density=density, dt=.004)
    assert len(captured) == 1  # One timing law; source vertices aren't start/stop boundaries.
    assert captured[0].cmd == "POLYLINE"
    np.testing.assert_array_equal(
        [[p.x, p.y] for p in [captured[0].start_pos, *captured[0].control_points]], xy,
    )
    with np.load(output) as data:
        samples = np.c_[data['x'], data['y']].astype(float)
        # Every output point lies on a source segment in monotonic source order.
        edges = np.diff(xy, axis=0)
        ratios = np.clip(np.sum((samples[:, None] - xy[:-1]) * edges, axis=2) /
                         np.sum(edges * edges, axis=1), 0., 1.)
        projections = xy[:-1] + ratios[:, :, None] * edges
        distances = np.linalg.norm(samples[:, None] - projections, axis=2)
        nearest = distances.argmin(axis=1)
        assert distances.min(axis=1).max() < 1e-5
        stations = np.r_[0., np.cumsum(lengths)][nearest] + ratios[np.arange(len(samples)), nearest] * lengths[nearest]
        assert np.diff(stations).min() >= -1e-5
        assert data['e'][-1] == pytest.approx(lengths.sum(), abs=1e-5)
        assert np.all(data['move_type'] == 1)  # PRINT, never PRINT_FIT.
        assert np.unique(data['path_id']).size == 1


@pytest.mark.parametrize("primeline_origin", [(0., 0.), (40., -10.)])
def test_final_preview_never_replaces_or_invents_primeline_geometry(tmp_path, primeline_origin):
    output = tmp_path / "final.npz"
    # Config requests a different straight primeline; only this curved exported run exists.
    points = np.array([[0., 0., .5], [40., 3., .5], [80., 0., .5]])
    np.savez(output, x=points[:, 0], y=points[:, 1], z=points[:, 2],
             tool_id=np.full(3, 2), move_type=np.ones(3), path_id=np.ones(3))
    preview = _preview_payload_from_final_core_npz(
        output, SliceConfig(), primeline_enabled=True,
        primeline_origin=primeline_origin, primeline_length_mm=80.,
    )
    entries = preview['layers'][0]['motion_paths']
    assert len(entries) == 1
    assert entries[0]['points'] == points.tolist()


def test_final_preview_keeps_collinear_reversals_even_below_display_tolerance(tmp_path):
    output = tmp_path / "backtracking.npz"
    x = np.array([0., 1., 1., .99, 2.])
    np.savez(output, x=x, y=np.zeros(5), z=np.ones(5),
             tool_id=np.ones(5), move_type=np.ones(5), path_id=np.ones(5))
    preview = _preview_payload_from_final_core_npz(output, SliceConfig())
    points = np.array(preview['layers'][0]['fiber_paths'][0])
    assert np.any(np.diff(points[:, 0]) < 0.)
    assert set(points[:, 0]).issubset(set(x))
