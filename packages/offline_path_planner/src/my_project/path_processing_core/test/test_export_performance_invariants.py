from __future__ import annotations

import json

import numpy as np

from path_processing_core.npz_exporter import export_npz
from path_processing_core.polynomial_interpolator import _arc_length_map_sample_count
from path_processing_core.types import MoveCommand, Position


def test_default_export_keeps_one_final_marker_per_path(tmp_path) -> None:
    """The O(1) marker index must preserve the consumer-visible path contract."""

    p0 = Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    p1 = Position(4.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    p2 = Position(4.0, 3.0, 0.5, 0.0, 0.0, 0.0)
    p3 = Position(0.0, 3.0, 0.5, 0.0, 0.0, 0.0)
    commands = [
        MoveCommand("PRINT", "G1", p0, p1, 4.0, 4.0, 600.0, 1, layer=0, subtype="RESIN"),
        MoveCommand("TRAVEL", "G0", p1, p2, 0.0, 0.0, 600.0, 2, layer=0, subtype="TRAVEL"),
        MoveCommand("PRINT", "G1", p2, p3, 8.0, 4.0, 600.0, 3, layer=0, subtype="RESIN"),
    ]
    output = tmp_path / "marker.npz"

    stats = export_npz(commands, str(output), dt=0.1, chunk_size=1_000)

    assert stats["detailed_sampling_timing"] is False
    with np.load(output, allow_pickle=False) as data:
        path_ids = data["path_id"]
        end_flags = data["path_end_flag"]
        for path_id in np.unique(path_ids[path_ids > 0]):
            indices = np.flatnonzero(path_ids == path_id)
            assert indices.size > 0
            assert end_flags[indices].sum() == 1
            assert end_flags[indices[-1]] == 1


def test_detailed_sampling_timing_remains_available_on_request(tmp_path) -> None:
    p0 = Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    p1 = Position(1.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    command = MoveCommand("PRINT", "G1", p0, p1, 1.0, 1.0, 600.0, 1)

    fast_output = tmp_path / "fast.npz"
    timed_output = tmp_path / "timed.npz"
    fast_stats = export_npz([command], str(fast_output), dt=0.1)
    stats = export_npz(
        [command],
        str(timed_output),
        dt=0.1,
        collect_detailed_timing=True,
    )

    assert fast_stats["detailed_sampling_timing"] is False
    assert stats["detailed_sampling_timing"] is True
    with np.load(fast_output, allow_pickle=False) as fast, np.load(
        timed_output, allow_pickle=False
    ) as timed:
        assert set(fast.files) == set(timed.files)
        for key in fast.files:
            assert np.array_equal(fast[key], timed[key])


def test_dense_spline_uses_refined_arc_length_map_budget() -> None:
    """High-density fitted curves need a sufficiently fine physical arc map."""

    assert _arc_length_map_sample_count(19) == 800
    assert _arc_length_map_sample_count(1_191) == 28_584


def test_travel_waypoints_share_one_short_polyline_profile(tmp_path) -> None:
    """Detour vertices must not become independent four-second stop profiles."""

    p0 = Position(0.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    p1 = Position(1.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    p2 = Position(1.0, 1.0, 0.5, 0.0, 0.0, 0.0)
    p3 = Position(2.0, 1.0, 0.5, 0.0, 0.0, 0.0)
    commands = [
        MoveCommand("TRAVEL", "G0", p0, p1, 0.0, 0.0, 1200.0, 1, layer=0, subtype="TRAVEL"),
        MoveCommand("TRAVEL", "G0", p1, p2, 0.0, 0.0, 1200.0, 2, layer=0, subtype="TRAVEL"),
        MoveCommand("TRAVEL", "G0", p2, p3, 0.0, 0.0, 1200.0, 3, layer=0, subtype="TRAVEL"),
    ]
    output = tmp_path / "travel_polyline.npz"

    export_npz(commands, str(output), dt=0.004, chunk_size=1_000)

    with np.load(output, allow_pickle=False) as data:
        path_ids = data["path_id"]
        end_flags = data["path_end_flag"]
        points = np.column_stack((data["x"], data["y"], data["z"]))
    assert np.unique(path_ids[path_ids > 0]).tolist() == [1]
    assert end_flags.sum() == 1
    # Samples stay on the original L-shaped polyline, not a spline shortcut.
    assert np.all(
        np.isclose(points[:, 1], 0.0)
        | np.isclose(points[:, 0], 1.0)
        | np.isclose(points[:, 1], 1.0)
    )
    # At the UI-selected 20 mm/s travel feed and 4 ms RSI period, each step
    # remains below the upper-computer 0.100020 mm limit without post-sampling
    # subdivision.
    assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) <= 0.10002

    timing = json.loads(output.with_suffix(".timing.json").read_text(encoding="utf-8"))
    segment = timing["segments"][0]
    assert segment["duration_s"] < 1.0
    assert segment["t_acc_s"] == segment["t_dec_s"]
    assert segment["t_acc_s"] < 2.0
