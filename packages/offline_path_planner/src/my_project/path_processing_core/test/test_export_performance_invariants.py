from __future__ import annotations

import numpy as np

from path_processing_core.npz_exporter import export_npz
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
