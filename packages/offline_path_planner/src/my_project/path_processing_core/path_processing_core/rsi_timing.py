"""Lightweight timing metadata for the RSI trajectory stream.

The accumulator models only rows consumed by RSI. Event rows are retained in
the NPZ for control semantics, but do not advance the planned motion clock.
"""

from __future__ import annotations


class RsiTimingAccumulator:
    """Accumulate the planned time represented by exported trajectory rows."""

    def __init__(self, dt: float):
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.dt = float(dt)
        self._time_s = 0.0
        self.trajectory_rows = 0
        self.event_rows_ignored = 0
        self.segments = []
        self._open_segment = None

    def append_trajectory_time(self) -> float:
        """Record one RSI trajectory row and return its cumulative timestamp."""
        value = self._time_s
        if self.trajectory_rows:
            self._time_s += self.dt
            value = self._time_s
        self.trajectory_rows += 1
        return value

    def append_event_time(self) -> float:
        """Return the current timestamp for a non-time-consuming event row."""
        self.event_rows_ignored += 1
        return self._time_s

    def trajectory_time(self) -> float:
        return self._time_s

    def start_segment(self, *, path_id: int, move_type: str, start_seq: int) -> None:
        if self._open_segment is not None:
            raise ValueError("cannot start a nested timing segment")
        self._open_segment = {
            "path_id": int(path_id),
            "move_type": str(move_type),
            "start_seq": int(start_seq),
            "start_time_s": self._time_s,
        }

    def finish_segment(
        self,
        *,
        t_acc_s: float,
        t_flat_s: float,
        t_dec_s: float,
        end_seq: int,
    ) -> None:
        if self._open_segment is None:
            raise ValueError("cannot finish a timing segment that is not open")
        segment = self._open_segment
        duration_s = max(0.0, self._time_s - segment.pop("start_time_s"))
        self.segments.append({
            "path_id": segment["path_id"],
            "move_type": segment["move_type"],
            "start_seq": segment["start_seq"],
            "end_seq": int(end_seq),
            "duration_s": duration_s,
            "t_acc_s": float(t_acc_s),
            "t_flat_s": float(t_flat_s),
            "t_dec_s": float(t_dec_s),
        })
        self._open_segment = None

    def summary(self) -> dict:
        """Return JSON-ready timing metadata for the exported NPZ."""
        return {
            "format": "rsi_print_timing",
            "version": 1,
            "sample_period_s": self.dt,
            "total_planned_time_s": self._time_s,
            "trajectory_rows": self.trajectory_rows,
            "event_rows_ignored": self.event_rows_ignored,
            "segments": list(self.segments),
        }
