from path_processing_core.rsi_timing import RsiTimingAccumulator


def test_trajectory_rows_advance_by_dt_but_event_rows_do_not():
    timing = RsiTimingAccumulator(dt=0.004)
    assert timing.trajectory_time() == 0.0
    assert timing.append_trajectory_time() == 0.0
    assert timing.append_event_time() == 0.0
    assert timing.append_trajectory_time() == 0.004
    assert timing.append_trajectory_time() == 0.008
    assert timing.trajectory_rows == 3
    assert timing.event_rows_ignored == 1


def test_segment_metadata_preserves_acc_flat_dec_and_sequence_range():
    timing = RsiTimingAccumulator(dt=0.1)
    timing.start_segment(path_id=7, move_type="TRAVEL", start_seq=10)
    timing.append_trajectory_time()
    timing.append_trajectory_time()
    timing.finish_segment(t_acc_s=2.0, t_flat_s=3.5, t_dec_s=2.0, end_seq=11)

    assert timing.segments == [{
        "path_id": 7,
        "move_type": "TRAVEL",
        "start_seq": 10,
        "end_seq": 11,
        "duration_s": 0.1,
        "t_acc_s": 2.0,
        "t_flat_s": 3.5,
        "t_dec_s": 2.0,
    }]


def test_summary_is_json_serializable_and_reports_total_time():
    timing = RsiTimingAccumulator(dt=0.25)
    timing.append_trajectory_time()
    timing.append_trajectory_time()
    summary = timing.summary()

    assert summary["format"] == "rsi_print_timing"
    assert summary["version"] == 1
    assert summary["sample_period_s"] == 0.25
    assert summary["total_planned_time_s"] == 0.25
    assert summary["trajectory_rows"] == 2
