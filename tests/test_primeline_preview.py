from types import SimpleNamespace

import numpy as np

from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths, TravelPaths
from kuka_slicer.slicer import SliceConfig
from kuka_slicer.ui_server import _core_preview_overlay_from_commands, _preview_payload


def _point(x, y):
    return SimpleNamespace(x=x, y=y, z=0.5)


def test_core_overlay_uses_actual_startup_and_post_primeline_travels():
    overlay = _core_preview_overlay_from_commands(
        [
            SimpleNamespace(raw="external_npz_start_xy_travel", type="TRAVEL", layer=0,
                            start_pos=_point(0, 0), pos=_point(0, -10)),
            SimpleNamespace(raw="external_npz_primeline", type="PRINT", subtype="RESIN_PRINT",
                            layer=0, start_pos=_point(0, -10), pos=_point(80, -10), control_points=[]),
            SimpleNamespace(raw="external_npz_travel", type="TRAVEL", layer=0,
                            start_pos=_point(80, -10), pos=_point(10, 5)),
        ]
    )

    assert [entry["role"] for entry in overlay["sequence"]] == [
        "core_travel", "primeline", "core_travel"
    ]
    assert [entry["anchor"] for entry in overlay["sequence"]] == [0, 0, 0]
    assert overlay["sequence"][0]["points"] == [[0.0, 0.0, 0.5], [0.0, -10.0, 0.5]]
    assert overlay["sequence"][2]["points"] == [[80.0, -10.0, 0.5], [10.0, 5.0, 0.5]]


def test_source_frame_startup_placeholders_do_not_enter_preview_or_bounds():
    retained = np.asarray([[20.0, 5.0, 0.5], [25.0, 8.0, 0.5]])
    travels = [
        np.asarray([[-85.0, -50.0, 0.5], [0.0, 0.0, 0.5]]),
        np.asarray([[0.0, 0.0, 0.5], [85.0, 30.0, 0.5]]),
        np.asarray([[10.0, 5.0, 0.5], [9.0, 5.0, 0.5]]),
        retained,
    ]
    job = ExternalSourceJob(
        material_paths=[MaterialPaths(0, "R", [np.asarray([[10.0, 5.0, 0.5], [20.0, 5.0, 0.5]])])],
        travel_paths=[TravelPaths(0, travels)],
        meta={
            "startup_travel_count": 1,
            "preview_startup_travel_count": 2,
            "motion_order": {"0": [
                {"kind": "travel", "index": index} for index in range(4)
            ] + [{"kind": "deposit", "index": 0}]},
        },
    )

    preview = _preview_payload(
        None,
        SliceConfig(line_width=2.0, start_x_mm=15.0, start_y_mm=5.0),
        job,
        hide_startup_source_travel=True,
        hide_initial_prusa_travel=True,
    )

    layer = preview["layers"][0]
    assert layer["travel_paths"] == [retained.tolist()]
    assert [entry["points"] for entry in layer["motion_paths"] if entry["kind"] == "travel"] == [
        retained.tolist()
    ]
    assert preview["bounds"]["min_x"] == 0.0
    assert preview["bounds"]["min_y"] == 0.0
