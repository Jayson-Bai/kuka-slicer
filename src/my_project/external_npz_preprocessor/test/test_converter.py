from pathlib import Path

import numpy as np
import pytest

import external_npz_preprocessor.converter as converter
from external_npz_preprocessor.converter import source_job_to_parsed_commands
from external_npz_preprocessor.process_params import (
    FiberProcessParams,
    ProcessParams,
    RESIN_FILAMENT_LENGTH_PER_MM3,
    ResinProcessParams,
)
from external_npz_preprocessor.source_npz import LayerPaths, MaterialPath, SourceJob
from path_processing_core.polynomial_interpolator import sample_global_curve_iter
from path_processing_core.types import (
    ExtrudeWait,
    GlobalCurveCommand,
    MCommand,
    MoveCommand,
    ResetECommand,
    ToolChangeCommand,
)


def _source_curves(commands):
    return [
        cmd
        for cmd in commands
        if isinstance(cmd, GlobalCurveCommand) and cmd.raw != "external_npz_primeline"
    ]


def _params():
    return ProcessParams(
        resin=ResinProcessParams(
            layer_height_mm=0.5,
            extrusion_scale=2.0,
            feed_mm_s=10.0,
        ),
        fiber=FiberProcessParams(
            extrusion_scale=0.25,
            feed_mm_s=6.0,
            first_layer_feed_mm_s=6.0,
        ),
        travel_feed_mm_s=20.0,
    )


def _straight_path(material, order, start_x, end_x, *, y=0.0, z=0.5):
    return MaterialPath(
        material=material,
        order=order,
        points=np.array(
            [
                [start_x, y, z, 0.0, 0.0, 0.0],
                [end_x, y, z, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def _job_with_paths(*, resin_paths=(), fiber_paths=()):
    return SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=list(resin_paths),
                fiber_paths=list(fiber_paths),
            )
        ],
    )


@pytest.mark.parametrize(
    "travel_feed_mm_s",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
    ids=["zero", "negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_rejects_non_finite_or_non_positive_travel_feed(travel_feed_mm_s):
    job = _job_with_paths(
        resin_paths=[_straight_path("R", 0, 0.0, 10.0)]
    )

    with pytest.raises(
        ValueError,
        match=r"^travel_feed_mm_s must be finite and > 0$",
    ):
        source_job_to_parsed_commands(
            job,
            ProcessParams(travel_feed_mm_s=travel_feed_mm_s),
        )


def test_converts_ordered_resin_and_fiber_paths_to_planner_commands_without_overriding_source_z():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [[0.0, 0.0, 2.5, 0.0, 0.0, 0.0],
                             [10.0, 0.0, 2.6, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[
                    MaterialPath(
                        material="F",
                        order=1,
                        points=np.array(
                            [[10.0, 0.0, 3.1, 0.0, 0.0, 0.0],
                             [10.0, 4.0, 3.3, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
            )
        ],
    )

    commands = source_job_to_parsed_commands(job, _params())

    tool_changes = [cmd for cmd in commands if isinstance(cmd, ToolChangeCommand)]
    tool_change_resets = [
        cmd
        for cmd in commands
        if isinstance(cmd, ResetECommand) and cmd.raw == "G92 E0"
    ]
    curves = _source_curves(commands)
    moves = [cmd for cmd in commands if isinstance(cmd, MoveCommand)]
    travel_moves = [
        cmd for cmd in moves
        if cmd.type == "TRAVEL" and cmd.raw == "external_npz_travel"
    ]
    assert [cmd.tool for cmd in tool_changes] == [1, 0]
    assert len(tool_change_resets) == 2
    assert [curve.subtype for curve in curves] == ["RESIN_PRINT", "FIBER_PRINT"]
    assert [curve.cmd for curve in curves] == ["POLYLINE", "POLYLINE"]
    assert curves[0].feedrate == 600.0
    assert round(curves[0].delta_e, 6) == round(
        (10.0 ** 2 + 0.1 ** 2) ** 0.5
        * 2.0
        * 0.5
        * 2.0
        * RESIN_FILAMENT_LENGTH_PER_MM3,
        6,
    )
    assert curves[1].feedrate == 360.0
    assert round(curves[1].delta_e, 6) == round((4.0 ** 2 + 0.2 ** 2) ** 0.5 * 0.25, 6)
    assert curves[0].start_pos.z == pytest.approx(2.5)
    assert curves[0].control_points[-1].z == pytest.approx(2.6)
    assert curves[1].start_pos.z == pytest.approx(3.1)
    assert curves[1].control_points[-1].z == pytest.approx(3.3)
    assert travel_moves[-1].start_pos.z == pytest.approx(2.6)
    assert travel_moves[-1].pos.z == pytest.approx(3.1)
    assert curves[0].layer == 0
    assert curves[1].layer == 0


def test_first_material_layers_and_destination_travels_use_dedicated_speeds():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[_straight_path("R", 0, 0.0, 10.0)],
                fiber_paths=[],
            ),
            LayerPaths(
                index=1,
                resin_paths=[_straight_path("R", 0, 20.0, 30.0, z=1.0)],
                fiber_paths=[_straight_path("F", 0, 40.0, 50.0, z=1.1)],
            ),
            LayerPaths(
                index=2,
                resin_paths=[],
                fiber_paths=[_straight_path("F", 0, 60.0, 70.0, z=1.6)],
            ),
        ],
    )
    params = ProcessParams(
        resin=ResinProcessParams(
            feed_mm_s=11.0,
            first_layer_feed_mm_s=2.0,
        ),
        fiber=FiberProcessParams(
            feed_mm_s=12.0,
            first_layer_feed_mm_s=3.0,
        ),
        travel_feed_mm_s=13.0,
        first_layer_travel_feed_mm_s=4.0,
    )

    commands = source_job_to_parsed_commands(job, params)

    curves = [cmd for cmd in commands if isinstance(cmd, GlobalCurveCommand)]
    travels = [
        cmd
        for cmd in commands
        if isinstance(cmd, MoveCommand)
        and cmd.type == "TRAVEL"
        and cmd.raw != "external_npz_resin_layer_end_travel"
    ]
    layer_end_travels = [
        cmd
        for cmd in commands
        if isinstance(cmd, MoveCommand)
        and cmd.raw == "external_npz_resin_layer_end_travel"
    ]
    assert [curve.feedrate for curve in curves] == [
        120.0,  # primeline uses first-layer resin speed
        120.0,  # first resin-bearing layer
        660.0,  # later resin layer
        180.0,  # first fiber-bearing layer, independently detected
        720.0,  # later fiber layer
    ]
    assert [travel.feedrate for travel in travels] == [
        240.0,  # initial positioning to the primeline
        240.0,  # primeline to first resin layer
        780.0,  # destination is a later resin layer
        240.0,  # destination is the first fiber layer
        780.0,  # destination is a later fiber layer
    ]
    assert [travel.feedrate for travel in layer_end_travels] == [
        240.0,  # first resin-bearing layer
        780.0,  # later resin layer
    ]


def test_marks_only_fiber_curves_with_custom_start_acceleration_time():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                             [10.0, 0.0, 0.5, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[
                    MaterialPath(
                        material="F",
                        order=1,
                        points=np.array(
                            [[10.0, 0.0, 0.6, 0.0, 0.0, 0.0],
                             [20.0, 0.0, 0.6, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
            )
        ],
    )
    params = ProcessParams(fiber=FiberProcessParams(start_accel_s=4.5))

    curves = _source_curves(source_job_to_parsed_commands(job, params))

    resin_curve = next(curve for curve in curves if curve.subtype == "RESIN_PRINT")
    fiber_curve = next(curve for curve in curves if curve.subtype == "FIBER_PRINT")
    assert resin_curve.time_acc_s is None
    assert fiber_curve.time_acc_s == pytest.approx(4.5)


def test_inserts_resin_primeline_as_first_regular_path():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [[5.0, 20.0, 0.5, 0.0, 0.0, 0.0],
                             [15.0, 20.0, 0.5, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[],
            )
        ],
    )
    params = _params()

    commands = source_job_to_parsed_commands(job, params)

    curves = [cmd for cmd in commands if isinstance(cmd, GlobalCurveCommand)]
    extrusion_waits = [
        cmd
        for cmd in commands
        if isinstance(cmd, ExtrudeWait) and abs(cmd.delta_e) > 1e-9
    ]
    assert len(curves) == 2
    primeline = curves[0]
    assert primeline.subtype == "RESIN_PRINT"
    assert primeline.raw == "external_npz_primeline"
    assert primeline.start_pos.x == pytest.approx(params.start_x_mm)
    assert primeline.start_pos.y == pytest.approx(params.start_y_mm + params.primeline_y_mm)
    assert primeline.start_pos.z == pytest.approx(params.resin.layer_height_mm)
    assert primeline.control_points[-1].x == pytest.approx(params.start_x_mm + params.primeline_x_mm + params.primeline_length_mm)
    assert primeline.control_points[-1].y == pytest.approx(params.start_y_mm + params.primeline_y_mm)
    assert primeline.delta_e == pytest.approx(params.primeline_length_mm * params.resin.e_per_mm())
    assert [(cmd.delta_e, cmd.subtype) for cmd in extrusion_waits[:3]] == [
        (-15.0, "RESIN_PRINT"),
        (18.0, "RESIN_PRINT"),
        (-15.0, "RESIN_PRINT"),
    ]


def test_resin_primeline_uses_configured_relative_position_and_length():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [[5.0, 20.0, 0.5, 0.0, 0.0, 0.0],
                             [15.0, 20.0, 0.5, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[],
            )
        ],
    )
    params = ProcessParams(
        start_x_mm=50.0,
        start_y_mm=60.0,
        primeline_x_mm=4.0,
        primeline_y_mm=-15.0,
        primeline_length_mm=80.0,
    )

    curves = [
        cmd
        for cmd in source_job_to_parsed_commands(job, params)
        if isinstance(cmd, GlobalCurveCommand)
    ]

    primeline = curves[0]
    assert primeline.raw == "external_npz_primeline"
    assert primeline.start_pos.x == pytest.approx(54.0)
    assert primeline.start_pos.y == pytest.approx(45.0)
    assert primeline.control_points[-1].x == pytest.approx(134.0)
    assert primeline.control_points[-1].y == pytest.approx(45.0)
    assert primeline.delta_e == pytest.approx(80.0 * params.resin.e_per_mm())


def test_adds_prime_before_paths_and_retract_after_resin_paths():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                             [10.0, 0.0, 0.5, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[
                    MaterialPath(
                        material="F",
                        order=0,
                        points=np.array(
                            [[10.0, 0.0, 0.6, 0.0, 0.0, 0.0],
                             [20.0, 0.0, 0.6, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
            )
        ],
    )

    waits = [
        cmd
        for cmd in source_job_to_parsed_commands(job, ProcessParams())
        if isinstance(cmd, ExtrudeWait) and abs(cmd.delta_e) > 1e-9
    ]

    assert [(cmd.delta_e, cmd.feedrate, cmd.subtype) for cmd in waits] == [
        (-15.0, 1800.0, "RESIN_PRINT"),
        (18.0, 900.0, "RESIN_PRINT"),
        (-15.0, 1800.0, "RESIN_PRINT"),
        (18.0, 900.0, "RESIN_PRINT"),
        (-15.0, 1800.0, "RESIN_PRINT"),
        (-10.0, 300.0, "FIBER_PRINT"),
        (12.0, 300.0, "FIBER_PRINT"),
        (-10.0, 300.0, "FIBER_PRINT"),
    ]
    assert [round(cmd.wait_sec, 6) for cmd in waits] == [
        0.5,
        1.2,
        0.5,
        1.2,
        0.5,
        2.0,
        2.4,
        2.0,
    ]


def test_final_resin_path_travels_20mm_outward_before_tool_change():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        "R",
                        0,
                        np.array(
                            [
                                [0.0, 0.0, 2.5, 1.0, 2.0, 3.0],
                                [0.0, 10.0, 2.5, 1.0, 2.0, 3.0],
                            ],
                            dtype=np.float32,
                        ),
                    ),
                    MaterialPath(
                        "R",
                        1,
                        np.array(
                            [
                                [10.0, 0.0, 2.5, 1.0, 2.0, 3.0],
                                [10.0, 10.0, 2.5, 1.0, 2.0, 3.0],
                            ],
                            dtype=np.float32,
                        ),
                    ),
                ],
                fiber_paths=[_straight_path("F", 0, 4.0, 6.0, y=5.0, z=2.6)],
            )
        ],
    )
    params = ProcessParams(first_layer_travel_feed_mm_s=4.0)

    commands = source_job_to_parsed_commands(job, params)
    resin_curves = [
        curve
        for curve in _source_curves(commands)
        if curve.subtype == "RESIN_PRINT"
    ]
    layer_end_travels = [
        cmd
        for cmd in commands
        if isinstance(cmd, MoveCommand)
        and cmd.raw == "external_npz_resin_layer_end_travel"
    ]

    assert len(layer_end_travels) == 1
    final_curve = resin_curves[-1]
    final_index = commands.index(final_curve)
    curve, retract, reset, anchor, travel = commands[final_index:final_index + 5]
    assert curve is final_curve
    assert isinstance(retract, ExtrudeWait)
    assert retract.raw == "external_npz_retract"
    assert isinstance(reset, ResetECommand)
    assert reset.raw == "external_npz_path_reset"
    assert isinstance(anchor, ExtrudeWait)
    assert anchor.raw == "external_npz_reset_anchor"
    assert travel is layer_end_travels[0]
    assert travel.e_val == pytest.approx(0.0)
    assert travel.delta_e == pytest.approx(0.0)
    assert travel.feedrate == pytest.approx(240.0)
    assert travel.start_pos == final_curve.control_points[-1]
    assert travel.pos.x == pytest.approx(10.0 + 20.0 / np.sqrt(2.0))
    assert travel.pos.y == pytest.approx(10.0 + 20.0 / np.sqrt(2.0))
    assert travel.pos.z == pytest.approx(2.5)
    assert (travel.pos.a, travel.pos.b, travel.pos.c) == pytest.approx((1.0, 2.0, 3.0))

    fiber_tool_change_index = next(
        index
        for index, cmd in enumerate(commands)
        if isinstance(cmd, ToolChangeCommand) and cmd.tool == 0
    )
    assert commands.index(travel) < fiber_tool_change_index


def test_layer_end_outward_travel_is_added_only_to_resin_bearing_layers():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[_straight_path("R", 0, 0.0, 10.0)],
                fiber_paths=[],
            ),
            LayerPaths(
                index=1,
                resin_paths=[],
                fiber_paths=[_straight_path("F", 0, 0.0, 10.0, y=5.0, z=1.0)],
            ),
            LayerPaths(
                index=2,
                resin_paths=[_straight_path("R", 0, 20.0, 30.0, z=1.5)],
                fiber_paths=[],
            ),
        ],
    )

    commands = source_job_to_parsed_commands(job, ProcessParams())
    layer_end_travels = [
        cmd
        for cmd in commands
        if isinstance(cmd, MoveCommand)
        and cmd.raw == "external_npz_resin_layer_end_travel"
    ]

    assert [travel.layer for travel in layer_end_travels] == [0, 2]
    assert all(
        np.hypot(
            travel.pos.x - travel.start_pos.x,
            travel.pos.y - travel.start_pos.y,
        )
        == pytest.approx(20.0)
        for travel in layer_end_travels
    )


def test_resin_path_end_resets_then_anchors_before_travel_with_zero_e():
    job = _job_with_paths(
        resin_paths=[
            _straight_path("R", 0, 0.0, 10.0),
            _straight_path("R", 1, 20.0, 30.0),
        ]
    )

    commands = source_job_to_parsed_commands(job, ProcessParams())
    first_path = _source_curves(commands)[0]
    path_index = commands.index(first_path)
    curve, retract, reset, anchor, travel = commands[path_index:path_index + 5]

    assert curve is first_path
    assert isinstance(retract, ExtrudeWait)
    assert retract.raw == "external_npz_retract"
    assert isinstance(reset, ResetECommand)
    assert reset.raw == "external_npz_path_reset"
    assert reset.val == pytest.approx(0.0)
    assert reset.pose == curve.control_points[-1]
    assert isinstance(anchor, ExtrudeWait)
    assert anchor.raw == "external_npz_reset_anchor"
    assert anchor.wait_sec == pytest.approx(0.004)
    assert anchor.delta_e == pytest.approx(0.0)
    assert anchor.feedrate == pytest.approx(600.0)
    assert isinstance(travel, MoveCommand)
    assert travel.raw == "external_npz_travel"
    assert travel.e_val == pytest.approx(0.0)
    assert [reset.line, anchor.line, travel.line] == [
        retract.line + 1,
        retract.line + 2,
        retract.line + 3,
    ]


def test_next_path_primes_from_zero_at_destination_then_settles_before_print():
    job = _job_with_paths(
        resin_paths=[
            _straight_path("R", 0, 0.0, 10.0),
            _straight_path("R", 1, 20.0, 30.0),
        ]
    )
    params = ProcessParams(prime_settle_s=0.75)

    commands = source_job_to_parsed_commands(job, params)
    next_path = _source_curves(commands)[1]
    path_index = commands.index(next_path)
    travel, prime, settle, curve = commands[path_index - 3:path_index + 1]

    assert isinstance(travel, MoveCommand)
    assert travel.raw == "external_npz_travel"
    assert travel.e_val == pytest.approx(0.0)
    assert isinstance(prime, ExtrudeWait)
    assert prime.raw == "external_npz_prime"
    assert prime.delta_e == pytest.approx(params.resin.prime_length_mm)
    assert travel.pos == curve.start_pos
    assert isinstance(settle, ExtrudeWait)
    assert settle.raw == "external_npz_prime_settle"
    assert settle.wait_sec == pytest.approx(0.75)
    assert settle.delta_e == pytest.approx(0.0)
    assert settle.feedrate == pytest.approx(prime.feedrate)
    assert settle.line == prime.line + 1
    assert curve.e_val == pytest.approx(prime.delta_e + curve.delta_e)


def test_each_printable_path_has_reset_pair_including_primeline_and_final_path():
    job = _job_with_paths(
        resin_paths=[
            _straight_path("R", 0, 0.0, 10.0),
            _straight_path("R", 1, 20.0, 30.0),
        ],
        fiber_paths=[_straight_path("F", 0, 40.0, 50.0, z=0.6)],
    )

    commands = source_job_to_parsed_commands(job, ProcessParams())
    printable_paths = [
        cmd
        for cmd in commands
        if isinstance(cmd, GlobalCurveCommand) and cmd.type == "PRINT"
    ]
    assert len(printable_paths) == 4
    boundary_pairs = []
    boundary_table = []
    for path_number, curve in enumerate(printable_paths):
        path_index = commands.index(curve)
        next_path_index = (
            commands.index(printable_paths[path_number + 1])
            if path_number + 1 < len(printable_paths)
            else len(commands)
        )
        path_tail = commands[path_index + 1:next_path_index]
        resets = [
            cmd
            for cmd in path_tail
            if isinstance(cmd, ResetECommand)
            and cmd.raw == "external_npz_path_reset"
        ]
        assert len(resets) == 1
        reset = resets[0]
        reset_index = path_tail.index(reset)
        assert reset_index + 1 < len(path_tail)
        anchor = path_tail[reset_index + 1]
        assert isinstance(anchor, ExtrudeWait)
        assert anchor.raw == "external_npz_reset_anchor"
        assert path_tail.index(anchor) == path_tail.index(reset) + 1
        assert reset.pose == curve.control_points[-1]
        boundary_pairs.append((reset, anchor))
        boundary_table.append(
            (
                "primeline"
                if curve.raw == "external_npz_primeline"
                else curve.subtype,
                reset.raw,
                anchor.raw,
            )
        )

    assert boundary_table == [
        ("primeline", "external_npz_path_reset", "external_npz_reset_anchor"),
        ("RESIN_PRINT", "external_npz_path_reset", "external_npz_reset_anchor"),
        ("RESIN_PRINT", "external_npz_path_reset", "external_npz_reset_anchor"),
        ("FIBER_PRINT", "external_npz_path_reset", "external_npz_reset_anchor"),
    ]
    assert commands[-2:] == list(boundary_pairs[-1])
    assert [cmd.line for cmd in commands] == list(range(len(commands)))


def test_zero_prime_settle_s_keeps_prime_without_settle():
    job = _job_with_paths(
        resin_paths=[_straight_path("R", 0, 0.0, 10.0)]
    )

    commands = source_job_to_parsed_commands(
        job, ProcessParams(prime_settle_s=0.0)
    )
    primes = [
        cmd
        for cmd in commands
        if isinstance(cmd, ExtrudeWait) and cmd.raw == "external_npz_prime"
    ]
    settles = [
        cmd
        for cmd in commands
        if isinstance(cmd, ExtrudeWait)
        and cmd.raw == "external_npz_prime_settle"
    ]

    assert len(primes) == 2
    assert settles == []


def test_zero_prime_length_suppresses_prime_and_settle():
    job = _job_with_paths(
        resin_paths=[_straight_path("R", 0, 0.0, 10.0)]
    )
    params = ProcessParams(
        resin=ResinProcessParams(prime_length_mm=0.0),
        prime_settle_s=0.75,
    )

    commands = source_job_to_parsed_commands(job, params)

    assert not any(
        isinstance(cmd, ExtrudeWait)
        and cmd.raw in {"external_npz_prime", "external_npz_prime_settle"}
        for cmd in commands
    )


def test_fiber_ui_actions_apply_only_at_layer_boundaries_with_absolute_resets():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[],
                fiber_paths=[
                    _straight_path("F", 0, 0.0, 10.0, z=0.6),
                    _straight_path("F", 1, 20.0, 30.0, z=0.6),
                ],
            ),
            LayerPaths(
                index=1,
                resin_paths=[],
                fiber_paths=[
                    _straight_path("F", 0, 40.0, 50.0, z=1.2),
                    _straight_path("F", 1, 60.0, 70.0, z=1.2),
                ],
            ),
        ],
    )

    params = ProcessParams(
        fiber=FiberProcessParams(
            prime_length_mm=6.0,
            retract_length_mm=4.0,
            prime_speed_mm_s=3.0,
            retract_speed_mm_s=2.0,
        ),
        prime_settle_s=0.75,
    )
    commands = source_job_to_parsed_commands(job, params)
    fiber_paths = [
        cmd
        for cmd in commands
        if isinstance(cmd, GlobalCurveCommand)
        and cmd.subtype == "FIBER_PRINT"
    ]
    assert len(fiber_paths) == 4
    assert all(
        fiber_path.e_val - fiber_path.delta_e == pytest.approx(0.0)
        for fiber_path in fiber_paths
    )

    primes = [
        cmd for cmd in commands
        if isinstance(cmd, ExtrudeWait) and cmd.raw == "external_npz_prime"
        and cmd.subtype == "FIBER_PRINT"
    ]
    initial_retracts = [
        cmd for cmd in commands
        if isinstance(cmd, ExtrudeWait)
        and cmd.raw == "external_npz_fiber_initial_retract"
    ]
    layer_retracts = [
        cmd for cmd in commands
        if isinstance(cmd, ExtrudeWait)
        and cmd.raw == "external_npz_fiber_layer_retract"
    ]
    assert [(cmd.layer, cmd.delta_e) for cmd in primes] == [
        (0, pytest.approx(6.0)),
        (1, pytest.approx(6.0)),
    ]
    assert [(cmd.layer, cmd.delta_e) for cmd in initial_retracts] == [
        (0, pytest.approx(-4.0)),
    ]
    assert [(cmd.layer, cmd.delta_e) for cmd in layer_retracts] == [
        (0, pytest.approx(-4.0)),
        (1, pytest.approx(-4.0)),
    ]

    first_fiber_index = commands.index(fiber_paths[0])
    assert [cmd.raw for cmd in commands[first_fiber_index - 9:first_fiber_index]] == [
        "external_npz_fiber_prepare_reset",
        "external_npz_reset_anchor",
        "external_npz_fiber_initial_retract",
        "external_npz_fiber_prime_reset",
        "external_npz_reset_anchor",
        "external_npz_prime",
        "external_npz_prime_settle",
        "external_npz_fiber_print_reset",
        "external_npz_reset_anchor",
    ]

    second_layer_first_index = commands.index(fiber_paths[2])
    assert [
        cmd.raw for cmd in commands[second_layer_first_index - 6:second_layer_first_index]
    ] == [
        "external_npz_fiber_prime_reset",
        "external_npz_reset_anchor",
        "external_npz_prime",
        "external_npz_prime_settle",
        "external_npz_fiber_print_reset",
        "external_npz_reset_anchor",
    ]

    for layer_last_curve in (fiber_paths[1], fiber_paths[3]):
        curve_index = commands.index(layer_last_curve)
        cut, retract_reset, retract_anchor, retract, path_reset, path_anchor = commands[
            curve_index + 1:curve_index + 7
        ]
        assert isinstance(cut, MCommand) and cut.code == "CUT"
        assert isinstance(retract_reset, ResetECommand)
        assert retract_reset.raw == "external_npz_fiber_layer_retract_reset"
        assert retract_anchor.raw == "external_npz_reset_anchor"
        assert retract.raw == "external_npz_fiber_layer_retract"
        assert isinstance(path_reset, ResetECommand)
        assert path_reset.raw == "external_npz_path_reset"
        assert path_anchor.raw == "external_npz_reset_anchor"


def test_single_fiber_path_is_both_layer_first_and_layer_last():
    job = _job_with_paths(
        fiber_paths=[_straight_path("F", 0, 0.0, 10.0, z=0.6)]
    )

    commands = source_job_to_parsed_commands(job, ProcessParams())

    relevant_raws = [
        cmd.raw for cmd in commands
        if (cmd.raw or "").startswith("external_npz_fiber_")
    ]
    assert relevant_raws.count("external_npz_fiber_initial_retract") == 1
    assert relevant_raws.count("external_npz_fiber_layer_retract") == 1


def test_initializes_both_heads_before_first_path_and_resets_after_tool_change():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                             [10.0, 0.0, 0.5, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[
                    MaterialPath(
                        material="F",
                        order=0,
                        points=np.array(
                            [[10.0, 0.0, 0.6, 0.0, 0.0, 0.0],
                             [20.0, 0.0, 0.6, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
            )
        ],
    )

    commands = source_job_to_parsed_commands(job, ProcessParams())

    startup_events = [cmd for cmd in commands[:4] if isinstance(cmd, MCommand)]
    assert [(cmd.code, cmd.tool, cmd.params) for cmd in startup_events] == [
        ("M106", 1, {"T": 1.0}),
        ("M106", 0, {"T": 0.0}),
        ("M104", 1, {"S": 250.0, "T": 1.0}),
        ("M104", 0, {"S": 250.0, "T": 0.0}),
    ]

    command_kinds = [
        type(cmd).__name__
        for cmd in commands
        if isinstance(cmd, (MCommand, ToolChangeCommand, ResetECommand))
    ]
    assert command_kinds[:6] == [
        "MCommand",
        "MCommand",
        "MCommand",
        "MCommand",
        "ToolChangeCommand",
        "ResetECommand",
    ]

    fiber_tool_idx = next(
        idx for idx, cmd in enumerate(commands)
        if isinstance(cmd, ToolChangeCommand) and cmd.tool == 0
    )
    assert isinstance(commands[fiber_tool_idx + 1], ResetECommand)
    assert commands[fiber_tool_idx + 1].raw == "G92 E0"

    tool_change_resets = [
        cmd
        for cmd in commands
        if isinstance(cmd, ResetECommand) and cmd.raw == "G92 E0"
    ]
    path_resets = [
        cmd
        for cmd in commands
        if isinstance(cmd, ResetECommand)
        and cmd.raw == "external_npz_path_reset"
    ]
    assert len(tool_change_resets) == 2
    assert len(path_resets) == 3


def test_skips_fiber_startup_events_when_job_has_no_fiber_paths():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                             [10.0, 0.0, 0.5, 0.0, 0.0, 0.0]],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[],
            )
        ],
    )

    commands = source_job_to_parsed_commands(job, ProcessParams())

    startup_events = [cmd for cmd in commands if isinstance(cmd, MCommand)]
    assert [(cmd.code, cmd.tool, cmd.params) for cmd in startup_events] == [
        ("M106", 1, {"T": 1.0}),
        ("M104", 1, {"S": 250.0, "T": 1.0}),
    ]


def test_process_layer_heights_are_extrusion_references_only_not_z_generation():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[MaterialPath("R", 0, np.array([[0.0, 0.0, 8.0, 0.0, 0.0, 0.0], [1.0, 0.0, 8.2, 0.0, 0.0, 0.0]], dtype=np.float32))],
                fiber_paths=[MaterialPath("F", 0, np.array([[0.0, 1.0, 9.0, 0.0, 0.0, 0.0], [1.0, 1.0, 9.4, 0.0, 0.0, 0.0]], dtype=np.float32))],
            ),
        ],
    )
    params = ProcessParams(
        resin=ResinProcessParams(layer_height_mm=0.01),
        fiber=FiberProcessParams(layer_height_mm=99.0),
    )

    moves = _source_curves(source_job_to_parsed_commands(job, params))

    assert [(cmd.subtype, pytest.approx(cmd.start_pos.z), pytest.approx(cmd.control_points[-1].z)) for cmd in moves] == [
        ("RESIN_PRINT", pytest.approx(8.0), pytest.approx(8.2)),
        ("FIBER_PRINT", pytest.approx(9.0), pytest.approx(9.4)),
    ]


def test_start_xy_offsets_source_paths_and_inserts_initial_travel_without_z_override():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [
                                [0.0, 0.0, 2.5, 1.0, 2.0, 3.0],
                                [10.0, 0.0, 2.6, 1.0, 2.0, 3.0],
                            ],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[],
            )
        ],
    )
    params = ProcessParams(
        travel_feed_mm_s=20.0,
        default_a=4.0,
        default_b=5.0,
        default_c=6.0,
        start_x_mm=50.0,
        start_y_mm=60.0,
    )

    moves = [
        cmd for cmd in source_job_to_parsed_commands(job, params)
        if isinstance(cmd, MoveCommand)
    ]
    travel_moves = [
        cmd
        for cmd in moves
        if cmd.type == "TRAVEL"
        and cmd.raw != "external_npz_resin_layer_end_travel"
    ]
    print_curves = _source_curves(source_job_to_parsed_commands(job, params))

    assert len(travel_moves) == 2
    assert travel_moves[0].raw == "external_npz_start_xy_travel"
    assert travel_moves[0].start_pos.x == pytest.approx(0.0)
    assert travel_moves[0].start_pos.y == pytest.approx(0.0)
    assert travel_moves[0].start_pos.z == pytest.approx(0.5)
    assert travel_moves[0].pos.x == pytest.approx(50.0)
    assert travel_moves[0].pos.y == pytest.approx(50.0)
    assert travel_moves[0].pos.z == pytest.approx(0.5)
    assert (
        travel_moves[0].start_pos.a,
        travel_moves[0].start_pos.b,
        travel_moves[0].start_pos.c,
    ) == (4.0, 5.0, 6.0)
    assert print_curves[0].start_pos.x == pytest.approx(50.0)
    assert print_curves[0].start_pos.y == pytest.approx(60.0)
    assert print_curves[0].start_pos.z == pytest.approx(2.5)
    assert print_curves[0].control_points[-1].x == pytest.approx(60.0)
    assert print_curves[0].control_points[-1].y == pytest.approx(60.0)
    assert print_curves[0].control_points[-1].z == pytest.approx(2.6)


def test_external_npz_start_xy_places_part_lower_left_at_requested_position():
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [
                                [-10.0, 5.0, 0.5, 0.0, 0.0, 0.0],
                                [-5.0, 5.0, 0.5, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                    ),
                    MaterialPath(
                        material="R",
                        order=1,
                        points=np.array(
                            [
                                [-8.0, -3.0, 0.5, 0.0, 0.0, 0.0],
                                [-6.0, -3.0, 0.5, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                    ),
                ],
                fiber_paths=[],
            )
        ],
    )

    curves = _source_curves(
        source_job_to_parsed_commands(
            job, ProcessParams(start_x_mm=50.0, start_y_mm=60.0)
        )
    )
    points = []
    for curve in curves:
        points.append(curve.start_pos)
        points.extend(curve.control_points)

    assert min(point.x for point in points) == pytest.approx(50.0)
    assert min(point.y for point in points) == pytest.approx(60.0)
    assert curves[0].start_pos.x == pytest.approx(50.0)
    assert curves[0].start_pos.y == pytest.approx(68.0)


def test_external_npz_print_path_uses_polyline_fast_path_without_extra_bspline_fit(monkeypatch):
    class FakePlanner:
        def __init__(self):
            self.last_fit_profile = {}

        def fit_global_curve(self, moves, **kwargs):
            return GlobalCurveCommand(
                type="PRINT_FIT",
                cmd="SPLINE",
                start_pos=moves[0].start_pos,
                control_points=[move.pos for move in moves],
                e_val=moves[-1].e_val,
                delta_e=sum(move.delta_e for move in moves),
                feedrate=moves[0].feedrate,
                line=moves[0].line,
                raw="fake_fit",
                original_moves=list(moves),
            )

    monkeypatch.setattr(converter, "GlobalSplinePlanner", FakePlanner)
    monkeypatch.setattr(converter, "_curve_max_bidirectional_error_mm", lambda curve, points: 0.01, raising=False)
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [
                                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [1.0, 0.2, 0.5, 0.0, 0.0, 0.0],
                                [2.0, 0.4, 0.5, 0.0, 0.0, 0.0],
                                [3.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                                [4.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[],
            )
        ],
    )

    curves = _source_curves(
        source_job_to_parsed_commands(job, ProcessParams(spline_max_error_mm=0.05))
    )

    assert len(curves) == 1
    assert curves[0].cmd == "POLYLINE"
    assert curves[0].type == "PRINT"
    assert curves[0].raw == "external_npz_polyline"
    assert curves[0].subtype == "RESIN_PRINT"
    assert curves[0].layer == 0


def test_external_npz_print_path_smooths_sharp_corners_before_polyline_fast_path(monkeypatch):
    class FakePlanner:
        def __init__(self):
            self.last_fit_profile = {}

        def fit_global_curve(self, moves, **kwargs):
            return GlobalCurveCommand(
                type="PRINT_FIT",
                cmd="SPLINE",
                start_pos=moves[0].start_pos,
                control_points=[move.pos for move in moves],
                e_val=moves[-1].e_val,
                delta_e=sum(move.delta_e for move in moves),
                feedrate=moves[0].feedrate,
                line=moves[0].line,
                raw="fake_fit",
                original_moves=list(moves),
            )

    errors = iter([0.2, 0.01])
    monkeypatch.setattr(converter, "GlobalSplinePlanner", FakePlanner)
    monkeypatch.setattr(
        converter,
        "_curve_max_bidirectional_error_mm",
        lambda curve, points: next(errors),
        raising=False,
    )
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [
                                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                                [2.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                    )
                ],
                fiber_paths=[],
            )
        ],
    )

    curves = _source_curves(
        source_job_to_parsed_commands(job, ProcessParams(spline_max_error_mm=0.05))
    )

    assert len(curves) == 1
    assert curves[0].cmd == "POLYLINE"
    assert curves[0].raw == "external_npz_smoothed_polyline"
    assert curves[0].type == "PRINT"
    assert converter._polyline_max_turn_angle_deg(
        [curves[0].start_pos] + curves[0].control_points,
        min_segment_mm=0.01,
    ) < 45.0


def test_external_npz_resin_hairpin_smoothing_enforces_turn_angle_limit():
    source = (
        Path(__file__).resolve().parents[4]
        / "data/input_gcode/i/single_lug_connector_thickness_5mm_source (2).npz"
    )
    with np.load(source, allow_pickle=True) as data:
        resin_path = data["layer_0000_R"][5]
    resin_path = resin_path[np.isfinite(resin_path).all(axis=1)]
    resin_path = np.hstack(
        (resin_path, np.zeros((resin_path.shape[0], 3), dtype=np.float32))
    ).astype(np.float32)
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[MaterialPath("R", 0, resin_path)],
                fiber_paths=[],
            )
        ],
    )
    params = ProcessParams(
        corner_angle_deg=45.0,
        corner_retreat_ratio=0.65,
        corner_retreat_max_mm=0.4,
        corner_blend_segments=8,
        source_merge_distance_mm=0.04,
    )

    curves = _source_curves(source_job_to_parsed_commands(job, params))
    sampled_points = [
        sample.pos
        for sample in sample_global_curve_iter(
            curves[0],
            dt=params.dt,
            target_velocity=params.resin.feed_mm_s,
        )
    ]

    assert curves[0].raw == "external_npz_smoothed_polyline"
    assert converter._polyline_max_turn_angle_deg(
        sampled_points, min_segment_mm=0.02
    ) <= params.corner_angle_deg


def test_external_npz_rejects_bspline_when_sampled_turn_angle_exceeds_limit(monkeypatch):
    curve = GlobalCurveCommand(
        type="PRINT",
        cmd="SPLINE",
        start_pos=converter.Position(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        control_points=[
            converter.Position(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            converter.Position(1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            converter.Position(2.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        ],
        e_val=1.0,
        delta_e=1.0,
        feedrate=600.0,
        line=1,
        raw="fake_fit",
    )
    monkeypatch.setattr(
        converter,
        "_sample_spline_positions",
        lambda curve, points, sample_count=None: [curve.start_pos] + curve.control_points,
    )

    assert converter._curve_max_turn_angle_deg(curve, [], min_segment_mm=0.01) == pytest.approx(90.0)


def test_external_npz_polyline_fast_path_does_not_cross_separate_source_paths(monkeypatch):
    fit_starts = []

    class FakePlanner:
        def __init__(self):
            self.last_fit_profile = {}

        def fit_global_curve(self, moves, **kwargs):
            fit_starts.append((moves[0].start_pos.x, moves[0].start_pos.y))
            return GlobalCurveCommand(
                type="PRINT_FIT",
                cmd="SPLINE",
                start_pos=moves[0].start_pos,
                control_points=[move.pos for move in moves],
                e_val=moves[-1].e_val,
                delta_e=sum(move.delta_e for move in moves),
                feedrate=moves[0].feedrate,
                line=moves[0].line,
                raw="fake_fit",
                original_moves=list(moves),
            )

    monkeypatch.setattr(converter, "GlobalSplinePlanner", FakePlanner)
    monkeypatch.setattr(converter, "_curve_max_bidirectional_error_mm", lambda curve, points: 0.0, raising=False)
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        material="R",
                        order=0,
                        points=np.array(
                            [
                                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                    ),
                    MaterialPath(
                        material="R",
                        order=1,
                        points=np.array(
                            [
                                [10.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [11.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                    ),
                ],
                fiber_paths=[],
            )
        ],
    )

    commands = source_job_to_parsed_commands(job, ProcessParams())
    curves = _source_curves(commands)
    travels = [
        cmd for cmd in commands
        if isinstance(cmd, MoveCommand)
        and cmd.type == "TRAVEL"
        and cmd.raw == "external_npz_travel"
    ]

    assert len(curves) == 2
    assert [curve.cmd for curve in curves] == ["POLYLINE", "POLYLINE"]
    assert fit_starts == []
    assert curves[1].start_pos.x == pytest.approx(10.0)
    assert curves[1].start_pos.y == pytest.approx(0.0)
    assert len(travels) == 2
    assert travels[-1].start_pos.x == pytest.approx(1.0)
    assert travels[-1].start_pos.y == pytest.approx(1.0)
    assert travels[-1].pos.x == pytest.approx(10.0)
    assert travels[-1].pos.y == pytest.approx(0.0)
