import numpy as np
import pytest

from external_npz_preprocessor.converter import source_job_to_parsed_commands
from external_npz_preprocessor.process_params import (
    FiberProcessParams,
    ProcessParams,
    ResinProcessParams,
)
from external_npz_preprocessor.source_npz import (
    LayerPaths,
    MaterialPath,
    SourceJob,
    TravelPath,
)
from path_processing_core.types import (
    ExtrudeWait,
    GlobalCurveCommand,
    MCommand,
    MoveCommand,
    ResetECommand,
    ToolChangeCommand,
)


def _path(material: str, points, extrusion=None) -> MaterialPath:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[1] == 3:
        points = np.hstack((points, np.zeros((len(points), 3), dtype=np.float32)))
    return MaterialPath(
        material=material,
        order=0,
        points=points,
        extrusion=(None if extrusion is None else np.asarray(extrusion, dtype=np.float32)),
    )


def _job(*, resin_paths=(), fiber_paths=(), travel_paths=()) -> SourceJob:
    return SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=list(resin_paths),
                fiber_paths=list(fiber_paths),
                travel_paths=list(travel_paths),
            )
        ],
    )


def _print_moves(commands):
    return [
        command
        for command in commands
        if isinstance(command, MoveCommand) and command.type == "PRINT"
    ]


def _travel_moves(commands):
    return [
        command
        for command in commands
        if isinstance(command, MoveCommand) and command.type == "TRAVEL"
    ]


@pytest.mark.parametrize("feed", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_travel_feed(feed):
    with pytest.raises(ValueError, match="travel_feed_mm_s"):
        source_job_to_parsed_commands(
            _job(resin_paths=[_path("R", [[0, 0, 0.5], [1, 0, 0.5]])]),
            ProcessParams(travel_feed_mm_s=feed),
        )


def test_print_paths_remain_moves_with_source_xyz_and_material_metadata():
    commands = source_job_to_parsed_commands(
        _job(
            resin_paths=[_path("R", [[0, 0, 2.5], [10, 0, 2.6]])],
            fiber_paths=[_path("F", [[10, 0, 3.1], [10, 4, 3.3]])],
        ),
        ProcessParams(primeline_enabled=False),
    )

    moves = _print_moves(commands)

    assert [(move.subtype, move.start_pos.z, move.pos.z) for move in moves] == [
        ("RESIN_PRINT", pytest.approx(2.5), pytest.approx(2.6)),
        ("FIBER_PRINT", pytest.approx(3.1), pytest.approx(3.3)),
    ]
    assert [move.feedrate for move in moves] == [600.0, 900.0]
    assert not any(isinstance(command, GlobalCurveCommand) for command in commands)


def test_fiber_paths_do_not_redefine_resin_start_xy_origin():
    commands = source_job_to_parsed_commands(
        _job(
            resin_paths=[_path("R", [[20, 30, 0.5], [25, 30, 0.5]])],
            fiber_paths=[_path("F", [[-100, -100, 0.6], [-99, -100, 0.6]])],
        ),
        ProcessParams(primeline_enabled=False, start_x_mm=10.0, start_y_mm=15.0),
    )

    resin_move = next(move for move in _print_moves(commands) if move.subtype == "RESIN_PRINT")

    assert (resin_move.start_pos.x, resin_move.start_pos.y) == pytest.approx((10.0, 15.0))


def test_prusa_source_e_profile_is_preserved_per_move_before_core_fitting():
    commands = source_job_to_parsed_commands(
        _job(
            resin_paths=[
                _path(
                    "R",
                    [[0, 0, 0.5], [5, 0, 0.5], [10, 0, 0.5]],
                    extrusion=[100.0, 103.0, 111.0],
                )
            ]
        ),
        ProcessParams(primeline_enabled=False),
    )

    moves = _print_moves(commands)

    assert [move.delta_e for move in moves] == pytest.approx([3.0, 8.0])
    assert moves[-1].e_val - moves[0].e_val == pytest.approx(8.0)


def test_zero_e_segment_inside_material_path_stays_a_print_move():
    """Macro-partition connectors are print-context motion, never Travel."""

    commands = source_job_to_parsed_commands(
        _job(
            resin_paths=[
                _path(
                    "R",
                    [[0, 0, 0.5], [5, 0, 0.5], [5, 5, 0.5], [10, 5, 0.5]],
                    extrusion=[40.0, 43.0, 43.0, 47.0],
                )
            ]
        ),
        ProcessParams(primeline_enabled=False),
    )

    moves = _print_moves(commands)

    assert [move.delta_e for move in moves] == pytest.approx([3.0, 0.0, 4.0])
    assert moves[1].type == "PRINT"
    assert moves[1].cmd == "G1"
    assert not any(
        isinstance(command, MoveCommand)
        and command.type == "TRAVEL"
        and command.start_pos == moves[1].start_pos
        and command.pos == moves[1].pos
        for command in commands
    )


def test_collinear_source_travel_is_collapsed_before_core_handoff():
    commands = source_job_to_parsed_commands(
        _job(
            resin_paths=[_path("R", [[4, 0, 0.5], [6, 0, 0.5]])],
            travel_paths=[
                TravelPath(
                    order=0,
                    points=np.asarray(
                        [
                            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                            [2.5, 0.0, 0.5, 0.0, 0.0, 0.0],
                            [4.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                        ],
                        dtype=np.float64,
                    ),
                )
            ],
        ),
        ProcessParams(primeline_enabled=False),
    )

    travels = _travel_moves(commands)

    assert len(travels) == 1
    assert travels[0].raw == "external_npz_prusa_travel"
    assert (travels[0].start_pos.x, travels[0].start_pos.y) == pytest.approx((6.0, 10.0))
    assert (travels[0].pos.x, travels[0].pos.y) == pytest.approx((10.0, 10.0))


def test_turning_source_travel_keeps_hole_avoidance_waypoints():
    commands = source_job_to_parsed_commands(
        _job(
            resin_paths=[_path("R", [[4, 2, 0.5], [6, 2, 0.5]])],
            travel_paths=[
                TravelPath(
                    order=0,
                    points=np.asarray(
                        [
                            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                            [0.0, 2.0, 0.5, 0.0, 0.0, 0.0],
                            [4.0, 2.0, 0.5, 0.0, 0.0, 0.0],
                        ],
                        dtype=np.float64,
                    ),
                )
            ],
        ),
        ProcessParams(primeline_enabled=False),
    )

    travels = _travel_moves(commands)

    assert len(travels) == 2
    np.testing.assert_allclose(
        [
            (move.start_pos.x, move.start_pos.y, move.pos.x, move.pos.y)
            for move in travels
        ],
        [
            (6.0, 8.0, 6.0, 10.0),
            (6.0, 10.0, 10.0, 10.0),
        ],
    )


def test_path_events_and_reset_boundaries_remain_outside_core_spline_input():
    commands = source_job_to_parsed_commands(
        _job(
            resin_paths=[_path("R", [[0, 0, 0.5], [10, 0, 0.5]])],
            fiber_paths=[_path("F", [[10, 0, 0.6], [20, 0, 0.6]])],
        ),
        ProcessParams(primeline_enabled=False),
    )

    assert [command.tool for command in commands if isinstance(command, ToolChangeCommand)] == [1, 0]
    assert len(_print_moves(commands)) == 2
    assert any(
        isinstance(command, ResetECommand)
        and command.raw == "external_npz_path_reset"
        for command in commands
    )
    assert any(
        isinstance(command, MCommand) and command.code == "CUT"
        for command in commands
    )
    assert any(
        isinstance(command, ExtrudeWait) and command.raw == "external_npz_prime"
        for command in commands
    )


def test_logical_layers_stay_in_source_order_without_regrouping_by_z():
    job = SourceJob(
        meta={
            "layer_semantics": {
                "format": "logical_layer_v1",
                "layer_key": "logical_deposition_layer",
                "z_coordinate": "per_point_trajectory",
                "ordering": "ascending_layer_key_then_source_path_order",
                "reconstruct_layers_from_z": False,
            }
        },
        layers=[
            LayerPaths(index=0, resin_paths=[_path("R", [[0, 0, 1.4], [5, 0, 0.9]])]),
            LayerPaths(index=1, resin_paths=[_path("R", [[5, 0, 0.6], [10, 0, 1.8]])]),
        ],
    )

    moves = _print_moves(source_job_to_parsed_commands(job, ProcessParams(primeline_enabled=False)))

    assert [move.layer for move in moves] == [0, 1]
    assert [move.start_pos.z for move in moves] == pytest.approx([1.4, 0.6])


def test_first_layer_material_speeds_are_applied_to_move_commands():
    commands = source_job_to_parsed_commands(
        _job(
            resin_paths=[_path("R", [[0, 0, 0.5], [10, 0, 0.5]])],
            fiber_paths=[_path("F", [[10, 0, 0.6], [20, 0, 0.6]])],
        ),
        ProcessParams(
            primeline_enabled=False,
            resin=ResinProcessParams(feed_mm_s=11.0, first_layer_feed_mm_s=2.0),
            fiber=FiberProcessParams(feed_mm_s=12.0, first_layer_feed_mm_s=3.0),
        ),
    )

    assert [move.feedrate for move in _print_moves(commands)] == [120.0, 180.0]
