import numpy as np
import pytest

from external_npz_preprocessor.converter import source_job_to_parsed_commands
from external_npz_preprocessor.process_params import ProcessParams
from external_npz_preprocessor.source_npz import LayerPaths, MaterialPath, SourceJob
from path_processing_core import npz_exporter
from path_processing_core.types import GlobalCurveCommand, MoveCommand


def test_external_source_print_path_is_fitted_by_core_with_density(tmp_path, monkeypatch):
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        "R",
                        0,
                        np.asarray(
                            [
                                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [5.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [5.0, 5.0, 0.5, 0.0, 0.0, 0.0],
                                [10.0, 5.0, 0.5, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                    )
                ],
            )
        ],
    )
    commands = source_job_to_parsed_commands(
        job, ProcessParams(primeline_enabled=False)
    )
    print_moves = [
        command
        for command in commands
        if isinstance(command, MoveCommand) and command.type == "PRINT"
    ]
    assert len(print_moves) == 3
    assert not any(isinstance(command, GlobalCurveCommand) for command in commands)

    captured_density = []
    original_fit = npz_exporter.GlobalSplinePlanner.fit_global_curve

    def record_fit(self, moves, **kwargs):
        captured_density.append(kwargs["density"])
        return original_fit(self, moves, **kwargs)

    monkeypatch.setattr(
        npz_exporter.GlobalSplinePlanner, "fit_global_curve", record_fit
    )
    npz_exporter.export_npz(
        commands,
        str(tmp_path / "core.npz"),
        density=1,
        corner_angle_deg=45.0,
        corner_retreat_ratio=0.65,
    )

    assert captured_density == [1]


def test_generic_external_source_path_does_not_split_at_short_input_segments(tmp_path, monkeypatch):
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        "R",
                        0,
                        np.asarray(
                            [
                                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [0.1, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [0.2, 0.02, 0.5, 0.0, 0.0, 0.0],
                                [10.0, 0.02, 0.5, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                    )
                ],
            )
        ],
    )
    commands = source_job_to_parsed_commands(job, ProcessParams(primeline_enabled=False))
    captured: list[GlobalCurveCommand] = []
    original_sample = npz_exporter.sample_global_curve_iter

    def record_sample(curve, **kwargs):
        if curve.type in {"PRINT", "PRINT_FIT"}:
            captured.append(curve)
        return original_sample(curve, **kwargs)

    monkeypatch.setattr(npz_exporter, "sample_global_curve_iter", record_sample)
    npz_exporter.export_npz(
        commands,
        str(tmp_path / "generic-short-prefix.npz"),
        corner_angle_deg=45.0,
        corner_retreat_ratio=0.25,
        density=1,
    )

    assert len(captured) == 1
    assert captured[0].cmd == "SPLINE"
    assert len(captured[0].original_moves) == 3


def test_declared_continuous_source_path_is_one_fitted_core_timing_curve(tmp_path, monkeypatch):
    job = SourceJob(
        meta={
            "path_roles": {"R": {"0": ["brim"]}},
            "continuous_deposition_roles": {"R": ["brim"]},
        },
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[
                    MaterialPath(
                        "R",
                        0,
                        np.asarray(
                            [
                                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [0.1, 0.0, 0.5, 0.0, 0.0, 0.0],
                                [0.2, 0.02, 0.5, 0.0, 0.0, 0.0],
                                [10.0, 0.02, 0.5, 0.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                        extrusion=np.asarray([0.0, 0.1, 0.2, 10.0], dtype=np.float32),
                    )
                ],
            )
        ],
    )
    commands = source_job_to_parsed_commands(job, ProcessParams(primeline_enabled=False))
    captured: list[GlobalCurveCommand] = []
    original_sample = npz_exporter.sample_global_curve_iter

    def record_sample(curve, **kwargs):
        if curve.type in {"PRINT", "PRINT_FIT"}:
            captured.append(curve)
        return original_sample(curve, **kwargs)

    monkeypatch.setattr(npz_exporter, "sample_global_curve_iter", record_sample)
    npz_exporter.export_npz(
        commands,
        str(tmp_path / "continuous.npz"),
        dt=0.02,
        corner_angle_deg=45.0,
        corner_retreat_ratio=0.25,
        density=1,
        preserve_source_e_profile=True,
    )

    assert len(captured) == 1
    assert captured[0].cmd == "SPLINE"
    assert len(captured[0].original_moves) == 3
    # The converter primes resin before every source path; source-profile
    # deltas, rather than its reset origin, are the preservation contract.
    assert captured[0].source_e_values[0] == pytest.approx(
        captured[0].original_moves[0].e_val - captured[0].original_moves[0].delta_e
    )
    assert captured[0].source_e_values[-1] == pytest.approx(
        captured[0].original_moves[-1].e_val
    )


def test_adjacent_continuous_source_paths_keep_distinct_core_strokes(tmp_path, monkeypatch):
    points = lambda x: np.asarray(
        [
            [x, 0.0, 0.5, 0.0, 0.0, 0.0],
            [x + 5.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [x + 5.0, 5.0, 0.5, 0.0, 0.0, 0.0],
            [x + 10.0, 5.0, 0.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    job = SourceJob(
        meta={
            "path_roles": {"R": {"0": ["brim", "brim"]}},
            "continuous_deposition_roles": {"R": ["brim"]},
        },
        layers=[
            LayerPaths(
                index=0,
                resin_paths=[MaterialPath("R", 0, points(0.0)), MaterialPath("R", 1, points(10.0))],
            )
        ],
    )
    commands = source_job_to_parsed_commands(job, ProcessParams(primeline_enabled=False))
    captured: list[GlobalCurveCommand] = []
    original_sample = npz_exporter.sample_global_curve_iter

    def record_sample(curve, **kwargs):
        if curve.type in {"PRINT", "PRINT_FIT"}:
            captured.append(curve)
        return original_sample(curve, **kwargs)

    monkeypatch.setattr(npz_exporter, "sample_global_curve_iter", record_sample)
    npz_exporter.export_npz(commands, str(tmp_path / "two-strokes.npz"))

    assert len(captured) == 2
    assert all(curve.cmd == "SPLINE" for curve in captured)
    assert [len(curve.original_moves) for curve in captured] == [3, 3]
    assert {
        move.source_path_id
        for curve in captured
        for move in curve.original_moves
    } == {"external-source:0:R:0", "external-source:0:R:1"}
