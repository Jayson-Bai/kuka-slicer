import numpy as np

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
