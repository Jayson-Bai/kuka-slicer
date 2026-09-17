from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from external_npz_preprocessor.converter import source_job_to_parsed_commands
from external_npz_preprocessor.process_params import ProcessParams
from external_npz_preprocessor.source_npz import LayerPaths, MaterialPath, SourceJob
from path_processing_core.npz_exporter import export_npz
from path_processing_core.parallel_npz_exporter import (
    _WORKER_NUMERIC_THREAD_ENVS,
    _request_full_execution_speed,
    _single_thread_worker_numeric_environment,
    export_npz_parallel_by_layer,
    shutdown_parallel_worker_pool,
)


def test_worker_execution_speed_request_is_a_noop_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(os, "name", "posix")

    assert _request_full_execution_speed() is False


def test_parallel_worker_numeric_environment_is_scoped(monkeypatch) -> None:
    for index, name in enumerate(_WORKER_NUMERIC_THREAD_ENVS):
        monkeypatch.setenv(name, str(index + 2))

    with _single_thread_worker_numeric_environment():
        assert {name: os.environ[name] for name in _WORKER_NUMERIC_THREAD_ENVS} == {
            name: "1" for name in _WORKER_NUMERIC_THREAD_ENVS
        }

    assert {name: os.environ[name] for name in _WORKER_NUMERIC_THREAD_ENVS} == {
        name: str(index + 2) for index, name in enumerate(_WORKER_NUMERIC_THREAD_ENVS)
    }


def _path(material: str, order: int, x: float, z: float) -> MaterialPath:
    return MaterialPath(
        material,
        order,
        np.asarray(
            [
                [x, 0.0, z, 0.0, 2.0, 0.0],
                [x + 1.0, 0.5, z + 0.1, 0.0, 2.5, 0.0],
                [x + 2.0, 0.0, z + 0.2, 0.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def test_layer_parallel_export_is_byte_identical_to_serial_export(tmp_path: Path) -> None:
    job = SourceJob(
        meta={},
        layers=[
            LayerPaths(index=0, resin_paths=[_path("R", 0, 0.0, 0.2)]),
            LayerPaths(
                index=1,
                resin_paths=[_path("R", 0, 3.0, 0.4)],
                fiber_paths=[_path("F", 0, 6.0, 0.5)],
            ),
            LayerPaths(
                index=2,
                resin_paths=[_path("R", 0, 9.0, 0.6)],
                fiber_paths=[_path("F", 0, 12.0, 0.7)],
            ),
        ],
    )
    params = ProcessParams(primeline_enabled=False)
    commands = source_job_to_parsed_commands(job, params)
    serial = tmp_path / "serial.npz"
    parallel = tmp_path / "parallel.npz"
    progress: list[float] = []
    kwargs = {
        "dt": 0.05,
        "chunk_size": 1_000_000,
        "default_feed_mm_s": params.travel_feed_mm_s,
        "density": 0,
        "enable_extrude_wait": True,
        "external_npz_cut_absolute_e": True,
        "cut_lift_mm": 1.0,
        "cut_wait_s": 1.0,
    }

    export_npz(commands, str(serial), **kwargs)
    try:
        export_npz_parallel_by_layer(
            commands,
            str(parallel),
            max_workers=2,
            progress_callback=progress.append,
            reuse_workers=True,
            **kwargs,
        )
    finally:
        shutdown_parallel_worker_pool()

    assert progress[0] == 0.0
    assert serial.read_bytes() == parallel.read_bytes()
    assert serial.with_suffix(".offset.json").read_bytes() == parallel.with_suffix(
        ".offset.json"
    ).read_bytes()
    assert serial.with_suffix(".timing.json").read_bytes() == parallel.with_suffix(
        ".timing.json"
    ).read_bytes()
