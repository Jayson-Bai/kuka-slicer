"""Launch the real UI compute server outside the caller's Windows Job.

This is a local benchmark harness. It keeps the server alive until stdin is
closed, so density timing can be collected through the same HTTP/UI workflow
as the desktop app while selecting a worker count for controlled experiments.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from kuka_slicer.app_session import _launch_server_process, _stop_process


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs-benchmark")
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    base_executable = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
    venv_root = Path(sys.prefix).resolve()
    venv_paths: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            Path(entry).resolve().relative_to(venv_root)
        except ValueError:
            continue
        venv_paths.append(entry)
    bootstrap = (
        "import os,runpy,sys;"
        f"os.environ['KUKA_SLICER_MAX_CPU_CORES']={str(args.workers)!r};"
        f"os.environ['KUKA_CORE_DETAILED_TIMING']={('1' if args.detailed else '')!r};"
        f"sys.path[:0]={venv_paths!r};"
        "sys.argv=['kuka_slicer',*sys.argv[1:]];"
        "runpy.run_module('kuka_slicer',run_name='__main__')"
    )
    process = _launch_server_process(
        [
            base_executable,
            "-c",
            bootstrap,
            "ui",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--output-dir",
            args.output_dir,
        ]
    )
    print(f"SERVER_PID={process.pid}", flush=True)
    try:
        input()
    finally:
        _stop_process(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
