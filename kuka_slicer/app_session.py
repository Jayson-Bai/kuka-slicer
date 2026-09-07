"""Launch one local KUKA web tool for the lifetime of its browser window."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


_TOOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ui": ("ui", ("--output-dir", "outputs")),
    "surface-preview": ("surface-preview", ()),
    "surface-map": ("surface-map", ()),
}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def spawn_app_session(tool: str) -> subprocess.Popen[bytes]:
    """Start a detached browser-bound session for one supported local tool."""

    if tool not in _TOOLS:
        raise ValueError(f"unsupported local tool: {tool}")
    return subprocess.Popen(
        [sys.executable, "-m", "kuka_slicer.app_session", tool],
        cwd=_PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_app_session(tool: str) -> int:
    """Run the server only while its dedicated browser app window is open."""

    command, extra_args = _tool_spec(tool)
    port = _find_available_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "kuka_slicer", command, "--host", "127.0.0.1", "--port", str(port), *extra_args],
        cwd=_PROJECT_ROOT,
    )
    profile_dir: Path | None = None
    try:
        _wait_for_port(port, server)
        browser, profile_dir = _launch_browser_app(f"http://127.0.0.1:{port}", tool)
        browser.wait()
        return 0
    finally:
        _stop_process(server)
        if profile_dir is not None:
            shutil.rmtree(profile_dir, ignore_errors=True)


def _tool_spec(tool: str) -> tuple[str, tuple[str, ...]]:
    try:
        return _TOOLS[tool]
    except KeyError as exc:
        raise ValueError(f"unsupported local tool: {tool}") from exc


def _find_available_port() -> int:
    """Allocate a fresh loopback port for one browser-bound local session."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_port(port: int, server: subprocess.Popen[bytes], timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"local server exited with code {server.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"local server did not start on port {port}")


def _launch_browser_app(url: str, tool: str) -> tuple[subprocess.Popen[bytes], Path]:
    browser_path = _find_browser()
    profile_dir = Path(tempfile.mkdtemp(prefix=f"kuka-slicer-{tool}-"))
    try:
        browser = subprocess.Popen(
            [
                str(browser_path),
                f"--app={url}",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={profile_dir}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        shutil.rmtree(profile_dir, ignore_errors=True)
        raise
    return browser, profile_dir


def _find_browser() -> Path:
    configured = os.environ.get("KUKA_SLICER_BROWSER")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        Path(root) / relative
        for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"))
        if root
        for relative in (
            Path("Google/Chrome/Application/chrome.exe"),
            Path("Microsoft/Edge/Application/msedge.exe"),
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("未找到 Microsoft Edge 或 Google Chrome，无法创建受控界面窗口")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m kuka_slicer.app_session <ui|surface-preview|surface-map>")
    raise SystemExit(run_app_session(sys.argv[1]))
