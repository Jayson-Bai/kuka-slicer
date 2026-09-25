from __future__ import annotations

from pathlib import Path

import pytest

from kuka_slicer import app_session
from kuka_slicer.ui_server import _index_html


class _Process:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_app_session_stops_server_when_browser_window_exits(monkeypatch, tmp_path: Path) -> None:
    server = _Process()
    browser = _Process()
    profile = tmp_path / "isolated-browser-profile"
    profile.mkdir()
    monkeypatch.setattr(app_session.subprocess, "Popen", lambda *args, **kwargs: server)
    monkeypatch.setattr(app_session.sys, "platform", "linux")
    monkeypatch.setattr(app_session, "_wait_for_port", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_session, "_launch_browser_app", lambda *args, **kwargs: (browser, profile))

    assert app_session.run_app_session("surface-preview") == 0

    assert browser.waited
    assert server.terminated
    assert not profile.exists()


def test_app_session_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="unsupported local tool"):
        app_session.run_app_session("unknown")


def test_app_session_uses_a_fresh_port_for_each_server(monkeypatch, tmp_path: Path) -> None:
    server = _Process()
    browser = _Process()
    profile = tmp_path / "isolated-browser-profile"
    profile.mkdir()
    launched = []

    def fake_popen(command, **kwargs):
        launched.append(command)
        return server

    monkeypatch.setattr(app_session.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(app_session.sys, "platform", "linux")
    monkeypatch.setattr(app_session, "_find_available_port", lambda: 43210)
    monkeypatch.setattr(app_session, "_wait_for_port", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_session, "_launch_browser_app", lambda *args, **kwargs: (browser, profile))

    assert app_session.run_app_session("ui") == 0

    assert "43210" in launched[0]


def test_server_process_requests_windows_job_breakaway(monkeypatch) -> None:
    server = _Process()
    calls: list[dict[str, object]] = []

    def fake_popen(_command, **kwargs):
        calls.append(kwargs)
        return server

    monkeypatch.setattr(app_session.sys, "platform", "win32")
    monkeypatch.setattr(
        app_session.subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000, raising=False
    )
    monkeypatch.setattr(app_session.subprocess, "Popen", fake_popen)

    assert app_session._launch_server_process(["python", "-m", "kuka_slicer"]) is server
    assert calls == [{"creationflags": 0x01000000, "cwd": app_session._PROJECT_ROOT}]


def test_server_process_uses_cim_when_job_rejects_breakaway(monkeypatch) -> None:
    server = _Process()
    calls: list[dict[str, object]] = []

    def fake_popen(_command, **kwargs):
        calls.append(kwargs)
        error = OSError("breakaway denied")
        error.winerror = 5
        raise error

    monkeypatch.setattr(app_session.sys, "platform", "win32")
    monkeypatch.setattr(
        app_session.subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000, raising=False
    )
    monkeypatch.setattr(app_session.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        app_session,
        "_launch_server_process_via_cim",
        lambda _command: server,
    )

    assert app_session._launch_server_process(["python", "-m", "kuka_slicer"]) is server
    assert calls == [{"creationflags": 0x01000000, "cwd": app_session._PROJECT_ROOT}]


def test_server_process_uses_cim_when_breakaway_child_remains_in_job(monkeypatch) -> None:
    server = _Process()
    detached = _Process()
    monkeypatch.setattr(app_session.sys, "platform", "win32")
    monkeypatch.setattr(
        app_session.subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000, raising=False
    )
    monkeypatch.setattr(app_session.subprocess, "Popen", lambda *_args, **_kwargs: server)
    monkeypatch.setattr(app_session, "_windows_process_is_in_job", lambda _pid: True)
    monkeypatch.setattr(app_session, "_launch_server_process_via_cim", lambda _command: detached)

    assert app_session._launch_server_process(["python", "-m", "kuka_slicer"]) is detached
    assert server.terminated


def test_direct_server_command_bypasses_windows_venv_launcher(monkeypatch, tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    launcher = venv / "Scripts" / "pythonw.exe"
    base = tmp_path / "base" / "pythonw.exe"
    site_packages = venv / "Lib" / "site-packages"
    monkeypatch.setattr(app_session.sys, "platform", "win32")
    monkeypatch.setattr(app_session.sys, "executable", str(launcher))
    monkeypatch.setattr(app_session.sys, "_base_executable", str(base), raising=False)
    monkeypatch.setattr(app_session.sys, "prefix", str(venv))
    monkeypatch.setattr(app_session.sys, "path", [str(site_packages), str(tmp_path / "other")])

    command = app_session._direct_server_command(
        [str(launcher), "-m", "kuka_slicer", "ui", "--port", "1234"]
    )

    assert command[:2] == [str(base.resolve()), "-c"]
    assert repr(str(site_packages)) in command[2]
    assert command[3:] == ["ui", "--port", "1234"]


def test_app_session_prefers_google_chrome_over_edge(monkeypatch, tmp_path: Path) -> None:
    chrome = tmp_path / "Google" / "Chrome" / "Application" / "chrome.exe"
    edge = tmp_path / "Microsoft" / "Edge" / "Application" / "msedge.exe"
    chrome.parent.mkdir(parents=True)
    edge.parent.mkdir(parents=True)
    chrome.touch()
    edge.touch()
    monkeypatch.delenv("KUKA_SLICER_BROWSER", raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setattr(app_session.sys, "platform", "win32")

    assert app_session._find_browser() == chrome


def test_app_session_finds_macos_chrome_for_designer_window(monkeypatch, tmp_path: Path) -> None:
    chrome = tmp_path / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome"
    edge = tmp_path / "Microsoft Edge.app" / "Contents" / "MacOS" / "Microsoft Edge"
    chrome.parent.mkdir(parents=True)
    edge.parent.mkdir(parents=True)
    chrome.touch()
    edge.touch()
    monkeypatch.delenv("KUKA_SLICER_BROWSER", raising=False)
    monkeypatch.setattr(app_session.sys, "platform", "darwin")
    monkeypatch.setattr(app_session, "_MACOS_BROWSER_PATHS", (chrome, edge))

    assert app_session._find_browser() == chrome


def test_macos_designer_browser_activation_uses_its_containing_app(monkeypatch, tmp_path: Path) -> None:
    chrome = tmp_path / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome"
    chrome.parent.mkdir(parents=True)
    chrome.touch()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(app_session.sys, "platform", "darwin")
    monkeypatch.setattr(app_session.subprocess, "run", fake_run)

    app_session._activate_macos_browser_window(chrome)

    assert calls == [
        (
            ["/usr/bin/osascript", "-e", 'tell application "Google Chrome" to activate'],
            {
                "check": False,
                "stdout": app_session.subprocess.DEVNULL,
                "stderr": app_session.subprocess.DEVNULL,
                "timeout": 5.0,
            },
        )
    ]


def test_macos_browser_session_uses_an_isolated_profile_without_background_mode(monkeypatch, tmp_path: Path) -> None:
    chrome = tmp_path / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome"
    chrome.parent.mkdir(parents=True)
    chrome.touch()
    process = _Process()
    commands: list[list[str]] = []

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return process

    monkeypatch.setattr(app_session.sys, "platform", "darwin")
    monkeypatch.setattr(app_session, "_find_browser", lambda: chrome)
    monkeypatch.setattr(app_session, "_activate_macos_browser_window", lambda _path: None)
    monkeypatch.setattr(app_session.subprocess, "Popen", fake_popen)

    _browser, profile = app_session._launch_browser_app("http://127.0.0.1:45678", "surface-preview")

    assert commands[0][:6] == [
        "/usr/bin/open",
        "-n",
        "-a",
        str(chrome.parents[2]),
        "--args",
        "--app=http://127.0.0.1:45678",
    ]
    assert "--disable-background-mode" in commands[0]
    assert "--app=http://127.0.0.1:45678" in commands[0]
    profile.rmdir()


def test_macos_session_waits_for_the_profile_process_to_close(monkeypatch, tmp_path: Path) -> None:
    profile = tmp_path / "isolated-browser-profile"
    profile.mkdir()
    observed = iter([(), (1234,), (), ()])
    monotonic_values = iter([0.0, 0.0, 0.5, 1.5])
    monkeypatch.setattr(
        app_session,
        "_macos_browser_profile_pids",
        lambda _profile, _browser: next(observed),
    )
    monkeypatch.setattr(app_session.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(app_session.time, "sleep", lambda _seconds: None)

    app_session._wait_for_macos_browser_profile(profile, tmp_path / "Google Chrome")


def test_main_ui_exposes_surface_tool_launchers() -> None:
    html = _index_html()

    assert 'id="surfacePreviewButton"' in html
    assert 'id="surfaceMapperButton"' not in html
    assert "surfaceToolButtons['surface-preview'].addEventListener" in html
    assert "surfaceToolButtons['surface-map'].addEventListener" not in html
    assert "/launch-tool?tool=" in html
