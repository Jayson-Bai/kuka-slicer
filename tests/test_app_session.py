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
    monkeypatch.setattr(app_session, "_find_available_port", lambda: 43210)
    monkeypatch.setattr(app_session, "_wait_for_port", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_session, "_launch_browser_app", lambda *args, **kwargs: (browser, profile))

    assert app_session.run_app_session("ui") == 0

    assert "43210" in launched[0]


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

    assert app_session._find_browser() == chrome


def test_main_ui_exposes_surface_tool_launchers() -> None:
    html = _index_html()

    assert 'id="surfacePreviewButton"' in html
    assert 'id="surfaceMapperButton"' in html
    assert "surfaceToolButtons['surface-preview'].addEventListener" in html
    assert "surfaceToolButtons['surface-map'].addEventListener" in html
    assert "/launch-tool?tool=" in html
