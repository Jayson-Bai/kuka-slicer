"""Launch one local KUKA web tool for the lifetime of its browser window."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Protocol


_TOOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ui": ("ui", ("--output-dir", "outputs")),
    "surface-preview": ("surface-preview", ()),
    "surface-map": ("surface-map", ()),
}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BREAKAWAY_FALLBACK_WINERRORS = {5, 87}
_MACOS_BROWSER_PATHS = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)


class _ManagedProcess(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


class _WindowsDetachedProcess:
    """Small Popen-compatible handle for a process created by Win32 CIM."""

    _SYNCHRONIZE = 0x00100000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 258
    _INFINITE = 0xFFFFFFFF
    _STILL_ACTIVE = 259

    def __init__(self, pid: int, command: list[str]):
        import ctypes
        from ctypes import wintypes

        self.pid = int(pid)
        self.args = list(command)
        self.returncode: int | None = None
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._kernel32.WaitForSingleObject.restype = wintypes.DWORD
        self._kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        self._kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateProcess.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        access = (
            self._SYNCHRONIZE
            | self._PROCESS_TERMINATE
            | self._PROCESS_QUERY_LIMITED_INFORMATION
        )
        self._handle = self._kernel32.OpenProcess(access, False, self.pid)
        if not self._handle:
            raise OSError(
                ctypes.get_last_error(),
                f"cannot open detached server process {self.pid}",
            )

    def _read_returncode(self) -> int | None:
        code = self._wintypes.DWORD()
        if not self._kernel32.GetExitCodeProcess(self._handle, self._ctypes.byref(code)):
            raise OSError(self._ctypes.get_last_error(), "GetExitCodeProcess failed")
        if int(code.value) == self._STILL_ACTIVE:
            return None
        self.returncode = int(code.value)
        return self.returncode

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        result = int(self._kernel32.WaitForSingleObject(self._handle, 0))
        if result == self._WAIT_TIMEOUT:
            return None
        if result != self._WAIT_OBJECT_0:
            raise OSError(self._ctypes.get_last_error(), "WaitForSingleObject failed")
        return self._read_returncode()

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is not None:
            return self.returncode
        milliseconds = (
            self._INFINITE
            if timeout is None
            else max(0, min(self._INFINITE - 1, int(float(timeout) * 1000)))
        )
        result = int(self._kernel32.WaitForSingleObject(self._handle, milliseconds))
        if result == self._WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired(self.args, timeout)
        if result != self._WAIT_OBJECT_0:
            raise OSError(self._ctypes.get_last_error(), "WaitForSingleObject failed")
        return int(self._read_returncode() or 0)

    def terminate(self) -> None:
        if self.poll() is None and not self._kernel32.TerminateProcess(self._handle, 1):
            raise OSError(self._ctypes.get_last_error(), "TerminateProcess failed")

    def kill(self) -> None:
        self.terminate()

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._kernel32.CloseHandle(handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


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
    server = _launch_server_process(
        [sys.executable, "-m", "kuka_slicer", command, "--host", "127.0.0.1", "--port", str(port), *extra_args]
    )
    profile_dir: Path | None = None
    try:
        _wait_for_port(port, server)
        browser, profile_dir = _launch_browser_app(f"http://127.0.0.1:{port}", tool)
        _wait_for_browser_session(browser, profile_dir)
        return 0
    finally:
        _stop_process(server)
        if profile_dir is not None:
            shutil.rmtree(profile_dir, ignore_errors=True)


def _launch_server_process(command: list[str]) -> _ManagedProcess:
    """Start the compute server outside a restrictive launcher Job when possible.

    Windows may attach a GUI-launched ``pythonw`` process tree to a Job Object
    with a CPU-rate cap.  The browser-bound supervisor remains in that Job,
    while the explicitly managed server breaks away so the project's own CPU
    and memory limits remain authoritative.  Locked-down Jobs reject the
    breakaway flag; Win32 CIM then creates the server through the WMI service,
    outside the launcher's Job, instead of silently retaining its CPU cap.
    """

    command = _direct_server_command(command)
    kwargs = {"cwd": _PROJECT_ROOT}
    breakaway_flag = (
        getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        if sys.platform == "win32"
        else 0
    )
    if breakaway_flag:
        try:
            process = subprocess.Popen(
                command,
                creationflags=breakaway_flag,
                **kwargs,
            )
            in_job = _windows_process_is_in_job(getattr(process, "pid", None))
            if in_job is not True:
                return process
            _stop_process(process)
            return _launch_server_process_via_cim(command)
        except OSError as exc:
            if getattr(exc, "winerror", None) not in _BREAKAWAY_FALLBACK_WINERRORS:
                raise
            return _launch_server_process_via_cim(command)
    return subprocess.Popen(command, **kwargs)


def _direct_server_command(command: list[str]) -> list[str]:
    """Avoid the Windows venv launcher reattaching the real server to a Job.

    A Windows virtual-environment ``pythonw.exe`` is a launcher process.  It
    can successfully break away while the base interpreter it subsequently
    starts is placed back in the desktop launcher's Job.  Run that base
    interpreter directly, while explicitly restoring the venv-only import
    paths, so the PID verified and managed below is the actual UI server.
    """

    if sys.platform != "win32" or len(command) < 3:
        return command
    try:
        launcher = Path(sys.executable).resolve()
        requested = Path(command[0]).resolve()
        base = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    except OSError:
        return command
    if requested != launcher or base == launcher or command[1:3] != ["-m", "kuka_slicer"]:
        return command

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
        "import runpy,sys;"
        f"sys.path[:0]={venv_paths!r};"
        "sys.argv=['kuka_slicer',*sys.argv[1:]];"
        "runpy.run_module('kuka_slicer',run_name='__main__')"
    )
    return [str(base), "-c", bootstrap, *command[3:]]


def _windows_process_is_in_job(pid: int | None) -> bool | None:
    """Return a verified Job membership state for one Windows process."""

    if sys.platform != "win32" or pid is None:
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    try:
        result = wintypes.BOOL()
        if not kernel32.IsProcessInJob(handle, None, ctypes.byref(result)):
            return None
        return bool(result.value)
    finally:
        kernel32.CloseHandle(handle)


def _launch_server_process_via_cim(command: list[str]) -> _WindowsDetachedProcess:
    """Create the compute server outside the caller's Windows Job Object."""

    import base64
    import json

    command_line = subprocess.list2cmdline(command)
    encoded_command = base64.b64encode(command_line.encode("utf-16-le")).decode("ascii")
    encoded_cwd = base64.b64encode(str(_PROJECT_ROOT).encode("utf-16-le")).decode("ascii")
    script = (
        "$ErrorActionPreference='Stop';"
        f"$cmd=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded_command}'));"
        f"$cwd=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded_cwd}'));"
        "$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        "-Arguments @{CommandLine=$cmd;CurrentDirectory=$cwd};"
        "if($r.ReturnValue -ne 0){throw ('Win32_Process.Create failed: '+$r.ReturnValue)};"
        "$r.ProcessId"
    )
    encoded_script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded_script,
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    try:
        pid = int(output_lines[-1])
    except (IndexError, ValueError) as exc:
        detail = json.dumps(
            {"stdout": completed.stdout, "stderr": completed.stderr},
            ensure_ascii=False,
        )
        raise RuntimeError(f"CIM did not return a server process id: {detail}") from exc
    return _WindowsDetachedProcess(pid, command)


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


def _wait_for_port(port: int, server: _ManagedProcess, timeout_s: float = 60.0) -> None:
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
        browser_args = [
            f"--app={url}",
            "--no-first-run",
            "--no-default-browser-check",
            # A browser-bound session must terminate when its app window is
            # closed. Otherwise Chrome can keep the temporary profile alive
            # in the background and retain its local slicer port indefinitely.
            "--disable-background-mode",
            f"--user-data-dir={profile_dir}",
        ]
        if sys.platform == "darwin":
            # ``-n`` forces a new Chrome app instance for the temporary
            # profile instead of forwarding the URL to an unrelated existing
            # browser. Its launcher returns promptly; the profile monitor
            # below owns the actual window lifetime.
            command = ["/usr/bin/open", "-n", "-a", str(browser_path.parents[2]), "--args", *browser_args]
        else:
            command = [str(browser_path), *browser_args]
        browser = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _activate_macos_browser_window(browser_path)
    except Exception:
        shutil.rmtree(profile_dir, ignore_errors=True)
        raise
    return browser, profile_dir


def _wait_for_browser_session(browser: _ManagedProcess, profile_dir: Path) -> None:
    """Wait for the actual browser window lifetime on every supported OS."""

    if sys.platform != "darwin":
        browser.wait()
        return
    _wait_for_macos_browser_profile(profile_dir, _find_browser())


def _wait_for_macos_browser_profile(profile_dir: Path, browser_path: Path) -> None:
    """Wait for the main browser process carrying this session's profile.

    Chrome on macOS forks its application process, so the PID returned by
    ``Popen`` can exit while the visible app window is still open. The unique
    user-data directory is a stable ownership marker for the real app. Once
    that main process has appeared and subsequently disappears, the UI window
    is definitively closed and its local server may be stopped.
    """

    deadline = time.monotonic() + 10.0
    profile_stable_since: float | None = None
    observed_stable_profile = False
    missing_since: float | None = None
    while True:
        now = time.monotonic()
        if _macos_browser_profile_pids(profile_dir, browser_path):
            if profile_stable_since is None:
                profile_stable_since = now
            # ``open -n`` can briefly create one Chrome process for the
            # isolated profile, then replace it with the real app process.
            # Treating that first transient process as the window owner can
            # make the supervisor stop the server just before the visible
            # app window appears.  A profile must remain present briefly
            # before its disappearance means that the user closed the UI.
            if now - profile_stable_since >= 2.0:
                observed_stable_profile = True
            missing_since = None
        else:
            if not observed_stable_profile:
                # A startup-only Chrome process disappeared.  Keep the
                # server alive while the app finishes replacing it, and make
                # the eventual real process establish its own stable window.
                profile_stable_since = None
            else:
            # ``pgrep`` can momentarily miss a process during a macOS app
            # activation or child-process change. Do not tear down a working
            # slicer server on one transient observation.
                if missing_since is None:
                    missing_since = now
                # Chrome can take several seconds to replace its initial app
                # process even after the window was already visible.  A
                # longer grace period prevents that hand-off from leaving an
                # orphaned app window that has no local UI server.
                elif now - missing_since >= 5.0:
                    return
            if now >= deadline:
                raise RuntimeError("macOS browser session did not create its isolated window process")
        # The launcher may have exited already; keep waiting until the unique
        # profile process appears or the bounded startup deadline is reached.
        time.sleep(0.1)


def _macos_browser_profile_pids(profile_dir: Path, browser_path: Path) -> tuple[int, ...]:
    """Return live main-browser PIDs using a particular temporary profile."""

    profile_argument = f"--user-data-dir={profile_dir}"
    try:
        completed = subprocess.run(
            # Match the executable at the command-line start. This excludes
            # the ``open`` launcher and transient Chrome helper subprocesses.
            [
                "/usr/bin/pgrep",
                "-f",
                rf"^{re.escape(str(browser_path))} .*{re.escape(profile_argument)}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if completed.returncode not in (0, 1):
        return ()
    return tuple(int(line) for line in completed.stdout.splitlines() if line.strip().isdigit())


def _activate_macos_browser_window(browser_path: Path) -> None:
    """Bring a newly-created macOS browser app window to the foreground.

    Directly launching Chrome's executable is necessary to own the temporary
    profile and accurately stop the matching design-server session. Unlike
    ``open -a``, though, it may leave the app window behind the main slicer or
    on a different Space.  Activating its containing ``.app`` makes a designer
    launch visible without changing how Windows sessions behave.
    """

    if sys.platform != "darwin":
        return
    try:
        app_bundle = browser_path.parents[2]
        app_name = app_bundle.stem
    except IndexError:
        return
    if not app_name:
        return
    try:
        subprocess.run(
            ["/usr/bin/osascript", "-e", f'tell application "{app_name}" to activate'],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        # The window may already be frontmost, and activation must never turn
        # a working local designer session into a failed launch.
        pass


def _find_browser() -> Path:
    configured = os.environ.get("KUKA_SLICER_BROWSER")
    candidates = [Path(configured)] if configured else []
    if sys.platform == "darwin":
        candidates.extend(_MACOS_BROWSER_PATHS)
    else:
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


def _stop_process(process: _ManagedProcess) -> None:
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
