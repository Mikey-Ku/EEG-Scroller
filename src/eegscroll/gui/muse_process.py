"""Runs `muselsl stream --lsltime` for the app and reports what it says."""

import shutil
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

# Lines muselsl prints -> short status for the headset panel
STATUS_LINES = [
    ("No Muses found", "No headset found"),
    ("Disconnected", "Headset disconnected"),
    ("Streaming", "Streaming"),
    ("Connected", "Connected"),
    ("Connecting", "Connecting..."),
    ("Searching", "Searching for headset..."),
    ("Looking for", "Searching for headset..."),
]


def find_muselsl() -> str | None:
    """muselsl from PATH, or where `uv tool install` puts it."""
    found = shutil.which("muselsl")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "muselsl"
    return str(local) if local.exists() else None


class MuseProcess(QObject):
    """One muselsl stream process, kept for as long as the app runs."""

    status = Signal(str)
    line = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")  # see lines as they happen
        self._proc.setProcessEnvironment(env)
        self._proc.readyReadStandardOutput.connect(self._read)
        self._proc.finished.connect(lambda *_: self.status.emit("Stream stopped"))
        self._proc.errorOccurred.connect(lambda e: self.status.emit(f"muselsl failed: {e.name}"))

    @property
    def running(self) -> bool:
        return self._proc.state() != QProcess.ProcessState.NotRunning

    def start(self, name: str = "") -> None:
        exe = find_muselsl()
        if exe is None:
            self.status.emit("muselsl not found (uv tool install muselsl)")
            return
        args = ["stream", "--lsltime"] + (["--name", name] if name else [])
        self.status.emit("Starting muselsl...")
        self.line.emit(f"$ muselsl {' '.join(args)}")
        self._proc.start(exe, args)

    def stop(self) -> None:
        if not self.running:
            return
        self._proc.terminate()
        if not self._proc.waitForFinished(3000):
            self._proc.kill()
            self._proc.waitForFinished(1000)

    def _read(self) -> None:
        text = bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            self.line.emit(line)
            for needle, status in STATUS_LINES:
                if needle in line:
                    self.status.emit(status)
                    break
