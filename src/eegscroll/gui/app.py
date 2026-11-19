"""EEG Scroller app: headset, live EEG, webcam, session controls, and Reels in one window."""

import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QProcess, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QFont, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from eegscroll.browser import REELS_URL, ReelWatcher
from eegscroll.clock import now
from eegscroll.data import CHANNELS, SESSIONS_DIR, SFREQ
from eegscroll.gui.eeg_monitor import SIM_STREAM, EEGMonitor, bandpass, run_fake_stream
from eegscroll.gui.muse_process import MuseProcess
from eegscroll.gui.reels_view import ReelsView
from eegscroll.quality import session_report
from eegscroll.session import new_session_id, save_recorded
from eegscroll.webcam import WebcamRecorder, default_camera, list_cameras

PLOT_S = 5.0  # seconds of EEG on screen
PLOT_UV = 150.0  # y range, +/- microvolts
PANEL_W = 460  # control column width; Reels gets the rest

# Where each Muse 2 sensor sits on the head
PLACES = {"TP9": "left ear", "AF7": "left forehead", "AF8": "right forehead", "TP10": "right ear"}

COLORS = {"good": "#2e9d4f", "ok": "#d39b1c", "poor": "#d0453b", "off": "#8a8f98"}
RETRAIN_CODE = "from eegscroll.train import retrain; retrain(real_only=True)"


CAMERA_BLOCKED = (
    "Camera access is blocked. Allow it for the app you launched this from "
    "(Terminal, or Claude) in System Settings > Privacy & Security > Camera, then restart."
)


def request_camera(done) -> None:
    """Ask macOS for camera access from the main thread; done(granted) may run on any thread.

    OpenCV asks from its own thread, which macOS ignores, so the prompt never shows.
    """
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    if status == 3:  # authorized
        done(True)
    elif status == 0:  # not asked yet: show the macOS prompt

        def answered(granted) -> None:  # WebKit/AVFoundation blocks must return None
            done(bool(granted))

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, answered)
    else:  # denied or restricted
        done(False)


def dot(color: str) -> str:
    return f'<span style="color:{color}; font-size:16px">&#9679;</span>'


class MainWindow(QMainWindow):
    def __init__(self, simulate: bool = False, url: str = REELS_URL, sessions_dir: Path = SESSIONS_DIR) -> None:
        super().__init__()
        self.setWindowTitle("EEG Scroller")
        self.simulate = simulate
        self.url = url
        self.sessions_dir = sessions_dir

        # Session state; watcher is not None while recording
        self.watcher: ReelWatcher | None = None
        self.session_id = ""
        self.t_start = 0.0
        self.last_saved: Path | None = None

        # Background inputs, alive for the whole app
        self.eeg = EEGMonitor(SIM_STREAM if simulate else None)
        self.settings = QSettings("EEGScroller", "app")
        self.cam = WebcamRecorder(self._saved_camera())
        self.muse = MuseProcess(self)
        self._fake_stop = threading.Event()
        self.retrain_proc: QProcess | None = None

        self._build_ui()
        self._build_edit_menu()
        self.muse.status.connect(self._on_muse_status)
        self.muse.line.connect(lambda line: self.log(f"muselsl: {line}"))

        # Start inputs, then refresh the panel 20 times a second
        self.eeg.start()
        self.camera_allowed: bool | None = None  # set by the macOS prompt, started in _refresh
        self.camera_started = False
        request_camera(lambda granted: setattr(self, "camera_allowed", granted))
        if simulate:
            threading.Thread(target=run_fake_stream, args=(self._fake_stop,), daemon=True).start()
            self.muse_status = "Simulated EEG"
        else:
            self.muse_status = "Checking for a running stream..."
            QTimer.singleShot(2500, self._auto_connect)
        self.reels.load(url)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(50)

    # ---------- layout ----------

    def _build_ui(self) -> None:
        panel = QWidget()
        col = QVBoxLayout(panel)

        # Headset
        box = QGroupBox("Headset")
        g = QVBoxLayout(box)
        row = QHBoxLayout()
        self.headset_label = QLabel()
        self.headset_label.setTextFormat(Qt.TextFormat.RichText)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._toggle_muse)
        row.addWidget(self.headset_label, 1)
        row.addWidget(self.connect_btn)
        g.addLayout(row)
        lights = QGridLayout()
        self.lights: dict[str, QLabel] = {}
        for i, name in enumerate(CHANNELS):
            lab = QLabel()
            lab.setTextFormat(Qt.TextFormat.RichText)
            lights.addWidget(lab, 0, i)
            self.lights[name] = lab
        g.addLayout(lights)
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        g.addWidget(self.hint_label)

        # Live EEG, one row per channel, 1-40 Hz
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        self.plot = pg.GraphicsLayoutWidget()
        self.plot.setMinimumHeight(200)
        self.curves = []
        for i, name in enumerate(CHANNELS):
            p = self.plot.addPlot(row=i, col=0)
            p.setYRange(-PLOT_UV, PLOT_UV, padding=0)
            p.setXRange(-PLOT_S, 0, padding=0)
            p.setMouseEnabled(False, False)
            p.hideButtons()
            p.getAxis("left").setWidth(34)
            label = pg.TextItem(f"{name} · {PLACES[name]}", color="#333", anchor=(0, 0))
            label.setPos(-PLOT_S, PLOT_UV)
            p.addItem(label)
            if i < len(CHANNELS) - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            self.curves.append(p.plot(pen=pg.mkPen("#2b5fab", width=1)))

        # Webcam preview
        cam_box = QGroupBox("Webcam")
        cam_col = QVBoxLayout(cam_box)
        self.camera_box = QComboBox()
        for cam in list_cameras():
            self.camera_box.addItem(cam["name"], cam)
            if cam["index"] == self.cam.camera_index:
                self.camera_box.setCurrentIndex(self.camera_box.count() - 1)
        self.camera_box.currentIndexChanged.connect(self._switch_camera)
        cam_col.addWidget(self.camera_box)
        cv = QHBoxLayout()
        cam_col.addLayout(cv)
        self.cam_view = QLabel("starting...")
        self.cam_view.setFixedSize(192, 108)
        self.cam_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_view.setStyleSheet("background:#222; color:#bbb")
        self.cam_label = QLabel()
        self.cam_label.setTextFormat(Qt.TextFormat.RichText)
        self.cam_label.setWordWrap(True)
        cv.addWidget(self.cam_view)
        cv.addWidget(self.cam_label, 1)

        # Session controls
        s_box = QGroupBox("Session")
        sv = QVBoxLayout(s_box)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. david-desk")
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("anything odd: tired, noisy room...")
        form.addRow("Name", self.name_edit)
        form.addRow("Notes", self.notes_edit)
        sv.addLayout(form)
        self.start_btn = QPushButton("Start session")
        self.start_btn.setMinimumHeight(40)
        f = self.start_btn.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.start_btn.setFont(f)
        self.start_btn.clicked.connect(self._toggle_session)
        sv.addWidget(self.start_btn)
        self.session_label = QLabel("Not recording")
        sv.addWidget(self.session_label)
        self.reel_list = QListWidget()
        self.reel_list.setMaximumHeight(80)
        sv.addWidget(self.reel_list)
        tools = QHBoxLayout()
        self.reload_btn = QPushButton("Reload Reels")  # connected once the view exists
        open_btn = QPushButton("Open sessions folder")
        open_btn.clicked.connect(lambda: QProcess.startDetached("open", [str(self.sessions_dir)]))
        tools.addWidget(self.reload_btn)
        tools.addWidget(open_btn)
        sv.addLayout(tools)

        # Log
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setFont(QFont("Menlo", 10))
        self.log_box.setFixedHeight(90)

        # Order: what you check first, then the controls you use most
        col.addWidget(box)
        col.addWidget(s_box)
        col.addWidget(self.plot, 1)
        col.addWidget(cam_box)
        col.addWidget(self.log_box)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(PANEL_W)
        scroll.setMaximumWidth(PANEL_W + 80)

        # Reels fill the rest of the window
        self.reels = ReelsView(self._on_reel_event)
        self.reload_btn.clicked.connect(self.reels.reload)
        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.reels)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([PANEL_W, 1000])
        self.setCentralWidget(split)

    def _build_edit_menu(self) -> None:
        """Edit menu, so Cmd+C/V/X/A/Z also work inside the embedded Reels page."""
        menu = self.menuBar().addMenu("Edit")
        items = [
            ("Undo", QKeySequence.StandardKey.Undo, "undo:", "undo"),
            ("Redo", QKeySequence.StandardKey.Redo, "redo:", "redo"),
            ("Cut", QKeySequence.StandardKey.Cut, "cut:", "cut"),
            ("Copy", QKeySequence.StandardKey.Copy, "copy:", "copy"),
            ("Paste", QKeySequence.StandardKey.Paste, "paste:", "paste"),
            ("Select All", QKeySequence.StandardKey.SelectAll, "selectAll:", "selectAll"),
        ]
        for label, key, web_action, qt_method in items:
            action = QAction(label, self)
            action.setShortcut(key)
            action.triggered.connect(lambda _=False, w=web_action, q=qt_method: self._edit(w, q))
            menu.addAction(action)

    def _edit(self, web_action: str, qt_method: str) -> None:
        # Page has focus: let WebKit do it; else the focused text box
        if self.reels.has_focus():
            self.reels.send_edit(web_action)
            return
        widget = QApplication.focusWidget()
        if widget is not None and hasattr(widget, qt_method):
            getattr(widget, qt_method)()

    def log(self, text: str) -> None:
        self.log_box.appendPlainText(text)

    # ---------- headset ----------

    def _auto_connect(self) -> None:
        # Reuse a stream that's already running (e.g. muselsl in a terminal)
        if self.eeg.streaming:
            self.muse_status = "Using existing stream"
            self.log(f"Found a running EEG stream ({self.eeg.stream_name}); not starting muselsl.")
        else:
            self.muse.start()

    def _toggle_muse(self) -> None:
        if self.simulate:
            return
        if self.muse.running:
            self.muse.stop()
        else:
            self.muse.start()

    def _on_muse_status(self, status: str) -> None:
        self.muse_status = status
        self.log(status)

    def _refresh_headset(self) -> None:
        streaming = self.eeg.streaming
        rate = self.eeg.measured_rate() if streaming else None
        if self.eeg.error:
            color, text = COLORS["poor"], self.eeg.error
        elif streaming:
            color = COLORS["good"] if rate and rate > 0.95 * SFREQ else COLORS["ok"]
            text = f"Streaming: {self.eeg.stream_name}, {rate:.0f} Hz" if rate else "Streaming"
        else:
            color, text = COLORS["off"], self.muse_status
        self.headset_label.setText(f"{dot(color)} {text}")
        self.connect_btn.setVisible(not self.simulate)
        self.connect_btn.setText("Disconnect" if self.muse.running else "Connect")

        # Per-channel lights and a hint for the worst problem
        quality = self.eeg.quality() if streaming else []
        for name in CHANNELS:
            q = next((q for q in quality if q["channel"] == name), None)
            if q is None:
                self.lights[name].setText(f"{dot(COLORS['off'])} {name}<br><small>{PLACES[name]}</small>")
            else:
                self.lights[name].setText(
                    f"{dot(COLORS[q['level']])} {name} {q['uv']:.0f} uV<br><small>{PLACES[name]}</small>"
                )
        hints = []
        if any(q["hum"] > 0.5 for q in quality):
            hints.append("60 Hz hum: unplug the charger.")
        poor = [q["channel"] for q in quality if q["level"] == "poor"]
        if poor:
            where = " and ".join(
                part for part, chans in (("ear pads", {"TP9", "TP10"}), ("forehead strip", {"AF7", "AF8"}))
                if chans & set(poor)
            )
            hints.append(f"Poor contact on {', '.join(poor)}: adjust the {where}, keep still.")
        self.hint_label.setText(" ".join(hints) or ("Signal looks good." if quality else ""))

    def _refresh_plot(self) -> None:
        x, t = self.eeg.recent(PLOT_S + 1.0)
        if t.size < 64:
            for c in self.curves:
                c.setData([], [])
            return
        y = bandpass(x)
        keep = t >= t[-1] - PLOT_S
        rel = t[keep] - t[-1]
        for i, c in enumerate(self.curves):
            c.setData(rel, np.clip(y[i, keep], -PLOT_UV, PLOT_UV))

    # ---------- webcam ----------

    def _refresh_cam(self) -> None:
        # Start the camera once macOS says yes
        if self.camera_allowed and not self.camera_started:
            self.camera_started = True
            self.cam.start()
        elif self.camera_allowed is False:
            self.cam.error = CAMERA_BLOCKED
        elif self.camera_allowed is None:
            self.cam_view.setText("waiting...")
            self.cam_label.setText(f"{dot(COLORS['off'])} waiting for camera permission (check for a macOS prompt)")
            return
        if self.cam.error:
            self.cam_view.setText("no camera")
            self.cam_label.setText(f"{dot(COLORS['poor'])} {self.cam.error}")
            return
        frame, row = self.cam.latest_frame, self.cam.latest_row
        if frame is not None:
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            h, w, _ = rgb.shape
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.cam_view.setPixmap(
                QPixmap.fromImage(img).scaled(self.cam_view.size(), Qt.AspectRatioMode.KeepAspectRatio)
            )
        if row is None:
            self.cam_label.setText(f"{dot(COLORS['off'])} starting camera...")
            return
        age = now() - row["t"]
        if age > 2:
            self.cam_label.setText(f"{dot(COLORS['poor'])} camera stopped {age:.0f} s ago (another app using it?)")
        elif not row["face"]:
            self.cam_label.setText(f"{dot(COLORS['ok'])} no face found: face the camera, check the light")
        else:
            blink = row["blink"]
            state = "blink" if blink > 0.3 else "eyes open"
            self.cam_label.setText(f"{dot(COLORS['good'])} face found<br>{state} ({blink:.2f})")

        # Between sessions, don't keep old webcam rows
        if self.watcher is None and len(self.cam.rows) > 600:
            self.cam.rows.clear()

    def _saved_camera(self) -> int:
        """Camera picked last time (by id, since numbers can shift), else the built-in one."""
        saved = self.settings.value("camera_id", "")
        match = [c["index"] for c in list_cameras() if c["id"] == saved]
        return match[0] if match else default_camera()

    def _switch_camera(self, _row: int) -> None:
        cam = self.camera_box.currentData()
        if cam is None or cam["index"] == self.cam.camera_index:
            return
        self.settings.setValue("camera_id", cam["id"])
        if self.camera_started:
            self.cam.stop()
        self.cam = WebcamRecorder(cam["index"])
        self.camera_started = False  # _refresh_cam starts it
        self.cam_view.clear()
        self.log(f"Webcam: {cam['name']}")

    # ---------- sessions ----------

    def _on_reel_event(self, event: dict) -> None:
        if self.watcher is None:
            return  # browsing between sessions isn't recorded
        n_before = len(self.watcher.videos)
        self.watcher.handle(event)
        if len(self.watcher.videos) > n_before:
            v = self.watcher.videos[-1]
            length = f"{v['duration']:.0f} s" if v["duration"] is not None else "no video"
            self.reel_list.addItem(f"{len(self.watcher.videos)}. {length}  {v['caption'][:40]}")
            self.reel_list.scrollToBottom()

    def _toggle_session(self) -> None:
        if self.watcher is None:
            self.start_session()
        else:
            folder = self.stop_session()
            self.review_session(folder)

    def start_session(self, ask: bool = True) -> None:
        if ask and not self.eeg.streaming:
            answer = QMessageBox.question(self, "No EEG", "The headset isn't streaming. Start the session anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.session_id = new_session_id(self.name_edit.text())
        self.watcher = ReelWatcher()
        self.t_start = now()
        self.eeg.begin_recording()
        self.cam.rows.clear()
        self.reel_list.clear()
        self.start_btn.setText("Stop session")
        self.start_btn.setStyleSheet("background:#d0453b; color:white")
        self.name_edit.setEnabled(False)
        self.camera_box.setEnabled(False)  # same camera for the whole session
        self.log(f"Recording {self.session_id}. Scroll with the arrow keys; Stop or q to end.")
        self.reels.focus_page()

    def stop_session(self) -> Path:
        """Stop recording and save; returns the session folder."""
        t_end = now()
        watcher, self.watcher = self.watcher, None
        eeg_data, eeg_times = self.eeg.end_recording(self.t_start, t_end)
        frames = None
        if self.camera_started and not self.cam.error:
            frames = self.cam.to_frame()
            frames = frames[frames["t"] >= self.t_start].reset_index(drop=True)
        folder = save_recorded(
            self.session_id,
            "simulated" if self.simulate else "muse",
            eeg_data,
            eeg_times,
            watcher,
            frames,
            self.t_start,
            t_end,
            url=self.url,
            notes=self.notes_edit.text(),
            root=self.sessions_dir,
        )
        self.last_saved = folder
        self.start_btn.setText("Start session")
        self.start_btn.setStyleSheet("")
        self.name_edit.setEnabled(True)
        self.camera_box.setEnabled(True)
        self.log(f"Saved {folder.name}: {len(watcher.videos)} reels, {eeg_times.size} EEG samples")
        return folder

    def review_session(self, folder: Path) -> None:
        """Show the quality report; keep, keep and retrain, or delete."""
        try:
            text, problems = session_report(folder)
        except Exception as exc:
            text, problems = f"Could not check {folder.name}: {exc}", ["Report failed."]
        verdict = "Looks good." if not problems else "Problems:\n" + "\n".join(f"- {p}" for p in problems)
        self.log(text + "\n" + verdict)

        box = QMessageBox(self)
        box.setWindowTitle("Session saved")
        box.setText(f"{folder.name}\n\n{verdict}")
        box.setDetailedText(text)
        retrain_btn = box.addButton("Keep and retrain", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Keep", QMessageBox.ButtonRole.RejectRole)
        delete_btn = box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(retrain_btn)
        box.exec()
        if box.clickedButton() is delete_btn:
            shutil.rmtree(folder)
            self.log(f"Deleted {folder.name}")
        elif box.clickedButton() is retrain_btn:
            self.retrain()

    def retrain(self) -> None:
        if self.retrain_proc and self.retrain_proc.state() != QProcess.ProcessState.NotRunning:
            self.log("Retraining is already running.")
            return
        self.log("Retraining on all real sessions...")
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self.log(bytes(proc.readAllStandardOutput()).decode(errors="replace").rstrip())
        )
        proc.finished.connect(lambda code, _s: self.log("Retrain done." if code == 0 else f"Retrain failed ({code})."))
        proc.start(sys.executable, ["-c", RETRAIN_CODE])
        self.retrain_proc = proc

    # ---------- refresh / close ----------

    def _refresh(self) -> None:
        self._refresh_headset()
        self._refresh_plot()
        self._refresh_cam()
        if self.watcher is not None:
            elapsed = now() - self.t_start
            self.session_label.setText(
                f"Recording {self.session_id}  {int(elapsed // 60)}:{int(elapsed % 60):02d}  "
                f"reels {len(self.watcher.videos)}"
            )
            if self.watcher.stop.is_set():  # q pressed in the page
                self._toggle_session()
        else:
            self.session_label.setText("Not recording")

    def closeEvent(self, event) -> None:
        if self.watcher is not None:
            answer = QMessageBox.question(self, "Recording", "A session is recording. Stop and save it before quitting?")
            if answer == QMessageBox.StandardButton.Yes:
                self.review_session(self.stop_session())
            else:
                self.watcher = None
        self.timer.stop()
        self.muse.stop()
        self._fake_stop.set()
        self.eeg.stop()
        if self.camera_started:
            self.cam.stop()
        event.accept()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="EEG Scroller app")
    parser.add_argument("--sim", action="store_true", help="simulated EEG instead of the Muse")
    parser.add_argument("--url", default=REELS_URL, help="page to show (default Instagram Reels)")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("EEG Scroller")
    win = MainWindow(simulate=args.sim, url=args.url)
    win.showMaximized()
    sys.exit(app.exec())
