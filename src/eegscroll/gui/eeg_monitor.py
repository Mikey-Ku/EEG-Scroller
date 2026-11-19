"""Always-on EEG reader for the app: live buffer, signal quality, recording."""

import threading

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from eegscroll.clock import now
from eegscroll.data import CHANNELS, SFREQ

KEEP_S = 30.0  # seconds kept between sessions (enough for the plot)
QUIET_S = 3.0  # no samples this long: look for the stream again
SIM_STREAM = "Simulated Muse"  # name of the app's --sim stream

# Signal size after a 1-40 Hz filter, microvolts: good below, poor above
GOOD_UV, POOR_UV = 50.0, 100.0

_BAND = butter(4, [1, 40], btype="band", fs=SFREQ, output="sos")


def bandpass(x: np.ndarray) -> np.ndarray:
    """1-40 Hz along the last axis; removes drift and mains hum."""
    if x.shape[-1] < 64:
        return x - x.mean(axis=-1, keepdims=True)
    return sosfiltfilt(_BAND, x, axis=-1)


class EEGMonitor:
    """Background thread reading the LSL EEG stream for the app's lifetime.

    Keeps the last KEEP_S seconds for plotting; while recording,
    keeps everything so the session can be cut out afterwards.
    """

    def __init__(self, stream_name: str | None = None) -> None:
        self.wanted_name = stream_name  # None = any EEG stream
        self._lock = threading.Lock()
        self._times: list[np.ndarray] = []
        self._samples: list[np.ndarray] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._recording = False
        self.stream_name: str | None = None
        self.last_data: float | None = None  # clock time the last chunk arrived
        self.error: str | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def streaming(self) -> bool:
        return self.last_data is not None and now() - self.last_data < 2.0

    def _run(self) -> None:
        from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_byprop

        while not self._stop.is_set():
            # Wait for an EEG stream (muselsl, or the simulated one)
            if self.wanted_name:
                streams = resolve_byprop("name", self.wanted_name, timeout=1)
            else:
                # Real headset: never fall back to a simulated stream
                streams = [s for s in resolve_byprop("type", "EEG", timeout=1) if s.name() != SIM_STREAM]
            if not streams:
                continue
            info = streams[0]
            inlet = StreamInlet(info, processing_flags=proc_clocksync | proc_dejitter)
            self.stream_name = info.name()
            last = now()

            # Read until stopped or the stream goes quiet
            while not self._stop.is_set():
                x, t = inlet.pull_chunk(timeout=0.2)
                if not t:
                    if now() - last > QUIET_S:
                        break
                    continue
                last = now()
                t = np.asarray(t)

                # muselsl without --lsltime stamps Unix time; it would never line up
                if abs(t[-1] - last) > 3600:
                    self.error = "EEG timestamps are Unix time: restart the stream with --lsltime"
                    continue
                self.error = None
                self.last_data = last
                with self._lock:
                    self._times.append(t)
                    self._samples.append(np.asarray(x)[:, : len(CHANNELS)])
                    if not self._recording:
                        self._trim(last - KEEP_S)
            inlet.close_stream()
            self.stream_name = None

    def _trim(self, before: float) -> None:
        # Drop whole chunks that ended before `before` (lock held)
        while self._times and self._times[0][-1] < before:
            self._times.pop(0)
            self._samples.pop(0)

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        # (channels, samples), times (lock held)
        if not self._times:
            return np.empty((len(CHANNELS), 0)), np.empty(0)
        return np.concatenate(self._samples).T, np.concatenate(self._times)

    def recent(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """Last `seconds` of EEG: (channels, samples), times."""
        with self._lock:
            x, t = self._arrays()
        n = int(seconds * SFREQ)
        return x[:, -n:], t[-n:]

    def measured_rate(self, seconds: float = 4.0) -> float | None:
        _, t = self.recent(seconds)
        if t.size < 2 or t[-1] == t[0]:
            return None
        return (t.size - 1) / (t[-1] - t[0])

    def quality(self, seconds: float = 2.0) -> list[dict]:
        """Per channel: filtered signal size (uV), mains-hum ratio, and a verdict."""
        x, _ = self.recent(seconds + 2.0)  # extra for filter edges
        if x.shape[1] < SFREQ:
            return []
        y = bandpass(x)[:, -int(seconds * SFREQ) :]
        f, p = welch(x, fs=SFREQ, nperseg=min(256, x.shape[1]))
        hum = p[:, (f >= 58) & (f <= 62)].sum(1) / np.maximum(p[:, (f >= 1) & (f < 40)].sum(1), 1e-12)
        out = []
        for i, name in enumerate(CHANNELS):
            size = float(y[i].std())
            level = "good" if size < GOOD_UV else "ok" if size < POOR_UV else "poor"
            out.append({"channel": name, "uv": size, "hum": float(hum[i]), "level": level})
        return out

    def begin_recording(self) -> None:
        with self._lock:
            self._recording = True

    def end_recording(self, t_start: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
        """Stop keeping everything; return the EEG between t_start and t_end."""
        with self._lock:
            x, t = self._arrays()
            self._recording = False
            self._trim(now() - KEEP_S)
        keep = (t >= t_start) & (t <= t_end)
        return x[:, keep], t[keep]


def run_fake_stream(stop: threading.Event) -> None:
    """Publish simulated Muse-like EEG on LSL (for trying the app without a headset)."""
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(SIM_STREAM, "EEG", len(CHANNELS) + 1, SFREQ, "float32", "eegscroll-sim")
    outlet = StreamOutlet(info)
    rng = np.random.default_rng()
    chunk, sent, t0 = 16, 0, now()
    while not stop.is_set():
        stop.wait(max(0.0, t0 + (sent + chunk) / SFREQ - now()))
        t = t0 + (sent + np.arange(chunk)) / SFREQ
        x = rng.normal(0, 10, (chunk, len(CHANNELS) + 1)) + 15 * np.sin(2 * np.pi * 10 * t)[:, None]
        outlet.push_chunk(x.tolist(), float(t[-1]))  # stamp of the newest sample
        sent += chunk
