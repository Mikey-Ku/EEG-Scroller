"""EEG inputs: a simulated Muse now, the real Muse later."""

import threading

import numpy as np

from eegscroll.clock import now
from eegscroll.data import CHANNELS, SFREQ


class EEGSource:
    """Base class: background thread collecting timestamped samples."""

    name = "base"

    def __init__(self) -> None:
        self._times: list[np.ndarray] = []
        self._samples: list[np.ndarray] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error: str | None = None

    # Start and stop the background thread
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        raise NotImplementedError

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (eeg (channels, samples), times)."""
        if not self._times:
            return np.empty((len(CHANNELS), 0)), np.empty(0)
        return np.concatenate(self._samples).T, np.concatenate(self._times)


class SimulatedMuse(EEGSource):
    """Template input: fake 4-channel EEG at 256 Hz."""

    name = "simulated"

    def _run(self) -> None:
        rng = np.random.default_rng()
        chunk, sent, t0 = 16, 0, now()
        while not self._stop.is_set():
            # Wait until the next chunk is due
            self._stop.wait(max(0.0, t0 + (sent + chunk) / SFREQ - now()))

            # Noise plus a little 10 Hz rhythm
            t = t0 + (sent + np.arange(chunk)) / SFREQ
            x = rng.normal(0, 10, (chunk, len(CHANNELS))) + 3 * np.sin(2 * np.pi * 10 * t)[:, None]
            self._times.append(t)
            self._samples.append(x)
            sent += chunk


class MuseLSL(EEGSource):
    """Real Muse 2: run `muselsl stream` first, then this."""

    name = "muse"

    def _run(self) -> None:
        from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_byprop

        # Find the EEG stream muselsl publishes
        streams = resolve_byprop("type", "EEG", timeout=10)
        if not streams:
            self.error = "No EEG stream found. Is `muselsl stream` running?"
            print(self.error)
            return

        # Clock sync maps Muse times onto our clock
        inlet = StreamInlet(streams[0], processing_flags=proc_clocksync | proc_dejitter)
        while not self._stop.is_set():
            x, t = inlet.pull_chunk(timeout=0.5)
            if t:
                # Muse sends 5 channels; keep first 4
                self._times.append(np.asarray(t))
                self._samples.append(np.asarray(x)[:, : len(CHANNELS)])


def make_source(kind: str) -> EEGSource:
    """'sim' or 'muse' -> an EEG source."""
    return {"sim": SimulatedMuse, "muse": MuseLSL}[kind]()

