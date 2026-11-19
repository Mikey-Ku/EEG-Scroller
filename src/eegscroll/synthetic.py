"""Fake sessions shaped like real ones, with a known effect."""

import numpy as np
import pandas as pd

from eegscroll.data import CHANNELS, SFREQ, Recording, Session
from eegscroll.labels import STAY_S, WINDOW_S


def make_synthetic_session(
    session_id: str,
    n_videos: int = 60,
    alpha_effect: float = 0.5,
    window_s: float = WINDOW_S,
    seed: int = 0,
) -> Session:
    """A Session like record.py saves, minus webcam.

    alpha_effect = 0 means no real signal; models should
    score near chance (0.5 balanced accuracy).
    """
    # Reproducible randomness
    rng = np.random.default_rng(seed)

    # Random reel lengths, about half watched (stayed past STAY_S)
    durations = rng.uniform(8, 60, n_videos)
    watched = rng.random(n_videos) < 0.5
    watch_times = np.where(
        watched,
        rng.uniform(STAY_S + 0.5, 30, n_videos),  # stayed
        rng.uniform(1.6, STAY_S - 0.5, n_videos),  # left early
    )

    # Reel start times; one extra final reel
    starts = 1.0 + np.concatenate([[0.0], np.cumsum(watch_times)])

    # Background EEG: white noise, about 10 µV
    n_samples = int((starts[-1] + 5) * SFREQ)
    t = np.arange(n_samples) / SFREQ
    eeg = rng.normal(0, 10, (len(CHANNELS), n_samples))

    # Inject effect: extra 10 Hz on watched reels
    n_win = int(window_s * SFREQ)
    for start, w in zip(starts[:-1], watched):
        if w:
            i = np.searchsorted(t, start)
            phase = rng.uniform(0, 2 * np.pi)
            eeg[:, i : i + n_win] += alpha_effect * 5 * np.sin(2 * np.pi * 10 * t[i : i + n_win] + phase)

    # Same tables the browser recorder produces
    videos = pd.DataFrame(
        {"t": starts, "video_id": [str(i) for i in range(n_videos + 1)], "duration": np.append(durations, 30.0)}
    )
    keys = pd.DataFrame({"t": starts[1:], "key": "ArrowDown"})
    rec = Recording(session_id, eeg, t)
    meta = {"session_id": session_id, "eeg_source": "synthetic", "sfreq": SFREQ, "channels": CHANNELS}
    return Session(rec, videos, None, keys, meta)
