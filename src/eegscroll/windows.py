"""Cut a fixed EEG window from each video's start."""

import numpy as np
import pandas as pd

from eegscroll.labels import WINDOW_S


def covered_mask(eeg_times: np.ndarray, videos: pd.DataFrame, window_s: float, sfreq: float) -> np.ndarray:
    """True where EEG fully covers the video's window."""
    # Allow a few missing samples, not more
    starts = videos["start_time"].to_numpy()
    if len(eeg_times) == 0:  # session with no EEG at all
        return np.zeros(len(starts), dtype=bool)
    idx = np.searchsorted(eeg_times, starts)
    n = int(window_s * sfreq)
    ok = idx + n <= len(eeg_times)
    safe = np.minimum(idx + n - 1, len(eeg_times) - 1)
    span = eeg_times[safe] - starts
    return ok & (span <= window_s + 0.1) & (span >= window_s - 0.1)


def cut_windows(
    eeg: np.ndarray,
    eeg_times: np.ndarray,
    videos: pd.DataFrame,
    window_s: float = WINDOW_S,
    sfreq: float = 256.0,
) -> np.ndarray:
    """Return array of shape (n_videos, n_channels, n_samples).

    Uses only [start_time, start_time + window_s). The seconds before
    the key press contain planning the press, so never use them.
    """
    # Window can't reach past video end
    if (videos["start_time"] + window_s > videos["end_time"]).any():
        raise ValueError("A window would pass its video's end; filter videos first.")

    # Same sample count for every window
    n = int(window_s * sfreq)

    # First EEG sample at each video start
    idx = np.searchsorted(eeg_times, videos["start_time"].to_numpy())

    # Slice and stack into one 3D array
    return np.stack([eeg[:, i : i + n] for i in idx])
