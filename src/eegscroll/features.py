"""EEG band power and webcam features per video."""

import numpy as np
import pandas as pd
from scipy.signal import welch

# Frequency bands in Hz: (low, high)
BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Range used as "total" for relative power
TOTAL_RANGE: tuple[float, float] = (1.0, 40.0)


def band_power(windows: np.ndarray, sfreq: float, ch_names: list[str]) -> pd.DataFrame:
    """Windows (n_videos, n_channels, n_samples) -> feature table.

    Columns like AF7_alpha_abs, AF7_alpha_rel, AF7_engagement.
    """
    # Power spectrum per window and channel
    nperseg = min(windows.shape[-1], int(2 * sfreq))
    freqs, psd = welch(windows, fs=sfreq, nperseg=nperseg, axis=-1)
    df = freqs[1] - freqs[0]

    # Area under the spectrum between two frequencies
    def power(lo: float, hi: float) -> np.ndarray:
        mask = (freqs >= lo) & (freqs < hi)
        return psd[..., mask].sum(axis=-1) * df

    total = power(*TOTAL_RANGE)
    absolute = {band: power(lo, hi) for band, (lo, hi) in BANDS.items()}

    # Readable columns per channel and band
    cols: dict[str, np.ndarray] = {}
    for c, ch in enumerate(ch_names):
        for band in BANDS:
            cols[f"{ch}_{band}_abs"] = absolute[band][:, c]
            cols[f"{ch}_{band}_rel"] = absolute[band][:, c] / total[:, c]

        # Engagement index: beta / (alpha + theta)
        cols[f"{ch}_engagement"] = absolute["beta"][:, c] / (absolute["alpha"][:, c] + absolute["theta"][:, c])
    return pd.DataFrame(cols)


def zscore_within_session(features: pd.DataFrame, session_ids: pd.Series) -> pd.DataFrame:
    """Z-score every column using only that session's rows."""
    # Group by session, never look at labels
    by_session = features.groupby(np.asarray(session_ids))

    # Subtract session mean, divide by session std
    mean = by_session.transform("mean")
    std = by_session.transform("std", ddof=0).replace(0, 1.0)
    return (features - mean) / std


def webcam_features(frames: pd.DataFrame, videos: pd.DataFrame, window_s: float = 3.0) -> pd.DataFrame:
    """Per-video face features from the per-frame webcam table.

    Uses the same [start, start + window_s) window as the EEG.
    """
    rows = []
    for start in videos["start_time"]:
        # Frames inside this video's window
        w = frames[(frames["t"] >= start) & (frames["t"] < start + window_s)]
        face = w[w["face"] == 1]
        if face.empty:
            rows.append({"face_present": 0.0, "blink_rate": np.nan, "head_motion": np.nan})
            continue

        # Blink = eyes-closed score rising past 0.5
        closed = (face["blink"] > 0.5).astype(int)
        blinks = int((closed.diff() == 1).sum())

        # Head motion: average angle change per frame
        motion = (face["yaw"].diff().abs() + face["pitch"].diff().abs()).mean()
        rows.append(
            {"face_present": len(face) / max(len(w), 1), "blink_rate": blinks / window_s, "head_motion": motion}
        )
    return pd.DataFrame(rows)
