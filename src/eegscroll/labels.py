"""Turn reel start times into one labeled row per video."""

import numpy as np
import pandas as pd

# Label values used everywhere in the project
WATCH: int = 0
SKIP: int = 1

# Defaults for every script; change them here
WINDOW_S: float = 1.0  # EEG used per reel: the first second after it starts
GAP_S: float = 0.5  # ignore the last half second before a scroll (planning the key press)
STAY_S: float = 5.0  # stayed this long = WATCH, left sooner = SKIP


def label_video(watch_time: float, video_duration: float, stay_s: float = STAY_S) -> int:
    """SKIP if left before stay_s; reels shorter than that count as watched if finished."""
    # A 4 s reel can't be watched for 5 s; finishing 90% of it counts
    needed = stay_s if not np.isfinite(video_duration) else min(stay_s, 0.9 * video_duration)
    return SKIP if watch_time < needed else WATCH


def boundaries(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive times -> (starts, ends). Last reel has no end."""
    # Each start is ended by the next one
    times = np.asarray(times, dtype=float)
    return times[:-1], times[1:]


def build_video_table(
    session_id: str,
    start_times: np.ndarray,
    end_times: np.ndarray,
    video_durations: np.ndarray,
    window_s: float = WINDOW_S,
    stay_s: float = STAY_S,
    gap_s: float = GAP_S,
) -> pd.DataFrame:
    """One labeled row per video.

    Drops videos shorter than the window, and reels left before
    window_s + gap_s (their window would include planning the scroll,
    or the next reel).
    """
    # Basic timing columns
    table = pd.DataFrame(
        {
            "session_id": session_id,
            "start_time": start_times,
            "end_time": end_times,
            "watch_time": np.asarray(end_times) - np.asarray(start_times),
            "video_duration": video_durations,
        }
    )

    # Apply the labeling rule per video
    table["label"] = [
        label_video(w, d, stay_s) for w, d in zip(table["watch_time"], table["video_duration"])
    ]

    # Window must fit inside the video, and end gap_s before the scroll
    keep = (table["video_duration"] >= window_s) & (table["watch_time"] >= window_s + gap_s)
    return table[keep].reset_index(drop=True)
