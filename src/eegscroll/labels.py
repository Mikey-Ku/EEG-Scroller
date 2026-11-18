"""Turn reel start times into one labeled row per video."""

import numpy as np
import pandas as pd

# Label values used everywhere in the project
WATCH: int = 0
SKIP: int = 1


def label_video(watch_time: float, video_duration: float, skip_fraction: float = 0.9) -> int:
    """Return SKIP if left before skip_fraction of the video."""
    # Compare watch time to the cutoff
    return SKIP if watch_time < skip_fraction * video_duration else WATCH


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
    window_s: float = 3.0,
    skip_fraction: float = 0.9,
) -> pd.DataFrame:
    """One labeled row per video.

    Drops videos shorter than the window, and reels left before the
    window ends (their window would include the next reel).
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
        label_video(w, d, skip_fraction) for w, d in zip(table["watch_time"], table["video_duration"])
    ]

    # Window must fit inside video and viewing
    keep = (table["video_duration"] >= window_s) & (table["watch_time"] >= window_s)
    return table[keep].reset_index(drop=True)
