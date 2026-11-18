"""Load and save session folders."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Muse 2 channel order and sampling rate
CHANNELS: list[str] = ["TP9", "AF7", "AF8", "TP10"]
SFREQ: float = 256.0

# Where record.py saves session folders
SESSIONS_DIR = Path(__file__).resolve().parents[2] / "data" / "sessions"


# Raw EEG on the shared clock
@dataclass
class Recording:
    session_id: str
    eeg: np.ndarray  # shape (n_channels, n_samples), microvolts
    eeg_times: np.ndarray  # shape (n_samples,), seconds


# Everything recorded in one sitting
@dataclass
class Session:
    recording: Recording
    videos: pd.DataFrame  # one row per reel start: t, video_id, duration
    frames: pd.DataFrame | None = None  # webcam rows; None without camera
    keys: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["t", "key"]))
    meta: dict = field(default_factory=dict)
    auto_scrolls: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["t", "video_id", "watched"]))


def save_session(session: Session, out_dir: Path) -> None:
    """Write one session as CSV files plus meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = session.recording

    # EEG: one row per sample
    eeg = pd.DataFrame(rec.eeg.T, columns=CHANNELS)
    eeg.insert(0, "t", rec.eeg_times)
    eeg.to_csv(out_dir / "eeg.csv", index=False, float_format="%.6f")

    # Browser events and webcam rows
    session.videos.to_csv(out_dir / "videos.csv", index=False)
    session.keys.to_csv(out_dir / "keys.csv", index=False)
    if session.frames is not None:
        session.frames.to_csv(out_dir / "face.csv", index=False, float_format="%.4f")

    # Scrolls made by the program, not the viewer
    if len(session.auto_scrolls):
        session.auto_scrolls.to_csv(out_dir / "auto_scrolls.csv", index=False)

    # Settings and notes for this session
    (out_dir / "meta.json").write_text(json.dumps(session.meta, indent=2))


def load_session(folder: Path) -> Session:
    """Read a folder written by save_session."""
    meta = json.loads((folder / "meta.json").read_text())

    # EEG back to (channels, samples)
    eeg = pd.read_csv(folder / "eeg.csv")
    keys = pd.read_csv(folder / "keys.csv")
    rec = Recording(meta["session_id"], eeg[CHANNELS].to_numpy().T, eeg["t"].to_numpy())

    # Webcam file only exists if camera worked
    face_path = folder / "face.csv"
    frames = pd.read_csv(face_path) if face_path.exists() else None
    videos = pd.read_csv(folder / "videos.csv", dtype={"video_id": str})
    session = Session(rec, videos, frames, keys, meta)

    # Auto-scroll file only exists for auto-scrolled sessions
    auto_path = folder / "auto_scrolls.csv"
    if auto_path.exists():
        session.auto_scrolls = pd.read_csv(auto_path, dtype={"video_id": str})
    return session


def load_all_sessions(root: Path = SESSIONS_DIR, real_only: bool = False) -> list[Session]:
    """Every session folder under root, oldest first."""
    sessions = [load_session(p.parent) for p in sorted(root.glob("*/meta.json"))]

    # Optionally skip sessions with fake EEG
    if real_only:
        sessions = [s for s in sessions if s.meta.get("eeg_source") == "muse"]
    return sessions
