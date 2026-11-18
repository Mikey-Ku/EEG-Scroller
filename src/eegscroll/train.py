"""Turn sessions into a dataset, evaluate, and retrain."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from eegscroll.data import CHANNELS, SESSIONS_DIR, SFREQ, Session, load_all_sessions
from eegscroll.evaluate import evaluate_models, permutation_p_value, print_feature_importance
from eegscroll.features import band_power, webcam_features, zscore_within_session
from eegscroll.labels import boundaries, build_video_table
from eegscroll.models import get_models
from eegscroll.windows import covered_mask, cut_windows

# Where the retrained model is saved
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "latest.joblib"


def session_features(
    session: Session, window_s: float, skip_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One session -> (video table, raw feature table)."""
    rec = session.recording

    # Reel starts -> labeled videos
    starts, ends = boundaries(session.videos["t"].to_numpy())
    durations = session.videos["duration"].to_numpy()[:-1]

    # Drop reels the program scrolled away; only the viewer's choices are labels
    auto_ids = set(session.auto_scrolls["video_id"].astype(str))
    human = ~session.videos["video_id"].astype(str).isin(auto_ids).to_numpy()[:-1]
    videos = build_video_table(
        rec.session_id, starts[human], ends[human], durations[human], window_s, skip_fraction
    )

    # Keep videos with full EEG coverage
    videos = videos[covered_mask(rec.eeg_times, videos, window_s, SFREQ)].reset_index(drop=True)
    if videos.empty:
        return videos, pd.DataFrame()

    # EEG band power per video window
    windows = cut_windows(rec.eeg, rec.eeg_times, videos, window_s, SFREQ)
    X = band_power(windows, SFREQ, CHANNELS)

    # Add face features when the webcam ran
    if session.frames is not None:
        X = pd.concat([X, webcam_features(session.frames, videos, window_s)], axis=1)
    return videos, X


def build_dataset(
    sessions: list[Session], window_s: float = 3.0, skip_fraction: float = 0.9
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """All sessions -> X, y, groups, videos."""
    parts = [session_features(s, window_s, skip_fraction) for s in sessions]
    parts = [(v, X) for v, X in parts if not v.empty]
    if not parts:
        return pd.DataFrame(), np.array([]), np.array([]), pd.DataFrame()

    # Only keep columns every session has
    videos = pd.concat([v for v, _ in parts], ignore_index=True)
    X = pd.concat([X for _, X in parts], ignore_index=True, join="inner")

    # Per-session z-score; missing values -> session average
    X = zscore_within_session(X, videos["session_id"]).fillna(0.0)
    return X, videos["label"].to_numpy(), videos["session_id"].to_numpy(), videos


def evaluate_and_pick(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_permutations: int = 100
) -> str:
    """Print evaluation report; return the best model's name."""
    models = get_models()

    # Leave-one-session-out needs at least two sessions
    if len(np.unique(groups)) < 2:
        print("Only one session: skipping evaluation, defaulting to random_forest.")
        return "random_forest"

    # Per-session and mean balanced accuracy
    results = evaluate_models(models, X, y, groups)
    print("\nBalanced accuracy per held-out session:")
    print(results.pivot(index="held_out_session", columns="model", values="balanced_accuracy").round(3))
    means = results.groupby("model")["balanced_accuracy"].mean()
    print("\nMean balanced accuracy:")
    print(means.round(3).to_string())

    # Best real model; baseline is only a reference
    best = means.drop("baseline").idxmax()
    p = permutation_p_value(models[best], X, y, groups, n_permutations)
    print(f"\nBest model: {best}  (permutation p = {p:.3f})")
    if means[best] <= means["baseline"]:
        print("Warning: best model does not beat the baseline.")

    # What the models relied on
    print_feature_importance(models, X, y)
    return best


def retrain(
    root: Path = SESSIONS_DIR,
    real_only: bool = False,
    window_s: float = 3.0,
    skip_fraction: float = 0.9,
    model_path: Path = MODEL_PATH,
) -> Path | None:
    """Rebuild from every saved session and save the best model."""
    sessions = load_all_sessions(root, real_only)

    # Warn when training on fake EEG
    fake = [s.meta["session_id"] for s in sessions if s.meta.get("eeg_source") != "muse"]
    if fake:
        print(f"Note: {len(fake)} session(s) use simulated EEG: {', '.join(fake)}")

    X, y, groups, videos = build_dataset(sessions, window_s, skip_fraction)
    print(f"{len(sessions)} sessions, {len(videos)} usable reels, {int((y == 1).sum())} skips")

    # Need enough reels and both labels
    if len(videos) < 10 or len(np.unique(y)) < 2:
        print("Not enough labeled reels yet to train.")
        return None

    best = evaluate_and_pick(X, y, groups)

    # Final model sees all data
    model = clone(get_models()[best]).fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "name": best,
            "features": list(X.columns),
            "window_s": window_s,
            "skip_fraction": skip_fraction,
            "sessions": sorted(set(groups)),
        },
        model_path,
    )
    print(f"Saved {best} to {model_path}")
    return model_path
