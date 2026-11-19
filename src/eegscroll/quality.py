"""Quality report for one saved session: is it worth keeping for training?"""

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from eegscroll.data import CHANNELS, SFREQ, load_session
from eegscroll.train import session_features

GOOD_UV, POOR_UV = 50.0, 100.0  # 1-40 Hz signal size per channel
BURST_UV = 150.0  # a second this noisy is movement or loose contact
HUM_RATIO = 0.5  # 60 Hz power vs 1-40 Hz power
BLINK_CUTOFF = 0.3  # same as features.py


def session_report(folder: Path) -> tuple[str, list[str]]:
    """Return (report text, list of problems). No problems = looks good."""
    s = load_session(folder)
    meta, x, t = s.meta, s.recording.eeg, s.recording.eeg_times
    length = meta["end"] - meta["start"]
    lines = [f"Session {folder.name}  ({length:.0f} s, notes: {meta.get('notes') or '-'})"]
    problems: list[str] = []

    # EEG: coverage and timing
    if t.size < SFREQ:
        problems.append("No EEG recorded.")
        lines.append("EEG: none")
    else:
        rate = (t.size - 1) / (t[-1] - t[0])
        gap = float(np.diff(t).max())
        lines.append(f"EEG: {t.size} samples, {rate:.1f} Hz, longest gap {gap * 1000:.0f} ms")
        if rate < 0.95 * SFREQ:
            problems.append(f"EEG dropped samples ({rate:.0f} Hz instead of {SFREQ:.0f}).")
        if gap > 1.0:
            problems.append(f"EEG cut out for {gap:.1f} s.")

        # EEG: signal size per channel, noisy seconds, mains hum
        y = sosfiltfilt(butter(4, [1, 40], btype="band", fs=SFREQ, output="sos"), x, axis=1)
        f, p = welch(x, fs=SFREQ, nperseg=min(512, x.shape[1]))
        hum = p[:, (f >= 58) & (f <= 62)].sum(1) / np.maximum(p[:, (f >= 1) & (f < 40)].sum(1), 1e-12)
        n = y.shape[1] // int(SFREQ) * int(SFREQ)
        per_sec = y[:, :n].reshape(len(CHANNELS), -1, int(SFREQ)).std(2)
        for i, name in enumerate(CHANNELS):
            size = float(np.median(per_sec[i])) if per_sec.size else float(y[i].std())
            mark = "good" if size < GOOD_UV else "high" if size < POOR_UV else "POOR"
            lines.append(f"  {name:5s} {size:5.0f} uV  {mark}" + ("  60 Hz hum" if hum[i] > HUM_RATIO else ""))
            if mark == "POOR":
                problems.append(f"{name} has poor contact ({size:.0f} uV).")
        if (hum > HUM_RATIO).any():
            problems.append("60 Hz mains hum: unplug the charger, move away from power cables.")
        if per_sec.size:
            bursts = int((per_sec.mean(0) > BURST_UV).sum())
            lines.append(f"  noisy seconds (movement/contact): {bursts} of {per_sec.shape[1]}")
            if bursts > 0.2 * per_sec.shape[1]:
                problems.append(f"{bursts} s of large artifacts: sit still, relax face and jaw.")

    # Webcam
    face = s.frames
    if face is None or face.empty:
        lines.append("Webcam: none")
        problems.append("No webcam data.")
    else:
        fps = len(face) / max(length, 1e-9)
        seen = float(face["face"].mean())
        blink = face["blink"].dropna()
        blinks = int(((blink > BLINK_CUTOFF).astype(int).diff() == 1).sum())
        lines.append(f"Webcam: {len(face)} frames ({fps:.0f} fps), face {seen:.0%}, {blinks} blinks")
        if face["t"].iloc[-1] < meta["end"] - 2:
            problems.append(f"Webcam stopped {face['t'].iloc[-1] - meta['start']:.0f} s into the session.")
        if seen < 0.8:
            problems.append(f"Face found in only {seen:.0%} of frames.")

    # Reels and labels, exactly as training will see them
    n_reels = len(s.videos)
    videos, _ = session_features(s)
    if videos.empty:
        lines.append(f"Reels: {n_reels} logged, 0 usable for training")
        problems.append("No reels usable for training.")
    else:
        skips = int((videos["label"] == 1).sum())
        lines.append(
            f"Reels: {n_reels} logged, {len(videos)} usable ({skips} skipped, {len(videos) - skips} watched)"
        )

    return "\n".join(lines), problems


if __name__ == "__main__":
    import sys

    from eegscroll.data import SESSIONS_DIR

    # Default: the newest session
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else max(SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
    text, issues = session_report(target)
    print(text)
    print("\n".join(["", "Problems:"] + [f"- {p}" for p in issues]) if issues else "\nLooks good.")
