"""Eyeball one session: EEG with reels marked, plus skip vs. watch band power.

Run: python scripts/plot_session.py                 (newest session)
     python scripts/plot_session.py data/sessions/<id>
Saves <session>/plot.png and opens it.
"""

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from eegscroll.data import CHANNELS, SESSIONS_DIR, SFREQ, load_session
from eegscroll.labels import WINDOW_S
from eegscroll.train import session_features

# Okabe-Ito colors, readable with common color blindness
SKIP, WATCH = "#E69F00", "#0072B2"


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else max(SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
    s = load_session(folder)
    x, t = s.recording.eeg, s.recording.eeg_times
    t0 = s.meta["start"]
    y = sosfiltfilt(butter(4, [1, 40], btype="band", fs=SFREQ, output="sos"), x, axis=1)
    reels, feats = session_features(s)

    fig, axes = plt.subplots(len(CHANNELS) + 2, 1, figsize=(16, 12), sharex=False,
                             gridspec_kw={"height_ratios": [1] * len(CHANNELS) + [0.6, 1.4]})

    # EEG traces, reels shaded: orange = skipped, blue = watched
    for i, (ax, ch) in enumerate(zip(axes, CHANNELS)):
        ax.plot(t - t0, y[i], lw=0.4, color="k")
        ax.set_ylabel(f"{ch}\nµV"); ax.set_ylim(-150, 150)
        for _, r in reels.iterrows():
            ax.axvspan(r.start_time - t0, r.end_time - t0, color=SKIP if r.label == 1 else WATCH, alpha=0.18)
            ax.axvspan(r.start_time - t0, r.start_time - t0 + WINDOW_S, facecolor="none", edgecolor="0.3", hatch="///", lw=0)
    # Blink score from the webcam
    ax = axes[len(CHANNELS)]
    if s.frames is not None and "blink" in s.frames:
        ax.plot(s.frames.t - t0, s.frames.blink, lw=0.6, color="k"); ax.axhline(0.3, ls="--", color="gray")
    ax.set_ylabel("blink"); ax.set_xlabel("seconds into session")
    for a in axes[: len(CHANNELS) + 1]:
        a.set_xlim(0, t[-1] - t0)

    # First-3-s features per reel, skipped vs. watched
    ax = axes[-1]
    cols = [c for c in feats.columns if c.endswith(("alpha_rel", "theta_rel", "beta_rel", "engagement"))]
    for j, c in enumerate(cols):
        for lab, color, dx in ((1, SKIP, -0.15), (0, WATCH, 0.15)):
            v = feats.loc[reels.label == lab, c]
            ax.scatter(np.full(len(v), j + dx), v, color=color, s=14)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("feature value")
    ax.set_title(f"First 3 s of each reel: orange = skipped ({(reels.label == 1).sum()}), blue = watched ({(reels.label == 0).sum()})")

    axes[0].set_title(f"{folder.name}  —  orange = skipped reel, blue = watched, hatched = first 3 s (what the model sees)")
    fig.tight_layout()
    out = folder / "plot.png"
    fig.savefig(out, dpi=110)
    print(out)
    subprocess.run(["open", str(out)])


if __name__ == "__main__":
    main()
