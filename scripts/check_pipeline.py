"""Prove the recording pipeline works without a Muse or an Instagram login.

Opens a self-driving fake Reels page in Chrome, records it with simulated
EEG, saves a session to a temp folder, and retrains on it (about 1 minute).
Run: python scripts/check_pipeline.py
"""

import tempfile
from pathlib import Path

import eegscroll.browser as browser
from eegscroll.session import record_session
from eegscroll.train import retrain

# Fake page and its two clips live next to this script
FAKE_PAGE = Path(__file__).resolve().parent / "fake_reels" / "reels.html"


def main() -> None:
    # Everything goes to a temp folder, never data/sessions
    tmp = Path(tempfile.mkdtemp(prefix="eegscroll_check_"))
    browser.PROFILE_DIR = tmp / "profile"

    # Page plays 14 fake reels, then presses q itself
    out = record_session("sim", use_webcam=False, url=FAKE_PAGE.as_uri(), root=tmp / "sessions", notes="pipeline check")

    # Expect 12 usable reels: 6 skips, 6 watches
    print("\nRetraining on the fake session...")
    retrain(tmp / "sessions", model_path=tmp / "model.joblib")
    print(f"\nPipeline check done. Session folder: {out}")


if __name__ == "__main__":
    main()
