"""Record a session on Instagram Reels, then retrain.

Run: python scripts/record.py                          (you scroll; simulated EEG + webcam)
     python scripts/record.py --auto-scroll 5 --max-reels 20   (program scrolls every 5 s, 20 reels)
     python scripts/record.py --eeg muse               (after `muselsl stream`)
"""

import argparse

from eegscroll.session import record_session
from eegscroll.train import retrain


def main() -> None:
    # Command-line options
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg", choices=["sim", "muse"], default="sim", help="EEG input")
    parser.add_argument("--no-webcam", action="store_true", help="skip face tracking")
    parser.add_argument("--no-retrain", action="store_true", help="only record")
    parser.add_argument("--real-only", action="store_true", help="train on real-Muse sessions only")
    parser.add_argument("--notes", default="", help="who, where, anything odd")
    parser.add_argument("--auto-scroll", type=float, metavar="SEC", help="program scrolls after SEC seconds (timer stand-in for the model)")
    parser.add_argument("--max-reels", type=int, metavar="N", help="stop after N reels")
    args = parser.parse_args()

    # Record until q, window closed, or max reels
    record_session(
        args.eeg,
        use_webcam=not args.no_webcam,
        notes=args.notes,
        scroll_after=args.auto_scroll,
        max_reels=args.max_reels,
    )

    # Retrain on every saved session
    if not args.no_retrain:
        print("\nRetraining on all sessions...")
        retrain(real_only=args.real_only)


if __name__ == "__main__":
    main()
