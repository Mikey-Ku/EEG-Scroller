"""Evaluate and retrain on every saved session in data/sessions/."""

import argparse

from eegscroll.labels import GAP_S, STAY_S, WINDOW_S
from eegscroll.train import retrain


def main() -> None:
    # Same settings record.py uses by default
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-only", action="store_true", help="skip simulated-EEG sessions")
    parser.add_argument("--window", type=float, default=WINDOW_S, help="seconds of EEG from reel start")
    parser.add_argument("--stay", type=float, default=STAY_S, help="stayed this long = watch, sooner = skip")
    parser.add_argument("--gap", type=float, default=GAP_S, help="seconds before a scroll never used")
    args = parser.parse_args()
    retrain(real_only=args.real_only, window_s=args.window, stay_s=args.stay, gap_s=args.gap)


if __name__ == "__main__":
    main()
