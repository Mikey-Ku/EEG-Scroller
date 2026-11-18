"""Evaluate and retrain on every saved session in data/sessions/."""

import argparse

from eegscroll.train import retrain


def main() -> None:
    # Same settings record.py uses by default
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-only", action="store_true", help="skip simulated-EEG sessions")
    parser.add_argument("--window", type=float, default=3.0, help="seconds from reel start")
    parser.add_argument("--skip-fraction", type=float, default=0.9, help="watched below this = skip")
    args = parser.parse_args()
    retrain(real_only=args.real_only, window_s=args.window, skip_fraction=args.skip_fraction)


if __name__ == "__main__":
    main()
