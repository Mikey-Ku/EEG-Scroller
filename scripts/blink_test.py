"""Webcam-only blink test: is the blink cutoff in features.py right for you?

Run: python scripts/blink_test.py            (10 blinks)
     python scripts/blink_test.py --blinks 5

Phase 1: look at the screen normally, try not to blink (baseline).
Phase 2: blink firmly the number of times asked.
Then prints the blink scores and how many blinks each cutoff would count.
"""

import argparse
import time

import numpy as np

from eegscroll.webcam import WebcamRecorder

CURRENT_CUTOFF = 0.3  # features.py: blink > 0.3


def count(scores: np.ndarray, cutoff: float) -> int:
    closed = (scores > cutoff).astype(int)
    return int((np.diff(closed) == 1).sum() + closed[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blinks", type=int, default=10, help="blinks to do in phase 2")
    parser.add_argument("--seconds", type=float, default=15, help="length of phase 2")
    args = parser.parse_args()

    cam = WebcamRecorder()
    cam.start()
    print("Starting camera...")
    while not cam.rows and not cam.error:
        time.sleep(0.1)
    if cam.error:
        return

    print("\nPhase 1 (5 s): look at the screen, eyes open, try not to blink.")
    time.sleep(5)
    t_mid = cam.rows[-1]["t"]
    print(f"Phase 2 ({args.seconds:.0f} s): blink firmly {args.blinks} times, a second or so apart. GO")
    time.sleep(args.seconds)
    cam.stop()
    print("Done.\n")

    df = cam.to_frame()
    seen = df[df.face == 1]
    print(f"{len(df)} frames, face found in {100 * len(seen) / max(len(df), 1):.0f}%")
    if seen.empty:
        print("No face found: check lighting and that the camera can see you.")
        return

    base = seen[seen.t <= t_mid].blink.to_numpy()
    test = seen[seen.t > t_mid].blink.to_numpy()
    print(f"Eyes open (phase 1): median {np.median(base):.2f}, max {base.max():.2f}")
    print(f"Blinking  (phase 2): max {test.max():.2f}")

    print(f"\nBlinks counted in phase 2 (you did {args.blinks}):")
    for cutoff in (0.2, 0.3, 0.4, 0.5, 0.6):
        mark = "  <- current" if cutoff == CURRENT_CUTOFF else ""
        print(f"  cutoff {cutoff:.1f}: {count(test, cutoff):2d} blinks, {count(base, cutoff)} false in phase 1{mark}")


if __name__ == "__main__":
    main()
