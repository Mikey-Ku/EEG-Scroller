"""End-to-end demo on fake data. Run: python scripts/run_synthetic.py"""

from eegscroll.synthetic import make_synthetic_session
from eegscroll.train import build_dataset, evaluate_and_pick

N_SESSIONS = 5


def main() -> None:
    # Fake sessions with a known alpha effect
    sessions = [make_synthetic_session(f"syn{i}", seed=i) for i in range(N_SESSIONS)]

    # Same dataset code as real recordings
    X, y, groups, videos = build_dataset(sessions)
    print(f"{len(videos)} reels, {X.shape[1]} features, {int((y == 1).sum())} skips")

    # Leave-one-session-out report
    evaluate_and_pick(X, y, groups)


if __name__ == "__main__":
    main()
