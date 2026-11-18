# EEG Scroller To-do

Goal: skip vs. watch on Instagram Reels from the first 3 s of Muse 2 EEG.

## Understanding

- [ ] **The loop:** Understand the path from sensors to windowed features, prediction, and scroll. Know offline replay vs. a real-time closed loop and where delay comes from.
- [ ] **Syncing:** Understand why each device has its own clock, how LSL timestamps align streams, and what clock drift is.
- [ ] **Labels:** Understand why a skip is only a stand-in for engagement and how a key press can leak into the data.
- [ ] **EEG basics:** Learn what EEG measures, where the four Muse sensors sit, and what delta, theta, alpha, beta, and gamma are linked to.
- [ ] **EEG noise:** Recognize how blinks, eye movements, jaw clenches, head movement, and poor contact look in the raw signal.
- [ ] **Band power:** Understand how an FFT or Welch's method estimates power per frequency and how window length changes the result.
- [ ] **Webcam features:** Learn how face landmarks and blendshapes work and how blinks and head movement are measured.
- [ ] **Classifiers:** Understand how logistic regression and random forests decide and what their outputs mean.
- [ ] **Honest testing:** Understand baselines, balanced accuracy, session-based splits, and overfitting with a few hundred examples.

## Setup

- [x] Create the shared GitHub repo.
- [x] Set up the Python environment and pin dependencies in `requirements.txt` (pylsl, playwright, opencv-python, mediapipe, scipy, scikit-learn).
- [x] Create folders for raw recordings and sessions (`data/`); each session gets its own folder with `meta.json` notes.
- [ ] Confirm the webcam, browser, and LSL tools work on both laptops.

## Recording tools

- [x] Write the scroll logger: timestamp every key press on Instagram Reels web.
- [x] Write a browser script that logs each reel's start and length.
- [x] Write the webcam recorder: face features plus a timestamp for every frame.
- [x] Check the whole recording path without a Muse or login: `python scripts/check_pipeline.py`.
- [x] Test on real instagram.com/reels with a login: reels detected, lengths right, `q` ends the session.
- [x] Auto-scroller: program presses ArrowDown on a timer (`--auto-scroll`), logs it, and those reels are excluded from labels.
- [ ] Log the reel author reliably (caption is best-effort now; the reel code can be looked up later).
- [ ] **11/20:** Record the first labeled webcam session.

## EEG

- [ ] Install `muselsl` and confirm it runs on the Mac (`muselsl list`), before the headset arrives.
- [ ] Run the band-power code on a practice Muse dataset and look at the spectrum (1/f slope, alpha bump, 60 Hz line). Needs a small CSV loader for that dataset.
- [ ] Get the Muse 2 streaming into LSL with `muselsl stream`, then `python scripts/record.py --eeg muse`.
- [ ] Do a few hard blinks and confirm they line up in the EEG and the webcam rows.

## Data collection

- [ ] Record sessions until there are a few hundred labeled reels.
- [ ] Log every session: date, who, and notes (`--notes`).

## Processing

- [x] Turn reel starts and lengths into skip/watch labels (cutoff 0.9; tune on real data).
- [x] Cut a 3 s window from the start of each reel (tune the length on real data).
- [x] Compute EEG features: band power per channel and the engagement index.
- [x] Compute webcam features: blink rate and head movement.
- [ ] Add a high-pass filter and drop windows with blink/motion artifacts.
- [ ] Add a margin so the window ends well before the key press (motor planning leak).

## Modeling

- [x] Set up a baseline that always guesses the most common label.
- [x] Set up logistic regression and random forest with leave-one-session-out testing.
- [ ] Train on EEG only, webcam only, and both together; report balanced accuracy against the baseline.
- [ ] Check whether EEG adds anything beyond the webcam blink features.

## Demo

- [ ] Write a replay script: load a saved session + `models/latest.joblib`, predict per reel, print predicted vs. real scroll.
- [ ] Plug the model into `decide()` so the EEG window at 3 s decides the scroll (replaces the timer).
- [ ] **12/8:** Run the best model over a recorded session and show its scrolls next to the real ones.

## Final deliverables

- [ ] Write a README that lets a classmate rerun the full pipeline on our data.
- [ ] Write a short summary of what we did and found.
- [ ] Make about five minutes of slides plus the demo, and submit them on Canvas.

## Stretch

- [ ] Build a "learning mode" that slows scrolling on educational reels.
