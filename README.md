# EEG-Scroller
**Goal:** predict whether you will *skip* or *keep watching* an Instagram Reel from the
**first second of EEG** after it starts, recorded with a Muse 2 headband.

The pipeline records EEG, reel timing and key presses while you scroll Reels normally,
labels each reel skip/watch from how long you stayed, cuts the EEG window at reel start,
turns it into band power, and tests whether a simple model beats guessing.
Webcam face data is a side signal: it is there to check that an EEG result is not just blinks.

## Pipeline
```
record (session) → save (data) → label (labels) → cut windows (windows)
→ band power (features) → model (models) → leave-one-session-out test (evaluate) → retrain + save (train)
```

## Setup
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```
Needs Google Chrome installed. The first webcam run downloads a ~4 MB face model into `models/`.

## Check the pipeline (no Muse, no Instagram login)
```bash
python scripts/check_pipeline.py   # Chrome opens a fake Reels page, records ~1 min with simulated EEG, retrains
python scripts/run_synthetic.py    # fake sessions with a known alpha effect; models should score well above 0.5
```
`check_pipeline.py` should end with `12 usable reels, 6 skips` and a saved model.

## The app (easiest way to record, macOS)
```bash
python scripts/app.py          # real Muse
python scripts/app.py --sim    # simulated EEG, to try it without the headset
```
One window holds everything; leave it open across sessions:
- **Headset:** starts `muselsl stream --lsltime` itself (or reuses a stream already running).
  A light per sensor shows signal size: green under 50 uV, yellow under 100, red above, plus a hint
  when there's 60 Hz hum or poor contact.
- **Live EEG:** the last 5 s of each channel, filtered 1-40 Hz.
- **Webcam:** preview, face found, live blink score; warns if the camera stops.
- **Session:** name + notes, Start/Stop (or `q` in Reels). Saved as `data/sessions/<date-time>_<name>/`.
  After Stop it shows a quality report and asks: **Keep and retrain**, **Keep**, or **Delete**.
- **Reels:** embedded on the right using Safari's engine (log in once; it's remembered).
  Click the reels before scrolling with the arrow keys.

Check any saved session from the terminal: `python -m eegscroll.quality [session folder]` (default: newest).

## Record a real session
```bash
python scripts/record.py              # you scroll; simulated EEG + webcam
python scripts/record.py --eeg muse   # real Muse: run `muselsl stream` first
python scripts/record.py --no-webcam --notes "Mikey, library"
```
1. A Chrome window opens on instagram.com/reels. First time only: log in (saved in `.browser_profile/`).
2. Scroll normally (arrow keys, wheel or trackpad all work). The terminal prints each reel's length.
3. Press `q` in the browser (or close the window) to end.
4. The session is saved to `data/sessions/<date-time>/`, then the model retrains on **all** sessions
   and is saved to `models/latest.joblib`.

### With the Muse 2
In one terminal, keep the headset streaming (`--lsltime` is required; without it EEG
timestamps won't line up with the reels):
```bash
muselsl stream --lsltime
```
Then in another terminal, for real sessions:
```bash
python scripts/record.py --eeg muse --real-only --notes "..."
```

## Auto-scroller
```bash
python scripts/record.py --auto-scroll 5 --max-reels 20 --no-webcam   # program scrolls every 5 s, stops after 20 reels
```
The program presses ArrowDown itself. Right now the rule is a timer; later the EEG model decides.
The seam is `ReelWatcher(decide=...)` in `browser.py`: `decide(reel, seconds_watched) -> True` means scroll now.
Reels the program scrolled away are listed in `auto_scrolls.csv` and **never used as training labels**
(they were not the viewer's choice).

macOS will ask for camera permission for your terminal/VS Code the first time.

```bash
python scripts/run_session.py    # retrain on saved sessions without recording (--real-only to skip simulated EEG)
```

## Session folder
| File | Contents |
|---|---|
| `eeg.csv` | `t, TP9, AF7, AF8, TP10` – one row per sample at 256 Hz |
| `videos.csv` | `t, video_id, duration, caption` – one row per reel; `video_id` is Instagram's reel code, `duration` is empty if its video never played (ad, photo, stall), `caption` is best-effort on-screen text |
| `auto_scrolls.csv` | `t, video_id, watched` – scrolls made by the program (only in `--auto-scroll` sessions) |
| `keys.csv` | `t, key` – every key press |
| `face.csv` | `t, face, blink, yaw, pitch` – per webcam frame (if webcam on) |
| `meta.json` | session id, EEG source (`simulated`/`muse`), notes |

All `t` values use the same LSL clock, so streams line up.

## Modules (`src/eegscroll/`)
| File | What it does |
|---|---|
| `clock.py` | Shared clock (LSL) for every stream |
| `browser.py` | Opens Reels in Chrome; a page script reports key presses and each reel's start, length and caption, keyed by the reel URL. Holds the `decide` hook that scrolls for you |
| `eeg_sources.py` | `SimulatedMuse` (stand-in until the headset arrives) and `MuseLSL` (real headset) |
| `webcam.py` | MediaPipe face tracking: eyes-closed score and head angles per frame |
| `session.py` | Runs browser + EEG + webcam together and saves the session |
| `data.py` | Save/load session folders |
| `labels.py` | Reel starts become one row per reel with a skip/watch label |
| `windows.py` | Cuts the first N seconds of EEG from each reel (never the pre-scroll seconds) |
| `features.py` | Welch band power (theta/alpha/beta), engagement index, per-session z-score; blink rate and head motion from the webcam |
| `models.py` | Dummy baseline, logistic regression, random forest |
| `evaluate.py` | Leave-one-session-out balanced accuracy, permutation test, feature importances |
| `train.py` | Builds the dataset from all sessions, picks the best model, saves it |
| `synthetic.py` | Fake sessions with a known effect, for checking the pipeline |

### Labeling rules
- A reel ends when the next reel appears (URL changes). The last reel of a session is dropped (it wasn't scrolled away).
- Reels whose video never played (ads, photos, stalls) are dropped, but they still end the reel before them.
- Each reel's EEG window is its **first 1 s** (`WINDOW_S` in `labels.py`, `--window`).
- **Watch** if you stayed at least **5 s** (`STAY_S`, `--stay`); **skip** if you left sooner.
  A reel shorter than that counts as watched if you finished 90% of it.
- Reels left within window + **0.5 s** (`GAP_S`, `--gap`), i.e. under 1.5 s, are dropped: the half second
  before a key press contains planning the press, which would leak the answer.
- Reels shorter than the window are dropped.

The windows → band power → classifier structure follows MNE's
[sleep staging tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html), using `scipy.signal.welch` instead of MNE.
