"""Record one session: browser, EEG and webcam together."""

from datetime import datetime
from pathlib import Path

from eegscroll.browser import REELS_URL, ReelWatcher
from eegscroll.clock import now
from eegscroll.data import CHANNELS, SESSIONS_DIR, SFREQ, Recording, Session, save_session
from eegscroll.eeg_sources import make_source
from eegscroll.webcam import WebcamRecorder


def record_session(
    eeg_kind: str = "sim",
    use_webcam: bool = True,
    url: str = REELS_URL,
    root: Path = SESSIONS_DIR,
    notes: str = "",
    scroll_after: float | None = None,
    max_reels: int | None = None,
) -> Path:
    """Run until q / window closed / max_reels; save; return folder.

    scroll_after: seconds before the program scrolls on by itself
    (timer stand-in for the EEG model). None = the viewer scrolls.
    """
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Timer policy now; the EEG model replaces this later
    decide = (lambda reel, watched: watched >= scroll_after) if scroll_after else None

    # Start background recorders before the browser
    eeg = make_source(eeg_kind)
    cam = WebcamRecorder() if use_webcam else None
    watcher = ReelWatcher(decide, max_reels)
    t_start = now()
    eeg.start()
    if cam:
        cam.start()

    # Browser runs here until the session ends
    print(f"Session {session_id}: scroll reels, press q in the browser to stop.")
    try:
        watcher.run(url)
    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")
    finally:
        # Stop everything, even after errors
        t_end = now()
        eeg.stop()
        if cam:
            cam.stop()

    # Gather everything into one Session
    eeg_data, eeg_times = eeg.to_arrays()
    keys = watcher.keys_frame()
    frames = cam.to_frame() if cam and not cam.error else None
    meta = {
        "session_id": session_id,
        "eeg_source": eeg.name,
        "sfreq": SFREQ,
        "channels": CHANNELS,
        "url": url,
        "start": t_start,
        "end": t_end,
        "webcam": frames is not None,
        "scroll_after": scroll_after,
        "notes": notes,
    }
    session = Session(
        Recording(session_id, eeg_data, eeg_times),
        watcher.videos_frame(),
        frames,
        keys,
        meta,
        watcher.auto_scrolls_frame(),
    )

    # Save to data/sessions/<id>/
    out = root / session_id
    save_session(session, out)
    n_frames = 0 if frames is None else len(frames)
    print(
        f"Saved {out}\n  {len(watcher.videos)} reels, {len(keys)} key presses, "
        f"{len(watcher.auto_scrolls)} auto scrolls, {eeg_times.size} EEG samples, {n_frames} webcam frames"
    )
    return out
