"""Instagram Reels in a controlled browser; reports keys and reels."""

import threading
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from eegscroll.clock import now, unix_to_clock_offset

REELS_URL = "https://www.instagram.com/reels/"

# Saved login, so you sign in only once
PROFILE_DIR = Path(__file__).resolve().parents[2] / ".browser_profile"

# Runs inside every page; reports to Python
WATCHER_JS = r"""
(() => {
  // Only the top page, only once
  if (window.top !== window || window.__eegWatcher) return;
  window.__eegWatcher = true;

  // Precise Unix time in seconds
  const now = () => (performance.timeOrigin + performance.now()) / 1000;

  // Log key presses, ignoring typing in text boxes
  document.addEventListener('keydown', (e) => {
    const el = e.target;
    const tag = (el.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || el.isContentEditable) return;
    window.eegReport({ type: 'key', t: now(), key: e.key });
  }, true);

  // How much of a video is on screen
  const visibleArea = (v) => {
    const r = v.getBoundingClientRect();
    const w = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
    const h = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
    return w * h;
  };

  // Most visible playing video, or null
  const playingVideo = () => {
    const playing = [...document.querySelectorAll('video')]
      .filter((v) => !v.paused && !v.ended && v.readyState > 2);
    playing.sort((a, b) => visibleArea(b) - visibleArea(a));
    return playing[0] || null;
  };

  // Video element identity: stable id plus its source
  let nextId = 0;
  const elemKey = (v) => {
    if (!v.dataset.eegId) v.dataset.eegId = String(nextId++);
    return v.dataset.eegId + '|' + v.currentSrc;
  };

  // Reel identity: Instagram's reel code in the URL, else the video element
  const reelKey = (v) => {
    const m = location.pathname.match(/^\/reels?\/([^/]+)\/?$/);
    return m ? m[1] : (v ? elemKey(v) : null);
  };

  // Best effort caption: longest text on screen right now
  const onScreen = (el) => {
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight;
  };
  const caption = () => {
    const texts = [...document.querySelectorAll('span, h1')]
      .filter(onScreen).map((s) => s.innerText.trim()).filter((t) => t.length > 12 && t.length < 500);
    return texts.sort((a, b) => b.length - a.length)[0] || '';
  };

  // Report a reel start; duration null if its video never played
  let currentKey = null, pending = null, lastElem = null;
  const flush = (duration) => {
    window.eegReport({ type: 'video', t: pending.t, id: pending.id, duration, caption: caption() });
    pending = null;
  };

  // Every 100 ms: which reel is on screen now?
  setInterval(() => {
    const v = playingVideo();
    const key = reelKey(v);
    if (key === null) return;

    // New reel: close the old one, remember when this one began
    if (key !== currentKey) {
      if (pending) flush(null);
      currentKey = key;
      pending = { t: now(), id: key, prev: lastElem };
    }
    if (v) lastElem = elemKey(v);
    if (!pending) return;

    // Length must come from this reel's own video, not the last one still playing
    if (v && elemKey(v) !== pending.prev && isFinite(v.duration) && v.duration > 0) {
      flush(v.duration);
    } else if (now() - pending.t > 3) {
      flush(null);
    }
  }, 100);
})();
"""


class ReelWatcher:
    """Opens Reels, collects key presses and reel starts.

    decide(reel, watched_s) -> True makes the watcher scroll on;
    this is where the EEG model plugs in later. None = scroll by hand.
    """

    def __init__(
        self,
        decide: Callable[[dict, float], bool] | None = None,
        max_reels: int | None = None,
    ) -> None:
        self.keys: list[dict] = []
        self.videos: list[dict] = []
        self.auto_scrolls: list[dict] = []
        self.decide = decide
        self.max_reels = max_reels
        self.stop = threading.Event()
        self._offset = unix_to_clock_offset()
        self._left: dict | None = None

    def handle(self, event: dict) -> None:
        """One event from WATCHER_JS (Playwright here, or the app's embedded view)."""
        # Browser Unix time -> shared clock time
        t = event["t"] + self._offset
        if event["type"] == "key":
            self.keys.append({"t": t, "key": event["key"]})
            if event["key"] in ("q", "Q"):
                self.stop.set()
        elif event["type"] == "video":
            # Duration is None when the reel's video never played (ad, photo, stall)
            duration = event["duration"]
            self.videos.append(
                {"t": t, "video_id": event["id"], "duration": duration, "caption": event.get("caption", "")}
            )
            print(f"  reel {len(self.videos)}: " + (f"{duration:.1f} s" if duration is not None else "no video played"))

            # Enough reels scrolled away: end the session
            if self.max_reels and len(self.videos) > self.max_reels:
                self.stop.set()

    def _maybe_auto_scroll(self, page) -> None:
        """Ask decide() whether to leave the current reel."""
        if self.decide is None or not self.videos:
            return
        reel = self.videos[-1]
        if reel is self._left:
            return  # already scrolled; waiting for the next reel to appear

        # Scroll with a real key press so the page logs it like a human one
        watched = now() - reel["t"]
        if self.decide(reel, watched):
            page.keyboard.press("ArrowDown")
            self._left = reel
            self.auto_scrolls.append({"t": now(), "video_id": reel["video_id"], "watched": watched})

    def run(self, url: str = REELS_URL) -> None:
        """Block until q is pressed or the window closes."""
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            # Use installed Chrome with a saved profile
            context = p.chromium.launch_persistent_context(
                str(PROFILE_DIR),
                channel="chrome",
                headless=False,
                viewport=None,
                args=["--autoplay-policy=no-user-gesture-required"],
            )

            # Connect page script to Python, then open Reels
            context.expose_function("eegReport", self.handle)
            context.add_init_script(WATCHER_JS)
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(url)

            # Short waits let Playwright deliver page events
            try:
                while not self.stop.is_set() and not page.is_closed():
                    page.wait_for_timeout(100)
                    self._maybe_auto_scroll(page)
            except PlaywrightError:
                pass  # window closed by the user
            finally:
                try:
                    context.close()
                except PlaywrightError:
                    pass

    def keys_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.keys, columns=["t", "key"])

    def videos_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.videos, columns=["t", "video_id", "duration", "caption"])

    def auto_scrolls_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.auto_scrolls, columns=["t", "video_id", "watched"])
