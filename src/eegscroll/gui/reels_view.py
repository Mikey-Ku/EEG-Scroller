"""Instagram Reels embedded in the app, using Safari's engine (WKWebView).

Qt's own browser engine can't play H.264/AAC, which Instagram mostly
serves, so this wraps macOS's WKWebView in a Qt widget instead.
The same WATCHER_JS as the Playwright recorder reports reels and keys.
"""

import json
from collections.abc import Callable

import objc
from AppKit import NSApplication, NSView
from Foundation import NSURL, NSURLRequest, NSUUID, NSMakeRect, NSObject, NSProcessInfo

# WebKit keeps cookies under ~/Library/WebKit/<process name>/, and that name is
# "python" or "python3" depending on how the app was started. Fix it, so the
# Instagram login is found every time. Must happen before WebKit starts.
APP_NAME = "EEGScroller"
NSProcessInfo.processInfo().setProcessName_(APP_NAME)
from PySide6.QtGui import QWindow
from PySide6.QtWidgets import QVBoxLayout, QWidget
from WebKit import (
    WKUserScript,
    WKWebView,
    WKWebViewConfiguration,
    WKWebsiteDataStore,
)

from eegscroll.browser import WATCHER_JS

# Fixed id = the same saved login every time the app opens
DATA_STORE_ID = "5E7C2A1F-3B8D-4E6A-9C0F-EE65C0C0FFEE"

# WATCHER_JS calls window.eegReport(event); send it to Python as JSON
BRIDGE_JS = "window.eegReport = (e) => window.webkit.messageHandlers.eeg.postMessage(JSON.stringify(e));"

# Look like Safari so Instagram serves the normal site
SAFARI_NAME = "Version/18.0 Safari/605.1.15"

_INJECT_AT_DOCUMENT_END = 1
_NO_USER_ACTION_NEEDED = 0  # autoplay video and audio


class _MessageHandler(NSObject, protocols=[objc.protocolNamed("WKScriptMessageHandler")]):
    """Receives window.webkit.messageHandlers.eeg.postMessage(...) calls."""

    def initWithCallback_(self, callback):
        self = objc.super(_MessageHandler, self).init()
        if self is not None:
            self._callback = callback
        return self

    def userContentController_didReceiveScriptMessage_(self, controller, message):
        try:
            self._callback(json.loads(str(message.body())))
        except Exception as exc:  # never let a bad event crash the app
            print(f"Reels event error: {exc}")


class ReelsView(QWidget):
    """Qt widget showing a WKWebView; on_event gets WATCHER_JS events."""

    def __init__(self, on_event: Callable[[dict], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        config = WKWebViewConfiguration.alloc().init()
        config.setMediaTypesRequiringUserActionForPlayback_(_NO_USER_ACTION_NEEDED)
        config.setApplicationNameForUserAgent_(SAFARI_NAME)

        # Persistent cookies under a fixed id (macOS 14+), else the default store
        if hasattr(WKWebsiteDataStore, "dataStoreForIdentifier_"):
            store_id = NSUUID.alloc().initWithUUIDString_(DATA_STORE_ID)
            config.setWebsiteDataStore_(WKWebsiteDataStore.dataStoreForIdentifier_(store_id))

        # Bridge first, then the watcher, at the end of every page load
        controller = config.userContentController()
        self._handler = _MessageHandler.alloc().initWithCallback_(on_event)
        controller.addScriptMessageHandler_name_(self._handler, "eeg")
        script = WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
            BRIDGE_JS + WATCHER_JS, _INJECT_AT_DOCUMENT_END, True
        )
        controller.addUserScript_(script)

        self.web = WKWebView.alloc().initWithFrame_configuration_(NSMakeRect(0, 0, 800, 900), config)
        container = QWidget.createWindowContainer(QWindow.fromWinId(objc.pyobjc_id(self.web)), self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)

    def load(self, url: str) -> None:
        if url.startswith("file://"):
            # Local test pages need read access to their folder
            page = NSURL.URLWithString_(url)
            self.web.loadFileURL_allowingReadAccessToURL_(page, page.URLByDeletingLastPathComponent())
        else:
            self.web.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(url)))

    def reload(self) -> None:
        self.web.reload()

    def focus_page(self) -> None:
        """Give the page keyboard focus so arrow keys scroll reels."""
        window = self.web.window()
        if window is not None:
            window.makeFirstResponder_(self.web)

    def has_focus(self) -> bool:
        """True when the page (or a box in it) has keyboard focus."""
        window = self.web.window()
        responder = window.firstResponder() if window is not None else None
        return bool(responder is not None and responder.isKindOfClass_(NSView) and responder.isDescendantOf_(self.web))

    def send_edit(self, action: str) -> None:
        """Send an edit command (paste:, copy:, ...) to whatever has focus in the page."""
        NSApplication.sharedApplication().sendAction_to_from_(action, None, None)

    def run_js(self, code: str, done: Callable[[object], None] | None = None) -> None:
        def finished(result, _error) -> None:  # WebKit requires a None return
            if done:
                done(result)

        self.web.evaluateJavaScript_completionHandler_(code, finished)
