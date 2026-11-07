"""Entry point: ``python -m app``.

Loads the bundled fonts and ``theme.qss``, then opens the main window. All
algorithm work happens in ``autofocus``, called from background workers
(``workers.py``); this module only starts the Qt event loop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from . import APP_NAME, APP_VERSION
from .main_window import MainWindow


def _prefer_xcb_on_wayland() -> None:
    """Avoid PySide6's native Wayland backend, which has real rendering bugs.

    Verified directly on this app: with ``QT_QPA_PLATFORM=wayland``, labels
    occasionally render flipped or show stale/overlapping content after a
    pixmap update (e.g. the Focus Search "Best image" card during a live
    run). The same window renders correctly end to end under ``xcb``
    (X11/XWayland). This only acts when the session is actually Wayland, an
    X display is available to fall back to (XWayland), and the user hasn't
    already chosen a platform explicitly via ``QT_QPA_PLATFORM``.
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    if os.environ.get("XDG_SESSION_TYPE") == "wayland" and os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

#: Bundled under app/fonts/ (Inter: SIL OFL 1.1, JetBrains Mono: Apache 2.0)
#: so the design's typography renders the same regardless of what fonts
#: happen to be installed on the machine running the app.
_FONT_FILES = [
    "Inter-Regular.otf",
    "Inter-SemiBold.otf",
    "JetBrainsMono-Regular.ttf",
    "JetBrainsMono-Medium.ttf",
    "JetBrainsMono-SemiBold.ttf",
    "JetBrainsMono-Bold.ttf",
]


def load_stylesheet() -> str:
    return (Path(__file__).resolve().parent / "theme.qss").read_text()


def load_bundled_fonts() -> None:
    fonts_dir = Path(__file__).resolve().parent / "fonts"
    for filename in _FONT_FILES:
        QFontDatabase.addApplicationFont(str(fonts_dir / filename))


def _fit_to_screen(window: MainWindow, app: QApplication) -> None:
    """Size and center the window within the actual screen, never past it.

    A bare ``show()`` sizes a top-level window from its layout's size hint,
    which here is the *largest* size hint across every page in the
    ``QStackedWidget`` -- not just the one shown first (``QStackedWidget``
    reports the max over all of its pages, by Qt design). On a screen
    shorter than that hint, the window used to open partly off-screen, with
    no way to resize or scroll it back into view. Clamping to the screen's
    available geometry (excluding taskbars/docks) guarantees it always
    opens fully visible, regardless of how tall any one page's content is.
    """
    screen = app.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    width = min(window.sizeHint().width(), available.width())
    height = min(window.sizeHint().height(), available.height())
    width = max(width, min(window.minimumWidth(), available.width()))
    height = max(height, min(window.minimumHeight(), available.height()))
    window.resize(width, height)
    window.move(available.center() - window.rect().center())


def main(argv: list = None) -> int:
    _prefer_xcb_on_wayland()
    app = QApplication(list(sys.argv if argv is None else argv))
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    load_bundled_fonts()
    app.setStyleSheet(load_stylesheet())

    window = MainWindow()
    _fit_to_screen(window, app)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
