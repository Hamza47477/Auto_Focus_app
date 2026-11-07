"""Small helpers shared by every widget module."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget


def set_class(widget: QWidget, class_str: str) -> None:
    """Set the dynamic "class" property used by theme.qss selectors and re-polish.

    Qt caches which QSS rules apply to a widget; changing the ``class``
    property at runtime (e.g. a stat value turning orange) needs an
    unpolish/polish cycle to actually take effect.
    """
    widget.setProperty("class", class_str)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
