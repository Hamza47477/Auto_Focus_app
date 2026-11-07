"""Small label-over-value card used throughout the app for result stats."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from .common import set_class


class StatCard(QFrame):
    """A "stat-label" on top, a big "stat-value" below (e.g. "Images read" / "42 of 199")."""

    def __init__(self, label: str, value: str = "—", accent: str = "", parent=None):
        super().__init__(parent)
        set_class(self, "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        self._label = QLabel(label)
        set_class(self._label, "stat-label")
        self._value = QLabel(value)
        self._accent = accent
        set_class(self._value, f"stat-value {accent}".strip())
        self._value.setWordWrap(True)

        layout.addWidget(self._label)
        layout.addWidget(self._value)

    def set_value(self, value: str, accent: Optional[str] = None) -> None:
        self._value.setText(value)
        if accent is not None and accent != self._accent:
            self._accent = accent
            set_class(self._value, f"stat-value {accent}".strip())

    def set_label(self, label: str) -> None:
        self._label.setText(label)
