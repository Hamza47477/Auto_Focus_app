"""Evaluation log table: one row per focus evaluation (section 2.3)."""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

from autofocus.results import TraceStep

COLUMNS = ["Step", "Index", "Position (mm)", "Phase", "NV", "Note", "Elapsed (ms)"]
_RIGHT_ALIGNED = {0, 1, 2, 4, 6}


class LogTable(QTableWidget):
    """Append-only table of :class:`TraceStep` rows, with CSV export support."""

    def __init__(self, parent=None):
        super().__init__(0, len(COLUMNS), parent)
        self.setHorizontalHeaderLabels(COLUMNS)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

        header = self.horizontalHeader()
        for col in range(len(COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)  # Note column

    def clear_rows(self) -> None:
        self.setRowCount(0)

    def append_step(self, step: TraceStep, elapsed_ms: float) -> None:
        row = self.rowCount()
        self.insertRow(row)
        values = [
            str(step.step_no),
            str(step.index),
            f"{step.position_mm:.4f}",
            step.phase,
            f"{step.focus:.6f}",
            step.note,
            f"{elapsed_ms:.2f}",
        ]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            if col in _RIGHT_ALIGNED:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, col, item)
        self.scrollToBottom()

    def rows_as_lists(self) -> List[List[str]]:
        return [
            [self.item(r, c).text() if self.item(r, c) else "" for c in range(self.columnCount())]
            for r in range(self.rowCount())
        ]
