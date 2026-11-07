"""Walkthroughs page (section 2.5): step-by-step replay of a finished run.

One tab per method. Replays an already-finished ``AutofocusResult.trace``
from ``state.run_history`` -- no search runs here. The gradient-ascent tab's
slope / direction / step size / random factor / accepted-reject fields come
from parsing the ``key=value`` tokens search.py now writes into
``TraceStep.note`` for exactly this purpose (see search.py's ``_climb``).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autofocus.results import TraceStep

from ..state import AppState, RunRecord
from ..widgets.common import set_class
from ..widgets.log_table import LogTable
from ..widgets.plot_widget import FocusCurveCanvas

_KV_RE = re.compile(r"(\w+)=([+-]?[\w.]+)")


def parse_note_kv(note: str) -> Dict[str, str]:
    """Pull ``key=value`` tokens out of an enriched trace note.

    e.g. ``"candidate lam=21.602 step=22 u=1.137 accepted=no"`` ->
    ``{"lam": "21.602", "step": "22", "u": "1.137", "accepted": "no"}``.
    A GUI-only parser for the plain-text format search.py writes; it never
    touches the algorithm itself.
    """
    return dict(_KV_RE.findall(note))


DETAIL_FIELDS = [
    ("index", "Index"),
    ("position_mm", "Position (mm)"),
    ("focus", "NV"),
    ("phase", "Phase"),
    ("slope", "Slope"),
    ("direction", "Direction"),
    ("step_size", "Step size Λn"),
    ("random_factor", "Random factor u"),
    ("accepted", "Accepted"),
    ("restarts_so_far", "Restarts so far"),
]


class MethodWalkthroughTab(QWidget):
    """One method's replay: playback controls, plot, step detail, step log."""

    def __init__(self, state: AppState, method: str, parent=None):
        super().__init__(parent)
        self.state = state
        self.method = method
        self._run: Optional[RunRecord] = None
        self._position = 0
        self._rendered_up_to = 0
        self._restarts_so_far = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 12, 0, 0)
        root.setSpacing(12)

        root.addLayout(self._build_controls())

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self._build_plot_card(), 3)
        body.addWidget(self._build_detail_card(), 2)
        root.addLayout(body, 2)

        log_frame = QFrame()
        set_class(log_frame, "card")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(16, 14, 16, 14)
        log_title = QLabel("Step log")
        set_class(log_title, "card-title")
        log_layout.addWidget(log_title)
        self.log_table = LogTable()
        self.log_table.setMinimumHeight(260)
        log_layout.addWidget(self.log_table)
        root.addWidget(log_frame, 1)

        self.state.run_recorded.connect(self._on_run_recorded)
        self._refresh_run_combo()

    # -- layout -----------------------------------------------------------

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Run"))
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(240)
        self.run_combo.currentIndexChanged.connect(self._on_run_selected)
        row.addWidget(self.run_combo)
        row.addStretch(1)

        self.prev_btn = QPushButton("⏮ Previous")
        self.prev_btn.clicked.connect(self._on_previous)
        row.addWidget(self.prev_btn)
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._on_play_pause)
        row.addWidget(self.play_btn)
        self.next_btn = QPushButton("Next ⏭")
        self.next_btn.clicked.connect(self._on_next)
        row.addWidget(self.next_btn)

        row.addWidget(QLabel("Speed"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(5)
        self.speed_slider.setFixedWidth(100)
        row.addWidget(self.speed_slider)

        self.step_label = QLabel("Step 0 / 0")
        set_class(self.step_label, "mono dim")
        row.addWidget(self.step_label)
        return row

    def _build_plot_card(self) -> QFrame:
        frame = QFrame()
        set_class(frame, "card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        title = QLabel("Replay")
        set_class(title, "card-title")
        layout.addWidget(title)
        self.plot = FocusCurveCanvas()
        self.plot.setMinimumHeight(420)
        layout.addWidget(self.plot, 1)
        return frame

    def _build_detail_card(self) -> QFrame:
        frame = QFrame()
        set_class(frame, "card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        title = QLabel("Step detail")
        set_class(title, "card-title")
        layout.addWidget(title)

        self.detail_rows: Dict[str, QLabel] = {}
        for key, label in DETAIL_FIELDS:
            if key in ("slope", "direction", "step_size", "random_factor", "accepted") and self.method != "gradient_ascent":
                continue
            row = QHBoxLayout()
            l = QLabel(label)
            set_class(l, "muted")
            v = QLabel("—")
            set_class(v, "mono")
            row.addWidget(l)
            row.addStretch(1)
            row.addWidget(v)
            layout.addLayout(row)
            self.detail_rows[key] = v
        layout.addStretch(1)
        return frame

    # -- run selection ---------------------------------------------------

    def _runs(self) -> List[RunRecord]:
        return self.state.runs_for_method(self.method)

    def _refresh_run_combo(self) -> None:
        runs = self._runs()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        self.run_combo.addItems([r.label for r in runs])
        self.run_combo.blockSignals(False)
        if runs:
            self.run_combo.setCurrentIndex(len(runs) - 1)
            self._load_run(runs[-1])

    def _on_run_recorded(self, record: RunRecord) -> None:
        if record.result.method == self.method:
            self._refresh_run_combo()

    def _on_run_selected(self, index: int) -> None:
        runs = self._runs()
        if 0 <= index < len(runs):
            self._load_run(runs[index])

    def _load_run(self, run: RunRecord) -> None:
        self._timer.stop()
        self.play_btn.setText("▶ Play")
        self._run = run
        self._position = 0
        self._rendered_up_to = 0
        self._restarts_so_far = 0
        self.log_table.clear_rows()
        self.plot.reset()
        self._clear_detail()
        self._update_step_label()

    # -- playback ----------------------------------------------------------

    def _on_play_pause(self) -> None:
        if self._run is None:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            interval_ms = int(1000 / max(1, self.speed_slider.value()))
            self._timer.start(interval_ms)
            self.play_btn.setText("⏸ Pause")

    def _on_tick(self) -> None:
        if self._run is None or self._position >= len(self._run.result.trace):
            self._timer.stop()
            self.play_btn.setText("▶ Play")
            return
        self._step_forward()

    def _on_next(self) -> None:
        self._timer.stop()
        self.play_btn.setText("▶ Play")
        self._step_forward()

    def _on_previous(self) -> None:
        self._timer.stop()
        self.play_btn.setText("▶ Play")
        if self._run is None or self._position == 0:
            return
        self._position -= 1
        self._render_up_to(self._position)

    def _step_forward(self) -> None:
        if self._run is None or self._position >= len(self._run.result.trace):
            return
        self._position += 1
        self._render_up_to(self._position)
        if self._position >= len(self._run.result.trace):
            self._timer.stop()
            self.play_btn.setText("▶ Play")

    # -- rendering ----------------------------------------------------------

    def _render_up_to(self, position: int) -> None:
        if self._run is None:
            return
        result = self._run.result
        trace = result.trace

        if position < self._rendered_up_to:
            self.plot.reset()
            self.log_table.clear_rows()
            self._rendered_up_to = 0
            self._restarts_so_far = 0

        for i in range(self._rendered_up_to, position):
            step = trace[i]
            is_start = i == 0 and self.method == "gradient_ascent"
            self.plot.add_point(step.position_mm, step.focus, step.phase, step.step_no, is_start=is_start)
            self.log_table.append_step(step, elapsed_ms=0.0)
            if step.phase == "restart":
                self._restarts_so_far += 1
        self._rendered_up_to = position

        if position > 0:
            self._show_detail(trace[position - 1])
        else:
            self._clear_detail()

        if position >= len(trace) and result.fit_points:
            a0, a1, a2 = result.fit_coefficients
            xs = np.linspace(-result.settings.fit_half_window, result.settings.fit_half_window, 100)
            ys = a0 + a1 * xs + a2 * xs ** 2
            loaded = self.state.datasets.get(self._run.dataset_name)
            if loaded is not None:
                positions = loaded.dataset.d_min_mm + (result.coarse_index + xs) * loaded.dataset.step_mm
                self.plot.set_fit(positions, ys, result.fine_position_mm)

        self._update_step_label()

    def _show_detail(self, step: TraceStep) -> None:
        kv = parse_note_kv(step.note)
        values = {
            "index": str(step.index),
            "position_mm": f"{step.position_mm:.4f}",
            "focus": f"{step.focus:.6f}",
            "phase": step.phase,
            "slope": kv.get("slope", "—"),
            "direction": kv.get("dir", "—"),
            "step_size": kv.get("lam", "—"),
            "random_factor": kv.get("u", "—"),
            "accepted": kv.get("accepted", "—"),
            "restarts_so_far": str(self._restarts_so_far),
        }
        for key, label_widget in self.detail_rows.items():
            label_widget.setText(values.get(key, "—"))

    def _clear_detail(self) -> None:
        for label_widget in self.detail_rows.values():
            label_widget.setText("—")

    def _update_step_label(self) -> None:
        total = len(self._run.result.trace) if self._run else 0
        self.step_label.setText(f"Step {self._position} / {total}")


class WalkthroughsPage(QWidget):
    """"Walkthroughs" screen: one tab per method, replaying finished runs."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(10)

        title = QLabel("Walkthroughs")
        set_class(title, "page-title")
        subtitle = QLabel("Replay a finished run, step by step")
        set_class(subtitle, "page-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        tabs = QTabWidget()
        self.tabs: Dict[str, MethodWalkthroughTab] = {}
        for method, label in [
            ("gradient_ascent", "Gradient ascent"),
            ("golden_section", "Golden-section"),
            ("full_sweep", "Full sweep"),
        ]:
            tab = MethodWalkthroughTab(state, method)
            tabs.addTab(tab, label)
            self.tabs[method] = tab
        layout.addWidget(tabs, 1)
