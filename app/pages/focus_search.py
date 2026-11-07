"""Focus Search page (section 2.3): the main screen.

Runs exactly one ``autofocus`` search function (gradient ascent, golden
section or full sweep) in a background :class:`~app.workers.SearchWorker`
and renders its live trace: best image, evaluation strip, result stats,
focus curve and log table. This page computes no focus values itself.
"""

from __future__ import annotations

import csv
import time
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from autofocus import AutofocusResult, full_sweep, golden_section, gradient_ascent
from autofocus.results import TraceStep

from ..colors import PHASE_COLORS, PHASE_LABELS
from ..state import AppState
from ..widgets.common import set_class
from ..widgets.image_view import EvaluationStrip, load_pixmap
from ..widgets.log_table import LogTable
from ..widgets.plot_widget import FocusCurveCanvas
from ..widgets.stat_card import StatCard
from ..workers import SearchWorker

METHOD_LABELS = {
    "gradient_ascent": "Gradient ascent",
    "golden_section": "Golden-section",
    "full_sweep": "Full sweep",
}
METHOD_FUNCS = {
    "gradient_ascent": gradient_ascent,
    "golden_section": golden_section,
    "full_sweep": full_sweep,
}


class LegendChip(QWidget):
    def __init__(self, color_hex: str, text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {color_hex}; font-size: 12px;")
        label = QLabel(text)
        set_class(label, "dim")
        layout.addWidget(dot)
        layout.addWidget(label)


class FocusSearchPage(QWidget):
    """"Focus Search" screen: header controls, best image + strip, results, plot, log."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker: Optional[SearchWorker] = None
        self._run_start: float = 0.0
        self._images_seen: set = set()
        self._best_index: Optional[int] = None
        self._best_focus: float = float("-inf")
        self._current_method: str = "gradient_ascent"
        self._current_result: Optional[AutofocusResult] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(14)

        root.addLayout(self._build_header())

        # A plain two-column row, not a QSplitter: QSplitter panes don't
        # recompute their internal layout correctly when the splitter itself
        # sits inside a QScrollArea's resizable content (verified directly --
        # child widgets inside a pane get positioned with stale geometry,
        # e.g. the best-image label overlapping the label below it). A fixed
        # 2:3 column row has none of that risk.
        columns = QHBoxLayout()
        columns.setSpacing(16)
        columns.addWidget(self._build_left_column(), 2)
        columns.addWidget(self._build_right_column(), 3)
        columns_widget = QWidget()
        columns_widget.setLayout(columns)
        columns_widget.setMinimumHeight(640)
        root.addWidget(columns_widget, 1)

        self.state.datasets_changed.connect(self._refresh_dataset_combo)
        self.state.active_dataset_changed.connect(self._on_active_dataset_changed)
        self._refresh_dataset_combo()
        self._on_method_changed(self.method_combo.currentIndex())

    # -- layout --------------------------------------------------------

    def _build_header(self) -> QVBoxLayout:
        root = QVBoxLayout()
        root.setSpacing(10)

        title = QLabel("Focus Search")
        set_class(title, "page-title")
        subtitle = QLabel("Watch the algorithm pick images and converge on the sharpest one")
        set_class(subtitle, "page-subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        # Controls live on their own row below the title so they never
        # compete with it for space at the app's minimum width (1280px).
        row = QHBoxLayout()
        row.setSpacing(8)

        row.addWidget(QLabel("Dataset"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(110)
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_combo_changed)
        row.addWidget(self.dataset_combo)

        row.addWidget(QLabel("Method"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([METHOD_LABELS[k] for k in ("gradient_ascent", "golden_section", "full_sweep")])
        self.method_combo.setMinimumWidth(130)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        row.addWidget(self.method_combo)

        row.addWidget(QLabel("Start"))
        self.start_spin = QSpinBox()
        self.start_spin.setFixedWidth(65)
        row.addWidget(self.start_spin)
        self.random_start_btn = QPushButton("Random")
        self.random_start_btn.clicked.connect(self._on_random_start)
        row.addWidget(self.random_start_btn)
        self.start_note = QLabel("")
        set_class(self.start_note, "dim")
        row.addWidget(self.start_note)

        row.addStretch(1)

        self.step_btn = QPushButton("Step")
        self.step_btn.clicked.connect(self._on_step_clicked)
        row.addWidget(self.step_btn)
        self.run_btn = QPushButton("▶  Run")
        set_class(self.run_btn, "primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
        row.addWidget(self.run_btn)
        self.stop_btn = QPushButton("Stop")
        set_class(self.stop_btn, "danger")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.stop_btn)

        root.addLayout(row)
        return root

    def _build_left_column(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(14)

        best_frame = QFrame()
        set_class(best_frame, "card")
        best_layout = QVBoxLayout(best_frame)
        best_layout.setContentsMargins(16, 14, 16, 14)
        best_layout.setSpacing(8)

        header_row = QHBoxLayout()
        title = QLabel("Best image")
        set_class(title, "card-title")
        header_row.addWidget(title)
        header_row.addStretch(1)
        self.best_badge = QLabel("—")
        set_class(self.best_badge, "badge badge-orange")
        header_row.addWidget(self.best_badge)
        best_layout.addLayout(header_row)

        self.best_image_label = QLabel("No run yet")
        self.best_image_label.setAlignment(Qt.AlignCenter)
        self.best_image_label.setMinimumHeight(220)
        set_class(self.best_image_label, "card-flat")
        best_layout.addWidget(self.best_image_label, 1)

        self.best_info_label = QLabel("—")
        set_class(self.best_info_label, "mono muted")
        best_layout.addWidget(self.best_info_label)

        strip_label_row = QHBoxLayout()
        strip_label = QLabel("Images read")
        set_class(strip_label, "dim")
        self.images_read_label = QLabel("0 of 0 (0%)")
        set_class(self.images_read_label, "dim mono")
        strip_label_row.addWidget(strip_label)
        strip_label_row.addStretch(1)
        strip_label_row.addWidget(self.images_read_label)
        best_layout.addLayout(strip_label_row)

        self.strip = EvaluationStrip()
        self.strip.cell_clicked.connect(self._on_strip_clicked)
        best_layout.addWidget(self.strip)

        legend_row = QHBoxLayout()
        legend_row.setSpacing(10)
        for phase in ("start", "ascent", "safety_sample", "restart", "fit"):
            legend_row.addWidget(LegendChip(PHASE_COLORS[phase], PHASE_LABELS[phase]))
        legend_row.addStretch(1)
        best_layout.addLayout(legend_row)

        layout.addWidget(best_frame)

        self.stat_grid_frame = QFrame()
        set_class(self.stat_grid_frame, "card")
        grid_outer = QVBoxLayout(self.stat_grid_frame)
        grid_outer.setContentsMargins(16, 14, 16, 14)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.stats: Dict[str, StatCard] = {
            "coarse_index": StatCard("Coarse index"),
            "focus_value": StatCard("Focus value (NV)"),
            "coarse_position": StatCard("Coarse position (mm)"),
            "fine_position": StatCard("Fine position (mm)"),
            "images_read": StatCard("Images read"),
            "time": StatCard("Time"),
            "restarts": StatCard("Restarts"),
        }
        positions = [
            ("coarse_index", 0, 0), ("focus_value", 0, 1),
            ("coarse_position", 1, 0), ("fine_position", 1, 1),
            ("images_read", 2, 0), ("time", 2, 1),
            ("restarts", 3, 0),
        ]
        for key, r, c in positions:
            grid.addWidget(self.stats[key], r, c)
        grid_outer.addLayout(grid)
        layout.addWidget(self.stat_grid_frame)

        self.warnings_frame = QFrame()
        set_class(self.warnings_frame, "card")
        warnings_layout = QVBoxLayout(self.warnings_frame)
        warnings_layout.setContentsMargins(16, 12, 16, 12)
        warn_title = QLabel("Warnings")
        set_class(warn_title, "card-title accent-red")
        warnings_layout.addWidget(warn_title)
        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        set_class(self.warnings_label, "muted")
        warnings_layout.addWidget(self.warnings_label)
        self.warnings_frame.setVisible(False)
        layout.addWidget(self.warnings_frame)

        layout.addStretch(1)
        return container

    def _build_right_column(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(14)

        plot_frame = QFrame()
        set_class(plot_frame, "card")
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(16, 14, 16, 10)
        plot_header = QHBoxLayout()
        plot_title = QLabel("Focus curve")
        set_class(plot_title, "card-title")
        plot_header.addWidget(plot_title)
        plot_header.addStretch(1)
        save_png_btn = QPushButton("Save plot PNG")
        save_png_btn.clicked.connect(self._on_save_png)
        plot_header.addWidget(save_png_btn)
        plot_layout.addLayout(plot_header)
        self.plot = FocusCurveCanvas()
        self.plot.setMinimumHeight(280)
        plot_layout.addWidget(self.plot, 1)
        layout.addWidget(plot_frame, 3)

        log_frame = QFrame()
        set_class(log_frame, "card")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(16, 14, 16, 14)
        log_header = QHBoxLayout()
        log_title = QLabel("Evaluation log")
        set_class(log_title, "card-title")
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self._on_export_csv)
        log_header.addWidget(export_btn)
        log_layout.addLayout(log_header)
        self.log_table = LogTable()
        log_layout.addWidget(self.log_table)
        layout.addWidget(log_frame, 2)

        return container

    # -- combo / dataset sync --------------------------------------------

    def _refresh_dataset_combo(self) -> None:
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(list(self.state.datasets.keys()))
        if self.state.active_dataset_name:
            self.dataset_combo.setCurrentText(self.state.active_dataset_name)
        self.dataset_combo.blockSignals(False)
        self._sync_start_bounds()

    def _on_dataset_combo_changed(self, name: str) -> None:
        if name:
            self.state.set_active_dataset(name)

    def _on_active_dataset_changed(self, name: str) -> None:
        if self.dataset_combo.currentText() != name:
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.setCurrentText(name)
            self.dataset_combo.blockSignals(False)
        self._sync_start_bounds()

    def _sync_start_bounds(self) -> None:
        loaded = self.state.active_dataset()
        n = len(loaded.dataset) if loaded else 0
        self.start_spin.setMaximum(max(0, n - 1))
        self.strip.set_total(n)

    def _on_method_changed(self, index: int) -> None:
        self._current_method = ("gradient_ascent", "golden_section", "full_sweep")[index]
        is_gradient = self._current_method == "gradient_ascent"
        self.start_spin.setEnabled(is_gradient)
        self.random_start_btn.setEnabled(is_gradient)
        self.start_note.setText("" if is_gradient else "ignores the start image")

    def _on_random_start(self) -> None:
        loaded = self.state.active_dataset()
        if loaded is None:
            return
        rng = np.random.default_rng()
        self.start_spin.setValue(int(rng.integers(0, len(loaded.dataset))))

    # -- running ----------------------------------------------------------

    def _on_run_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.run_to_completion()
            return
        self._start_worker(single_step=False)

    def _on_step_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.advance()
            return
        self._start_worker(single_step=True)

    def _on_stop_clicked(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def _start_worker(self, single_step: bool) -> None:
        loaded = self.state.active_dataset()
        if loaded is None:
            QMessageBox.warning(self, "No dataset", "Load and select a dataset on the Datasets page first.")
            return

        self._reset_run_ui()
        fn = METHOD_FUNCS[self._current_method]
        if self._current_method == "gradient_ascent":
            args = (loaded.dataset, self.state.settings, self.start_spin.value())
        else:
            args = (loaded.dataset, self.state.settings)

        worker = SearchWorker(fn, args, {}, single_step=single_step)
        worker.step.connect(self._on_step)
        worker.finished_ok.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        self._run_start = time.perf_counter()
        self._set_running_ui(True)
        self.state.worker_state_changed.emit("running")
        self._update_status()
        worker.start()

    def _reset_run_ui(self) -> None:
        self.strip.clear()
        self._sync_start_bounds()
        self.log_table.clear_rows()
        self._images_seen = set()
        self._best_index = None
        self._best_focus = float("-inf")
        self.best_image_label.clear()
        self.best_image_label.setText("Searching…")
        self.best_badge.setText("—")
        self.best_info_label.setText("—")
        self.images_read_label.setText("0 of 0 (0%)")
        for stat in self.stats.values():
            stat.set_value("—")
        self.warnings_frame.setVisible(False)
        self.plot.reset()

        loaded = self.state.active_dataset()
        if loaded is not None and loaded.sweep_result is not None and self._current_method != "full_sweep":
            positions = [loaded.dataset.position_mm(s.index) for s in loaded.sweep_result.trace]
            values = [s.focus for s in loaded.sweep_result.trace]
            order = np.argsort(positions)
            self.plot.set_reference_curve(np.array(positions)[order], np.array(values)[order])

    def _set_running_ui(self, running: bool) -> None:
        self.run_btn.setEnabled(not running or (self._worker is not None and self._worker.single_step))
        self.step_btn.setEnabled(True)
        self.stop_btn.setEnabled(running)
        self.dataset_combo.setEnabled(not running)
        self.method_combo.setEnabled(not running)

    def _on_step(self, step: TraceStep) -> None:
        self._images_seen.add(step.index)
        loaded = self.state.active_dataset()
        n = len(loaded.dataset) if loaded else 0

        is_start = step.step_no == 1 and self._current_method == "gradient_ascent"
        self.strip.mark(step.index, "start" if is_start else step.phase)
        self.plot.add_point(step.position_mm, step.focus, step.phase, step.step_no, is_start=is_start)

        elapsed_ms = (time.perf_counter() - self._run_start) * 1000
        self.log_table.append_step(step, elapsed_ms)

        read = len(self._images_seen)
        pct = (100.0 * read / n) if n else 0.0
        self.images_read_label.setText(f"{read} of {n} ({pct:.0f}%)")

        if step.focus > self._best_focus:
            self._best_focus = step.focus
            self._best_index = step.index
            self.strip.set_best(step.index)
            self._show_best_image(step.index, step.focus, step.position_mm)

        self._update_status()

    def _show_best_image(self, index: int, focus: float, position_mm: float) -> None:
        loaded = self.state.active_dataset()
        if loaded is None:
            return
        try:
            pixmap = load_pixmap(loaded.dataset.path_for(index))
        except IOError:
            return
        self.best_image_label.setPixmap(
            pixmap.scaled(self.best_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.best_badge.setText(f"#{index} · NV {focus:.3f}")
        self.best_info_label.setText(f"#{index}  ·  {position_mm:.4f} mm")

    def _on_strip_clicked(self, index: int) -> None:
        loaded = self.state.active_dataset()
        if loaded is None:
            return
        try:
            pixmap = load_pixmap(loaded.dataset.path_for(index))
        except IOError:
            return
        self.best_image_label.setPixmap(
            pixmap.scaled(self.best_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.best_info_label.setText(f"#{index}  ·  {loaded.dataset.position_mm(index):.4f} mm  (clicked)")

    def _on_finished(self, result: AutofocusResult) -> None:
        self._current_result = result
        self._set_running_ui(False)
        self.run_btn.setEnabled(True)
        self.state.worker_state_changed.emit("done")
        self.state.record_run(result, self.state.active_dataset_name or "?", self._run_label(result))

        self.strip.set_best(result.coarse_index)
        self._show_best_image(result.coarse_index, result.coarse_focus, result.coarse_position_mm)

        self.stats["coarse_index"].set_value(f"#{result.coarse_index}", "accent-orange")
        self.stats["focus_value"].set_value(f"{result.coarse_focus:.6f}")
        self.stats["coarse_position"].set_value(f"{result.coarse_position_mm:.4f}")
        self.stats["fine_position"].set_value(f"{result.fine_position_mm:.4f}")
        self.stats["images_read"].set_value(f"{result.images_read}/{result.total_images} ({result.percent_read:.0f}%)")
        self.stats["time"].set_value(f"{result.elapsed_s * 1000:.1f} ms")
        self.stats["restarts"].set_value(str(result.restarts))

        if result.fit_points:
            self._draw_fit(result)

        if result.warnings:
            self.warnings_label.setText("\n".join(f"⚠ {w}" for w in result.warnings))
            self.warnings_frame.setVisible(True)
        else:
            self.warnings_frame.setVisible(False)

        self._update_status(done=True)
        self._worker = None

    def _draw_fit(self, result: AutofocusResult) -> None:
        loaded = self.state.active_dataset()
        if loaded is None:
            return
        a0, a1, a2 = result.fit_coefficients
        xs = np.linspace(-result.settings.fit_half_window, result.settings.fit_half_window, 100)
        ys = a0 + a1 * xs + a2 * xs ** 2
        positions = loaded.dataset.d_min_mm + (result.coarse_index + xs) * loaded.dataset.step_mm
        self.plot.set_fit(positions, ys, result.fine_position_mm)

    def _on_failed(self, message: str) -> None:
        self._set_running_ui(False)
        self.state.worker_state_changed.emit("error")
        self._update_status()
        self._worker = None
        QMessageBox.critical(self, "Search failed", message)

    def _run_label(self, result: AutofocusResult) -> str:
        dataset_name = self.state.active_dataset_name or "?"
        if result.method == "gradient_ascent":
            return f"{METHOD_LABELS[result.method]} — {dataset_name} — start #{result.start_index}"
        return f"{METHOD_LABELS[result.method]} — {dataset_name}"

    def _update_status(self, done: bool = False) -> None:
        dataset_name = self.state.active_dataset_name or "—"
        method_label = METHOD_LABELS[self._current_method]
        n_eval = len(self._images_seen)
        if self._worker is not None and self._worker.isRunning():
            worker_state = "stepping (paused)" if self._worker.single_step else "running"
        elif done:
            worker_state = "idle"
        else:
            worker_state = "idle"
        self.state.status_changed.emit(
            f"{method_label} · {dataset_name} · {n_eval} evaluations · worker: {worker_state}"
        )

    # -- export -------------------------------------------------------

    def _on_export_csv(self) -> None:
        if self.log_table.rowCount() == 0:
            QMessageBox.information(self, "Nothing to export", "Run a search first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "focus_search_log.csv", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            if self._current_result is not None:
                r = self._current_result
                writer.writerow(["# result summary"])
                writer.writerow(["method", r.method])
                writer.writerow(["start_index", r.start_index])
                writer.writerow(["coarse_index", r.coarse_index])
                writer.writerow(["coarse_focus", r.coarse_focus])
                writer.writerow(["coarse_position_mm", r.coarse_position_mm])
                writer.writerow(["fine_index", r.fine_index])
                writer.writerow(["fine_position_mm", r.fine_position_mm])
                writer.writerow(["images_read", r.images_read])
                writer.writerow(["total_images", r.total_images])
                writer.writerow(["percent_read", r.percent_read])
                writer.writerow(["elapsed_s", r.elapsed_s])
                writer.writerow(["restarts", r.restarts])
                writer.writerow(["seed", r.seed])
                writer.writerow([])
            from ..widgets.log_table import COLUMNS

            writer.writerow(["# evaluation log"])
            writer.writerow(COLUMNS)
            writer.writerows(self.log_table.rows_as_lists())

    def _on_save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save plot PNG", "focus_curve.png", "PNG files (*.png)")
        if path:
            self.plot.save_png(path)
