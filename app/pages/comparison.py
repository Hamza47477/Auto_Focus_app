"""Comparison page (section 2.4): all 3 methods on all loaded datasets.

Runs full sweep + golden section + N random-start gradient ascents per
dataset through :class:`~app.workers.ComparisonWorker`, then renders bar
charts, a results table and a "where each method landed" plot. All numbers
come from ``autofocus`` results -- this page only aggregates/formats them.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..colors import PHASE_COLORS
from ..state import AppState, LoadedDataset
from ..widgets.common import set_class
from ..widgets.image_view import load_pixmap
from ..widgets.plot_widget import FocusCurveCanvas, LineChartCanvas
from ..workers import ComparisonWorker, DatasetComparison

#: (result key, display title) for the three result-image thumbnails.
RESULT_THUMB_METHODS = [
    ("full_sweep", "Full sweep"),
    ("golden_section", "Golden-section"),
    ("gradient_ascent", "Gradient ascent"),
]

RESULTS_COLUMNS = [
    "Method", "Images read", "Time (ms)", "Coarse index", "Coarse position (mm)",
    "Fine position (mm)", "Error (images)", "Error (µm)", "Result",
]


def _error_vs_sweep(result, sweep) -> tuple:
    error_images = abs(result.coarse_index - sweep.coarse_index)
    error_um = abs(result.coarse_position_mm - sweep.coarse_position_mm) * 1000.0
    return error_images, error_um


class ComparisonPage(QWidget):
    """"Comparison" screen: run all methods on all datasets and compare them."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker: Optional[ComparisonWorker] = None
        self._comparisons: List[DatasetComparison] = []

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
        root.addLayout(self._build_controls())

        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)
        self.chart_images = LineChartCanvas()
        self.chart_time = LineChartCanvas()
        self.chart_success = LineChartCanvas()
        for chart in (self.chart_images, self.chart_time, self.chart_success):
            frame = QFrame()
            set_class(frame, "card")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.addWidget(chart)
            frame.setMinimumHeight(260)
            charts_row.addWidget(frame)
        charts_widget = QWidget()
        charts_widget.setLayout(charts_row)
        root.addWidget(charts_widget)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(14)
        bottom_row.addWidget(self._build_table_card(), 3)
        bottom_row.addWidget(self._build_landed_card(), 2)
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_row)
        bottom_widget.setMinimumHeight(360)
        root.addWidget(bottom_widget)

    # -- layout ------------------------------------------------------------

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        col = QVBoxLayout()
        title = QLabel("Comparison")
        set_class(title, "page-title")
        self.subtitle = QLabel("Run gradient ascent, golden-section and full sweep on every loaded dataset")
        set_class(self.subtitle, "page-subtitle")
        col.addWidget(title)
        col.addWidget(self.subtitle)
        row.addLayout(col)
        row.addStretch(1)
        export_btn = QPushButton("Export report")
        export_btn.clicked.connect(self._on_export_report)
        row.addWidget(export_btn)
        return row

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Random starts per dataset"))
        self.n_starts_spin = QSpinBox()
        self.n_starts_spin.setRange(1, 500)
        self.n_starts_spin.setValue(20)
        row.addWidget(self.n_starts_spin)

        row.addWidget(QLabel("Seed"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setValue(0)
        row.addWidget(self.seed_spin)

        self.run_btn = QPushButton("Run comparison")
        set_class(self.run_btn, "primary")
        self.run_btn.clicked.connect(self._on_run)
        row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        set_class(self.stop_btn, "danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self.stop_btn)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        row.addWidget(self.progress, 1)
        return row

    def _build_table_card(self) -> QFrame:
        frame = QFrame()
        set_class(frame, "card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)

        header = QHBoxLayout()
        title = QLabel("Results table")
        set_class(title, "card-title")
        header.addWidget(title)
        header.addStretch(1)
        self.table_dataset_combo = QComboBox()
        self.table_dataset_combo.currentTextChanged.connect(self._refresh_table)
        header.addWidget(self.table_dataset_combo)
        layout.addLayout(header)

        thumbs_row = QHBoxLayout()
        thumbs_row.setSpacing(12)
        self.result_thumbs: Dict[str, QLabel] = {}
        self.result_captions: Dict[str, QLabel] = {}
        for key, title in RESULT_THUMB_METHODS:
            col = QVBoxLayout()
            col.setSpacing(4)
            name_label = QLabel(title)
            set_class(name_label, "dim")
            img_label = QLabel("—")
            img_label.setFixedSize(130, 96)
            img_label.setAlignment(Qt.AlignCenter)
            set_class(img_label, "card-flat")
            caption = QLabel("—")
            caption.setWordWrap(True)
            set_class(caption, "mono dim")
            col.addWidget(name_label)
            col.addWidget(img_label)
            col.addWidget(caption)
            self.result_thumbs[key] = img_label
            self.result_captions[key] = caption
            thumbs_row.addLayout(col)
        thumbs_row.addStretch(1)
        layout.addLayout(thumbs_row)

        self.table = QTableWidget(0, len(RESULTS_COLUMNS))
        self.table.setHorizontalHeaderLabels(RESULTS_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        summary_row = QHBoxLayout()
        self.summary_label = QLabel("Run a comparison to see results.")
        set_class(self.summary_label, "dim")
        summary_row.addWidget(self.summary_label)
        layout.addLayout(summary_row)
        return frame

    def _build_landed_card(self) -> QFrame:
        frame = QFrame()
        set_class(frame, "card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        header = QHBoxLayout()
        title = QLabel("Where each method landed")
        set_class(title, "card-title")
        header.addWidget(title)
        header.addStretch(1)
        self.landed_dataset_combo = QComboBox()
        self.landed_dataset_combo.currentTextChanged.connect(self._refresh_landed_plot)
        header.addWidget(self.landed_dataset_combo)
        layout.addLayout(header)
        self.landed_plot = FocusCurveCanvas()
        layout.addWidget(self.landed_plot, 1)
        return frame

    # -- running -------------------------------------------------------

    def _on_run(self) -> None:
        datasets = list(self.state.datasets.values())
        if not datasets:
            QMessageBox.warning(self, "No datasets", "Load at least one dataset on the Datasets page first.")
            return

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self._comparisons = []

        worker = ComparisonWorker(datasets, self.state.settings, self.n_starts_spin.value(), self.seed_spin.value())
        worker.progress.connect(lambda done, total: self.progress.setValue(int(100 * done / max(1, total))))
        worker.dataset_done.connect(self._on_dataset_done)
        worker.finished_ok.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def _on_dataset_done(self, comparison: DatasetComparison) -> None:
        self._comparisons.append(comparison)

    def _on_finished(self, comparisons: List[DatasetComparison]) -> None:
        self._comparisons = comparisons
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        names = [c.dataset_name for c in comparisons]
        for combo in (self.table_dataset_combo, self.landed_dataset_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.blockSignals(False)

        self._refresh_charts()
        if names:
            self._refresh_table(names[0])
            self._refresh_landed_plot(names[0])

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Comparison failed", message)

    # -- charts -------------------------------------------------------

    def _refresh_charts(self) -> None:
        names = [c.dataset_name for c in self._comparisons]
        images = {"Full sweep": [], "Golden-section": [], "Gradient ascent": []}
        times = {"Full sweep": [], "Golden-section": [], "Gradient ascent": []}
        success = {"Full sweep": [], "Golden-section": [], "Gradient ascent": []}

        for c in self._comparisons:
            images["Full sweep"].append(c.full_sweep.images_read)
            images["Golden-section"].append(c.golden_section.images_read)
            images["Gradient ascent"].append(float(np.mean([r.images_read for r in c.gradient_runs])) if c.gradient_runs else 0)

            times["Full sweep"].append(c.full_sweep.elapsed_s * 1000)
            times["Golden-section"].append(c.golden_section.elapsed_s * 1000)
            times["Gradient ascent"].append(
                float(np.mean([r.elapsed_s for r in c.gradient_runs])) * 1000 if c.gradient_runs else 0
            )

            golden_ok = abs(c.golden_section.coarse_index - c.full_sweep.coarse_index) <= 1
            success["Full sweep"].append(100.0)
            success["Golden-section"].append(100.0 if golden_ok else 0.0)
            if c.gradient_runs:
                rate = 100.0 * sum(
                    1 for r in c.gradient_runs if abs(r.coarse_index - c.full_sweep.coarse_index) <= 1
                ) / len(c.gradient_runs)
            else:
                rate = 0.0
            success["Gradient ascent"].append(rate)

        colors = {"Full sweep": "#5b6472", "Golden-section": PHASE_COLORS["golden_probe"], "Gradient ascent": PHASE_COLORS["ascent"]}

        def series(data):
            return [(label, data[label], colors[label]) for label in ("Full sweep", "Golden-section", "Gradient ascent")]

        self.chart_images.plot_series(names, series(images), "images read", "Images read", log_scale=True)
        self.chart_time.plot_series(names, series(times), "ms", "Search time", log_scale=True)
        self.chart_success.plot_series(names, series(success), "%", "Success rate (peak ±1)")

    # -- table ----------------------------------------------------------

    def _find_comparison(self, name: str) -> Optional[DatasetComparison]:
        for c in self._comparisons:
            if c.dataset_name == name:
                return c
        return None

    def _refresh_table(self, name: str) -> None:
        comparison = self._find_comparison(name)
        self.table.setRowCount(0)
        loaded = self.state.datasets.get(name)
        if comparison is None or loaded is None:
            self._clear_result_thumbs()
            return

        sweep = comparison.full_sweep
        rows = []

        rows.append([
            "Full sweep", str(sweep.images_read), f"{sweep.elapsed_s * 1000:.1f}", str(sweep.coarse_index),
            f"{sweep.coarse_position_mm:.4f}", f"{sweep.fine_position_mm:.4f}", "0", "0.0", "ground truth",
        ])
        self._set_result_thumb("full_sweep", loaded, sweep)

        golden = comparison.golden_section
        err_i, err_um = _error_vs_sweep(golden, sweep)
        rows.append([
            "Golden-section", str(golden.images_read), f"{golden.elapsed_s * 1000:.1f}", str(golden.coarse_index),
            f"{golden.coarse_position_mm:.4f}", f"{golden.fine_position_mm:.4f}", str(err_i), f"{err_um:.2f}",
            "Global max" if err_i <= 1 else "Local max",
        ])
        self._set_result_thumb("golden_section", loaded, golden)

        if comparison.gradient_runs:
            errs = [_error_vs_sweep(r, sweep) for r in comparison.gradient_runs]
            mean_images = np.mean([r.images_read for r in comparison.gradient_runs])
            mean_time = np.mean([r.elapsed_s for r in comparison.gradient_runs]) * 1000
            mean_err_i = np.mean([e[0] for e in errs])
            mean_err_um = np.mean([e[1] for e in errs])
            n_ok = sum(1 for e in errs if e[0] <= 1)
            rows.append([
                f"Gradient ascent (N={len(comparison.gradient_runs)})", f"{mean_images:.1f}", f"{mean_time:.1f}",
                "varies", "varies", "varies", f"{mean_err_i:.2f}", f"{mean_err_um:.1f}",
                f"{n_ok}/{len(comparison.gradient_runs)} global max",
            ])
            # Gradient ascent has no single result across N random starts, so the
            # thumbnail shows the most common landing (the mode), not just run 0.
            counts = Counter(r.coarse_index for r in comparison.gradient_runs)
            mode_index, mode_count = counts.most_common(1)[0]
            representative = next(r for r in comparison.gradient_runs if r.coarse_index == mode_index)
            self._set_result_thumb(
                "gradient_ascent", loaded, representative, note=f"most common · {mode_count}/{len(comparison.gradient_runs)} runs"
            )
        else:
            self._clear_result_thumb("gradient_ascent")

        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(text))

        n_ok_text = rows[-1][-1] if comparison.gradient_runs else "—"
        self.summary_label.setText(f"{name}: golden-section {rows[1][-1].lower()} · gradient ascent {n_ok_text}")

    def _set_result_thumb(self, key: str, loaded: LoadedDataset, result, note: str = "") -> None:
        img_label = self.result_thumbs[key]
        try:
            pixmap = load_pixmap(loaded.dataset.path_for(result.coarse_index))
        except IOError:
            img_label.setText("unavailable")
            self.result_captions[key].setText("—")
            return
        img_label.setText("")
        img_label.setPixmap(pixmap.scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        caption = f"#{result.coarse_index} · NV {result.coarse_focus:.3f}\n{result.coarse_position_mm:.4f} mm"
        if note:
            caption += f"\n{note}"
        self.result_captions[key].setText(caption)

    def _clear_result_thumb(self, key: str) -> None:
        self.result_thumbs[key].clear()
        self.result_thumbs[key].setText("—")
        self.result_captions[key].setText("—")

    def _clear_result_thumbs(self) -> None:
        for key, _title in RESULT_THUMB_METHODS:
            self._clear_result_thumb(key)

    # -- landed plot ------------------------------------------------------

    def _refresh_landed_plot(self, name: str) -> None:
        comparison = self._find_comparison(name)
        self.landed_plot.reset()
        if comparison is None:
            return
        loaded = self.state.datasets.get(name)
        if loaded is None:
            return

        sweep = comparison.full_sweep
        curve = sorted(((s.index, s.focus) for s in sweep.trace), key=lambda p: p[0])
        positions = [loaded.dataset.position_mm(i) for i, _ in curve]
        values = [f for _, f in curve]
        self.landed_plot.set_reference_curve(positions, values)

        if comparison.gradient_runs:
            gx = [loaded.dataset.position_mm(r.coarse_index) for r in comparison.gradient_runs]
            gy = [r.coarse_focus for r in comparison.gradient_runs]
            self.landed_plot.scatter(gx, gy, PHASE_COLORS["ascent"], size=24, label="Gradient ascent landing")

        self.landed_plot.scatter(
            [loaded.dataset.position_mm(comparison.golden_section.coarse_index)],
            [comparison.golden_section.coarse_focus],
            PHASE_COLORS["golden_probe"], size=50, marker="D", label="Golden-section landing",
        )
        self.landed_plot.scatter(
            [loaded.dataset.position_mm(sweep.coarse_index)], [sweep.coarse_focus],
            PHASE_COLORS["start"], size=70, marker="*", label="Full sweep (reference peak)",
        )

    # -- export -------------------------------------------------------

    def _on_export_report(self) -> None:
        if not self._comparisons:
            QMessageBox.information(self, "Nothing to export", "Run a comparison first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not folder:
            return
        out = Path(folder)

        with open(out / "comparison_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset"] + RESULTS_COLUMNS)
            for comparison in self._comparisons:
                current = self.table_dataset_combo.currentText()
                self.table_dataset_combo.blockSignals(True)
                self.table_dataset_combo.setCurrentText(comparison.dataset_name)
                self.table_dataset_combo.blockSignals(False)
                self._refresh_table(comparison.dataset_name)
                for r in range(self.table.rowCount()):
                    row = [self.table.item(r, c).text() for c in range(self.table.columnCount())]
                    writer.writerow([comparison.dataset_name] + row)
                self.table_dataset_combo.setCurrentText(current)

        for chart, filename in [
            (self.chart_images, "comparison_images_read.png"),
            (self.chart_time, "comparison_time.png"),
            (self.chart_success, "comparison_success_rate.png"),
            (self.landed_plot, "comparison_where_landed.png"),
        ]:
            chart.figure.savefig(out / filename, dpi=200, facecolor=chart.figure.get_facecolor())

        QMessageBox.information(self, "Export complete", f"Report written to:\n{out}")
