"""Datasets page (section 2.1): load, browse and quick-search a dataset.

Loading only lists files (via ``autofocus.ImageDataset``); no pixel is read
until the user picks an image to preview (``ImagePreview``, which bypasses
the focus evaluator) or runs one of the three search methods (which does go
through the corresponding ``autofocus`` function in a background worker).

Only a *full sweep* result is ever stored as the dataset's reference peak
(``LoadedDataset.sweep_result``, used elsewhere: the dataset card badge, the
Focus Search reference curve, the Comparison baseline) -- it is the only
method guaranteed to have read every image. Golden-section and gradient
ascent results are shown here for a quick look but are not persisted as
that reference, since neither guarantees the true global peak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autofocus import AutofocusResult

from ..state import AppState, LoadedDataset, discover_dataset_folders, display_name_for
from ..widgets.common import set_class
from ..widgets.image_view import ImagePreview, ThumbnailStrip
from ..widgets.stat_card import StatCard
from ..workers import SearchWorker
from .focus_search import METHOD_FUNCS, METHOD_LABELS

METHOD_ORDER = ("full_sweep", "golden_section", "gradient_ascent")


def count_local_maxima(result: AutofocusResult) -> int:
    """Display-only analytic: count local maxima in a finished full sweep's curve.

    Pure post-processing over an already-computed ``AutofocusResult.trace``
    (not part of the ``autofocus`` package -- no new focus values here).
    """
    curve = sorted(((step.index, step.focus) for step in result.trace), key=lambda p: p[0])
    values = [f for _, f in curve]
    count = 0
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            count += 1
    return count


class DatasetCard(QFrame):
    """Small clickable summary card for one loaded dataset (design's DS-N cards)."""

    clicked = Signal(str)

    def __init__(self, loaded: LoadedDataset, parent=None):
        super().__init__(parent)
        self.name = loaded.name
        set_class(self, "card")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        top = QHBoxLayout()
        title = QLabel(loaded.name)
        set_class(title, "card-title")
        top.addWidget(title)
        top.addStretch(1)
        self.badge = QLabel("Not scanned")
        set_class(self.badge, "badge badge-grey")
        top.addWidget(self.badge)
        layout.addLayout(top)

        self.info_label = QLabel(f"{len(loaded.dataset)} images")
        set_class(self.info_label, "muted")
        layout.addWidget(self.info_label)

        self.peak_label = QLabel("Peak —")
        set_class(self.peak_label, "dim mono")
        layout.addWidget(self.peak_label)

    def set_active(self, active: bool) -> None:
        set_class(self, "card-selected" if active else "card")

    def set_sweep_result(self, result: AutofocusResult) -> None:
        set_class(self.badge, "badge badge-green")
        self.badge.setText(f"Global max #{result.coarse_index}")
        self.peak_label.setText(f"Peak NV {result.coarse_focus:.3f}")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit(self.name)
        super().mousePressEvent(event)


class DatasetsPage(QWidget):
    """"Datasets" screen: load folders, preview images, run the reference sweep."""

    navigate_requested = Signal(str)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._cards: Dict[str, DatasetCard] = {}
        self._sweep_worker: Optional[SearchWorker] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        root.addLayout(self._build_header())

        self._cards_row = QHBoxLayout()
        self._cards_row.setSpacing(12)
        cards_container = QWidget()
        cards_container.setLayout(self._cards_row)
        root.addWidget(cards_container)

        content_row = QHBoxLayout()
        content_row.setSpacing(16)
        content_row.addWidget(self._build_preview_card(), 3)
        content_row.addWidget(self._build_info_panel(), 2)
        root.addLayout(content_row, 1)

        self.state.datasets_changed.connect(self._refresh_cards)
        self.state.active_dataset_changed.connect(self._on_active_changed)
        self.state.sweep_recorded.connect(self._on_sweep_recorded)

    # -- layout ------------------------------------------------------------

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel("Datasets")
        set_class(title, "page-title")
        self.subtitle = QLabel("No datasets loaded")
        set_class(self.subtitle, "page-subtitle")
        title_col.addWidget(title)
        title_col.addWidget(self.subtitle)
        row.addLayout(title_col)
        row.addStretch(1)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Dataset folder, or a root folder with speed sub-folders…")
        self.path_edit.setFixedWidth(360)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        load_btn = QPushButton("Load")
        set_class(load_btn, "primary")
        load_btn.clicked.connect(self._on_load)
        run_btn = QPushButton("Run focus search")
        set_class(run_btn, "primary")
        run_btn.clicked.connect(lambda: self.navigate_requested.emit("focus_search"))

        row.addWidget(self.path_edit)
        row.addWidget(browse_btn)
        row.addWidget(load_btn)
        row.addSpacing(12)
        row.addWidget(run_btn)
        return row

    def _build_preview_card(self) -> QFrame:
        frame = QFrame()
        set_class(frame, "card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Image viewer")
        set_class(title, "card-title")
        header.addWidget(title)
        self.sharpest_badge = QLabel("SHARPEST")
        set_class(self.sharpest_badge, "badge badge-orange")
        self.sharpest_badge.setVisible(False)
        header.addWidget(self.sharpest_badge)
        header.addStretch(1)
        self.stats_label = QLabel("")
        set_class(self.stats_label, "mono dim")
        header.addWidget(self.stats_label)
        layout.addLayout(header)

        self.preview = ImagePreview(show_stats=True)
        self.preview.stats_changed.connect(self._on_preview_stats)
        self.preview.index_changed.connect(self._on_preview_index_changed)
        layout.addWidget(self.preview, 1)

        self.thumbnails = ThumbnailStrip()
        self.thumbnails.thumbnail_clicked.connect(self.preview.show_index)
        layout.addWidget(self.thumbnails)
        return frame

    def _on_preview_stats(self, mu: float, variance: float, nv: float) -> None:
        self.stats_label.setText(f"NV {nv:.3f}   μ {mu:.1f}   σ² {variance:.1f}")

    def _on_preview_index_changed(self, index: int) -> None:
        loaded = self.state.active_dataset()
        is_sharpest = (
            loaded is not None and loaded.sweep_result is not None and loaded.sweep_result.coarse_index == index
        )
        self.sharpest_badge.setVisible(is_sharpest)
        if is_sharpest:
            self.sharpest_badge.setText(f"SHARPEST · #{index}")

    def _build_info_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        info_frame = QFrame()
        set_class(info_frame, "card")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(16, 14, 16, 14)
        info_layout.setSpacing(6)
        self.info_title = QLabel("—")
        set_class(self.info_title, "card-title")
        info_layout.addWidget(self.info_title)

        self.info_rows: Dict[str, QLabel] = {}
        for key, label in [("folder", "Folder"), ("images", "Images"), ("range", "Distance range"), ("format", "Format")]:
            row = QHBoxLayout()
            l = QLabel(label)
            set_class(l, "muted")
            v = QLabel("—")
            set_class(v, "mono")
            row.addWidget(l)
            row.addStretch(1)
            row.addWidget(v)
            info_layout.addLayout(row)
            self.info_rows[key] = v
        layout.addWidget(info_frame)

        ref_frame = QFrame()
        set_class(ref_frame, "card")
        ref_layout = QVBoxLayout(ref_frame)
        ref_layout.setContentsMargins(16, 14, 16, 14)
        ref_layout.setSpacing(10)

        ref_header = QHBoxLayout()
        ref_title = QLabel("Result")
        set_class(ref_title, "card-title")
        ref_header.addWidget(ref_title)
        ref_header.addStretch(1)
        self.method_combo = QComboBox()
        self.method_combo.addItems([METHOD_LABELS[k] for k in METHOD_ORDER])
        self.method_combo.currentIndexChanged.connect(self._on_method_combo_changed)
        ref_header.addWidget(self.method_combo)
        self.sweep_btn = QPushButton("Run full sweep")
        set_class(self.sweep_btn, "primary")
        self.sweep_btn.clicked.connect(self._on_run_sweep)
        ref_header.addWidget(self.sweep_btn)
        ref_layout.addLayout(ref_header)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.stat_peak = StatCard("Peak index")
        self.stat_images_read = StatCard("Images read")
        self.stat_nv = StatCard("Peak NV")
        self.stat_time = StatCard("Scan time")
        grid.addWidget(self.stat_peak, 0, 0)
        grid.addWidget(self.stat_images_read, 0, 1)
        grid.addWidget(self.stat_nv, 1, 0)
        grid.addWidget(self.stat_time, 1, 1)
        ref_layout.addLayout(grid)

        self.ref_note = QLabel("Not scanned yet — click “Run full sweep”.")
        set_class(self.ref_note, "dim")
        self.ref_note.setWordWrap(True)
        ref_layout.addWidget(self.ref_note)

        actions_row = QHBoxLayout()
        search_btn = QPushButton("Search this dataset")
        set_class(search_btn, "primary")
        search_btn.clicked.connect(lambda: self.navigate_requested.emit("focus_search"))
        compare_btn = QPushButton("Compare")
        compare_btn.clicked.connect(lambda: self.navigate_requested.emit("comparison"))
        actions_row.addWidget(search_btn, 1)
        actions_row.addWidget(compare_btn, 1)
        ref_layout.addLayout(actions_row)

        layout.addWidget(ref_frame)
        layout.addStretch(1)
        return container

    # -- actions -------------------------------------------------------

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if folder:
            self.path_edit.setText(folder)

    def _on_load(self) -> None:
        text = self.path_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "No folder", "Choose a dataset folder first.")
            return
        root = Path(text)
        if not root.is_dir():
            QMessageBox.critical(self, "Folder not found", f"{root} is not a folder.")
            return

        folders = discover_dataset_folders(root)
        if not folders:
            QMessageBox.critical(self, "No images found", f"No supported images found under:\n{root}")
            return

        errors = []
        loaded_any = False
        for folder in folders:
            try:
                self.state.add_dataset(folder, display_name_for(folder))
                loaded_any = True
            except (FileNotFoundError, ValueError, IOError) as exc:
                errors.append(str(exc))
        if errors and not loaded_any:
            QMessageBox.critical(self, "Could not load dataset", "\n".join(errors))
        elif errors:
            QMessageBox.warning(self, "Some folders skipped", "\n".join(errors))

    def _current_method(self) -> str:
        return METHOD_ORDER[self.method_combo.currentIndex()]

    def _on_method_combo_changed(self, _index: int) -> None:
        self.sweep_btn.setText(f"Run {METHOD_LABELS[self._current_method()].lower()}")
        loaded = self.state.active_dataset()
        if loaded is not None and loaded.sweep_result is not None and self._current_method() == "full_sweep":
            self._show_sweep_stats(loaded.sweep_result)
        else:
            self._clear_sweep_stats()

    def _on_run_sweep(self) -> None:
        loaded = self.state.active_dataset()
        if loaded is None:
            QMessageBox.warning(self, "No dataset", "Load and select a dataset first.")
            return
        method = self._current_method()
        self.sweep_btn.setEnabled(False)
        self.method_combo.setEnabled(False)
        self.sweep_btn.setText("Running…")
        if method == "gradient_ascent":
            start = int(np.random.default_rng().integers(0, len(loaded.dataset)))
            args = (loaded.dataset, self.state.settings, start)
        else:
            args = (loaded.dataset, self.state.settings)
        worker = SearchWorker(METHOD_FUNCS[method], args, {})
        worker.finished_ok.connect(lambda result, name=loaded.name, m=method: self._on_sweep_done(name, result, m))
        worker.failed.connect(self._on_sweep_failed)
        self._sweep_worker = worker
        worker.start()

    def _on_sweep_done(self, dataset_name: str, result: AutofocusResult, method: str) -> None:
        self.sweep_btn.setEnabled(True)
        self.method_combo.setEnabled(True)
        self.sweep_btn.setText(f"Run {METHOD_LABELS[method].lower()}")
        if method == "full_sweep":
            # Only a full sweep is guaranteed to have read every image, so only
            # it is kept as this dataset's reference peak (card badge, Focus
            # Search reference curve, Comparison baseline).
            self.state.set_sweep_result(dataset_name, result)
        else:
            self.state.record_run(result, dataset_name, f"{METHOD_LABELS[method]} — {dataset_name}")
            if dataset_name == self.state.active_dataset_name:
                self._show_sweep_stats(result)
                self.preview.show_index(result.coarse_index)

    def _on_sweep_failed(self, message: str) -> None:
        self.sweep_btn.setEnabled(True)
        self.method_combo.setEnabled(True)
        self.sweep_btn.setText(f"Run {METHOD_LABELS[self._current_method()].lower()}")
        QMessageBox.critical(self, "Search failed", message)

    # -- state reactions -------------------------------------------------

    def _refresh_cards(self) -> None:
        for i in reversed(range(self._cards_row.count())):
            item = self._cards_row.takeAt(i)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards = {}

        for name, loaded in self.state.datasets.items():
            card = DatasetCard(loaded)
            card.clicked.connect(self.state.set_active_dataset)
            card.set_active(name == self.state.active_dataset_name)
            if loaded.sweep_result is not None:
                card.set_sweep_result(loaded.sweep_result)
            self._cards_row.addWidget(card)
            self._cards[name] = card
        self._cards_row.addStretch(1)

        total_images = sum(len(loaded.dataset) for loaded in self.state.datasets.values())
        n = len(self.state.datasets)
        if n == 0:
            self.subtitle.setText("No datasets loaded")
        else:
            self.subtitle.setText(f"{n} dataset{'s' if n != 1 else ''} loaded · {total_images} images total")

    def _on_active_changed(self, name: str) -> None:
        for key, card in self._cards.items():
            card.set_active(key == name)

        loaded = self.state.active_dataset()
        if loaded is None:
            return
        self.preview.set_dataset(loaded.dataset)
        self.thumbnails.set_dataset(loaded.dataset)
        self.info_title.setText(loaded.name)
        self.info_rows["folder"].setText(str(loaded.path))
        self.info_rows["images"].setText(str(len(loaded.dataset)))
        lo = loaded.dataset.position_mm(0)
        hi = loaded.dataset.position_mm(len(loaded.dataset) - 1)
        self.info_rows["range"].setText(f"{lo:.3f} to {hi:.3f} mm")
        self.info_rows["format"].setText(loaded.dataset.files[0].suffix.upper().lstrip("."))

        if loaded.sweep_result is not None and self._current_method() == "full_sweep":
            self._show_sweep_stats(loaded.sweep_result)
        else:
            self._clear_sweep_stats()

    def _on_sweep_recorded(self, dataset_name: str) -> None:
        if dataset_name in self._cards:
            self._cards[dataset_name].set_sweep_result(self.state.datasets[dataset_name].sweep_result)
        if dataset_name == self.state.active_dataset_name:
            result = self.state.datasets[dataset_name].sweep_result
            if self._current_method() == "full_sweep":
                self._show_sweep_stats(result)
            self.preview.show_index(result.coarse_index)

    def _show_sweep_stats(self, result: AutofocusResult) -> None:
        self.stat_peak.set_value(f"#{result.coarse_index}", "accent-orange")
        self.stat_images_read.set_value(f"{result.images_read}/{result.total_images} ({result.percent_read:.0f}%)")
        self.stat_nv.set_value(f"{result.coarse_focus:.3f}")
        self.stat_time.set_value(f"{result.elapsed_s * 1000:.0f} ms")
        if result.method == "full_sweep":
            self.ref_note.setText(
                f"Found {count_local_maxima(result)} local maxima across the full scan "
                f"({result.images_read}/{result.total_images} images) — set as this dataset's reference peak."
            )
        else:
            self.ref_note.setText(
                f"{METHOD_LABELS[result.method]} converged without scanning the full dataset "
                f"({result.images_read}/{result.total_images} images read)."
            )

    def _clear_sweep_stats(self) -> None:
        self.stat_peak.set_value("—")
        self.stat_images_read.set_value("—")
        self.stat_nv.set_value("—")
        self.stat_time.set_value("—")
        self.ref_note.setText(f"Not scanned yet — click “Run {METHOD_LABELS[self._current_method()].lower()}”.")
