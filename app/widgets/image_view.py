"""Image preview and the per-image evaluation strip.

Both widgets display dataset images directly with OpenCV/Qt -- they never
go through ``autofocus.FocusEvaluator``, so browsing images for display
never counts as a focus evaluation (section 2.1 of the spec).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QMouseEvent, QPainter, QPaintEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from autofocus import ImageDataset
from autofocus.focus import normalized_variance

from ..colors import PHASE_COLORS
from .common import set_class


def load_pixmap(path: Path) -> QPixmap:
    """Decode one image for display (any extension ``autofocus`` accepts)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image for preview: {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())  # detach from the numpy buffer


def compute_preview_stats(path: Path) -> Tuple[float, float, float]:
    """(mean, variance, normalized variance) of one image, for display only.

    Decodes the image a second time in grayscale -- deliberately bypassing
    ``autofocus.FocusEvaluator`` so that showing these numbers next to the
    preview never counts as a focus evaluation or touches its cache.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"Could not read image for preview: {path}")
    gray_f = gray.astype(np.float64)
    mu = float(gray_f.mean())
    variance = float(np.mean((gray_f - mu) ** 2))
    return mu, variance, normalized_variance(gray_f)


class ImagePreview(QWidget):
    """One dataset image plus an index slider/spinbox and a position (mm) label.

    When ``show_stats`` is on, also emits :attr:`stats_changed` with
    ``(mu, variance, normalized_variance)`` for the displayed image, computed
    directly (not through the focus evaluator) purely for display.
    """

    index_changed = Signal(int)
    stats_changed = Signal(float, float, float)

    def __init__(self, parent=None, show_stats: bool = False):
        super().__init__(parent)
        self._dataset: Optional[ImageDataset] = None
        self._pixmap: Optional[QPixmap] = None
        self._show_stats = show_stats

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._image_label = QLabel("No image loaded")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumHeight(260)
        set_class(self._image_label, "card-flat")
        layout.addWidget(self._image_label, 1)

        info_row = QHBoxLayout()
        self._info_label = QLabel("—")
        set_class(self._info_label, "mono muted")
        info_row.addWidget(self._info_label)
        info_row.addStretch(1)
        layout.addLayout(info_row)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)
        self._slider = QSlider(Qt.Horizontal)
        self._spin = QSpinBox()
        self._spin.setFixedWidth(80)
        slider_row.addWidget(QLabel("Image"))
        slider_row.addWidget(self._slider, 1)
        slider_row.addWidget(self._spin)
        layout.addLayout(slider_row)

        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def set_dataset(self, dataset: Optional[ImageDataset]) -> None:
        self._dataset = dataset
        n = len(dataset) if dataset is not None else 0
        for w in (self._slider, self._spin):
            w.blockSignals(True)
            w.setMinimum(0)
            w.setMaximum(max(0, n - 1))
            w.blockSignals(False)
        if dataset is not None and n > 0:
            self.show_index(0)
        else:
            self._pixmap = None
            self._image_label.setText("No image loaded")
            self._info_label.setText("—")

    def show_index(self, index: int) -> None:
        if self._dataset is None or len(self._dataset) == 0:
            return
        index = max(0, min(len(self._dataset) - 1, index))
        for w, v in ((self._slider, index), (self._spin, index)):
            w.blockSignals(True)
            w.setValue(index)
            w.blockSignals(False)
        try:
            self._pixmap = load_pixmap(self._dataset.path_for(index))
        except IOError as exc:
            self._image_label.setText(str(exc))
            self._pixmap = None
        else:
            self._rescale()
        position = self._dataset.position_mm(index)
        self._info_label.setText(f"#{index}  ·  {position:.4f} mm")
        self.index_changed.emit(index)

        if self._show_stats:
            try:
                mu, variance, nv = compute_preview_stats(self._dataset.path_for(index))
            except IOError:
                pass
            else:
                self.stats_changed.emit(mu, variance, nv)

    def current_index(self) -> int:
        return self._spin.value()

    def _on_slider(self, value: int) -> None:
        self.show_index(value)

    def _on_spin(self, value: int) -> None:
        self.show_index(value)

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._rescale()


# QPainter needs QColor objects; built once from the shared hex palette.
_PHASE_QCOLORS: Dict[str, QColor] = {phase: QColor(hex_) for phase, hex_ in PHASE_COLORS.items()}


class EvaluationStrip(QWidget):
    """One cell per dataset image, colored by how/whether it was evaluated.

    section 2.3: "a strip of small cells (one per image) colored: start /
    ascent / safety sample / fit / best / not read."
    """

    cell_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n = 0
        self._states: Dict[int, str] = {}
        self._best_index: Optional[int] = None
        self.setMinimumHeight(26)
        self.setMaximumHeight(26)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def set_total(self, n: int) -> None:
        self._n = n
        self._states = {}
        self._best_index = None
        self.update()

    def mark(self, index: int, phase: str) -> None:
        if phase in _PHASE_QCOLORS:
            self._states[index] = phase
        self.update()

    def set_best(self, index: Optional[int]) -> None:
        self._best_index = index
        self.update()

    def clear(self) -> None:
        self._states = {}
        self._best_index = None
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        if self._n <= 0:
            return
        painter = QPainter(self)
        w, h = self.width(), self.height()
        cell_w = w / self._n
        for i in range(self._n):
            color = _PHASE_QCOLORS.get(self._states.get(i, "not_read"), _PHASE_QCOLORS["not_read"])
            if i == self._best_index:
                color = _PHASE_QCOLORS["best"]
            x0, x1 = int(i * cell_w), int((i + 1) * cell_w)
            painter.fillRect(x0, 1, max(1, x1 - x0 - 1), h - 2, color)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._n <= 0:
            return
        i = int(event.position().x() / (self.width() / self._n))
        self.cell_clicked.emit(max(0, min(self._n - 1, i)))


class ThumbnailStrip(QWidget):
    """A row of evenly-spaced thumbnails across the dataset, click to jump.

    Display only -- decodes each thumbnail directly, bypassing the focus
    evaluator, same as :class:`ImagePreview`.
    """

    thumbnail_clicked = Signal(int)

    def __init__(self, count: int = 9, parent=None):
        super().__init__(parent)
        self._count = count
        self._buttons: List[QToolButton] = []
        self._connected = [False] * count

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for _ in range(count):
            btn = QToolButton()
            btn.setFixedSize(68, 80)
            btn.setIconSize(QSize(60, 60))
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setVisible(False)
            layout.addWidget(btn)
            self._buttons.append(btn)
        layout.addStretch(1)

    def set_dataset(self, dataset: Optional[ImageDataset]) -> None:
        for i, btn in enumerate(self._buttons):
            btn.setVisible(False)
            if self._connected[i]:
                btn.clicked.disconnect()
                self._connected[i] = False

        if dataset is None or len(dataset) == 0:
            return

        n = len(dataset)
        indices = sorted(set(np.linspace(0, n - 1, min(self._count, n)).round().astype(int).tolist()))
        for i, (btn, index) in enumerate(zip(self._buttons, indices)):
            try:
                pixmap = load_pixmap(dataset.path_for(index)).scaled(
                    60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                btn.setIcon(QIcon(pixmap))
            except IOError:
                btn.setIcon(QIcon())
            btn.setText(f"#{index}")
            btn.setVisible(True)
            btn.clicked.connect(lambda _checked=False, idx=int(index): self.thumbnail_clicked.emit(idx))
            self._connected[i] = True
