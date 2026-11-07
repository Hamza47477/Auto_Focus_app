"""Settings form mapped 1:1 to AutofocusSettings, in 3 groups (section 2.2).

Only the fields the spec's Algorithms page explicitly lists are exposed:
NV preprocessing, Search (gradient ascent), Fine position. ``d_min_mm`` /
``step_mm`` and ``use_overshoot_control`` are intentionally not exposed here
-- the first two are fixed at the package defaults per section 2.1's
distance-range formula, and overshoot control is presented as a fixed part
of the main method's pipeline (section 2.2), not a toggle.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from autofocus import AutofocusSettings

from .common import set_class


class SettingsForm(QWidget):
    """Live-editing form for :class:`AutofocusSettings`, shared by every page."""

    settings_changed = Signal(object)  # AutofocusSettings

    def __init__(self, initial: AutofocusSettings, parent=None, horizontal: bool = False):
        super().__init__(parent)
        self._settings = initial
        self._updating = False

        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(0, 0, 0, 0)

        # Side by side when there's a full page width to use (Algorithms page,
        # now that Settings sits below the pipeline instead of beside it);
        # stacked otherwise (e.g. if reused in a narrower panel).
        preprocessing_group = self._build_preprocessing_group()
        search_group = self._build_search_group()
        fit_group = self._build_fit_group()
        groups = QHBoxLayout() if horizontal else QVBoxLayout()
        groups.setSpacing(14)
        if horizontal:
            # Qt vertically *centers* a widget within a QHBoxLayout row by
            # default when its own size policy isn't Expanding -- without
            # Qt.AlignTop, the two shorter group cards would float in the
            # middle of the row instead of lining up with the tallest one.
            groups.addWidget(preprocessing_group, 0, Qt.AlignTop)
            groups.addWidget(search_group, 0, Qt.AlignTop)
            groups.addWidget(fit_group, 0, Qt.AlignTop)
        else:
            groups.addWidget(preprocessing_group)
            groups.addWidget(search_group)
            groups.addWidget(fit_group)
        root.addLayout(groups)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        self.reset_btn = QPushButton("Reset to defaults")
        set_class(self.reset_btn, "ghost")
        self.reset_btn.clicked.connect(self._on_reset)
        reset_row.addWidget(self.reset_btn)
        root.addLayout(reset_row)
        if not horizontal:
            root.addStretch(1)

        self._load_from_settings(initial)

    # -- group builders ------------------------------------------------

    def _card(self, title: str):
        frame = QFrame()
        set_class(frame, "card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        label = QLabel(title)
        set_class(label, "card-title")
        layout.addWidget(label)
        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft)
        layout.addLayout(form)
        return frame, form

    def _build_preprocessing_group(self) -> QFrame:
        frame, form = self._card("NV preprocessing")

        self.chk_crop = QCheckBox("Crop bottom info bar")
        self.spin_crop_px = QSpinBox()
        self.spin_crop_px.setRange(0, 500)
        self.spin_crop_px.setSuffix(" px")
        form.addRow(self.chk_crop)
        form.addRow("Crop height", self.spin_crop_px)

        self.chk_denoise = QCheckBox("Denoise before NV")
        self.combo_denoise = QComboBox()
        self.combo_denoise.addItems(["gaussian", "median"])
        self.spin_ksize = QSpinBox()
        self.spin_ksize.setRange(1, 31)
        self.spin_ksize.setSingleStep(2)
        form.addRow(self.chk_denoise)
        form.addRow("Blur method", self.combo_denoise)
        form.addRow("Kernel size", self.spin_ksize)

        for widget, signal in [
            (self.chk_crop, "toggled"),
            (self.spin_crop_px, "valueChanged"),
            (self.chk_denoise, "toggled"),
            (self.combo_denoise, "currentTextChanged"),
            (self.spin_ksize, "valueChanged"),
        ]:
            getattr(widget, signal).connect(self._on_any_change)

        return frame

    def _build_search_group(self) -> QFrame:
        frame, form = self._card("Search (gradient ascent)")

        self.spin_alpha0 = QSpinBox()
        self.spin_alpha0.setRange(0, 1000)
        self.spin_alpha0.setSpecialValueText("auto (N // 10)")
        form.addRow("Initial step α0", self.spin_alpha0)

        self.chk_stochastic = QCheckBox("Stochastic step factor")
        self.spin_u_low = QDoubleSpinBox()
        self.spin_u_low.setRange(0.0, 5.0)
        self.spin_u_low.setSingleStep(0.1)
        self.spin_u_high = QDoubleSpinBox()
        self.spin_u_high.setRange(0.0, 5.0)
        self.spin_u_high.setSingleStep(0.1)
        form.addRow(self.chk_stochastic)
        bounds_row = QHBoxLayout()
        bounds_row.addWidget(self.spin_u_low)
        bounds_row.addWidget(QLabel("to"))
        bounds_row.addWidget(self.spin_u_high)
        form.addRow("u bounds", bounds_row)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 999_999)
        self.spin_seed.setSpecialValueText("none")
        form.addRow("Random seed", self.spin_seed)

        self.spin_shrink = QDoubleSpinBox()
        self.spin_shrink.setRange(0.05, 0.95)
        self.spin_shrink.setSingleStep(0.05)
        form.addRow("Overshoot shrink factor", self.spin_shrink)

        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 10_000)
        form.addRow("Max iterations", self.spin_max_iter)

        self.chk_neighbor = QCheckBox("Neighbor check")
        form.addRow(self.chk_neighbor)

        self.chk_safety = QCheckBox("Safety check (sparse samples)")
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 64)
        self.spin_restarts = QSpinBox()
        self.spin_restarts.setRange(0, 20)
        form.addRow(self.chk_safety)
        form.addRow("K samples", self.spin_k)
        form.addRow("Max restarts", self.spin_restarts)

        for widget, signal in [
            (self.spin_alpha0, "valueChanged"),
            (self.chk_stochastic, "toggled"),
            (self.spin_u_low, "valueChanged"),
            (self.spin_u_high, "valueChanged"),
            (self.spin_seed, "valueChanged"),
            (self.spin_shrink, "valueChanged"),
            (self.spin_max_iter, "valueChanged"),
            (self.chk_neighbor, "toggled"),
            (self.chk_safety, "toggled"),
            (self.spin_k, "valueChanged"),
            (self.spin_restarts, "valueChanged"),
        ]:
            getattr(widget, signal).connect(self._on_any_change)

        return frame

    def _build_fit_group(self) -> QFrame:
        frame, form = self._card("Fine position (quadratic fit)")
        self.spin_window = QSpinBox()
        self.spin_window.setRange(2, 20)
        form.addRow("Fit window ±w", self.spin_window)
        self.spin_window.valueChanged.connect(self._on_any_change)
        return frame

    # -- load/save -------------------------------------------------------

    def _load_from_settings(self, settings: AutofocusSettings) -> None:
        self._updating = True
        self.chk_crop.setChecked(settings.do_crop_bottom)
        self.spin_crop_px.setValue(settings.crop_bottom_px)
        self.chk_denoise.setChecked(settings.do_denoise)
        self.combo_denoise.setCurrentText(settings.denoise_method)
        self.spin_ksize.setValue(settings.denoise_ksize)

        self.spin_alpha0.setValue(settings.alpha0 if settings.alpha0 is not None else 0)
        self.chk_stochastic.setChecked(settings.use_stochastic_factor)
        self.spin_u_low.setValue(settings.stochastic_low)
        self.spin_u_high.setValue(settings.stochastic_high)
        self.spin_seed.setValue(settings.seed if settings.seed is not None else -1)
        self.spin_shrink.setValue(settings.overshoot_shrink)
        self.spin_max_iter.setValue(settings.max_iterations)
        self.chk_neighbor.setChecked(settings.use_neighbor_check)
        self.chk_safety.setChecked(settings.use_safety_check)
        self.spin_k.setValue(settings.safety_samples_k)
        self.spin_restarts.setValue(settings.max_restarts)

        self.spin_window.setValue(settings.fit_half_window)
        self._updating = False

    def current_settings(self) -> AutofocusSettings:
        """Build a fresh :class:`AutofocusSettings` from the current widget values."""
        return AutofocusSettings(
            d_min_mm=self._settings.d_min_mm,
            step_mm=self._settings.step_mm,
            do_crop_bottom=self.chk_crop.isChecked(),
            crop_bottom_px=self.spin_crop_px.value(),
            do_denoise=self.chk_denoise.isChecked(),
            denoise_method=self.combo_denoise.currentText(),
            denoise_ksize=self.spin_ksize.value(),
            alpha0=(self.spin_alpha0.value() or None),
            use_stochastic_factor=self.chk_stochastic.isChecked(),
            stochastic_low=self.spin_u_low.value(),
            stochastic_high=self.spin_u_high.value(),
            use_overshoot_control=True,
            overshoot_shrink=self.spin_shrink.value(),
            max_iterations=self.spin_max_iter.value(),
            use_neighbor_check=self.chk_neighbor.isChecked(),
            use_safety_check=self.chk_safety.isChecked(),
            safety_samples_k=self.spin_k.value(),
            max_restarts=self.spin_restarts.value(),
            fit_half_window=self.spin_window.value(),
            seed=(self.spin_seed.value() if self.spin_seed.value() >= 0 else None),
        )

    def _on_any_change(self, *_args) -> None:
        if self._updating:
            return
        self._settings = self.current_settings()
        self.settings_changed.emit(self._settings)

    def _on_reset(self) -> None:
        defaults = AutofocusSettings(d_min_mm=self._settings.d_min_mm, step_mm=self._settings.step_mm)
        self._settings = defaults
        self._load_from_settings(defaults)
        self.settings_changed.emit(defaults)
