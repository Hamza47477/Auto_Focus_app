"""Algorithms page (section 2.2): formula, the 3 methods, the pipeline, settings.

Purely informational + the shared settings form -- it does not run any
search itself.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..state import AppState
from ..widgets.common import set_class
from ..widgets.settings_form import SettingsForm

METHOD_CARDS = [
    {
        "tag": "MAIN · FROM START IMAGE",
        "tag_class": "badge-orange",
        "title": "Gradient ascent",
        "how": "Climbs from the start image using the slope between neighboring "
        "images (Eq. 6.13, centered difference) and a decreasing step size "
        "(Eq. 6.16), until it settles on a peak.",
        "solves_label": "Solves",
        "solves": "Reads a small, bounded number of images — never the whole stack.",
        "risk_label": "Risk",
        "risk": "A plain hill-climb can stop at a local maximum; overshoot control, "
        "a neighbor check and sparse safety sampling guard against it.",
    },
    {
        "tag": "COMPARISON · IGNORES START",
        "tag_class": "badge-purple",
        "title": "Golden-section search",
        "how": "Shrinks the search bracket by the golden ratio (≈0.618) each "
        "iteration, reusing one of the two probes every step.",
        "solves_label": "Solves",
        "solves": "O(log N) reads — typically a dozen or so images for a 200-image stack.",
        "risk_label": "Risk",
        "risk": "Assumes the focus curve has one peak; if the first probes land "
        "near a smaller local maximum, it can lock onto the wrong hill.",
    },
    {
        "tag": "REFERENCE · IGNORES START",
        "tag_class": "badge-grey",
        "title": "Full sweep",
        "how": "Computes the normalized variance for every image, in order, "
        "and keeps a running best.",
        "solves_label": "Solves",
        "solves": "Always finds the true global maximum — the ground truth the "
        "other two methods are checked against.",
        "risk_label": "Cost",
        "risk": "Reads all N images. O(N).",
    },
]

PIPELINE_STEPS = [
    "Start image",
    "Gradient ascent\n(stochastic step + overshoot control)",
    "Neighbor check",
    "Safety check\n(sparse samples)",
    "Quadratic fit",
    "Coarse + fine\nposition (mm)",
]


class MethodCard(QFrame):
    def __init__(self, spec: dict, parent=None):
        super().__init__(parent)
        set_class(self, "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        tag = QLabel(spec["tag"])
        set_class(tag, f"badge {spec['tag_class']}")
        tag.setFixedWidth(tag.sizeHint().width() + 4)
        layout.addWidget(tag, 0, Qt.AlignLeft)

        title = QLabel(spec["title"])
        set_class(title, "card-title")
        layout.addWidget(title)

        how = QLabel(spec["how"])
        how.setWordWrap(True)
        set_class(how, "muted")
        layout.addWidget(how)

        layout.addSpacing(4)
        for label_key, text_key, accent in [("solves_label", "solves", "accent-green"), ("risk_label", "risk", "accent-red")]:
            row_label = QLabel(spec[label_key])
            set_class(row_label, f"dim {accent}")
            row_text = QLabel(spec[text_key])
            row_text.setWordWrap(True)
            set_class(row_text, "muted")
            layout.addWidget(row_label)
            layout.addWidget(row_text)
        layout.addStretch(1)


class PipelineStep(QFrame):
    def __init__(self, number: int, text: str, parent=None):
        super().__init__(parent)
        set_class(self, "card-flat")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        badge = QLabel(str(number))
        set_class(badge, "badge badge-blue")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(20, 20)
        layout.addWidget(badge)
        label = QLabel(text)
        label.setWordWrap(True)
        set_class(label, "muted")
        layout.addWidget(label, 1)


class AlgorithmsPage(QWidget):
    """"Algorithms" screen: formula, method cards, pipeline, settings form."""

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
        root = QVBoxLayout(content)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        root.addLayout(self._build_header())

        methods_row = QHBoxLayout()
        methods_row.setSpacing(12)
        for spec in METHOD_CARDS:
            methods_row.addWidget(MethodCard(spec), 1)
        root.addLayout(methods_row)

        # Pipeline (full width, horizontal steps) above Settings (full width)
        # rather than side by side -- side by side needs more width than the
        # app's minimum window size (1280px) has to give both at once.
        root.addWidget(self._build_pipeline_card())
        root.addWidget(self._build_settings_panel())
        root.addStretch(1)

    def _build_header(self) -> QVBoxLayout:
        root = QVBoxLayout()
        root.setSpacing(10)

        title = QLabel("Algorithms")
        set_class(title, "page-title")
        subtitle = QLabel("Gradient ascent from a start image, and two reference methods")
        set_class(subtitle, "page-subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)
        return root

    def _build_pipeline_card(self) -> QFrame:
        frame = QFrame()
        set_class(frame, "card-selected")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Main method pipeline")
        set_class(title, "card-title")
        header.addWidget(title)
        header.addStretch(1)
        badge = QLabel("Gradient ascent")
        set_class(badge, "badge badge-orange")
        header.addWidget(badge)
        layout.addLayout(header)

        grid = QHBoxLayout()
        grid.setSpacing(8)
        for i, step in enumerate(PIPELINE_STEPS, start=1):
            grid.addWidget(PipelineStep(i, step), 1)
        layout.addLayout(grid)

        note = QLabel(
            "Golden-section and full sweep ignore the start image and search the "
            "full index range directly."
        )
        note.setWordWrap(True)
        set_class(note, "dim")
        layout.addWidget(note)
        layout.addStretch(1)
        return frame

    def _build_settings_panel(self) -> QFrame:
        frame = QFrame()
        set_class(frame, "card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        title = QLabel("Settings")
        set_class(title, "card-title")
        layout.addWidget(title)

        self.settings_form = SettingsForm(self.state.settings, horizontal=True)
        self.settings_form.settings_changed.connect(self.state.update_settings)
        layout.addWidget(self.settings_form)
        return frame
