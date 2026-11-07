"""Main window: left sidebar navigation, page stack, shared status bar."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import APP_NAME, APP_VERSION
from .pages.algorithms import AlgorithmsPage
from .pages.comparison import ComparisonPage
from .pages.datasets import DatasetsPage
from .pages.focus_search import FocusSearchPage
from .pages.walkthroughs import WalkthroughsPage
from .state import AppState
from .widgets.common import set_class

PAGES = [
    ("datasets", "Datasets"),
    ("algorithms", "Algorithms"),
    ("focus_search", "Focus Search"),
    ("comparison", "Comparison"),
    ("walkthroughs", "Walkthroughs"),
]

_WORKER_STATE_ACCENT = {
    "running": "accent-blue",
    "done": "accent-green",
    "stopped": "accent-orange",
    "error": "accent-red",
    "idle": "accent-green",
}


class MainWindow(QMainWindow):
    """Top-level window: sidebar + page stack + status bar, wired to one AppState."""

    @staticmethod
    def _min_size_for_screen(width: int, height: int) -> tuple[int, int]:
        """Clamp the declared minimum size to what the actual screen can show.

        ``QMainWindow.setMinimumSize`` is a hard floor: no later ``resize()``
        (manual or programmatic) can ever go below it. A fixed 1280x800 floor
        on a screen with less available height than that (e.g. a smaller
        laptop panel, or one with a tall taskbar/dock) would make the window
        permanently taller than the screen with no way to shrink it back into
        view. Capping the floor itself to ``availableGeometry()`` keeps the
        app always resizable to fit whatever screen it's running on.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return width, height
        available = screen.availableGeometry()
        return min(width, available.width()), min(height, available.height())

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(*self._min_size_for_screen(1280, 800))

        self.state = AppState()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.pages = {
            "datasets": DatasetsPage(self.state),
            "algorithms": AlgorithmsPage(self.state),
            "focus_search": FocusSearchPage(self.state),
            "comparison": ComparisonPage(self.state),
            "walkthroughs": WalkthroughsPage(self.state),
        }
        for key, _label in PAGES:
            self.stack.addWidget(self.pages[key])
        root.addWidget(self.stack, 1)

        self.pages["datasets"].navigate_requested.connect(self._select_page)

        self._build_status_bar()
        self.state.status_changed.connect(self.status_text.setText)
        self.state.worker_state_changed.connect(self._on_worker_state_changed)
        self.state.active_dataset_changed.connect(self._update_active_dataset_card)
        self.state.datasets_changed.connect(self._update_active_dataset_card)

        self._select_page("datasets")

    # -- sidebar ------------------------------------------------------

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(4)

        brand_row = QHBoxLayout()
        icon = QLabel("◎")
        icon.setStyleSheet("font-size: 18px; color: #5aa9ff;")
        brand_col = QVBoxLayout()
        brand_col.setSpacing(0)
        name = QLabel(APP_NAME)
        name.setObjectName("BrandTitle")
        version = QLabel(f"v{APP_VERSION} · PySide6")
        version.setObjectName("BrandSubtitle")
        brand_col.addWidget(name)
        brand_col.addWidget(version)
        brand_row.addWidget(icon)
        brand_row.addLayout(brand_col)
        brand_row.addStretch(1)
        layout.addLayout(brand_row)
        layout.addSpacing(18)

        section = QLabel("WORKSPACE")
        section.setObjectName("SidebarSectionLabel")
        layout.addWidget(section)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = {}
        for key, label in PAGES:
            btn = QPushButton(label)
            btn.setObjectName("NavItem")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, k=key: self._select_page(k))
            layout.addWidget(btn)
            self.nav_group.addButton(btn)
            self.nav_buttons[key] = btn

        layout.addStretch(1)

        self.active_card = QFrame()
        self.active_card.setObjectName("ActiveDatasetCard")
        active_layout = QVBoxLayout(self.active_card)
        active_layout.setContentsMargins(12, 10, 12, 10)
        active_layout.setSpacing(2)
        active_label = QLabel("ACTIVE DATASET")
        set_class(active_label, "dim")
        self.active_name_label = QLabel("None")
        set_class(self.active_name_label, "card-title")
        self.active_info_label = QLabel("—")
        set_class(self.active_info_label, "muted")
        active_layout.addWidget(active_label)
        active_layout.addWidget(self.active_name_label)
        active_layout.addWidget(self.active_info_label)
        layout.addWidget(self.active_card)

        settings_btn = QPushButton("⚙  Settings")
        settings_btn.setObjectName("NavItem")
        settings_btn.clicked.connect(lambda: self._select_page("algorithms"))
        layout.addWidget(settings_btn)

        return sidebar

    # -- status bar ------------------------------------------------------

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.status_dot = QLabel("●")
        set_class(self.status_dot, "accent-green")
        bar.addWidget(self.status_dot)
        self.status_text = QLabel("Ready")
        bar.addWidget(self.status_text)
        right_label = QLabel("NumPy · OpenCV · Matplotlib")
        set_class(right_label, "dim")
        bar.addPermanentWidget(right_label)

    # -- navigation -----------------------------------------------------

    def _select_page(self, key: str) -> None:
        if key not in self.pages:
            return
        self.stack.setCurrentWidget(self.pages[key])
        self.nav_buttons[key].setChecked(True)

    def _on_worker_state_changed(self, worker_state: str) -> None:
        set_class(self.status_dot, _WORKER_STATE_ACCENT.get(worker_state, "muted"))

    def _update_active_dataset_card(self, *_args) -> None:
        loaded = self.state.active_dataset()
        if loaded is None:
            self.active_name_label.setText("None")
            self.active_info_label.setText("—")
            return
        self.active_name_label.setText(loaded.name)
        self.active_info_label.setText(f"{len(loaded.dataset)} images")
