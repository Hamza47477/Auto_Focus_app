"""Matplotlib focus-curve plot, embedded via ``backend_qtagg``.

Chosen over pyqtgraph because the app's plots (a results.csv worth of
points per run, three comparison line charts, replay curves) are small and
update once per evaluation rather than per animation frame -- matplotlib's Qt backend
is more than fast enough, and it is already a project dependency (used by
``experiments/make_report_figures.py``), so there is no need for a second
plotting library.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from ..colors import PHASE_COLORS, PHASE_LABELS

_FIG_BG = "#181b1f"
_AX_BG = "#10141a"
_GRID = "#22262c"
_SPINE = "#2a3038"
_TEXT = "#8b93a1"


def style_axes(ax, xlabel: str = "working distance (mm)", ylabel: str = "normalized variance") -> None:
    """Apply the app's dark-theme styling to a matplotlib Axes."""
    ax.set_facecolor(_AX_BG)
    for spine in ax.spines.values():
        spine.set_color(_SPINE)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.label.set_color("#c4cad4")
    ax.yaxis.label.set_color("#c4cad4")
    ax.title.set_color("#e9edf1")
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


def new_figure(figsize=(5, 4)) -> Figure:
    fig = Figure(figsize=figsize, facecolor=_FIG_BG)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.14)
    return fig


class _HoverCanvas(FigureCanvasQTAgg):
    """Shared hover-tooltip behavior: point at any marker/bar, see its data.

    Subclasses track their own plotted data and implement :meth:`_find_hit`;
    everything else (the tooltip artist, showing/hiding it, wiring the Qt
    mouse-move event) lives here once so every chart in the app gets it.

    The tooltip annotation is a matplotlib *artist*: ``Axes.clear()`` (called
    before every redraw) destroys it along with everything else on the axes,
    so it must be recreated -- not just repositioned -- after each clear via
    :meth:`_make_tooltip`.
    """

    def _init_hover(self, ax) -> None:
        self._hover_ax = ax
        self._tooltip = None
        self._make_tooltip()
        self.mpl_connect("motion_notify_event", self._on_hover_move)
        self.mpl_connect("figure_leave_event", lambda _e: self._hide_tooltip())

    def _make_tooltip(self) -> None:
        self._tooltip = self._hover_ax.annotate(
            "",
            xy=(0, 0),
            xytext=(14, 14),
            textcoords="offset points",
            fontsize=8,
            color="#e9edf1",
            bbox=dict(boxstyle="round,pad=0.4", fc="#20252c", ec=_SPINE, lw=0.8),
            zorder=10,
        )
        self._tooltip.set_visible(False)

    def _hide_tooltip(self) -> None:
        if self._tooltip is not None and self._tooltip.get_visible():
            self._tooltip.set_visible(False)
            self.draw_idle()

    def _show_tooltip(self, x: float, y: float, text: str) -> None:
        self._tooltip.xy = (x, y)
        self._tooltip.set_text(text)
        self._tooltip.set_visible(True)
        self.draw_idle()

    def _on_hover_move(self, event) -> None:
        if self._tooltip is None or event.inaxes != self._hover_ax:
            self._hide_tooltip()
            return
        hit = self._find_hit(event)
        if hit is None:
            self._hide_tooltip()
            return
        x, y, text = hit
        self._show_tooltip(x, y, text)

    def _find_hit(self, event) -> Optional[Tuple[float, float, str]]:
        raise NotImplementedError


class FocusCurveCanvas(_HoverCanvas):
    """Live-updating normalized-variance-vs-position plot for Focus Search / Walkthroughs.

    Points are added one at a time via :meth:`add_point` as a search's
    ``on_step`` callback fires; :meth:`set_fit` draws the quadratic fit
    (Eq. 6.10) once a search finishes.
    """

    def __init__(self, parent=None):
        self._figure = new_figure()
        super().__init__(self._figure)
        self.setParent(parent)
        self.ax = self._figure.add_subplot(111)
        style_axes(self.ax)
        self._fit_artists: List = []
        self._ref_artist = None
        self._points: List[Tuple[float, float, str]] = []
        self._init_hover(self.ax)
        self.draw_idle()

    def reset(self, normalized: bool = False) -> None:
        self.ax.clear()
        style_axes(self.ax, ylabel="NV / max NV" if normalized else "normalized variance")
        self._fit_artists = []
        self._ref_artist = None
        self._points = []
        self._make_tooltip()
        self.draw_idle()

    def set_reference_curve(self, positions: Sequence[float], values: Sequence[float]) -> None:
        if self._ref_artist is not None:
            try:
                self._ref_artist.remove()
            except ValueError:
                pass
        (self._ref_artist,) = self.ax.plot(positions, values, color="#5b6472", lw=1, ls="--", zorder=1)
        self.draw_idle()

    def add_point(self, position_mm: float, focus: float, phase: str, step_no: int, is_start: bool = False) -> None:
        color = PHASE_COLORS["start"] if is_start else PHASE_COLORS.get(phase, "#c4cad4")
        size = 40 if is_start else 22
        self.ax.scatter([position_mm], [focus], s=size, color=color, zorder=3, edgecolors=_FIG_BG, linewidths=0.6)
        self.ax.annotate(
            str(step_no), (position_mm, focus), fontsize=6, color=_TEXT, xytext=(3, 3), textcoords="offset points"
        )
        phase_label = "Start" if is_start else PHASE_LABELS.get(phase, phase.replace("_", " ").title())
        self._points.append(
            (position_mm, focus, f"{phase_label} · step {step_no}\n{focus:.4f} NV\n{position_mm:.4f} mm")
        )
        self.draw_idle()

    def set_fit(self, curve_positions: Sequence[float], curve_values: Sequence[float], fine_position: float) -> None:
        for artist in self._fit_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._fit_artists = []
        (line,) = self.ax.plot(curve_positions, curve_values, color=PHASE_COLORS["fit"], lw=1.5, zorder=2)
        vline = self.ax.axvline(fine_position, color=PHASE_COLORS["fit"], ls="--", lw=1, zorder=2)
        self._fit_artists = [line, vline]
        self.draw_idle()

    def scatter(
        self,
        positions: Sequence[float],
        values: Sequence[float],
        color: str,
        size: int = 30,
        marker: str = "o",
        label: str = "",
    ) -> None:
        """One-off scatter without per-point step numbers (used by the Comparison page).

        ``label`` identifies what kind of point this is (e.g. "Golden-section
        landing") and is shown, along with its value, in the hover tooltip.
        """
        self.ax.scatter(positions, values, s=size, color=color, marker=marker, zorder=3, edgecolors=_FIG_BG, linewidths=0.5)
        prefix = f"{label}\n" if label else ""
        for x, y in zip(positions, values):
            self._points.append((x, y, f"{prefix}{y:.4f} NV\n{x:.4f} mm"))
        self.draw_idle()

    def _find_hit(self, event) -> Optional[Tuple[float, float, str]]:
        if not self._points:
            return None
        threshold_px = 14
        best = None
        best_d2 = threshold_px ** 2
        for x, y, text in self._points:
            px, py = self.ax.transData.transform((x, y))
            d2 = (px - event.x) ** 2 + (py - event.y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (x, y, text)
        return best

    def save_png(self, path: str) -> None:
        self._figure.savefig(path, dpi=200, facecolor=self._figure.get_facecolor())


class LineChartCanvas(_HoverCanvas):
    """Per-category line+marker chart used by the Comparison page.

    One line per method, one marker per dataset -- used instead of grouped
    bars because the three methods' values routinely span two orders of
    magnitude (e.g. full sweep reads ~199 images, golden-section ~13): on a
    linear bar chart the smaller series flatten to near-invisible slivers.
    A log-scale line/point plot keeps every series readably sized and makes
    each method's trend across datasets easy to follow, matching the style
    of the other plots in the app (e.g. "Where each method landed").
    """

    def __init__(self, parent=None, figsize=(5, 3.2)):
        self._figure = new_figure(figsize)
        super().__init__(self._figure)
        self.setParent(parent)
        self.ax = self._figure.add_subplot(111)
        style_axes(self.ax, xlabel="", ylabel="")
        self._points: List[Tuple[float, float, str]] = []
        self._init_hover(self.ax)
        self.draw_idle()

    def plot_series(
        self,
        categories: Sequence[str],
        series: List[Tuple[str, Sequence[float], str]],
        ylabel: str,
        title: str = "",
        log_scale: bool = False,
    ) -> None:
        """``series`` is a list of (label, values, color_hex), one line per method."""
        self.ax.clear()
        style_axes(self.ax, xlabel="", ylabel=ylabel)
        if title:
            self.ax.set_title(title, fontsize=10)

        x = list(range(len(categories)))
        self._points = []
        for label, values, color in series:
            self.ax.plot(
                x, values, color=color, marker="o", markersize=6, linewidth=1.8,
                markeredgecolor=_FIG_BG, markeredgewidth=0.7, label=label, zorder=3,
            )
            series_prefix = f"{label} · " if len(series) > 1 else ""
            for xi, yi, category in zip(x, values, categories):
                self._points.append((xi, yi, f"{series_prefix}{category}\n{yi:.3g} {ylabel}".strip()))

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(categories, fontsize=8)
        self.ax.set_xlim(-0.3, max(0, len(categories) - 1) + 0.3)
        if log_scale:
            self.ax.set_yscale("log")
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            self.ax.yaxis.set_major_formatter(formatter)
        if len(series) > 1:
            legend = self.ax.legend(fontsize=7, facecolor=_FIG_BG, edgecolor=_SPINE, labelcolor="#c4cad4", loc="best")
            legend.get_frame().set_alpha(0.9)
        self._make_tooltip()
        self.draw_idle()

    def _find_hit(self, event) -> Optional[Tuple[float, float, str]]:
        if not self._points:
            return None
        threshold_px = 14
        best = None
        best_d2 = threshold_px ** 2
        for x, y, text in self._points:
            px, py = self.ax.transData.transform((x, y))
            d2 = (px - event.x) ** 2 + (py - event.y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (x, y, text)
        return best
