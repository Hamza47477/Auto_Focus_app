"""Plain data returned by the autofocus searches.

Both dataclasses are pure data (no behaviour) so they can be freely passed
to a GUI thread, serialized, or dropped into a pandas/matplotlib report
without importing anything else from this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .settings import AutofocusSettings

#: Allowed values of :attr:`TraceStep.phase`.
TRACE_PHASES = (
    "start",
    "ascent",
    "neighbor_check",
    "safety_sample",
    "restart",
    "fit",
    "golden_probe",
    "sweep",
)


@dataclass
class TraceStep:
    """One focus evaluation, in the order it happened.

    A GUI can pass ``on_step`` into any search call and receive one of these
    after every new focus evaluation to draw points live.
    """

    step_no: int
    index: int
    position_mm: float
    focus: float
    phase: str
    note: str = ""


@dataclass
class AutofocusResult:
    """Everything a search produced, in one object.

    Attributes:
        method: ``"gradient_ascent"``, ``"golden_section"`` or ``"full_sweep"``.
        start_index: Start index given by the user (``None`` for methods
            that search the full range regardless of a start point).
        coarse_index: Index of the sharpest image found by the coarse search.
        coarse_focus: Normalized-variance value (Eq. 6.5) at ``coarse_index``.
        coarse_position_mm: Working distance of ``coarse_index``, in mm.
        fine_index: Sub-index peak position from the quadratic fit (may be
            fractional).
        fine_position_mm: Working distance at ``fine_index``, in mm.
        fit_coefficients: ``(a0, a1, a2)`` of Eq. 6.10, with x centered on
            ``coarse_index`` (x = index - coarse_index).
        fit_points: ``(index, focus)`` pairs that were fitted.
        images_read: Number of distinct images actually decoded from disk.
        total_images: Number of images in the dataset.
        percent_read: ``100 * images_read / total_images``.
        elapsed_s: Wall-clock time of the whole search (coarse + fit).
        restarts: Number of safety-check restarts performed (gradient ascent
            only; 0 for the other methods).
        seed: Seed used for the stochastic step factor.
        settings: The settings object the search was run with.
        warnings: Human-readable notes about fallbacks that happened
            (fit degenerate, safety check exhausted its restarts, ...).
        trace: Every focus evaluation, in order.
    """

    method: str
    start_index: Optional[int]
    coarse_index: int
    coarse_focus: float
    coarse_position_mm: float
    fine_index: float
    fine_position_mm: float
    fit_coefficients: Tuple[float, float, float]
    fit_points: List[Tuple[int, float]]
    images_read: int
    total_images: int
    percent_read: float
    elapsed_s: float
    restarts: int
    seed: Optional[int]
    settings: AutofocusSettings
    warnings: List[str] = field(default_factory=list)
    trace: List[TraceStep] = field(default_factory=list)
