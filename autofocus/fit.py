"""Fine (sub-index) peak position by a quadratic fit, course Eq. 6.10.

Separated from ``search.py`` because all three search methods share the
exact same fine-position step once they have produced a coarse index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .dataset import ImageDataset
from .focus import FocusEvaluator
from .settings import AutofocusSettings


@dataclass
class QuadraticFitOutcome:
    """Result of the quadratic fit around a coarse peak."""

    fine_index: float
    fine_position_mm: float
    coefficients: Tuple[float, float, float]  # (a0, a1, a2), x centered on peak_index
    points: List[Tuple[int, float]]
    warnings: List[str] = field(default_factory=list)


def quadratic_fine_position(
    evaluator: FocusEvaluator,
    dataset: ImageDataset,
    peak_index: int,
    settings: AutofocusSettings,
) -> QuadraticFitOutcome:
    """Refine a coarse peak to a (possibly fractional) sub-index position.

    Fits ``f = a0 + a1*x + a2*x**2`` (Eq. 6.10) over the images in
    ``[peak_index - w, peak_index + w]`` (clipped to the dataset range),
    with x centered on ``peak_index`` for numerical stability. The vertex
    ``x* = -a1 / (2*a2)`` is only accepted if the parabola opens downward
    (``a2 < 0``) and the vertex falls inside the fitted window; otherwise
    the coarse index is kept and a warning is added.

    Args:
        evaluator: Shared evaluator (reused so already-cached focus values
            are not re-read).
        dataset: Dataset the peak belongs to (for the position mapping).
        peak_index: Coarse peak index to refine.
        settings: Provides ``fit_half_window``.

    Returns:
        The fitted sub-index peak, its position in mm, the fit coefficients
        and the points used, or a fallback to ``peak_index`` with a warning
        if the fit is degenerate.
    """
    n = len(dataset)
    w = settings.fit_half_window
    lo = max(0, peak_index - w)
    hi = min(n - 1, peak_index + w)
    indices = list(range(lo, hi + 1))
    warnings: List[str] = []

    if len(indices) < 3:
        warnings.append(
            f"quadratic fit needs >= 3 points, only {len(indices)} available "
            f"around index {peak_index}; falling back to the coarse index"
        )
        return QuadraticFitOutcome(
            fine_index=float(peak_index),
            fine_position_mm=dataset.position_mm(peak_index),
            coefficients=(0.0, 0.0, 0.0),
            points=[],
            warnings=warnings,
        )

    values = [evaluator.evaluate(i, phase="fit", note=f"fit window for peak {peak_index}") for i in indices]
    xs = np.array([i - peak_index for i in indices], dtype=np.float64)  # center x on the peak, Eq. 6.10
    ys = np.array(values, dtype=np.float64)

    # numpy.polyfit returns the highest power first: a2*x**2 + a1*x + a0
    a2, a1, a0 = np.polyfit(xs, ys, 2)

    x_lo, x_hi = float(xs.min()), float(xs.max())
    vertex_x = -a1 / (2.0 * a2) if a2 < 0 else None

    if vertex_x is not None and x_lo <= vertex_x <= x_hi:
        fine_index = peak_index + vertex_x
    else:
        warnings.append(
            "quadratic fit vertex is invalid (a2 >= 0 or the vertex falls outside "
            "the fitted window); falling back to the coarse index"
        )
        fine_index = float(peak_index)

    return QuadraticFitOutcome(
        fine_index=fine_index,
        fine_position_mm=dataset.position_mm(fine_index),
        coefficients=(float(a0), float(a1), float(a2)),
        points=list(zip(indices, values)),
        warnings=warnings,
    )
