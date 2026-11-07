"""Static autofocus algorithms for SEM pollen-grain image stacks.

This package implements the coarse/fine autofocus pipeline described in the
course notes (normalized-variance focus measure, derivative-based gradient
ascent, golden-section search and a quadratic sub-index fit) with lazy,
cached image access so large datasets are never read in full by the search
algorithms.

It contains no GUI code. A PySide6 front end is built on top of this API
separately; every public function here is designed to be safe to call from
a worker thread and to report progress through plain callbacks.
"""

from .settings import AutofocusSettings
from .dataset import ImageDataset
from .focus import FocusEvaluator, normalized_variance
from .results import AutofocusResult, TraceStep
from .fit import QuadraticFitOutcome, quadratic_fine_position
from .search import gradient_ascent, golden_section, full_sweep

__all__ = [
    "AutofocusSettings",
    "ImageDataset",
    "FocusEvaluator",
    "normalized_variance",
    "AutofocusResult",
    "TraceStep",
    "QuadraticFitOutcome",
    "quadratic_fine_position",
    "gradient_ascent",
    "golden_section",
    "full_sweep",
]
