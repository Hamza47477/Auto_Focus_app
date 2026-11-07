"""
Algorithm Package for Electron Microscope Autofocus

Extracts standalone implementations of key search and focus score algorithms:
- Search:
    * Coarse-and-Fine Hybrid Search (coarse_and_fine_search.py)
    * Global Linear Scan Search (global_scan_search.py)
    * Gradient Descent / Ascent Search (gradient_descent_search.py)
    * Coarse Search with Quadratic Fit (coarse_quadratic_search.py)
- Refinement:
    * Quadratic Polynomial Curve Fitting (quadratic_fit.py)
- Focus Score:
    * Normalized Variance (normalized_variance.py)
    * Laplacian (variance, LoG, modified / SML) (laplacian.py)
"""

from __future__ import annotations

from .coarse_and_fine_search import (
    coarse_and_fine_search,
    search_hybrid_coarse_fine,
    hybrid_coarse_fine_search,
    SearchResult,
)
from .global_scan_search import (
    global_scan_search,
    global_search,
)
from .gradient_descent_search import (
    gradient_descent_search,
    gradient_ascent_search,
    gradient_search,
    gradient_ascent,
    verify_peak,
)
from .quadratic_fit import (
    quadratic_fit,
    fit_quadratic_vertex,
    QuadraticFitResult,
)
from .coarse_quadratic_search import (
    coarse_quadratic_search,
    coarse_fit_search,
    coarse_quadratic,
    coarse_fit,
)
from .normalized_variance import (
    normalized_variance,
    mean_normalized_variance,
)
from .laplacian import (
    variance_of_laplacian,
    laplacian_of_gaussian,
    modified_laplacian,
    laplacian_focus_score,
)

__all__ = [
    # Search algorithms
    "coarse_and_fine_search",
    "search_hybrid_coarse_fine",
    "hybrid_coarse_fine_search",
    "global_scan_search",
    "global_search",
    "gradient_descent_search",
    "gradient_ascent_search",
    "gradient_search",
    "gradient_ascent",
    "verify_peak",
    "coarse_quadratic_search",
    "coarse_fit_search",
    "coarse_quadratic",
    "coarse_fit",
    "SearchResult",
    # Fine refinement
    "quadratic_fit",
    "fit_quadratic_vertex",
    "QuadraticFitResult",
    # Focus score algorithms
    "normalized_variance",
    "mean_normalized_variance",
    "variance_of_laplacian",
    "laplacian_of_gaussian",
    "modified_laplacian",
    "laplacian_focus_score",
]
