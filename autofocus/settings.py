"""All tunable parameters of the autofocus pipeline, in one place.

Keeping every parameter in a single dataclass (instead of scattering default
arguments across functions) is what makes the package GUI-friendly: a future
PySide6 front end can build one form from :class:`AutofocusSettings`, let the
user edit it, and pass the same object into any search function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AutofocusSettings:
    """Parameters shared by every autofocus search and the focus measure.

    Attributes:
        d_min_mm: Working distance of image index 0, in mm (section 1).
        step_mm: Working-distance increment per image index, in mm
            (section 1): ``d(i) = d_min_mm + i * step_mm``.

        do_crop_bottom: Whether to remove the SEM info bar before computing
            the focus measure.
        crop_bottom_px: Number of rows removed from the bottom of each image
            when ``do_crop_bottom`` is on. Inspect a sample image from the
            dataset to pick a sensible value (see the project README).
        do_denoise: Whether to blur the image before computing the focus
            measure.
        denoise_method: ``"gaussian"`` or ``"median"``.
        denoise_ksize: Kernel size of the blur (forced to the next odd
            number if even).

        delta: Step used by the centered-difference slope, Eq. 6.13.
        alpha0: Initial step gain used in Eq. 6.16. When ``None``, it
            defaults to ``max(2, N // 10)`` where N is the number of images
            in the dataset (computed inside the search, since it depends on
            the dataset size).
        use_stochastic_factor: Multiply the step size by a random factor
            (lab suggestion) so the search does not always retrace the exact
            same path.
        stochastic_low: Lower bound of the random factor u ~ U(low, high).
        stochastic_high: Upper bound of the random factor u ~ U(low, high).
        use_overshoot_control: Reject a step that makes the focus worse and
            shrink alpha instead of taking it.
        overshoot_shrink: Multiplicative shrink factor applied to alpha
            after a rejected step.
        max_iterations: Hard cap on the number of climbing steps of a single
            gradient-ascent run (shared by the neighbor-check continuation).
        use_neighbor_check: After convergence, look one image to either side
            and keep climbing if either is better.
        use_safety_check: Sample a handful of evenly spaced images; if one
            beats the found peak, restart the ascent from there.
        safety_samples_k: Number of evenly spaced samples used by the safety
            check (including both ends of the dataset).
        max_restarts: Maximum number of safety-check restarts.

        fit_half_window: Half-width (in images) of the window used by the
            quadratic fit around the coarse peak, Eq. 6.10.

        seed: Seed of the ``numpy.random.Generator`` used for the stochastic
            step factor, so runs are repeatable.
    """

    # --- dataset / position mapping (section 1) ---
    d_min_mm: float = 10.0
    step_mm: float = 0.010

    # --- preprocessing (section 2) ---
    do_crop_bottom: bool = False
    crop_bottom_px: int = 0
    do_denoise: bool = False
    denoise_method: str = "gaussian"  # "gaussian" | "median"
    denoise_ksize: int = 5

    # --- gradient ascent, course section 6.3.2 ---
    delta: int = 1
    alpha0: Optional[int] = None
    use_stochastic_factor: bool = False
    stochastic_low: float = 0.5
    stochastic_high: float = 1.5
    use_overshoot_control: bool = True
    overshoot_shrink: float = 0.5
    max_iterations: int = 100
    use_neighbor_check: bool = True
    use_safety_check: bool = True
    safety_samples_k: int = 8
    max_restarts: int = 3

    # --- quadratic fit, Eq. 6.10 ---
    fit_half_window: int = 3

    # --- reproducibility ---
    seed: Optional[int] = None

    def resolve_alpha0(self, total_images: int) -> float:
        """Return ``alpha0`` if set, otherwise the dataset-dependent default."""
        if self.alpha0 is not None:
            return float(self.alpha0)
        return float(max(2, total_images // 10))
