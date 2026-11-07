"""The three search strategies: gradient ascent, golden-section, full sweep.

Only ``full_sweep`` reads every image on purpose (it is the ground-truth
baseline). ``gradient_ascent`` and ``golden_section`` read a small, bounded
number of images through a shared :class:`~autofocus.focus.FocusEvaluator`.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from .dataset import ImageDataset
from .fit import quadratic_fine_position
from .focus import FocusEvaluator
from .results import AutofocusResult, TraceStep
from .settings import AutofocusSettings

OnStep = Callable[[TraceStep], None]
ShouldStop = Callable[[], bool]


def _no_stop() -> bool:
    return False


def _climb(
    evaluator: FocusEvaluator,
    settings: AutofocusSettings,
    rng: np.random.Generator,
    start_index: int,
    warnings: List[str],
    stop: ShouldStop,
    restart_no: int = 0,
) -> Tuple[int, int]:
    """Gradient ascent (6.3.2) with optional neighbor-check continuation.

    Runs the hill-climb to a local peak, then (if enabled) checks the two
    neighboring images; if either is better, the climb resumes from there.
    The whole thing shares one ``max_iterations`` budget.

    ``restart_no`` is purely descriptive (0 for the initial climb, 1+ for a
    safety-check restart); it only affects the trace notes below, not the
    search itself, so a GUI walkthrough can label which restart a step
    belongs to.

    Returns:
        ``(peak_index, iterations_used)``.
    """
    n = len(evaluator.dataset)
    d = start_index
    start_note = "ascent start" if restart_no == 0 else f"ascent start (restart #{restart_no})"
    f0 = evaluator.evaluate(d, phase="ascent", note=start_note)
    alpha = settings.resolve_alpha0(n)
    delta = settings.delta
    iterations = 0

    while True:
        # --- hill-climb to a local peak ---
        while iterations < settings.max_iterations:
            if stop():
                warnings.append("cancelled by should_stop() during ascent")
                return d, iterations
            iterations += 1

            f_minus = evaluator.evaluate(d - delta, phase="ascent", note="f(d-delta)") if d - delta >= 0 else None
            f_plus = evaluator.evaluate(d + delta, phase="ascent", note="f(d+delta)") if d + delta <= n - 1 else None
            fn = evaluator.evaluate(d, phase="ascent", note="f(d)")

            if f_plus is not None and f_minus is not None:
                slope = (f_plus - f_minus) / (2 * delta)  # Eq. 6.13, centered difference
            elif f_plus is not None:
                slope = (f_plus - fn) / delta  # Eq. 6.13, one-sided (left boundary)
            elif f_minus is not None:
                slope = (fn - f_minus) / delta  # Eq. 6.13, one-sided (right boundary)
            else:
                break  # single-image dataset: nowhere to go

            if slope == 0:
                break  # flat: already at a maximum

            direction = 1 if slope > 0 else -1  # Eq. 6.15, Gamma = sign(slope)
            # Descriptive only: records the just-computed slope/direction on the
            # trace entry already logged above, for the Walkthroughs screen.
            evaluator.trace[-1].note = f"f(d) slope={slope:+.5f} dir={direction:+d}"

            ratio = (f0 / fn) if fn != 0 else 0.0
            lam = alpha * (ratio ** 2) if ratio < 1.0 else alpha  # Eq. 6.16
            u = None
            if settings.use_stochastic_factor:
                u = rng.uniform(settings.stochastic_low, settings.stochastic_high)
                lam *= u
            step = max(1, round(lam))

            candidate = int(np.clip(d + direction * step, 0, n - 1))  # Eq. 6.14
            if candidate == d:
                break  # boundary reached in the uphill direction: local peak

            f_candidate = evaluator.evaluate(candidate, phase="ascent", note="candidate")
            if settings.use_overshoot_control:
                accepted = f_candidate > fn
                if accepted:
                    d = candidate
                else:
                    alpha *= settings.overshoot_shrink
            else:
                accepted = True
                d = candidate
            # Descriptive only: records the step size, random factor and the
            # accept/reject decision on the candidate's trace entry.
            u_text = f" u={u:.3f}" if u is not None else ""
            evaluator.trace[-1].note = (
                f"candidate lam={lam:.3f} step={step}{u_text} accepted={'yes' if accepted else 'no'}"
            )
            if settings.use_overshoot_control and not accepted and step == 1:
                break  # smallest possible step was rejected: local peak
            # else: stay at d, retry next iteration with a smaller alpha

        # --- neighbor check (optional), after the climb has converged ---
        if not settings.use_neighbor_check or iterations >= settings.max_iterations:
            break

        fn = evaluator.evaluate(d, phase="neighbor_check", note="f(d)")
        best_d, best_f = d, fn
        if d - 1 >= 0:
            f_left = evaluator.evaluate(d - 1, phase="neighbor_check", note="f(d-1)")
            if f_left > best_f:
                best_d, best_f = d - 1, f_left
        if d + 1 <= n - 1:
            f_right = evaluator.evaluate(d + 1, phase="neighbor_check", note="f(d+1)")
            if f_right > best_f:
                best_d, best_f = d + 1, f_right

        if best_d == d:
            break  # neither neighbor is better: converged

        evaluator.trace[-1].note = f"f(d{'+1' if best_d > d else '-1'}) better, continue ascent"
        d = best_d  # continue the ascent from the better neighbor

    return d, iterations


def gradient_ascent(
    dataset: ImageDataset,
    settings: AutofocusSettings,
    start_index: int,
    on_step: Optional[OnStep] = None,
    should_stop: Optional[ShouldStop] = None,
) -> AutofocusResult:
    """Coarse sharpest image by derivative-based gradient ascent (6.3.2) + fine fit (6.10).

    Starts from ``start_index`` and climbs in the integer index domain. With
    the overshoot, neighbor-check and safety-check protections on (the
    defaults), it does not get stuck on a local maximum and never reads the
    whole dataset.

    Args:
        dataset: Image dataset to search.
        settings: Algorithm parameters.
        start_index: Index to start climbing from.
        on_step: Optional callback invoked after every focus evaluation.
        should_stop: Optional callback polled to allow cancellation.

    Raises:
        IndexError: ``start_index`` is outside ``[0, len(dataset) - 1]``.
    """
    n = len(dataset)
    if not (0 <= start_index < n):
        raise IndexError(f"start index {start_index} out of range [0, {n - 1}]")

    t0 = time.perf_counter()
    evaluator = FocusEvaluator(dataset, settings, on_step=on_step)
    rng = np.random.default_rng(settings.seed)
    stop = should_stop if should_stop is not None else _no_stop
    warnings: List[str] = []
    restarts = 0

    peak_index, _ = _climb(evaluator, settings, rng, start_index, warnings, stop)
    peak_focus = evaluator.evaluate(peak_index, phase="ascent", note="coarse peak")

    if settings.use_safety_check:
        # Fixed, evenly spaced sample points (including both ends); reused
        # across restarts, so only the first round of safety samples costs
        # new image reads.
        sample_indices = sorted(set(np.linspace(0, n - 1, settings.safety_samples_k).round().astype(int)))

        while not stop():
            samples = [(i, evaluator.evaluate(i, phase="safety_sample", note="safety sample")) for i in sample_indices]
            better = [(i, f) for i, f in samples if f > peak_focus]
            if not better:
                break  # safety check confirms the current peak

            best_sample_index, best_sample_focus = max(better, key=lambda pair: pair[1])
            if restarts >= settings.max_restarts:
                warnings.append(
                    f"safety check still finds a higher sample at index {best_sample_index} "
                    f"after {settings.max_restarts} restarts; keeping the best peak found so far"
                )
                break

            restarts += 1
            candidate_peak, _ = _climb(evaluator, settings, rng, best_sample_index, warnings, stop, restart_no=restarts)
            candidate_focus = evaluator.evaluate(candidate_peak, phase="restart", note=f"restart #{restarts} peak")
            if candidate_focus > peak_focus:
                peak_index, peak_focus = candidate_peak, candidate_focus

    fit = quadratic_fine_position(evaluator, dataset, peak_index, settings)
    elapsed = time.perf_counter() - t0

    return AutofocusResult(
        method="gradient_ascent",
        start_index=start_index,
        coarse_index=peak_index,
        coarse_focus=peak_focus,
        coarse_position_mm=dataset.position_mm(peak_index),
        fine_index=fit.fine_index,
        fine_position_mm=fit.fine_position_mm,
        fit_coefficients=fit.coefficients,
        fit_points=fit.points,
        images_read=evaluator.images_read,
        total_images=n,
        percent_read=100.0 * evaluator.images_read / n,
        elapsed_s=elapsed,
        restarts=restarts,
        seed=settings.seed,
        settings=settings,
        warnings=warnings + fit.warnings,
        trace=evaluator.trace,
    )


def golden_section(
    dataset: ImageDataset,
    settings: AutofocusSettings,
    on_step: Optional[OnStep] = None,
    should_stop: Optional[ShouldStop] = None,
) -> AutofocusResult:
    """Coarse sharpest image by golden-section search (6.3.1) + fine fit (6.10).

    Searches the full index range ``[0, N-1]``; it ignores any start index.
    Reuses one of the two probe evaluations every iteration (only one new
    image is read per iteration) and stops once the bracket is 2 images
    wide or less, then evaluates the (at most 3) remaining images in the
    bracket to pick the best one.
    """
    n = len(dataset)
    t0 = time.perf_counter()
    evaluator = FocusEvaluator(dataset, settings, on_step=on_step)
    stop = should_stop if should_stop is not None else _no_stop
    warnings: List[str] = []

    phi = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate ~= 0.618, section 6.3.1
    low, high = 0, n - 1
    c = int(round(high - phi * (high - low)))
    d_pt = int(round(low + phi * (high - low)))
    f_c = evaluator.evaluate(c, phase="golden_probe", note="c")
    f_d = evaluator.evaluate(d_pt, phase="golden_probe", note="d")

    while (high - low) > 2 and not stop():
        if f_c > f_d:
            high = d_pt
            d_pt, f_d = c, f_c
            c = int(round(high - phi * (high - low)))
            if c == d_pt:
                break  # degenerate bracket: nothing left to narrow
            f_c = evaluator.evaluate(c, phase="golden_probe", note="c")
        else:
            low = c
            c, f_c = d_pt, f_d
            d_pt = int(round(low + phi * (high - low)))
            if d_pt == c:
                break
            f_d = evaluator.evaluate(d_pt, phase="golden_probe", note="d")

    bracket_values = [(i, evaluator.evaluate(i, phase="golden_probe", note="bracket")) for i in range(low, high + 1)]
    coarse_index, coarse_focus = max(bracket_values, key=lambda pair: pair[1])

    fit = quadratic_fine_position(evaluator, dataset, coarse_index, settings)
    elapsed = time.perf_counter() - t0

    return AutofocusResult(
        method="golden_section",
        start_index=None,
        coarse_index=coarse_index,
        coarse_focus=coarse_focus,
        coarse_position_mm=dataset.position_mm(coarse_index),
        fine_index=fit.fine_index,
        fine_position_mm=fit.fine_position_mm,
        fit_coefficients=fit.coefficients,
        fit_points=fit.points,
        images_read=evaluator.images_read,
        total_images=n,
        percent_read=100.0 * evaluator.images_read / n,
        elapsed_s=elapsed,
        restarts=0,
        seed=settings.seed,
        settings=settings,
        warnings=warnings + fit.warnings,
        trace=evaluator.trace,
    )


def full_sweep(
    dataset: ImageDataset,
    settings: AutofocusSettings,
    on_step: Optional[OnStep] = None,
    should_stop: Optional[ShouldStop] = None,
) -> AutofocusResult:
    """Baseline: read every image and return the true argmax + fine fit (6.10).

    Ground truth for the report graphs and the tests; not meant to be used
    on large datasets.
    """
    n = len(dataset)
    t0 = time.perf_counter()
    evaluator = FocusEvaluator(dataset, settings, on_step=on_step)
    stop = should_stop if should_stop is not None else _no_stop

    best_index, best_focus = 0, -np.inf
    for i in range(n):
        if stop():
            break
        f = evaluator.evaluate(i, phase="sweep", note="full sweep")
        if f > best_focus:
            best_index, best_focus = i, f

    fit = quadratic_fine_position(evaluator, dataset, best_index, settings)
    elapsed = time.perf_counter() - t0

    return AutofocusResult(
        method="full_sweep",
        start_index=None,
        coarse_index=best_index,
        coarse_focus=best_focus,
        coarse_position_mm=dataset.position_mm(best_index),
        fine_index=fit.fine_index,
        fine_position_mm=fit.fine_position_mm,
        fit_coefficients=fit.coefficients,
        fit_points=fit.points,
        images_read=evaluator.images_read,
        total_images=n,
        percent_read=100.0 * evaluator.images_read / n,
        elapsed_s=elapsed,
        restarts=0,
        seed=settings.seed,
        settings=settings,
        warnings=list(fit.warnings),
        trace=evaluator.trace,
    )
