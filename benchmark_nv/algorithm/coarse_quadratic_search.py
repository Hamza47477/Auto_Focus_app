"""
Coarse Search with Quadratic Fit (Coarse-Fit / Coarse-Quadratic Autofocus Search)

Algorithm Overview:
This algorithm implements a robust two-stage static autofocus search:
1. Stage 1 (Coarse Search):
   Rapidly locates the discrete focus peak kc using either:
   - "gradient" (Default): Derivative-based gradient ascent with an adaptive
     decreasing step factor, centered finite differences, and statistical
     noise-floor verification with multi-start basin restarts.
   - "grid": Sparse coarse grid sampling with step s = max(3, round(sqrt(N))).
   Saves ~70-90% of frame reads by sampling sparsely across the focal stack.

2. Stage 2 (Fine Quadratic Fit):
   Refines the discrete peak kc to sub-frame continuous accuracy using parabolic
   fitting on window [kc - half_window, kc + half_window]:
       f(d) = A_0 + A_1 * d + A_2 * d^2
   Finds analytical continuous vertex:
       d* = -A_1 / (2 * A_2)
       z* = kc + d* / step
   Provides goodness-of-fit R^2 and polynomial coefficients (A0, A1, A2).

Aliases:
   - coarse_quadratic_search
   - coarse_fit_search
   - coarse_quadratic
   - coarse_fit

Returns:
   SearchResult container:
       Unpackable as (best_idx, evals, history)
       or accessed via rich attributes:
       - result.best_idx / result.coarse_idx: Discrete optimum frame
       - result.fine_idx: Continuous sub-frame optimum (e.g. 73.38)
       - result.fine_value: Interpolated focus score at continuous peak
       - result.fit_ok: True if quadratic fit converged to valid concave peak
       - result.fit_r2: Goodness of fit R^2
       - result.fit_coeffs: (A0, A1, A2)
       - result.evals: All evaluated frame scores {idx: score}
       - result.history: Step-by-step audit log covering both coarse and fine stages
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any, Optional, Sequence, Union
import math
import inspect
import numpy as np

try:
    from .gradient_descent_search import gradient_descent_search, _invoke_eval, _clamp
    from .quadratic_fit import quadratic_fit, QuadraticFitResult
except (ImportError, ValueError):
    from gradient_descent_search import gradient_descent_search, _invoke_eval, _clamp
    from quadratic_fit import quadratic_fit, QuadraticFitResult


class SearchResult:
    """
    Search result container for Coarse-Fit / Coarse-Quadratic Search.
    Fully compatible with standard 3-tuple unpacking:
        best_idx, evals, history = result
    while exposing coarse discrete peak and fine quadratic fit properties:
        result.best_idx, result.coarse_idx, result.fine_idx, result.fine_value,
        result.fine_wd, result.fit_ok, result.fit_r2, result.fit_coeffs,
        result.fit_message, result.fit_window, result.evals, result.history,
        result.verified, result.restarts, result.noise_sigma.
    """
    def __init__(
        self,
        best_idx: int,
        evals: Dict[int, float],
        history: List[Dict[str, Any]],
        coarse_idx: int,
        fine_idx: float,
        fine_value: float,
        fine_wd: float,
        fit_ok: bool,
        fit_r2: float,
        fit_coeffs: Optional[Tuple[float, float, float]],
        fit_message: str = "ok",
        fit_window: Tuple[int, int] = (0, 0),
        verified: Optional[bool] = None,
        restarts: int = 0,
        noise_sigma: float = float("nan"),
    ):
        self.best_idx = int(best_idx)
        self.coarse_idx = int(coarse_idx)
        self.fine_idx = float(fine_idx)
        self.fine_value = float(fine_value)
        self.fine_wd = float(fine_wd)
        self.fit_ok = bool(fit_ok)
        self.fit_r2 = float(fit_r2)
        self.fit_coeffs = fit_coeffs
        self.fit_message = str(fit_message)
        self.fit_window = fit_window
        self.evals = evals
        self.history = history
        self.verified = verified
        self.restarts = restarts
        self.noise_sigma = float(noise_sigma)

    # Standard 3-tuple unpacking: best_idx, evals, history = result
    def __iter__(self):
        return iter((self.best_idx, self.evals, self.history))

    def __getitem__(self, index: int):
        return (self.best_idx, self.evals, self.history)[index]

    def __len__(self) -> int:
        return 3

    def __repr__(self) -> str:
        return (
            f"SearchResult(coarse_idx={self.coarse_idx}, fine_idx={self.fine_idx:.3f}, "
            f"fit_ok={self.fit_ok}, R2={self.fit_r2:.4f}, evals={len(self.evals)})"
        )


def _coarse_grid_sweep(
    eval_func: Callable[..., float],
    max_idx: int,
    step_s: Optional[int] = None,
    stride: int = 4,
    fmo_name: Optional[str] = None,
    evals: Optional[Dict[int, float]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Helper coarse grid sweep across [0, max_idx]."""
    if evals is None:
        evals = {}
    if history is None:
        history = []

    num_frames = max_idx + 1
    step = step_s if step_s is not None else max(3, int(round(math.sqrt(num_frames))))
    grid_indices = list(range(0, num_frames, step))
    if grid_indices[-1] != max_idx:
        grid_indices.append(max_idx)

    for idx in grid_indices:
        if idx not in evals:
            val = _invoke_eval(eval_func, idx, stride=stride, fmo_name=fmo_name)
            evals[idx] = val
            history.append({
                "step": len(history) + 1,
                "idx": idx,
                "score": val,
                "stride": stride,
                "stage": "coarse_grid",
            })

    return max(grid_indices, key=evals.__getitem__)


def coarse_quadratic_search(
    eval_func: Callable[..., float],
    max_idx: int,
    coarse_method: str = "gradient",
    half_window: int = 12,
    start_idx: Optional[int] = None,
    alpha: float = 24.0,
    delta: int = 3,
    verify_dist: int = 10,
    verify_k: float = 3.0,
    max_restarts: int = 8,
    coarse_step: Optional[int] = None,
    coarse_stride: int = 1,
    fine_stride: int = 1,
    coarse_fmo: Optional[str] = None,
    fine_fmo: Optional[str] = None,
    wd: Optional[Sequence[float]] = None,
) -> SearchResult:
    """
    Executes a two-stage coarse autofocus search combined with fine quadratic curve fitting.

    Stage 1: Coarse search isolates discrete focal peak kc.
             - coarse_method="gradient": Derivative-based gradient ascent with noise verification
             - coarse_method="grid": Sparse grid sampling (stride max(3, sqrt(N)))
    Stage 2: Fine quadratic fit computes sub-frame continuous peak z*, score, and R^2.

    Args:
        eval_func: Focus evaluation function:
                   eval_func(idx) or eval_func(idx, stride) or eval_func(idx, stride, fmo).
        max_idx: Maximum frame index in stack (0-indexed, inclusive).
        coarse_method: Coarse search strategy: 'gradient' (default) or 'grid'.
        half_window: Frames on each side of coarse peak to fit quadratic parabola (default 12).
        start_idx: Initial probe frame index for gradient ascent (default max_idx // 2).
        alpha: Initial step size parameter beta for gradient search (default 24.0).
        delta: Minimum derivative finite-difference span h (default 3).
        verify_dist: Verification probe distance for gradient search (default 10).
        verify_k: Multiplier for noise threshold k*sigma in verification (default 3.0).
        max_restarts: Maximum basin relocation attempts (default 8).
        coarse_step: Step size between coarse probes when coarse_method='grid'.
        coarse_stride: Pixel subsampling stride during coarse stage.
        fine_stride: Pixel subsampling stride during fine quadratic fit (default 1).
        coarse_fmo: Focus measure identifier used during coarse stage.
        fine_fmo: Focus measure identifier used during fine quadratic fit stage.
        wd: Optional sequence or array of physical working distances (mm or um).

    Returns:
        SearchResult container with both discrete and sub-frame continuous results:
            - Unpackable as: best_idx, evals, history = result
            - Attributes:
                result.best_idx / result.coarse_idx: Discrete optimum frame
                result.fine_idx: Continuous sub-frame optimum (e.g. 73.38)
                result.fine_value: Interpolated focus score at peak
                result.fine_wd: Physical working distance at continuous peak
                result.fit_ok: True if quadratic fit succeeded
                result.fit_r2: Goodness of fit R^2
                result.fit_coeffs: (A0, A1, A2) in f(d) = A0 + A1*d + A2*d^2
                result.fit_message: Status message
                result.fit_window: (lo, hi) frame range used for fit
                result.evals: All evaluated frame scores {idx: score}
                result.history: Execution step log
                result.verified: Fake-maximum test result (if gradient search)
                result.restarts: Relocation restart count (if gradient search)

    Example:
        >>> result = coarse_quadratic_search(my_eval_fn, max_idx=199)
        >>> print(f"Discrete: Frame {result.best_idx}, Sub-frame: {result.fine_idx:.2f}")
    """
    evals: Dict[int, float] = {}
    history: List[Dict[str, Any]] = []

    verified: Optional[bool] = None
    restarts = 0
    noise_sigma = float("nan")

    # --- Stage 1: Coarse Search ---
    if coarse_method.lower() in ("grid", "sweep_grid", "coarse_grid"):
        coarse_idx = _coarse_grid_sweep(
            eval_func=eval_func,
            max_idx=max_idx,
            step_s=coarse_step,
            stride=coarse_stride,
            fmo_name=coarse_fmo,
            evals=evals,
            history=history,
        )
    else:  # default: gradient search from autofocus_core.py
        coarse_res = gradient_descent_search(
            eval_func=eval_func,
            max_idx=max_idx,
            start_idx=start_idx,
            alpha=alpha,
            delta=delta,
            verify_dist=verify_dist,
            verify_k=verify_k,
            max_restarts=max_restarts,
            verify_peaks=True,
            stride=coarse_stride,
            fmo_name=coarse_fmo,
        )
        coarse_idx = coarse_res.best_idx
        evals.update(coarse_res.evals)
        history.extend(coarse_res.history)
        verified = coarse_res.verified
        restarts = coarse_res.restarts
        noise_sigma = coarse_res.noise_sigma

    # --- Stage 2: Fine Quadratic Curve Fit ---
    fit_res: QuadraticFitResult = quadratic_fit(
        eval_func=eval_func,
        coarse_idx=coarse_idx,
        max_idx=max_idx,
        half_window=half_window,
        wd=wd,
        evals=evals,
        stride=fine_stride,
        fmo_name=fine_fmo,
    )

    # Log quadratic fit points in history if newly evaluated
    lo, hi = fit_res.lo, fit_res.hi
    for step_item in history:
        if lo <= step_item["idx"] <= hi and step_item["stage"] == "coarse_grid":
            pass  # Already tagged

    for idx in range(lo, hi + 1):
        if not any(h["idx"] == idx for h in history):
            history.append({
                "step": len(history) + 1,
                "idx": idx,
                "score": evals[idx],
                "stride": fine_stride,
                "stage": "quadratic_fit_window",
            })

    # Fallback to coarse position if fit was not acceptable
    fine_idx = fit_res.idx if fit_res.ok else float(coarse_idx)
    fine_val = fit_res.value if fit_res.ok else float(evals[coarse_idx])
    fine_wd = fit_res.wd if fit_res.ok else (float(wd[coarse_idx]) if wd is not None else float(coarse_idx))

    return SearchResult(
        best_idx=coarse_idx,
        evals=evals,
        history=history,
        coarse_idx=coarse_idx,
        fine_idx=fine_idx,
        fine_value=fine_val,
        fine_wd=fine_wd,
        fit_ok=fit_res.ok,
        fit_r2=fit_res.r2,
        fit_coeffs=fit_res.coeffs,
        fit_message=fit_res.msg,
        fit_window=(lo, hi),
        verified=verified,
        restarts=restarts,
        noise_sigma=noise_sigma,
    )


# Canonical naming aliases
coarse_fit_search = coarse_quadratic_search
coarse_quadratic = coarse_quadratic_search
coarse_fit = coarse_quadratic_search


if __name__ == "__main__":
    print("Testing Coarse-Fit / Coarse-Quadratic Search Algorithm...")

    TRUE_SUBFRAME_PEAK = 73.35
    TOTAL_FRAMES = 200
    MAX_IDX = TOTAL_FRAMES - 1

    def synthetic_eval(idx: int, stride: int = 1, fmo: Optional[str] = None) -> float:
        # Gaussian curve with true sub-frame continuous peak
        sigma = 8.0
        val = 100.0 * np.exp(-((idx - TRUE_SUBFRAME_PEAK) ** 2) / (2 * sigma ** 2))
        return float(val)

    # 1. Test Coarse Quadratic with Gradient Coarse Method (Default)
    res_grad = coarse_quadratic(synthetic_eval, max_idx=MAX_IDX, coarse_method="gradient")
    print(f"\n[1. Coarse Quadratic (Gradient + Fit)]")
    print(f"Discrete Coarse Peak : Frame {res_grad.best_idx}")
    print(f"Sub-Frame Fine Peak  : Frame {res_grad.fine_idx:.3f} (True: {TRUE_SUBFRAME_PEAK:.3f})")
    print(f"Fitted Peak Score    : {res_grad.fine_value:.4f}")
    print(f"Fit Status           : {res_grad.fit_ok} (R^2 = {res_grad.fit_r2:.4f})")
    print(f"Frames Evaluated     : {len(res_grad.evals)} / {TOTAL_FRAMES}")
    reduction = (1.0 - len(res_grad.evals) / TOTAL_FRAMES) * 100.0
    print(f"Frame Read Reduction : {reduction:.1f}%")

    subframe_err = abs(res_grad.fine_idx - TRUE_SUBFRAME_PEAK)
    print(f"Sub-Frame Error      : {subframe_err:.4f} frames")
    assert abs(res_grad.best_idx - round(TRUE_SUBFRAME_PEAK)) <= 1
    assert subframe_err < 0.15
    assert res_grad.fit_ok is True
    assert res_grad.fit_r2 > 0.95

    # 2. Test Coarse Fit with Grid Coarse Method
    res_grid = coarse_fit(synthetic_eval, max_idx=MAX_IDX, coarse_method="grid")
    print(f"\n[2. Coarse Fit (Grid + Fit)]")
    print(f"Discrete Coarse Peak : Frame {res_grid.best_idx}")
    print(f"Sub-Frame Fine Peak  : Frame {res_grid.fine_idx:.3f} (True: {TRUE_SUBFRAME_PEAK:.3f})")
    print(f"Frames Evaluated     : {len(res_grid.evals)} / {TOTAL_FRAMES}")
    subframe_err_grid = abs(res_grid.fine_idx - TRUE_SUBFRAME_PEAK)
    print(f"Sub-Frame Error      : {subframe_err_grid:.4f} frames")
    assert subframe_err_grid < 0.25, f"Sub-frame error {subframe_err_grid} >= 0.25"

    # Unpack test
    b, e, h = res_grad
    assert b == res_grad.best_idx
    assert len(e) == len(res_grad.evals)
    print("\n[OK] Coarse-Fit / Coarse-Quadratic tests passed successfully.")
