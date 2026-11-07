"""
Quadratic Curve Fitting Algorithm (Fine Autofocus Refinement)

Algorithm Overview:
In discrete focal stacks, the true optical focus peak typically lies between
discrete frame captures (sub-frame focus optimum).

Fine search models the focus curve locally around the discrete coarse optimum kc
as a second-order parabola:
    f(d) = A_0 + A_1 * d + A_2 * d^2
where:
    d = wd[i] - wd[kc]   (centred relative working distance or frame offset)

Key Processing Steps:
1. Window Selection:
   Probes frames in the bracket [kc - half_window, kc + half_window].
   Requires at least 5 sample points for numerical stability.
2. Numerical Conditioning:
   Z-score normalizes focus values ys = (y - mean(y)) / (std(y) + eps) before
   fitting polynomial coefficients [a2, a1, a0] via least-squares polyfit.
3. Goodness of Fit (R^2):
   Computes coefficient of determination R^2 = 1 - SS_res / SS_tot.
4. Vertex & Concavity Verification:
   Ensures A_2 < 0 (concave downward, signifying a true maximum).
   Computes analytical vertex offset:
       d* = -A_1 / (2 * A_2)
       wd* = wd[kc] + d*
       z*  = kc + d* / step   (continuous fractional frame index)
   Verifies that vertex falls strictly within the evaluation window [wd_lo, wd_hi].

Returns:
   QuadraticFitResult supporting both attribute access (.ok, .idx, .value, .r2, .coeffs)
   and dictionary-style indexing (['ok'], ['idx'], ['wd']).
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any, Optional, Sequence, Union
import inspect
import numpy as np


class QuadraticFitResult:
    """
    Result container for Quadratic Curve Fitting.
    Supports both named attribute access:
        result.ok, result.idx, result.value, result.wd, result.r2, result.coeffs
    and dictionary-style access:
        result["ok"], result["idx"], result["coeffs"]
    """
    def __init__(
        self,
        ok: bool,
        msg: str,
        idx: float,
        value: float,
        wd: float,
        coeffs: Optional[Tuple[float, float, float]],
        r2: float,
        lo: int,
        hi: int,
        evals: Optional[Dict[int, float]] = None,
    ):
        self.ok = bool(ok)
        self.msg = str(msg)
        self.idx = float(idx)
        self.value = float(value)
        self.wd = float(wd)
        self.coeffs = coeffs
        self.r2 = float(r2)
        self.lo = int(lo)
        self.hi = int(hi)
        self.evals = evals or {}

    def __getitem__(self, key: str) -> Any:
        # Dictionary compatibility with autofocus_core.py
        mapping = {
            "ok": self.ok,
            "msg": self.msg,
            "idx": self.idx,
            "value": self.value,
            "wd": self.wd,
            "coeffs": self.coeffs,
            "r2": self.r2,
            "lo": self.lo,
            "hi": self.hi,
            "evals": self.evals,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __repr__(self) -> str:
        return (
            f"QuadraticFitResult(ok={self.ok}, idx={self.idx:.3f}, "
            f"value={self.value:.4f}, R2={self.r2:.4f}, msg='{self.msg}')"
        )


def _invoke_eval(
    eval_func: Callable[..., float],
    idx: int,
    stride: int = 1,
    fmo_name: Optional[str] = None
) -> float:
    """
    Safely invokes eval_func across differing function signatures:
    - eval_func(idx, stride, fmo_name)
    - eval_func(idx, stride)
    - eval_func(idx)
    """
    try:
        sig = inspect.signature(eval_func)
        params = list(sig.parameters.values())
        has_var_args = any(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in params)
        num_pos = sum(1 for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))

        if has_var_args or num_pos >= 3:
            return float(eval_func(idx, stride, fmo_name))
        elif num_pos == 2:
            return float(eval_func(idx, stride))
        else:
            return float(eval_func(idx))
    except (TypeError, ValueError):
        try:
            return float(eval_func(idx, stride, fmo_name))
        except TypeError:
            try:
                return float(eval_func(idx, stride))
            except TypeError:
                return float(eval_func(idx))


def fit_quadratic_vertex(
    d: Union[np.ndarray, Sequence[float]],
    y: Union[np.ndarray, Sequence[float]],
) -> Tuple[bool, str, Optional[Tuple[float, float, float]], float, float, float]:
    """
    Low-level mathematical quadratic fit:
        f(d) = A_0 + A_1 * d + A_2 * d^2

    Args:
        d: Relative position coordinates (e.g. frame offsets or centered WD).
        y: Focus score values.

    Returns:
        (ok, msg, coeffs, d_star, value_star, r2)
        where coeffs is (A0, A1, A2).
    """
    d_arr = np.asarray(d, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if len(d_arr) < 5:
        return False, "window too small for a quadratic fit (need >= 5 samples)", None, float("nan"), float("nan"), float("nan")

    # Normalize y for numerical conditioning
    s = float(y_arr.std()) + 1e-12
    m = float(y_arr.mean())
    ys = (y_arr - m) / s

    # Fit quadratic: ys = a2 * d^2 + a1 * d + a0
    a2, a1, a0 = np.polyfit(d_arr, ys, 2)
    pred = np.polyval([a2, a1, a0], d_arr)

    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    # De-normalize coefficients
    A2 = float(a2 * s)
    A1 = float(a1 * s)
    A0 = float(a0 * s + m)
    coeffs = (A0, A1, A2)

    if A2 >= 0:
        return False, "fit is convex (no maximum) - window not centered on a peak", coeffs, float("nan"), float("nan"), r2

    # Vertex position
    d_star = -A1 / (2.0 * A2)
    value_star = A0 + A1 * d_star + A2 * (d_star ** 2)

    # Check bounds
    if not (d_arr.min() <= d_star <= d_arr.max()):
        return False, "vertex lies outside the fitting window", coeffs, float(d_star), float(value_star), r2

    return True, "ok", coeffs, float(d_star), float(value_star), r2


def quadratic_fit(
    eval_func: Callable[..., float],
    coarse_idx: int,
    max_idx: int,
    half_window: int = 12,
    wd: Optional[Sequence[float]] = None,
    evals: Optional[Dict[int, float]] = None,
    stride: int = 1,
    fmo_name: Optional[str] = None,
) -> QuadraticFitResult:
    """
    Performs fine autofocus refinement using quadratic polynomial curve fitting.

    Fits f(d) = A_0 + A_1 * d + A_2 * d^2 on frames [coarse_idx - half_window, coarse_idx + half_window].

    Args:
        eval_func: Focus evaluation function:
                   eval_func(idx) or eval_func(idx, stride) or eval_func(idx, stride, fmo).
        coarse_idx: Discrete focal peak index from coarse search.
        max_idx: Maximum valid frame index (total frames - 1).
        half_window: Number of frames on either side of coarse_idx to include (default 12).
        wd: Optional sequence or array of physical working distances (in mm or um).
            If None, uniform index-based spacing is used.
        evals: Optional dictionary of pre-evaluated {idx: score} to reuse.
        stride: Pixel subsampling stride during evaluation (default 1 = full resolution).
        fmo_name: Optional focus measure operator identifier.

    Returns:
        QuadraticFitResult:
            - ok: True if valid concave maximum was located within window.
            - idx: Sub-frame continuous peak index (e.g. 64.35).
            - value: Focus score interpolated at vertex.
            - wd: Physical working distance at vertex (if wd provided, else same as idx).
            - r2: Coefficient of determination R^2.
            - coeffs: Tuple (A0, A1, A2) where f(d) = A0 + A1*d + A2*d^2.
            - lo, hi: Evaluated frame window bounds.
            - evals: Dictionary of all evaluated {idx: score} pairs.
    """
    if evals is None:
        evals = {}

    coarse_idx = max(0, min(max_idx, int(round(coarse_idx))))
    lo = max(0, coarse_idx - half_window)
    hi = min(max_idx, coarse_idx + half_window)

    frame_indices = list(range(lo, hi + 1))
    if len(frame_indices) < 5:
        return QuadraticFitResult(
            ok=False,
            msg="window too small for a quadratic fit",
            idx=float(coarse_idx),
            value=evals.get(coarse_idx, float("nan")),
            wd=float(wd[coarse_idx]) if wd is not None else float(coarse_idx),
            coeffs=None,
            r2=float("nan"),
            lo=lo,
            hi=hi,
            evals=evals
        )

    # Evaluate all frames within window, reusing existing cache
    for idx in frame_indices:
        if idx not in evals:
            evals[idx] = _invoke_eval(eval_func, idx, stride=stride, fmo_name=fmo_name)

    y = np.array([evals[i] for i in frame_indices], dtype=np.float64)

    # Define relative coordinate d
    if wd is not None and len(wd) > max_idx:
        wd_arr = np.asarray(wd, dtype=np.float64)
        d = wd_arr[frame_indices] - wd_arr[coarse_idx]
        step = (wd_arr[-1] - wd_arr[0]) / max(1, max_idx)
    else:
        d = np.array(frame_indices, dtype=np.float64) - float(coarse_idx)
        step = 1.0

    ok, msg, coeffs, d_star, value_star, r2 = fit_quadratic_vertex(d, y)

    if not ok:
        fallback_wd = float(wd[coarse_idx]) if wd is not None else float(coarse_idx)
        fallback_val = float(evals.get(coarse_idx, float("nan")))
        return QuadraticFitResult(
            ok=False,
            msg=msg,
            idx=float(coarse_idx),
            value=fallback_val,
            wd=fallback_wd,
            coeffs=coeffs,
            r2=r2,
            lo=lo,
            hi=hi,
            evals=evals
        )

    # Compute continuous sub-frame peak and physical WD
    if wd is not None and len(wd) > max_idx:
        wd_star = float(wd[coarse_idx] + d_star)
        idx_star = float(coarse_idx + d_star / step)
    else:
        wd_star = float(coarse_idx + d_star)
        idx_star = float(coarse_idx + d_star)

    return QuadraticFitResult(
        ok=True,
        msg="ok",
        idx=idx_star,
        value=value_star,
        wd=wd_star,
        coeffs=coeffs,
        r2=r2,
        lo=lo,
        hi=hi,
        evals=evals
    )


if __name__ == "__main__":
    print("Testing Quadratic Curve Fitting Algorithm...")

    TRUE_CONTINUOUS_PEAK = 73.42  # Sub-frame peak
    TOTAL_FRAMES = 200
    MAX_IDX = TOTAL_FRAMES - 1

    def synthetic_eval(idx: int, stride: int = 1, fmo: Optional[str] = None) -> float:
        # Gaussian curve with known true sub-frame continuous peak
        sigma = 8.0
        val = 100.0 * np.exp(-((idx - TRUE_CONTINUOUS_PEAK) ** 2) / (2 * sigma ** 2))
        return float(val)

    COARSE_PEAK = int(round(TRUE_CONTINUOUS_PEAK))  # frame 73

    # Run quadratic fit with half_window = 10
    fit_res = quadratic_fit(synthetic_eval, coarse_idx=COARSE_PEAK, max_idx=MAX_IDX, half_window=10)

    print(f"Discrete Coarse Peak  : Frame {COARSE_PEAK}")
    print(f"True Continuous Peak  : {TRUE_CONTINUOUS_PEAK:.3f}")
    print(f"Fitted Sub-Frame Peak : {fit_res.idx:.3f}")
    print(f"Fitted Peak Focus     : {fit_res.value:.4f}")
    print(f"Fit Status (ok)       : {fit_res.ok} ({fit_res.msg})")
    print(f"Goodness of Fit R^2   : {fit_res.r2:.5f}")
    print(f"Coefficients (A0,A1,A2: {fit_res.coeffs}")
    print(f"Fitting Window        : [{fit_res.lo}, {fit_res.hi}] ({len(fit_res.evals)} points)")

    # Dict compatibility test
    assert fit_res["ok"] == fit_res.ok
    assert fit_res["idx"] == fit_res.idx

    subframe_error = abs(fit_res.idx - TRUE_CONTINUOUS_PEAK)
    print(f"Sub-Frame Error       : {subframe_error:.4f} frames")
    assert fit_res.ok is True, f"Fit failed: {fit_res.msg}"
    assert subframe_error < 0.15, f"Sub-frame error {subframe_error} >= 0.15 frames"
    assert fit_res.r2 > 0.95, f"R2 {fit_res.r2} <= 0.95"
    print("\n[OK] Quadratic Curve Fitting tests passed successfully.")
