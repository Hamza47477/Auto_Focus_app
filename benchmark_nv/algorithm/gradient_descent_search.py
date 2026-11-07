"""
Gradient Descent / Ascent Autofocus Search Algorithm

Algorithm Overview:
1. Derivative-Based Gradient Ascent:
   Optimizes focal position by climbing the discrete focus curve gradient:
       d_{n+1} = d_n + Gamma_n * Lambda_n
   where:
       Gamma_n  = sign(grad f_n)                         (direction of motion, +/-1)
       Lambda_n = beta * (f0 / fn)^2 if fn > f0 else beta (adaptive decreasing step factor)
       grad f_n = [f(d + h) - f(d - h)] / (2 h)          (centred finite difference)

   The baseline difference span h = max(delta, beta // 2) is wide far from the peak
   to reject high-frequency noise ripples, shrinking to delta near the optimum.
   The step parameter beta starts at alpha frames and is halved each time the search
   overshoots (direction reverses). A final +/- 1 frame hill-climb locks onto the local peak.

2. Fake-Maximum Rejection (Statistical Noise Verification):
   High scanning speeds introduce stochastic noise and local ripples.
   A candidate peak c is verified against symmetric baseline probes at distances
   p in {c - 2*dist, c - dist, c + dist, c + 2*dist}:
       median(f(c-1), f(c), f(c+1)) - max(f(p-1), f(p), f(p+1)) > k * sigma
   Noise sigma is measured on the fly from probe second differences:
       f(p-1) - 2*f(p) + f(p+1) ~ N(0, 6 * sigma^2).

3. Multistart Basin Relocation:
   If a candidate fails verification, the search restarts from the higher probe,
   or relocates to the center of the largest unexplored region in [0, max_idx].

Returns:
   SearchResult container unpackable as (best_idx, evals, history) or accessible via
   named attributes (.best_idx, .evals, .history, .verified, .restarts, .noise_sigma).
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any, Optional, Iterable
import inspect
import math
import numpy as np


class SearchResult:
    """
    Search result container for Gradient Search.
    Fully compatible with standard 3-tuple unpacking:
        best_idx, evals, history = result
    while also exposing rich attributes:
        result.best_idx, result.evals, result.history,
        result.verified, result.restarts, result.noise_sigma,
        result.path, result.rejected.
    """
    def __init__(
        self,
        best_idx: int,
        evals: Dict[int, float],
        history: List[Dict[str, Any]],
        verified: Optional[bool] = None,
        restarts: int = 0,
        noise_sigma: float = float("nan"),
        path: Optional[List[int]] = None,
        rejected: Optional[List[int]] = None,
    ):
        self.best_idx = int(best_idx)
        self.evals = evals
        self.history = history
        self.verified = verified
        self.restarts = restarts
        self.noise_sigma = float(noise_sigma)
        self.path = path or []
        self.rejected = rejected or []

    # Enable standard 3-tuple unpacking: best_idx, evals, history = result
    def __iter__(self):
        return iter((self.best_idx, self.evals, self.history))

    def __getitem__(self, index: int):
        return (self.best_idx, self.evals, self.history)[index]

    def __len__(self) -> int:
        return 3

    def __repr__(self) -> str:
        return (
            f"SearchResult(best_idx={self.best_idx}, evals_count={len(self.evals)}, "
            f"steps={len(self.history)}, verified={self.verified}, restarts={self.restarts})"
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


def _clamp(idx: int, max_idx: int) -> int:
    """Clamps index within [0, max_idx]."""
    return max(0, min(int(max_idx), int(round(idx))))


def _largest_gap_centre(seen: Iterable[int], n_frames: int) -> int:
    """Finds the midpoint of the largest unexplored stretch of frames."""
    pts = sorted(set(seen))
    if not pts:
        return n_frames // 2
    edges = [(0, pts[0])] + list(zip(pts[:-1], pts[1:])) + [(pts[-1], n_frames - 1)]
    a, b = max(edges, key=lambda e: e[1] - e[0])
    return (a + b) // 2


def gradient_ascent(
    eval_func: Callable[..., float],
    max_idx: int,
    start_idx: int,
    alpha: float = 24.0,
    delta: int = 3,
    max_iter: int = 80,
    evals: Optional[Dict[int, float]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    stride: int = 1,
    fmo_name: Optional[str] = None,
) -> Tuple[int, List[int]]:
    """
    Performs single-trajectory derivative-based gradient ascent on the discrete focus curve.

    Args:
        eval_func: Focus evaluation function.
        max_idx: Maximum valid frame index (stack size - 1).
        start_idx: Initial probe frame index.
        alpha: Initial step size in frames (default 24.0).
        delta: Minimum derivative baseline span h (default 3).
        max_iter: Maximum step iterations.
        evals: In-out cache mapping index -> score.
        history: In-out step log.
        stride: Pixel subsampling stride.
        fmo_name: Optional focus measure name.

    Returns:
        (candidate_idx, path)
    """
    if evals is None:
        evals = {}
    if history is None:
        history = []

    def probe(i: int, stage: str = "gradient_step") -> float:
        ci = _clamp(i, max_idx)
        if ci not in evals:
            val = _invoke_eval(eval_func, ci, stride=stride, fmo_name=fmo_name)
            evals[ci] = val
            history.append({
                "step": len(history) + 1,
                "idx": ci,
                "score": val,
                "stride": stride,
                "stage": stage,
            })
        return evals[ci]

    d = _clamp(start_idx, max_idx)
    f0 = probe(d, "gradient_init")
    path = [d]
    beta = float(alpha)
    prev_dir = 0

    for _ in range(max_iter):
        h = max(int(delta), int(beta) // 2, 1)
        lo, hi = _clamp(d - h, max_idx), _clamp(d + h, max_idx)
        if hi == lo:
            break

        flo = probe(lo, "grad_probe_lo")
        fd = probe(d, "grad_probe_curr")
        fhi = probe(hi, "grad_probe_hi")

        g = (fhi - flo) / (hi - lo)
        direction = (g > 0) - (g < 0)
        if direction == 0:
            break

        if prev_dir != 0 and direction != prev_dir:
            beta /= 2.0  # Overshot: halve step size
            if beta < 1.0:
                break
        prev_dir = direction

        ratio = f0 / fd if fd > 0 else 1.0
        lam = beta * (ratio ** 2) if ratio < 1.0 else beta
        step_len = max(1, int(round(lam)))
        nd = _clamp(d + direction * step_len, max_idx)

        if nd == d:
            break  # Reached boundary
        d = nd
        path.append(d)

    # Local hill-climb on +/- 1 frame around best point in neighbourhood
    h = max(int(delta), 1)
    local_candidates = [i for i in evals if abs(i - d) <= h]
    c = max(local_candidates, key=evals.__getitem__) if local_candidates else d

    while True:
        fl = probe(c - 1, "hill_climb_left") if c > 0 else -float("inf")
        fc = probe(c, "hill_climb_center")
        fr = probe(c + 1, "hill_climb_right") if c < max_idx else -float("inf")

        nxt = c - 1 if fl > fc and fl >= fr else (c + 1 if fr > fc else c)
        nxt = _clamp(nxt, max_idx)
        if nxt == c:
            break
        c = nxt
        path.append(c)

    return c, path


def verify_peak(
    eval_func: Callable[..., float],
    max_idx: int,
    candidate: int,
    dist: int = 10,
    k: float = 3.0,
    evals: Optional[Dict[int, float]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    stride: int = 1,
    fmo_name: Optional[str] = None,
) -> Tuple[bool, float, List[int]]:
    """
    Fake-maximum verification test against statistical noise floor.

    A genuine focus peak stands out above the noise floor at probes
    p in {c - 2*dist, c - dist, c + dist, c + 2*dist}:
        median(f(c-1), f(c), f(c+1)) - max(f(p-1), f(p), f(p+1)) > k * sigma

    Returns:
        (is_real_peak, noise_sigma, higher_probes_list)
    """
    if evals is None:
        evals = {}
    if history is None:
        history = []

    def probe(i: int) -> float:
        ci = _clamp(i, max_idx)
        if ci not in evals:
            val = _invoke_eval(eval_func, ci, stride=stride, fmo_name=fmo_name)
            evals[ci] = val
            history.append({
                "step": len(history) + 1,
                "idx": ci,
                "score": val,
                "stride": stride,
                "stage": "verify_probe",
            })
        return evals[ci]

    c = _clamp(candidate, max_idx)
    n = max_idx + 1
    probes = [
        c + s * m * dist
        for m in (1, 2)
        for s in (-1, 1)
        if 0 <= c + s * m * dist < n
    ]

    if not probes or n < 3:
        return True, float("nan"), []

    centre = lambda p: min(max(p, 1), n - 2)
    trip = lambda p: [centre(p) - 1, centre(p), centre(p) + 1]

    # Evaluate triplet around candidate and around each probe
    for idx in trip(c):
        probe(idx)
    for p in probes:
        for idx in trip(p):
            probe(idx)

    top = sorted(evals[i] for i in trip(c))[1]  # Median of candidate triplet
    low = {p: max(evals[i] for i in trip(p)) for p in probes}

    # Estimate noise standard deviation sigma from second differences
    sd = [abs(evals[q - 1] - 2 * evals[q] + evals[q + 1]) for q in map(centre, probes)]
    # E|N(0, 1)| = sqrt(2/pi) approx 0.7979
    sigma = float(np.median(sd)) / math.sqrt(6.0) / 0.7979
    sigma = max(sigma, 1e-9 * abs(top))

    real = all(top - low[p] > k * sigma for p in probes)
    higher = sorted([p for p in probes if low[p] > top], key=low.get, reverse=True)

    return real, sigma, higher


def gradient_descent_search(
    eval_func: Callable[..., float],
    max_idx: int,
    start_idx: Optional[int] = None,
    alpha: float = 24.0,
    delta: int = 3,
    max_iter: int = 80,
    verify_dist: int = 10,
    verify_k: float = 3.0,
    max_restarts: int = 8,
    verify_peaks: bool = True,
    stride: int = 1,
    fmo_name: Optional[str] = None,
) -> SearchResult:
    """
    Executes derivative-based gradient search (gradient ascent/descent)
    with optional fake-maximum rejection and multistart basin relocation.

    Args:
        eval_func: Focus evaluation function:
                   eval_func(idx) or eval_func(idx, stride) or eval_func(idx, stride, fmo).
        max_idx: Maximum frame index in stack (0-indexed, inclusive).
        start_idx: Initial probe frame. If None, defaults to max_idx // 2.
        alpha: Initial decreasing step size beta (default 24.0 frames).
        delta: Minimum centred finite-difference span h (default 3 frames).
        max_iter: Maximum iterations per gradient ascent trajectory (default 80).
        verify_dist: Step distance to verification probes (default 10 frames).
        verify_k: Multiplier for noise sigma threshold (default 3.0).
        max_restarts: Maximum basin relocation restarts if candidate fails verification.
        verify_peaks: Whether to verify candidate against noise floor (default True).
        stride: Pixel subsampling stride during search (default 1).
        fmo_name: Optional focus measure operator identifier.

    Returns:
        SearchResult container:
            - Unpackable as (best_idx, evals, history)
            - Fields: best_idx, evals, history, verified, restarts, noise_sigma, path, rejected.

    Example:
        >>> best_idx, evals, history = gradient_descent_search(my_eval_fn, max_idx=199)
        >>> print(f"Gradient search locked peak at frame {best_idx} ({len(evals)} reads)")
    """
    evals: Dict[int, float] = {}
    history: List[Dict[str, Any]] = []
    num_frames = max_idx + 1

    s = max_idx // 2 if start_idx is None else _clamp(start_idx, max_idx)
    rejected: List[Tuple[int, float]] = []
    path: List[int] = []
    sigma = float("nan")

    if not verify_peaks:
        # Simple gradient ascent without rejection
        c, p = gradient_ascent(
            eval_func, max_idx, s,
            alpha=alpha, delta=delta, max_iter=max_iter,
            evals=evals, history=history, stride=stride, fmo_name=fmo_name
        )
        return SearchResult(
            best_idx=c,
            evals=evals,
            history=history,
            verified=True,
            restarts=0,
            noise_sigma=sigma,
            path=p,
            rejected=[]
        )

    # Gradient ascent with fake-maximum verification and basin restarts
    for attempt in range(max_restarts + 1):
        c, p = gradient_ascent(
            eval_func, max_idx, s,
            alpha=alpha, delta=delta, max_iter=max_iter,
            evals=evals, history=history, stride=stride, fmo_name=fmo_name
        )
        path += p

        real, sigma, higher = verify_peak(
            eval_func, max_idx, c,
            dist=verify_dist, k=verify_k,
            evals=evals, history=history, stride=stride, fmo_name=fmo_name
        )

        if real:
            return SearchResult(
                best_idx=c,
                evals=evals,
                history=history,
                verified=True,
                restarts=attempt,
                noise_sigma=sigma,
                path=path,
                rejected=[r for r, _ in rejected]
            )

        revisit = any(abs(c - r) <= verify_dist for r, _ in rejected)
        rejected.append((c, evals.get(c, 0.0)))

        if higher and not revisit:
            s = higher[0]
        else:
            s = _largest_gap_centre(evals.keys(), num_frames)

    # Fallback to best seen candidate among rejected if no peak satisfied k * sigma
    best = max(rejected, key=lambda t: t[1])[0] if rejected else c
    return SearchResult(
        best_idx=best,
        evals=evals,
        history=history,
        verified=False,
        restarts=max_restarts,
        noise_sigma=sigma,
        path=path,
        rejected=[r for r, _ in rejected if r != best]
    )


# Standard canonical aliases matching terminology across literature and codebases
gradient_ascent_search = gradient_descent_search
gradient_search = gradient_descent_search
coarse_search = gradient_descent_search


if __name__ == "__main__":
    print("Testing Gradient Descent / Ascent Search Algorithm...")

    TRUE_PEAK = 68
    TOTAL_FRAMES = 200
    MAX_IDX = TOTAL_FRAMES - 1

    def synthetic_eval(idx: int, stride: int = 1, fmo: Optional[str] = None) -> float:
        # Gaussian curve with peak at TRUE_PEAK and small noise
        sigma = 7.5
        val = 100.0 * np.exp(-((idx - TRUE_PEAK) ** 2) / (2 * sigma ** 2))
        return float(val)

    # Run search with default middle start (standard autofocus operational mode)
    res = gradient_descent_search(synthetic_eval, max_idx=MAX_IDX)

    print(f"Total Frames        : {TOTAL_FRAMES}")
    print(f"True Peak Index     : {TRUE_PEAK}")
    print(f"Found Peak Index    : {res.best_idx}")
    print(f"Peak Focus Score    : {res.evals[res.best_idx]:.4f}")
    print(f"Peak Verified       : {res.verified}")
    print(f"Restarts            : {res.restarts}")
    print(f"Frames Evaluated    : {len(res.evals)} / {TOTAL_FRAMES}")
    reduction = (1.0 - len(res.evals) / TOTAL_FRAMES) * 100.0
    print(f"Frame Read Reduction: {reduction:.1f}%")
    print(f"Execution Steps     : {len(res.history)}")

    # Unpack test
    best_idx, evals_dict, hist_list = res
    assert best_idx == res.best_idx
    assert len(evals_dict) == len(res.evals)

    error = abs(res.best_idx - TRUE_PEAK)
    print(f"Error               : {error} frames")
    assert error <= 1, f"Expected peak near {TRUE_PEAK}, got {res.best_idx}"
    assert len(res.evals) <= 45, f"Gradient search evaluated {len(res.evals)} frames (> 45)"
    print("\n[OK] Gradient Descent / Ascent Search passed successfully.")
