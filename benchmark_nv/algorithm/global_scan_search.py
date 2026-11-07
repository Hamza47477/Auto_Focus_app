"""
Global Exhaustive Linear Scan Search Algorithm

The Global Scan algorithm evaluates every single frame in the focal stack sequentially.
While O(N) in computational and I/O cost (requiring 100% of stack frames to be evaluated),
it serves as the ground-truth reference baseline to which sub-linear search algorithms
(such as Coarse-and-Fine, Gradient Search, and Quadratic Fit) are benchmarked.

Features:
- Guaranteed global optimum (no vulnerability to local minima or premature convergence).
- Produces the complete focal curve across all indices [0, max_idx].
- Supports optional first_frame prioritization (e.g. evaluating user-specified probe first).
- Supports flexible evaluation functions:
    * Single argument: eval_func(idx)
    * Two arguments:   eval_func(idx, stride)
    * Three arguments: eval_func(idx, stride, fmo_name)
- Returns a SearchResult NamedTuple unpackable as (best_idx, evals, history).
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any, Optional, NamedTuple
import inspect


class SearchResult(NamedTuple):
    """
    Search result container.
    Unpackable as a standard 3-tuple: (best_idx, evals, history)
    or accessed via named fields: result.best_idx, result.evals, result.history.
    """
    best_idx: int
    evals: Dict[int, float]
    history: List[Dict[str, Any]]


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
        # Fallback to direct trial
        try:
            return float(eval_func(idx, stride, fmo_name))
        except TypeError:
            try:
                return float(eval_func(idx, stride))
            except TypeError:
                return float(eval_func(idx))


def global_scan_search(
    eval_func: Callable[..., float],
    max_idx: int,
    stride: int = 1,
    fmo_name: Optional[str] = None,
    first_frame: Optional[int] = None
) -> SearchResult:
    """
    Executes an exhaustive linear scan across the entire focal range [0, max_idx].

    Args:
        eval_func: Focus evaluation function. Can take (idx), (idx, stride),
                   or (idx, stride, fmo_name).
        max_idx: Maximum frame index in the focal stack (inclusive, 0-indexed).
        stride: Pixel subsampling stride (default 1 = full resolution).
        fmo_name: Optional focus measure operator identifier.
        first_frame: Optional index to probe before beginning sequential scan.

    Returns:
        SearchResult(best_idx, evals, history):
            - best_idx: Index of the frame with the maximum focus score.
            - evals: Dictionary mapping evaluated indices to their scores {idx: score}.
            - history: Ordered list of execution step records.

    Example:
        >>> best_idx, evals, history = global_scan_search(my_eval_fn, max_idx=199)
        >>> print(f"Best focus at frame {best_idx}")
    """
    history: List[Dict[str, Any]] = []
    evals: Dict[int, float] = {}

    best_idx = 0
    best_score = -float("inf")

    # Order of evaluation: probe first_frame first if specified, then remaining indices
    order = list(range(max_idx + 1))
    if first_frame is not None and 0 <= first_frame <= max_idx:
        order = [first_frame] + [i for i in order if i != first_frame]

    for idx in order:
        val = _invoke_eval(eval_func, idx, stride=stride, fmo_name=fmo_name)
        evals[idx] = val
        history.append({
            "step": len(history) + 1,
            "idx": idx,
            "score": val,
            "stride": stride,
            "stage": "initial_frame" if idx == first_frame else "full_linear_scan"
        })
        if val > best_score:
            best_score = val
            best_idx = idx

    return SearchResult(best_idx=best_idx, evals=evals, history=history)


# Canonical alias
global_search = global_scan_search


if __name__ == "__main__":
    print("Testing Global Scan Search Algorithm...")

    # Synthetic focal curve: Gaussian peak centered at frame 42
    TRUE_PEAK = 42
    TOTAL_FRAMES = 100
    MAX_IDX = TOTAL_FRAMES - 1

    def synthetic_eval(idx: int, stride: int = 1, fmo: Optional[str] = None) -> float:
        # Gaussian curve with peak at TRUE_PEAK
        sigma = 6.0
        return float(100.0 * np.exp(-((idx - TRUE_PEAK) ** 2) / (2 * sigma ** 2)))

    import numpy as np

    result = global_scan_search(synthetic_eval, max_idx=MAX_IDX, first_frame=0)

    print(f"Total Frames        : {TOTAL_FRAMES}")
    print(f"True Peak Index     : {TRUE_PEAK}")
    print(f"Found Peak Index    : {result.best_idx}")
    print(f"Peak Focus Score    : {result.evals[result.best_idx]:.4f}")
    print(f"Total Frame Reads   : {len(result.evals)}")
    print(f"Execution Steps     : {len(result.history)}")

    assert result.best_idx == TRUE_PEAK, f"Expected peak {TRUE_PEAK}, got {result.best_idx}"
    assert len(result.evals) == TOTAL_FRAMES, f"Expected {TOTAL_FRAMES} evaluations"
    print("\n[OK] Global Scan Search tests passed successfully.")
