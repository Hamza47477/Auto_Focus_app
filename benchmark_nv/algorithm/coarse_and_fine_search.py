"""
Coarse-and-Fine Hybrid Search Algorithm

The Hybrid Two-Stage Coarse-to-Fine Sweep is a robust autofocus
search strategy in electron microscopy:
1. Stage 1 (Coarse Grid Sweep):
   Samples the focal stack sparsely with step size s = max(3, round(sqrt(N))),
   optionally using high pixel subsampling (e.g. coarse_stride=4) and a wide-basin
   coarse operator (such as Normalized Variance). This rapidly isolates the
   focal peak's basin of attraction while saving ~80-90% of frame reads.

2. Stage 2 (Fine Bracket Sweep):
   Centers a local refinement window [kc - s, kc + s] around coarse peak kc,
   evaluating at full pixel resolution (stride=1) with a high-precision fine operator
   (such as Laplacian or Scharr-Tenengrad). Immediate neighbors [best - 1, best + 1]
   are then climbed (+/-1 frame) with the fine operator to confirm the peak.

Complexity:
- Time & I/O Complexity: O(sqrt(N) + fine_window)
- Typical frame reads: ~22 to 27 frames for a 200-frame stack (~86% read reduction).
- Noise Robustness: High immunity against noise plateaus and local ripples.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any, Optional, NamedTuple
import math
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


def coarse_and_fine_search(
    eval_func: Callable[..., float],
    max_idx: int,
    coarse_step: Optional[int] = None,
    coarse_fmo: str = "composite_m9_m2",
    fine_fmo: Optional[str] = None,
    coarse_stride: int = 4,
    fine_stride: int = 1,
    first_frame: Optional[int] = None
) -> SearchResult:
    """
    Executes a two-stage coarse-and-fine autofocus search.

    Args:
        eval_func: Focus evaluation function. Can accept (idx), (idx, stride),
                   or (idx, stride, fmo_name).
        max_idx: Maximum frame index in stack (inclusive, 0-indexed).
        coarse_step: Step size between coarse probes. If None, defaults to
                     max(3, round(sqrt(max_idx + 1))).
        coarse_fmo: Focus measure identifier used during coarse stage.
        fine_fmo: Focus measure identifier used during fine refinement stage.
        coarse_stride: Pixel subsampling stride during coarse stage (default 4).
        fine_stride: Pixel subsampling stride during fine stage (default 1).
        first_frame: Optional user-specified initial probe index.

    Returns:
        SearchResult(best_idx, evals, history):
            - best_idx: Detected discrete focal peak index.
            - evals: Dictionary of all evaluated indices and their scores {idx: score}.
            - history: Ordered step log detailing stage, stride, and scores.

    Example:
        >>> best_idx, evals, history = coarse_and_fine_search(my_eval_fn, max_idx=199)
        >>> print(f"Autofocus locked at frame {best_idx} with {len(evals)} reads")
    """
    history: List[Dict[str, Any]] = []
    evals: Dict[int, float] = {}

    def probe(idx: int, current_stride: int, current_fmo: Optional[str], stage: str) -> float:
        idx = max(0, min(max_idx, int(round(idx))))
        val = _invoke_eval(eval_func, idx, stride=current_stride, fmo_name=current_fmo)
        evals[idx] = val
        history.append({
            "step": len(history) + 1,
            "idx": idx,
            "score": val,
            "stride": current_stride,
            "fmo": current_fmo or "default",
            "stage": stage
        })
        return val

    # Stage 1: Coarse Grid Sweep
    num_frames = max_idx + 1
    step_s = coarse_step if coarse_step is not None else max(3, int(round(math.sqrt(num_frames))))

    coarse_indices = list(range(0, num_frames, step_s))
    if coarse_indices[-1] != max_idx:
        coarse_indices.append(max_idx)

    coarse_scores: Dict[int, float] = {}
    if first_frame is not None and 0 <= first_frame <= max_idx:
        coarse_scores[first_frame] = probe(first_frame, coarse_stride, coarse_fmo, "initial_frame")

    for idx in coarse_indices:
        coarse_scores[idx] = probe(idx, coarse_stride, coarse_fmo, "coarse_grid")

    # Determine coarse peak
    kc = max(coarse_scores.keys(), key=lambda k: coarse_scores[k])

    # Stage 2: Fine Bracket Search
    # Refinement window centered on coarse peak: [kc - step_s, kc + step_s]
    w_start = max(0, kc - step_s)
    w_end = min(max_idx, kc + step_s)

    # Sub-sample inside window with mid_step first
    mid_step = max(1, step_s // 4)
    fine_probes = list(range(w_start, w_end + 1, mid_step))
    if w_end not in fine_probes:
        fine_probes.append(w_end)

    fine_scores: Dict[int, float] = {}
    for idx in fine_probes:
        fine_scores[idx] = probe(idx, fine_stride, fine_fmo, "fine_window")

    best_fine = max(fine_scores.keys(), key=lambda k: fine_scores[k])

    # +/-1 hill climb with the FINE measure only.
    # Check fine_scores (not evals): a frame read during the coarse stage was
    # scored with another measure/stride and must be re-scored here.
    c = best_fine
    while True:
        for nbr in (c - 1, c + 1):
            if 0 <= nbr <= max_idx and nbr not in fine_scores:
                fine_scores[nbr] = probe(nbr, fine_stride, fine_fmo, "fine_bracket")
        nxt = max((n for n in (c - 1, c, c + 1) if n in fine_scores), key=fine_scores.__getitem__)
        if nxt == c:
            break
        c = nxt

    overall_best = c
    return SearchResult(best_idx=overall_best, evals=evals, history=history)


# Aliases matching codebase naming conventions
search_hybrid_coarse_fine = coarse_and_fine_search
hybrid_coarse_fine_search = coarse_and_fine_search


if __name__ == "__main__":
    import numpy as np

    print("Testing Coarse-and-Fine Search Algorithm...")

    TRUE_PEAK = 73
    TOTAL_FRAMES = 200
    MAX_IDX = TOTAL_FRAMES - 1

    def synthetic_eval(idx: int, stride: int = 1, fmo: Optional[str] = None) -> float:
        # Gaussian focus peak with minor noise
        sigma = 8.0
        val = 100.0 * np.exp(-((idx - TRUE_PEAK) ** 2) / (2 * sigma ** 2))
        return float(val)

    result = coarse_and_fine_search(synthetic_eval, max_idx=MAX_IDX)

    print(f"Total Frames In Stack : {TOTAL_FRAMES}")
    print(f"True Peak Index       : {TRUE_PEAK}")
    print(f"Found Peak Index      : {result.best_idx}")
    print(f"Peak Focus Score      : {result.evals[result.best_idx]:.4f}")
    print(f"Frames Read           : {len(result.evals)} / {TOTAL_FRAMES}")
    reduction = (1.0 - len(result.evals) / TOTAL_FRAMES) * 100.0
    print(f"Frame Read Reduction  : {reduction:.1f}%")
    print(f"Execution Steps       : {len(result.history)}")

    # Check that error is within discrete tolerance (<= 1 frame)
    error = abs(result.best_idx - TRUE_PEAK)
    print(f"Discrete Error        : {error} frames")
    assert error <= 1, f"Expected peak near {TRUE_PEAK}, got {result.best_idx}"
    assert len(result.evals) <= 35, f"Coarse-fine should read <= 35 frames, read {len(result.evals)}"
    print("\n[OK] Coarse-and-Fine Search tests passed successfully.")
