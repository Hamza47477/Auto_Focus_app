"""Normalized-variance focus measure and the lazy, cached image reader.

Only :func:`normalized_variance` (Eq. 6.5) is implemented as a focus
measure, as required by the assignment. Optional preprocessing (info-bar
crop, denoise) happens before the measure is computed, never inside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from .dataset import ImageDataset
from .results import TraceStep
from .settings import AutofocusSettings


def normalized_variance(gray_f64: np.ndarray) -> float:
    """Normalized variance focus measure, course Eq. 6.5.

    ``f(I) = (1 / (M*N*mu)) * sum((I - mu) ** 2)``, mu = mean intensity.

    Args:
        gray_f64: Grayscale image as a float64 array.

    Returns:
        The focus value, or 0.0 if the mean intensity is 0.
    """
    mu = float(gray_f64.mean())
    if mu == 0.0:
        return 0.0
    m, n = gray_f64.shape
    return float(np.sum((gray_f64 - mu) ** 2) / (m * n * mu))


def _preprocess(gray_u8: np.ndarray, settings: AutofocusSettings) -> np.ndarray:
    """Apply the optional info-bar crop and denoise, in that order."""
    out = gray_u8

    if settings.do_crop_bottom and settings.crop_bottom_px > 0:
        crop = min(settings.crop_bottom_px, out.shape[0] - 1)
        out = out[: out.shape[0] - crop, :]

    if settings.do_denoise:
        ksize = settings.denoise_ksize
        if ksize % 2 == 0:
            ksize += 1
        if settings.denoise_method == "median":
            out = cv2.medianBlur(out, ksize)
        else:
            out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    return out


def compute_focus(path: Path, settings: AutofocusSettings) -> float:
    """Read one image from disk and compute its normalized-variance focus value.

    Raises:
        IOError: The file exists but could not be decoded as an image.
    """
    img_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise IOError(f"Could not read image: {path}")
    img_u8 = _preprocess(img_u8, settings)
    return normalized_variance(img_u8.astype(np.float64))


class FocusEvaluator:
    """Lazy, cached access to the focus value of each dataset image.

    An image is decoded at most once: the first call to :meth:`evaluate`
    for a given index reads the file and computes the normalized variance;
    every later call for the same index returns the cached value. The
    number of distinct images decoded is available as :attr:`images_read`,
    and every call (cached or not) is appended to :attr:`trace` and
    forwarded to ``on_step`` so a GUI can draw points live.
    """

    def __init__(
        self,
        dataset: ImageDataset,
        settings: AutofocusSettings,
        on_step: Optional[Callable[[TraceStep], None]] = None,
    ) -> None:
        self.dataset = dataset
        self.settings = settings
        self.on_step = on_step
        self._cache: Dict[int, float] = {}
        self.trace: list = []

    @property
    def images_read(self) -> int:
        """Number of distinct images decoded so far."""
        return len(self._cache)

    def evaluate(self, index: int, phase: str, note: str = "") -> float:
        """Return the focus value of ``index``, reading the image only if needed.

        Args:
            index: Image index to evaluate.
            phase: One of ``results.TRACE_PHASES``, recorded in the trace.
            note: Short free-text note, recorded in the trace.

        Raises:
            IndexError: ``index`` is outside the dataset's range.
            IOError: The image file could not be decoded.
        """
        if index not in self._cache:
            path = self.dataset.path_for(index)  # raises IndexError if out of range
            self._cache[index] = compute_focus(path, self.settings)

        value = self._cache[index]
        step = TraceStep(
            step_no=len(self.trace) + 1,
            index=index,
            position_mm=self.dataset.position_mm(index),
            focus=value,
            phase=phase,
            note=note,
        )
        self.trace.append(step)
        if self.on_step is not None:
            self.on_step(step)
        return value
