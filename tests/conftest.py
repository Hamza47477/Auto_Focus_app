"""Synthetic image-stack generator shared by the test suite.

Builds a small stack of PNG images on disk whose normalized-variance focus
curve has one clear global peak (increasing blur sigma with distance from
it) plus a smaller secondary local maximum elsewhere, so the tests can
exercise the local-maximum protections without needing the real dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest


def make_synthetic_stack(
    tmp_path: Path,
    n: int = 60,
    peak: int = 40,
    bump: int = 10,
    size: Tuple[int, int] = (96, 128),
    seed: int = 0,
    dirname: str = "stack",
) -> Path:
    """Write a synthetic image stack to ``tmp_path / dirname`` and return its path.

    Image ``i`` is a fixed textured pattern blurred with sigma growing with
    ``|i - peak|`` (so ``peak`` is the sharpest, global-maximum image),
    with a secondary, smaller sharpening applied near ``bump`` to create a
    local maximum that is clearly lower than the global one.
    """
    rng = np.random.default_rng(seed)
    texture = rng.integers(40, 216, size=size).astype(np.float64)

    folder = tmp_path / dirname
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        dist_main = abs(i - peak)
        sigma_main = 0.35 * dist_main
        main_img = cv2.GaussianBlur(texture, (0, 0), sigmaX=max(sigma_main, 0.35))

        dist_bump = abs(i - bump)
        if dist_bump <= 4:
            # Blend in a sharper (but still capped) version near the bump so
            # it forms a local maximum well below the global peak's focus.
            sigma_bump = 1.2 + 0.5 * dist_bump
            bump_img = cv2.GaussianBlur(texture, (0, 0), sigmaX=sigma_bump)
            weight = (5 - dist_bump) / 5.0  # 1.0 at the bump, 0 four images away
            img = (1.0 - 0.55 * weight) * main_img + (0.55 * weight) * bump_img
        else:
            img = main_img

        noise = rng.normal(0, 2.5, size=size)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(folder / f"img{i}.png"), img)

    return folder


@pytest.fixture
def synthetic_stack(tmp_path):
    """Default-sized synthetic stack: 60 images, global peak at 40, bump at 10."""
    return make_synthetic_stack(tmp_path, n=60, peak=40, bump=10, seed=0), 40, 10
