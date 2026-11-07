"""
Normalized Variance Focus Score Algorithm

Normalized Variance measures the spread of pixel intensities relative to the image
mean. In microscopy, out-of-focus images exhibit low contrast dispersion (low variance),
while in-focus images display sharp structural edges and high intensity divergence.

Two formulations are supported:
1. Standard Normalized Variance:
      Score = Var(I) / (Mean(I) + eps)
   Computes intensity variance normalized by the first moment of brightness.

2. Mean-Normalized Variance:
      Score = Var(I) / (Mean(I)^2 + eps)
   Normalizes by Mean(I)^2, making the score invariant to global illumination
   scaling and beam current fluctuations.
"""

from __future__ import annotations
from typing import Union
import cv2
import numpy as np


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Converts multi-channel image to 2D grayscale if necessary."""
    if img.ndim == 3:
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 1:
            return img[:, :, 0]
    return img


def normalized_variance(
    img: np.ndarray,
    squared_mean: bool = False,
    eps: float = 1e-5
) -> float:
    """
    Computes the Normalized Variance focus score.

    Args:
        img: Input image as a 2D grayscale NumPy array (or 3D BGR image).
        squared_mean: If False (default), computes Var(I) / (Mean(I) + eps).
                      If True, computes Var(I) / (Mean(I)^2 + eps) (illumination invariant).
        eps: Small epsilon to prevent division by zero.

    Returns:
        float: Computed focus score (higher value indicates better focus).

    Example:
        >>> score = normalized_variance(image)
        >>> invariant_score = normalized_variance(image, squared_mean=True)
    """
    if img is None or img.size == 0:
        return 0.0

    gray = _to_grayscale(img)
    x = gray.astype(np.float32)

    mean_val = float(np.mean(x))
    var_val = float(np.var(x))

    if squared_mean:
        # Scale/exposure invariant formulation
        denominator = (mean_val * mean_val) + eps
    else:
        # Classical normalized variance formulation
        denominator = mean_val + eps

    return float(var_val / denominator)


def mean_normalized_variance(img: np.ndarray, eps: float = 1e-5) -> float:
    """
    Convenience function for Mean-Normalized Variance:
        Score = Var(I) / (Mean(I)^2 + eps)

    Provides illumination-scale invariance, highly suited for coarse autofocus sweeps
    due to its broad monotonic basin.

    Args:
        img: Input image as a NumPy array.
        eps: Stability constant against division by zero.

    Returns:
        float: Mean-normalized variance focus score.
    """
    return normalized_variance(img, squared_mean=True, eps=eps)


if __name__ == "__main__":
    print("Testing Normalized Variance Algorithm...")

    # Create synthetic sharp vs blurred test images
    np.random.seed(42)
    sharp_image = np.zeros((128, 128), dtype=np.uint8)
    # Add high-contrast checkerboard / edges
    sharp_image[::16, :] = 255
    sharp_image[:, ::16] = 255
    sharp_image[32:96, 32:96] = 200

    # Defocused image via Gaussian blur
    blurred_image = cv2.GaussianBlur(sharp_image, (25, 25), 8.0)

    score_sharp_std = normalized_variance(sharp_image, squared_mean=False)
    score_blur_std = normalized_variance(blurred_image, squared_mean=False)

    score_sharp_m9 = mean_normalized_variance(sharp_image)
    score_blur_m9 = mean_normalized_variance(blurred_image)

    print(f"Standard Normalized Variance:")
    print(f"  Sharp Image Score   : {score_sharp_std:.4f}")
    print(f"  Blurred Image Score : {score_blur_std:.4f}")
    print(f"  Discrimination Ratio: {score_sharp_std / max(1e-6, score_blur_std):.2f}x")

    print(f"\nMean-Normalized Variance (M9 Invariant):")
    print(f"  Sharp Image Score   : {score_sharp_m9:.6f}")
    print(f"  Blurred Image Score : {score_blur_m9:.6f}")
    print(f"  Discrimination Ratio: {score_sharp_m9 / max(1e-6, score_blur_m9):.2f}x")
    assert score_sharp_std > score_blur_std, "Sharp score should exceed blurred score"
    assert score_sharp_m9 > score_blur_m9, "Sharp score should exceed blurred score"
    print("\n[OK] Normalized Variance tests passed successfully.")
