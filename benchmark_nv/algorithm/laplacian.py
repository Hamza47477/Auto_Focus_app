"""
Laplacian Focus Score Algorithms

Laplacian-based focus measures evaluate second-order spatial derivatives (inflection
points and edge curvature). In-focus micrographs have sharp, steep transitions yielding
high Laplacian variance, whereas out-of-focus images appear smoothed with near-zero
curvature.

Three primary formulations are included:
1. Variance of Laplacian (VoL):
   Computes Var(Laplacian(I, ksize)). Standard, fast, widely used in computer vision.

2. Laplacian of Gaussian (LoG):
   Applies a Gaussian smoothing kernel prior to the Laplacian operator to suppress
   high-frequency sensor / electron shot noise before measuring edge inflection.

3. Sum of Modified Laplacian (SML):
   Gaussian pre-smoothed absolute second-order partial derivatives:
     (|d2I/dx2| + |d2I/dy2|)^2
   Eliminates trace cancellation at saddle points where d2I/dx2 and d2I/dy2 have
   opposite signs.
"""

from __future__ import annotations
from typing import Optional
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


def variance_of_laplacian(img: np.ndarray, ksize: int = 3) -> float:
    """
    Computes the Variance of Laplacian (VoL) focus measure.

    Highlights inflection points of edges. High variance indicates a sharp,
    in-focus image, whereas low variance indicates blur.

    Args:
        img: Input image as a 2D grayscale array (or 3D BGR image).
        ksize: Aperture size for the Laplacian operator (must be 1, 3, 5, or 7).

    Returns:
        float: Variance of the Laplacian response.
    """
    if img is None or img.size == 0:
        return 0.0

    gray = _to_grayscale(img)
    img_f64 = gray.astype(np.float64)
    lap = cv2.Laplacian(img_f64, cv2.CV_64F, ksize=ksize)
    return float(np.var(lap))


def laplacian_of_gaussian(
    img: np.ndarray,
    ksize: int = 3,
    blur_ksize: int = 5,
    sigma: float = 1.0
) -> float:
    """
    Computes the Laplacian of Gaussian (LoG) focus measure.

    Acts as a bandpass filter: Gaussian blur attenuates high-frequency noise
    (e.g., SEM electron shot noise) before the second derivative is computed.

    Args:
        img: Input image as a 2D grayscale array (or 3D BGR image).
        ksize: Aperture size for the Laplacian operator.
        blur_ksize: Gaussian filter aperture size (must be positive odd integer).
        sigma: Gaussian kernel standard deviation.

    Returns:
        float: Variance of the LoG filtered response.
    """
    if img is None or img.size == 0:
        return 0.0

    gray = _to_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
    return float(np.var(lap))


def modified_laplacian(
    img: np.ndarray,
    sigma: float = 1.0,
    ksize_blur: int = 3
) -> float:
    """
    Computes Gaussian-Smoothed Modified Laplacian (SML).

    Instead of the standard Laplacian trace (d2I/dx2 + d2I/dy2), SML sums the
    absolute values of partial second derivatives:
        Phi = Mean( (|d2I/dx2| + |d2I/dy2|)^2 )
    This prevents cancellation at saddle points where orthogonal curvatures
    have opposite signs.

    Args:
        img: Input image as a 2D grayscale array (or 3D BGR image).
        sigma: Gaussian pre-smoothing standard deviation.
        ksize_blur: Gaussian kernel size (must be positive odd integer).

    Returns:
        float: Modified Laplacian energy.
    """
    if img is None or img.size == 0:
        return 0.0

    gray = _to_grayscale(img)
    s = cv2.GaussianBlur(gray, (ksize_blur, ksize_blur), sigma).astype(np.float32)
    h, w = s.shape[:2]
    if h < 3 or w < 3:
        return 0.0

    # Finite difference second partial derivatives:
    # d2I/dx2 = I(x+1, y) - 2*I(x, y) + I(x-1, y)
    # d2I/dy2 = I(x, y+1) - 2*I(x, y) + I(x, y-1)
    lx = np.abs(s[:, 2:] - 2.0 * s[:, 1:-1] + s[:, :-2])[1:-1, :]
    ly = np.abs(s[2:, :] - 2.0 * s[1:-1, :] + s[:-2, :])[:, 1:-1]

    return float(np.mean((lx + ly) ** 2))


def laplacian_focus_score(
    img: np.ndarray,
    method: str = "variance",
    **kwargs
) -> float:
    """
    Unified entry point for Laplacian-based focus scoring.

    Args:
        img: Input image as a NumPy array.
        method: One of 'variance' (VoL), 'log' (Laplacian of Gaussian),
                or 'modified' / 'sml' (Modified Laplacian).
        **kwargs: Additional parameters passed to the underlying algorithm
                  (e.g., ksize, blur_ksize, sigma).

    Returns:
        float: Focus score.
    """
    method_lower = method.lower().strip()
    if method_lower in ("variance", "vol", "standard"):
        ksize = kwargs.get("ksize", 3)
        return variance_of_laplacian(img, ksize=ksize)
    elif method_lower in ("log", "laplacian_of_gaussian", "gaussian"):
        ksize = kwargs.get("ksize", 3)
        blur_ksize = kwargs.get("blur_ksize", 5)
        sigma = kwargs.get("sigma", 1.0)
        return laplacian_of_gaussian(img, ksize=ksize, blur_ksize=blur_ksize, sigma=sigma)
    elif method_lower in ("modified", "sml", "m3"):
        sigma = kwargs.get("sigma", 1.0)
        ksize_blur = kwargs.get("ksize_blur", 3)
        return modified_laplacian(img, sigma=sigma, ksize_blur=ksize_blur)
    else:
        raise ValueError(
            f"Unknown Laplacian method '{method}'. Supported: 'variance', 'log', 'modified'"
        )


if __name__ == "__main__":
    print("Testing Laplacian Focus Score Algorithms...")

    # Create synthetic sharp vs blurred test images
    np.random.seed(42)
    sharp_image = np.zeros((128, 128), dtype=np.uint8)
    sharp_image[::16, :] = 255
    sharp_image[:, ::16] = 255
    sharp_image[32:96, 32:96] = 200

    blurred_image = cv2.GaussianBlur(sharp_image, (25, 25), 8.0)

    vol_sharp = variance_of_laplacian(sharp_image)
    vol_blur = variance_of_laplacian(blurred_image)

    log_sharp = laplacian_of_gaussian(sharp_image)
    log_blur = laplacian_of_gaussian(blurred_image)

    sml_sharp = modified_laplacian(sharp_image)
    sml_blur = modified_laplacian(blurred_image)

    print(f"Variance of Laplacian (VoL):")
    print(f"  Sharp Image Score   : {vol_sharp:.2f}")
    print(f"  Blurred Image Score : {vol_blur:.2f}")
    print(f"  Discrimination Ratio: {vol_sharp / max(1e-6, vol_blur):.2f}x")

    print(f"\nLaplacian of Gaussian (LoG):")
    print(f"  Sharp Image Score   : {log_sharp:.2f}")
    print(f"  Blurred Image Score : {log_blur:.2f}")
    print(f"  Discrimination Ratio: {log_sharp / max(1e-6, log_blur):.2f}x")

    print(f"\nModified Laplacian (SML / M3):")
    print(f"  Sharp Image Score   : {sml_sharp:.2f}")
    print(f"  Blurred Image Score : {sml_blur:.2f}")
    print(f"  Discrimination Ratio: {sml_sharp / max(1e-6, sml_blur):.2f}x")

    assert vol_sharp > vol_blur, "VoL sharp score must exceed blurred score"
    assert log_sharp > log_blur, "LoG sharp score must exceed blurred score"
    assert sml_sharp > sml_blur, "SML sharp score must exceed blurred score"
    print("\n[OK] All Laplacian focus score tests passed successfully.")
