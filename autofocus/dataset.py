"""Listing, natural sorting and index <-> position mapping.

Nothing in this module reads pixels. It only looks at file names, which is
what makes the "lazy image access" requirement possible: a dataset can be
opened and indexed without touching the (potentially large) image files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Union

#: Extensions recognised as dataset images (case-insensitive).
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

_NUMBER_RE = re.compile(r"(\d+)")


def _natural_key(path: Path) -> tuple:
    """Sort key that orders ``img2`` before ``img10``.

    Splits the file name on runs of digits and converts each digit run to
    an int, so comparisons are numeric where the name is numeric and
    lexicographic everywhere else.
    """
    parts = _NUMBER_RE.split(path.name)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def list_image_files(folder: Path) -> List[Path]:
    """Return the dataset's image files, naturally sorted.

    Args:
        folder: Directory to scan (non-recursive).

    Returns:
        Image file paths in natural number order.
    """
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=_natural_key)
    return files


class ImageDataset:
    """A folder of images, listed once, read lazily by a :class:`~autofocus.focus.FocusEvaluator`.

    Only the file list is built on construction; no image is decoded here.
    """

    def __init__(self, folder: Union[str, Path], d_min_mm: float = 10.0, step_mm: float = 0.010) -> None:
        """Open a dataset folder.

        Args:
            folder: Path to the image folder.
            d_min_mm: Working distance of index 0, in mm (section 1).
            step_mm: Working-distance increment per index, in mm (section 1).

        Raises:
            FileNotFoundError: ``folder`` does not exist or is not a directory.
            ValueError: ``folder`` contains no supported image files.
        """
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self.folder}")

        self.files: List[Path] = list_image_files(self.folder)
        if not self.files:
            raise ValueError(f"No supported images ({', '.join(IMAGE_EXTENSIONS)}) found in: {self.folder}")

        self.d_min_mm = d_min_mm
        self.step_mm = step_mm

    def __len__(self) -> int:
        return len(self.files)

    def path_for(self, index: int) -> Path:
        """Return the file path of image ``index`` without reading it.

        Raises:
            IndexError: ``index`` is outside ``[0, len(self) - 1]``.
        """
        if not (0 <= index < len(self.files)):
            raise IndexError(f"image index {index} out of range [0, {len(self.files) - 1}]")
        return self.files[index]

    def position_mm(self, index: Union[int, float]) -> float:
        """Working distance of ``index``, in mm.

        Implements ``d(i) = d_min_mm + i * step_mm`` (section 1). ``index``
        may be fractional, which is how the quadratic fit's sub-index peak
        is converted to a position.
        """
        return self.d_min_mm + float(index) * self.step_mm
