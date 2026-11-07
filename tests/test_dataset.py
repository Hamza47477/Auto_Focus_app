"""Tests for listing, natural sorting and index <-> position mapping."""

from __future__ import annotations

import pytest

from autofocus import ImageDataset
from autofocus.dataset import list_image_files


def test_natural_sort_order(tmp_path):
    names = ["img10.png", "img2.png", "img1.png", "img20.png", "img3.png"]
    for name in names:
        (tmp_path / name).write_bytes(b"not a real image, only the name matters here")

    files = list_image_files(tmp_path)
    assert [p.name for p in files] == ["img1.png", "img2.png", "img3.png", "img10.png", "img20.png"]


def test_non_image_files_are_ignored(tmp_path):
    (tmp_path / "img1.png").write_bytes(b"x")
    (tmp_path / "readme.txt").write_bytes(b"x")
    (tmp_path / "notes.yml").write_bytes(b"x")

    files = list_image_files(tmp_path)
    assert [p.name for p in files] == ["img1.png"]


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageDataset(tmp_path / "does_not_exist")


def test_empty_folder_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        ImageDataset(empty)


def test_position_mm_formula(tmp_path):
    (tmp_path / "img0.png").write_bytes(b"x")
    dataset = ImageDataset(tmp_path, d_min_mm=10.0, step_mm=0.01)

    assert dataset.position_mm(0) == pytest.approx(10.0)
    assert dataset.position_mm(5) == pytest.approx(10.05)
    assert dataset.position_mm(2.5) == pytest.approx(10.025)  # fractional index, used by the fine fit


def test_path_for_out_of_range_raises(tmp_path):
    (tmp_path / "img0.png").write_bytes(b"x")
    dataset = ImageDataset(tmp_path)
    with pytest.raises(IndexError):
        dataset.path_for(5)
