"""Tests for the normalized-variance focus measure and the lazy evaluator."""

from __future__ import annotations

import numpy as np

from autofocus import AutofocusSettings, FocusEvaluator, ImageDataset, normalized_variance
from conftest import make_synthetic_stack


def test_normalized_variance_higher_for_sharper_image():
    rng = np.random.default_rng(0)
    texture = rng.integers(40, 216, size=(64, 64)).astype(np.float64)

    sharp = texture
    blurry = texture.copy()
    # A crude blur: average with a shifted copy, enough to reduce local contrast.
    blurry[:, 1:] = 0.5 * blurry[:, 1:] + 0.5 * blurry[:, :-1]

    assert normalized_variance(sharp) > normalized_variance(blurry)


def test_normalized_variance_zero_mean_returns_zero():
    img = np.zeros((10, 10), dtype=np.float64)
    assert normalized_variance(img) == 0.0


def test_evaluator_reads_each_image_at_most_once(tmp_path):
    folder = make_synthetic_stack(tmp_path, n=20, peak=10, bump=3, seed=1)
    dataset = ImageDataset(folder)
    settings = AutofocusSettings()
    evaluator = FocusEvaluator(dataset, settings)

    for _ in range(3):
        evaluator.evaluate(5, phase="ascent")
    evaluator.evaluate(6, phase="ascent")

    assert evaluator.images_read == 2
    assert len(evaluator.trace) == 4  # every call is traced, cached or not


def test_evaluator_values_are_cached_and_consistent(tmp_path):
    folder = make_synthetic_stack(tmp_path, n=20, peak=10, bump=3, seed=1)
    dataset = ImageDataset(folder)
    settings = AutofocusSettings()
    evaluator = FocusEvaluator(dataset, settings)

    first = evaluator.evaluate(7, phase="ascent")
    second = evaluator.evaluate(7, phase="ascent")
    assert first == second


def test_on_step_callback_is_called_for_every_evaluation(tmp_path):
    folder = make_synthetic_stack(tmp_path, n=20, peak=10, bump=3, seed=1)
    dataset = ImageDataset(folder)
    settings = AutofocusSettings()

    seen = []
    evaluator = FocusEvaluator(dataset, settings, on_step=seen.append)
    evaluator.evaluate(0, phase="start")
    evaluator.evaluate(1, phase="ascent")
    evaluator.evaluate(0, phase="ascent")  # cache hit, still traced

    assert [step.index for step in seen] == [0, 1, 0]
    assert [step.step_no for step in seen] == [1, 2, 3]
