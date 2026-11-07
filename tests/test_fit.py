"""Tests for the quadratic sub-index fit, Eq. 6.10."""

from __future__ import annotations

import pytest

from autofocus import AutofocusSettings, FocusEvaluator, ImageDataset, quadratic_fine_position
from conftest import make_synthetic_stack


class _FakeDataset:
    """Minimal stand-in exposing exactly what quadratic_fine_position needs."""

    def __init__(self, n, d_min_mm=10.0, step_mm=0.01):
        self.n = n
        self.d_min_mm = d_min_mm
        self.step_mm = step_mm

    def __len__(self):
        return self.n

    def position_mm(self, index):
        return self.d_min_mm + float(index) * self.step_mm


class _FakeEvaluator:
    """Evaluator stand-in that serves pre-computed focus values from a dict."""

    def __init__(self, values):
        self.values = values
        self.calls = []

    def evaluate(self, index, phase, note=""):
        self.calls.append(index)
        return self.values[index]


def test_quadratic_fit_recovers_known_sub_index_peak():
    # f(x) = 10 - 0.5*(x - 23.5)**2, true vertex at index 23.5, peak_index = 24 (nearest int)
    true_vertex = 23.5
    peak_index = 24
    n = 40
    values = {i: 10.0 - 0.5 * (i - true_vertex) ** 2 for i in range(n)}

    dataset = _FakeDataset(n)
    evaluator = _FakeEvaluator(values)
    settings = AutofocusSettings(fit_half_window=3)

    outcome = quadratic_fine_position(evaluator, dataset, peak_index, settings)

    assert outcome.fine_index == pytest.approx(true_vertex, abs=1e-6)
    assert outcome.fine_position_mm == pytest.approx(dataset.position_mm(true_vertex), abs=1e-6)
    assert not outcome.warnings


def test_quadratic_fit_upward_parabola_falls_back_with_warning():
    # a2 >= 0 (upward parabola / valley): the vertex is a minimum, not a peak.
    peak_index = 10
    n = 30
    values = {i: 0.5 * (i - peak_index) ** 2 for i in range(n)}

    dataset = _FakeDataset(n)
    evaluator = _FakeEvaluator(values)
    settings = AutofocusSettings(fit_half_window=3)

    outcome = quadratic_fine_position(evaluator, dataset, peak_index, settings)

    assert outcome.fine_index == float(peak_index)
    assert outcome.warnings  # a warning was recorded


def test_quadratic_fit_vertex_outside_window_falls_back_with_warning():
    # Vertex far outside the fitted window: a2 < 0 but x* is way beyond +/- w.
    peak_index = 10
    n = 30
    true_vertex = 25  # outside [10-3, 10+3]
    values = {i: 10.0 - 0.1 * (i - true_vertex) ** 2 for i in range(n)}

    dataset = _FakeDataset(n)
    evaluator = _FakeEvaluator(values)
    settings = AutofocusSettings(fit_half_window=3)

    outcome = quadratic_fine_position(evaluator, dataset, peak_index, settings)

    assert outcome.fine_index == float(peak_index)
    assert outcome.warnings


def test_quadratic_fit_reads_missing_points_through_the_evaluator(tmp_path):
    folder = make_synthetic_stack(tmp_path, n=40, peak=20, bump=5, seed=2)
    dataset = ImageDataset(folder)
    settings = AutofocusSettings(fit_half_window=3)
    evaluator = FocusEvaluator(dataset, settings)

    # Nothing cached yet: the fit must read its own window through the evaluator.
    assert evaluator.images_read == 0
    outcome = quadratic_fine_position(evaluator, dataset, 20, settings)
    assert evaluator.images_read == len(outcome.points)
    assert 17 <= outcome.fine_index <= 23
