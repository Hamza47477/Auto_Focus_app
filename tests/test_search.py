"""Tests for the three search strategies.

Uses the synthetic stack (see conftest.py): global focus peak at index 40,
a smaller local-maximum "fake bump" at index 10, 60 images total.
"""

from __future__ import annotations

import pytest

from autofocus import AutofocusSettings, ImageDataset, full_sweep, golden_section, gradient_ascent
from conftest import make_synthetic_stack

N_IMAGES = 60
PEAK = 40
BUMP = 10


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    folder = make_synthetic_stack(tmp_path_factory.mktemp("search"), n=N_IMAGES, peak=PEAK, bump=BUMP, seed=0)
    return ImageDataset(folder)


def test_full_sweep_finds_true_argmax(dataset):
    settings = AutofocusSettings()
    result = full_sweep(dataset, settings)

    assert result.coarse_index == PEAK
    assert result.images_read == N_IMAGES
    assert result.total_images == N_IMAGES
    assert result.percent_read == pytest.approx(100.0)


def test_golden_section_returns_a_valid_index(dataset):
    settings = AutofocusSettings()
    result = golden_section(dataset, settings)

    assert 0 <= result.coarse_index < N_IMAGES
    assert result.start_index is None
    assert result.images_read < N_IMAGES
    assert abs(result.coarse_index - PEAK) <= 1


def test_protected_gradient_ascent_succeeds_from_most_starts(dataset):
    """With overshoot, neighbor-check and safety-check on, ascent should not
    get trapped by the fake bump and should rarely need to read all images.
    """
    settings = AutofocusSettings(seed=1)  # all protections on by default

    starts = list(range(0, N_IMAGES, 2))  # 30 fixed, deterministic starts
    successes = 0
    for start in starts:
        result = gradient_ascent(dataset, settings, start)
        if abs(result.coarse_index - PEAK) <= 1:
            successes += 1
        assert result.images_read < N_IMAGES  # never reads the whole stack

    success_rate = successes / len(starts)
    assert success_rate >= 0.95


def test_plain_gradient_ascent_can_get_stuck_on_the_fake_bump(dataset):
    """With every protection off, the hill-climber has no way to escape the
    local maximum, so it must fail to reach the global peak for at least
    one start index -- this is what the protections in the default
    settings are there to fix.
    """
    plain_settings = AutofocusSettings(
        use_overshoot_control=False,
        use_neighbor_check=False,
        use_safety_check=False,
    )

    starts = list(range(0, N_IMAGES, 1))
    failures = [
        start
        for start in starts
        if abs(gradient_ascent(dataset, plain_settings, start).coarse_index - PEAK) > 1
    ]

    assert len(failures) > 0


def test_gradient_ascent_from_index_zero(dataset):
    settings = AutofocusSettings(seed=2)
    result = gradient_ascent(dataset, settings, 0)
    assert abs(result.coarse_index - PEAK) <= 1


def test_gradient_ascent_from_last_index(dataset):
    settings = AutofocusSettings(seed=3)
    result = gradient_ascent(dataset, settings, N_IMAGES - 1)
    assert abs(result.coarse_index - PEAK) <= 1


def test_gradient_ascent_starting_at_the_peak(dataset):
    settings = AutofocusSettings(seed=4)
    result = gradient_ascent(dataset, settings, PEAK)
    assert result.coarse_index == PEAK
    assert result.images_read < N_IMAGES


def test_gradient_ascent_start_index_out_of_range_raises(dataset):
    settings = AutofocusSettings()
    with pytest.raises(IndexError):
        gradient_ascent(dataset, settings, N_IMAGES)
    with pytest.raises(IndexError):
        gradient_ascent(dataset, settings, -1)


def test_gradient_ascent_never_rereads_an_image(dataset):
    settings = AutofocusSettings(seed=5)
    result = gradient_ascent(dataset, settings, 5)

    read_counts = {}
    for step in result.trace:
        read_counts[step.index] = read_counts.get(step.index, 0) + 1
    # images_read counts distinct images actually decoded; the trace can
    # reference the same index many times (cache hits), but the evaluator
    # must still have decoded each one only once.
    assert result.images_read == len(read_counts)


def test_results_are_repeatable_with_a_fixed_seed(dataset):
    settings_a = AutofocusSettings(seed=7, use_stochastic_factor=True)
    settings_b = AutofocusSettings(seed=7, use_stochastic_factor=True)

    result_a = gradient_ascent(dataset, settings_a, 3)
    result_b = gradient_ascent(dataset, settings_b, 3)

    assert result_a.coarse_index == result_b.coarse_index
    assert result_a.images_read == result_b.images_read
