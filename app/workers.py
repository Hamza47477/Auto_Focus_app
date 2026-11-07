"""Background threads that call into ``autofocus``. The GUI thread never does.

Every autofocus search function (``gradient_ascent``, ``golden_section``,
``full_sweep``) shares the same ``on_step`` / ``should_stop`` keyword
contract, so one generic ``SearchWorker`` wraps all three. ``ComparisonWorker``
runs a batch of them for the Comparison page.

Single-step execution (the Focus Search "Step" button) is not something the
``autofocus`` package offers -- its search functions run to completion in one
call. Rather than change the algorithms, ``SearchWorker`` adds a thin,
GUI-only adapter: in single-step mode, its ``on_step`` callback (passed to
the unmodified package function, exactly like the live-update case) blocks
the worker thread on a ``threading.Event`` after every evaluation until the
GUI calls :meth:`SearchWorker.advance`. This changes nothing about what is
computed -- it only pauses the *existing* callback between evaluations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from PySide6.QtCore import QThread, Signal

from autofocus import AutofocusResult, AutofocusSettings, full_sweep, golden_section, gradient_ascent
from autofocus.results import TraceStep

from .state import LoadedDataset


class SearchWorker(QThread):
    """Runs one ``autofocus`` search (gradient ascent / golden section / full sweep)."""

    step = Signal(object)  # TraceStep
    finished_ok = Signal(object)  # AutofocusResult
    failed = Signal(str)

    def __init__(self, fn: Callable, args: tuple, kwargs: dict, single_step: bool = False, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = dict(kwargs)
        self._stop_event = threading.Event()
        self._advance_event = threading.Event()
        self.single_step = single_step
        self._kwargs["on_step"] = self._on_step
        self._kwargs["should_stop"] = self._stop_event.is_set

    def _on_step(self, step: TraceStep) -> None:
        self.step.emit(step)
        if self.single_step and not self._stop_event.is_set():
            self._advance_event.clear()
            self._advance_event.wait()

    def advance(self) -> None:
        """Let a paused single-step worker evaluate exactly one more step."""
        self._advance_event.set()

    def run_to_completion(self) -> None:
        """Switch a paused single-step worker to run freely to the end."""
        self.single_step = False
        self._advance_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._advance_event.set()  # unblock a paused worker so it can observe should_stop()

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished_ok.emit(result)
        except Exception as exc:  # surfaced to the GUI as an error dialog, not a crash
            self.failed.emit(str(exc))


@dataclass
class DatasetComparison:
    """All three methods run on one dataset, for the Comparison page."""

    dataset_name: str
    full_sweep: AutofocusResult
    golden_section: AutofocusResult
    gradient_runs: List[AutofocusResult]  # N runs, independent random start images


class ComparisonWorker(QThread):
    """Runs full sweep + golden section + N random-start gradient ascents, per dataset."""

    progress = Signal(int, int)  # completed, total
    dataset_done = Signal(object)  # DatasetComparison
    finished_ok = Signal(list)  # List[DatasetComparison]
    failed = Signal(str)

    def __init__(
        self,
        datasets: List[LoadedDataset],
        settings: AutofocusSettings,
        n_starts: int,
        start_seed: int,
        parent=None,
    ):
        super().__init__(parent)
        self.datasets = datasets
        self.settings = settings
        self.n_starts = n_starts
        self.start_seed = start_seed
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            total = len(self.datasets) * (2 + self.n_starts)
            done = 0
            results: List[DatasetComparison] = []
            start_rng = np.random.default_rng(self.start_seed)

            for loaded in self.datasets:
                if self._stop_event.is_set():
                    break
                ds = loaded.dataset

                sweep = full_sweep(ds, self.settings, should_stop=self._stop_event.is_set)
                done += 1
                self.progress.emit(done, total)

                golden = golden_section(ds, self.settings, should_stop=self._stop_event.is_set)
                done += 1
                self.progress.emit(done, total)

                gradient_runs: List[AutofocusResult] = []
                starts = start_rng.integers(0, len(ds), size=self.n_starts)
                for start in starts:
                    if self._stop_event.is_set():
                        break
                    r = gradient_ascent(ds, self.settings, int(start), should_stop=self._stop_event.is_set)
                    gradient_runs.append(r)
                    done += 1
                    self.progress.emit(done, total)

                comparison = DatasetComparison(loaded.name, sweep, golden, gradient_runs)
                results.append(comparison)
                self.dataset_done.emit(comparison)

            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))
