"""Shared application state: loaded datasets, current settings, run history.

One ``AppState`` instance is created in ``main_window.py`` and passed to
every page. Pages never talk to each other directly -- they read/write
``AppState`` and react to its Qt signals, which keeps the pages independent
of each other (e.g. Focus Search does not need to know Comparison exists).

No pixel is read and no focus value is computed here: this module only
wraps ``autofocus.ImageDataset`` (listing only) and stores results that the
search workers (``workers.py``) produced by calling into ``autofocus``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from autofocus import AutofocusResult, AutofocusSettings, ImageDataset
from autofocus.dataset import list_image_files


@dataclass
class LoadedDataset:
    """One opened image folder.

    ``sweep_result`` stays ``None`` until the user explicitly runs a full
    sweep on this dataset (Datasets page or Comparison page) -- loading a
    dataset must never read pixels.
    """

    name: str
    path: Path
    dataset: ImageDataset
    sweep_result: Optional[AutofocusResult] = None


@dataclass
class RunRecord:
    """One finished search, kept for the Walkthroughs replay and history."""

    result: AutofocusResult
    dataset_name: str
    label: str
    run_id: int


def resolve_image_folder(path: Path) -> Optional[Path]:
    """Return ``path`` if it directly holds images, else a same-named nested folder that does.

    Handles this lab's two dataset layouts: ``pollen.spd1/*.png`` directly,
    or ``pollen.spd2/pollen.spd2/*.png`` nested one level. Returns ``None``
    if neither layout yields any images (only file names are inspected).
    """
    if list_image_files(path):
        return path
    if path.is_dir():
        for child in sorted(p for p in path.iterdir() if p.is_dir()):
            if list_image_files(child):
                return child
    return None


def discover_dataset_folders(root: Path) -> List[Path]:
    """Return one image folder per speed sub-folder of ``root`` (section 2.1 cards).

    If ``root`` itself directly contains images, it is treated as a single
    dataset and returned as a one-item list.
    """
    direct = resolve_image_folder(root)
    if direct is not None and list_image_files(root):
        return [root]
    found = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        resolved = resolve_image_folder(child)
        if resolved is not None:
            found.append(resolved)
    return found


def display_name_for(folder: Path) -> str:
    """"pollen.spd5" -> "Speed 5"; anything else falls back to the folder name."""
    name = folder.name
    if name.lower().startswith("pollen.spd"):
        suffix = name[len("pollen.spd"):]
        if suffix.isdigit():
            return f"Speed {suffix}"
    return name


class AppState(QObject):
    """Central, observable state shared by every page."""

    datasets_changed = Signal()
    active_dataset_changed = Signal(str)
    settings_changed = Signal()
    run_recorded = Signal(object)  # RunRecord
    sweep_recorded = Signal(str)  # dataset name
    status_changed = Signal(str)  # free text for the status bar
    worker_state_changed = Signal(str)  # "idle" | "running" | "done" | "stopped" | "error"

    def __init__(self) -> None:
        super().__init__()
        self.settings = AutofocusSettings()
        self.datasets: Dict[str, LoadedDataset] = {}
        self.active_dataset_name: Optional[str] = None
        self.run_history: List[RunRecord] = []
        self._next_run_id = 1

    # -- datasets ---------------------------------------------------------

    def add_dataset(self, folder: Path, display_name: Optional[str] = None) -> LoadedDataset:
        """Open ``folder`` as a dataset (list files only) and register it.

        Raises:
            FileNotFoundError, ValueError: propagated from ``ImageDataset``.
        """
        dataset = ImageDataset(folder, d_min_mm=self.settings.d_min_mm, step_mm=self.settings.step_mm)
        name = display_name or display_name_for(folder)
        loaded = LoadedDataset(name=name, path=folder, dataset=dataset)
        self.datasets[name] = loaded
        if self.active_dataset_name is None:
            self.active_dataset_name = name
        self.datasets_changed.emit()
        if self.active_dataset_name == name:
            self.active_dataset_changed.emit(name)
        return loaded

    def active_dataset(self) -> Optional[LoadedDataset]:
        if self.active_dataset_name is None:
            return None
        return self.datasets.get(self.active_dataset_name)

    def set_active_dataset(self, name: str) -> None:
        if name in self.datasets and name != self.active_dataset_name:
            self.active_dataset_name = name
            self.active_dataset_changed.emit(name)

    # -- settings -----------------------------------------------------------

    def update_settings(self, settings: AutofocusSettings) -> None:
        self.settings = settings
        self.settings_changed.emit()

    # -- results ------------------------------------------------------------

    def record_run(self, result: AutofocusResult, dataset_name: str, label: str) -> RunRecord:
        record = RunRecord(result=result, dataset_name=dataset_name, label=label, run_id=self._next_run_id)
        self._next_run_id += 1
        self.run_history.append(record)
        self.run_recorded.emit(record)
        return record

    def set_sweep_result(self, dataset_name: str, result: AutofocusResult) -> None:
        loaded = self.datasets.get(dataset_name)
        if loaded is not None:
            loaded.sweep_result = result
        self.record_run(result, dataset_name, f"Full sweep — {dataset_name}")
        self.sweep_recorded.emit(dataset_name)

    def runs_for_method(self, method: str) -> List[RunRecord]:
        return [r for r in self.run_history if r.result.method == method]
