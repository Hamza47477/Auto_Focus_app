"""Single source of truth for the phase colors used across widgets/pages.

Both the Qt-painted evaluation strip (widgets/image_view.py) and the
matplotlib focus-curve plot (widgets/plot_widget.py) color points by
``TraceStep.phase``; keeping one dict means a color only has to be chosen
once.
"""

PHASE_COLORS = {
    "not_read": "#22262c",
    "start": "#ffa94d",
    "ascent": "#5aa9ff",
    "neighbor_check": "#7fc1ff",
    "safety_sample": "#a78bfa",
    "restart": "#ff6b6b",
    "fit": "#51cf66",
    "golden_probe": "#a78bfa",
    "sweep": "#5aa9ff",
    "skipped": "#22262c",
    "best": "#ffa94d",
}

PHASE_LABELS = {
    "start": "Start",
    "ascent": "Ascent",
    "neighbor_check": "Neighbor check",
    "safety_sample": "Safety sample",
    "restart": "Restart",
    "fit": "Fit",
    "golden_probe": "Probe",
    "sweep": "Sweep",
}
