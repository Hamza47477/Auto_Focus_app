"""PySide6 GUI for the autofocus algorithms in the ``autofocus`` package.

This package only builds the interface: dataset browsing, settings forms,
live search visualization, a method comparison and step-by-step replays. It
never computes a focus value itself -- every number shown comes from calling
``autofocus`` (gradient_ascent, golden_section, full_sweep, FocusEvaluator,
quadratic_fine_position) in a background thread.
"""

APP_NAME = "AutoFocus Lab"
APP_VERSION = "0.1"
