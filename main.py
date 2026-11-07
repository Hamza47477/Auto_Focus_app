"""Convenience entry point: ``python main.py`` (same as ``python -m app``).

Everything lives in the :mod:`app` package; this is a one-line delegate so
the app can be launched without the ``-m`` flag.
"""

from __future__ import annotations

import sys

from app.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
