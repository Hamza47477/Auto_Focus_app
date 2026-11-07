"""Tiny CLI for manual testing of the autofocus algorithms. No GUI.

Usage::

    python -m autofocus <folder> --start 25 --method gradient|golden|sweep --seed 0
"""

from __future__ import annotations

import argparse
import sys

from .dataset import ImageDataset
from .results import AutofocusResult
from .search import full_sweep, golden_section, gradient_ascent
from .settings import AutofocusSettings


def _parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m autofocus",
        description="Run one autofocus search on a folder of images and print the result.",
    )
    parser.add_argument("folder", help="Folder containing one image set.")
    parser.add_argument("--start", type=int, default=0, help="Start index for the gradient-ascent search.")
    parser.add_argument(
        "--method",
        choices=["gradient", "golden", "sweep"],
        default="gradient",
        help="Search strategy to run.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for the stochastic step factor.")
    parser.add_argument("--d-min-mm", type=float, default=10.0, help="Working distance of index 0, in mm.")
    parser.add_argument("--step-mm", type=float, default=0.010, help="Working-distance increment per index, in mm.")
    return parser.parse_args(argv)


def _print_result(result: AutofocusResult) -> None:
    print(f"method            : {result.method}")
    print(f"start_index       : {result.start_index}")
    print(f"coarse_index      : {result.coarse_index}")
    print(f"coarse_focus      : {result.coarse_focus:.6f}")
    print(f"coarse_position_mm: {result.coarse_position_mm:.4f}")
    print(f"fine_index        : {result.fine_index:.3f}")
    print(f"fine_position_mm  : {result.fine_position_mm:.4f}")
    print(f"images_read       : {result.images_read} / {result.total_images} ({result.percent_read:.1f}%)")
    print(f"restarts          : {result.restarts}")
    print(f"elapsed_s         : {result.elapsed_s:.4f}")
    print(f"seed              : {result.seed}")
    if result.warnings:
        print("warnings:")
        for w in result.warnings:
            print(f"  - {w}")


def main(argv: list = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    settings = AutofocusSettings(seed=args.seed, d_min_mm=args.d_min_mm, step_mm=args.step_mm)
    dataset = ImageDataset(args.folder, d_min_mm=args.d_min_mm, step_mm=args.step_mm)

    if args.method == "gradient":
        result = gradient_ascent(dataset, settings, args.start)
    elif args.method == "golden":
        result = golden_section(dataset, settings)
    else:
        result = full_sweep(dataset, settings)

    _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
