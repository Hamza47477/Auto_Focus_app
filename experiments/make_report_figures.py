"""Generate the lab-report figures and CSV tables for all speed datasets.

Usage::

    python experiments/make_report_figures.py <root_folder_with_speed_folders> --out results/

``root_folder`` must contain one sub-folder per scanning speed (e.g.
``pollen.spd1`` .. ``pollen.spd5``); each speed folder may hold its images
directly or one level deeper in a same-named sub-folder (both layouts are
auto-detected). This script only calls into the ``autofocus`` package; it
never reads pixels except through it. Matplotlib uses the non-interactive
``Agg`` backend throughout -- no windows are opened.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from autofocus import AutofocusSettings, ImageDataset, full_sweep, golden_section, gradient_ascent
from autofocus.dataset import list_image_files

RNG_SEED = 0
STAGE4_N_STARTS = 50


# --------------------------------------------------------------------------- #
# dataset discovery
# --------------------------------------------------------------------------- #


def discover_speed_folders(root: Path) -> list:
    """Return (name, image_folder) for every speed sub-folder under root.

    Each speed folder's images may live directly inside it, or one level
    deeper in a sub-folder sharing its name (as shipped for this lab).
    """
    speeds = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if list_image_files(child):
            speeds.append((child.name, child))
            continue
        for grandchild in sorted(p for p in child.iterdir() if p.is_dir()):
            if list_image_files(grandchild):
                speeds.append((child.name, grandchild))
                break
    return speeds


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #


def curve_from_trace(result) -> tuple:
    """Return (indices, focus) for every DISTINCT index seen in a trace, sorted."""
    by_index = {}
    for step in result.trace:
        by_index[step.index] = step.focus
    indices = sorted(by_index)
    return np.array(indices), np.array([by_index[i] for i in indices])


def parabola(coeffs, xs_centered):
    a0, a1, a2 = coeffs
    return a0 + a1 * xs_centered + a2 * xs_centered ** 2


def save(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# stage 1: NV curves, peaks, zoom, best image, CSV
# --------------------------------------------------------------------------- #


def stage1(datasets, out_dir: Path) -> dict:
    sweeps = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, dataset) in zip(axes.flat, datasets):
        settings = AutofocusSettings()
        result = full_sweep(dataset, settings)
        sweeps[name] = result

        positions = np.array([dataset.position_mm(i) for i in range(len(dataset))])
        indices, values = curve_from_trace(result)
        curve_positions = positions[indices]

        ax.plot(curve_positions, values, color="tab:blue", lw=1)
        ax.axvline(result.coarse_position_mm, color="tab:red", ls="--", lw=1, label="coarse peak")
        ax.set_title(name)
        ax.set_xlabel("working distance (mm)")
        ax.set_ylabel("normalized variance (Eq. 6.5)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "stage1_nv_vs_position_grid.png")

    # all 4 curves normalized to [0, 1] on one plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, dataset in datasets:
        result = sweeps[name]
        positions = np.array([dataset.position_mm(i) for i in range(len(dataset))])
        indices, values = curve_from_trace(result)
        values = (values - values.min()) / (values.max() - values.min())
        ax.plot(positions[indices], values, label=name)
    ax.set_xlabel("working distance (mm)")
    ax.set_ylabel("normalized variance (0-1 per curve)")
    ax.set_title("All speeds, focus curves normalized to [0, 1]")
    ax.legend()
    save(fig, out_dir, "stage1_all_speeds_normalized.png")

    # zoom on the peak with fit points, parabola and the fine position
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, dataset) in zip(axes.flat, datasets):
        result = sweeps[name]
        fit_indices = np.array([i for i, _ in result.fit_points])
        fit_values = np.array([f for _, f in result.fit_points])
        xs_centered = np.linspace(fit_indices.min() - result.coarse_index, fit_indices.max() - result.coarse_index, 200)
        curve = parabola(result.fit_coefficients, xs_centered)

        ax.scatter(fit_indices, fit_values, color="tab:blue", zorder=3, label="fit points")
        ax.plot(xs_centered + result.coarse_index, curve, color="tab:orange", label="parabola fit (Eq. 6.10)")
        ax.axvline(result.fine_index, color="tab:red", ls="--", label=f"fine index {result.fine_index:.2f}")
        ax.set_title(name)
        ax.set_xlabel("image index")
        ax.set_ylabel("normalized variance")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "stage1_peak_zoom_fit.png")

    # coarse sharpest image per speed
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    for ax, (name, dataset) in zip(axes, datasets):
        result = sweeps[name]
        img = cv2.imread(str(dataset.path_for(result.coarse_index)), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{name}\nindex {result.coarse_index}")
        ax.axis("off")
    fig.tight_layout()
    save(fig, out_dir, "stage1_coarse_sharpest_images.png")

    # CSV table
    csv_path = out_dir / "stage1_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["speed", "coarse_index", "coarse_position_mm", "coarse_focus", "fine_position_mm", "images_read"])
        for name, dataset in datasets:
            r = sweeps[name]
            writer.writerow([name, r.coarse_index, f"{r.coarse_position_mm:.4f}", f"{r.coarse_focus:.6f}", f"{r.fine_position_mm:.4f}", r.images_read])

    return sweeps


# --------------------------------------------------------------------------- #
# stage 2: probes overlay + images-read bar chart
# --------------------------------------------------------------------------- #


def stage2(datasets, sweeps, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    read_counts = {"full_sweep": [], "gradient_ascent": [], "golden_section": []}
    names = [name for name, _ in datasets]

    for ax, (name, dataset) in zip(axes.flat, datasets):
        sweep = sweeps[name]
        settings = AutofocusSettings(seed=RNG_SEED)
        start = len(dataset) // 5  # an off-peak start, to make the climb visible
        ga = gradient_ascent(dataset, settings, start)

        positions = np.array([dataset.position_mm(i) for i in range(len(dataset))])
        indices, values = curve_from_trace(sweep)
        ax.plot(positions[indices], values, color="0.7", lw=1, zorder=1, label="full curve")

        ga_indices, ga_values = curve_from_trace(ga)
        ax.scatter(positions[ga_indices], ga_values, color="tab:red", s=18, zorder=3, label="ascent probes")
        order = {idx: n for n, idx in enumerate(step.index for step in ga.trace if step.phase in ("start", "ascent", "neighbor_check", "safety_sample", "restart"))}
        for idx in ga_indices:
            n = order.get(int(idx))
            if n is not None:
                ax.annotate(str(n), (dataset.position_mm(idx), sweep_focus_lookup(sweep, idx)), fontsize=6, xytext=(2, 2), textcoords="offset points")

        ax.set_title(name)
        ax.set_xlabel("working distance (mm)")
        ax.set_ylabel("normalized variance")
        ax.legend(fontsize=7)

        read_counts["full_sweep"].append(sweep.images_read)
        read_counts["gradient_ascent"].append(ga.images_read)
        read_counts["golden_section"].append(golden_section(dataset, settings).images_read)

    fig.tight_layout()
    save(fig, out_dir, "stage2_ascent_probes.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.25
    for offset, (method, color) in enumerate([("full_sweep", "0.6"), ("gradient_ascent", "tab:red"), ("golden_section", "tab:green")]):
        ax.bar(x + (offset - 1) * width, read_counts[method], width, label=method, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("images read")
    ax.set_title("Images read: full sweep vs gradient ascent vs golden section")
    ax.legend()
    save(fig, out_dir, "stage2_images_read_bar.png")


def sweep_focus_lookup(sweep_result, index):
    for step in sweep_result.trace:
        if step.index == index:
            return step.focus
    return float("nan")


# --------------------------------------------------------------------------- #
# stage 3: preprocessing on/off
# --------------------------------------------------------------------------- #


def stage3(datasets, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, dataset) in zip(axes.flat, datasets):
        plain = full_sweep(dataset, AutofocusSettings())
        processed = full_sweep(
            dataset,
            AutofocusSettings(do_crop_bottom=True, crop_bottom_px=40, do_denoise=True, denoise_method="gaussian", denoise_ksize=5),
        )

        positions = np.array([dataset.position_mm(i) for i in range(len(dataset))])
        i1, v1 = curve_from_trace(plain)
        i2, v2 = curve_from_trace(processed)
        ax.plot(positions[i1], v1, label="raw", color="tab:blue")
        ax.plot(positions[i2], v2, label="crop + denoise", color="tab:orange")
        ax.set_title(name)
        ax.set_xlabel("working distance (mm)")
        ax.set_ylabel("normalized variance")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "stage3_preprocessing_comparison.png")


# --------------------------------------------------------------------------- #
# stage 4: random-start robustness study
# --------------------------------------------------------------------------- #

STAGE4_VERSIONS = [
    ("plain", dict(use_overshoot_control=False, use_neighbor_check=False, use_safety_check=False)),
    ("+overshoot", dict(use_overshoot_control=True, use_neighbor_check=False, use_safety_check=False)),
    ("+stochastic", dict(use_overshoot_control=True, use_neighbor_check=False, use_safety_check=False, use_stochastic_factor=True)),
    ("+neighbor+safety", dict(use_overshoot_control=True, use_neighbor_check=True, use_safety_check=True, use_stochastic_factor=True)),
]


def stage4(datasets, sweeps, out_dir: Path, n_starts: int = STAGE4_N_STARTS) -> None:
    rng = np.random.default_rng(RNG_SEED)
    names = [name for name, _ in datasets]
    success_rates = {v: [] for v, _ in STAGE4_VERSIONS}
    mean_reads = {v: [] for v, _ in STAGE4_VERSIONS}
    csv_rows = []

    example_paths = {}  # name -> (failed_result_or_None, success_result_or_None)

    for name, dataset in datasets:
        n = len(dataset)
        true_peak = sweeps[name].coarse_index
        starts = rng.integers(0, n, size=n_starts)
        example_paths[name] = [None, None]

        for version_name, overrides in STAGE4_VERSIONS:
            successes = 0
            reads = []
            for seed_offset, start in enumerate(starts):
                settings = AutofocusSettings(seed=RNG_SEED + seed_offset, **overrides)
                result = gradient_ascent(dataset, settings, int(start))
                reads.append(result.images_read)
                ok = abs(result.coarse_index - true_peak) <= 1
                if ok:
                    successes += 1
                    if version_name == "+neighbor+safety" and example_paths[name][1] is None:
                        example_paths[name][1] = result
                else:
                    if version_name == "plain" and example_paths[name][0] is None:
                        example_paths[name][0] = result

            rate = successes / n_starts
            success_rates[version_name].append(rate)
            mean_reads[version_name].append(float(np.mean(reads)))
            csv_rows.append([name, version_name, f"{rate:.3f}", f"{np.mean(reads):.2f}"])

    x = np.arange(len(names))
    width = 0.8 / len(STAGE4_VERSIONS)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (version_name, _) in enumerate(STAGE4_VERSIONS):
        ax.bar(x + (i - 1.5) * width, success_rates[version_name], width, label=version_name)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(f"success rate (peak +/-1, {n_starts} random starts)")
    ax.set_title("Gradient-ascent success rate by protection level")
    ax.legend(fontsize=8)
    save(fig, out_dir, "stage4_success_rate.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (version_name, _) in enumerate(STAGE4_VERSIONS):
        ax.bar(x + (i - 1.5) * width, mean_reads[version_name], width, label=version_name)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean images read")
    ax.set_title("Gradient-ascent images read by protection level")
    ax.legend(fontsize=8)
    save(fig, out_dir, "stage4_images_read.png")

    csv_path = out_dir / "stage4_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["speed", "version", "success_rate", "mean_images_read"])
        writer.writerows(csv_rows)

    # one failed path and one successful path on the full curve, per speed
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, dataset) in zip(axes.flat, datasets):
        positions = np.array([dataset.position_mm(i) for i in range(len(dataset))])
        indices, values = curve_from_trace(sweeps[name])
        ax.plot(positions[indices], values, color="0.75", lw=1, label="full curve")

        failed, succeeded = example_paths[name]
        for result, color, label in [(failed, "tab:red", "plain: failed path"), (succeeded, "tab:green", "protected: successful path")]:
            if result is None:
                continue
            r_idx, r_val = curve_from_trace(result)
            ax.plot(positions[r_idx], r_val, "o-", color=color, ms=3, lw=1, label=label)

        ax.set_title(name)
        ax.set_xlabel("working distance (mm)")
        ax.set_ylabel("normalized variance")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "stage4_example_paths.png")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Folder containing the speed sub-folders.")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Output folder for figures and CSVs.")
    parser.add_argument("--stage4-starts", type=int, default=STAGE4_N_STARTS, help="Random starts per speed/version in stage 4.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    args.out.mkdir(parents=True, exist_ok=True)
    speed_folders = discover_speed_folders(args.root)
    if not speed_folders:
        raise SystemExit(f"No speed sub-folders with images found under {args.root}")

    datasets = [(name, ImageDataset(folder)) for name, folder in speed_folders]

    print(f"Found {len(datasets)} speed datasets: {[n for n, _ in datasets]}")
    print("Stage 1: NV curves, peaks, zoom, best images, CSV ...")
    sweeps = stage1(datasets, args.out)
    print("Stage 2: ascent probes overlay, images-read bar chart ...")
    stage2(datasets, sweeps, args.out)
    print("Stage 3: preprocessing on/off comparison ...")
    stage3(datasets, args.out)
    print(f"Stage 4: {args.stage4_starts} random starts x {len(STAGE4_VERSIONS)} versions robustness study ...")
    stage4(datasets, sweeps, args.out, n_starts=args.stage4_starts)
    print(f"Done. Figures and CSVs written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
