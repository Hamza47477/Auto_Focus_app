"""
Step 1: score every frame of every image set with every focus measure, once.
Saves results/scores_spd<N>.npz (full-res and stride-4 scores) + per-measure cost.

Usage: python3 cache_scores.py <Datasets folder> [speeds...]
"""
import sys, os, glob, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algorithm.normalized_variance import normalized_variance
from algorithm.laplacian import variance_of_laplacian, laplacian_of_gaussian, modified_laplacian

MEASURES = {
    "NV": normalized_variance,
    "VoL": variance_of_laplacian,
    "LoG": laplacian_of_gaussian,
    "SML": modified_laplacian,
}


def frames_of(ds_root, spd):
    pat = os.path.join(ds_root, f"pollen.spd{spd}", "**", f"pollen.spd{spd}.*.png")
    return sorted(glob.glob(pat, recursive=True))


def main():
    ds_root = sys.argv[1]
    speeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3, 5]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for spd in speeds:
        files = frames_of(ds_root, spd)
        n = len(files)
        out = {}
        cost = {m: 0.0 for m in MEASURES}
        for m in MEASURES:
            out[f"{m}_s1"] = np.zeros(n)
            out[f"{m}_s4"] = np.zeros(n)
        for i, f in enumerate(files):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            small = img[::4, ::4]
            for m, fn in MEASURES.items():
                t = time.perf_counter()
                out[f"{m}_s1"][i] = fn(img)
                cost[m] += time.perf_counter() - t
                out[f"{m}_s4"][i] = fn(small)
        for m in MEASURES:
            out[f"cost_ms_{m}"] = np.array(1000.0 * cost[m] / n)
        np.savez(os.path.join(HERE, "results", f"scores_spd{spd}.npz"), **out)
        print(f"speed {spd}: {n} frames scored; ms/frame " +
              ", ".join(f"{m}={1000*cost[m]/n:.2f}" for m in MEASURES))


if __name__ == "__main__":
    main()
