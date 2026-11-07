"""
Step 2: every focus measure x every search strategy, on the cached scores.
"Reads" = distinct frames a strategy had to look at (each one = one image acquisition).

Usage: python3 run_benchmark.py  (after cache_scores.py)
Writes results/benchmark.csv and results/robustness.csv
"""
import sys, os, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from algorithm.global_scan_search import global_scan_search
from algorithm.coarse_and_fine_search import coarse_and_fine_search   # fixed version
from algorithm.gradient_descent_search import gradient_descent_search
from algorithm.coarse_quadratic_search import coarse_quadratic

SPEEDS = [1, 2, 3, 5]
MEASURES = ["NV", "VoL", "LoG", "SML"]
STARTS = list(range(10, 199, 20))   # 10, 30, ..., 190: robustness to the start frame


def evaluator(S, m, coarse_m="NV"):
    """eval_func(idx, stride, fmo): coarse stage of the hybrid uses NV on stride-4 images."""
    def ev(idx, stride=1, fmo=None):
        name = coarse_m if fmo == "coarse" else m
        return float(S[f"{name}_s4" if stride > 1 else f"{name}_s1"][idx])
    return ev


def main():
    rows, rob = [], []
    for spd in SPEEDS:
        S = dict(np.load(os.path.join(HERE, "results", f"scores_spd{spd}.npz")))
        n = len(S["NV_s1"]); mx = n - 1
        ref = int(np.argmax(S["VoL_s1"]))          # report's reference peak
        for m in MEASURES:
            ev = evaluator(S, m)
            own = int(np.argmax(S[f"{m}_s1"]))      # this measure's own global optimum
            cost = float(S[f"cost_ms_{m}"])

            def add(name, idx, reads, extra=None):
                r = dict(speed=spd, measure=m, strategy=name, peak=idx,
                         err_ref=abs(idx - ref), err_own=abs(idx - own), reads=reads,
                         saving=round(100 * (1 - reads / n), 1),
                         compute_ms=round(reads * cost, 1), ref=ref, own=own,
                         ms_per_frame=round(cost, 2))
                r.update(extra or {})
                rows.append(r)

            g = global_scan_search(ev, mx)
            add("global", g.best_idx, len(g.evals))

            h = coarse_and_fine_search(ev, mx, coarse_fmo="coarse", fine_fmo=m)
            add("hybrid_grid", h.best_idx, len(h.evals))

            gr = gradient_descent_search(ev, mx)
            add("gradient", gr.best_idx, len(gr.evals),
                dict(restarts=gr.restarts, verified=gr.verified))

            cq = coarse_quadratic(ev, mx, coarse_method="gradient")
            add("gradient_fit", cq.best_idx, len(cq.evals),
                dict(fine=round(cq.fine_idx, 3), r2=round(cq.fit_r2, 3), fit_ok=cq.fit_ok))

            cg = coarse_quadratic(ev, mx, coarse_method="grid")
            add("grid_fit", cg.best_idx, len(cg.evals),
                dict(fine=round(cg.fine_idx, 3), r2=round(cg.fit_r2, 3), fit_ok=cg.fit_ok))

            # robustness of gradient ascent to the start frame
            for st in STARTS:
                r = gradient_descent_search(ev, mx, start_idx=st)
                rob.append(dict(speed=spd, measure=m, start=st, peak=r.best_idx,
                                err_ref=abs(r.best_idx - ref), err_own=abs(r.best_idx - own),
                                reads=len(r.evals), restarts=r.restarts, verified=r.verified))

    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(os.path.join(HERE, "results", "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(HERE, "results", "robustness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rob[0])); w.writeheader(); w.writerows(rob)
    print("wrote", len(rows), "benchmark rows and", len(rob), "robustness rows")


if __name__ == "__main__":
    main()
