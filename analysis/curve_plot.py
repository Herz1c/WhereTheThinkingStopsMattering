"""Length/accuracy plane: every checkpoint against the same baseline, with intervals.

    python analysis/curve_plot.py --base runs/base_s2 --out figures/length_accuracy.png \
        "math m0=runs/stoppoint_s2" "code=runs/stoppoint_code_s2" "all three=runs/stoppoint_all3_s2" ...

Each point is compare.py's own average over the four benchmarks (tokens saved as the geometric
mean of per-benchmark ratios, accuracy change as the mean), with its 95% paired-bootstrap
intervals as error bars. Nothing is recomputed differently from the evaluation's comparison.
Also writes the numbers next to the figure as JSON.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.eval.compare import compare  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", nargs="+",
                        help='"label=run_dir"; a label ending in "*" marks the scaled-adapter family; '
                             '"label@dx,dy=run_dir" places the label dx,dy points from its marker')
    arguments = parser.parse_args()

    points = []
    for item in arguments.runs:
        label, run = item.split("=", 1)
        offset = (7, 6)
        if "@" in label:
            label, where = label.split("@")
            offset = tuple(float(v) for v in where.split(","))
        average = compare(arguments.base, run)["average"]
        intervals = average["intervals"]
        points.append({"label": label, "run": run, "offset": offset,
                       "tokens_saved": average["token_reduction"], "tokens_ci": intervals["token_reduction"],
                       "accuracy": average["accuracy_delta"], "accuracy_ci": intervals["accuracy_delta"]})
        print(f"{label:28s} saved {average['token_reduction']:+.1%} {intervals['token_reduction']}  "
              f"accuracy {average['accuracy_delta'] * 100:+.2f} pp", flush=True)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.with_suffix(".json").write_text(json.dumps(points, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    single, family = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0, color=muted, linewidth=0.8)
    # Scaled versions of one adapter, joined in the order given (e.g. x1.0, x0.75, x0.5). Dashed:
    # the points are the measurements; nothing says the path between them is a straight line.
    curve = [p for p in points if p["label"].endswith("*")]
    if len(curve) > 1:
        ax.plot([p["tokens_saved"] * 100 for p in curve], [p["accuracy"] * 100 for p in curve],
                color=family, linewidth=1.2, linestyle="--", zorder=1)
    for p in points:
        color = family if p["label"].endswith("*") else single
        x, y = p["tokens_saved"] * 100, p["accuracy"] * 100
        ax.errorbar(x, y, xerr=[[x - p["tokens_ci"][0] * 100], [p["tokens_ci"][1] * 100 - x]],
                    yerr=[[y - p["accuracy_ci"][0] * 100], [p["accuracy_ci"][1] * 100 - y]],
                    fmt="o", color=color, ecolor=color, elinewidth=1, capsize=0, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        offset = p["offset"]
        ax.annotate(p["label"].rstrip("*"), (x, y), xytext=offset, textcoords="offset points",
                    ha="right" if offset[0] < 0 else "left",
                    fontsize=8.5, color=ink)
    ax.set_xlabel("tokens saved, geometric mean over the 4 benchmarks (%)")
    ax.set_ylabel("accuracy change, mean over the 4 benchmarks (pp)")
    ax.set_title("Length vs accuracy, every checkpoint (quick evaluation, 2 samples)", color=ink, loc="left",
                 fontsize=10)
    ax.text(1.0, -0.13, "blue: separate runs   orange: the three-domain adapter at reduced strength   bars: 95% intervals",
            transform=ax.transAxes, ha="right", color=muted, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=grid, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(arguments.out, dpi=160)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
