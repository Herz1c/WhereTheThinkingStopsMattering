"""Per-benchmark mean tokens and accuracy, base against one checkpoint, from full evaluation records.

    python analysis/benchmark_bars.py --base runs/base --mine runs/grpo_full --label "GRPO" \
        --out figures/benchmark_bars.png

Mean generated tokens per response (thinking included, capped responses at their cap) and accuracy
per benchmark, BFCL's three subsets pooled as in compare.py. Descriptive only; the intervals are
in compare.py's output.
"""

import argparse
import json
from pathlib import Path

ORDER = [("mbpp", "MBPP+ (code)"), ("bfcl", "BFCL (function calling)"), ("ifeval", "IFEval (instructions)"),
         ("svamp", "SVAMP (math)")]


def summary(run):
    out = {}
    for line in (Path(run) / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        b = "bfcl" if r["dataset"].startswith("bfcl") else r["dataset"]
        s = out.setdefault(b, [0, 0, 0])
        s[0] += r["output_tokens"]; s[1] += bool(r.get("correct")); s[2] += 1
    return {b: (t / n, c / n) for b, (t, c, n) in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--mine", required=True)
    parser.add_argument("--label", default="this work")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    base, mine = summary(arguments.base), summary(arguments.mine)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, grey, green = "#0b0b0b", "#52514e", "#e6e5e0", "#9aa0a6", "#1a7f37"
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ys = range(len(ORDER))
    for y, (key, name) in zip(ys, ORDER):
        bt, ba = base[key]
        mt, ma = mine[key]
        ax.barh(y + 0.18, bt, height=0.34, color=grey)
        ax.barh(y - 0.18, mt, height=0.34, color=green)
        ax.text(bt + 25, y + 0.18, f"{bt:,.0f} tokens · {ba:.1%} correct", va="center", fontsize=8, color=muted)
        ax.text(mt + 25, y - 0.18, f"{mt:,.0f} tokens · {ma:.1%} correct  ({1 - mt / bt:.0%} fewer)",
                va="center", fontsize=8, color=green)
    ax.set_yticks(list(ys), [n for _, n in ORDER], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean generated tokens per response (thinking included)", fontsize=9)
    ax.set_xlim(0, max(v[0] for v in base.values()) * 1.75)
    ax.set_title(f"Qwen3-0.6B base (grey) vs. {arguments.label} (green), full evaluation", color=ink,
                 loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(arguments.out, dpi=160)
    print(f"wrote {arguments.out}")
    for key, name in ORDER:
        print(f"{name:26s} base {base[key][0]:7.0f} tok {base[key][1]:.1%}   mine {mine[key][0]:7.0f} tok {mine[key][1]:.1%}")


if __name__ == "__main__":
    main()
