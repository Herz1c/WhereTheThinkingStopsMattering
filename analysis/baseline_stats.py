"""Where do the base model's tokens go? Reads one evaluate.py run and describes it.

    python analysis/baseline_stats.py runs/base
    python analysis/baseline_stats.py runs/base --examples 3    # also print suspected loops

Writes <run>/analysis/summary.json and summary.md, plus two figures when
matplotlib is installed. Nothing here touches the questions or the grading: it
only reads records.jsonl, so it is safe to run on the evaluation run itself.

Two things the metric makes worth knowing, and this script measures:

- Tokens saved on a benchmark is a ratio of *sums*, so long responses dominate it.
  `tail_share` says how much of a benchmark's tokens the longest 10% of responses
  hold, and `truncated_token_share` how much is spent on answers that hit the cap
  and score as wrong anyway.
- Two cheap upper bounds on what training could save, each from the model's own
  samples. `oracle_shortest_correct`: every question answered at the length of its
  shortest correct sample. `oracle_no_truncation`: every truncated sample replaced
  by the median length of that question's finished samples. Both keep accuracy
  fixed by construction; neither is reachable, they size the opportunity.

Looping is flagged by how well the end of a response compresses (zlib): a model
repeating itself produces text that compresses far better than real reasoning.
The threshold is a heuristic -- read a few `--examples` before trusting it.
It only catches degenerate repetition. A response can also hit the cap by thinking
on and on without converging, which compresses like normal text; so truncated
responses are split by where they were cut: still inside <think>, or after it.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import zlib

TAIL_CHARS = 4000        # loops sit at the end of a response, so judge the end
LOOP_RATIO = 0.10        # compressed / raw size below this counts as looping
LOOP_MIN_TOKENS = 1024   # short answers compress poorly anyway; don't judge them
TAIL_FRACTION = 0.10
LENGTH_BINS = 5


def benchmark_of(dataset):
    """BFCL's three subsets are one benchmark, as in utils/eval/compare.py."""
    return "bfcl" if dataset.startswith("bfcl_") else dataset


def compression_ratio(text):
    tail = text[-TAIL_CHARS:].encode()
    return len(zlib.compress(tail, 9)) / len(tail) if tail else 1.0


def is_looping(record):
    return (record["output_tokens"] >= LOOP_MIN_TOKENS
            and compression_ratio(record["text"]) < LOOP_RATIO)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def oracle_savings(records):
    """(shortest-correct, no-truncation) token savings for one benchmark."""
    by_question = defaultdict(list)
    for r in records:
        by_question[r["id"]].append(r)
    total = sum(r["output_tokens"] for r in records)
    shortest = no_truncation = 0
    for samples in by_question.values():
        correct = [r["output_tokens"] for r in samples if r["correct"]]
        if correct:
            shortest += min(correct) * len(samples)
        else:
            shortest += sum(r["output_tokens"] for r in samples)
        finished = [r["output_tokens"] for r in samples if not r["truncated"]]
        fill = statistics.median(finished) if finished else None
        no_truncation += sum(fill if r["truncated"] and fill is not None else r["output_tokens"]
                             for r in samples)
    return 1 - shortest / total, 1 - no_truncation / total


def accuracy_by_length(records):
    """Accuracy in equal-count length bins, shortest first: does longer mean worse?"""
    ordered = sorted(records, key=lambda r: r["output_tokens"])
    size = math.ceil(len(ordered) / LENGTH_BINS)
    bins = []
    for start in range(0, len(ordered), size):
        chunk = ordered[start:start + size]
        bins.append({"max_tokens": chunk[-1]["output_tokens"],
                     "accuracy": sum(r["correct"] for r in chunk) / len(chunk)})
    return bins


def describe(records):
    tokens = [r["output_tokens"] for r in records]
    total = sum(tokens)
    longest = sorted(tokens, reverse=True)[:max(1, int(TAIL_FRACTION * len(tokens)))]
    loops = [r for r in records if is_looping(r)]
    shortest_correct, no_truncation = oracle_savings(records)
    return {
        "responses": len(records),
        "questions": len({r["id"] for r in records}),
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "mean_tokens": total / len(records),
        "median_tokens": statistics.median(tokens),
        "p90_tokens": percentile(tokens, 0.9),
        "max_tokens": max(tokens),
        "reasoning_share": sum(r["reasoning_tokens"] for r in records) / total,
        "truncation_rate": sum(r["truncated"] for r in records) / len(records),
        "truncated_token_share": sum(r["output_tokens"] for r in records if r["truncated"]) / total,
        "truncated_in_thinking": (sum(r["truncated"] and "</think>" not in r["text"] for r in records)
                                  / max(1, sum(r["truncated"] for r in records))),
        "looping_rate": len(loops) / len(records),
        "looping_token_share": sum(r["output_tokens"] for r in loops) / total,
        "tail_share": sum(longest) / total,
        "oracle_shortest_correct": shortest_correct,
        "oracle_no_truncation": no_truncation,
        "accuracy_by_length": accuracy_by_length(records),
    }


def geometric_saving(savings):
    """Average saving over benchmarks the way compare.py averages: 1 - geomean of ratios."""
    return 1 - math.exp(sum(math.log(1 - s) for s in savings) / len(savings))


def load(run):
    records = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]
    ungraded = sum(r.get("correct") is None for r in records)
    if ungraded:
        print(f"Leaving out {ungraded} ungraded responses")
    groups = defaultdict(list)
    for r in records:
        if r.get("correct") is not None:
            groups[benchmark_of(r["dataset"])].append(r)
    return dict(sorted(groups.items()))


def markdown(stats, average):
    rows = [
        ("Accuracy", "accuracy", "{:.1%}"),
        ("Mean tokens", "mean_tokens", "{:.0f}"),
        ("Median tokens", "median_tokens", "{:.0f}"),
        ("90th percentile", "p90_tokens", "{:.0f}"),
        ("Thinking share", "reasoning_share", "{:.1%}"),
        ("Truncation rate", "truncation_rate", "{:.1%}"),
        ("Tokens in truncated", "truncated_token_share", "{:.1%}"),
        ("Truncated still thinking", "truncated_in_thinking", "{:.0%}"),
        ("Looping rate", "looping_rate", "{:.1%}"),
        ("Tokens in loops", "looping_token_share", "{:.1%}"),
        ("Tokens in longest 10%", "tail_share", "{:.1%}"),
        ("Oracle: shortest correct", "oracle_shortest_correct", "{:.1%}"),
        ("Oracle: no truncation", "oracle_no_truncation", "{:.1%}"),
    ]
    names = list(stats)
    lines = ["| | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for label, key, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt.format(stats[n][key]) for n in names) + " |")
    lines += ["", "Average saving over benchmarks (geometric, as compare.py):",
              f"- oracle shortest correct: {average['oracle_shortest_correct']:.1%}",
              f"- oracle no truncation: {average['oracle_no_truncation']:.1%}", "",
              "Accuracy by length bin (equal-count bins, shortest first):", ""]
    for name in names:
        cells = ", ".join(f"<={b['max_tokens']}: {b['accuracy']:.0%}"
                          for b in stats[name]["accuracy_by_length"])
        lines.append(f"- {name}: {cells}")
    return "\n".join(lines) + "\n"


def plot(groups, out):
    """Two figures: length distributions, and how concentrated the tokens are."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    # Categorical slots 1-4 of the validated palette, in fixed order. Every panel
    # and line is also labelled by name, so identity never rests on color alone.
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ink, muted = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})

    names = list(groups)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 2.8), sharey=False)
    for ax, name, color in zip(axes, names, colors):
        tokens = [r["output_tokens"] for r in groups[name]]
        edges = [2 ** (i / 4) for i in range(16, 4 * 14)]   # 16 .. 16k, log-spaced
        ax.hist(tokens, bins=edges, color=color, edgecolor="white", linewidth=0.5)
        ax.set_xscale("log", base=2)
        truncated = sum(r["truncated"] for r in groups[name]) / len(tokens)
        ax.set_title(f"{name}  ·  truncated {truncated:.1%}", color=ink, loc="left")
        ax.set_xlabel("tokens per response")
    axes[0].set_ylabel("responses")
    fig.tight_layout()
    fig.savefig(out / "length_histograms.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, color in zip(names, colors):
        tokens = sorted((r["output_tokens"] for r in groups[name]), reverse=True)
        total, running, xs, ys = sum(tokens), 0, [0.0], [0.0]
        for i, t in enumerate(tokens, 1):
            running += t
            xs.append(i / len(tokens))
            ys.append(running / total)
        at = min(range(len(xs)), key=lambda i: abs(xs[i] - TAIL_FRACTION))
        ax.plot(xs, ys, color=color, linewidth=2,
                label=f"{name}: longest 10% hold {ys[at]:.0%}")
    ax.legend(loc="lower right", frameon=False, labelcolor=ink)
    ax.axvline(TAIL_FRACTION, color=muted, linewidth=0.8, linestyle=":")
    ax.set_xlabel("share of responses, longest first")
    ax.set_ylabel("share of all tokens")
    ax.set_title("How concentrated the tokens are", color=ink, loc="left")
    ax.grid(color="#e6e5e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out / "token_concentration.png", dpi=160)
    plt.close(fig)


def print_loops(groups, count):
    for name, records in groups.items():
        loops = sorted((r for r in records if is_looping(r)), key=lambda r: compression_ratio(r["text"]))
        for r in loops[:count]:
            print(f"\n--- {name} {r['id']} sample {r['sample_index']}: {r['output_tokens']} tokens, "
                  f"ratio {compression_ratio(r['text']):.3f} ---")
            print(r["text"][-800:])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="A directory written by evaluate.py")
    parser.add_argument("--examples", type=int, default=0,
                        help="Print the end of this many suspected loops per benchmark")
    arguments = parser.parse_args()

    groups = load(arguments.run)
    stats = {name: describe(records) for name, records in groups.items()}
    average = {key: geometric_saving([s[key] for s in stats.values()])
               for key in ("oracle_shortest_correct", "oracle_no_truncation")}

    out = arguments.run / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"benchmarks": stats, "average": average}, indent=2))
    report = markdown(stats, average)
    (out / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    plot(groups, out)
    print(f"Wrote {out}")
    if arguments.examples:
        print_loops(groups, arguments.examples)


if __name__ == "__main__":
    main()
