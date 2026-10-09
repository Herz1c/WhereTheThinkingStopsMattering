"""Side-by-side summary of rollout runs on the same validation questions.

    python analysis/compare_rollouts.py runs/val_base runs/val_m0 runs/val_m2 runs/val_m4

Each directory is written by stopping/generate_rollouts.py (with --adapter for a
checkpoint) on held-out questions. This is where the stop margin is chosen, never on
the evaluation benchmarks. Intervals are 95% from resampling whole questions.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

RESAMPLES = 2000


def load(directory):
    by_question = defaultdict(list)
    for line in (directory / "rollouts.jsonl").read_text().splitlines():
        row = json.loads(line)
        by_question[row["question_id"]].append(row)
    return by_question


def interval(questions, measure, rng):
    """Mean of `measure` over rollouts, with a question-level bootstrap interval."""
    keys = list(questions)
    def mean(sample):
        rows = [r for k in sample for r in questions[k]]
        return sum(measure(r) for r in rows) / len(rows)
    draws = sorted(mean([rng.choice(keys) for _ in keys]) for _ in range(RESAMPLES))
    return mean(keys), draws[int(0.025 * RESAMPLES)], draws[int(0.975 * RESAMPLES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    arguments = parser.parse_args()

    runs = {d.name: load(d) for d in arguments.runs}
    shared = set.intersection(*(set(q) for q in runs.values()))
    rng = random.Random(0)
    print(f"{len(shared)} shared questions\n")
    print(f"{'run':16s} {'strict accuracy':>24s} {'knows':>24s} {'mean tokens':>22s} {'truncated':>10s}")
    for name, questions in runs.items():
        questions = {k: v for k, v in questions.items() if k in shared}
        strict = interval(questions, lambda r: r["correct"], rng)
        known = interval(questions, lambda r: r["knows"], rng)
        tokens = interval(questions, lambda r: r["tokens"], rng)
        truncated = sum(not r["finished"] for v in questions.values() for r in v) / sum(map(len, questions.values()))
        print(f"{name:16s} {strict[0]:6.1%} [{strict[1]:5.1%}, {strict[2]:5.1%}] "
              f"{known[0]:6.1%} [{known[1]:5.1%}, {known[2]:5.1%}] "
              f"{tokens[0]:6.0f} [{tokens[1]:5.0f}, {tokens[2]:5.0f}] {truncated:9.1%}")


if __name__ == "__main__":
    main()
