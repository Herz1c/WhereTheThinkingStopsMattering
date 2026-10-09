"""Reference baseline data: for each question, the shortest of the model's own correct rollouts.

    python stopping/build_shortest.py runs/rollouts_gsm8k --out runs/sft_shortest.jsonl

This is the simplest way to make a model shorter with its own words (self-training
on best-of-N, as in "Self-Training Elicits Concise Reasoning"), and the bar the
stop-point method has to clear. A rollout qualifies only if it finished on its own
and passes the strict grader, so the model also learns to end on a clean
"Final answer:" line, the format the evaluation reads.

Each output row is the prompt (chat messages) and the rollout's own token ids, so
training sees exactly what the model generated.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path, help="A directory written by generate_rollouts.py")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    by_question = defaultdict(list)
    for line in (arguments.rollouts / "rollouts.jsonl").read_text().splitlines():
        row = json.loads(line)
        by_question[row["question_id"]].append(row)

    chosen, mean_lengths = [], []
    for rows in by_question.values():
        good = [r for r in rows if r["correct"] and r["finished"]]
        if not good:
            continue
        best = min(good, key=lambda r: r["tokens"])
        chosen.append({"question_id": best["question_id"], "source": best["source"],
                       "prompt": best["prompt"], "completion_ids": best["token_ids"],
                       "tokens": best["tokens"]})
        mean_lengths.append(statistics.mean(r["tokens"] for r in rows))

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text("".join(json.dumps(r) + "\n" for r in chosen))
    picked = [r["tokens"] for r in chosen]
    print(f"{len(chosen)} of {len(by_question)} questions have a correct, finished rollout")
    print(f"chosen: mean {statistics.mean(picked):.0f} tokens, median {statistics.median(picked):.0f}; "
          f"all rollouts of those questions: mean {statistics.mean(mean_lengths):.0f}")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
