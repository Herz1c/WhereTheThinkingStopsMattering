"""Round-2 training data: the stop model's own correct answers, and the base model's where it fails.

    python stopping/build_round2.py --domain runs/rollouts_gsm8k:runs/r2_rollouts_math \
        --domain ... --out runs/train_round2.jsonl

Each --domain pairs the base model's rollouts with the stop model's rollouts on the same
training questions (both written by generate_rollouts.py; question ids match). Per question:

- the stop model has correct, finished answers: keep the --keep shortest (on-policy, so what the
  model already does well, at its own length);
- it has none, but the base model has one: keep the base model's shortest correct, finished
  rollout in full ("weak spot": think as long as the base model needed);
- neither: nothing.

Rows are in the trainer's format (prompt + completion token ids).
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def by_question(path):
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        groups[row["question_id"]].append(row)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", action="append", required=True, help="base_rollouts_dir:stop_rollouts_dir")
    parser.add_argument("--keep", type=int, default=2, help="Shortest correct stop-model answers per question")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    rows = []
    for spec in arguments.domain:
        base_dir, stop_dir = spec.split(":")
        base, stop = by_question(Path(base_dir) / "rollouts.jsonl"), by_question(Path(stop_dir) / "rollouts.jsonl")
        kinds, lengths = Counter(), defaultdict(list)
        for q, answers in stop.items():
            good = sorted((r for r in answers if r["correct"] and r["finished"]), key=lambda r: r["tokens"])
            if good:
                picked, kind = good[:arguments.keep], "own correct"
            else:
                base_good = sorted((r for r in base.get(q, []) if r["correct"] and r["finished"]),
                                   key=lambda r: r["tokens"])
                if not base_good:
                    kinds["no correct answer"] += 1
                    continue
                picked, kind = base_good[:1], "weak spot: base rollout"
            for r in picked:
                rows.append({"question_id": q, "source": r["source"], "prompt": r["prompt"],
                             "completion_ids": r["token_ids"], "tokens": r["tokens"], "kind": kind})
                kinds[kind] += 1
                lengths[kind].append(r["tokens"])
        stop_correct = sum(r["correct"] for a in stop.values() for r in a) / sum(len(a) for a in stop.values())
        print(f"{stop_dir}: stop model correct {stop_correct:.1%} of answers; " + ", ".join(
            f"{k} {n}" + (f" (mean {sum(lengths[k]) / len(lengths[k]):.0f} tokens)" if lengths[k] else "")
            for k, n in kinds.most_common()), flush=True)
    arguments.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {arguments.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
