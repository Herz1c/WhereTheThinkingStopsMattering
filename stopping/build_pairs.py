"""Preference pairs for stop decisions: stop where the answer is settled, not at the first guess.

    python stopping/build_pairs.py --domain runs/rollouts_gsm8k:runs/stop_points_regress:runs/continue_math \
        --domain ... --out runs/dpo_pairs.jsonl

Supervised stop-point data only shows the model good stops. The stop-trained models lose
accuracy by stopping at the first answer they reach when it is wrong, and no example ever shows
that this is bad. These pairs do, from answers the model already wrote during the stop-point
search (find_stop_points.py) and the keep-thinking walk (continue_examples.py):

- *chosen*: a training example from those runs: the rollout's own reasoning up to a verified stop,
  `</think>`, and a correct answer sampled there;
- *rejected*: the same rollout's reasoning cut at the **latest earlier boundary** where a forced
  `</think>` produced a **wrong** answer (finished, graded wrong), with that answer.

Both completions are the model's own tokens; they share the reasoning up to the earlier boundary
and differ in the decision taken there: keep thinking (chosen) or close and answer (rejected).

Each --domain is rollouts:search:continue (continue may be omitted). The search directory must
be the margin-0 run whose train.jsonl the chosen examples come from.
"""

import argparse
from collections import Counter
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.find_stop_points import DOUBLE_NEWLINE, MODEL, THINK_CLOSE, boundaries  # noqa: E402


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
FINAL = re.compile(r"Final answer:\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\.?\s*")


def reached(answer_text, prefix, gold):
    """A wrong math answer that is a single number already written in the reasoning, and not the
    prompt's format example 42. Answers that are wrong only by format (an expression such as
    "5800 + 3600 = 9400") are excluded: the failure under study is the value."""
    last = answer_text.strip().splitlines()[-1] if answer_text.strip() else ""
    match = FINAL.fullmatch(last.strip())
    if not match:
        return False
    value = match.group(1).replace(",", "")
    if value == "42" and gold != "42":
        return False
    return value in {n.replace(",", "") for n in NUMBER.findall(prefix)}


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def domain_pairs(tokenizer, rollouts_dir, search_dir, continue_dir):
    rows = read(Path(rollouts_dir) / "rollouts.jsonl")
    known_questions = {r["question_id"] for r in rows if r["knows"]}
    rows = [r for r in rows if r["question_id"] in known_questions]
    outcomes = read(Path(search_dir) / "stop_points.jsonl")
    state = json.loads((Path(search_dir) / "state.json").read_text())
    assert len(rows) == len(outcomes) == len(state["lo"])
    answers = {tuple(map(int, k.split(":"))): v for k, v in state["answers"].items()}
    if any("knows" not in a for v in answers.values() for a in v):
        raise SystemExit(f"{search_dir}/state.json has ungraded answers; use a graded search state")

    chosen = {}       # rollout index -> (completion ids, index of the chosen cut, kind)
    examples = iter(read(Path(search_dir) / "train.jsonl"))
    for i, o in enumerate(outcomes):
        if o["result"] == "example":
            example = next(examples)
            assert example["question_id"] == rows[i]["question_id"]
            chosen[i] = (example["completion_ids"], state["lo"][i], "verified stop")
    if continue_dir:
        by_question = {}
        for i, r in enumerate(rows):
            by_question.setdefault(r["question_id"], []).append(i)
        for example in read(Path(continue_dir) / "train.jsonl"):
            ids = example["completion_ids"]
            stop = ids.index(THINK_CLOSE) if THINK_CLOSE in ids else None
            if stop is None:
                continue
            matches = [i for i in by_question[example["question_id"]] if rows[i]["token_ids"][:stop] == ids[:stop]]
            if len(matches) != 1 or matches[0] in chosen:
                continue
            i = matches[0]
            if example["kind"].startswith("walked"):
                chosen[i] = (ids, None, "keep thinking")
            elif example["kind"].startswith("full"):
                chosen[i] = (ids, None, "keep thinking (full)")

    pairs, skipped = [], Counter()
    for i, (completion, chosen_index, kind) in chosen.items():
        cuts = boundaries(tokenizer, rows[i])
        stop_position = completion.index(THINK_CLOSE)
        if chosen_index is None:
            later = [k for k, c in enumerate(cuts) if c >= stop_position]
            chosen_index = later[0] if later else len(cuts)
        earlier = sorted((k for (j, k) in answers if j == i and k < chosen_index), reverse=True)
        rejected = None
        for k in earlier:
            wrong = [a for a in answers[(i, k)] if a["finished"] and not a["knows"]]
            if rows[i]["task"] == "math":
                # Keep only the failure under study: committing to a number the reasoning has already
                # reached. A forced answer that copies the prompt's format example ("like: Final
                # answer: 42") or a number never written is not that failure.
                prefix = tokenizer.decode(rows[i]["token_ids"][:cuts[k]])
                wrong = [a for a in wrong if reached(a["text"], prefix, rows[i]["answer"])]
            if wrong:
                answer = min(wrong, key=lambda a: len(a["ids"]))
                rejected = rows[i]["token_ids"][:cuts[k]] + [THINK_CLOSE, DOUBLE_NEWLINE] + answer["ids"]
                break
        if rejected is None:
            skipped["no wrong earlier answer"] += 1
            continue
        pairs.append({"question_id": rows[i]["question_id"], "source": rows[i]["source"], "prompt": rows[i]["prompt"],
                      "chosen_ids": completion, "rejected_ids": rejected, "kind": kind,
                      "chosen_tokens": len(completion), "rejected_tokens": len(rejected)})
    return pairs, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", action="append", required=True, help="rollouts:search[:continue]")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    everything = []
    for spec in arguments.domain:
        parts = spec.split(":")
        pairs, skipped = domain_pairs(tokenizer, parts[0], parts[1], parts[2] if len(parts) > 2 else None)
        kinds = Counter(p["kind"] for p in pairs)
        print(f"{parts[0]}: {len(pairs)} pairs {dict(kinds)}; skipped {dict(skipped)}; "
              f"chosen {sum(p['chosen_tokens'] for p in pairs) / max(1, len(pairs)):.0f} tokens, "
              f"rejected {sum(p['rejected_tokens'] for p in pairs) / max(1, len(pairs)):.0f}", flush=True)
        everything += pairs
    arguments.out.write_text("".join(json.dumps(p) + "\n" for p in everything))
    print(f"wrote {arguments.out}: {len(everything)} pairs")


if __name__ == "__main__":
    main()
