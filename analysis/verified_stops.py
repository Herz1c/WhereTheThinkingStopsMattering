"""How much later would stops be if the model must have checked its answer first?

    python analysis/verified_stops.py runs/rollouts_code runs/stop_points_code

The stop points of find_stop_points.py are where an *oracle* (unit tests, the gold number) says
the forced answer is right. The model never sees that oracle, so it can learn "stop here" at
places where, from its own reasoning, nothing is settled yet. A stricter target: the earliest
cut at or after the oracle stop where the reasoning so far contains a *check by the model
itself* after the point where the answer was formed: tracing an example or test for code,
re-verifying a computation for math.

This measures, on the existing search, how far that moves each stop and what it does to the
length of the training examples, before anything is trained. A check is a paragraph matching
the domain's cue pattern, located in the window from two paragraphs before the oracle stop
onwards (a check written just before the stop counts).
"""

import argparse
import json
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CUES = {
    "code": re.compile(r"\b(test(?:ing|s)?|let me (?:check|verify|see|try)|let's (?:check|verify|test|try|see)|"
                       r"for example|e\.g\.|sample input|should (?:return|output|give)|would return|returns? \S+ which|"
                       r"assert|trace|walk through|edge case|example input|input is|output is)\b", re.IGNORECASE),
    "math": re.compile(r"\b(check|verify|double[- ]check|confirm|let me (?:recheck|re-check|recalculate|make sure)|"
                       r"plug(?:ging)? (?:it |this |that )?back|substitut|that matches|makes sense|consistent|"
                       r"sanity)\b", re.IGNORECASE),
}


CONCRETE = re.compile(r"\d|'[^']{1,40}'|\"[^\"]{1,40}\"|\[[^\]]{0,40}\]")
OUTCOME = re.compile(r"\b(correct|works|which matches|matches|that's right|yes|returns?|gives|equals|outputs?|"
                     r"result is|is indeed)\b|==|=|→|->", re.IGNORECASE)


def is_check(paragraph, cue):
    """A paragraph in which the model tries a concrete case and looks at what comes out:
    a check cue, a concrete value (number, string, list) and an outcome word or sign. Mere
    mentions ("the function should return an empty string") do not count."""
    return bool(cue.search(paragraph) and CONCRETE.search(paragraph) and OUTCOME.search(paragraph))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("search", type=Path)
    arguments = parser.parse_args()

    from transformers import AutoTokenizer
    from stopping.find_stop_points import MODEL, THINK_CLOSE, boundaries
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    rows = [json.loads(l) for l in (arguments.rollouts / "rollouts.jsonl").read_text().splitlines()]
    known = {r["question_id"] for r in rows if r["knows"]}
    rows = [r for r in rows if r["question_id"] in known]
    outcomes = [json.loads(l) for l in (arguments.search / "stop_points.jsonl").read_text().splitlines()]
    state = json.loads((arguments.search / "state.json").read_text())
    task = rows[0]["task"]
    cue = CUES["code" if task == "code" else "math"]

    oracle_len, verified_len, full_len, moved, never = [], [], [], 0, 0
    for i, (r, o) in enumerate(zip(rows, outcomes)):
        if o["result"] != "example":
            continue
        cuts = boundaries(tokenizer, r)
        lo = state["lo"][i]
        stop = None
        for k in range(max(0, lo - 2), len(cuts)):
            paragraph = tokenizer.decode(r["token_ids"][cuts[k - 1] if k > 0 else 0:cuts[k]])
            if is_check(paragraph, cue):
                stop = cuts[max(k, lo)]
                break
        end = r["token_ids"].index(THINK_CLOSE) if r["closed"] else len(r["token_ids"])
        if stop is None:
            never += 1
            stop = end
        moved += stop > cuts[lo]
        oracle_len.append(cuts[lo])
        verified_len.append(stop)
        full_len.append(end)

    n = len(oracle_len)
    print(f"{arguments.search}: {n} stop examples ({task})")
    print(f"  thinking tokens kept, median / mean:  oracle stop {statistics.median(oracle_len):.0f} / "
          f"{statistics.mean(oracle_len):.0f}   verified stop {statistics.median(verified_len):.0f} / "
          f"{statistics.mean(verified_len):.0f}   full rollout {statistics.median(full_len):.0f} / "
          f"{statistics.mean(full_len):.0f}")
    print(f"  share of the rollout's thinking cut: oracle {1 - sum(oracle_len) / sum(full_len):.0%}, "
          f"verified {1 - sum(verified_len) / sum(full_len):.0%}")
    print(f"  stops moved later: {moved / n:.0%}; no check before the rollout's end: {never / n:.0%}")


if __name__ == "__main__":
    main()
