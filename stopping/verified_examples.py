"""Stop-point examples that stop only after the model has checked its own answer.

    python stopping/verified_examples.py runs/rollouts_code runs/stop_points_code --mode strict \
        --out runs/verified_code_strict --answer-tokens 1024 --max-rollout-tokens 6144

find_stop_points.py places a stop where an oracle (unit tests, the gold number) says the forced
answer is right. The model cannot see that oracle, so it learns to stop where, from its own
reasoning, nothing is settled yet; on code it then writes programs it has not traced and more of
them fail. Here the stop is the first paragraph boundary at or after the oracle stop where the
model's reasoning contains a check of its own (analysis/verified_stops.py):

- lenient: a check cue ("let me test", "for example", "should return", ...) anywhere from two
  paragraphs before the oracle stop on;
- strict: a paragraph with a check cue, a concrete value and an outcome ("for example, if n is
  4 ... 2*2=4. Correct.").
- draft (code): the function has been written out in the reasoning (a `def` line or a closed
  code block). Both cue rules above fire on the model restating the task's test ("the test case
  is f(4) should return 8"), so models trained on them still stop after a one-paragraph plan and
  write untried code straight into the answer.
- derived (math): the reasoning has computed the answer ("... = 42"). The math analogue of draft:
  oracle stops on math come where a forced answer happens to be right, often before the model
  has done the last step, and a model trained on them answers with the first plausible number.

The cut found is verified like any other: 6 answers sampled there, at least 5 must know the
answer and one must pass the strict grader. If not, walk forward a paragraph at a time (up to 6);
if the reasoning never checks or never settles, the whole rollout is used when it is correct.
Oracle stops that already satisfy the rule keep their existing example. Writes <out>/train.jsonl.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.verified_stops import CUES, is_check  # noqa: E402
from stopping.find_stop_points import DOUBLE_NEWLINE, MODEL, THINK_CLOSE, boundaries, forced_ids  # noqa: E402
from stopping.grading import grade  # noqa: E402
from utils.eval.recipe import SAMPLING  # noqa: E402

SAMPLES, NEEDED, MAX_STEPS = 6, 5, 6
DRAFT = re.compile(r"```|(^|\n)\s*def \w+\s*\(")     # the function written out, not just planned


def derived_pattern(answer):
    """The answer as the result of a computation in the reasoning ("= 42", "is $42", "gives 42.0"),
    not merely a number that occurs in the question."""
    value = str(answer).replace(",", "").strip()
    if re.fullmatch(r"-?\d+\.0+", value):
        value = value.split(".")[0]
    number = re.escape(value) + (r"(?:\.0+)?" if "." not in value else "")
    return re.compile(r"(?:=|≈|\bis\b|\bare\b|\bequals\b|\bgives\b|\bget\b|\bmakes\b|\btotal of\b)\s*"
                      r"(?:\\\(|\\\$|\$|\\boxed\{)?\s*" + number + r"(?![\d,]*\d|\.\d)")


def check_cut(tokenizer, row, cuts, lo, mode, cue):
    """Index of the first cut at or after `lo` that follows a check (in draft mode a complete code
    draft, in derived mode the computed answer), or None."""
    ids = row["token_ids"]
    start_index = max(0, lo - 2)
    if mode in ("draft", "derived"):
        if mode == "draft":
            done = lambda text: DRAFT.search(text) and text.count("```") % 2 == 0
        else:
            done = derived_pattern(row["answer"]).search
        thinking = tokenizer.decode(ids[:cuts[lo - 1]]) if lo > 0 else ""
        for k in range(lo, len(cuts)):
            thinking += tokenizer.decode(ids[cuts[k - 1] if k > 0 else 0:cuts[k]])
            if done(thinking):
                return k
        return None
    if mode == "lenient":
        start = cuts[start_index - 1] if start_index > 0 else 0
        for k in range(lo, len(cuts)):
            if cue.search(tokenizer.decode(ids[start:cuts[k]])):
                return k
        return None
    for k in range(start_index, len(cuts)):
        paragraph = tokenizer.decode(ids[cuts[k - 1] if k > 0 else 0:cuts[k]])
        if is_check(paragraph, cue):
            return max(k, lo)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("search", type=Path, help="find_stop_points.py output (margin 0)")
    parser.add_argument("--mode", choices=("lenient", "strict", "draft", "derived"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--answer-tokens", type=int, default=384)
    parser.add_argument("--max-rollout-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    rows = [json.loads(l) for l in (arguments.rollouts / "rollouts.jsonl").read_text().splitlines()]
    known = {r["question_id"] for r in rows if r["knows"]}
    rows = [r for r in rows if r["question_id"] in known]
    outcomes = [json.loads(l) for l in (arguments.search / "stop_points.jsonl").read_text().splitlines()]
    state = json.loads((arguments.search / "state.json").read_text())
    existing = iter(json.loads(l) for l in (arguments.search / "train.jsonl").read_text().splitlines())
    cue = CUES["code" if rows[0]["task"] == "code" else "math"]

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    examples, kinds = [], Counter()

    def add(r, completion, kind):
        examples.append({"question_id": r["question_id"], "source": r["source"], "prompt": r["prompt"],
                         "completion_ids": completion, "tokens": len(completion), "kind": kind,
                         "original_tokens": r["tokens"]})
        kinds[kind] += 1

    def full_or_drop(r, reason):
        if r["correct"] and r["finished"]:
            add(r, r["token_ids"], f"full rollout: {reason}")
        else:
            kinds[f"dropped: {reason}"] += 1

    pending, cuts_of = {}, {}
    for i, o in enumerate(outcomes):
        if o["result"] != "example":
            continue
        example = next(existing)
        r = rows[i]
        cuts = boundaries(tokenizer, r)
        lo = state["lo"][i]
        k = check_cut(tokenizer, r, cuts, lo, arguments.mode, cue)
        if k is None:
            full_or_drop(r, "no check")
        elif k == lo:
            examples.append(example | {"kind": "oracle stop already after a check"})
            kinds["oracle stop already after a check"] += 1
        else:
            pending[i], cuts_of[i] = k, cuts
    print(f"{len(pending)} stops to move later and verify", flush=True)

    if pending:
        from vllm import LLM, SamplingParams
        prompts = {}
        for i in pending:
            q = rows[i]["question_id"]
            if q not in prompts:
                ids = tokenizer.apply_chat_template(rows[i]["prompt"], tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=True)
                prompts[q] = list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)
        llm = LLM(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=0.85,
                  max_model_len=arguments.max_rollout_tokens + 1024 + arguments.answer_tokens, seed=arguments.seed)
        for step in range(MAX_STEPS):
            jobs = [i for i in pending if pending[i] < len(cuts_of[i])]
            if not jobs:
                break
            params = SamplingParams(**SAMPLING, n=SAMPLES, max_tokens=arguments.answer_tokens,
                                    seed=arguments.seed + 200 + step)
            outputs = llm.generate([{"prompt_token_ids": forced_ids(prompts[rows[i]["question_id"]], rows[i],
                                                                    cuts_of[i][pending[i]])} for i in jobs], params)
            flat = [(i, o) for i, result in zip(jobs, outputs) for o in result.outputs]
            verdicts = grade([(rows[i], "</think>" + o.text) for i, o in flat])
            by_job = {}
            for (i, o), (strict, knows) in zip(flat, verdicts):
                by_job.setdefault(i, []).append((o, strict, knows and o.finish_reason == "stop"))
            settled = 0
            for i, sampled in by_job.items():
                clean = [o for o, strict, k in sampled if strict and k]
                if sum(k for _, _, k in sampled) >= NEEDED and clean:
                    answer = min(clean, key=lambda o: len(o.token_ids))
                    cut = cuts_of[i][pending.pop(i)]
                    add(rows[i], rows[i]["token_ids"][:cut] + [THINK_CLOSE, DOUBLE_NEWLINE] + list(answer.token_ids),
                        "moved to a checked, verified stop")
                    settled += 1
                else:
                    pending[i] += 1
            print(f"step {step + 1}: {len(jobs)} probed, {settled} settled, {len(pending)} left", flush=True)
        for i in list(pending):
            full_or_drop(rows[i], "no verified stop after the check")

    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    before = sum(e["original_tokens"] for e in examples)
    after = sum(e["tokens"] for e in examples)
    print(f"\n{arguments.mode}: {len(examples)} examples, mean {after / max(1, len(examples)):.0f} tokens, "
          f"{1 - after / max(1, before):.0%} fewer than their rollouts")
    for kind, n in kinds.most_common():
        print(f"  {kind:40s} {n}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)     # see find_stop_points.py: vLLM teardown can hang after outputs are written
