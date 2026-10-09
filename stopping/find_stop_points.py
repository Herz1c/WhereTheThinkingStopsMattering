"""Stop-point distillation data: where in its own reasoning could each rollout have stopped?

    python stopping/find_stop_points.py runs/rollouts_gsm8k --out runs/stop_points

For every rollout written by generate_rollouts.py, finished or truncated, this finds the
earliest paragraph boundary in the thinking from which the model, made to close
</think> there, still answers correctly. The training example is then the model's own
reasoning up to that boundary, </think>, and one of its own correct answers from there.
Nothing in the example is written by anyone but the model: it differs from the rollout
only in where the thinking ends.

Truncated rollouts matter most. They hold the largest share of tokens and always score
as wrong; if the model already knew the answer somewhere inside one, the example teaches
it to leave there.

Search, per rollout, over the token positions right after a token that ends a paragraph
("\\n\\n"), from the end of the first paragraph on (the model must keep thinking):

- A boundary is *good* when both of 2 sampled answers from it know the answer (the lenient
  grader of analysis/forced_stop_sweep.py: last "Final answer:" or \\boxed{}).
- Binary search for the earliest good boundary, assuming that once the answer is known it
  stays known. All rollouts advance one step per round, so each round is one batched vLLM
  call with prefix caching.
- The boundary found is then verified with 4 more answers (at least 3 must know).
- The example's answer is the shortest of the 6 sampled from that boundary that passes the
  strict grader, so it ends on the "Final answer:" line the evaluation reads. Without one,
  the rollout is dropped.

Rollouts of questions where no rollout knew the answer are skipped: there is nothing to find.
For a finished rollout the search can end at its own </think>; such a rollout has no earlier
stop and yields no example.

Writes <out>/stop_points.jsonl (one row per searched rollout, with the outcome) and
<out>/train.jsonl (rows for stopping/train_lora.py). State is saved after every round, so an
interrupted run resumes.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.forced_stop_sweep import knows  # noqa: E402
from utils.eval.recipe import SAMPLING  # noqa: E402
from utils.train.rewards import is_task_correct  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
THINK_CLOSE = 151668        # </think>
DOUBLE_NEWLINE = 271        # "\n\n", which the model writes right after </think>
PROBE_SAMPLES = 2
VERIFY_SAMPLES = 4
VERIFY_NEEDED = 3
ANSWER_TOKENS = 384


def boundaries(tokenizer, row):
    """Token positions just after each paragraph end inside the thinking, first paragraph excluded,
    plus the rollout's own </think> position when it closed one."""
    ids = row["token_ids"]
    end = ids.index(THINK_CLOSE) if row["closed"] else len(ids)
    pieces = tokenizer.convert_ids_to_tokens(ids[:end])
    cuts = [i + 1 for i, piece in enumerate(pieces) if piece.endswith("ĊĊ")][1:]
    if row["closed"] and (not cuts or cuts[-1] < end):
        cuts.append(end)    # stopping where it did: the search may conclude nothing earlier is known
    return cuts


def forced_ids(prompt_ids, row, cut):
    """Prompt, own thinking up to `cut`, then </think> and the blank line the model writes after it.

    At the rollout's own end the thinking already ends in "\\n", exactly as the model wrote it.
    At a paragraph boundary it ends in "\\n\\n", one newline more than the model's own closing.
    """
    return prompt_ids + row["token_ids"][:cut] + [THINK_CLOSE, DOUBLE_NEWLINE]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path, help="A directory written by generate_rollouts.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Only the first N rollouts (for a test)")
    parser.add_argument("--margin", type=int, default=0,
                        help="Stop this many paragraph boundaries after the earliest good one: "
                             "less aggressive, one point on the length/accuracy curve")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    rows = [json.loads(line) for line in (arguments.rollouts / "rollouts.jsonl").read_text().splitlines()]
    known_questions = {r["question_id"] for r in rows if r["knows"]}
    rows = [r for r in rows if r["question_id"] in known_questions]
    if arguments.limit:
        rows = rows[:arguments.limit]
    arguments.out.mkdir(parents=True, exist_ok=True)
    state_path = arguments.out / "state.json"

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = {}
    for r in rows:
        if r["question_id"] not in prompts:
            ids = tokenizer.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True,
                                                enable_thinking=True)
            prompts[r["question_id"]] = list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)
    cuts = [boundaries(tokenizer, r) for r in rows]

    # Search state per rollout: [lo, hi) over its cut list; hi == len(cuts) means "none good yet".
    # answers[(rollout, cut index)] keeps every sampled answer, so nothing is generated twice.
    if state_path.exists():
        state = json.loads(state_path.read_text())
        lo, hi, rounds = state["lo"], state["hi"], state["rounds"]
        answers = {tuple(map(int, k.split(":"))): v for k, v in state["answers"].items()}
        print(f"resuming after round {rounds}", flush=True)
    else:
        lo, hi, rounds = [0] * len(rows), [len(c) for c in cuts], 0
        answers = {}

    def save():
        state_path.write_text(json.dumps({"lo": lo, "hi": hi, "rounds": rounds,
                                          "answers": {f"{a}:{b}": v for (a, b), v in answers.items()}}))

    llm = LLM(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=0.85,
              max_model_len=4096 + 1024 + ANSWER_TOKENS, seed=arguments.seed)

    def sample(jobs, n):
        """jobs: [(rollout index, cut index)] -> stores n more answers for each."""
        if not jobs:
            return
        params = SamplingParams(**SAMPLING, n=n, max_tokens=ANSWER_TOKENS, seed=arguments.seed + rounds)
        inputs = [{"prompt_token_ids": forced_ids(prompts[rows[i]["question_id"]], rows[i], cuts[i][c])}
                  for i, c in jobs]
        for (i, c), result in zip(jobs, llm.generate(inputs, params)):
            answers.setdefault((i, c), [])
            answers[(i, c)] += [{"ids": list(o.token_ids), "text": o.text,
                                 "finished": o.finish_reason == "stop"} for o in result.outputs]

    def known(i, c, needed):
        texts = answers[(i, c)]
        return sum(a["finished"] and knows("</think>" + a["text"], rows[i]["answer"], rows[i]["task"])
                   for a in texts) >= needed

    started = time.monotonic()
    while True:
        jobs = [(i, (lo[i] + hi[i]) // 2) for i in range(len(rows)) if lo[i] < hi[i]]
        if not jobs:
            break
        sample([j for j in jobs if j not in answers], PROBE_SAMPLES)
        for i, mid in jobs:
            if known(i, mid, PROBE_SAMPLES):    # both probes know: an earlier cut may too
                hi[i] = mid
            else:
                lo[i] = mid + 1
        rounds += 1
        save()
        print(f"round {rounds}: {len(jobs)} probes, {sum(l < h for l, h in zip(lo, hi))} rollouts still searching, "
              f"{(time.monotonic() - started) / 60:.0f} min", flush=True)

    # The stop actually used: the earliest good cut, moved `margin` boundaries later (never past
    # the last cut). A moved cut is checked on its own: it gets 2 + 4 answers like any other.
    chosen = [min(lo[i] + arguments.margin, len(cuts[i]) - 1) if lo[i] < len(cuts[i]) else lo[i]
              for i in range(len(rows))]
    found = [(i, chosen[i]) for i in range(len(rows)) if lo[i] < len(cuts[i])]
    for have in range(PROBE_SAMPLES + VERIFY_SAMPLES):
        sample([(i, c) for i, c in found if len(answers.get((i, c), [])) == have],
               PROBE_SAMPLES + VERIFY_SAMPLES - have)
    save()

    outcomes, train = [], []
    for i, r in enumerate(rows):
        c = chosen[i]
        outcome = {"question_id": r["question_id"], "sample": r["sample"], "tokens": r["tokens"],
                   "closed": r["closed"], "correct": r["correct"], "knows": r["knows"],
                   "boundaries": len(cuts[i])}
        if c >= len(cuts[i]):
            outcome["result"] = "never known"
        else:
            cut = cuts[i][c]
            sampled = answers[(i, c)]
            verified = sum(a["finished"] and knows("</think>" + a["text"], r["answer"], r["task"])
                           for a in sampled[PROBE_SAMPLES:]) >= VERIFY_NEEDED
            clean = [a for a in sampled if a["finished"]
                     and is_task_correct("</think>" + a["text"], r["answer"], r["task"])]
            own_end = r["closed"] and cut == r["token_ids"].index(THINK_CLOSE)
            outcome |= {"stop": cut, "stop_fraction": cut / (r["thinking_tokens"] if r["closed"] else r["tokens"])}
            if own_end:
                outcome["result"] = "no earlier stop"
            elif not verified:
                outcome["result"] = "failed verification"
            elif not clean:
                outcome["result"] = "no clean answer"
            else:
                answer = min(clean, key=lambda a: len(a["ids"]))
                completion = r["token_ids"][:cut] + [THINK_CLOSE, DOUBLE_NEWLINE] + answer["ids"]
                outcome |= {"result": "example", "new_tokens": len(completion)}
                train.append({"question_id": r["question_id"], "source": r["source"], "prompt": r["prompt"],
                              "completion_ids": completion, "tokens": len(completion),
                              "original_tokens": r["tokens"], "original_closed": r["closed"],
                              "original_correct": r["correct"]})
        outcomes.append(outcome)

    (arguments.out / "stop_points.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outcomes))
    (arguments.out / "train.jsonl").write_text("".join(json.dumps(t) + "\n" for t in train))

    counts = defaultdict(int)
    for o in outcomes:
        counts[(o["closed"], o["result"])] += 1
    print(f"\nmargin {arguments.margin}: {len(rows)} rollouts searched in {rounds} rounds, "
          f"{(time.monotonic() - started) / 60:.0f} min")
    for (closed, result), n in sorted(counts.items()):
        print(f"  {'finished ' if closed else 'truncated'}  {result:20s} {n}")
    if train:
        before = sum(t["original_tokens"] for t in train)
        after = sum(t["tokens"] for t in train)
        print(f"{len(train)} examples: {after / len(train):.0f} tokens on average, "
              f"{1 - after / before:.0%} fewer than their rollouts; "
              f"{sum(not t['original_closed'] for t in train)} rescued from truncation, "
              f"{sum(t['original_closed'] and not t['original_correct'] for t in train)} from wrong finished rollouts")


if __name__ == "__main__":
    main()
