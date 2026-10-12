"""Give-up examples: after a long attempt that is going nowhere, close the thinking and answer.

    python stopping/give_up_examples.py runs/rollouts_code --out runs/give_up_code \
        --budget 2048 --answer-tokens 1024 --max-rollout-tokens 6144 --max-examples 600

Responses that hit the evaluation's token cap are graded wrong and, being the longest, hold a
large share of all tokens (a quarter or more of a stop-trained model's). Ending them early saves
those tokens at no accuracy cost, provided the model only gives up on attempts that would not
have finished anyway.

Source: rollouts that hit the rollout cap (`finished` false), at most one per question. Each is
cut at the first paragraph boundary at or after `--budget` thinking tokens, the thinking is
closed there and answers are sampled. The example keeps the model's own reasoning up to the cut,
then `</think>` and:

- a right, strictly formatted answer when one was sampled ("rescued at the budget"), else
- a wrong answer in the evaluation's format: one fenced block defining the entry point for code,
  a final "Final answer: <number>" line for math ("best guess at the budget").

A forced answer is the model's best guess from what it has worked out, never a refusal: a
refusal is always graded wrong, a guess sometimes passes. The budget sits above most correct
answers' length (see PROGRESS.md), so solvable questions are rarely reached by it.
Writes <out>/train.jsonl.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.find_stop_points import DOUBLE_NEWLINE, MODEL, THINK_CLOSE, boundaries, forced_ids  # noqa: E402
from stopping.grading import grade  # noqa: E402
from utils.eval.recipe import SAMPLING  # noqa: E402
from utils.train.rewards import final_answer  # noqa: E402

SAMPLES = 4
NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def well_formed(row, text):
    """Is the answer in the format the evaluation reads, right or wrong?"""
    if row["task"] == "code":
        from stopping.code_tasks import strict_code
        code = strict_code(text)
        return code is not None and f"def {row['entry_point']}" in code
    predicted = final_answer(text.replace("**", ""))
    return predicted is not None and bool(NUMBER.fullmatch(predicted.strip().rstrip(".")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=2048, help="Thinking tokens before giving up")
    parser.add_argument("--answer-tokens", type=int, default=384)
    parser.add_argument("--max-rollout-tokens", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=600)
    parser.add_argument("--include-long", action="store_true",
                        help="Also cut finished rollouts whose thinking runs past 1.25 x budget; a correct one "
                             "is kept only when the forced answer is still right (never teach giving up on "
                             "an attempt that would have succeeded)")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    rows = [json.loads(l) for l in (arguments.rollouts / "rollouts.jsonl").read_text().splitlines()]
    chosen, seen = [], set()
    for r in rows:
        long = arguments.include_long and r["thinking_tokens"] > 1.25 * arguments.budget
        if (r["finished"] and not long) or r["question_id"] in seen:
            continue
        cuts = [c for c in boundaries(tokenizer, r) if c >= arguments.budget]
        if not cuts or (r["closed"] and cuts[0] >= r["token_ids"].index(THINK_CLOSE)):
            continue    # no paragraph boundary inside the thinking after the budget
        seen.add(r["question_id"])
        chosen.append((r, cuts[0]))
        if len(chosen) == arguments.max_examples:
            break
    print(f"{len(chosen)} rollouts to cut at >= {arguments.budget} thinking tokens "
          f"({sum(not r['finished'] for r, _ in chosen)} capped)", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=0.85,
              max_model_len=arguments.max_rollout_tokens + 1024 + arguments.answer_tokens, seed=arguments.seed)
    prompts = []
    for r, cut in chosen:
        ids = tokenizer.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=True)
        prompts.append({"prompt_token_ids": forced_ids(list(ids["input_ids"] if hasattr(ids, "input_ids") else ids), r, cut)})
    params = SamplingParams(**SAMPLING, n=SAMPLES, max_tokens=arguments.answer_tokens, seed=arguments.seed + 300)
    outputs = llm.generate(prompts, params)
    flat = [(k, o) for k, result in enumerate(outputs) for o in result.outputs]
    verdicts = grade([(chosen[k][0], "</think>" + o.text) for k, o in flat])

    examples, kinds = [], Counter()
    by_row = {}
    for (k, o), (strict, _) in zip(flat, verdicts):
        if o.finish_reason == "stop":
            by_row.setdefault(k, []).append((o, strict, well_formed(chosen[k][0], o.text)))
    for k, (r, cut) in enumerate(chosen):
        sampled = by_row.get(k, [])
        right = [o for o, strict, _ in sampled if strict]
        guess = [o for o, strict, formed in sampled if formed and not strict]
        if right:
            answer, kind = min(right, key=lambda o: len(o.token_ids)), "rescued at the budget"
        elif r["finished"] and r["correct"]:
            kinds["dropped: long attempt that succeeds, not known at the budget"] += 1
            continue
        elif guess:
            answer, kind = min(guess, key=lambda o: len(o.token_ids)), "best guess at the budget"
        else:
            kinds["dropped: no well-formed answer"] += 1
            continue
        completion = r["token_ids"][:cut] + [THINK_CLOSE, DOUBLE_NEWLINE] + list(answer.token_ids)
        examples.append({"question_id": r["question_id"], "source": r["source"], "prompt": r["prompt"],
                         "completion_ids": completion, "tokens": len(completion), "kind": kind,
                         "original_tokens": r["tokens"], "cut": cut})
        kinds[kind] += 1

    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    before = sum(e["original_tokens"] for e in examples)
    after = sum(e["tokens"] for e in examples)
    print(f"\n{len(examples)} give-up examples, mean {after / max(1, len(examples)):.0f} tokens "
          f"(their rollouts {before / max(1, len(examples)):.0f})")
    for kind, n in kinds.most_common():
        print(f"  {kind:40s} {n}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)     # see find_stop_points.py: vLLM teardown can hang after outputs are written
