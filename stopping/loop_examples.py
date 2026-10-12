"""End-the-answer examples: the model's own reply cut where it starts repeating itself.

    python stopping/loop_examples.py --out runs/loop_dolly --prompts 4000

Some replies close the thinking, answer, and then repeat the answer (a line, a paragraph, a
list) until the token cap. Those repeats add nothing and are a large share of the tokens on
open-ended instructions. Source prompts: databricks-dolly-15k (human-written instructions,
CC BY-SA 3.0), unrelated to any evaluation benchmark; prompts sharing 8-grams with an evaluation
question are dropped (analysis/overlap_check.py). The base model answers each prompt twice with
the evaluation's sampling; replies that hit the cap after `</think>` with a repeated block of
lines are kept up to the first repeat, and the trainer appends the end-of-turn token. The
thinking is left exactly as the model wrote it. Writes <out>/train.jsonl.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.overlap_check import grams, question_text  # noqa: E402
from stopping.find_stop_points import MODEL, THINK_CLOSE  # noqa: E402
from utils.eval.recipe import SAMPLING  # noqa: E402

REPEAT = 3          # a block of this many lines seen before marks the loop


def loop_start(answer):
    """Character offset where the reply starts repeating an earlier block of lines, or None."""
    lines, offsets, position = [], [], 0
    for line in answer.split("\n"):
        if line.strip():
            lines.append(line.strip())
            offsets.append(position)
        position += len(line) + 1
    seen = {}
    for i in range(len(lines) - REPEAT + 1):
        block = tuple(lines[i:i + REPEAT])
        if block in seen and sum(len(l) for l in block) >= 20:
            return offsets[i]
        seen.setdefault(block, i)
    return None


def repeat_start(text, max_period=400, min_repeats=4, min_length=24):
    """Start of the periodic tail of `text` (a unit of up to `max_period` characters repeated at
    least `min_repeats` times up to the end), or None. Catches line loops and character loops such
    as a quotation mark on a new line written until the cap."""
    n, best = len(text), None
    for k in range(1, min(max_period, n // min_repeats) + 1):
        run = 0
        while run < n - k and text[n - 1 - run] == text[n - 1 - run - k]:
            run += 1
        if run + k >= max(k * min_repeats, min_length):
            start = n - run - k
            best = start if best is None else min(best, start)
    return best


def from_rollouts(arguments):
    """Loop cuts from constraint rollouts: replies capped after </think> whose tail repeats are cut
    before the repeats, the wrapper closed if the constraint has one, and kept if the reply then
    passes its constraints."""
    from transformers import AutoTokenizer
    from stopping.constraint_tasks import WRAPPERS, grade_constraint
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in (arguments.rollouts / "rollouts.jsonl").read_text().splitlines()]
    examples, kinds, used = [], Counter(), set()
    for r in rows:
        if r["finished"] or not r["closed"]:
            continue
        kinds["capped after </think>"] += 1
        ids = r["token_ids"]
        close = ids.index(THINK_CLOSE)
        answer = tokenizer.decode(ids[close + 1:], skip_special_tokens=True)
        start = repeat_start(answer)
        if start is None or r["question_id"] in used:
            kinds["no periodic tail"] += 1
            continue
        reply = answer[:start].rstrip()
        for c in r["answer"]:
            if c["kind"] in WRAPPERS and not reply.endswith(WRAPPERS[c["kind"]][2]):
                reply += "\n" + WRAPPERS[c["kind"]][2]
        if not grade_constraint(r, "</think>" + reply)[0]:
            kinds["cut reply fails its constraints"] += 1
            continue
        used.add(r["question_id"])
        completion = ids[:close + 1] + tokenizer.encode(reply, add_special_tokens=False)
        examples.append({"question_id": r["question_id"], "source": r["source"], "prompt": r["prompt"],
                         "completion_ids": completion, "tokens": len(completion), "kind": "reply cut at its loop",
                         "original_tokens": len(ids)})
        kinds["kept"] += 1
    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    mean = sum(e["tokens"] for e in examples) / max(1, len(examples))
    print(f"\n{len(examples)} loop examples from {len(rows)} rollouts, mean {mean:.0f} tokens")
    for kind, n in kinds.most_common():
        print(f"  {kind:32s} {n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, help="Cut loops in these constraint rollouts instead of "
                                                        "generating Dolly replies")
    parser.add_argument("--prompts", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--questions", default="questions/questions.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.rollouts:
        return from_rollouts(arguments)

    from datasets import load_dataset
    rows = list(load_dataset("databricks/databricks-dolly-15k", split="train"))
    random.Random(arguments.seed).shuffle(rows)
    evaluation = set()
    for line in Path(arguments.questions).read_text().splitlines():
        evaluation |= grams(question_text(json.loads(line)["messages"]))
    prompts, overlapping = [], 0
    for r in rows:
        text = r["instruction"].strip() + (f"\n\n{r['context'].strip()}" if r["context"].strip() else "")
        if len(text) > 2000:
            continue
        g = grams(text)
        if g and len(g & evaluation) / len(g) >= 0.15:
            overlapping += 1
            continue
        prompts.append({"question_id": f"dolly-{len(prompts)}", "source": "dolly", "category": r["category"],
                        "prompt": [{"role": "user", "content": text}]})
        if len(prompts) == arguments.prompts:
            break
    print(f"{len(prompts)} Dolly prompts ({overlapping} dropped for sharing 8-grams with evaluation questions)",
          flush=True)

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, gpu_memory_utilization=0.85, max_model_len=arguments.max_tokens + 2048,
              seed=arguments.seed)
    params = SamplingParams(**SAMPLING, n=arguments.samples, max_tokens=arguments.max_tokens,
                            seed=arguments.seed + 400)
    inputs = []
    for p in prompts:
        ids = tokenizer.apply_chat_template(p["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=True)
        inputs.append({"prompt_token_ids": list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)})
    outputs = llm.generate(inputs, params)

    examples, kinds, used = [], Counter(), set()
    for p, result in zip(prompts, outputs):
        for o in result.outputs:
            ids = list(o.token_ids)
            kinds["replies"] += 1
            if o.finish_reason != "length":
                continue
            if THINK_CLOSE not in ids:
                kinds["capped inside the thinking"] += 1
                continue
            kinds["capped after </think>"] += 1
            close = ids.index(THINK_CLOSE)
            answer = tokenizer.decode(ids[close + 1:])
            start = loop_start(answer)
            if start is None or len(answer[:start].strip()) < 20 or p["question_id"] in used:
                kinds["no clean repeat to cut"] += 1
                continue
            completion = ids[:close + 1] + tokenizer.encode(answer[:start].rstrip(), add_special_tokens=False)
            used.add(p["question_id"])
            examples.append({"question_id": p["question_id"], "source": "dolly", "prompt": p["prompt"],
                             "completion_ids": completion, "tokens": len(completion), "kind": "reply cut at its loop",
                             "original_tokens": len(ids), "category": p["category"]})
            kinds["kept"] += 1

    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    mean = sum(e["tokens"] for e in examples) / max(1, len(examples))
    print(f"\n{len(examples)} loop examples, mean {mean:.0f} tokens (capped at {arguments.max_tokens})")
    for kind, n in kinds.most_common():
        print(f"  {kind:32s} {n}")
    print("  categories:", dict(Counter(e["category"] for e in examples)))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)     # see find_stop_points.py: vLLM teardown can hang after outputs are written
