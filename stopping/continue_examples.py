""""Keep thinking" examples: where stopping early would have been wrong.

    python stopping/continue_examples.py runs/rollouts_gsm8k runs/stop_points --out runs/continue_math

Stop-point data (find_stop_points.py) teaches only one thing: close </think> at the earliest
point where the answer is known. Rollouts that needed their full reasoning were dropped, so the
model never sees a case where stopping at the first plausible answer is wrong. On SVAMP the
stop-trained models lost accuracy exactly that way: they committed to the first number reached.
This adds the missing side, from the same rollouts and the same finished search:

A. *Unstable stops* ("failed verification"): the 2 probes at the earliest good cut knew the
   answer, but 4 more samples there did not confirm it, so the answer at that point was still
   unsettled. Walk forward one paragraph at a time and take the first cut where at least 5 of 6
   sampled answers know the answer and one passes the strict grader; if none does before the
   rollout's own end and the rollout itself is correct, take the whole rollout. The example then
   keeps thinking past the unstable point.
B. *No earlier stop*: correct, finished rollouts whose earliest good cut was their own end. The
   whole rollout is the example. Capped per domain by --max-full (tool calls have thousands).

Rows are written in the trainer's format (prompt + completion token ids) to <out>/train.jsonl.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.find_stop_points import (DOUBLE_NEWLINE, MODEL, THINK_CLOSE, boundaries,  # noqa: E402
                                       forced_ids)
from stopping.grading import grade  # noqa: E402
from utils.eval.recipe import SAMPLING  # noqa: E402

STABLE_NEEDED = 5       # of 6 sampled answers
SAMPLES = 6
MAX_STEPS = 12          # paragraphs to walk forward before falling back to the full rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("search", type=Path, help="The find_stop_points.py output directory (margin 0)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--answer-tokens", type=int, default=384)
    parser.add_argument("--max-rollout-tokens", type=int, default=4096)
    parser.add_argument("--max-full", type=int, default=1000, help="Cap on type-B examples")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    # Rebuild the search's row list exactly as find_stop_points.py did, so outcomes line up.
    rows = [json.loads(line) for line in (arguments.rollouts / "rollouts.jsonl").read_text().splitlines()]
    known_questions = {r["question_id"] for r in rows if r["knows"]}
    rows = [r for r in rows if r["question_id"] in known_questions]
    outcomes = [json.loads(line) for line in (arguments.search / "stop_points.jsonl").read_text().splitlines()]
    state = json.loads((arguments.search / "state.json").read_text())
    assert len(outcomes) == len(rows) == len(state["lo"]), "search output does not match the rollouts"
    for r, o in zip(rows, outcomes):
        assert (r["question_id"], r["sample"]) == (o["question_id"], o["sample"])

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    examples, kinds = [], Counter()

    def add(r, completion, kind):
        examples.append({"question_id": r["question_id"], "source": r["source"], "prompt": r["prompt"],
                         "completion_ids": completion, "tokens": len(completion), "kind": kind,
                         "original_tokens": r["tokens"]})
        kinds[kind] += 1

    # B: whole correct rollouts with no earlier stop.
    full = [i for i, o in enumerate(outcomes) if o["result"] == "no earlier stop" and rows[i]["correct"]
            and rows[i]["finished"]]
    random.Random(arguments.seed).shuffle(full)
    for i in full[:arguments.max_full]:
        add(rows[i], rows[i]["token_ids"], "full: no earlier stop")

    # A: unstable stops, walked forward.
    unstable = [i for i, o in enumerate(outcomes) if o["result"] == "failed verification"]
    print(f"{len(full)} no-earlier-stop correct rollouts (using {min(len(full), arguments.max_full)}), "
          f"{len(unstable)} unstable stops to walk forward", flush=True)
    if unstable:
        prompts = {}
        for i in unstable:
            q = rows[i]["question_id"]
            if q not in prompts:
                ids = tokenizer.apply_chat_template(rows[i]["prompt"], tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=True)
                prompts[q] = list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)
        cuts = {i: boundaries(tokenizer, rows[i]) for i in unstable}
        position = {i: state["lo"][i] + 1 for i in unstable}
        llm = LLM(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=0.85,
                  max_model_len=arguments.max_rollout_tokens + 1024 + arguments.answer_tokens, seed=arguments.seed)
        pending = list(unstable)
        for step in range(MAX_STEPS):
            # Positions past the rollout's last cut cannot be probed: those fall back below.
            jobs = [i for i in pending if position[i] < len(cuts[i])]
            if not jobs:
                break
            params = SamplingParams(**SAMPLING, n=SAMPLES, max_tokens=arguments.answer_tokens,
                                    seed=arguments.seed + 100 + step)
            outputs = llm.generate([{"prompt_token_ids": forced_ids(prompts[rows[i]["question_id"]], rows[i],
                                                                    cuts[i][position[i]])} for i in jobs], params)
            flat = [(i, o) for i, result in zip(jobs, outputs) for o in result.outputs]
            verdicts = grade([(rows[i], "</think>" + o.text) for i, o in flat])
            by_job = {}
            for (i, o), (strict, knows) in zip(flat, verdicts):
                by_job.setdefault(i, []).append((o, strict, knows and o.finish_reason == "stop"))
            done = set()
            for i, sampled in by_job.items():
                if sum(k for _, _, k in sampled) >= STABLE_NEEDED:
                    clean = [o for o, strict, k in sampled if strict and k]
                    if clean:
                        answer = min(clean, key=lambda o: len(o.token_ids))
                        cut = cuts[i][position[i]]
                        add(rows[i], rows[i]["token_ids"][:cut] + [THINK_CLOSE, DOUBLE_NEWLINE] + list(answer.token_ids),
                            "walked forward to a stable stop")
                        done.add(i)
                        continue
                position[i] += 1
            pending = [i for i in pending if i not in done]
            print(f"step {step + 1}: {len(jobs)} probed, {len(done)} settled, {len(pending)} left", flush=True)
        for i in pending:
            if rows[i]["correct"] and rows[i]["finished"]:
                add(rows[i], rows[i]["token_ids"], "full: never stable earlier")
            else:
                kinds["dropped: unstable and wrong"] += 1

    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    print(f"\n{len(examples)} keep-thinking examples -> {arguments.out / 'train.jsonl'}")
    for kind, n in kinds.most_common():
        print(f"  {kind:32s} {n}")
    if examples:
        print(f"mean tokens {sum(e['tokens'] for e in examples) / len(examples):.0f}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)     # see find_stop_points.py: vLLM teardown can hang after outputs are written
