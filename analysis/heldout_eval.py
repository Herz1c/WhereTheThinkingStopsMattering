"""Held-out check: tokens and accuracy on benchmarks never looked at during development.

    python analysis/heldout_eval.py --out runs/heldout/base
    python analysis/heldout_eval.py --out runs/heldout/stoppoint_mixG --adapter checkpoints/stoppoint_mixG
    python analysis/heldout_eval.py --compare runs/heldout/base runs/heldout/stoppoint_mixG

Every design decision of this project was made while looking at the four evaluation benchmarks,
so their numbers can be flattered by selection even though nothing was trained on them. This
runs a model once on two sets that played no part in any decision:

- HumanEval (openai/openai_humaneval, 164 tasks, 4 samples): code, like MBPP but other tasks; the
  task's own `check` run in the code sandbox (stopping/code_tasks.py);
- GSM8K test (openai/gsm8k main/test, the first 500 of 1,319 problems, 2 samples): grade-school math word problems
  like SVAMP; only GSM8K's train split was ever used for training.

Prompts are written once here, in the style of the training tasks; the evaluation's sampling and
4,096-token cap (code 8,192) apply. Reported per set: tokens saved against base (ratio of sums,
as compare.py), accuracy, capped responses, paired bootstrap intervals over questions.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.grading import grade  # noqa: E402
from utils.eval.recipe import SAMPLING  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
CODE_PROMPT = ("Complete the following Python function. Give the complete function, including the signature, "
               "in a single fenced Python code block.\n\n```python\n{prompt}```")
MATH_PROMPT = "{question}\n\nPut the final answer on the last line, like: Final answer: 42"


def questions(code_samples=4, math_questions=500):
    from datasets import load_dataset
    rows = []
    for r in load_dataset("openai/openai_humaneval", split="test"):
        tests = f"{r['test']}\n\ndef test_humaneval():\n    check({r['entry_point']})\n"
        rows.append({"set": "humaneval", "id": r["task_id"], "task": "code", "samples": code_samples, "max_tokens": 8192,
                     "tests": tests, "entry_point": r["entry_point"],
                     "prompt": [{"role": "user", "content": CODE_PROMPT.format(prompt=r["prompt"])}]})
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test")):
        if i == math_questions:
            break
        rows.append({"set": "gsm8k_test", "id": f"gsm8k-test-{i}", "task": "math", "samples": 2, "max_tokens": 4096,
                     "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
                     "prompt": [{"role": "user", "content": MATH_PROMPT.format(question=r["question"])}]})
    return rows


def run(arguments):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rank = json.loads((arguments.adapter / "adapter_config.json").read_text())["r"] if arguments.adapter else None
    llm = LLM(model=MODEL, gpu_memory_utilization=0.85, max_model_len=8192 + 1024, seed=9001,
              enable_lora=rank is not None, max_lora_rank=rank or 16)
    lora = LoRARequest("candidate", 1, str(arguments.adapter.resolve())) if arguments.adapter else None
    rows = questions()
    records = []
    for group in ({"humaneval"}, {"gsm8k_test"}):
        part = [r for r in rows if r["set"] in group]
        params = SamplingParams(**SAMPLING, n=part[0]["samples"], max_tokens=part[0]["max_tokens"], seed=9001)
        prompts = [tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=True) for r in part]
        for r, result in zip(part, llm.generate(prompts, params, lora_request=lora)):
            for k, o in enumerate(result.outputs):
                text = tokenizer.decode(list(o.token_ids), skip_special_tokens=True)
                records.append({"set": r["set"], "id": r["id"], "sample": k, "tokens": len(o.token_ids),
                                "truncated": o.finish_reason == "length", "row": r, "text": text})
    verdicts = grade([(rec["row"], rec["text"]) for rec in records])
    arguments.out.mkdir(parents=True, exist_ok=True)
    with (arguments.out / "records.jsonl").open("w") as stream:
        for rec, (strict, knows) in zip(records, verdicts):
            rec.pop("row")
            stream.write(json.dumps(rec | {"correct": strict, "knows": knows}) + "\n")
    print(f"wrote {len(records)} records to {arguments.out}")


def compare(base_dir, mine_dir, resamples=2000):
    def load(d):
        out = {}
        for line in (d / "records.jsonl").read_text().splitlines():
            r = json.loads(line)
            out.setdefault(r["set"], {}).setdefault(r["id"], []).append(r)
        return out
    base, mine = load(base_dir), load(mine_dir)
    rng = random.Random(0)
    for name in sorted(base):
        ids = sorted(set(base[name]) & set(mine.get(name, {})))
        def stats(sel):
            bt = sum(r["tokens"] for q in sel for r in base[name][q])
            mt = sum(r["tokens"] for q in sel for r in mine[name][q])
            ba = sum(r["correct"] for q in sel for r in base[name][q]) / sum(len(base[name][q]) for q in sel)
            ma = sum(r["correct"] for q in sel for r in mine[name][q]) / sum(len(mine[name][q]) for q in sel)
            return 1 - mt / bt, 100 * (ma - ba), ba, ma
        saved, delta, ba, ma = stats(ids)
        boot = sorted((stats([rng.choice(ids) for _ in ids])[:2] for _ in range(resamples)))
        s_lo, s_hi = sorted(b[0] for b in boot)[int(.025 * resamples)], sorted(b[0] for b in boot)[int(.975 * resamples)]
        a_lo, a_hi = sorted(b[1] for b in boot)[int(.025 * resamples)], sorted(b[1] for b in boot)[int(.975 * resamples)]
        capped = lambda d: sum(r["truncated"] for q in ids for r in d[name][q])
        print(f"{name}  ({len(ids)} questions)  base acc {ba:.1%} -> {ma:.1%}")
        print(f"  tokens saved     {saved:+.1%}  CI [{s_lo:+.1%}, {s_hi:+.1%}]")
        print(f"  accuracy change  {delta:+.2f} pp  CI [{a_lo:+.2f}, {a_hi:+.2f}]   capped {capped(base)} -> {capped(mine)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    arguments = parser.parse_args()
    if arguments.compare:
        compare(*arguments.compare)
    else:
        run(arguments)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
