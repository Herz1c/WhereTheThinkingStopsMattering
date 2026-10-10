"""The base model's own rollouts on training questions: the raw material for every method here.

    python stopping/generate_rollouts.py --out runs/rollouts_gsm8k --sources gsm8k --questions 1500
    python stopping/generate_rollouts.py --out runs/val_stoppoint --data data/val.jsonl --questions 200 \
        --adapter checkpoints/stoppoint      # a checkpoint on held-out questions

Both the reference baseline (SFT on the shortest correct rollout) and stop-point
distillation start from the same thing: several samples of Qwen3-0.6B answering a
training question, with the evaluation's sampling. This writes them once.

Questions come from data/train.jsonl (utils/train/prepare_data.py), never from the
evaluation benchmarks or from data/val.jsonl. Each row keeps the generated token ids,
so later steps can cut and train on exactly what the model produced instead of a
re-tokenization of its text.

Resumable: questions are generated in chunks and appended; running the same command
again skips the questions already written. Every rollout is graded twice, strictly
(as the evaluation would) and leniently (does it know the answer), see
analysis/forced_stop_sweep.py.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stopping.grading import grade  # noqa: E402
from utils.eval.recipe import SAMPLING  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 4096
CARRY = ("tests", "entry_point", "kodcode_id", "subset")     # task-specific fields graders need
CHUNK = 100     # questions per vLLM call; a stop loses at most one chunk


def pick(path, sources, count, seed):
    """`count` training questions from `sources`, in a fixed random order; ids are line numbers."""
    rows = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        row = json.loads(line)
        if row["source"] in sources:
            rows.append({"question_id": index, **row})
    random.Random(seed).shuffle(rows)
    return rows[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--sources", default="gsm8k", help="Comma-separated `source` values")
    parser.add_argument("--questions", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=4, help="Rollouts per question")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Generation cap per rollout")
    parser.add_argument("--adapter", type=Path, help="A LoRA to generate with, e.g. to compare "
                                                      "checkpoints on validation questions")
    arguments = parser.parse_args()

    questions = pick(arguments.data, set(arguments.sources.split(",")), arguments.questions, arguments.seed)
    arguments.out.mkdir(parents=True, exist_ok=True)
    path = arguments.out / "rollouts.jsonl"
    done = set()
    # Keep only whole questions: an interruption can cut a question's samples short.
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip().endswith("}")]
        counts = {}
        for r in rows:
            counts[r["question_id"]] = counts.get(r["question_id"], 0) + 1
        done = {q for q, n in counts.items() if n == arguments.samples}
        path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["question_id"] in done))
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{len(questions) - len(todo)} of {len(questions)} questions done, {len(todo)} to go", flush=True)
    if not todo:
        return

    # As in evaluate.py: FlashInfer's sampler compiles a kernel with nvcc, which
    # driver-only machines do not have. The PyTorch sampler needs no toolchain.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    # Prefix caching lets the samples of one question share its prompt.
    rank = None
    if arguments.adapter:
        rank = json.loads((arguments.adapter / "adapter_config.json").read_text())["r"]
    llm = LLM(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=0.85,
              max_model_len=arguments.max_tokens + 1024, seed=arguments.seed,
              enable_lora=rank is not None, max_lora_rank=rank or 16)
    lora = LoRARequest("candidate", 1, str(arguments.adapter.resolve())) if arguments.adapter else None
    params = SamplingParams(**SAMPLING, n=arguments.samples, max_tokens=arguments.max_tokens, seed=arguments.seed)

    started = time.monotonic()
    with path.open("a") as stream:
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start:start + CHUNK]
            prompts = [tokenizer.apply_chat_template(q["prompt"], tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=True) for q in chunk]
            records = []
            for q, result in zip(chunk, llm.generate(prompts, params, lora_request=lora)):
                for k, output in enumerate(result.outputs):
                    ids = list(output.token_ids)
                    # Drops <|im_end|>, which would otherwise end the "Final answer:" line;
                    # <think> and </think> are not special tokens in Qwen3 and stay.
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    closed = close_id in ids
                    records.append({
                        "question_id": q["question_id"], "sample": k, "source": q["source"],
                        "task": q["task"], "answer": q["answer"], "prompt": q["prompt"],
                        **{key: q[key] for key in CARRY if key in q},
                        "tokens": len(ids), "closed": closed,
                        "thinking_tokens": ids.index(close_id) + 1 if closed else len(ids),
                        "finished": output.finish_reason == "stop",
                        "text": text, "token_ids": ids})
            for record, (strict, known_) in zip(records, grade([(r, r["text"]) for r in records])):
                record["correct"], record["knows"] = strict, known_
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            finished = start + len(chunk)
            elapsed = (time.monotonic() - started) / 60
            print(f"{len(questions) - len(todo) + finished}/{len(questions)} questions, "
                  f"{elapsed:.0f} min, ~{elapsed / finished * (len(todo) - finished):.0f} min left", flush=True)


if __name__ == "__main__":
    main()
    # Everything is written and closed by now. vLLM's engine teardown occasionally hangs after a
    # long run that also forked sandbox workers (seen once, 30 min, after all outputs were saved),
    # which would stall a chain of runs: leave without it.
    sys.stdout.flush()
    os._exit(0)
