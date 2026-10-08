"""evaluate.py that can be stopped and resumed, for machines that cannot run for hours.

    python scripts/resumable_eval.py --out runs/base --stop-after 110   # stop cleanly after ~110 min
    python scripts/resumable_eval.py --out runs/base                    # later: carry on where it stopped

Takes exactly evaluate.py's flags (plus --stop-after) and calls evaluate.py's own
functions for everything: question selection, prompts, seeds, sampling, the engine,
grading and results.json. Nothing about what is generated or how it is scored is
reimplemented here; only the generation loop differs, in two ways:

- It skips the batches already in records.jsonl and appends the rest.
- It can stop between batches once --stop-after minutes have passed.

evaluate.py notes that vLLM's output at a fixed seed depends on how requests are
batched. So a run resumes only at a batch boundary: the batches are the same
slices of the same job list as in an uninterrupted run, and a batch cut short by
an interruption is generated again in full. Each resume is logged to resumes.jsonl
next to the results, so the run documents that it was split.

Grading and results.json happen once, when every batch is present.
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate as ev  # noqa: E402

RESUMES = "resumes.jsonl"


def take_stop_after():
    """Pull --stop-after N out of argv so evaluate.py's parser sees only its own flags."""
    if "--stop-after" not in sys.argv:
        return None
    i = sys.argv.index("--stop-after")
    minutes = float(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    return minutes


def complete_batches(out, jobs):
    """Records of the whole batches already on disk, checked against the job list."""
    path = out / ev.RECORDS
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break   # a line cut off by the interruption ends what can be trusted
    keep = len(records) // ev.BATCH * ev.BATCH
    records = records[:keep]
    for record, job in zip(records, jobs):
        if (record["id"], record["sample_index"], record["seed"]) != \
                (job["row"]["id"], job["sample"], job["seed"]):
            raise SystemExit(f"{path} does not match this run's jobs; was it written with other flags?")
    return records


def generate(questions, arguments, out, stop_after):
    """evaluate.generate, resuming at the first missing batch. Returns (records, finished)."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(arguments.model, revision=arguments.revision)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    open_id, close_id = (tokenizer.convert_tokens_to_ids(tag) for tag in ("<think>", "</think>"))
    jobs = ev.build_jobs(questions, tokenizer, arguments)
    longest = max(len(job["ids"]) for job in jobs)
    if longest > ev.PROMPT_BUDGET:
        raise SystemExit(f"A question is {longest} tokens, over the {ev.PROMPT_BUDGET}-token prompt budget")

    records = complete_batches(out, jobs)
    if len(records) == len(jobs):
        return records, True
    with (out / RESUMES).open("a") as log:
        log.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "from_job": len(records), "of": len(jobs)}) + "\n")
    print(f"Resuming at job {len(records)} of {len(jobs)}", flush=True)

    model, lora = ev.load_engine(arguments, max(ev.caps_in_use(questions, arguments).values()))
    started = time.monotonic()
    # Rewrite the trusted prefix first, dropping any partial batch after it.
    (out / ev.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    with (out / ev.RECORDS).open("a") as stream:
        for start in range(len(records), len(jobs), ev.BATCH):
            batch = jobs[start:start + ev.BATCH]
            settings = [SamplingParams(max_tokens=ev.token_cap(job["row"]["dataset"], arguments),
                                       seed=job["seed"], stop_token_ids=list(eos_ids),
                                       **ev.SAMPLING) for job in batch]
            outputs = model.generate([{"prompt_token_ids": job["ids"]} for job in batch],
                                     settings, lora_request=lora)
            if len(outputs) != len(batch):
                raise ValueError("vLLM returned a different number of outputs")
            for job, output in zip(batch, outputs):
                record = ev.to_record(job, output, lora, tokenizer, eos_ids, open_id, close_id)
                records.append(record)
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"{start + len(batch)}/{len(jobs)}", flush=True)
            if stop_after is not None and time.monotonic() - started > stop_after * 60 \
                    and len(records) < len(jobs):
                print(f"Stopped after {stop_after:g} min at {len(records)}/{len(jobs)}. "
                      "Run the same command again to continue.", flush=True)
                return records, False
    return records, True


def main():
    stop_after = take_stop_after()
    arguments = ev.parse_arguments()
    if arguments.regrade:
        raise SystemExit("Use evaluate.py --regrade for regrading")
    root, out = Path(arguments.questions).resolve(), Path(arguments.out).resolve()
    if (out / ev.RESULTS).exists():
        raise SystemExit(f"{out / ev.RESULTS} exists: this run is finished")
    suite = ev.load_suite(root, arguments.smoke)
    listing = "smoke" if arguments.smoke else "questions"
    questions = ev.select(ev.read_jsonl(ev.paths(root)[listing]), arguments.benchmarks)
    out.mkdir(parents=True, exist_ok=True)

    grading = ev.grading_context(suite, arguments)
    records, finished = generate(questions, arguments, out, stop_after)
    if not finished:
        sys.exit(3)
    ev.grade_all(questions, records, grading)
    ev.save_records(out, records)
    results, failed = ev.collect(questions, records, arguments.samples)
    ev.save_results(out, arguments, suite, grading, results, failed,
                    ev.caps_in_use(questions, arguments))
    print(ev.report(results, suite, failed))
    print(f"\nWrote {out / ev.RESULTS}")


if __name__ == "__main__":
    main()
