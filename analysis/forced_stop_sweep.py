"""How much of the thinking does the answer actually need? Forces </think> part-way.

    python utils/train/prepare_data.py                      # once: writes data/val.jsonl
    python analysis/forced_stop_sweep.py --out runs/sweep    # ~1 h on one consumer GPU
    python analysis/forced_stop_sweep.py --out runs/sweep --plot-only

For a sample of validation questions (never the evaluation benchmarks), the base
model writes full rollouts with the evaluation's sampling. Each rollout's thinking
is cut at paragraph boundaries ("\\n\\n") nearest to fixed fractions of its length,
</think> is inserted there, and the model finishes only the answer. Graded with
utils/train/rewards.py, this gives accuracy as a function of how much of its own
thinking the model was allowed to keep.

If accuracy reaches the full-rollout level well before 100%, the model knows the
answer before it stops thinking -- the premise of stop-point distillation.

Writes <out>/rollouts.jsonl, <out>/forced.jsonl, <out>/summary.json and a figure.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.eval.recipe import SAMPLING  # noqa: E402
from utils.train.rewards import is_task_correct  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_TOKENS = 4096
ANSWER_TOKENS = 512


def pick_questions(path, per_source, seed):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    rng = random.Random(seed)
    picked = []
    for source in sorted(by_source):
        rng.shuffle(by_source[source])
        picked += by_source[source][:per_source]
    return picked


def cut(thinking, fraction):
    """The longest paragraph-aligned prefix of the thinking within `fraction` of its length."""
    if fraction >= 1.0:
        return thinking
    limit = fraction * len(thinking)
    prefix, position = "", 0
    for paragraph in thinking.split("\n\n"):
        end = position + len(paragraph)
        if end > limit:
            break
        prefix = thinking[:end]
        position = end + 2
    return prefix


def forced_prompt(prompt_text, prefix):
    """The chat prompt, the kept thinking, and a closed think block, as Qwen3 writes it."""
    return f"{prompt_text}<think>\n{prefix.strip()}\n</think>\n\n"


def generate(out, questions, rollouts, answers, seed):
    # As in evaluate.py: FlashInfer's sampler compiles a kernel with nvcc, which
    # driver-only machines do not have. The PyTorch sampler needs no toolchain.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Prefix caching is off in evaluate.py for reproducible scores; here it only
    # saves recomputing the shared thinking prefixes, so it is on.
    llm = LLM(model=MODEL, enable_prefix_caching=True, gpu_memory_utilization=0.85,
              max_model_len=MAX_TOKENS + 2048, seed=seed)
    close_id = tokenizer.convert_tokens_to_ids("</think>")

    prompts = [tokenizer.apply_chat_template(q["prompt"], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True)
               for q in questions]
    full = llm.generate(prompts, SamplingParams(**SAMPLING, n=rollouts, max_tokens=MAX_TOKENS, seed=seed))

    rollout_rows, jobs = [], []
    for q, prompt_text, result in zip(questions, prompts, full):
        for k, output in enumerate(result.outputs):
            ids = list(output.token_ids)
            # Drops <|im_end|>, which would otherwise end the "Final answer:" line;
            # <think> and </think> are not special tokens in Qwen3 and stay.
            text = tokenizer.decode(ids, skip_special_tokens=True)
            closed = close_id in ids
            row = {"id": len(rollout_rows), "source": q["source"], "task": q["task"],
                   "answer": q["answer"], "tokens": len(ids), "closed": closed,
                   "thinking_tokens": ids.index(close_id) + 1 if closed else len(ids),
                   "correct": is_task_correct(text, q["answer"], q["task"]), "text": text}
            rollout_rows.append(row)
            if not closed:
                continue    # no finished thinking to cut: truncated rollouts are reported, not swept
            thinking = text.split("</think>")[0].removeprefix("<think>").strip("\n")
            for fraction in FRACTIONS:
                prefix = cut(thinking, fraction)
                jobs.append({"rollout": row["id"], "fraction": fraction,
                             "kept_tokens": len(tokenizer.encode(prefix)),
                             "prompt": forced_prompt(prompt_text, prefix)})

    forced = llm.generate([j["prompt"] for j in jobs],
                          SamplingParams(**SAMPLING, n=answers, max_tokens=ANSWER_TOKENS, seed=seed))
    forced_rows = []
    for job, result in zip(jobs, forced):
        row = rollout_rows[job["rollout"]]
        verdicts = [is_task_correct("</think>" + o.text, row["answer"], row["task"]) for o in result.outputs]
        forced_rows.append({"rollout": job["rollout"], "source": row["source"], "fraction": job["fraction"],
                            "kept_tokens": job["kept_tokens"], "accuracy": sum(verdicts) / len(verdicts)})

    write_jsonl(out / "rollouts.jsonl", rollout_rows)
    write_jsonl(out / "forced.jsonl", forced_rows)


def summarize(out):
    rollouts = {r["id"]: r for r in read_jsonl(out / "rollouts.jsonl")}
    forced = read_jsonl(out / "forced.jsonl")
    by_rollout = defaultdict(dict)
    for f in forced:
        by_rollout[f["rollout"]][f["fraction"]] = f

    summary = {}
    for source in sorted({r["source"] for r in rollouts.values()}):
        mine = [r for r in rollouts.values() if r["source"] == source]
        closed = [r for r in mine if r["closed"]]
        curve = {}
        for fraction in FRACTIONS:
            points = [by_rollout[r["id"]][fraction]["accuracy"] for r in closed]
            curve[fraction] = sum(points) / len(points) if points else None
        # Earliest fraction from which every later forced answer is right more often
        # than not, for rollouts that were right on their own: where they could have stopped.
        earliest, saved, spent = [], 0, 0
        for r in closed:
            if not r["correct"]:
                continue
            steps = by_rollout[r["id"]]
            # A rollout right on its own can still miss when re-answered after </think>;
            # then nothing is safe to cut and it counts as stopping at the end.
            stop = next((f for f in FRACTIONS
                         if all(steps[g]["accuracy"] >= 0.5 for g in FRACTIONS if g >= f)), 1.0)
            earliest.append(stop)
            saved += r["thinking_tokens"] - steps[stop]["kept_tokens"]
            spent += r["tokens"]
        summary[source] = {
            "rollouts": len(mine),
            "truncated_rate": 1 - len(closed) / len(mine),
            "full_accuracy": sum(r["correct"] for r in mine) / len(mine),
            "forced_accuracy_by_fraction": curve,
            "median_earliest_stop": sorted(earliest)[len(earliest) // 2] if earliest else None,
            "oracle_saving_on_correct": saved / spent if spent else None,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for source, s in summary.items():
        saving = s["oracle_saving_on_correct"]
        print(f"{source:16s} full acc {s['full_accuracy']:.1%}  truncated {s['truncated_rate']:.1%}  "
              f"median stop at {s['median_earliest_stop']}  "
              f"oracle saving {'n/a' if saving is None else f'{saving:.1%}'}")
        print("  forced acc: " + "  ".join(f"{f:.1f}:{a:.0%}" for f, a in
                                           s["forced_accuracy_by_fraction"].items() if a is not None))
    return summary


def plot(out, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    # Categorical slots 1-4 of the validated palette, fixed order; lines are also
    # labelled in the legend, so identity never rests on color alone.
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ink, muted = "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for (source, s), color in zip(summary.items(), colors):
        xs = [f for f, a in s["forced_accuracy_by_fraction"].items() if a is not None]
        ys = [s["forced_accuracy_by_fraction"][f] for f in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, label=source)
        ax.axhline(s["full_accuracy"], color=color, linewidth=0.8, linestyle=":")
    ax.set_xlabel("share of the model's own thinking kept before </think>")
    ax.set_ylabel("accuracy of the forced answer")
    ax.set_title("Where the thinking stops mattering", color=ink, loc="left")
    ax.text(1.0, 0.02, "dotted: accuracy of the full rollouts", transform=ax.transAxes,
            ha="right", color=muted, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e0", linewidth=0.6)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out / "forced_stop_curve.png", dpi=160)
    plt.close(fig)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", default="data/val.jsonl")
    parser.add_argument("--per-source", type=int, default=100, help="Questions per dataset")
    parser.add_argument("--rollouts", type=int, default=4, help="Full rollouts per question")
    parser.add_argument("--answers", type=int, default=2, help="Forced answers per cut")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-only", action="store_true", help="Re-summarize an existing run")
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    if not arguments.plot_only:
        questions = pick_questions(arguments.data, arguments.per_source, arguments.seed)
        generate(arguments.out, questions, arguments.rollouts, arguments.answers, arguments.seed)
    plot(arguments.out, summarize(arguments.out))


if __name__ == "__main__":
    main()
