"""How much of the thinking does the answer actually need? Forces </think> part-way.

    python utils/train/prepare_data.py                      # once: writes data/val.jsonl
    python analysis/forced_stop_sweep.py --out runs/sweep    # ~1 h on one consumer GPU
    python analysis/forced_stop_sweep.py --out runs/sweep --reuse-rollouts   # redo the cuts only
    python analysis/forced_stop_sweep.py --out runs/sweep --plot-only

For a sample of validation questions (never the evaluation benchmarks), the base
model writes full rollouts with the evaluation's sampling. Each rollout's thinking
is cut at paragraph boundaries ("\\n\\n") nearest to fixed fractions of its length,
</think> is inserted there, and the model finishes only the answer. This gives
accuracy as a function of how much of its own thinking the model was allowed to keep.

Every answer is graded twice. `accuracy` follows utils/train/rewards.py, which, like
the evaluation, wants "Final answer:" on the last line. `knows` accepts the last
"Final answer:" or \\boxed{} anywhere after </think>: Qwen3-0.6B often writes the
right final answer and then repeats it in a \\boxed{} line, which the strict grader
counts as wrong. When the model stops knowing is a different question from when it
stops formatting, and the stop point is about the first.

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
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.eval.recipe import SAMPLING  # noqa: E402
from utils.train.rewards import (clean_answer, is_task_correct, same_choice,  # noqa: E402
                                 same_number)

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


LAST_FINAL = re.compile(r"Final answer:[ \t]*([^\n]+)")
LAST_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def knows(completion, answer, task):
    """Lenient grade: the last "Final answer:" value, else the last \\boxed{}, anywhere after </think>."""
    response = completion.split("</think>")[-1].replace("**", "")
    finals, boxed = LAST_FINAL.findall(response), LAST_BOXED.findall(response)
    candidate = finals[-1] if finals else boxed[-1] if boxed else None
    if candidate is None:
        return False
    candidate = clean_answer(candidate.strip())
    return same_number(candidate, answer) if task == "math" else same_choice(candidate, answer)


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


def generate(out, questions, rollouts, answers, seed, reuse):
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
    if reuse:
        rollout_rows = read_jsonl(out / "rollouts.jsonl")
        for row in rollout_rows:
            row["question"] = row["id"] // rollouts     # rollouts were written question by question
    else:
        rollout_rows = []
        full = llm.generate(prompts, SamplingParams(**SAMPLING, n=rollouts, max_tokens=MAX_TOKENS, seed=seed))
        for index, (q, result) in enumerate(zip(questions, full)):
            for output in result.outputs:
                ids = list(output.token_ids)
                closed = close_id in ids
                rollout_rows.append({
                    "id": len(rollout_rows), "question": index, "source": q["source"], "task": q["task"],
                    "answer": q["answer"], "tokens": len(ids), "closed": closed,
                    "thinking_tokens": ids.index(close_id) + 1 if closed else len(ids),
                    # Drops <|im_end|>, which would otherwise end the "Final answer:" line;
                    # <think> and </think> are not special tokens in Qwen3 and stay.
                    "text": tokenizer.decode(ids, skip_special_tokens=True)})
    for row in rollout_rows:
        row["correct"] = is_task_correct(row["text"], row["answer"], row["task"])
        row["knows"] = knows(row["text"], row["answer"], row["task"])

    jobs = []
    for row in rollout_rows:
        if not row["closed"]:
            continue    # no finished thinking to cut: truncated rollouts are reported, not swept
        thinking = row["text"].split("</think>")[0].removeprefix("<think>").strip("\n")
        for fraction in FRACTIONS:
            prefix = cut(thinking, fraction)
            jobs.append({"rollout": row["id"], "fraction": fraction,
                         "kept_tokens": len(tokenizer.encode(prefix)),
                         "prompt": forced_prompt(prompts[row["question"]], prefix)})

    forced = llm.generate([j["prompt"] for j in jobs],
                          SamplingParams(**SAMPLING, n=answers, max_tokens=ANSWER_TOKENS, seed=seed))
    forced_rows = []
    for job, result in zip(jobs, forced):
        row = rollout_rows[job["rollout"]]
        texts = ["</think>" + o.text for o in result.outputs]
        strict = [is_task_correct(t, row["answer"], row["task"]) for t in texts]
        lenient = [knows(t, row["answer"], row["task"]) for t in texts]
        forced_rows.append({"rollout": job["rollout"], "source": row["source"], "fraction": job["fraction"],
                            "kept_tokens": job["kept_tokens"], "accuracy": sum(strict) / len(strict),
                            "knows": sum(lenient) / len(lenient), "answers": [o.text for o in result.outputs]})

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
        curves = {}
        for key in ("knows", "accuracy"):
            curve = {}
            for fraction in FRACTIONS:
                points = [by_rollout[r["id"]][fraction][key] for r in closed]
                curve[fraction] = sum(points) / len(points) if points else None
            curves[key] = curve
        # For rollouts that knew the answer on their own: the earliest cut from which
        # every later forced answer knows it more often than not. Where they could have stopped.
        earliest, saved, spent = [], 0, 0
        for r in closed:
            if not r["knows"]:
                continue
            steps = by_rollout[r["id"]]
            # A rollout right on its own can still miss when re-answered after </think>;
            # then nothing is safe to cut and it counts as stopping at the end.
            stop = next((f for f in FRACTIONS
                         if all(steps[g]["knows"] >= 0.5 for g in FRACTIONS if g >= f)), 1.0)
            earliest.append(stop)
            saved += r["thinking_tokens"] - steps[stop]["kept_tokens"]
            spent += r["tokens"]
        summary[source] = {
            "rollouts": len(mine),
            "truncated_rate": 1 - len(closed) / len(mine),
            "closed_knows": sum(r["knows"] for r in closed) / len(closed) if closed else None,
            "closed_accuracy": sum(r["correct"] for r in closed) / len(closed) if closed else None,
            "forced_knows_by_fraction": curves["knows"],
            "forced_accuracy_by_fraction": curves["accuracy"],
            "median_earliest_stop": sorted(earliest)[len(earliest) // 2] if earliest else None,
            "oracle_saving_on_known": saved / spent if spent else None,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for source, s in summary.items():
        saving = s["oracle_saving_on_known"]
        print(f"{source:16s} truncated {s['truncated_rate']:.0%}  finished rollouts: knows {s['closed_knows']:.0%}, "
              f"strict {s['closed_accuracy']:.0%}  median stop at {s['median_earliest_stop']}  "
              f"oracle saving {'n/a' if saving is None else f'{saving:.0%}'}")
        for key in ("knows", "accuracy"):
            print(f"  forced {key:8s}: " + "  ".join(
                f"{f:.1f}:{a:.0%}" for f, a in s[f"forced_{key}_by_fraction"].items() if a is not None))
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
    panels = [("knows", "closed_knows", "Answer known (last Final answer or \\boxed)"),
              ("accuracy", "closed_accuracy", "Graded like the evaluation (strict format)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (key, full_key, title) in zip(axes, panels):
        for (source, s), color in zip(summary.items(), colors):
            curve = s[f"forced_{key}_by_fraction"]
            xs = [f for f, a in curve.items() if a is not None]
            ax.plot(xs, [curve[f] for f in xs], color=color, linewidth=2, marker="o", markersize=4,
                    label=source)
            ax.axhline(s[full_key], color=color, linewidth=0.8, linestyle=":")
        ax.set_title(title, color=ink, loc="left", fontsize=10)
        ax.set_xlabel("share of the model's own thinking kept before </think>")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e6e5e0", linewidth=0.6)
    axes[0].set_ylabel("accuracy of the forced answer")
    axes[1].text(1.0, 0.02, "dotted: the finished rollouts, uncut", transform=axes[1].transAxes,
                 ha="right", color=muted, fontsize=8)
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Where the thinking stops mattering", color=ink, x=0.01, ha="left")
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
    parser.add_argument("--reuse-rollouts", action="store_true",
                        help="Keep <out>/rollouts.jsonl and redo only the forced answers")
    parser.add_argument("--plot-only", action="store_true", help="Re-summarize an existing run")
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    if not arguments.plot_only:
        questions = pick_questions(arguments.data, arguments.per_source, arguments.seed)
        generate(arguments.out, questions, arguments.rollouts, arguments.answers, arguments.seed,
                 arguments.reuse_rollouts)
    plot(arguments.out, summarize(arguments.out))


if __name__ == "__main__":
    main()
