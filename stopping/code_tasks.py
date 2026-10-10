"""Code tasks for stop-point distillation: data from KodCode, and a sandboxed test runner.

    python stopping/code_tasks.py build --train 1500 --val 100     # writes data/code_train.jsonl, data/code_val.jsonl

Source: KodCode/KodCode-Light-RL-10K (CC BY-NC 4.0; used for non-commercial research, not
redistributed). Each task asks for one Python function and comes with pytest-style unit
tests. Only tasks that

- are labelled `easy` (a 0.6B model needs a fair pass rate to give stop points),
- are not in the `Package` subset (third-party libraries),
- have tests that are plain functions without arguments (no fixtures, no parametrize),
- and whose reference solution passes those tests in this runner

are kept. MBPP, the evaluation's code benchmark, is never used; analysis/overlap_check.py
checks the selection against every evaluation question.

The prompt states the task and the function signature the tests import, and asks for one
Python code block. Rows follow data/train.jsonl, with task "code" and extra fields
`tests` and `entry_point`, so stopping/generate_rollouts.py and find_stop_points.py read
them like the math rows.

Grading (`grade_code`) runs every program in a forked child of a worker process inside
bubblewrap: no network, read-only system, private /tmp, memory and CPU limits, and a
per-item time limit (stopping/code_worker.py), the same isolation the evaluation uses for
MBPP+.
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

WORKER = Path(__file__).with_name("code_worker.py").resolve()
MEMORY_BYTES = 2 * 1024 ** 3
CPU_SECONDS = 600
# Own wording on purpose: no sentence of the evaluation's MBPP+ prompt is reused.
INSTRUCTION = ("Implement it as a Python function with the signature `{signature}`, "
               "and give the complete function in a single fenced Python code block.")
FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


# ---------- answers ----------

def strict_code(response):
    """The code of a reply with exactly one fenced block, or None."""
    blocks = FENCE.findall(response)
    if len(blocks) != 1 or response.count("```") != 2:
        return None
    return blocks[0]


def lenient_code(response):
    """The last fenced block, else the whole reply."""
    blocks = FENCE.findall(response)
    return blocks[-1] if blocks else response


# ---------- sandbox ----------

def sandbox_command():
    environment = Path(sys.prefix).resolve()
    runtime = Path(sys.base_prefix).resolve()
    mounts = [Path("/usr"), Path("/lib"), Path("/lib64"), environment, runtime, WORKER.parent]
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for path in dict.fromkeys(p for p in mounts if p.exists()):
        command += ["--ro-bind", str(path), str(path)]
    command += ["--chdir", "/tmp", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "1",
                "--setenv", "PYTHONHASHSEED", "0", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                "/usr/bin/prlimit", f"--as={MEMORY_BYTES}", f"--cpu={CPU_SECONDS}", "--",
                str(environment / "bin/python"), str(WORKER)]
    return command


def run_chunk(items):
    payload = "".join(json.dumps({"code": c, "tests": t}) + "\n" for c, t in items)
    finished = subprocess.run(sandbox_command(), input=payload, text=True, capture_output=True,
                              timeout=60 + 10 * len(items))
    verdicts = [json.loads(line) for line in finished.stdout.splitlines() if line.strip()]
    if len(verdicts) != len(items):
        raise RuntimeError(f"worker returned {len(verdicts)} of {len(items)}: {finished.stderr[-500:]}")
    return verdicts


def run_programs(items, workers=None, chunk=50):
    """[(code, tests)] -> [{"passed", "error"}], in order, run in parallel sandboxes."""
    if not items:
        return []
    workers = workers or max(1, (os.cpu_count() or 2) - 4)
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    out = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_chunk, c): i for i, c in enumerate(chunks)}
        for future, i in futures.items():
            try:
                out[i] = future.result()
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                out[i] = [{"passed": False, "error": f"sandbox: {error}"[:300]}] * len(chunks[i])
    return [v for chunk_out in out for v in chunk_out]


def grade_code(pairs):
    """[(row, completion text)] -> [(strict, lenient)]: both mean "passes the tests"; strict also
    requires exactly one fenced block after </think> (the shape the evaluation reads best)."""
    jobs, plan = [], []
    for row, text in pairs:
        response = text.split("</think>")[-1] if "</think>" in text else ""
        strict, lenient = strict_code(response), lenient_code(response) if response.strip() else None
        entries = []
        for code in (strict, lenient):
            if code is None:
                entries.append(None)
            elif jobs and jobs[-1] == (code, row["tests"]):
                entries.append(len(jobs) - 1)
            else:
                jobs.append((code, row["tests"]))
                entries.append(len(jobs) - 1)
        plan.append(entries)
    verdicts = run_programs(jobs)
    return [tuple(e is not None and verdicts[e]["passed"] for e in entries) for entries in plan]


# ---------- data ----------

def plain_tests(tests):
    """True when every test_* function takes no arguments and nothing uses fixtures."""
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return False
    found = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found = True
            if node.args.args or node.decorator_list:
                return False
    return found and "fixture" not in tests and "capsys" not in tests and "tmp_path" not in tests


def build(arguments):
    from datasets import load_dataset
    rows = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    candidates = []
    for r in rows:
        if r["gpt_difficulty"] != "easy" or r["subset"] == "Package" or len(r["test_info"]) != 1:
            continue
        if not plain_tests(r["test"]):
            continue
        info = r["test_info"][0]
        signature = info["function_declaration"].strip().removeprefix("def ").rstrip(":").strip()
        prompt = f"{r['question'].strip()}\n\n{INSTRUCTION.format(signature=signature)}"
        candidates.append({"prompt": [{"role": "user", "content": prompt}], "answer": info["function_name"],
                           "task": "code", "source": "kodcode", "kodcode_id": r["question_id"],
                           "subset": r["subset"], "tests": r["test"], "entry_point": info["function_name"],
                           "reference": r["solution"]})
    print(f"{len(candidates)} easy, single-function, plain-test tasks", flush=True)
    random.Random(0).shuffle(candidates)
    pool = candidates[:int((arguments.train + arguments.val) * 1.3)]
    verdicts = run_programs([(c["reference"], c["tests"]) for c in pool])
    passing = [c for c, v in zip(pool, verdicts) if v["passed"]]
    print(f"reference solution passes in this runner: {len(passing)} of {len(pool)}", flush=True)
    for c in passing:
        del c["reference"]
    val, train = passing[:arguments.val], passing[arguments.val:arguments.val + arguments.train]
    Path("data").mkdir(exist_ok=True)
    for name, part in (("code_train", train), ("code_val", val)):
        Path(f"data/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        print(f"data/{name}.jsonl: {len(part)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train", type=int, default=1500)
    b.add_argument("--val", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments)


if __name__ == "__main__":
    main()
