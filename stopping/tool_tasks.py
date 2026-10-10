"""Function-calling tasks for stop-point distillation, from Glaive function-calling v2.

    python stopping/tool_tasks.py build --train 1500 --val 100   # writes data/tool_train.jsonl, data/tool_val.jsonl

Source: glaiveai/glaive-function-calling-v2 (Apache-2.0). Each conversation offers one or
more functions as JSON schemas. The first user turn and the first assistant turn give a
task with a checkable answer:

- the assistant calls a function (`<functioncall> {"name": ..., "arguments": ...}`): the
  target is that call, with those arguments;
- the assistant makes no call (the request does not fit, or needs details the user has not
  given): the target is *no call*.

About 30% of kept tasks are no-call, close to the evaluation's BFCL mix (240 of 840).
BFCL itself is never used; analysis/overlap_check.py checks the selection against every
evaluation question.

The system prompt is our own wording and asks for the same answer syntax the evaluation
reads, a Python list of calls with keyword arguments: [func(a=1, b="x")].

Grading parses the reply with utils/eval/graders.decode_calls (ast parsing and literal_eval
only; nothing is executed). A call is right when the names match and every argument matches
the reference (numbers by value, strings ignoring case and surrounding space). No-call
follows BFCL's rule: anything that is not a parseable call is correct. *knows* also accepts a
reply wrapped in a code fence; *strict* asks for the bare list the evaluation expects.
"""

import argparse
import json
import keyword
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.eval.graders import decode_calls  # noqa: E402

SYSTEM = ("You can use the functions listed below. Read the user's request. If one or more of the "
          "functions fulfil it, reply with only the call(s), as a Python list with keyword arguments: "
          "[function_name(parameter=value, ...)]. Write nothing else. If none of the functions fits the "
          "request, or a required value is missing, do not call anything and say why.\n\n"
          "Functions (JSON):\n{functions}")
CALL = re.compile(r"<functioncall>\s*(\{.*\})", re.DOTALL)
QUOTED_CALL = re.compile(r'"name":\s*"([^"]+)",\s*"arguments":\s*\'(.*)\'\s*\}\s*$', re.DOTALL)


# ---------- grading ----------

def same_value(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return a == b


def matches(reply, target):
    """target: None for no call, else {"name": ..., "arguments": {...}}."""
    try:
        calls = decode_calls(reply)
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return target is None
    if target is None:
        return calls == []
    if len(calls) != 1:
        return False
    (name, arguments), = calls[0].items()
    return name == target["name"] and same_value(arguments, target["arguments"])


def grade_tool(row, text):
    """-> (strict, knows)."""
    if "</think>" not in text:
        return False, False
    response = text.split("</think>")[-1].strip()
    target = json.loads(row["answer"])
    knows = matches(response, target)
    strict = knows and not response.startswith("```")
    return strict, knows


# ---------- data ----------

def parse(sample):
    """A Glaive sample -> (functions, user turn, target) or None."""
    system = sample["system"].removeprefix("SYSTEM:").strip()
    start = system.find("{")
    if start < 0:
        return None
    functions, decoder, rest = [], json.JSONDecoder(), system[start:]
    while rest:
        try:
            obj, end = decoder.raw_decode(rest)
        except json.JSONDecodeError:
            return None
        functions.append(obj)
        rest = rest[end:].lstrip().lstrip(",").lstrip()
        if rest and not rest.startswith("{"):
            break
    if not functions or not all(isinstance(f, dict) and "name" in f for f in functions):
        return None
    turns = re.split(r"\n*(USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*", sample["chat"])
    pairs = [(turns[i], turns[i + 1].replace("<|endoftext|>", "").strip()) for i in range(1, len(turns) - 1, 2)]
    if len(pairs) < 2 or pairs[0][0] != "USER:" or pairs[1][0] != "ASSISTANT:":
        return None
    user, reply = pairs[0][1], pairs[1][1]
    call = CALL.search(reply)
    if call:
        # Glaive writes the arguments as a single-quoted JSON string inside the call object,
        # which is not valid JSON as a whole: read the name and the argument string apart.
        quoted = QUOTED_CALL.search(call.group(1))
        try:
            if quoted:
                name, arguments = quoted.group(1), json.loads(quoted.group(2))
            else:
                obj = json.loads(call.group(1))
                name, arguments = obj.get("name"), obj.get("arguments", {})
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(arguments, dict) or name not in {f["name"] for f in functions}:
            return None
        # A parameter called e.g. `from` cannot be written as a keyword argument in the answer
        # syntax ([f(from="USD")] does not parse), so no reply could be graded right.
        if not name.isidentifier() or any(not k.isidentifier() or keyword.iskeyword(k) for k in arguments):
            return None
        target = {"name": name, "arguments": arguments}
    else:
        target = None
    return functions, user, target


def build(arguments):
    from datasets import load_dataset
    rows = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    calls, none = [], []
    for sample in rows:
        parsed = parse(sample)
        if parsed is None:
            continue
        functions, user, target = parsed
        if not user or len(user) > 1500:
            continue
        row = {"prompt": [{"role": "system", "content": SYSTEM.format(functions=json.dumps(functions, indent=4))},
                          {"role": "user", "content": user}],
               "answer": json.dumps(target), "task": "tool", "source": "glaive"}
        (calls if target else none).append(row)
    print(f"{len(calls)} call tasks, {len(none)} no-call tasks", flush=True)
    rng = random.Random(0)
    rng.shuffle(calls)
    rng.shuffle(none)
    total = arguments.train + arguments.val
    n_none = round(total * 0.3)
    picked = calls[:total - n_none] + none[:n_none]
    rng.shuffle(picked)
    # Sanity: every reference answer, written in the answer syntax, must grade as right.
    for row in picked:
        target = json.loads(row["answer"])
        reply = "[]" if target is None else \
            f"[{target['name']}(" + ", ".join(f"{k}={v!r}" for k, v in target["arguments"].items()) + ")]"
        assert grade_tool(row, "</think>\n\n" + reply) == (True, True), (reply, row["answer"])
    val, train = picked[:arguments.val], picked[arguments.val:]
    Path("data").mkdir(exist_ok=True)
    for name, part in (("tool_train", train), ("tool_val", val)):
        Path(f"data/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        print(f"data/{name}.jsonl: {len(part)} rows ({sum(json.loads(r['answer']) is None for r in part)} no-call)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train", type=int, default=1500)
    b.add_argument("--val", type=int, default=100)
    arguments = parser.parse_args()
    build(arguments)


if __name__ == "__main__":
    main()
