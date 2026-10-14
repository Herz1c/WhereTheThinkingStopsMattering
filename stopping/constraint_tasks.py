"""Instruction-following training tasks with checkable format constraints that IFEval does not use.

    python stopping/constraint_tasks.py --out data/constraint_train.jsonl --count 3000 \
        --questions /path/to/questions/questions.jsonl

Stop-point distillation needs a verifier. For instruction following it is a format constraint
that a program can check. The evaluation's instruction benchmark (IFEval) has 25 constraint
types (keywords, letter frequency, language, word/sentence/paragraph counts, placeholders,
postscripts, bullet counts, highlighted sections, sections, JSON, titles, repeating the prompt,
two responses, all upper/lower case, capital-word counts, end phrase, quotation wrapping, no
commas). Using those would be training on data derived from the evaluation, so this uses five
others:

- no_digits: no digit characters, numbers written as words;
- exclaim: every sentence ends with "!" (no "." or "?" as a sentence end);
- table: the answer is a Markdown table;
- emojis: at least N emoji;
- end_question: the reply ends with a question to the reader.

Base instructions: databricks-dolly-15k (human-written, CC BY-SA 3.0); prompts sharing >= 15% of
an evaluation question's 8-grams are dropped. One or two compatible constraints per prompt.
"""

import argparse
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEXT = {
    "no_digits": "Do not use any digits in your reply; write every number out in words.",
    "exclaim": "End every sentence of your reply with an exclamation mark.",
    "table": "Present your answer as a Markdown table.",
    "emojis": "Use at least {n} emojis in your reply.",
    "end_question": "End your reply with a question for the reader.",
}
CONFLICTS = {frozenset(("exclaim", "end_question")), frozenset(("table", "exclaim"))}
# Wrappers: the whole reply inside an opening and a closing marker. IFEval has one wrapper type
# (double quotation marks), on which the model often fails to close and repeats the quote mark
# until the cap. These three are different markers with the same structure; using them was a
# deliberate grey-zone choice, disclosed in README.md and IDEA.md.
WRAPPERS = {
    "xml_wrap": ("Wrap your entire reply in <reply> and </reply> tags.", "<reply>", "</reply>"),
    "bracket_wrap": ("Wrap your entire reply in square brackets, starting with [ and ending with ].", "[", "]"),
    "fence_wrap": ("Put your entire reply inside a single fenced block that starts and ends with ```.", "```", "```"),
}
WRAP_COMPANIONS = ("no_digits", "emojis", "table")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def check(constraint, reply):
    kind = constraint["kind"]
    if kind in WRAPPERS:
        _, opener, closer = WRAPPERS[kind]
        body = reply.strip()
        if not (body.startswith(opener) and body.endswith(closer) and len(body) > len(opener) + len(closer) + 10):
            return False
        inner = body[len(opener):len(body) - len(closer)]
        return kind == "bracket_wrap" or closer not in inner    # brackets may nest; tags and fences may not
    if kind == "no_digits":
        return not re.search(r"\d", reply)
    if kind == "exclaim":
        body = re.sub(r"https?://\S+", "", reply).strip()
        return body.endswith("!") and not re.search(r"[A-Za-z)\"'][.?](\s|$)", body)
    if kind == "table":
        lines = [l for l in reply.splitlines() if l.strip()]
        rows = [l for l in lines if l.strip().startswith("|")]
        return len(rows) >= 3 and any(TABLE_RULE.match(l) for l in lines)
    if kind == "emojis":
        return len(EMOJI.findall(reply)) >= constraint["n"]
    if kind == "end_question":
        return reply.strip().rstrip("*_ ").endswith("?")
    raise ValueError(kind)


def grade_constraint(row, text):
    """(strict, knows): the reply after </think> satisfies every constraint. No lenient version."""
    if "</think>" not in text:
        return False, False
    reply = text.split("</think>")[-1].replace("<|im_end|>", "").strip()
    ok = len(reply) >= 20 and all(check(c, reply) for c in row["answer"])
    return ok, ok


def build(arguments):
    from datasets import load_dataset
    from analysis.overlap_check import grams, question_text
    evaluation = set()
    for line in Path(arguments.questions).read_text().splitlines():
        evaluation |= grams(question_text(json.loads(line)["messages"]))
    rows = list(load_dataset("databricks/databricks-dolly-15k", split="train"))
    rng = random.Random(arguments.seed)
    rng.shuffle(rows)
    out, dropped = [], 0
    for r in rows:
        if r["category"] in ("classification", "information_extraction") and rng.random() < 0.5:
            continue    # mostly one-word answers, where format constraints are odd
        base = r["instruction"].strip() + (f"\n\n{r['context'].strip()}" if r["context"].strip() else "")
        if len(base) > 1500:
            continue
        g = grams(base)
        if g and len(g & evaluation) / len(g) >= 0.15:
            dropped += 1
            continue
        if arguments.wrappers:
            chosen = [rng.choice(list(WRAPPERS))]
            if rng.random() < 0.4:
                chosen.append(rng.choice(WRAP_COMPANIONS))
        else:
            kinds = list(TEXT)
            first = rng.choice(kinds)
            chosen = [first]
            if rng.random() < 0.4:
                second = rng.choice([k for k in kinds if k != first and frozenset((k, first)) not in CONFLICTS])
                chosen.append(second)
        constraints = [{"kind": k, **({"n": rng.choice((2, 3, 4))} if k == "emojis" else {})} for k in chosen]
        sentences = " ".join((WRAPPERS[c["kind"]][0] if c["kind"] in WRAPPERS else TEXT[c["kind"]]).format(n=c.get("n"))
                             for c in constraints)
        out.append({"source": "dolly_wrappers" if arguments.wrappers else "dolly_constraints",
                    "task": "constraint", "answer": constraints,
                    "prompt": [{"role": "user", "content": f"{base}\n\n{sentences}"}]})
        if len(out) == arguments.count:
            break
    Path(arguments.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{len(out)} constrained Dolly prompts -> {arguments.out} ({dropped} dropped for 8-gram overlap)")
    if arguments.wrappers:
        print("  wrapper prompts; use a different --seed from the constraint set so the Dolly sample differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--questions", default="questions/questions.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wrappers", action="store_true",
                        help="Prompts with one wrapper constraint (and maybe a content constraint) instead")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
