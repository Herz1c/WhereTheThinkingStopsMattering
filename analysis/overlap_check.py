"""Does any training question overlap an evaluation question?

    python analysis/overlap_check.py --train data/train.jsonl --sources gsm8k

The rules forbid training on the evaluation benchmarks or data derived from them. This
compares every training question from the given sources with every evaluation question
in questions/questions.jsonl by word 8-gram overlap after normalisation (lower case,
numbers kept, punctuation dropped). A pair is reported when the share of the evaluation
question's 8-grams found in the training question passes --threshold. Exact duplicates
score 1.0; rewordings with a shared long span score high; independent problems on the
same topic score near 0.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

N = 8


def words(text):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def grams(text):
    w = words(text)
    return {tuple(w[i:i + N]) for i in range(len(w) - N + 1)}


def question_text(messages):
    """The user turn without the answer-format instruction the prompts append."""
    text = next(m["content"] for m in messages if m["role"] == "user")
    # Instructions are shared by every question of a source, so they would match each other:
    # drop the evaluation's MBPP+ preamble and the answer-format lines the prompts append.
    text = re.sub(r"^Write a complete Python solution[^\n]*\n\n", "", text)
    return re.split(r"\n\n(?:Put the final answer|Answer with the label|The last line of your reply"
                    r"|Implement it as a Python function)", text)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--sources", default="gsm8k")
    parser.add_argument("--questions", default="questions/questions.jsonl")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--write-clean", type=Path,
                        help="Also write the training file without every row in a reported pair")
    arguments = parser.parse_args()

    sources = set(arguments.sources.split(","))
    all_rows = [json.loads(l) for l in Path(arguments.train).read_text().splitlines()]
    train = [(i, question_text(r["prompt"])) for i, r in enumerate(all_rows) if r["source"] in sources]
    index = defaultdict(set)
    train_grams = {}
    for i, text in train:
        train_grams[i] = grams(text)
        for g in train_grams[i]:
            index[g].add(i)

    evaluation = [json.loads(l) for l in Path(arguments.questions).read_text().splitlines()]
    hits, checked = [], defaultdict(int)
    for q in evaluation:
        g = grams(question_text(q["messages"]))
        checked[q["dataset"]] += 1
        if not g:
            continue
        counts = defaultdict(int)
        for gram in g:
            for i in index.get(gram, ()):
                counts[i] += 1
        for i, c in counts.items():
            if c / len(g) >= arguments.threshold:
                hits.append((c / len(g), q["dataset"], q["id"], i))

    print(f"{len(train)} training questions from {sorted(sources)} vs "
          f"{sum(checked.values())} evaluation questions ({dict(checked)})")
    print(f"{len(hits)} pairs with >= {arguments.threshold:.0%} of the evaluation question's {N}-grams shared")
    texts = dict(train)
    for share, dataset, qid, i in sorted(hits, reverse=True)[:10]:
        print(f"\n{share:.0%}  {dataset} {qid}  vs  train line {i}")
        q = next(e for e in evaluation if e["id"] == qid and e["dataset"] == dataset)
        print("  eval :", question_text(q["messages"])[:200].replace("\n", " "))
        print("  train:", texts[i][:200].replace("\n", " "))
    if arguments.write_clean:
        flagged = {i for *_, i in hits}
        kept = [r for i, r in enumerate(all_rows) if i not in flagged]
        arguments.write_clean.write_text("".join(json.dumps(r) + "\n" for r in kept))
        print(f"\nwrote {arguments.write_clean}: {len(kept)} rows ({len(flagged)} removed)")


if __name__ == "__main__":
    main()
