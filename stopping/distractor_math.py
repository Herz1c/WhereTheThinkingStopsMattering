"""Math questions with one irrelevant numeric sentence: where the first number reached is a trap.

    python stopping/distractor_math.py --out data/math_distract.jsonl --questions 1000

Stop-point models learn to close `</think>` at the first answer they reach. On word problems
with irrelevant or changed facts that habit costs accuracy (distraction by irrelevant context:
Shi et al., 2023, "Large Language Models Can Be Easily Distracted by Irrelevant Context").
These questions put that trap into training data built only from GSM8K train:

- a GSM8K training question not used by the earlier math runs (questions are picked after
  skipping the 1,500 that stopping/generate_rollouts.py chose with the same seed);
- one sentence from a *different* GSM8K training question, inserted just before the question
  sentence. It contains a number, is not a question, and shares no capitalised word (names,
  places) with the target, so it cannot be read as part of the problem;
- the gold answer is unchanged, so the usual graders still verify answers automatically.

The stop-point search on these questions places a stop only where forced answers are right,
i.e. after the model has dealt with the distractor; rollouts that used it are wrong and give no
example. Rows follow data/train.jsonl (source "gsm8k_distract").
"""

import argparse
import json
from pathlib import Path
import random
import re

SENTENCE = re.compile(r"(?<=[.!?])\s+")
NUMBER = re.compile(r"\d")
CAPITAL = re.compile(r"\b[A-Z][a-z]+\b")
INSTRUCTION = "\n\nPut the final answer"


NOT_A_NAME = {"He", "She", "They", "His", "Her", "Their", "It", "Its", "This", "That", "These", "Those", "The",
              "A", "An", "Each", "Every", "If", "There", "In", "On", "At", "After", "Then", "So", "But", "And",
              "When", "Of", "For", "With", "All", "Both", "One", "Two", "Some", "Most", "Half", "We", "You", "I",
              "Before", "During", "While", "Since", "Because", "Although", "Last", "Yesterday", "Today", "Once",
              "Now", "Next", "First", "Later", "Over", "Out", "Every", "Tomorrow", "Instead", "Also"}
BACK_REFERENCE = re.compile(r"\b(?:than that|that much|this|these|those|the same|the rest|the other|remaining)\b",
                            re.IGNORECASE)


def self_contained(sentence):
    """A sentence that stands on its own: it starts with a name and does not point back to
    something earlier in its original problem ("His lunch has 25% more calories than that")."""
    first = sentence.split()[0].strip(",.'")
    return (CAPITAL.fullmatch(first) is not None and first not in NOT_A_NAME
            and not BACK_REFERENCE.search(sentence))


def split_question(text):
    """GSM8K prompt text -> (problem sentences, instruction suffix)."""
    body, _, rest = text.partition(INSTRUCTION)
    return [s for s in SENTENCE.split(body.strip()) if s], INSTRUCTION + rest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=1000)
    parser.add_argument("--skip", type=int, default=1500, help="Questions already used, in the same seeded order")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    rows = []
    for index, line in enumerate(Path(arguments.data).read_text().splitlines()):
        row = json.loads(line)
        if row["source"] == "gsm8k":
            rows.append({"question_id": index, **row})
    # Same order generate_rollouts.py uses (random.Random(seed).shuffle), so the first --skip
    # questions are exactly those already used.
    random.Random(arguments.seed).shuffle(rows)
    targets, donors = rows[arguments.skip:arguments.skip + arguments.questions], rows[arguments.skip + arguments.questions:]

    pool = []
    for donor in donors:
        sentences, _ = split_question(donor["prompt"][0]["content"])
        for s in sentences[:-1]:            # never the donor's own question sentence
            if NUMBER.search(s) and not s.rstrip().endswith("?") and 6 <= len(s.split()) <= 25 \
                    and self_contained(s):
                pool.append(s.strip())
    rng = random.Random(arguments.seed + 1)
    out = []
    for target in targets:
        sentences, suffix = split_question(target["prompt"][0]["content"])
        if len(sentences) < 2:
            continue
        names = set(CAPITAL.findall(" ".join(sentences)))
        for _ in range(50):
            distractor = rng.choice(pool)
            if not (set(CAPITAL.findall(distractor)) - {"The", "A", "An", "He", "She", "They", "It", "In", "On",
                                                         "Each", "Every", "If", "There"}) & names:
                break
        else:
            continue
        text = " ".join(sentences[:-1] + [distractor, sentences[-1]]) + suffix
        out.append({"prompt": [{"role": "user", "content": text}], "answer": target["answer"], "task": "math",
                    "source": "gsm8k_distract", "original_question_id": target["question_id"],
                    "distractor": distractor})
    arguments.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"wrote {arguments.out}: {len(out)} questions (pool of {len(pool)} distractor sentences)")


if __name__ == "__main__":
    main()
