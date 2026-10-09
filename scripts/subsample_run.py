"""A view of a finished run with only its first N samples per question, for quick comparisons.

    python scripts/subsample_run.py runs/base runs/base_s2 --samples 2
    python evaluate.py --out runs/mine_s2 --adapter ./lora --samples 2
    python compare.py runs/base_s2 runs/mine_s2

compare.py pairs runs only when their sample counts match. A quick evaluation with
--samples 2 asks every question with the same seeds as samples 0 and 1 of the full
run (seeds come from the question and the sample index), so the first two samples of
the full baseline are its natural partner. Nothing is regenerated: this copies the
matching records and rewrites `samples` in results.json. The intervals are wider,
as they should be; the result is a development signal, not a leaderboard row.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--samples", type=int, required=True)
    arguments = parser.parse_args()

    meta = json.loads((arguments.run / "results.json").read_text())
    if arguments.samples >= meta["samples"]:
        raise SystemExit(f"{arguments.run} has {meta['samples']} samples; nothing to drop")
    rows = [line for line in (arguments.run / "records.jsonl").read_text().splitlines()
            if json.loads(line)["sample_index"] < arguments.samples]
    arguments.out.mkdir(parents=True, exist_ok=False)
    (arguments.out / "records.jsonl").write_text("\n".join(rows) + "\n")
    meta["samples"] = arguments.samples
    meta["subsampled_from"] = str(arguments.run.resolve())
    (arguments.out / "results.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(rows)} records to {arguments.out}")


if __name__ == "__main__":
    main()
