"""Animated side-by-side demo for the README: the same question answered by two models, streamed at
the same speed, from real evaluation records.

    python analysis/demo_svg.py --base runs/base --mine runs/grpo_full --dataset svamp \
        --out figures/demo_svamp.svg

Picks a question both models answered correctly in the same sample slot, where the base answer
is long and this model's is short (typical, not the most extreme: the pair closest to the
benchmark's median token ratio among long base answers). The text is the models' own output,
thinking included, scrolled at a fixed number of tokens per second, so each pane runs for a time
proportional to its token count. Pure SVG with CSS animation (GitHub renders it in an <img>;
no script).
"""

import argparse
from html import escape
import json
from pathlib import Path
import statistics
import textwrap

WIDTH, PANE, GAP, MARGIN = 900, 430, 20, 10
WINDOW, LINE, FONT = 300, 14, 11
WRAP = 66
TOKENS_PER_SECOND = 90
PAUSE = 3.0


def load(run, dataset):
    out = {}
    for line in (Path(run) / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["dataset"] == dataset:
            out[(r["id"], r["sample_index"])] = r
    return out


def pick(base, mine, low, high):
    pairs = [(k, base[k], mine[k]) for k in base if k in mine and base[k]["correct"] and mine[k]["correct"]
             and low <= base[k]["output_tokens"] <= high and not base[k]["truncated"]]
    ratios = sorted(m["output_tokens"] / b["output_tokens"] for _, b, m in pairs)
    target = statistics.median(ratios)
    return min(pairs, key=lambda p: abs(p[2]["output_tokens"] / p[1]["output_tokens"] - target))


def lines_of(text):
    text = text.replace("<|im_end|>", "").strip()
    out = []
    for paragraph in text.split("\n"):
        out += textwrap.wrap(paragraph, WRAP) or [""]
    return out


def pane(x, title, lines, tokens, seconds, cycle, name, note):
    end = seconds / cycle * 100
    start_y = WINDOW - LINE
    end_y = WINDOW - LINE * len(lines)
    rows = "".join(f'<text x="8" y="{i * LINE + FONT}" class="t" xml:space="preserve">{escape(l)}</text>'
                   for i, l in enumerate(lines))
    return f"""
  <style>
    @keyframes scroll_{name} {{ 0% {{ transform: translateY({start_y}px) }} {end:.2f}% {{ transform: translateY({end_y}px) }}
      100% {{ transform: translateY({end_y}px) }} }}
    @keyframes bar_{name} {{ 0% {{ width: 0px }} {end:.2f}% {{ width: {tokens / BASE_TOKENS * PANE:.1f}px }}
      100% {{ width: {tokens / BASE_TOKENS * PANE:.1f}px }} }}
    @keyframes done_{name} {{ 0%, {max(0, end - 0.01):.2f}% {{ opacity: 0 }} {end:.2f}%, 100% {{ opacity: 1 }} }}
    .s_{name} {{ animation: scroll_{name} {cycle:.2f}s linear infinite }}
    .b_{name} {{ animation: bar_{name} {cycle:.2f}s linear infinite }}
    .d_{name} {{ animation: done_{name} {cycle:.2f}s step-end infinite }}
  </style>
  <text x="{x}" y="46" class="h">{escape(title)}</text>
  <rect x="{x}" y="56" width="{PANE}" height="{WINDOW}" rx="6" class="box"/>
  <svg x="{x}" y="56" width="{PANE}" height="{WINDOW}" overflow="hidden">
    <g class="s_{name}">{rows}</g>
  </svg>
  <rect x="{x}" y="{56 + WINDOW + 12}" width="{PANE}" height="8" rx="4" class="track"/>
  <rect x="{x}" y="{56 + WINDOW + 12}" height="8" rx="4" class="bar {name} b_{name}"/>
  <text x="{x}" y="{56 + WINDOW + 38}" class="n d_{name}">done: {tokens:,} tokens {escape(note)}</text>"""


BASE_TOKENS = 1


def main():
    global BASE_TOKENS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--mine", required=True)
    parser.add_argument("--dataset", default="svamp")
    parser.add_argument("--low", type=int, default=700)
    parser.add_argument("--high", type=int, default=1400)
    parser.add_argument("--label", default="this work (GRPO)")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    key, b, m = pick(load(arguments.base, arguments.dataset), load(arguments.mine, arguments.dataset),
                     arguments.low, arguments.high)
    BASE_TOKENS = b["output_tokens"]
    seconds_b = b["output_tokens"] / TOKENS_PER_SECOND
    seconds_m = m["output_tokens"] / TOKENS_PER_SECOND
    cycle = seconds_b + PAUSE
    saved = 1 - m["output_tokens"] / b["output_tokens"]
    height = 56 + WINDOW + 52
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">
  <style>
    .t {{ font: {FONT}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; fill: #c9d1d9 }}
    .h {{ font: 600 14px -apple-system, Segoe UI, Helvetica, Arial, sans-serif; fill: #24292f }}
    .c {{ font: 12px -apple-system, Segoe UI, Helvetica, Arial, sans-serif; fill: #57606a }}
    .n {{ font: 600 12px -apple-system, Segoe UI, Helvetica, Arial, sans-serif; fill: #1a7f37 }}
    .box {{ fill: #0d1117 }}
    .track {{ fill: #d0d7de }}
    .bar.base {{ fill: #8c959f }}
    .bar.mine {{ fill: #1a7f37 }}
  </style>
  <rect width="{WIDTH}" height="{height}" fill="#ffffff"/>
  <text x="{MARGIN}" y="20" class="c">{escape(arguments.dataset.upper())} {escape(str(key[0]))}: same question, same sampling, both answers correct, streamed at the same speed. Real outputs from the evaluation.</text>
{pane(MARGIN, "Qwen3-0.6B (base)", lines_of(b["text"]), b["output_tokens"], seconds_b, cycle, "base", "")}
{pane(MARGIN + PANE + GAP, f"Qwen3-0.6B + {arguments.label}", lines_of(m["text"]), m["output_tokens"], seconds_m, cycle, "mine", f"({saved:.0%} fewer)")}
</svg>
"""
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(svg, encoding="utf-8")
    print(f"{arguments.out}: {key}, base {b['output_tokens']} tokens, mine {m['output_tokens']} tokens, "
          f"{saved:.0%} fewer, cycle {cycle:.1f} s")


if __name__ == "__main__":
    main()
