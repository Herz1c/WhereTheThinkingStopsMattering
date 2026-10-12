# Where the Thinking Stops Mattering

Small reasoning models often reach the right answer and then keep writing: re-checking a sum
they already did, planning a function they never write out, repeating a closing quote until the
token limit. This project teaches Qwen3-0.6B to stop where its own thinking stops mattering, with
the change in the weights only: same prompts, same sampling, same token caps, thinking still on.

**Final result: 43.5% fewer generated tokens at −0.77 pp accuracy**, measured on 22,072 responses
over four benchmarks (code, function calling, instruction following, arithmetic), with 95%
paired-bootstrap intervals [42.2%, 44.9%] and [−1.49, −0.10] pp.

Independent student project, October 2026, on one RTX 5060 Ti (16 GB). Negative results are kept
below with the positive ones: most of what was tried did not work, and why it did not is most of
what was learned.

<p align="center">
  <img src="figures/demo_mbpp.svg" alt="The same MBPP+ task answered by base Qwen3-0.6B and by the trained model, streamed side by side at the same speed" width="900">
</p>

<sub>Real outputs from the evaluation run, the same task and sample slot for both models, both
answers correct, streamed at the same number of tokens per second. The example is the pair closest
to the benchmark's median length ratio among long base answers, not a hand-picked extreme
(`analysis/demo_svg.py`). A math example: [figures/demo_svamp.svg](figures/demo_svamp.svg).</sub>

## Result

| Benchmark | What it tests | Tokens saved | Accuracy change |
| --- | --- | ---: | ---: |
| MBPP+ | writing a Python function | **65.0%** [62.9, 67.2] | −0.23 pp [−1.92, +1.49] |
| IFEval | following format instructions | **39.0%** [35.1, 42.8] | −0.58 pp [−2.22, +1.16] |
| BFCL (3 subsets) | function calling | **26–35%** | −0.44 to −1.31 pp (all intervals cover 0) |
| SVAMP | grade-school arithmetic | **30.9%** [28.5, 33.1] | −1.21 pp [−2.00, −0.44] |
| **Average** | | **43.5%** [42.2, 44.9] | **−0.77 pp** [−1.49, −0.10] |

Tokens saved is the evaluation's own metric: per benchmark one minus the ratio of summed tokens
(thinking included, a response cut off at the cap counts in full and as wrong), then the geometric
mean over benchmarks. Accuracy change is the mean over benchmarks. 8 samples per question.

<p align="center"><img src="figures/benchmark_bars.png" alt="Mean tokens and accuracy per benchmark, base vs trained" width="760"></p>

What the numbers do not show: SVAMP is the one benchmark whose accuracy interval excludes zero.
Every design decision was made while looking at these four benchmarks (never training on them),
so a held-out check on HumanEval and GSM8K test is being run: [Held-out check](#held-out-check).

## What I built

The method has two stages, both trained as a LoRA (r = 16) on the model's own outputs, never on
evaluation data:

1. **Stop-point distillation** (supervised). For each training task, sample the base model's own
   reasoning, force `</think>` at paragraph boundaries, and find the earliest cut where the forced
   answer is verifiably right (unit tests, the gold number, the right function call, a checked
   format constraint). Train on *own reasoning up to that cut + `</think>` + answer*. Three rules
   were needed to keep accuracy, each found by reading failures:
   - **Code: stop only after the function is written out in the reasoning.** Oracle stops let the
     model stop after a one-paragraph plan and write untried code into the answer (MBPP −5.4 pp);
     requiring a drafted `def` before the stop brought MBPP back.
   - **Give up with a best guess after a long unproductive attempt.** Attempts that run to the cap
     are wrong anyway and hold nearly a third of the base model's code tokens; cutting them at 2,048
     thinking tokens with the model's own best answer cut capped MBPP responses from 49 to 6 per
     756, with no accuracy cost.
   - **Instruction following needs its own verifiable data.** Dolly instructions with format
     constraints of five types IFEval does not use (no digits, Markdown table, emoji, …) halved
     IFEval's reasoning length (296 → 134 median tokens) with no accuracy loss.
2. **GRPO** from that adapter, with reward `1[correct] · (1 − α · length / domain mean length)`:
   the length penalty is normalised per domain because the metric is a ratio of sums, and a wrong
   or capped answer earns 0 whatever its length. Supervised stops cannot teach *when* re-checking
   pays off; on arithmetic the model's checks sometimes catch an error and sometimes are waste.
   RL learned to keep the useful ones: SVAMP's saving went from 14% to 31% at the same accuracy.

Training data: GSM8K train, KodCode-Light-RL-10K, Glaive function calling v2, databricks-dolly-15k
(with own constraints). Every source was checked for 8-gram overlap with all evaluation questions;
11 overlapping function-calling rows were removed.

## Along the way

<p align="center"><img src="figures/final_curve.png" alt="Tokens saved vs accuracy change for four checkpoints, full evaluation" width="640"></p>

| Checkpoint (full evaluation) | Tokens saved | Accuracy change |
| --- | ---: | ---: |
| Shortest correct own answer, SFT (reference baseline; quick eval) | 2.3% | −0.53 pp |
| Mix C: oracle stops on math, code, tools | 37.2% | −1.80 pp |
| Mix E: + code drafts, give-up | 29.4% | −0.45 pp (all intervals cover 0) |
| Mix G: + instruction stops | 34.6% | −0.85 pp |
| **GRPO from mix G** | **43.5%** | **−0.77 pp** |

What did not work, each on the evaluation's quick 2-sample run unless noted:

| Idea | Result | Why |
| --- | --- | --- |
| DPO on "later correct vs earlier wrong" stop pairs | learned stopping collapsed (MBPP 762 → 1,769 tokens) | preference pairs undid the stop instead of sharpening it |
| Stop only after the model "checked" its answer (cue words) | no accuracy gain, fewer savings | the cues fired on the model restating the task's test case |
| Math stops only after the answer is computed | SVAMP −6.1 pp | 86% of stops already were; the re-checks after the answer are what keep it right |
| Give up on math after 1,024 tokens | SVAMP saving ×2 at −1.5 pp | long correct attempts get cut before they know |
| Scaling the adapter's strength (×0.75, ×0.5) | the stop vanishes entirely | a learned stop is a switch, not a dial |
| Math with irrelevant distractor sentences | SVAMP still −5.2 pp | the loss came from the math stop data itself, not from distraction |
| Cutting looping replies (Dolly, own wrapper formats) | 7 usable examples | plain instructions rarely loop; wrapper loops start at the first token, leaving nothing to keep |

## Held-out check

Pending: HumanEval (164 tasks × 4) and GSM8K test (500 × 2) for base, mix G and GRPO, sets that
played no part in any decision (`analysis/heldout_eval.py`).

## How the repository is laid out

| Path | Contents |
| --- | --- |
| `stopping/` | the method: rollouts, stop search, verified/draft stops, give-up and constraint data, LoRA and GRPO trainers, graders and the code sandbox |
| `analysis/` | measurements and figures: forced-stop sweep, overlap check, held-out check, plots, demo |
| `scripts/` | resumable full evaluation, subsampled baselines, adapter scaling |
| `evaluate.py`, `compare.py`, `utils/` | the evaluation harness, unchanged from the original test |
| `figures/` | everything shown here |

## Reproduce

```bash
python utils/eval/benchmarks.py                                   # freeze the evaluation questions
python scripts/resumable_eval.py --out runs/base                  # baseline (~3 h on a 5060 Ti)

# stop-point data per domain (math shown; code, tools and constraints the same way)
python stopping/generate_rollouts.py --out runs/rollouts_gsm8k --sources gsm8k --questions 1500
python stopping/find_stop_points.py runs/rollouts_gsm8k --out runs/stop_points
python stopping/verified_examples.py runs/rollouts_code runs/stop_points_code --mode draft --out runs/verified_code_draft
python stopping/give_up_examples.py runs/rollouts_code --out runs/give_up_code --budget 2048

# mix G = draft code stops (one per question) + give-up rows + tool and math rows + constraint stops, concatenated
python stopping/train_lora.py --data runs/train_mixG.jsonl --out checkpoints/stoppoint_mixG
python stopping/train_grpo.py --init checkpoints/stoppoint_mixG --out checkpoints/grpo --steps 50 --prompts 24
python scripts/resumable_eval.py --out runs/grpo_full --adapter checkpoints/grpo
python compare.py runs/base runs/grpo_full
```

Compute for everything in this repository, including all failed variants: about 65 GPU-hours on
one RTX 5060 Ti.

## Evidence ladder

| Level | Claim | Support |
| --- | --- | --- |
| A | 43.5% fewer tokens at −0.77 pp on the four benchmarks | full evaluation, 22,072 responses, paired bootstrap |
| B | each of the three stop rules is needed | ablations on the quick evaluation (2 samples), one run each |
| B | GRPO's gain comes from selective re-checking on math | SVAMP saving ×2 at unchanged accuracy; training-pool logs |
| C | the result holds off the four benchmarks | held-out check (pending) |
| Missing | the method scales to larger models | next: Qwen3-1.7B |

## Origin and license

This project started as a response to the
[ThinkingCap Test](https://github.com/BottleCapAI/ThinkingCap-Test) by BottleCapAI.
The evaluation harness (`evaluate.py`, `compare.py`, `utils/`), `train.py` and the original task description
([CHALLENGE.md](CHALLENGE.md)) come from that repository and are licensed under Apache-2.0 (see [LICENSE](LICENSE)).
The benchmarks and training datasets belong to their authors; see the sources table in
[CHALLENGE.md](CHALLENGE.md#sources-and-licenses).

Independent research.
