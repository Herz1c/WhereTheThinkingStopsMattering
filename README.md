# Where the Thinking Stops Mattering

Small reasoning models often reach the right answer and then keep writing.
This project measures how much of Qwen3-0.6B's reasoning comes *after* the answer is already settled,
and tries to teach the model to put `</think>` there itself, so that the shortening lives in the weights
rather than in a prompt or a decoding trick.

> **Status:** work in progress. Results, including negative ones, will be added as they come in.

## Question

At which point in a reasoning trace does further thinking stop changing the answer,
and can a 0.6B model learn to recognise that point across very different tasks:
arithmetic, code, function calling and instruction following?

## Approach (planned)

1. **Measure overthinking.** Force `</think>` at every paragraph boundary of the base model's own traces
   and record when the forced answer becomes as good as the full one, per domain.
2. **Stop-point self-distillation.** Fine-tune the model to emit `</think>` at the earliest point where its
   own answer is already correct, keeping the content of its reasoning unchanged.
3. **Counterfactual step pruning.** Remove reasoning steps whose omission barely changes the probability
   of the correct answer.
4. **Cross-domain transfer.** Train on one domain at a time and measure savings on all four.
5. **RL fine-tuning** with a length-aware reward that only pays for correct answers.

## Evaluation

Four benchmarks, each answered 8 times per question (22,072 responses per run):
MBPP+ (code), BFCL (function calling), SVAMP (arithmetic) and IFEval (instruction following).
Every generated token counts, thinking included; an answer cut off by the token cap counts as wrong.

- **Tokens saved:** geometric mean over the four benchmarks of the per-benchmark token ratio.
- **Accuracy change:** mean over the four benchmarks, in percentage points.

Both are reported with 95% confidence intervals from a paired bootstrap over questions.

```bash
python utils/eval/benchmarks.py                        # download and freeze the questions
python evaluate.py --out runs/base                     # baseline
python evaluate.py --out runs/mine --adapter ./lora    # a checkpoint
python compare.py runs/base runs/mine
```

## Results

| Checkpoint | Tokens saved | Accuracy change |
| --- | --- | --- |
| — | — | — |

## Origin and license

This project started as a response to the
[ThinkingCap Test](https://github.com/BottleCapAI/ThinkingCap-Test) by BottleCapAI.
The evaluation harness (`evaluate.py`, `compare.py`, `utils/`), `train.py` and the original task description
([CHALLENGE.md](CHALLENGE.md)) come from that repository and are licensed under Apache-2.0 (see [LICENSE](LICENSE)).
The benchmarks belong to their authors; see the sources table in [CHALLENGE.md](CHALLENGE.md#sources-and-licenses).

Independent research.
