"""DPO on stop decisions, starting from a stop-point LoRA.

    python stopping/train_dpo.py --pairs runs/dpo_pairs.jsonl --init checkpoints/stoppoint_all3_cont \
        --out checkpoints/stoppoint_dpo

Direct Preference Optimization (Rafailov et al., 2023) with the starting adapter as the frozen
reference: the loss is -log sigmoid(beta * [(log p(chosen) - log p_ref(chosen)) -
(log p(rejected) - log p_ref(rejected))]), plus alpha times the per-token negative
log-likelihood of the chosen completion, which keeps the policy on its own format (the usual
stabiliser, as in RPO). Chosen and rejected share the reasoning up to the earlier boundary, so
their shared tokens cancel in the preference term: what is trained is the decision at that
boundary (keep thinking vs close and answer) and what follows.

Reference log-probabilities are computed once, before training, with the starting adapter. The
same adapter then continues training, so the result loads like any other LoRA here (vLLM-
compatible targets, safetensors). Memory: one sequence at a time, gradient checkpointing,
log-probabilities computed over the completion in chunks without materialising full logits.
"""

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen3-0.6B"
CHUNK = 1024


def prompt_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    return list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)


def completion_logprobs(model, prompt, completion):
    """(sum of log p over the completion tokens, mean NLL over them), with graph if grad is on."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    ids = torch.tensor([prompt + completion], device="cuda")
    hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    states = hidden[len(prompt) - 1:-1]
    targets = ids[0, len(prompt):]
    total = states.new_zeros((), dtype=torch.float32)
    for start in range(0, states.shape[0], CHUNK):
        logits = base.lm_head(states[start:start + CHUNK]).float()
        total = total - F.cross_entropy(logits, targets[start:start + CHUNK], reduction="sum")
    return total, -total / targets.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True, help="Starting LoRA (also the reference)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.2, help="Weight of the chosen-completion NLL term")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch", type=int, default=8, help="Pairs per optimizer step")
    parser.add_argument("--max-length", type=int, default=7400)
    parser.add_argument("--limit", type=int, help="Only the first N pairs (for a test)")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(arguments.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pairs = [json.loads(line) for line in arguments.pairs.read_text().splitlines()]
    if arguments.limit:
        pairs = pairs[:arguments.limit]
    data = []
    for p in pairs:
        prompt = prompt_ids(tokenizer, p["prompt"])
        if len(prompt) + max(len(p["chosen_ids"]), len(p["rejected_ids"])) <= arguments.max_length:
            data.append((prompt, p["chosen_ids"], p["rejected_ids"]))
    print(f"{len(data)} pairs (of {len(pairs)})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, str(arguments.init), is_trainable=True)

    started = time.monotonic()
    model.eval()
    reference = []
    with torch.no_grad():
        for k, (prompt, chosen, rejected) in enumerate(data):
            reference.append((completion_logprobs(model, prompt, chosen)[0].item(),
                              completion_logprobs(model, prompt, rejected)[0].item()))
            if k % 500 == 0:
                print(f"reference {k}/{len(data)}, {(time.monotonic() - started) / 60:.0f} min", flush=True)
    print(f"reference done, {(time.monotonic() - started) / 60:.0f} min; reference prefers chosen in "
          f"{sum(c > r for c, r in reference) / len(reference):.0%} of pairs", flush=True)

    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=arguments.lr, weight_decay=0.0)
    steps = math.ceil(len(data) * arguments.epochs / arguments.batch)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    order = []
    rng = random.Random(arguments.seed)
    while len(order) < steps * arguments.batch:
        epoch = list(range(len(data)))
        rng.shuffle(epoch)
        order += epoch
    order = order[:steps * arguments.batch]

    started = time.monotonic()
    for step in range(steps):
        stats = {"loss": 0.0, "margin": 0.0, "right": 0, "nll": 0.0}
        batch = order[step * arguments.batch:(step + 1) * arguments.batch]
        for index in batch:
            prompt, chosen, rejected = data[index]
            ref_chosen, ref_rejected = reference[index]
            logp_chosen, nll_chosen = completion_logprobs(model, prompt, chosen)
            logp_rejected, _ = completion_logprobs(model, prompt, rejected)
            margin = arguments.beta * ((logp_chosen - ref_chosen) - (logp_rejected - ref_rejected))
            loss = (-F.logsigmoid(margin) + arguments.alpha * nll_chosen) / len(batch)
            loss.backward()
            stats["loss"] += loss.item()
            stats["margin"] += margin.item() / len(batch)
            stats["right"] += int(margin.item() > 0)
            stats["nll"] += nll_chosen.item() / len(batch)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            elapsed = (time.monotonic() - started) / 60
            print(f"step {step + 1}/{steps}  loss {stats['loss']:.4f}  margin {stats['margin']:+.3f}  "
                  f"pairs right {stats['right']}/{len(batch)}  chosen nll {stats['nll']:.3f}  "
                  f"{elapsed:.0f} min, ~{elapsed / (step + 1) * (steps - step - 1):.0f} min left", flush=True)

    arguments.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(arguments.out, safe_serialization=True)
    (arguments.out / "training_args.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(arguments).items()} | {"pairs": len(data)}, indent=2))
    print(f"saved {arguments.out}")


if __name__ == "__main__":
    main()
