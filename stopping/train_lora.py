"""LoRA fine-tuning of Qwen3-0.6B on (prompt, completion token ids) pairs.

    python stopping/train_lora.py --data runs/sft_shortest.jsonl --out checkpoints/shortest

One trainer for every supervised method in this project: each row carries a chat
prompt and the exact token ids the model should produce. The loss is computed on the
completion only; the prompt is context. Training sequences are built the way
evaluate.py builds its inputs (chat template, thinking enabled), so the model learns
in the format it is scored in.

The adapter targets only the projections vLLM can serve as a LoRA (attention and
MLP), so evaluate.py --adapter loads it unchanged. It is saved as safetensors.

Memory: one sequence per step with gradient checkpointing; with a 151,936-token
vocabulary the logits dominate, so the loss is computed in chunks of positions and
the full fp32 logits are never materialized. Fits a 16 GB GPU at 4k+ tokens.
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
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LOSS_CHUNK = 1024


def load_rows(path, tokenizer, max_length):
    """Rows -> (input_ids, number of prompt tokens). Drops sequences over max_length."""
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    examples, dropped = [], 0
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                               enable_thinking=True)
        if hasattr(prompt, "input_ids"):        # transformers 5 returns a BatchEncoding
            prompt = prompt["input_ids"]
        completion = list(row["completion_ids"])
        if not completion or completion[-1] != end:
            completion.append(end)              # the model must also learn to stop
        if len(prompt) + len(completion) > max_length:
            dropped += 1
            continue
        examples.append((prompt + completion, len(prompt)))
    return examples, dropped


def chunked_loss(model, ids, prompt_length):
    """Mean cross-entropy over the completion: run the transformer once, then project
    and score the completion positions in chunks. LoRA layers live inside the modules,
    so calling the inner model directly still trains them."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state[0]   # (T, d)
    # Position t predicts token t+1: score positions prompt_length-1 .. T-2.
    states = hidden[prompt_length - 1:-1]
    targets = ids[0, prompt_length:]
    total = states.new_zeros((), dtype=torch.float32)
    for start in range(0, states.shape[0], LOSS_CHUNK):
        logits = base.lm_head(states[start:start + LOSS_CHUNK]).float()
        total = total + F.cross_entropy(logits, targets[start:start + LOSS_CHUNK], reduction="sum")
    return total / targets.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch", type=int, default=16, help="Sequences per optimizer step")
    parser.add_argument("--max-length", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(arguments.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    examples, dropped = load_rows(arguments.data, tokenizer, arguments.max_length)
    print(f"{len(examples)} examples ({dropped} over {arguments.max_length} tokens dropped), "
          f"{sum(len(i) - p for i, p in examples)} completion tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=arguments.rank, lora_alpha=arguments.alpha, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=arguments.lr, weight_decay=0.0)
    steps = math.ceil(len(examples) * arguments.epochs / arguments.batch)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    rng = random.Random(arguments.seed)
    order = []
    while len(order) < steps * arguments.batch:
        epoch = list(range(len(examples)))
        rng.shuffle(epoch)
        order += epoch
    order = order[:steps * arguments.batch]

    model.train()
    started = time.monotonic()
    for step in range(steps):
        running = 0.0
        for index in order[step * arguments.batch:(step + 1) * arguments.batch]:
            ids, prompt_length = examples[index]
            ids = torch.tensor([ids], device="cuda")
            loss = chunked_loss(model, ids, prompt_length) / arguments.batch
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            elapsed = (time.monotonic() - started) / 60
            print(f"step {step + 1}/{steps}  loss {running:.4f}  lr {schedule.get_last_lr()[0]:.2e}  "
                  f"{elapsed:.0f} min, ~{elapsed / (step + 1) * (steps - step - 1):.0f} min left", flush=True)

    arguments.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(arguments.out, safe_serialization=True)
    (arguments.out / "training_args.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(arguments).items()} | {"examples": len(examples)}, indent=2))
    print(f"saved {arguments.out}")


if __name__ == "__main__":
    main()
