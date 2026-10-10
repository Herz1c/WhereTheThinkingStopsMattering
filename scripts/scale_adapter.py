"""A copy of a LoRA adapter with its update scaled by a factor, for a length/accuracy curve.

    python scripts/scale_adapter.py checkpoints/stoppoint_all3 checkpoints/stoppoint_all3_x050 --scale 0.5

A LoRA adds (lora_alpha / r) * B @ A to each adapted weight. Scaling lora_alpha by a factor
scales that update by the same factor and leaves the weights untouched, so each factor is a
model between the base (0) and the trained adapter (1) at no training cost. Transformers,
PEFT and vLLM all read lora_alpha from adapter_config.json, so the scaled copy loads and
evaluates exactly like the original.
"""

import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--scale", type=float, required=True)
    arguments = parser.parse_args()

    shutil.copytree(arguments.source, arguments.out)
    config_path = arguments.out / "adapter_config.json"
    config = json.loads(config_path.read_text())
    original = config["lora_alpha"]
    config["lora_alpha"] = original * arguments.scale
    config_path.write_text(json.dumps(config, indent=2))
    print(f"{arguments.out}: lora_alpha {original} -> {config['lora_alpha']} (r = {config['r']})")


if __name__ == "__main__":
    main()
