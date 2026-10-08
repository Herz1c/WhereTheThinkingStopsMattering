#!/usr/bin/env bash
# Prepares a rented GPU pod (RunPod PyTorch template or any Ubuntu + CUDA box)
# and runs the evaluation. Keeps everything on the persistent volume, so a
# stopped pod loses nothing.
#
#   bash scripts/setup_pod.sh setup    # uv, venv, requirements, frozen questions
#   bash scripts/setup_pod.sh smoke    # 30-question plumbing check (~1 min)
#   bash scripts/setup_pod.sh base     # full baseline -> runs/base (~2 h on a 4090)
#
# `base` runs in the background under nohup and logs to runs/base.log, so an
# SSH disconnect does not kill it: follow it with `tail -f runs/base.log`.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/Herz1c/WhereTheThinkingStopsMattering.git}"
REPO_DIR="$WORKSPACE/WhereTheThinkingStopsMattering"

# Model weights and datasets go to the volume, not to the pod's ephemeral disk.
export HF_HOME="$WORKSPACE/hf_cache"
export UV_CACHE_DIR="$WORKSPACE/uv_cache"
export PATH="$HOME/.local/bin:$PATH"

check_gpu() {
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    # bf16 needs Ampere (8.0) or newer; on older cards evaluate.py needs dtype="half",
    # and the baseline must then be scored the same way.
    if [[ "${cap%%.*}" -lt 8 ]]; then
        echo "WARNING: compute capability $cap has no bf16. Add dtype=\"half\" in load_engine" \
             "(evaluate.py) for both the baseline and every checkpoint." >&2
    fi
}

setup() {
    check_gpu
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        git clone "$REPO_URL" "$REPO_DIR"
    fi
    cd "$REPO_DIR"
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # vLLM ships wheels for these Pythons; the template's system Python may be newer.
    [[ -d .venv ]] || uv venv --python 3.12
    source .venv/bin/activate
    uv pip install -r requirements.txt
    uv pip install matplotlib    # analysis/ figures; the evaluation itself does not need it
    python utils/eval/benchmarks.py
    echo "Setup done. Next: bash scripts/setup_pod.sh smoke"
}

activate() {
    cd "$REPO_DIR"
    source .venv/bin/activate
}

smoke() {
    activate
    rm -rf runs/check
    python evaluate.py --out runs/check --smoke
}

base() {
    activate
    if [[ -e runs/base/results.json ]]; then
        echo "runs/base already finished; it is the reference, not overwriting it." >&2
        exit 1
    fi
    mkdir -p runs
    nohup python evaluate.py --out runs/base > runs/base.log 2>&1 &
    echo "Baseline started (pid $!). Follow it with: tail -f $REPO_DIR/runs/base.log"
}

case "${1:-}" in
    setup) setup ;;
    smoke) smoke ;;
    base)  base ;;
    *) echo "usage: $0 {setup|smoke|base}" >&2; exit 2 ;;
esac
