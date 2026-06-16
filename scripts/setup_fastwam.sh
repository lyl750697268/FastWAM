#!/usr/bin/env bash
# FastWAM setup script for local 1-GPU eval and (attempted) training.
# Downloads release checkpoint, dataset stats, and prepares environment.

set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/nas/luoyulin/qwen-oft/third_party/fastwam}"
cd "${FASTWAM_ROOT}"

# ============ Proxy for HuggingFace / GitHub ============
export http_proxy="${http_proxy:-http://127.0.0.1:7897}"
export https_proxy="${https_proxy:-http://127.0.0.1:7897}"

# ============ Step 1: Create conda env (optional) ============
# FastWAM requires torch==2.7.1+cu128, which may conflict with qwen-oft env.
# We try to reuse qwen-oft env if torch>=2.5 is available.
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/qwen-oft/bin/python}"
if ! "${PYTHON_BIN}" -c "import torch; assert torch.__version__ >= '2.5'" 2>/dev/null; then
    echo "[warn] qwen-oft torch version may be too old for FastWAM (needs >=2.5)."
    echo "       FastWAM officially requires torch==2.7.1+cu128."
    echo "       Continue anyway; some ops may fail."
fi

# ============ Step 2: Install FastWAM package ============
echo "[setup] Installing FastWAM package..."
"${PYTHON_BIN}" -m pip install -e "${FASTWAM_ROOT}" --quiet 2>/dev/null || {
    echo "[setup] pip install -e failed; trying without -e..."
    "${PYTHON_BIN}" -m pip install "${FASTWAM_ROOT}" --quiet
}

# ============ Step 3: Download release checkpoints ============
CKPT_DIR="${FASTWAM_ROOT}/checkpoints/fastwam_release"
mkdir -p "${CKPT_DIR}"

echo "[setup] Downloading FastWAM release checkpoints from HuggingFace..."
"${PYTHON_BIN}" -m pip install --quiet huggingface-hub 2>/dev/null || true

"${PYTHON_BIN}" -c "
from huggingface_hub import hf_hub_download
import os
repo = 'yuanty/fastwam'
files = [
    'robotwin_uncond_3cam_384.pt',
    'robotwin_uncond_3cam_384_dataset_stats.json',
    'libero_uncond_2cam224.pt',
    'libero_uncond_2cam224_dataset_stats.json',
]
for f in files:
    try:
        path = hf_hub_download(repo_id=repo, filename=f, local_dir='${CKPT_DIR}', local_dir_use_symlinks=False)
        print(f'[ok] {f} -> {path}')
    except Exception as e:
        print(f'[skip] {f}: {e}')
"

# ============ Step 4: Check RoboTwin assets ============
echo "[setup] Checking RoboTwin assets..."
ROBOTWIN_ASSETS="${FASTWAM_ROOT}/third_party/RoboTwin"
if [[ ! -d "${ROBOTWIN_ASSETS}" ]]; then
    echo "[warn] RoboTwin assets not found at ${ROBOTWIN_ASSETS}"
    echo "       For full eval, clone https://github.com/RoboTwin-Platform/RoboTwin"
    echo "       and download assets. Skipping for now."
fi

# ============ Step 5: Preprocess ActionDiT backbone (one-time) ============
BACKBONE_PT="${FASTWAM_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
if [[ ! -f "${BACKBONE_PT}" ]]; then
    echo "[setup] Preprocessing ActionDiT backbone (one-time, ~5 min)..."
    "${PYTHON_BIN}" "${FASTWAM_ROOT}/scripts/preprocess_action_dit_backbone.py" \
        --model-config "${FASTWAM_ROOT}/configs/model/fastwam.yaml" \
        --output "${BACKBONE_PT}" \
        --device cuda \
        --dtype bfloat16 2>&1 | tee "${FASTWAM_ROOT}/checkpoints/preprocess_backbone.log"
else
    echo "[setup] ActionDiT backbone already exists: ${BACKBONE_PT}"
fi

echo "=========================================="
echo "[setup] FastWAM setup complete."
echo "  Checkpoint dir: ${CKPT_DIR}"
echo "  Backbone: ${BACKBONE_PT}"
echo ""
echo "Next steps:"
echo "  1-GPU eval:  bash scripts/run_fastwam_eval_robotwin_1gpu.sh"
echo "  1-GPU train attempt: bash scripts/run_fastwam_train_robotwin_1gpu.sh"
echo "=========================================="
