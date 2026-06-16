#!/usr/bash
# FastWAM 1-GPU training attempt on RoboTwin data.
# WARNING: FastWAM is a 5B DiT model. 1 GPU (96GB H20) may still OOM.
# This script uses gradient checkpointing + bf16 + batch_size=1.
# If OOM, the model is simply too large for 1 GPU.

set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/luoyulin_code/luoyulin/code/qwen-oft/third_party/fastwam}"
cd "${FASTWAM_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

# Check if RoboTwin data exists
ROBOTWIN_DATA="${FASTWAM_ROOT}/data/robotwin2.0/robotwin2.0"
if [[ ! -d "${ROBOTWIN_DATA}" ]]; then
    echo "[error] RoboTwin data not found: ${ROBOTWIN_DATA}" >&2
    echo "        Download from: https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam" >&2
    echo "        Then extract to: data/robotwin2.0/" >&2
    exit 1
fi

# Check backbone
BACKBONE_PT="${FASTWAM_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
if [[ ! -f "${BACKBONE_PT}" ]]; then
    echo "[error] ActionDiT backbone not found: ${BACKBONE_PT}" >&2
    echo "        Run: bash scripts/setup_fastwam.sh" >&2
    exit 1
fi

GPUS="${GPUS:-0}"
PORT="${PORT:-29500}"

# 1-GPU overrides for memory
echo "=========================================="
echo "[FastWAM] 1-GPU training attempt on RoboTwin"
echo "  WARNING: 5B DiT model. May OOM on 1 GPU."
echo "  If OOM, you need 8+ GPUs for training."
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes 1 \
    --main_process_port "${PORT}" \
    "${FASTWAM_ROOT}/scripts/train.py" \
    task=robotwin_uncond_3cam_384_1e-4 \
    "output_dir=${FASTWAM_ROOT}/runs/robotwin_uncond_3cam_384_1e-4/1gpu_attempt" \
    "batch_size=1" \
    "gradient_accumulation_steps=8" \
    "mixed_precision=bf16" \
    "model.mot_checkpoint_mixed_attn=true" \
    "num_workers=2" \
    ${EXTRA_ARGS:-}

echo "=========================================="
echo "[FastWAM] Training attempt finished."
echo "  If OOM: need 8 GPUs (bash scripts/train_zero1.sh 8 ...)"
echo "  If success: checkpoint in runs/robotwin_uncond_3cam_384_1e-4/1gpu_attempt/"
echo "=========================================="
