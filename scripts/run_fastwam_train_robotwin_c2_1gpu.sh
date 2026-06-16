#!/usr/bin/env bash
# FastWAM 1-GPU training on C2 (skill-object compositional split).
# This is a TEST script to verify episode filtering works correctly.
# For real training, use run_fastwam_train_robotwin_c2_8gpu.sh

set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/nas/luoyulin/qwen-oft/third_party/fastwam}"
cd "${FASTWAM_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/qwen-oft/bin/python}"

# Paths
DATA_ROOT="${FASTWAM_ROOT}/data/robotwin2.0/robotwin2.0"
TRAIN_FILTER="${FASTWAM_ROOT}/data/c2_train_episode_indices.txt"
VAL_FILTER="${FASTWAM_ROOT}/data/c2_test_episode_indices.txt"
BACKBONE_PT="${FASTWAM_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

# Check data exists
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[error] RoboTwin data not found: ${DATA_ROOT}" >&2
    echo "        Download from: https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam" >&2
    echo "        Then extract to: data/robotwin2.0/" >&2
    exit 1
fi

# Check episode filter files exist
if [[ ! -f "${TRAIN_FILTER}" ]]; then
    echo "[error] Train episode filter not found: ${TRAIN_FILTER}" >&2
    echo "        Run: python scripts/build_episode_filter.py ..." >&2
    exit 1
fi

if [[ ! -f "${VAL_FILTER}" ]]; then
    echo "[error] Val episode filter not found: ${VAL_FILTER}" >&2
    exit 1
fi

# Check backbone
if [[ ! -f "${BACKBONE_PT}" ]]; then
    echo "[error] ActionDiT backbone not found: ${BACKBONE_PT}" >&2
    echo "        Run: bash scripts/setup_fastwam.sh" >&2
    exit 1
fi

GPUS="${GPUS:-0}"
PORT="${PORT:-29500}"

# Count episodes in filters
TRAIN_EPS=$(wc -l < "${TRAIN_FILTER}")
VAL_EPS=$(wc -l < "${VAL_FILTER}")

echo "=========================================="
echo "[FastWAM] 1-GPU C2 training (TEST)"
echo "  Train episodes: ${TRAIN_EPS}"
echo "  Val episodes:   ${VAL_EPS}"
echo "  Data root:      ${DATA_ROOT}"
echo "  WARNING: 5B DiT model. May OOM on 1 GPU."
echo "=========================================="

# 1-GPU overrides for memory + episode filter
CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes 1 \
    --main_process_port "${PORT}" \
    "${FASTWAM_ROOT}/scripts/train.py" \
    task=robotwin_uncond_3cam_384_1e-4 \
    "output_dir=${FASTWAM_ROOT}/runs/robotwin_uncond_3cam_384_c2/1gpu_test" \
    "batch_size=1" \
    "gradient_accumulation_steps=8" \
    "mixed_precision=bf16" \
    "model.mot_checkpoint_mixed_attn=true" \
    "num_workers=2" \
    "data.train.episode_filter_path=${TRAIN_FILTER}" \
    "data.val.episode_filter_path=${VAL_FILTER}" \
    "num_epochs=1" \
    "max_steps=10" \
    ${EXTRA_ARGS:-}

echo "=========================================="
echo "[FastWAM] C2 test run finished."
echo "  If OOM: need 8 GPUs"
echo "  If success: run 8-GPU script for full training"
echo "=========================================="
