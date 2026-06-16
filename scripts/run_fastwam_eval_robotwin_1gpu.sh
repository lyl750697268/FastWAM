#!/usr/bin/env bash
# FastWAM 1-GPU eval on RoboTwin with released checkpoint.
# This uses the official FastWAM eval framework (not our C1/C2 eval).
# Purpose: get a baseline score on RoboTwin for comparison.

set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/luoyulin_code/luoyulin/code/qwen-oft/third_party/fastwam}"
cd "${FASTWAM_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

# Release checkpoint
CKPT="${FASTWAM_ROOT}/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
STATS="${FASTWAM_ROOT}/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"

if [[ ! -f "${CKPT}" ]]; then
    echo "[error] Checkpoint not found: ${CKPT}" >&2
    echo "        Run: bash scripts/setup_fastwam.sh" >&2
    exit 1
fi

if [[ ! -f "${STATS}" ]]; then
    echo "[error] Dataset stats not found: ${STATS}" >&2
    echo "        Run: bash scripts/setup_fastwam.sh" >&2
    exit 1
fi

GPUS="${GPUS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "=========================================="
echo "[FastWAM] 1-GPU eval on RoboTwin"
echo "  ckpt: ${CKPT}"
echo "  stats: ${STATS}"
echo "  GPUs: ${GPUS}"
echo "=========================================="

# FastWAM eval uses its own RoboTwin wrapper in experiments/robotwin/
# NUM_GPUS is passed to MULTIRUN.num_gpus
CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" \
    "${FASTWAM_ROOT}/experiments/robotwin/run_robotwin_manager.py" \
    task=robotwin_uncond_3cam_384_1e-4 \
    ckpt="${CKPT}" \
    "EVALUATION.dataset_stats_path=${STATS}" \
    "MULTIRUN.num_gpus=${NUM_GPUS}" \
    "EVALUATION.instruction_type=unseen" \
    ${EXTRA_ARGS:-}

echo "=========================================="
echo "[FastWAM] Eval finished. Results in evaluate_results/"
echo "=========================================="
