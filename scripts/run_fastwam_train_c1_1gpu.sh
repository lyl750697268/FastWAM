#!/usr/bin/env bash
set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/nas/luoyulin/qwen-oft/third_party/fastwam}"
cd "${FASTWAM_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/qwen-oft/bin/python}"

C1_TRAIN="${FASTWAM_ROOT}/data/robotwin2.0_c1_new/train"
if [[ ! -d "${C1_TRAIN}" ]]; then
    echo "[error] C1 train data not found: ${C1_TRAIN}" >&2
    exit 1
fi

BACKBONE_PT="${FASTWAM_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
if [[ ! -f "${BACKBONE_PT}" ]]; then
    echo "[error] ActionDiT backbone not found: ${BACKBONE_PT}" >&2
    exit 1
fi

GPUS="${GPUS:-0}"
PORT="${PORT:-29500}"

echo "=========================================="
echo "[FastWAM] 1-GPU C1 training (TEST)"
echo "  Data: ${C1_TRAIN}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes 1 \
    --main_process_port "${PORT}" \
    "${FASTWAM_ROOT}/scripts/train.py" \
    task=robotwin_c1_uncond_3cam_384_1e-4 \
    "output_dir=${FASTWAM_ROOT}/runs/robotwin_c1_uncond_3cam_384_1e-4/1gpu_test" \
    "batch_size=1" \
    "gradient_accumulation_steps=8" \
    "mixed_precision=bf16" \
    "model.mot_checkpoint_mixed_attn=true" \
    "num_workers=2" \
    "num_epochs=1" \
    "max_steps=10" \
    ${EXTRA_ARGS:-}

echo "[FastWAM] C1 test run finished."
