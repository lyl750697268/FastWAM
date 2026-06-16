#!/usr/bin/env bash
# FastWAM 1-GPU training on C2 (skill-object compositional split).
# Uses our own HDF5-converted Lerobot dataset.
# This is a TEST script to verify dataset loading works.

set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/luoyulin_code/luoyulin/code/qwen-oft/third_party/fastwam}"
cd "${FASTWAM_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

# Check data exists
C2_TRAIN="${FASTWAM_ROOT}/data/robotwin2.0_c2/train"
C2_TEST="${FASTWAM_ROOT}/data/robotwin2.0_c2/test"
if [[ ! -d "${C2_TRAIN}" ]]; then
    echo "[error] C2 train data not found: ${C2_TRAIN}" >&2
    echo "        Run: python scripts/convert_c2_hdf5_to_lerobot.py ..." >&2
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

echo "=========================================="
echo "[FastWAM] 1-GPU C2 training (TEST)"
echo "  Data: ${C2_TRAIN}"
echo "  WARNING: 5B DiT model. May OOM on 1 GPU."
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes 1 \
    --main_process_port "${PORT}" \
    "${FASTWAM_ROOT}/scripts/train.py" \
    task=robotwin_c2_uncond_3cam_384_1e-4 \
    "output_dir=${FASTWAM_ROOT}/runs/robotwin_c2_uncond_3cam_384_1e-4/1gpu_test" \
    "batch_size=1" \
    "gradient_accumulation_steps=8" \
    "mixed_precision=bf16" \
    "model.mot_checkpoint_mixed_attn=true" \
    "num_workers=2" \
    "num_epochs=1" \
    "max_steps=10" \
    ${EXTRA_ARGS:-}

echo "=========================================="
echo "[FastWAM] C2 test run finished."
echo "=========================================="
