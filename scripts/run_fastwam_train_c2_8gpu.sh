#!/usr/bin/env bash
# FastWAM 8-GPU training on C2 (skill-object compositional split).
# Uses our own HDF5-converted Lerobot dataset.

set -euo pipefail

FASTWAM_ROOT="${FASTWAM_ROOT:-/mnt/nas/luoyulin/qwen-oft/third_party/fastwam}"
cd "${FASTWAM_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/qwen-oft/bin/python}"

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

NUM_GPUS="${NUM_GPUS:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-29500}"

echo "=========================================="
echo "[FastWAM] 8-GPU C2 training"
echo "  Train: ${C2_TRAIN}"
echo "  Test:  ${C2_TEST}"
echo "  GPUs:  ${NUM_GPUS}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --main_process_port "${PORT}" \
    "${FASTWAM_ROOT}/scripts/train.py" \
    task=robotwin_c2_uncond_3cam_384_1e-4 \
    "output_dir=${FASTWAM_ROOT}/runs/robotwin_c2_uncond_3cam_384_1e-4/8gpu_train" \
    ${EXTRA_ARGS:-}

echo "=========================================="
echo "[FastWAM] C2 training finished."
echo "=========================================="
