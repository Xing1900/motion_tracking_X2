#!/usr/bin/env bash
set -euo pipefail

RETARGET_ROOT=${RETARGET_ROOT:-/home/liuguoxing/Documents/dataset_new/retarget_gp02_v3}
NPZ_ROOT=${NPZ_ROOT:-/home/liuguoxing/Documents/dataset_new/retarget_gp02_v3_npz}
AMASS_MEMPATH=${AMASS_MEMPATH:-dataset/gp02_amass_all}
LAFAN_MEMPATH=${LAFAN_MEMPATH:-dataset/gp02_lafan_all}
NUM_CPUS=${NUM_CPUS:-8}

python scripts/data_process/gmr_pkl_to_motion_tracking_npz.py \
    "$RETARGET_ROOT/AMASS" \
    --output "$NPZ_ROOT/AMASS" \
    --robot gp02_v3 \
    --target-fps 50 \
    --num-cpus "$NUM_CPUS"

python scripts/data_process/gmr_pkl_to_motion_tracking_npz.py \
    "$RETARGET_ROOT/LAFAN" \
    --output "$NPZ_ROOT/LAFAN" \
    --robot gp02_v3 \
    --target-fps 50 \
    --num-cpus "$NUM_CPUS"

python scripts/data_process/generate_dataset.py \
    --dataset-root "$NPZ_ROOT/AMASS" \
    --mem-path "$AMASS_MEMPATH" \
    --amass-filter

python scripts/data_process/generate_dataset.py \
    --dataset-root "$NPZ_ROOT/LAFAN" \
    --mem-path "$LAFAN_MEMPATH"
