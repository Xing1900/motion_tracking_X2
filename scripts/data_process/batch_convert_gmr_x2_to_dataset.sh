#!/usr/bin/env bash
set -euo pipefail

# Convert GMR X2 pkl outputs to motion_tracking npz and build memmap dataset.

RETARGET_ROOT=/home/liuguoxing/Documents/dataset_new/retarget_x2
NPZ_ROOT=/home/liuguoxing/Documents/dataset_new/retarget_x2_npz
DATASET_MEMPATH=dataset/x2_amass_all
NUM_CPUS=8

# 1. pkl -> npz (with 30fps -> 50fps upsampling)
python scripts/data_process/gmr_pkl_to_motion_tracking_npz.py \
    "$RETARGET_ROOT/AMASS/ACCAD" \
    --output "$NPZ_ROOT/AMASS/ACCAD" \
    --target-fps 50 \
    --num-cpus "$NUM_CPUS"

# 2. npz -> memmap dataset
python scripts/data_process/generate_dataset.py \
    --dataset-root "$NPZ_ROOT/AMASS/ACCAD" \
    --mem-path "$DATASET_MEMPATH" \
    --amass-filter

echo "Done. Dataset at $DATASET_MEMPATH"
