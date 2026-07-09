#!/usr/bin/env bash
set -euo pipefail

# Convert all GMR X2 pkl outputs to motion_tracking npz and build the memmap dataset.

RETARGET_ROOT=/home/liuguoxing/Documents/dataset_new/retarget_x2
NPZ_ROOT=/home/liuguoxing/Documents/dataset_new/retarget_x2_npz
DATASET_MEMPATH=dataset/x2_amass_all
NUM_CPUS=16

mkdir -p "$NPZ_ROOT/AMASS"

# 1. pkl -> npz (with 30fps -> 50fps upsampling) for every subdataset that has pkl outputs.
for subdir in "$RETARGET_ROOT"/AMASS/*/; do
    name=$(basename "$subdir")
    pkl_count=$(find "$subdir" -name '*.pkl' | wc -l)
    if [ "$pkl_count" -eq 0 ]; then
        echo "[skip] $name: no pkl files"
        continue
    fi
    echo "[convert] $name ($pkl_count pkl files)"
    python scripts/data_process/gmr_pkl_to_motion_tracking_npz.py \
        "$subdir" \
        --output "$NPZ_ROOT/AMASS/$name" \
        --target-fps 50 \
        --num-cpus "$NUM_CPUS"
done

# 2. npz -> memmap dataset (all subdatasets together).
python scripts/data_process/generate_dataset.py \
    --dataset-root "$NPZ_ROOT/AMASS" \
    --mem-path "$DATASET_MEMPATH" \
    --amass-filter

echo "Done. Dataset at $DATASET_MEMPATH"
