mkdir -p dataset
# X2 retargeted dataset root (npz format)
DATASET_ROOT=/home/liuguoxing/Documents/dataset_new/retarget_x2_npz

python scripts/data_process/generate_dataset.py --dataset-root $DATASET_ROOT/AMASS/ACCAD --mem-path dataset/x2_amass_all --amass-filter
