#!/usr/bin/env bash
set -uo pipefail

readonly MT_DIR=/home/liuguoxing/Documents/motion_tracking
readonly PYTHON="$MT_DIR/.venv/bin/python"
readonly RUN_DIR="$MT_DIR/outputs/gp02_stage1_train"
readonly LOG="$MT_DIR/outputs/gp02_stage1_train.log"

mkdir -p "$RUN_DIR"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

latest_checkpoint() {
    find "$RUN_DIR" -type f -name 'checkpoint_*.pt' \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

final_checkpoint=$(find "$RUN_DIR" -type f -name 'checkpoint_final.pt' \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-)

if [[ -n "$final_checkpoint" && -f "$final_checkpoint" ]]; then
    ln -sfn "$final_checkpoint" "$RUN_DIR/checkpoint_final.pt"
    echo "[$(timestamp)] GP02 stage 1 already complete: $final_checkpoint" | tee -a "$LOG"
    exit 0
fi

checkpoint_args=()
wandb_args=()
resume_checkpoint=$(latest_checkpoint)

if [[ -n "$resume_checkpoint" && -f "$resume_checkpoint" ]]; then
    echo "[$(timestamp)] Resuming GP02 stage 1 from $resume_checkpoint" | tee -a "$LOG"
    checkpoint_args+=("checkpoint_path=$resume_checkpoint")

    wandb_id=$(
        "$PYTHON" -c \
            'import sys, torch; state=torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(state.get("wandb", {}).get("id", ""))' \
            "$resume_checkpoint" 2>/dev/null
    )
    if [[ -n "$wandb_id" ]]; then
        echo "[$(timestamp)] Resuming WandB run $wandb_id" | tee -a "$LOG"
        wandb_args+=("wandb.id=$wandb_id" "wandb.resume=allow")
    fi
else
    echo "[$(timestamp)] Starting GP02 stage 1 from scratch" | tee -a "$LOG"
fi

cd "$MT_DIR" || exit 1
"$PYTHON" scripts/train.py \
    task=GP02/GP02_tracking \
    +exp=train \
    wandb.project=gp02_motion_tracking \
    "${checkpoint_args[@]}" \
    "${wandb_args[@]}" \
    "hydra.run.dir=$RUN_DIR" \
    hydra.job.chdir=true \
    >> "$LOG" 2>&1
exit_code=$?

echo "[$(timestamp)] GP02 stage 1 process exited with code $exit_code" | tee -a "$LOG"

final_checkpoint=$(find "$RUN_DIR" -type f -name 'checkpoint_final.pt' \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-)
if [[ -n "$final_checkpoint" && -f "$final_checkpoint" ]]; then
    ln -sfn "$final_checkpoint" "$RUN_DIR/checkpoint_final.pt"
    echo "[$(timestamp)] GP02 stage 1 complete: $final_checkpoint" | tee -a "$LOG"
    exit 0
fi

echo "[$(timestamp)] No final checkpoint; systemd will restart and resume from the latest periodic checkpoint." | tee -a "$LOG"
exit 1
