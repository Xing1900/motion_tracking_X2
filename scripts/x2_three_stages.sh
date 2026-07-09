#!/bin/bash
set -uo pipefail

MT_DIR=/home/liuguoxing/Documents/motion_tracking
PYTHON=$MT_DIR/.venv/bin/python
LOG=$MT_DIR/outputs/x2_three_stages.log

mkdir -p $MT_DIR/outputs

STAGES=("train" "adapt" "finetune")
# Normal frame counts from exp configs
FRAME_TARGETS=(8000000000 1000000000 4000000000)

prev_checkpoint=""

for stage_idx in 0 1 2; do
    stage=${STAGES[$stage_idx]}
    run_dir=$MT_DIR/outputs/x2_${stage}
    
    mkdir -p $run_dir
    
    echo "[$(date)] ===== Stage $stage =====" | tee -a "$LOG"
    
    # Find final checkpoint recursively (train.py saves it under wandb/run-*/files/)
    final_ckpt=$(find "$run_dir" -name 'checkpoint_final.pt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
    
    # If final checkpoint exists, stage is done
    if [ -n "$final_ckpt" ] && [ -f "$final_ckpt" ]; then
        echo "[$(date)] Stage $stage already complete: $final_ckpt" | tee -a "$LOG"
        # Ensure a stable symlink exists at the run_dir root for downstream stages
        if [ "$final_ckpt" != "$run_dir/checkpoint_final.pt" ]; then
            ln -sf "$final_ckpt" "$run_dir/checkpoint_final.pt"
        fi
        prev_checkpoint=$final_ckpt
        continue
    fi
    
    # Find latest checkpoint in this stage's run dir
    latest_ckpt=$(find "$run_dir" -name 'checkpoint_*.pt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
    
    wandb_args=""
    ckpt_arg=""
    resume_args=""
    
    if [ -n "$latest_ckpt" ] && [ -f "$latest_ckpt" ]; then
        # Resume this stage
        echo "[$(date)] Resuming $stage from $latest_ckpt" | tee -a "$LOG"
        ckpt_arg="checkpoint_path=$latest_ckpt"
        wandb_id=$($PYTHON -c "import torch; ckpt=torch.load('$latest_ckpt', map_location='cpu', weights_only=False); print(ckpt.get('wandb',{}).get('id',''))" 2>/dev/null)
        if [ -n "$wandb_id" ]; then
            echo "[$(date)] Resuming wandb run: $wandb_id" | tee -a "$LOG"
            wandb_args="wandb.id=$wandb_id wandb.resume=allow"
        fi
    elif [ -n "$prev_checkpoint" ] && [ "$stage_idx" -gt  0 ]; then
        # Start this stage from previous stage's final checkpoint
        echo "[$(date)] Starting $stage from previous checkpoint: $prev_checkpoint" | tee -a "$LOG"
        ckpt_arg="checkpoint_path=$prev_checkpoint"
        # Reset iteration/frame counters so the new stage trains for its full budget
        resume_args="checkpoint_resume_iter=0 checkpoint_resume_env_frames=0"
    else
        echo "[$(date)] Starting $stage from scratch" | tee -a "$LOG"
    fi
    
    cd $MT_DIR
    $PYTHON scripts/train.py +exp=$stage $ckpt_arg $resume_args $wandb_args hydra.run.dir=$run_dir hydra.job.chdir=true >> "$LOG" 2>&1
    exit_code=$?
    
    echo "[$(date)] Stage $stage exited with code $exit_code" | tee -a "$LOG"
    
    # After exit, check if final checkpoint was created
    final_ckpt=$(find "$run_dir" -name 'checkpoint_final.pt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
    if [ -z "$final_ckpt" ] || [ ! -f "$final_ckpt" ]; then
        echo "[$(date)] Stage $stage did not complete (no checkpoint_final.pt). Will retry on restart." | tee -a "$LOG"
        exit 1
    fi
    
    # Create stable symlink for downstream stages
    ln -sf "$final_ckpt" "$run_dir/checkpoint_final.pt"
    prev_checkpoint=$final_ckpt
done

echo "[$(date)] All three stages complete!" | tee -a "$LOG"
