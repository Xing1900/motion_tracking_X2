#!/usr/bin/env python
"""Evaluate / play a local checkpoint without going through wandb.

Examples:
    # Visualize the trained policy in simulation (opens Viser viewer)
    .venv/bin/python scripts/eval_local.py \
        --checkpoint outputs/x2_train/wandb/run-20260707_094106-rbrb4cjt/files/checkpoint_final.pt \
        --play

    # Run a headless quantitative evaluation and print metrics
    .venv/bin/python scripts/eval_local.py \
        --checkpoint outputs/x2_train/wandb/run-20260707_094106-rbrb4cjt/files/checkpoint_final.pt

    # Record a video (requires a display / remote forwarding)
    .venv/bin/python scripts/eval_local.py \
        --checkpoint outputs/x2_train/wandb/run-20260707_094106-rbrb4cjt/files/checkpoint_final.pt \
        --video

    # Export a deployable torchscript / ONNX policy
    .venv/bin/python scripts/eval_local.py \
        --checkpoint outputs/x2_train/wandb/run-20260707_094106-rbrb4cjt/files/checkpoint_final.pt \
        --play --export
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import hydra
from omegaconf import OmegaConf, DictConfig

from scripts.utils.play import play
from scripts.utils.eval import eval as eval_fn


def main():
    parser = argparse.ArgumentParser(description="Local checkpoint evaluation / visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--task", type=str, default="X2/X2_tracking", help="Hydra task config")
    parser.add_argument("--algo", type=str, default="ppo_train", help="Hydra algo config")
    parser.add_argument("--play", action="store_true", help="Open interactive Viser viewer")
    parser.add_argument("--video", action="store_true", help="Render and record video")
    parser.add_argument("--success", action="store_true", help="Test success rate (headless, 2048 envs)")
    parser.add_argument("--export", action="store_true", help="Export deployable policy to exports/")
    parser.add_argument("--headless", action="store_true", help="Force headless mode even with --play")
    parser.add_argument("--motion-id", type=int, default=None, help="Fix a specific motion clip id from the dataset")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    overrides = [
        f"task={args.task}",
        f"+algo={args.algo}",
        f"checkpoint_path={args.checkpoint}",
        "vecnorm=eval",
    ]
    if args.motion_id is not None:
        overrides.append(f"+task.command.dataset.fix_motion_id={args.motion_id}")

    with hydra.initialize(config_path="../cfg", job_name="eval_local", version_base=None):
        cfg = hydra.compose(config_name="eval", overrides=overrides)

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Make sure the checkpoint we requested is the one that gets loaded
    cfg.checkpoint_path = args.checkpoint
    cfg.vecnorm = "eval"

    if args.play:
        if not args.success:
            cfg.app.headless = args.headless
            cfg.task.num_envs = 16
        cfg.export_policy = args.export
        cfg.perf_test = False
        play(cfg)
    else:
        if args.video:
            cfg.task.num_envs = 16
            cfg.eval_render = True
            cfg.app.enable_cameras = True
            cfg.app.headless = False
        eval_fn(cfg)


if __name__ == "__main__":
    main()
