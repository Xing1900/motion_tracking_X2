#!/usr/bin/env python3
"""Update task/success metadata for one finalized raw episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate an X2 raw VR episode")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--success", choices=["true", "false", "unknown"])
    parser.add_argument("--task")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = args.episode.expanduser().resolve()
    manifest_path = episode / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if args.success is not None:
        manifest["success"] = {"true": True, "false": False, "unknown": None}[args.success]
    if args.task is not None:
        if not args.task.strip():
            raise ValueError("--task must not be empty")
        manifest["task"] = args.task

    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(manifest_path)
    print(
        f"updated {manifest_path}: "
        f"task={manifest.get('task')!r}, success={manifest.get('success')}"
    )


if __name__ == "__main__":
    main()
