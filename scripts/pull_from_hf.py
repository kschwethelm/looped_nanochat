#!/usr/bin/env python3
"""
Download a folder from a Hugging Face repo into the nanochat cache directory.

Example:
  python -m scripts.pull_from_hf --repo-id Trelis/nanochat-recursive \
    --repo-path sft/d20 --stage sft --target-tag d20

Download the tokenizer
  python scripts/pull_from_hf.py \
    --repo-id Trelis/nanochat-recursive \
    --repo-path tokenizer \
    --dest-dir ~/.cache/nanochat/tokenizer
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_stage_dir(stage: str, base_dir: Path, target_tag: str | None) -> Path:
    stage_map = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
    }
    if stage not in stage_map:
        raise ValueError(f"Unknown stage '{stage}'")
    root = base_dir / stage_map[stage]
    return root / target_tag


def main() -> None:
    parser = argparse.ArgumentParser("Download HF artifacts into nanochat cache")
    parser.add_argument("--repo-id", required=True, help="HF repo slug (e.g. Trelis/nanochat)")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--repo-path", required=True, help="Path inside the repo (e.g. sft/d20)")
    parser.add_argument(
        "--stage",
        choices=["base", "mid", "sft"],
        help="Place files under $NANOCHAT_BASE_DIR/<stage>_checkpoints/<target-tag>",
    )
    parser.add_argument(
        "--target-tag",
        default=None,
        help="Folder name used when --stage is set (defaults to last segment of repo-path)",
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        help="Explicit destination directory (used when --stage is omitted)",
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get(
            "NANOCHAT_BASE_DIR", os.path.join(Path.home(), ".cache", "nanochat")
        ),
        help="Where nanochat checkpoints live locally",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="HF auth token (or set HF_TOKEN env / cached login)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    if args.stage:
        target_tag = args.target_tag or Path(args.repo_path).name
        dest_dir = resolve_stage_dir(args.stage, base_dir, target_tag)
    else:
        if not args.dest_dir:
            raise SystemExit("Either --stage or --dest-dir must be provided.")
        dest_dir = Path(args.dest_dir).expanduser()
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading hf://{args.repo_id}/{args.repo_path} -> {dest_dir}")
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            allow_patterns=[f"{args.repo_path}/**"],
            local_dir=tmpdir,
            token=args.token,
        )
        src = Path(tmpdir) / args.repo_path
        if not src.exists():
            raise FileNotFoundError(f"Downloaded folder {src} not found")
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(src), str(dest_dir))
    print("Download complete.")


if __name__ == "__main__":
    main()
