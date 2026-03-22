from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError


DEFAULT_REPO_ID = "taiyi09/EgoGazeVQA"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download EgoGazeVQA from Hugging Face to the local dataset directory.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--target-dir", default="dataset/egogazevqa", help="Local target directory.")
    parser.add_argument("--token", default=None, help="Hugging Face token. Falls back to HF_TOKEN or local login.")
    parser.add_argument(
        "--subset",
        nargs="*",
        default=["ego4d", "egoexo", "egtea"],
        help="Video subsets to download. Use any of: ego4d egoexo egtea.",
    )
    parser.add_argument("--metadata-only", action="store_true", help="Download metadata and narrations only, no videos.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    target_dir = Path(args.target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = ["README.md", "metadata.csv", "metadata.jsonl", "metadata.json", "*.json"]
    if not args.metadata_only:
        for subset in args.subset:
            allow_patterns.append(f"{subset}/**")

    ignore_patterns = ["SFT_model/**"]

    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=token,
        )
    except GatedRepoError as exc:
        parser.exit(
            1,
            "Access denied for the gated dataset. First open the dataset card, accept access, then run `hf auth login` or pass `--token`.\n"
            f"Dataset: https://huggingface.co/datasets/{args.repo_id}\n"
            f"Details: {exc}\n",
        )

    print(f"Downloaded EgoGazeVQA to: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
