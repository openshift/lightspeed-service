"""Utility script to download models from HuggingFace."""

import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to download models from HuggingFace"
    )
    parser.add_argument(
        "-l", "--local-dir", required=True, help="Directory to download model to"
    )
    parser.add_argument(
        "-r", "--hf-repo-id", required=True, help="Model repo id, ex. BAAI/bge-base-en"
    )
    args = parser.parse_args()

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.hf_repo_id, local_dir=args.local_dir, local_dir_use_symlinks=False
    )

    # workaround for https://github.com/UKPLab/sentence-transformers/pull/2460
    os.makedirs(os.path.join(args.local_dir, "2_Normalize"), exist_ok=True)

    # pretend local_dir is HF cache
    with open(os.path.join(args.local_dir, "version.txt"), "w", encoding="utf-8") as f:
        f.write("1")

    # remove pytorch_model.bin, load the model from model.safetensors
    os.remove(os.path.join(args.local_dir, "pytorch_model.bin"))
