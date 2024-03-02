"""Utility script to download a model from HuggingFace."""

import os
import sys

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python download_embeddings_model.py <local_dir> <hf_repo_id>")
        sys.exit(1)

    local_dir = sys.argv[1]
    hf_repo_id = sys.argv[2]

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=hf_repo_id, local_dir=local_dir, local_dir_use_symlinks=False
    )

    # workaround for https://github.com/UKPLab/sentence-transformers/pull/2460
    os.makedirs(os.path.join(local_dir, "2_Normalize"), exist_ok=True)

    # pretend local_dir is HF cache
    with open(os.path.join(local_dir, "version.txt"), "w", encoding="utf-8") as f:
        f.write("1")

    # remove pytorch_model.bin, load the model from model.safetensors
    os.remove(os.path.join(local_dir, "pytorch_model.bin"))
