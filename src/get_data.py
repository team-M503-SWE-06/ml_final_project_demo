"""
get_data.py — download the bina.az dataset from Kaggle into data/.

    python -m src.get_data

Uses kagglehub (works anonymously for this public dataset; if Kaggle asks
for credentials, put kaggle.json in ~/.kaggle or download the archive by
hand — see data/README.md). The raw file is NOT committed to git.
"""

import os
import shutil

from .data_prep import DATA_PATH

DATASET = "sehriyarmemmedli/binaaz-sale-project"
VERSION = 1   # version used for every number in the report


def main() -> None:
    if os.path.exists(DATA_PATH):
        print(f"{DATA_PATH} already exists — nothing to do.")
        return
    import kagglehub

    folder = kagglehub.dataset_download(f"{DATASET}/versions/{VERSION}")
    src = os.path.join(folder, os.path.basename(DATA_PATH))
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    shutil.copy(src, DATA_PATH)
    print(f"copied {src} -> {DATA_PATH} ({os.path.getsize(DATA_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
