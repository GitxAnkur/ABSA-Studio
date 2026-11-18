"""Download optional external assets (GloVe vectors and NLTK corpora).

Usage:
    python -m scripts.download_assets

The trained models committed in this repository already contain their learned
weights, so these assets are only required to *retrain* the pipeline.  The code
falls back gracefully if they are unavailable:

* missing GloVe -> randomly initialised trainable embeddings
* missing NLTK corpora -> built-in stop-word list and no lemmatisation
"""

from __future__ import annotations

import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import EXTERNAL_DIR, GLOVE_PATH, NLTK_DATA_DIR

GLOVE_URL = (
    "https://huggingface.co/datasets/antokun/glove.6B.50d/resolve/main/glove.6B.50d.txt"
)
MIN_GLOVE_BYTES = 100 * 1024 * 1024  # sanity check for a truncated download


def download_glove() -> None:
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    if os.path.exists(GLOVE_PATH) and os.path.getsize(GLOVE_PATH) > MIN_GLOVE_BYTES:
        print(f"[ASSETS] GloVe already present: {GLOVE_PATH}")
        return

    print(f"[ASSETS] Downloading GloVe 50d (~163 MB) from {GLOVE_URL}")
    temporary = GLOVE_PATH + ".part"
    try:
        urllib.request.urlretrieve(GLOVE_URL, temporary)
        if os.path.getsize(temporary) < MIN_GLOVE_BYTES:
            raise IOError("Downloaded file is unexpectedly small.")
        os.replace(temporary, GLOVE_PATH)
        print(f"[ASSETS] Saved to {GLOVE_PATH}")
    except Exception as exc:
        if os.path.exists(temporary):
            os.remove(temporary)
        print(
            "[ASSETS] GloVe download failed. Training will fall back to random "
            f"embeddings.\n         Reason: {exc}"
        )


def download_nltk() -> None:
    import nltk

    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    print("[ASSETS] Downloading NLTK stopwords / wordnet / omw-1.4")
    for package in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(package, download_dir=NLTK_DATA_DIR, quiet=True)
            print(f"[ASSETS]   {package}: ok")
        except Exception as exc:
            print(f"[ASSETS]   {package}: failed ({exc})")


def main() -> None:
    download_glove()
    download_nltk()
    print("[ASSETS] Done.")


if __name__ == "__main__":
    main()
