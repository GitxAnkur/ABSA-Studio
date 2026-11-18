"""Prepare the SemEval-2014 restaurant dataset.

Usage:
    python -m scripts.prepare_data
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import prepare_datasets


def main() -> None:
    prepare_datasets(verbose=True)


if __name__ == "__main__":
    main()
