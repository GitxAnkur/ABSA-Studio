"""Loading and preparing the SemEval-2014 Task 4 restaurant dataset.

The official XML files are parsed into two tabular views:

1. ``*_sentences.csv`` -- one row per review sentence with a multi-label
   vector over the five canonical aspect categories.  Used for aspect
   detection.
2. ``*_aspect_sentiment.csv`` -- one row per (sentence, aspect) annotation
   with a three-class polarity label.  Used for aspect-specific sentiment
   classification.

Run ``python -m scripts.prepare_data`` to (re)build the processed files.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List

import pandas as pd

from .config import (
    ASPECTS,
    CONFLICT_POLICY,
    PROCESSED_DIR,
    REPORTS_DIR,
    SENTIMENT_LABELS,
    TEST_XML,
    TRAIN_XML,
)
from .preprocessing import tokenize

CATEGORY_MAP = {"anecdotes/miscellaneous": "miscellaneous"}


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------
def parse_semeval_xml(path: str) -> List[Dict]:
    """Parse a SemEval-2014 ABSA XML file.

    Returns a list of dicts with the sentence text, its aspect terms and its
    aspect-category annotations.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SemEval XML file not found: {path}")

    root = ET.parse(path).getroot()
    sentences: List[Dict] = []

    for node in root.findall("sentence"):
        text = (node.findtext("text") or "").strip()
        terms = []
        for term in node.findall("./aspectTerms/aspectTerm"):
            terms.append(
                {
                    "term": term.get("term", ""),
                    "polarity": (term.get("polarity") or "").lower(),
                    "from": term.get("from"),
                    "to": term.get("to"),
                }
            )

        categories = []
        for cat in node.findall("./aspectCategories/aspectCategory"):
            raw = (cat.get("category") or "").lower()
            aspect = CATEGORY_MAP.get(raw, raw)
            categories.append(
                {
                    "aspect": aspect,
                    "polarity": (cat.get("polarity") or "").lower(),
                }
            )

        sentences.append(
            {
                "sentence_id": node.get("id"),
                "text": text,
                "aspect_terms": terms,
                "categories": categories,
            }
        )

    return sentences


# ---------------------------------------------------------------------------
# Tabular construction
# ---------------------------------------------------------------------------
def build_sentence_frame(sentences: List[Dict]) -> pd.DataFrame:
    """One row per sentence, with binary aspect columns."""
    rows = []
    for sentence in sentences:
        aspects = sorted({c["aspect"] for c in sentence["categories"] if c["aspect"]})
        row = {
            "sentence_id": sentence["sentence_id"],
            "text": sentence["text"],
            "aspects": "|".join(aspects),
            "num_aspects": len(aspects),
        }
        for aspect in ASPECTS:
            row[f"aspect_{aspect}"] = int(aspect in aspects)
        rows.append(row)
    return pd.DataFrame(rows)


def build_sentiment_frame(
    sentences: List[Dict], include_conflict: bool = False
) -> pd.DataFrame:
    """One row per (sentence, aspect) annotation with a polarity label."""
    rows = []
    for sentence in sentences:
        for category in sentence["categories"]:
            aspect = category["aspect"]
            polarity = category["polarity"]
            if aspect not in ASPECTS:
                continue
            if polarity == "conflict" and not include_conflict:
                continue
            if polarity not in SENTIMENT_LABELS:
                continue
            rows.append(
                {
                    "sentence_id": sentence["sentence_id"],
                    "text": sentence["text"],
                    "aspect": aspect,
                    "label": polarity,
                    "label_id": SENTIMENT_LABELS.index(polarity),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Integrity checks and statistics
# ---------------------------------------------------------------------------
def dataset_statistics(
    train_sentences: List[Dict], test_sentences: List[Dict]
) -> Dict:
    """Compute descriptive statistics for data quality and documentation."""
    stats: Dict = {
        "source": "SemEval-2014 Task 4 (ABSA), restaurant domain",
        "canonical_aspects": ASPECTS,
        "sentiment_labels": SENTIMENT_LABELS,
        "conflict_policy": CONFLICT_POLICY,
        "splits": {},
    }

    for split, sentences in (("train", train_sentences), ("test", test_sentences)):
        texts = [s["text"] for s in sentences]
        category_polars = Counter(
            c["polarity"] for s in sentences for c in s["categories"]
        )
        category_aspects = Counter(
            c["aspect"] for s in sentences for c in s["categories"]
        )
        aspect_sentiment = Counter(
            (c["aspect"], c["polarity"])
            for s in sentences
            for c in s["categories"]
        )
        aspect_per_sentence = Counter(
            len({c["aspect"] for c in s["categories"]}) for s in sentences
        )
        token_lengths = [len(tokenize(t)) for t in texts]

        stats["splits"][split] = {
            "sentences": len(sentences),
            "empty_texts": sum(1 for t in texts if not t.strip()),
            "duplicate_texts": len(texts) - len(set(texts)),
            "aspect_annotations": sum(len(s["categories"]) for s in sentences),
            "aspect_term_annotations": sum(len(s["aspect_terms"]) for s in sentences),
            "polarity_distribution": dict(category_polars),
            "aspect_distribution": dict(category_aspects),
            "aspect_sentiment_distribution": {
                f"{a}/{p}": n for (a, p), n in sorted(aspect_sentiment.items())
            },
            "aspects_per_sentence": {
                str(k): v for k, v in sorted(aspect_per_sentence.items())
            },
            "mean_tokens": round(sum(token_lengths) / max(len(token_lengths), 1), 2),
            "max_tokens": max(token_lengths) if token_lengths else 0,
            "sentences_without_aspect": sum(1 for s in sentences if not s["categories"]),
        }

    return stats


def validate_frames(sentences: pd.DataFrame, sentiments: pd.DataFrame) -> Dict:
    """Run sanity checks and return a small validation report."""
    expected_aspect_cols = {f"aspect_{a}" for a in ASPECTS}
    report = {
        "sentence_rows": int(len(sentences)),
        "sentiment_rows": int(len(sentiments)),
        "missing_text": int(sentences["text"].isna().sum()),
        "empty_text": int((sentences["text"].str.strip() == "").sum()),
        "duplicate_texts": int(sentences["text"].duplicated().sum()),
        "missing_aspect_columns": sorted(expected_aspect_cols - set(sentences.columns)),
        "invalid_aspects": sorted(set(sentiments["aspect"]) - set(ASPECTS)),
        "invalid_labels": sorted(set(sentiments["label"]) - set(SENTIMENT_LABELS)),
        "rows_with_zero_aspects": int((sentences["num_aspects"] == 0).sum()),
    }
    if report["missing_aspect_columns"]:
        raise ValueError(f"Missing aspect columns: {report['missing_aspect_columns']}")
    if report["invalid_aspects"]:
        raise ValueError(f"Unknown aspects: {report['invalid_aspects']}")
    if report["invalid_labels"]:
        raise ValueError(f"Unknown sentiment labels: {report['invalid_labels']}")
    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def prepare_datasets(verbose: bool = True) -> Dict:
    """Parse the XML files, write processed CSVs and a statistics report."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train = parse_semeval_xml(TRAIN_XML)
    test = parse_semeval_xml(TEST_XML)

    outputs = {}
    for split, sentences in (("train", train), ("test", test)):
        sentence_df = build_sentence_frame(sentences)
        sentiment_df = build_sentiment_frame(sentences)

        # Remove exact duplicate training texts (keep the first annotation).
        # The official test set is kept untouched for a fair evaluation.
        if split == "train":
            before = len(sentence_df)
            sentence_df = sentence_df.drop_duplicates(subset="text", keep="first")
            sentiment_df = sentiment_df[
                sentiment_df["sentence_id"].isin(sentence_df["sentence_id"])
            ]
            removed = before - len(sentence_df)
        else:
            removed = 0

        sentence_path = os.path.join(PROCESSED_DIR, f"{split}_sentences.csv")
        sentiment_path = os.path.join(PROCESSED_DIR, f"{split}_aspect_sentiment.csv")
        sentence_df.to_csv(sentence_path, index=False)
        sentiment_df.to_csv(sentiment_path, index=False)

        outputs[split] = {
            "sentences_path": sentence_path,
            "sentiment_path": sentiment_path,
            "sentences": len(sentence_df),
            "sentiment_rows": len(sentiment_df),
            "duplicates_removed": removed,
        }

    stats = dataset_statistics(train, test)
    stats["processed"] = outputs
    stats["validation"] = {
        "train": validate_frames(
            pd.read_csv(outputs["train"]["sentences_path"]),
            pd.read_csv(outputs["train"]["sentiment_path"]),
        ),
        "test": validate_frames(
            pd.read_csv(outputs["test"]["sentences_path"]),
            pd.read_csv(outputs["test"]["sentiment_path"]),
        ),
    }

    stats_path = os.path.join(PROCESSED_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    with open(os.path.join(REPORTS_DIR, "dataset_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    if verbose:
        for split in ("train", "test"):
            info = outputs[split]
            print(
                f"[DATA] {split}: {info['sentences']} sentences, "
                f"{info['sentiment_rows']} aspect-sentiment rows "
                f"({info['duplicates_removed']} duplicate texts removed)"
            )
        print(f"[DATA] Statistics written to {stats_path}")

    return stats


def load_sentences(split: str) -> pd.DataFrame:
    """Load the processed sentence-level frame for ``train`` or ``test``."""
    path = os.path.join(PROCESSED_DIR, f"{split}_sentences.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run: python -m scripts.prepare_data"
        )
    return pd.read_csv(path)


def load_aspect_sentiment(split: str) -> pd.DataFrame:
    """Load the processed aspect-sentiment frame for ``train`` or ``test``."""
    path = os.path.join(PROCESSED_DIR, f"{split}_aspect_sentiment.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run: python -m scripts.prepare_data"
        )
    return pd.read_csv(path)


def load_dataset_stats() -> Dict:
    """Load the dataset statistics JSON (useful for the Streamlit app)."""
    path = os.path.join(PROCESSED_DIR, "dataset_stats.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Dataset statistics not found. Run: python -m scripts.prepare_data"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
