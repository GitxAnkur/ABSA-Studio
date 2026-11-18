"""Unsupervised aspect lexicon induction with pLSA + TF-ICF.

Pipeline
--------
1. Fit pLSA on the training reviews (stop-words removed, lemmatised).
2. Cluster documents by their dominant topic.
3. Score terms in each cluster with TF-ICF.
4. Measure the cosine similarity between each aspect's seed lexicon and each
   topic's word distribution.
5. Transfer the strongest TF-ICF terms from aligned topics into the aspect
   lexicon.

The resulting lexicon powers a transparent, unsupervised baseline for aspect
detection.  It also serves as a qualitative window into what pLSA learned.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np

from .config import (
    ASPECT_SEED_TERMS,
    ASPECTS,
    PLSA_MAX_ITER,
    PLSA_MAX_VOCAB,
    PLSA_MIN_DF,
    PLSA_TOPICS,
    RANDOM_STATE,
    TFICF_EXPANSION_TERMS_PER_ASPECT,
    TFICF_TOP_TERMS_PER_TOPIC,
)
from .plsa import PLSA
from .preprocessing import tokenize
from .tf_icf import normalise_scores, tf_icf, top_terms


# ---------------------------------------------------------------------------
# Lexicon induction
# ---------------------------------------------------------------------------
def build_aspect_lexicon(
    texts: Sequence[str],
    n_topics: int = PLSA_TOPICS,
    max_iter: int = PLSA_MAX_ITER,
    expansion_size: int = TFICF_EXPANSION_TERMS_PER_ASPECT,
    verbose: bool = True,
    plsa: PLSA | None = None,
) -> Dict:
    """Induce an expanded lexicon for every aspect.

    If ``plsa`` is supplied it must already be fitted; otherwise a model is
    trained on ``texts``.  Returns a JSON-serialisable dictionary with seed
    terms, expanded terms, topic-word previews and aspect-topic similarities.
    """
    docs = [
        tokenize(text, remove_stopwords=True, lemmatize=True)
        for text in texts
    ]

    if plsa is None:
        plsa = PLSA(
            n_topics=n_topics,
            max_iter=max_iter,
            random_state=RANDOM_STATE,
            min_df=PLSA_MIN_DF,
            max_features=PLSA_MAX_VOCAB,
        ).fit(docs)
        fitted_here = True
    else:
        fitted_here = False

    if verbose:
        print(
            f"[PLSA] {'Fitted' if fitted_here else 'Using'} {n_topics} topics on "
            f"{len(docs)} documents with vocabulary size {len(plsa.vocab_)}."
        )

    # Hard cluster assignment by dominant topic.
    doc_topics = plsa.document_topic_matrix().argmax(axis=1)
    clusters: Dict[int, List[List[str]]] = {}
    for topic_id, tokens in zip(doc_topics, docs):
        clusters.setdefault(int(topic_id), []).append(tokens)

    tf_icf_scores = tf_icf(clusters)
    topic_terms = top_terms(tf_icf_scores, n=TFICF_TOP_TERMS_PER_TOPIC)

    # Aspect seed vectors in the PLSA vocabulary.
    vocab_index = {term: idx for idx, term in enumerate(plsa.vocab_)}
    topic_vectors = plsa.topic_word_vectors()  # (Z, V), L2-normalised

    aspect_seed_vectors = np.zeros((len(ASPECTS), len(plsa.vocab_)), dtype=float)
    for aspect_idx, aspect in enumerate(ASPECTS):
        for term in ASPECT_SEED_TERMS.get(aspect, []):
            for token in tokenize(term, remove_stopwords=True, lemmatize=True):
                idx = vocab_index.get(token)
                if idx is not None:
                    aspect_seed_vectors[aspect_idx, idx] = 1.0

    norms = np.linalg.norm(aspect_seed_vectors, axis=1, keepdims=True) + 1e-12
    aspect_seed_vectors = aspect_seed_vectors / norms
    similarity = aspect_seed_vectors @ topic_vectors.T  # (A, Z)

    lexicon: Dict[str, Dict] = {}
    for aspect_idx, aspect in enumerate(ASPECTS):
        seed_terms = sorted(
            {
                token
                for term in ASPECT_SEED_TERMS.get(aspect, [])
                for token in tokenize(term, remove_stopwords=True, lemmatize=True)
            }
        )

        aggregate: Dict[str, float] = {}
        topic_similarity = {}
        for topic_id in range(n_topics):
            sim = float(similarity[aspect_idx, topic_id])
            topic_similarity[str(topic_id)] = round(sim, 6)
            if sim <= 0:
                continue
            for term, score in topic_terms.get(topic_id, []):
                if term in seed_terms:
                    continue
                aggregate[term] = aggregate.get(term, 0.0) + sim * score

        expanded = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[
            :expansion_size
        ]

        lexicon[aspect] = {
            "seed_terms": seed_terms,
            "expanded_terms": [
                {"term": term, "weight": round(float(weight), 6)}
                for term, weight in expanded
            ],
            "topic_similarity": topic_similarity,
        }

    return {
        "method": "pLSA + TF-ICF aspect lexicon induction",
        "n_topics": n_topics,
        "n_documents": len(docs),
        "vocabulary_size": len(plsa.vocab_),
        "topics": {
            str(topic_id): [term for term, _ in terms]
            for topic_id, terms in topic_terms.items()
        },
        "aspect_topic_similarity": {
            aspect: lexicon[aspect]["topic_similarity"] for aspect in ASPECTS
        },
        "aspects": lexicon,
    }


def save_lexicon(lexicon: Dict, path: str) -> None:
    """Persist the induced lexicon as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(lexicon, handle, indent=2)


def load_lexicon(path: str) -> Dict:
    """Load a lexicon JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Aspect lexicon not found at {path}. Run the training pipeline first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def lexicon_term_weights(lexicon: Dict, aspect: str) -> Dict[str, float]:
    """Return a flat term -> weight mapping for one aspect."""
    if aspect not in lexicon.get("aspects", {}):
        return {}
    entry = lexicon["aspects"][aspect]
    weights: Dict[str, float] = {term: 1.0 for term in entry.get("seed_terms", [])}
    expanded = entry.get("expanded_terms", [])
    if expanded:
        maximum = max(item["weight"] for item in expanded) or 1.0
        for item in expanded:
            term = item["term"]
            weight = 0.6 * (item["weight"] / maximum)
            weights[term] = max(weights.get(term, 0.0), weight)
    return weights


# ---------------------------------------------------------------------------
# Baseline aspect detector
# ---------------------------------------------------------------------------
class LexiconAspectDetector:
    """Weighted lexicon-hit baseline for multi-label aspect detection.

    A sentence receives, for each aspect, the average weight of matched
    lexicon terms (normalised by the square root of the sentence length).
    Per-aspect thresholds are tuned on a development split.
    """

    def __init__(self, lexicon: Dict, thresholds: Dict[str, float] | None = None):
        self.lexicon = lexicon
        self.weights: Dict[str, Dict[str, float]] = {
            aspect: lexicon_term_weights(lexicon, aspect) for aspect in ASPECTS
        }
        self.thresholds = thresholds or {aspect: 0.25 for aspect in ASPECTS}
        self._normalised = {
            aspect: normalise_scores(weights)
            for aspect, weights in self.weights.items()
        }

    def score(self, text: str) -> Dict[str, float]:
        """Return normalised lexicon scores for every aspect."""
        tokens = tokenize(text)
        if not tokens:
            return {aspect: 0.0 for aspect in ASPECTS}
        token_set = set(tokens)
        length_penalty = np.sqrt(max(len(tokens), 1))

        scores = {}
        for aspect in ASPECTS:
            weights = self.weights.get(aspect, {})
            hits = [weights[token] for token in token_set if token in weights]
            scores[aspect] = float(sum(hits) / length_penalty)
        return scores

    def predict(self, text: str) -> Dict[str, int]:
        """Return binary aspect predictions."""
        scores = self.score(text)
        return {
            aspect: int(scores[aspect] >= self.thresholds.get(aspect, 0.25))
            for aspect in ASPECTS
        }

    def probabilities(self, text: str) -> Dict[str, float]:
        """Map scores to a 0-1 scale for display purposes only."""
        scores = self.score(text)
        probabilities = {}
        for aspect in ASPECTS:
            threshold = max(self.thresholds.get(aspect, 0.25), 1e-6)
            # Logistic squashing around the tuned threshold.
            probabilities[aspect] = float(
                1.0 / (1.0 + np.exp(-6.0 * (scores[aspect] - threshold) / threshold))
            )
        return probabilities

    def jsonable(self) -> Dict:
        return {"thresholds": self.thresholds}


def tune_thresholds(
    detector: LexiconAspectDetector,
    texts: Sequence[str],
    labels: np.ndarray,
    grid: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Tune one threshold per aspect to maximise per-aspect F1 on dev data."""
    from sklearn.metrics import f1_score

    if grid is None:
        grid = np.linspace(0.0, 1.2, 49)

    scores = [detector.score(text) for text in texts]
    thresholds: Dict[str, float] = {}

    for aspect_idx, aspect in enumerate(ASPECTS):
        y_true = labels[:, aspect_idx]
        if y_true.sum() == 0:
            thresholds[aspect] = 0.25
            continue
        best_threshold, best_f1 = 0.25, -1.0
        for threshold in grid:
            y_pred = [int(row[aspect] >= threshold) for row in scores]
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1, best_threshold = score, float(threshold)
        thresholds[aspect] = round(best_threshold, 4)

    detector.thresholds = thresholds
    return thresholds
