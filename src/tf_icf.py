"""TF-ICF term weighting for topic clusters.

TF-ICF (Term Frequency - Inverse Cluster Frequency) highlights terms that are
frequent inside one topic cluster but rare in the other clusters.  It is used
to expand the seed aspect lexicons with corpus-specific vocabulary discovered
by pLSA.

The formula follows the repository's original design:

    tf_icf(t, c) = tf(t, c) * log(N / cf(t))

where ``cf(t)`` is the number of clusters containing term ``t`` and ``N`` is
the number of clusters.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence


def tf_icf(
    clusters: Dict[int, List[Sequence[str]]],
) -> Dict[int, Dict[str, float]]:
    """Compute TF-ICF scores for every term in every cluster.

    Parameters
    ----------
    clusters:
        Mapping ``cluster_id -> list of tokenised documents``.

    Returns
    -------
    dict
        ``{cluster_id: {term: score}}``.
    """
    class_term_counts: Dict[int, Counter] = {}
    for cluster_id, docs in clusters.items():
        counts = Counter()
        for doc in docs:
            counts.update(set(doc))  # presence-based document frequency
        class_term_counts[cluster_id] = counts

    n_clusters = len(class_term_counts)
    if n_clusters == 0:
        return {}

    cluster_frequency: Counter = Counter()
    for counts in class_term_counts.values():
        for term in counts:
            cluster_frequency[term] += 1

    scores: Dict[int, Dict[str, float]] = defaultdict(dict)
    for cluster_id, counts in class_term_counts.items():
        for term, frequency in counts.items():
            cf = cluster_frequency[term]
            icf = math.log(n_clusters / cf) if cf > 0 else 0.0
            if icf > 0:
                scores[cluster_id][term] = float(frequency * icf)
    return dict(scores)


def top_terms(
    scores: Dict[int, Dict[str, float]], n: int = 30
) -> Dict[int, List[tuple]]:
    """Return the ``n`` highest-scoring terms per cluster."""
    return {
        cluster_id: sorted(term_scores.items(), key=lambda item: item[1], reverse=True)[:n]
        for cluster_id, term_scores in scores.items()
    }


def normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Scale a term-score dictionary so that its maximum is 1."""
    if not scores:
        return {}
    maximum = max(scores.values())
    if maximum <= 0:
        return {}
    return {term: value / maximum for term, value in scores.items()}
