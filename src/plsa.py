"""Probabilistic Latent Semantic Analysis (pLSA) with EM.

This is a compact, dependency-light implementation of the model introduced by
Hofmann (1999).  Documents are represented as bags of words.  The model learns

    P(w | z)   -- word distribution of each latent topic
    P(z | d)   -- topic mixture of each document

using Expectation-Maximisation on the observed word counts.

The implementation operates only on the non-zero entries of the
document-term matrix, which keeps training on a few thousand short reviews
fast on a CPU.

Reference:
    Hofmann, T. (1999). Probabilistic Latent Semantic Analysis.
    Proceedings of UAI 1999.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def _token_analyzer(tokens):
    """Module-level analyzer so that fitted models remain picklable."""
    return list(tokens)


class PLSA:
    """Probabilistic Latent Semantic Analysis.

    Parameters
    ----------
    n_topics:
        Number of latent topics ``Z``.
    max_iter:
        Number of EM iterations during training.
    random_state:
        Seed for reproducible initialisation.
    min_df, max_features:
        Vocabulary constraints passed to ``CountVectorizer``.
    """

    def __init__(
        self,
        n_topics: int = 20,
        max_iter: int = 30,
        random_state: int = 42,
        min_df: int = 3,
        max_features: int = 3000,
    ) -> None:
        self.n_topics = int(n_topics)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.min_df = int(min_df)
        self.max_features = int(max_features)

        self.vectorizer: CountVectorizer | None = None
        self.vocab_: List[str] = []
        self.P_w_z_: np.ndarray | None = None
        self.P_z_d_: np.ndarray | None = None
        self.log_likelihood_: List[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, docs: Sequence[Sequence[str]]) -> "PLSA":
        """Fit pLSA on a corpus of tokenised documents."""
        if len(docs) == 0:
            raise ValueError("PLSA.fit received an empty corpus.")

        self.vectorizer = CountVectorizer(
            analyzer=_token_analyzer,
            lowercase=False,
            min_df=self.min_df,
            max_features=self.max_features,
            dtype=np.float64,
        )
        X = self.vectorizer.fit_transform(docs)
        self.vocab_ = list(self.vectorizer.get_feature_names_out())

        if X.shape[1] == 0:
            raise ValueError(
                "PLSA vocabulary is empty. Lower min_df or provide more text."
            )

        n_docs, n_vocab = X.shape
        n_topics = self.n_topics
        rng = np.random.default_rng(self.random_state)

        # Random but normalised initial parameters.
        P_w_z = rng.random((n_topics, n_vocab)) + 1e-9
        P_w_z /= P_w_z.sum(axis=1, keepdims=True)
        P_z_d = rng.random((n_docs, n_topics)) + 1e-9
        P_z_d /= P_z_d.sum(axis=1, keepdims=True)

        X = X.tocsr()
        self.log_likelihood_ = []

        for _ in range(self.max_iter):
            P_w_z_new = np.zeros_like(P_w_z)
            P_z_d_new = np.zeros_like(P_z_d)
            log_likelihood = 0.0

            for d in range(n_docs):
                start, end = X.indptr[d], X.indptr[d + 1]
                word_idx = X.indices[start:end]
                counts = X.data[start:end]
                if counts.sum() <= 0:
                    continue

                # E-step: posterior over topics for each observed word.
                posterior = P_w_z[:, word_idx] * P_z_d[d][:, None]
                normaliser = posterior.sum(axis=0, keepdims=True) + 1e-12
                posterior /= normaliser

                # M-step sufficient statistics.
                P_w_z_new[:, word_idx] += posterior * counts[None, :]
                P_z_d_new[d] = posterior.sum(axis=1)

                # Observed-data log-likelihood for monitoring convergence.
                log_likelihood += float(np.sum(counts * np.log(normaliser[0])))

            P_w_z = P_w_z_new / (P_w_z_new.sum(axis=1, keepdims=True) + 1e-12)
            P_z_d = P_z_d_new / (P_z_d_new.sum(axis=1, keepdims=True) + 1e-12)
            self.log_likelihood_.append(log_likelihood)

        self.P_w_z_ = P_w_z
        self.P_z_d_ = P_z_d
        return self

    # ------------------------------------------------------------------
    # Inference for new documents
    # ------------------------------------------------------------------
    def transform(
        self, docs: Sequence[Sequence[str]], max_iter: int = 20
    ) -> np.ndarray:
        """Return topic activations ``P(z|d)`` for new documents.

        New documents are projected with the fixed ``P(w|z)`` learned during
        training ("fold-in"), so vocabulary items unseen in training are
        ignored.
        """
        if self.P_w_z_ is None or self.vectorizer is None:
            raise RuntimeError("PLSA model is not fitted.")

        X = self.vectorizer.transform(docs).tocsr()
        n_docs = X.shape[0]
        n_topics = self.n_topics

        # Uniform initialisation for the new documents.
        P_z_d = np.full((n_docs, n_topics), 1.0 / n_topics)

        for _ in range(max_iter):
            P_z_d_new = np.zeros_like(P_z_d)
            for d in range(n_docs):
                start, end = X.indptr[d], X.indptr[d + 1]
                word_idx = X.indices[start:end]
                counts = X.data[start:end]
                if counts.sum() <= 0:
                    P_z_d_new[d] = P_z_d[d]
                    continue
                posterior = self.P_w_z_[:, word_idx] * P_z_d[d][:, None]
                posterior /= posterior.sum(axis=0, keepdims=True) + 1e-12
                P_z_d_new[d] = posterior.sum(axis=1)
            P_z_d = P_z_d_new / (P_z_d_new.sum(axis=1, keepdims=True) + 1e-12)

        return P_z_d

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def top_words(self, n: int = 15) -> List[List[str]]:
        """Return the ``n`` most probable words for each topic."""
        if self.P_w_z_ is None:
            raise RuntimeError("PLSA model is not fitted.")
        topics = []
        for topic_idx in range(self.P_w_z_.shape[0]):
            order = np.argsort(self.P_w_z_[topic_idx])[::-1][:n]
            topics.append([self.vocab_[i] for i in order])
        return topics

    def topic_word_vectors(self) -> np.ndarray:
        """Return the normalised ``P(w|z)`` matrix (topics x vocabulary)."""
        if self.P_w_z_ is None:
            raise RuntimeError("PLSA model is not fitted.")
        vectors = self.P_w_z_.copy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def document_topic_matrix(self) -> np.ndarray:
        """Return ``P(z|d)`` for the training documents."""
        if self.P_z_d_ is None:
            raise RuntimeError("PLSA model is not fitted.")
        return self.P_z_d_

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "PLSA":
        import joblib

        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Object at {path} is not a PLSA model.")
        return model
