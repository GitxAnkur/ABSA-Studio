"""Text representation: TF-IDF feature extraction.

Word n-grams capture lexical sentiment cues ("not good", "great food"),
while character n-grams make the representation robust to spelling variation
and morphology ("delicious", "deliciously", "delish").
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .config import (
    TFIDF_CHAR_NGRAMS,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_WORD_NGRAMS,
)
from .preprocessing import normalize_text, tokenize


def build_vectorizer(
    word_ngrams: tuple = TFIDF_WORD_NGRAMS,
    char_ngrams: tuple = TFIDF_CHAR_NGRAMS,
    min_df: int = TFIDF_MIN_DF,
    max_features: int = TFIDF_MAX_FEATURES,
) -> FeatureUnion:
    """Build a word + character TF-IDF feature union.

    The vectorizer must be fitted on training text only to prevent leakage.
    """
    word_vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=word_ngrams,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents=None,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        preprocessor=normalize_text,
        lowercase=False,
        ngram_range=char_ngrams,
        min_df=min_df,
        sublinear_tf=True,
        max_features=max_features,
    )
    return FeatureUnion(
        [
            ("word", word_vectorizer),
            ("char", char_vectorizer),
        ]
    )


def ensure_writable(matrix):
    """Return a writable CSR copy of a sparse matrix.

    ``FeatureUnion`` may produce read-only sparse buffers under recent
    NumPy/SciPy releases, which makes scikit-learn's internal ``np.max`` check
    fail with ``WRITEBACKIFCOPY base is read-only``.  A compact CSR copy with
    owned buffers avoids the issue.
    """
    matrix = matrix.tocsr(copy=True)
    matrix.data = matrix.data.copy()
    matrix.indices = matrix.indices.copy()
    matrix.indptr = matrix.indptr.copy()
    return matrix
