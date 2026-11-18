"""Text normalisation and tokenisation utilities.

Design choices
--------------
* Lowercasing and unicode normalisation reduce sparsity.
* HTML/URL removal is needed because review text scraped from the web often
  contains markup.
* Contractions are expanded *before* punctuation removal so that negation is
  preserved (``wasn't`` -> ``was not``).  Negation is crucial for sentiment.
* Punctuation is removed but sentence-final ``!`` is converted to a token
  because emphasis carries sentiment (e.g. "amazing!!!").
* Stop-word removal and lemmatisation are optional.  They are used for topic
  modelling (PLSA), where function words add noise, but *not* for the
  supervised TF-IDF/LSTM models, where negation and function words carry
  sentiment information.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable, List

import nltk

from .config import NLTK_DATA_DIR

# Make the project-local NLTK data directory discoverable.
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# ---------------------------------------------------------------------------
# Contractions and negation
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "can not",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'll": "it will", "it's": "it is", "let's": "let us",
    "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what's": "what is", "where's": "where is",
    "who's": "who is", "won't": "will not", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will", "you're": "you are",
    "you've": "you have", "y'all": "you all", "cannot": "can not",
}

NEGATIONS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "nowhere", "hardly", "barely", "scarcely", "without", "cannot",
}

# Intensifiers are kept when stop-word removal is enabled.
INTENSIFIERS = {"very", "really", "extremely", "so", "too", "quite", "highly", "super"}

# Small fallback stop-word list used only if the NLTK corpus is unavailable.
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "than", "s", "t", "can", "will", "just", "should", "now", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "could", "ought",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "as", "until",
    "while", "because", "although", "though", "since", "unless", "also",
}


@lru_cache(maxsize=1)
def _stopwords() -> frozenset:
    """English stop-words minus negation and intensifier cues."""
    try:
        from nltk.corpus import stopwords

        words = set(stopwords.words("english"))
    except Exception:
        words = set(_FALLBACK_STOPWORDS)
    words -= NEGATIONS
    words -= INTENSIFIERS
    return frozenset(words)


@lru_cache(maxsize=1)
def _lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")  # triggers lazy corpus loading
        return lemmatizer
    except Exception:
        return None


_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u02bc]")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_WS_RE = re.compile(r"\s+")


def _expand_contractions(text: str) -> str:
    # Replace longest keys first so "won't" is not partially matched.
    for contraction in sorted(CONTRACTIONS, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(contraction) + r"\b", re.IGNORECASE)
        text = pattern.sub(CONTRACTIONS[contraction], text)
    return text


def normalize_text(text: str) -> str:
    """Clean and normalise a raw review.

    Returns a lowercase string with URLs/HTML removed, contractions expanded,
    punctuation converted to whitespace (except emphasis markers) and unicode
    normalised.
    """
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _APOSTROPHE_RE.sub("'", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _expand_contractions(text)
    text = text.lower()
    # Mark strong emphasis, then drop all remaining punctuation.
    text = re.sub(r"!+", " exclamationmark ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(
    text: str,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
) -> List[str]:
    """Tokenise normalised text into word tokens.

    ``remove_stopwords`` and ``lemmatize`` default to False because negation
    and inflection are useful for sentiment classification.  The topic-model
    pipeline enables them explicitly.
    """
    normalized = normalize_text(text)
    tokens = _TOKEN_RE.findall(normalized)
    # Normalise the explicit emphasis marker.
    tokens = ["!" if tok == "exclamationmark" else tok for tok in tokens]

    if remove_stopwords:
        stop = _stopwords()
        tokens = [tok for tok in tokens if tok not in stop]

    if lemmatize:
        lemmatizer = _lemmatizer()
        if lemmatizer is not None:
            tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    return [tok for tok in tokens if tok]


def clean_text(text: str) -> str:
    """Return the normalised text as a whitespace-joined string."""
    return " ".join(tokenize(text))


def preprocess_corpus(
    texts: Iterable[str],
    remove_stopwords: bool = False,
    lemmatize: bool = False,
) -> List[List[str]]:
    """Tokenise an iterable of reviews."""
    return [
        tokenize(text, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
        for text in texts
    ]


def is_valid_text(text: str, min_tokens: int = 1) -> bool:
    """Basic input validation used by the app and inference engine."""
    if not isinstance(text, str) or not text.strip():
        return False
    return len(tokenize(text)) >= min_tokens
