"""Central configuration for the ABSA project.

All paths are derived from the project root so that scripts work regardless of
the current working directory.  The configuration is intentionally lightweight
(no external dependency) because it is imported by every module, including the
Streamlit application.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
DEV_SIZE = 0.15  # fraction of the official training split used for tuning

# ---------------------------------------------------------------------------
# Canonical aspect categories (SemEval-2014 Task 4, restaurant domain)
# ---------------------------------------------------------------------------
ASPECTS = ["food", "service", "price", "ambience", "miscellaneous"]

ASPECT_DISPLAY = {
    "food": "Food",
    "service": "Service",
    "price": "Price",
    "ambience": "Ambience",
    "miscellaneous": "Overall / Miscellaneous",
}

ASPECT_DESCRIPTIONS = {
    "food": "Quality, taste, presentation, menu variety and drinks.",
    "service": "Staff attitude, speed, attentiveness and reservations.",
    "price": "Value for money, cost and pricing fairness.",
    "ambience": "Atmosphere, decor, noise, lighting and comfort.",
    "miscellaneous": "Holistic impressions and anything not covered above.",
}

# Polarity labels used throughout the project.
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# SemEval annotations contain a fourth label, "conflict", for genuinely mixed
# opinions.  It is excluded from the three-class sentiment task (documented in
# the report) because it is not a coherent polarity class.
CONFLICT_POLICY = "exclude"

# ---------------------------------------------------------------------------
# Seed lexicons used by the PLSA + TF-ICF aspect-expansion component.
# These are *seeds*, not an exhaustive dictionary: the unsupervised component
# expands them with corpus-specific terms (see src/aspect_lexicon.py).
# ---------------------------------------------------------------------------
ASPECT_SEED_TERMS = {
    "food": [
        "food", "meal", "dish", "dishes", "menu", "pizza", "pasta", "sushi",
        "burger", "steak", "salad", "dessert", "appetizer", "starter", "wine",
        "beer", "drink", "drinks", "coffee", "tea", "cocktail", "bread",
        "sauce", "cheese", "chicken", "fish", "beef", "pork", "rice", "soup",
        "sandwich", "taste", "tasty", "flavor", "flavour", "portion", "plate",
        "ingredient", "cooked", "fresh", "delicious", "bland", "seasoning",
        "seafood", "vegetarian", "vegan", "gluten", "spicy", "sweet", "salty",
        "breakfast", "lunch", "dinner", "brunch", "pancake", "egg", "eggs",
        "fries", "sushi", "noodle", "noodles", "curry", "grill", "grilled",
        "chocolate", "cake", "cream", "ice", "juice", "soda", "water",
    ],
    "service": [
        "service", "staff", "waiter", "waitress", "server", "waiters",
        "waitresses", "host", "hostess", "manager", "bartender", "employee",
        "reservation", "wait", "waiting", "attitude", "friendly", "rude",
        "helpful", "unhelpful", "attentive", "inattentive", "polite",
        "professional", "courteous", "slow", "quick", "prompt", "response",
        "greeting", "seated", "order", "delivery", "takeout", "check", "bill",
        "recommendation", "recommended", "knowledgeable", "apology",
    ],
    "price": [
        "price", "prices", "pricing", "cost", "costly", "value", "expensive",
        "cheap", "cheaper", "worth", "bill", "money", "overpriced",
        "underpriced", "affordable", "reasonable", "budget", "dollars",
        "dollar", "euro", "paid", "pay", "charge", "charged", "fee", "deal",
        "discount", "pricey", "inexpensive", "fair",
    ],
    "ambience": [
        "ambience", "ambiance", "atmosphere", "decor", "decoration", "music",
        "noise", "noisy", "quiet", "lighting", "lights", "seating", "seats",
        "chairs", "tables", "crowded", "cozy", "cosy", "romantic", "interior",
        "design", "view", "window", "patio", "outdoor", "indoor", "space",
        "vibe", "environment", "clean", "dirty", "comfortable", "uncomfortable",
        "loud", "smell", "bathroom", "restroom", "air", "temperature", "warm",
        "cold", "candle", "bar", "counter", "room", "layout",
    ],
    "miscellaneous": [
        "restaurant", "place", "experience", "overall", "recommend",
        "recommended", "return", "visit", "everything", "nothing", "general",
        "impression", "favorite", "favourite", "best", "worst", "worth",
        "amazing", "terrible", "excellent", "awful", "good", "bad", "great",
        "poor", "love", "like", "enjoy", "disappointed", "satisfied",
    ],
}

# Kept for compatibility with the original repository's semantic-similarity
# module: the "miscellaneous" bucket should not dominate aspect matching.
LIKE_SIMILARITY_PENALTY = 0.55

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
SEMEVAL_DIR = os.path.join(RAW_DIR, "semeval2014")
TRAIN_XML = os.path.join(SEMEVAL_DIR, "Restaurants_Train_v2.xml")
TEST_XML = os.path.join(SEMEVAL_DIR, "Restaurants_Test_Gold.xml")

MODELS_DIR = os.path.join(BASE_DIR, "models")
LSTM_DIR = os.path.join(MODELS_DIR, "lstm")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

GLOVE_PATH = os.path.join(EXTERNAL_DIR, "glove.6B.50d.txt")
NLTK_DATA_DIR = os.path.join(EXTERNAL_DIR, "nltk_data")
os.environ.setdefault("NLTK_DATA", NLTK_DATA_DIR)

# Artefacts produced by the training scripts.
ASPECT_DETECTOR_PATH = os.path.join(MODELS_DIR, "aspect_detector.joblib")
SENTIMENT_MODELS_PATH = os.path.join(MODELS_DIR, "sentiment_models.joblib")
LEXICON_PATH = os.path.join(MODELS_DIR, "aspect_lexicon.json")
PLSA_PATH = os.path.join(MODELS_DIR, "plsa_model.joblib")
LSTM_META_PATH = os.path.join(LSTM_DIR, "meta.json")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# Text preprocessing / representation
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 64
MIN_TOKEN_FREQ = 1
MAX_VOCAB = 20000
EMBED_DIM = 50  # GloVe 6B.50d (optional; random initialisation is used if absent)

TFIDF_WORD_NGRAMS = (1, 2)
TFIDF_CHAR_NGRAMS = (3, 5)
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 60000

# ---------------------------------------------------------------------------
# PLSA + TF-ICF
# ---------------------------------------------------------------------------
PLSA_TOPICS = 20
PLSA_MAX_ITER = 30
PLSA_MAX_VOCAB = 3000
PLSA_MIN_DF = 3
TFICF_TOP_TERMS_PER_TOPIC = 30
TFICF_EXPANSION_TERMS_PER_ASPECT = 60

# ---------------------------------------------------------------------------
# Aspect detection / sentiment training
# ---------------------------------------------------------------------------
ASPECT_THRESHOLD = 0.50
LR_C_GRID = [1.0, 4.0, 10.0]
LSTM_HIDDEN_DIM = 96
LSTM_DROPOUT = 0.35
LSTM_EPOCHS = 18
LSTM_BATCH_SIZE = 32
LSTM_LR = 1e-3
LSTM_PATIENCE = 3
