"""Aspect-specific sentiment classifier based on a BiLSTM.

Architecture
------------
    Embedding (GloVe-initialised, fine-tuned)
      -> BiLSTM (96 units per direction)
      -> masked mean-pooling + max-pooling
      -> Dropout
      -> Dense(128, ReLU)
      -> Softmax over {negative, neutral, positive}

Each aspect category receives its own classifier because the same sentence can
express different polarities for different aspects (e.g. "great food but rude
staff").  A shared vocabulary and embedding matrix is used across aspects.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import (
    EMBED_DIM,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN_DIM,
    LSTM_LR,
    LSTM_PATIENCE,
    MAX_SEQ_LEN,
    MAX_VOCAB,
    MIN_TOKEN_FREQ,
    RANDOM_STATE,
    SENTIMENT_LABELS,
)
from .preprocessing import tokenize


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class Vocabulary:
    """Simple word-to-index vocabulary with <pad> and <unk>."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self) -> None:
        self.token_to_idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx_to_token: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build(
        self,
        texts: Sequence[str],
        min_freq: int = MIN_TOKEN_FREQ,
        max_size: int = MAX_VOCAB,
    ) -> "Vocabulary":
        from collections import Counter

        counts: Counter = Counter()
        for text in texts:
            counts.update(tokenize(text))
        for token, frequency in counts.most_common():
            if frequency < min_freq:
                continue
            if len(self.token_to_idx) >= max_size:
                break
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
        return self

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        tokens = tokenize(text)[:max_len]
        ids = [self.token_to_idx.get(token, 1) for token in tokens]
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def to_dict(self) -> Dict:
        return {
            "token_to_idx": self.token_to_idx,
            "min_freq": MIN_TOKEN_FREQ,
            "max_size": MAX_VOCAB,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "Vocabulary":
        vocab = cls()
        token_to_idx = {str(k): int(v) for k, v in payload["token_to_idx"].items()}
        vocab.token_to_idx = token_to_idx
        vocab.idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        return vocab


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class EncodedDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocabulary):
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        ids = self.vocab.encode(self.texts[index])
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_matrix: np.ndarray | None = None,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        n_classes: int = len(SENTIMENT_LABELS),
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 4, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != 0).unsqueeze(-1).float()
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)

        # Masked mean pooling.
        summed = (outputs * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        mean_pooled = summed / lengths

        # Masked max pooling.
        masked_outputs = outputs.masked_fill(mask == 0, -1e9)
        max_pooled = masked_outputs.max(dim=1).values
        max_pooled = torch.where(
            torch.isfinite(max_pooled), max_pooled, torch.zeros_like(max_pooled)
        )

        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        hidden = self.relu(self.fc1(self.dropout(pooled)))
        return self.fc2(self.dropout(hidden))


# ---------------------------------------------------------------------------
# GloVe embeddings
# ---------------------------------------------------------------------------
def load_glove_subset(
    path: str, vocab: Vocabulary, embed_dim: int = EMBED_DIM
) -> Tuple[np.ndarray, float]:
    """Load only the vectors needed by ``vocab`` from a GloVe text file."""
    matrix = np.random.normal(0, 0.05, (len(vocab), embed_dim)).astype(np.float32)
    matrix[0] = 0.0
    if not os.path.exists(path):
        return matrix, 0.0

    wanted = set(vocab.token_to_idx.keys())
    found = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            if word not in wanted:
                continue
            idx = vocab.token_to_idx[word]
            matrix[idx] = np.asarray(parts[1:], dtype=np.float32)
            found += 1
    coverage = found / max(len(vocab), 1)
    return matrix, coverage


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    from sklearn.metrics import f1_score

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for input_ids, labels in loader:
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, len(SENTIMENT_LABELS)))
    labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=int)
    predictions = probs.argmax(axis=1) if len(probs) else np.zeros((0,), dtype=int)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return float(macro_f1), probs, labels


def _class_weight_tensor(labels: Sequence[int]) -> torch.Tensor:
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=np.asarray(labels)
    )
    class_weights = torch.ones(len(SENTIMENT_LABELS), dtype=torch.float32)
    for class_id, weight in zip(classes, weights):
        class_weights[int(class_id)] = float(weight)
    return class_weights


def train_aspect_model(
    aspect: str,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    dev_texts: Sequence[str],
    dev_labels: Sequence[int],
    vocab: Vocabulary,
    embedding_matrix: np.ndarray,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH_SIZE,
    verbose: bool = True,
    final_texts: Sequence[str] | None = None,
    final_labels: Sequence[int] | None = None,
) -> Tuple[SentimentLSTM, Dict]:
    """Train one aspect-specific BiLSTM with early stopping on dev macro-F1.

    When ``final_texts``/``final_labels`` are supplied, the model is refitted
    on those (usually the full aspect subset) for ``best_epoch`` epochs after
    early stopping, using half the original learning rate.  This recovers the
    15% of data held out for early stopping without changing the selected
    number of epochs.
    """
    from sklearn.utils.class_weight import compute_class_weight

    set_seed()

    train_dataset = EncodedDataset(train_texts, train_labels, vocab)
    dev_dataset = EncodedDataset(dev_texts, dev_labels, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)

    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_matrix=embedding_matrix,
        embed_dim=embedding_matrix.shape[1],
    )

    criterion = nn.CrossEntropyLoss(weight=_class_weight_tensor(train_labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    history = []
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, labels in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)

        dev_f1, _, _ = _evaluate(model, dev_loader)
        scheduler.step(dev_f1)
        average_loss = total_loss / max(len(train_dataset), 1)
        history.append(
            {"epoch": epoch, "train_loss": round(average_loss, 5), "dev_macro_f1": round(dev_f1, 5)}
        )
        if verbose:
            print(
                f"    epoch {epoch:02d} | loss={average_loss:.4f} | dev macro-F1={dev_f1:.4f}"
            )

        if dev_f1 > best_f1 + 1e-4:
            best_f1 = dev_f1
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= LSTM_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if final_texts is not None and final_labels is not None and best_epoch > 0:
        final_dataset = EncodedDataset(final_texts, final_labels, vocab)
        final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True)
        final_criterion = nn.CrossEntropyLoss(weight=_class_weight_tensor(final_labels))
        final_optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR * 0.5)
        model.train()
        for epoch in range(1, best_epoch + 1):
            for input_ids, labels in final_loader:
                final_optimizer.zero_grad()
                loss = final_criterion(model(input_ids), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                final_optimizer.step()
        if verbose:
            print(f"    refitted on {len(final_dataset)} samples for {best_epoch} epochs")

    metadata = {
        "aspect": aspect,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": round(float(best_f1), 6),
        "epochs_run": len(history),
        "train_size": len(train_dataset),
        "dev_size": len(dev_dataset),
        "refit_on_full": final_texts is not None,
        "history": history,
    }
    return model, metadata


def predict_proba(
    model: nn.Module, texts: Sequence[str], vocab: Vocabulary, batch_size: int = 64
) -> np.ndarray:
    """Return class probabilities for a list of texts."""
    model.eval()
    dataset = EncodedDataset(texts, [0] * len(texts), vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for input_ids, _ in loader:
            all_probs.append(torch.softmax(model(input_ids), dim=1).cpu().numpy())
    return np.concatenate(all_probs) if all_probs else np.zeros((0, len(SENTIMENT_LABELS)))


def save_model(
    model: nn.Module,
    vocab: Vocabulary,
    path: str,
    meta: Dict | None = None,
) -> None:
    """Save a model checkpoint together with its vocabulary."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab.to_dict(),
            "meta": meta or {},
            "embed_dim": model.embedding.embedding_dim,
            "hidden_dim": model.lstm.hidden_size,
            "n_classes": model.fc2.out_features,
        },
        path,
    )


def load_model(path: str) -> Tuple[SentimentLSTM, Vocabulary, Dict]:
    """Load a checkpoint saved by :func:`save_model`."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    vocab = Vocabulary.from_dict(checkpoint["vocab"])
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        n_classes=checkpoint["n_classes"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocab, checkpoint.get("meta", {})


def save_metadata(meta: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_metadata(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
