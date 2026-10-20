"""Learned re-ranker: a tiny logistic-regression model over embedding-similarity
features, trained on the labels paperBoy logs.

Why: the embedding ranker scores a paper by its single best cosine to any topic.
That is blunt - a paper that broadly shares ML vocabulary with every topic (high
mean similarity) looks the same as one that sharply matches a single topic (high
max, low mean), yet usually only the second is what you want. This re-ranker
learns, from the keep/drop + good/bad labels, how the *distribution* of a paper's
per-topic similarities maps to relevance, and replaces the raw cosine as the
ranking score.

Deliberately pure-Python and dependency-free (like rank.py): the feature vector
is 5-dim and the training set is hundreds of examples, so hand-rolled gradient
descent trains in milliseconds and inference is a dot product. No numpy, no
scikit-learn, no GPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# The feature vector is summary stats of a paper's per-topic cosine similarities.
FEATURE_NAMES = ("max", "mean", "min", "std", "gap")


def _sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def extract_features(cosines: list[float]) -> list[float]:
    """Summary stats of a paper's per-topic cosine similarities -> fixed 5-vector.

    Fixed length regardless of topic count, so the model is independent of how many
    topics are configured. `gap` (max minus second-highest) is large when one topic
    matches far more than the rest - a sharp, specific match rather than broad
    vocabulary overlap.
    """
    if not cosines:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    ordered = sorted(cosines, reverse=True)
    mx, mn = ordered[0], ordered[-1]
    mean = sum(ordered) / len(ordered)
    std = math.sqrt(sum((c - mean) ** 2 for c in ordered) / len(ordered))
    gap = (mx - ordered[1]) if len(ordered) > 1 else 0.0
    return [mx, mean, mn, std, gap]


@dataclass
class Reranker:
    """A trained logistic-regression relevance scorer over `extract_features`."""

    weights: list[float]
    bias: float
    mean: list[float]  # per-feature standardization (fit on the training set)
    std: list[float]
    n_train: int = 0
    created: str = ""

    def score(self, features: list[float]) -> float:
        """Relevance probability in [0, 1] for a feature vector."""
        z = self.bias
        for w, x, m, s in zip(self.weights, features, self.mean, self.std):
            z += w * ((x - m) / s if s else 0.0)
        return _sigmoid(z)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Reranker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = ("weights", "bias", "mean", "std", "n_train", "created")
        return cls(**{k: data[k] for k in fields if k in data})


def _standardize(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    """Per-feature mean and std over the training rows (std floored to 1.0 to avoid
    divide-by-zero on a constant feature)."""
    n, dim = len(rows), len(rows[0])
    mean = [sum(r[j] for r in rows) / n for j in range(dim)]
    std = []
    for j in range(dim):
        var = sum((r[j] - mean[j]) ** 2 for r in rows) / n
        std.append(math.sqrt(var) or 1.0)
    return mean, std


def train_reranker(
    examples: list[tuple[list[float], int]],
    *,
    epochs: int = 800,
    lr: float = 0.3,
    l2: float = 1e-3,
) -> Reranker:
    """Fit logistic regression on (features, label) pairs via batch gradient descent.

    Labels are 1 (relevant) / 0 (not). Features are standardized first so a single
    learning rate works across features on different scales. L2 regularization keeps
    weights small, which matters when labels are scarce.
    """
    if not examples:
        raise ValueError("no training examples")

    X = [f for f, _ in examples]
    y = [lbl for _, lbl in examples]
    mean, std = _standardize(X)
    Xn = [[(row[j] - mean[j]) / std[j] for j in range(len(row))] for row in X]

    dim, n = len(Xn[0]), len(Xn)
    w = [0.0] * dim
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * dim
        gb = 0.0
        for xi, yi in zip(Xn, y):
            p = _sigmoid(b + sum(w[j] * xi[j] for j in range(dim)))
            err = p - yi
            for j in range(dim):
                gw[j] += err * xi[j]
            gb += err
        for j in range(dim):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * (gb / n)

    return Reranker(
        weights=w,
        bias=b,
        mean=mean,
        std=std,
        n_train=n,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
