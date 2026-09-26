from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)
from torch.utils.data import DataLoader
from torch.nn import functional as F


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": precision,
            "recall_macro": recall, "f1_macro": f1,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(10)).tolist()}


@torch.no_grad()
def predict_torch(model: torch.nn.Module, loader: DataLoader,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    ys, predictions = [], []
    for images, targets in loader:
        predictions.append(model(images.to(device)).argmax(1).cpu().numpy())
        ys.append(targets.numpy())
    return np.concatenate(ys), np.concatenate(predictions)


@torch.no_grad()
def extract_embeddings(encoder: torch.nn.Module, loader: DataLoader,
                       device: torch.device, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval().to(device)
    xs, ys, seen = [], [], 0
    for images, targets in loader:
        xs.append(encoder(images.to(device)).cpu().numpy())
        ys.append(targets.numpy())
        seen += len(targets)
        if limit and seen >= limit:
            break
    x, y = np.concatenate(xs), np.concatenate(ys)
    return (x[:limit], y[:limit]) if limit else (x, y)


@torch.no_grad()
def embedding_similarity_samples(encoder: torch.nn.Module, loader: DataLoader,
                                 device: torch.device, max_batches: int = 8) -> dict[str, np.ndarray]:
    """Cosine similarities for matched views and deliberately mismatched views."""
    encoder.eval().to(device)
    positives, negatives = [], []
    for batch_number, batch in enumerate(loader):
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("similarity loader must expose exactly two unlabeled views")
        z1 = F.normalize(encoder(batch[0].to(device)), dim=1)
        z2 = F.normalize(encoder(batch[1].to(device)), dim=1)
        positives.append((z1 * z2).sum(1).cpu().numpy())
        negatives.append((z1 * torch.roll(z2, shifts=1, dims=0)).sum(1).cpu().numpy())
        if batch_number + 1 >= max_batches:
            break
    return {"positive": np.concatenate(positives), "negative": np.concatenate(negatives)}


def train_logistic(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                   max_iter: int, seed: int) -> tuple[dict, LogisticRegression]:
    def flatten(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for images, targets in loader:
            x.append(images.numpy().reshape(len(images), -1)); y.append(targets.numpy())
        return np.concatenate(x), np.concatenate(y)

    train_x, train_y = flatten(train_loader)
    val_x, val_y = flatten(val_loader)
    started = time.perf_counter()
    candidates = [0.1, 1.0, 10.0]
    best_model, best_score, best_c = None, -1.0, None
    for c in candidates:
        candidate = LogisticRegression(C=c, max_iter=max_iter, solver="saga", tol=1e-2,
                                       random_state=seed, n_jobs=-1)
        candidate.fit(train_x, train_y)
        score = candidate.score(val_x, val_y)
        if score > best_score:
            best_model, best_score, best_c = candidate, score, c
    assert best_model is not None
    # Access the held-out test set only after C has been fixed on validation.
    test_x, test_y = flatten(test_loader)
    metrics = metric_dict(test_y, best_model.predict(test_x))
    metrics.update({"best_C": best_c, "val_accuracy": best_score,
                    "runtime_seconds": time.perf_counter() - started})
    return metrics, best_model
