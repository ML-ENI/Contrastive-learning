from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import nt_xent_loss


def labelled_batch_size(sample_count: int, maximum: int) -> int:
    """Keep roughly >=20 batches/epoch for small labelled budgets."""
    return min(sample_count, maximum, max(16, math.ceil(sample_count / 20)))


def epochs_for_updates(base_epochs: int, loader_length: int, min_updates: int) -> int:
    return max(base_epochs, math.ceil(min_updates / max(1, loader_length)))


def _classification_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                          device: torch.device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    # A frozen probe encoder must remain in eval mode so BatchNorm statistics do not change.
    if training and hasattr(model, "encoder") and not any(p.requires_grad for p in model.encoder.parameters()):
        model.encoder.eval()
    total_loss = correct = count = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            count += targets.size(0)
    return total_loss / count, correct / count


def train_classifier(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int, learning_rate: float, weight_decay: float,
                     patience: int, device: torch.device, checkpoint: str | Path,
                     min_updates: int = 0) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss, stale, best_state = float("inf"), 0, None
    history: list[dict] = []
    started = time.perf_counter()
    updates = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _classification_epoch(model, train_loader, criterion, device, optimizer)
        updates += len(train_loader)
        val_loss, val_acc = _classification_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc,
                        "optimizer_updates": updates})
        if val_loss < best_loss - 1e-5:
            best_loss, stale, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience and updates >= min_updates:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "history": history, "best_val_loss": best_loss}, checkpoint)
    history[-1]["total_seconds"] = time.perf_counter() - started
    return history


@torch.no_grad()
def _cache_features(encoder: nn.Module, source: DataLoader, device: torch.device) -> TensorDataset:
    encoder.eval().to(device)
    features, targets = [], []
    for images, labels in source:
        features.append(encoder(images.to(device)).cpu())
        targets.append(labels.clone())
    return TensorDataset(torch.cat(features), torch.cat(targets))


def train_linear_probe(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                       epochs: int, learning_rate: float, weight_decay: float,
                       patience: int, device: torch.device, checkpoint: str | Path,
                       min_updates: int = 0) -> list[dict]:
    """Fit the head on cached frozen features; mathematically identical, much faster on CPU."""
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise ValueError("linear-probe encoder must be frozen")
    started = time.perf_counter()
    cached_train = DataLoader(_cache_features(model.encoder, train_loader, device),
                              batch_size=train_loader.batch_size, shuffle=True)
    cached_val = DataLoader(_cache_features(model.encoder, val_loader, device),
                            batch_size=val_loader.batch_size, shuffle=False)
    head = model.classifier.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss, stale, best_state = float("inf"), 0, None
    history: list[dict] = []
    updates = 0
    for epoch in range(1, epochs + 1):
        values = []
        for current, training in ((cached_train, True), (cached_val, False)):
            head.train(training); total = correct = count = 0
            with torch.set_grad_enabled(training):
                for features, targets in current:
                    features, targets = features.to(device), targets.to(device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = head(features); loss = criterion(logits, targets)
                    if training:
                        loss.backward(); optimizer.step()
                    total += loss.item() * len(targets)
                    correct += (logits.argmax(1) == targets).sum().item(); count += len(targets)
            values.extend([total / count, correct / count])
        train_loss, train_acc, val_loss, val_acc = values
        updates += len(cached_train)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc,
                        "optimizer_updates": updates})
        if val_loss < best_loss - 1e-5:
            best_loss, stale, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience and updates >= min_updates:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint = Path(checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "history": history, "best_val_loss": best_loss}, checkpoint)
    history[-1]["total_seconds"] = time.perf_counter() - started
    return history


def _contrastive_epoch(model: nn.Module, loader: DataLoader, temperature: float,
                       device: torch.device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            # ContrastiveDataset exposes exactly two views. No label is accepted here.
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError("contrastive loader must return exactly two unlabeled views")
            view1, view2 = batch[0].to(device), batch[1].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = nt_xent_loss(model(view1), model(view2), temperature)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * view1.size(0)
            count += view1.size(0)
    return total_loss / count


def train_contrastive(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                      epochs: int, learning_rate: float, weight_decay: float,
                      temperature: float, patience: int, device: torch.device,
                      checkpoint: str | Path) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, stale, best_state = float("inf"), 0, None
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = _contrastive_epoch(model, train_loader, temperature, device, optimizer)
        val_loss = _contrastive_epoch(model, val_loader, temperature, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, stale, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "history": history, "best_val_loss": best_loss}, checkpoint)
    history[-1]["total_seconds"] = time.perf_counter() - started
    return history
