from __future__ import annotations

from pathlib import Path
import platform
import subprocess

# Some sandboxed macOS systems return a valid but empty system_profiler plist;
# Matplotlib 3.10 assumes an `_items` key and crashes while discovering fonts.
_original_check_output = subprocess.check_output
if platform.system() == "Darwin":
    def _safe_check_output(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and command and command[0] == "system_profiler":
            raise OSError("system_profiler font enumeration disabled in sandbox")
        return _original_check_output(command, *args, **kwargs)
    subprocess.check_output = _safe_check_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
subprocess.check_output = _original_check_output
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_eda(raw_dataset, raw_test_dataset, split_indices: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, indices in split_indices.items():
        source = raw_test_dataset if name == "test" else raw_dataset
        labels = np.asarray(source.targets)[indices]
        offsets = {"train": -0.25, "val": 0.0, "test": 0.25}
        axes[0].bar(np.arange(10) + offsets[name], np.bincount(labels, minlength=10),
                    width=0.24, label=name)
    axes[0].set(title="MNIST class distribution", xlabel="Digit", ylabel="Count")
    axes[0].legend()
    rng = np.random.default_rng(0)
    sample = rng.choice(split_indices["train"], size=16, replace=False)
    grid = np.zeros((4 * 28, 4 * 28), dtype=np.uint8)
    for k, idx in enumerate(sample):
        image, _ = raw_dataset[int(idx)]
        grid[(k // 4) * 28:(k // 4 + 1) * 28, (k % 4) * 28:(k % 4 + 1) * 28] = np.asarray(image)
    axes[1].imshow(grid, cmap="gray"); axes[1].set_title("Training examples"); axes[1].axis("off")
    axes[2].axis("off")
    axes[2].text(0, .8, "Image shape: 1 × 28 × 28\nClasses: 10 (digits 0–9)\n"
                 + "\n".join(f"{k}: {len(v):,}" for k, v in split_indices.items()), fontsize=12)
    _save(fig, output / "eda_mnist.png")


def plot_history(history: list[dict], path: Path, title: str, classification: bool) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2 if classification else 1, figsize=(10 if classification else 6, 4))
    axes = np.atleast_1d(axes)
    axes[0].plot(frame.epoch, frame.train_loss, marker="o", label="train")
    axes[0].plot(frame.epoch, frame.val_loss, marker="o", label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{title}: loss"); axes[0].legend()
    if classification:
        axes[1].plot(frame.epoch, frame.train_accuracy, marker="o", label="train")
        axes[1].plot(frame.epoch, frame.val_accuracy, marker="o", label="validation")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy", title=f"{title}: accuracy"); axes[1].legend()
    _save(fig, path)


def plot_comparison(results: pd.DataFrame, output: Path) -> None:
    methods = list(results.method.unique())
    budgets = sorted(results.label_fraction.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metric, ax in zip(["accuracy", "f1_macro"], axes):
        summary = results.groupby(["method", "label_fraction"])[metric].agg(["mean", "std"]).reset_index()
        for method in methods:
            part = summary[summary.method == method]
            errors = part["std"].fillna(0)
            ax.errorbar(part.label_fraction * 100, part["mean"], yerr=errors, marker="o", capsize=3,
                        label=method)
        ax.set(xlabel="Labelled training data (%)", ylabel=metric.replace("_", " ").title(),
               xticks=np.asarray(budgets) * 100); ax.grid(alpha=.25); ax.legend()
    _save(fig, output / "method_comparison.png")


def plot_confusion(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(title=title, xlabel="Predicted", ylabel="True", xticks=range(10), yticks=range(10))
    threshold = matrix.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if matrix[i, j] > threshold else "black")
    _save(fig, path)


def plot_pca(before: np.ndarray, after: np.ndarray, labels: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for embedding, title, ax in zip([before, after], ["Before pretraining", "After pretraining"], axes):
        points = PCA(n_components=2).fit_transform(embedding)
        scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=8, alpha=.7)
        ax.set(title=title, xlabel="PC1", ylabel="PC2")
    fig.colorbar(scatter, ax=axes, label="Digit", ticks=range(10), fraction=.025)
    _save(fig, path)
