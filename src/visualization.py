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


def plot_pretraining_gain(results: pd.DataFrame, output: Path) -> None:
    """Paired accuracy gain of SimCLR features over identical random features."""
    required = {"SimCLR linear probe", "Random encoder probe"}
    if not required.issubset(set(results.method)):
        return
    paired = results[results.method.isin(required)].pivot_table(
        index=["seed", "label_fraction"], columns="method", values="accuracy"
    ).dropna()
    paired["gain"] = paired["SimCLR linear probe"] - paired["Random encoder probe"]
    summary = paired.groupby("label_fraction")["gain"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#2ca02c" if value >= 0 else "#d62728" for value in summary["mean"]]
    ax.bar(summary.label_fraction * 100, summary["mean"],
           yerr=summary["std"].fillna(0), width=[.7, 3, 12][:len(summary)],
           color=colours, alpha=.8, capsize=4)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Labelled training data (%)", ylabel="Accuracy gain",
           title="Effect of SimCLR pretraining vs. a random frozen encoder",
           xticks=summary.label_fraction * 100)
    ax.grid(axis="y", alpha=.25)
    _save(fig, output / "simclr_pretraining_gain.png")


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


def plot_similarity(before: dict[str, np.ndarray], after: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    bins = np.linspace(-1, 1, 35)
    for values, title, ax in ((before, "Random encoder", axes[0]),
                              (after, "After SimCLR pretraining", axes[1])):
        ax.hist(values["negative"], bins=bins, alpha=.65, density=True, label="mismatched views")
        ax.hist(values["positive"], bins=bins, alpha=.65, density=True, label="positive views")
        gap = values["positive"].mean() - values["negative"].mean()
        ax.set(title=f"{title}\nmean positive-negative gap = {gap:.3f}",
               xlabel="Cosine similarity", ylabel="Density")
        ax.legend()
    _save(fig, path)


def plot_nearest_neighbors(before: np.ndarray, after: np.ndarray, images: np.ndarray,
                           labels: np.ndarray, path: Path, queries: int = 5,
                           neighbors: int = 4) -> None:
    """Compare cosine-nearest retrieval without using labels to choose neighbors."""
    query_ids = np.linspace(0, len(images) - 1, queries, dtype=int)
    fig, axes = plt.subplots(queries * 2, neighbors + 1,
                             figsize=(2 * (neighbors + 1), 2 * queries * 2))
    for block, (embedding, name) in enumerate(((before, "Random encoder"),
                                                (after, "SimCLR encoder"))):
        normalized = embedding / np.clip(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12, None)
        similarities = normalized @ normalized.T
        np.fill_diagonal(similarities, -np.inf)
        for row, query in enumerate(query_ids):
            axis_row = block * queries + row
            found = np.argsort(similarities[query])[-neighbors:][::-1]
            axes[axis_row, 0].imshow(images[query], cmap="gray")
            axes[axis_row, 0].set_title(f"{name}\nquery: {labels[query]}", fontsize=8)
            for column, neighbor in enumerate(found, start=1):
                axes[axis_row, column].imshow(images[neighbor], cmap="gray")
                axes[axis_row, column].set_title(
                    f"label {labels[neighbor]}\ncos={similarities[query, neighbor]:.2f}", fontsize=8)
            for ax in axes[axis_row]:
                ax.axis("off")
    fig.suptitle("Nearest neighbors in embedding space (labels shown only for interpretation)",
                 fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, .975), h_pad=1.2)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
