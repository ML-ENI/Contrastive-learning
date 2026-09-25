from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def validate(self) -> None:
        # Train/validation indices share the official-train namespace; test is
        # from a different official dataset and therefore checked separately.
        if np.intersect1d(self.train, self.val).size:
            raise ValueError("train and validation overlap")
        if len(np.unique(self.test)) != len(self.test):
            raise ValueError("duplicate test indices")


def stratified_indices(labels: Sequence[int], size: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    if size > len(labels):
        raise ValueError(f"requested {size} examples from only {len(labels)}")
    if size == len(labels):
        return np.arange(len(labels), dtype=np.int64)
    chosen, _ = train_test_split(
        np.arange(len(labels)), train_size=size, random_state=seed, stratify=labels
    )
    return np.sort(chosen.astype(np.int64))


def make_splits(train_labels: Sequence[int], test_labels: Sequence[int], train_size: int,
                val_size: int, test_size: int, seed: int) -> SplitIndices:
    labels = np.asarray(train_labels)
    if train_size + val_size > len(labels):
        raise ValueError("train_size + val_size exceeds official MNIST train size")
    selected = stratified_indices(labels, train_size + val_size, seed)
    local_train, local_val = train_test_split(
        selected, train_size=train_size, random_state=seed,
        stratify=labels[selected]
    )
    test = stratified_indices(test_labels, test_size, seed)
    result = SplitIndices(np.sort(local_train), np.sort(local_val), test)
    result.validate()
    return result


def labelled_subset(indices: Sequence[int], labels: Sequence[int], fraction: float,
                     seed: int) -> np.ndarray:
    indices = np.asarray(indices)
    n = max(10, int(round(len(indices) * fraction)))
    local_indices = stratified_indices(np.asarray(labels)[indices], n, seed)
    return np.sort(indices[local_indices])


class GaussianNoise:
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)


def base_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])


def contrastive_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.10, 0.10), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        GaussianNoise(0.04),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


class ContrastiveDataset(Dataset):
    """Returns two independent views only—never a label."""

    def __init__(self, dataset: Dataset, indices: Sequence[int], transform=None):
        self.dataset = dataset
        self.indices = np.asarray(indices)
        self.transform = transform or contrastive_transform()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, _ignored_label = self.dataset[int(self.indices[item])]
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image), mode="L")
        return self.transform(image), self.transform(image)


@dataclass
class MNISTData:
    raw_train: datasets.MNIST
    raw_test: datasets.MNIST
    train: Subset
    val: Subset
    test: Subset
    splits: SplitIndices


def load_mnist(data_dir: str | Path, train_size: int, val_size: int, test_size: int,
               seed: int, download: bool = True) -> MNISTData:
    root = str(Path(data_dir))
    raw_train = datasets.MNIST(root, train=True, download=download, transform=None)
    raw_test = datasets.MNIST(root, train=False, download=download, transform=None)
    splits = make_splits(raw_train.targets.numpy(), raw_test.targets.numpy(), train_size,
                         val_size, test_size, seed)
    transformed_train = datasets.MNIST(root, train=True, download=False, transform=base_transform())
    transformed_test = datasets.MNIST(root, train=False, download=False, transform=base_transform())
    return MNISTData(raw_train, raw_test, Subset(transformed_train, splits.train),
                     Subset(transformed_train, splits.val), Subset(transformed_test, splits.test), splits)
