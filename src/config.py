from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ExperimentConfig:
    mode: str = "quick"
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "outputs/quick"
    train_size: int = 12_000
    val_size: int = 2_000
    test_size: int = 2_000
    label_fractions: tuple[float, ...] = (0.01, 0.10, 1.0)
    batch_size: int = 256
    labelled_batch_size: int = 128
    num_workers: int = 0
    feature_dim: int = 128
    projection_dim: int = 64
    temperature: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    contrastive_epochs: int = 2
    supervised_epochs: int = 3
    probe_epochs: int = 5
    patience: int = 2
    min_label_updates: int = 60
    seeds: tuple[int, ...] = (42,)
    pca_samples: int = 1_000
    similarity_batches: int = 8
    retrieval_samples: int = 1_000
    logistic_max_iter: int = 100
    reuse_pretraining: bool = False
    reuse_logistic: bool = False

    @classmethod
    def for_mode(cls, mode: str, output_dir: str | None = None) -> "ExperimentConfig":
        if mode == "quick":
            cfg = cls(mode=mode)
        elif mode == "full":
            cfg = cls(
                mode=mode,
                output_dir="outputs/full",
                train_size=48_000,
                val_size=12_000,
                test_size=10_000,
                batch_size=512,
                labelled_batch_size=128,
                contrastive_epochs=30,
                supervised_epochs=30,
                probe_epochs=30,
                patience=5,
                min_label_updates=300,
                seeds=(42, 123, 2024),
                logistic_max_iter=1_000,
            )
        else:
            raise ValueError("mode must be 'quick' or 'full'")
        if output_dir:
            cfg.output_dir = output_dir
        return cfg

    def as_dict(self) -> dict:
        return asdict(self)
