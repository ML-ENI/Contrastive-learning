from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .config import ExperimentConfig
from .data import ContrastiveDataset, labelled_subset, load_mnist
from .evaluation import extract_embeddings, metric_dict, predict_torch, train_logistic
from .models import Classifier, CompactEncoder, SimCLR, make_linear_probe
from .reporting import write_presentation, write_report, write_result_fragments
from .training import train_classifier, train_contrastive, train_linear_probe
from .utils import choose_device, dependency_versions, save_json, set_seed
from .visualization import (plot_comparison, plot_confusion, plot_eda, plot_history, plot_pca)


def loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=generator, pin_memory=torch.cuda.is_available())


def run(cfg: ExperimentConfig) -> pd.DataFrame:
    output = Path(cfg.output_dir)
    figures, checkpoints, histories = output / "figures", output / "checkpoints", output / "histories"
    for directory in (figures, checkpoints, histories, Path("reports")):
        directory.mkdir(parents=True, exist_ok=True)
    save_json(cfg.as_dict(), output / "config.json")
    save_json(dependency_versions(), output / "versions.json")
    device = choose_device()
    print(f"Mode={cfg.mode}; device={device}; output={output}", flush=True)
    all_results: list[dict] = []

    for seed_number, seed in enumerate(cfg.seeds):
        print(f"\nSeed {seed} ({seed_number + 1}/{len(cfg.seeds)})", flush=True)
        set_seed(seed)
        data = load_mnist(cfg.data_dir, cfg.train_size, cfg.val_size, cfg.test_size, seed)
        save_json({"train": data.splits.train.tolist(), "val": data.splits.val.tolist(),
                   "test": data.splits.test.tolist(),
                   "note": "test IDs use the separate official-test index namespace"},
                  output / f"split_indices_seed_{seed}.json")
        if seed_number == 0:
            plot_eda(data.raw_train, data.raw_test,
                     {"train": data.splits.train, "val": data.splits.val, "test": data.splits.test}, figures)

        train_labels = data.raw_train.targets.numpy()
        transformed_train = data.train.dataset
        val_loader = loader(data.val, cfg.batch_size, False, seed, cfg.num_workers)
        test_loader = loader(data.test, cfg.batch_size, False, seed, cfg.num_workers)

        # Pretraining uses only two-view image batches; ContrastiveDataset drops labels.
        set_seed(seed)
        encoder = CompactEncoder(cfg.feature_dim)
        initial_encoder = copy.deepcopy(encoder)
        simclr = SimCLR(encoder, cfg.feature_dim, cfg.projection_dim)
        contrast_train = ContrastiveDataset(data.raw_train, data.splits.train)
        contrast_val = ContrastiveDataset(data.raw_train, data.splits.val)
        contrast_history = train_contrastive(
            simclr,
            loader(contrast_train, cfg.batch_size, True, seed, cfg.num_workers),
            loader(contrast_val, cfg.batch_size, False, seed, cfg.num_workers),
            cfg.contrastive_epochs, cfg.learning_rate, cfg.weight_decay, cfg.temperature,
            cfg.patience, device, checkpoints / f"simclr_seed_{seed}.pt"
        )
        pd.DataFrame(contrast_history).to_csv(histories / f"simclr_seed_{seed}.csv", index=False)
        if seed_number == 0:
            plot_history(contrast_history, figures / "contrastive_loss.png", "SimCLR pretraining", False)
            before, pca_labels = extract_embeddings(initial_encoder, test_loader, device, cfg.pca_samples)
            after, after_labels = extract_embeddings(simclr.encoder, test_loader, device, cfg.pca_samples)
            if not np.array_equal(pca_labels, after_labels):
                raise RuntimeError("PCA samples are misaligned")
            plot_pca(before, after, pca_labels, figures / "embedding_pca.png")

        for fraction in cfg.label_fractions:
            budget = f"{int(fraction * 100):03d}pct"
            labelled_ids = labelled_subset(data.splits.train, train_labels, fraction, seed)
            save_json(labelled_ids.tolist(), output / f"labelled_ids_seed_{seed}_{budget}.json")
            labelled_train = Subset(transformed_train, labelled_ids.tolist())
            train_loader = loader(labelled_train, cfg.batch_size, True, seed, cfg.num_workers)
            eval_train_loader = loader(labelled_train, cfg.batch_size, False, seed, cfg.num_workers)
            print(f"  Labels {fraction:.0%}: n={len(labelled_ids)}", flush=True)

            set_seed(seed)
            supervised = Classifier(CompactEncoder(cfg.feature_dim), cfg.feature_dim)
            sup_history = train_classifier(
                supervised, train_loader, val_loader, cfg.supervised_epochs, cfg.learning_rate,
                cfg.weight_decay, cfg.patience, device,
                checkpoints / f"supervised_seed_{seed}_{budget}.pt"
            )
            pd.DataFrame(sup_history).to_csv(histories / f"supervised_seed_{seed}_{budget}.csv", index=False)
            y_true, y_pred = predict_torch(supervised, test_loader, device)
            metrics = metric_dict(y_true, y_pred)
            metrics.update(method="Supervised CNN", label_fraction=fraction, seed=seed,
                           labelled_samples=len(labelled_ids), runtime_seconds=sup_history[-1]["total_seconds"])
            all_results.append(metrics)
            plot_confusion(np.asarray(metrics["confusion_matrix"]),
                           f"Supervised CNN — {fraction:.0%} labels — seed {seed}",
                           figures / f"confusion_supervised_{budget}_seed_{seed}.png")
            if seed_number == 0 and fraction == 1.0:
                plot_history(sup_history, figures / "supervised_curves.png", "Supervised CNN", True)

            probe_encoder = copy.deepcopy(simclr.encoder)
            probe = make_linear_probe(probe_encoder, cfg.feature_dim)
            probe_history = train_linear_probe(
                probe, train_loader, val_loader, cfg.probe_epochs, cfg.learning_rate,
                cfg.weight_decay, cfg.patience, device,
                checkpoints / f"linear_probe_seed_{seed}_{budget}.pt"
            )
            pd.DataFrame(probe_history).to_csv(histories / f"linear_probe_seed_{seed}_{budget}.csv", index=False)
            y_true, y_pred = predict_torch(probe, test_loader, device)
            metrics = metric_dict(y_true, y_pred)
            metrics.update(method="SimCLR linear probe", label_fraction=fraction, seed=seed,
                           labelled_samples=len(labelled_ids), runtime_seconds=probe_history[-1]["total_seconds"])
            all_results.append(metrics)
            plot_confusion(np.asarray(metrics["confusion_matrix"]),
                           f"SimCLR linear probe — {fraction:.0%} labels — seed {seed}",
                           figures / f"confusion_probe_{budget}_seed_{seed}.png")

            log_metrics, _ = train_logistic(eval_train_loader, val_loader, test_loader,
                                             cfg.logistic_max_iter, seed)
            log_metrics.update(method="Logistic regression", label_fraction=fraction, seed=seed,
                               labelled_samples=len(labelled_ids))
            all_results.append(log_metrics)
            plot_confusion(np.asarray(log_metrics["confusion_matrix"]),
                           f"Logistic regression — {fraction:.0%} labels — seed {seed}",
                           figures / f"confusion_logistic_{budget}_seed_{seed}.png")

            serializable = [{k: (json.dumps(v) if k == "confusion_matrix" else v) for k, v in row.items()}
                            for row in all_results]
            pd.DataFrame(serializable).to_csv(output / "metrics.csv", index=False)

    result_frame = pd.read_csv(output / "metrics.csv")
    plot_comparison(result_frame, figures)
    write_result_fragments(result_frame, output, Path("reports"), cfg.mode)
    write_report(result_frame, Path("reports"), cfg.mode, cfg.seeds)
    write_presentation(Path("reports"), cfg.mode)
    save_json({"status": "complete", "mode": cfg.mode, "device": str(device),
               "seeds_completed": list(cfg.seeds), "test_used_for_selection": False},
              output / "run_manifest.json")
    print(f"\nComplete. Metrics: {output / 'metrics.csv'}", flush=True)
    return result_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, help="Override quick mode seed (or use one seed in full mode).")
    parser.add_argument("--contrastive-epochs", type=int)
    parser.add_argument("--supervised-epochs", type=int)
    parser.add_argument("--probe-epochs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.for_mode(args.mode, args.output_dir)
    if args.seed is not None:
        cfg.seeds = (args.seed,)
        cfg.seed = args.seed
    for name in ("contrastive_epochs", "supervised_epochs", "probe_epochs"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    run(cfg)


if __name__ == "__main__":
    main()
