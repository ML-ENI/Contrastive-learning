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
from .evaluation import (embedding_similarity_samples, extract_embeddings, metric_dict,
                         predict_torch, train_logistic)
from .models import Classifier, CompactEncoder, SimCLR, make_linear_probe
from .reporting import write_presentation, write_report, write_result_fragments
from .training import (epochs_for_updates, labelled_batch_size, train_classifier,
                       train_contrastive, train_linear_probe)
from .utils import choose_device, dependency_versions, save_json, set_seed
from .visualization import (plot_comparison, plot_confusion, plot_eda, plot_history,
                            plot_nearest_neighbors, plot_pca, plot_pretraining_gain,
                            plot_similarity)


def loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=generator, pin_memory=torch.cuda.is_available())


def run(cfg: ExperimentConfig) -> pd.DataFrame:
    output = Path(cfg.output_dir)
    figures, checkpoints, histories = output / "figures", output / "checkpoints", output / "histories"
    for directory in (figures, checkpoints, histories, Path("reports")):
        directory.mkdir(parents=True, exist_ok=True)
    cached_logistic: dict[tuple[int, float], dict] = {}
    if cfg.reuse_logistic and (output / "metrics.csv").exists():
        previous = pd.read_csv(output / "metrics.csv")
        for _, row in previous[previous.method == "Logistic regression"].iterrows():
            restored = row.to_dict()
            restored["confusion_matrix"] = json.loads(restored["confusion_matrix"])
            cached_logistic[(int(restored["seed"]), float(restored["label_fraction"]))] = restored
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
        simclr_checkpoint = checkpoints / f"simclr_seed_{seed}.pt"
        simclr_history_path = histories / f"simclr_seed_{seed}.csv"
        if cfg.reuse_pretraining and simclr_checkpoint.exists() and simclr_history_path.exists():
            payload = torch.load(simclr_checkpoint, map_location=device)
            simclr.load_state_dict(payload["model_state"])
            contrast_history = pd.read_csv(simclr_history_path).to_dict("records")
            print("  Reused existing SimCLR checkpoint", flush=True)
        else:
            contrast_history = train_contrastive(
                simclr,
                loader(contrast_train, cfg.batch_size, True, seed, cfg.num_workers),
                loader(contrast_val, cfg.batch_size, False, seed, cfg.num_workers),
                cfg.contrastive_epochs, cfg.learning_rate, cfg.weight_decay, cfg.temperature,
                cfg.patience, device, simclr_checkpoint
            )
            pd.DataFrame(contrast_history).to_csv(simclr_history_path, index=False)

        # Directly measure the contrastive objective in representation space.
        set_seed(seed)
        random_similarity = embedding_similarity_samples(
            initial_encoder,
            loader(contrast_val, cfg.batch_size, False, seed, cfg.num_workers),
            device, cfg.similarity_batches,
        )
        set_seed(seed)
        pretrained_similarity = embedding_similarity_samples(
            simclr.encoder,
            loader(contrast_val, cfg.batch_size, False, seed, cfg.num_workers),
            device, cfg.similarity_batches,
        )
        similarity_summary = {
            "seed": seed,
            "random_positive_mean": float(random_similarity["positive"].mean()),
            "random_negative_mean": float(random_similarity["negative"].mean()),
            "pretrained_positive_mean": float(pretrained_similarity["positive"].mean()),
            "pretrained_negative_mean": float(pretrained_similarity["negative"].mean()),
        }
        similarity_summary["random_gap"] = (
            similarity_summary["random_positive_mean"] - similarity_summary["random_negative_mean"]
        )
        similarity_summary["pretrained_gap"] = (
            similarity_summary["pretrained_positive_mean"] - similarity_summary["pretrained_negative_mean"]
        )
        save_json(similarity_summary, output / f"similarity_seed_{seed}.json")
        if seed_number == 0:
            plot_history(contrast_history, figures / "contrastive_loss.png", "SimCLR pretraining", False)
            plot_similarity(random_similarity, pretrained_similarity,
                            figures / "positive_negative_similarity.png")
            analysis_limit = max(cfg.pca_samples, cfg.retrieval_samples)
            before, analysis_labels = extract_embeddings(initial_encoder, test_loader, device, analysis_limit)
            after, after_labels = extract_embeddings(simclr.encoder, test_loader, device, analysis_limit)
            if not np.array_equal(analysis_labels, after_labels):
                raise RuntimeError("PCA samples are misaligned")
            plot_pca(before[:cfg.pca_samples], after[:cfg.pca_samples],
                     analysis_labels[:cfg.pca_samples], figures / "embedding_pca.png")
            retrieval_count = min(cfg.retrieval_samples, len(before))
            raw_images = np.stack([
                np.asarray(data.raw_test[int(index)][0])
                for index in data.splits.test[:retrieval_count]
            ])
            plot_nearest_neighbors(before[:retrieval_count], after[:retrieval_count], raw_images,
                                   analysis_labels[:retrieval_count],
                                   figures / "embedding_nearest_neighbors.png")

        for fraction in cfg.label_fractions:
            budget = f"{int(fraction * 100):03d}pct"
            labelled_ids = labelled_subset(data.splits.train, train_labels, fraction, seed)
            save_json(labelled_ids.tolist(), output / f"labelled_ids_seed_{seed}_{budget}.json")
            labelled_train = Subset(transformed_train, labelled_ids.tolist())
            label_batch = labelled_batch_size(len(labelled_train), cfg.labelled_batch_size)
            train_loader = loader(labelled_train, label_batch, True, seed, cfg.num_workers)
            eval_train_loader = loader(labelled_train, label_batch, False, seed, cfg.num_workers)
            supervised_epochs = epochs_for_updates(
                cfg.supervised_epochs, len(train_loader), cfg.min_label_updates
            )
            probe_epochs = epochs_for_updates(cfg.probe_epochs, len(train_loader), cfg.min_label_updates)
            print(f"  Labels {fraction:.0%}: n={len(labelled_ids)}, batch={label_batch}, "
                  f"max epochs supervised/probe={supervised_epochs}/{probe_epochs}", flush=True)

            set_seed(seed)
            supervised = Classifier(CompactEncoder(cfg.feature_dim), cfg.feature_dim)
            sup_history = train_classifier(
                supervised, train_loader, val_loader, supervised_epochs, cfg.learning_rate,
                cfg.weight_decay, cfg.patience, device,
                checkpoints / f"supervised_seed_{seed}_{budget}.pt", cfg.min_label_updates
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
            set_seed(seed)
            probe = make_linear_probe(probe_encoder, cfg.feature_dim)
            probe_history = train_linear_probe(
                probe, train_loader, val_loader, probe_epochs, cfg.learning_rate,
                cfg.weight_decay, cfg.patience, device,
                checkpoints / f"linear_probe_seed_{seed}_{budget}.pt", cfg.min_label_updates
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

            # Essential control: identical frozen architecture without contrastive pretraining.
            set_seed(seed)
            random_probe = make_linear_probe(copy.deepcopy(initial_encoder), cfg.feature_dim)
            random_history = train_linear_probe(
                random_probe, train_loader, val_loader, probe_epochs, cfg.learning_rate,
                cfg.weight_decay, cfg.patience, device,
                checkpoints / f"random_probe_seed_{seed}_{budget}.pt", cfg.min_label_updates
            )
            pd.DataFrame(random_history).to_csv(
                histories / f"random_probe_seed_{seed}_{budget}.csv", index=False
            )
            y_true, y_pred = predict_torch(random_probe, test_loader, device)
            metrics = metric_dict(y_true, y_pred)
            metrics.update(method="Random encoder probe", label_fraction=fraction, seed=seed,
                           labelled_samples=len(labelled_ids),
                           runtime_seconds=random_history[-1]["total_seconds"])
            all_results.append(metrics)
            plot_confusion(np.asarray(metrics["confusion_matrix"]),
                           f"Random encoder probe — {fraction:.0%} labels — seed {seed}",
                           figures / f"confusion_random_probe_{budget}_seed_{seed}.png")

            cached_key = (seed, float(fraction))
            if cached_key in cached_logistic:
                log_metrics = cached_logistic[cached_key]
                print("    Reused logistic-regression result", flush=True)
            else:
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
    plot_pretraining_gain(result_frame, figures)
    write_result_fragments(result_frame, output, Path("reports"), cfg.mode)
    write_report(result_frame, Path("reports"), cfg.mode, cfg.seeds, output)
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
    parser.add_argument("--reuse-pretraining", action="store_true",
                        help="Reuse a compatible SimCLR checkpoint already in the output directory.")
    parser.add_argument("--reuse-logistic", action="store_true",
                        help="Reuse compatible logistic rows already in metrics.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.for_mode(args.mode, args.output_dir)
    cfg.reuse_pretraining = args.reuse_pretraining
    cfg.reuse_logistic = args.reuse_logistic
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
