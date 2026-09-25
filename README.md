# MNIST: supervised vs. contrastive learning

Reproducible PRDL/MLLB assignment comparing a supervised CNN, a small self-supervised SimCLR encoder with a frozen linear probe, and logistic regression at 1%, 10%, and 100% labelled-data budgets. The neural methods use the same compact encoder. SimCLR sees all training images without labels during pretraining, so its image exposure and compute are intentionally documented as unequal to direct supervision.

## Setup

Python 3.11+ is supported. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

On systems where the interpreter is named `python3` (including many macOS installations), use `python3` for the first command. Once the virtual environment is active, both `python` and `python3` normally resolve inside `.venv`. NumPy, SciPy, contourpy, PyTorch, and torchvision are constrained to a mutually compatible ABI range because the PyTorch 2.2 Intel-macOS wheel cannot interoperate with NumPy 2.x.

MNIST is downloaded automatically by `torchvision.datasets.MNIST` into `data/` on the first run. This directory is ignored by Git. The official training set alone supplies training and validation; official test examples are reserved for final metrics.

## Run

```bash
python3 -m src.run_experiment --mode quick
python3 -m src.run_experiment --mode full
```

Quick mode is a pipeline smoke test: one seed, 12,000/2,000/2,000 stratified train/validation/test examples, and short schedules. It must not be used for general conclusions. Full mode uses all 60,000 official training images as a 48,000/12,000 train/validation split, all 10,000 test images, three seeds, and longer schedules. It can require substantial CPU/GPU time. Optional flags include `--seed`, `--output-dir`, `--contrastive-epochs`, `--supervised-epochs`, and `--probe-epochs`.

For CUDA, install the matching PyTorch build from the official PyTorch selector before installing the remaining requirements. Apple Silicon automatically uses MPS when available; otherwise the code uses CPU. All paths are relative.

## Experimental safeguards

- Split and labelled-example IDs are saved under `outputs/<mode>/`.
- Labelled IDs are identical across the three methods for each seed/budget.
- `ContrastiveDataset` returns exactly two views and discards the source label.
- Moderate rotation, translation, scale, and Gaussian noise are used; flips are excluded.
- Early stopping and logistic-regression `C` selection use validation only.
- Test data enters only final prediction and post-training PCA visualization.
- Standard deviations are reported only when multiple seeds were actually run.

## Outputs

Each run writes its configuration, dependency versions, split IDs, metrics, summary, histories, checkpoints, and PNG figures to `outputs/<mode>/`. Tables in `reports/results_table.md`, the English report, and the presentation outline are regenerated from the real metrics. Checkpoints and downloaded datasets are ignored by Git.

The notebook [notebooks/contrastive_learning_mnist.ipynb](notebooks/contrastive_learning_mnist.ipynb) reuses the Python modules and is suitable for VS Code or Colab. In Colab, upload/clone the repository, change into its root, install requirements, then run the same module command.

## Project map

```text
src/config.py          configuration and quick/full presets
src/data.py            MNIST splits, labelled budgets, augmentations
src/models.py          shared CNN, classifier, projection head
src/losses.py          explicit NT-Xent implementation
src/training.py        supervised and contrastive training
src/evaluation.py      test metrics, embeddings, logistic regression
src/visualization.py   EDA, curves, comparisons, PCA, confusion matrices
src/reporting.py       result-driven report/table/presentation text
src/run_experiment.py  command-line orchestration
tests/                 automatic invariant tests
```

The original SimCLR source used for the theoretical design is Chen et al. (2020), [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html).
