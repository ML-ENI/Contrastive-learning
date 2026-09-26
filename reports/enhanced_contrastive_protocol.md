# Enhanced contrastive-learning protocol

This protocol keeps MNIST and makes representation learning—not merely final classification—the centre of the experiment.

## Primary question

Does self-supervised SimCLR pretraining make digit identity more linearly accessible when labels are scarce?

## Direct control

For every seed and label budget, fit the same linear head on:

1. a frozen randomly initialized encoder;
2. the frozen SimCLR-pretrained encoder.

Their paired accuracy difference isolates the observed contribution of contrastive pretraining. It is more direct than comparing only against an end-to-end supervised CNN.

## Optimisation safeguard

Labelled batch size adapts to subset size and each neural classifier receives a configurable minimum number of optimizer updates. This prevents the 1% condition from receiving only one gradient update per epoch. Early stopping remains validation-only and cannot trigger before the minimum update budget has been reached.

## Representation diagnostics

- Positive-versus-mismatched cosine similarity before and after pretraining.
- Nearest-neighbor image retrieval in random and pretrained embedding spaces.
- PCA before and after pretraining, with labels used only for post-hoc colouring.
- Paired SimCLR-minus-random-probe accuracy at 1%, 10%, and 100% labels.

Labels never define contrastive pairs or enter NT-Xent.

## Final protocol

- Official MNIST train: 48,000 training and 12,000 validation images.
- Official MNIST test: all 10,000 images, used only after model selection.
- Label budgets: 1%, 10%, and 100%, with fixed IDs within each seed.
- Seeds: 42, 123, and 2024.
- SimCLR pretraining: up to 30 epochs, AdamW, temperature 0.5.
- Supervised/probe training: adaptive labelled batches, at least 300 optimizer updates, validation early stopping.
- Report mean and standard deviation only after all three seeds complete.

## Interpretation

A positive SimCLR-minus-random-probe gain supports the claim that pretraining learned linearly useful information. It does not by itself prove superiority over end-to-end supervision. Conversely, a zero or negative gain is evidence about this architecture, augmentation policy, optimization budget, and dataset—not a universal rejection of contrastive learning.
