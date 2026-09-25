# Supervised and Contrastive Representation Learning on MNIST

## Introduction

Deep classifiers normally require labels, while self-supervised contrastive learning learns invariances from augmented views. This study tests whether a small SimCLR-style pretraining stage is useful when only 1%, 10%, or 100% of training labels are available. No direction of the result was assumed.

## Objectives

We compare (A) a compact supervised CNN, (B) the identical encoder pretrained without labels and evaluated by a frozen linear probe, and (C) multinomial logistic regression on normalized pixels. All methods receive identical labelled example IDs within each seed and budget.

## Theoretical Background

For supervised classification, cross-entropy minimizes $-\log p_\theta(y_i\mid x_i)$ and directly adjusts both representation and decision boundary toward the supplied class label. Contrastive pretraining instead makes two independently augmented views of each image. For an anchor $i$, its other view $j$ is the positive; the remaining $2N-2$ views in a batch are negatives. With normalized projections and temperature $\tau$, NT-Xent is

$$\ell_{i,j}=-\log\frac{\exp(\mathrm{sim}(z_i,z_j)/\tau)}{\sum_{k\ne i}\exp(\mathrm{sim}(z_i,z_k)/\tau)}.$$

An embedding space is useful when semantically related inputs admit a simple downstream boundary. The nonlinear projection head is used only to optimize the contrastive objective; the encoder representation beneath it is retained. A frozen linear probe then measures linear accessibility of digit identity without allowing the encoder to adapt. Thus a good representation and a good end classifier are related but distinct: the latter also depends on optimization, labels, and classifier capacity.

## Dataset

MNIST contains 28×28 grayscale images in ten digit classes. The quick protocol draws stratified, disjoint subsets of 12,000 training and 2,000 validation examples from the official training set and 2,000 examples from the official test set. The test set is untouched until final evaluation. Labels in contrastive pretraining are used only to construct splits and labelled budgets; batches contain two images and no labels.

## Methodology

The CNN encoder uses three convolutional blocks and a 128-dimensional representation. The supervised model adds a ten-class layer. SimCLR adds a two-layer MLP projection head and optimizes NT-Xent ($\tau=0.5$) from two moderate affine/noise views, with no flips. Its encoder is frozen for linear probing. Logistic regression tunes $C\in\{0.1,1,10\}$ on validation data. The contrastive method has additional access to every unlabelled training image during pretraining, so labelled budgets are matched but compute and image exposure are not equivalent.

## Experimental Setup

Mode: **quick**. Seeds actually executed: **[42]**. AdamW is used for neural models. Early stopping and hyperparameter selection use validation data only; test predictions are computed after training decisions. This is a smoke-test run and must not be interpreted as a general empirical conclusion. The full three-seed experiment was not executed in the CPU-only development environment.

## Results

The table below is generated from the saved metrics; `n/a` standard deviations mean only one seed was run.

<!-- BEGIN GENERATED TABLE -->
| method              |   label_fraction |   accuracy |   precision_macro |   recall_macro |   f1_macro |
|:--------------------|-----------------:|-----------:|------------------:|---------------:|-----------:|
| Logistic regression |           0.0100 |     0.7840 |            0.7859 |         0.7790 |     0.7786 |
| Logistic regression |           0.1000 |     0.8735 |            0.8735 |         0.8715 |     0.8717 |
| Logistic regression |           1.0000 |     0.9080 |            0.9071 |         0.9062 |     0.9063 |
| SimCLR linear probe |           0.0100 |     0.0880 |            0.0395 |         0.0883 |     0.0411 |
| SimCLR linear probe |           0.1000 |     0.3450 |            0.4188 |         0.3341 |     0.2644 |
| SimCLR linear probe |           1.0000 |     0.6460 |            0.6495 |         0.6388 |     0.6309 |
| Supervised CNN      |           0.0100 |     0.1015 |            0.0268 |         0.1005 |     0.0203 |
| Supervised CNN      |           0.1000 |     0.1175 |            0.0916 |         0.1040 |     0.0284 |
| Supervised CNN      |           1.0000 |     0.9445 |            0.9462 |         0.9444 |     0.9439 |
<!-- END GENERATED TABLE -->

![Method comparison](../outputs/quick/figures/method_comparison.png)

Observed outcomes:

- At 1% labels, the highest observed mean test accuracy was 0.7840 (Logistic regression). This is an observation, not proof of superiority.
- At 10% labels, the highest observed mean test accuracy was 0.8735 (Logistic regression). This is an observation, not proof of superiority.
- At 100% labels, the highest observed mean test accuracy was 0.9445 (Supervised CNN). This is an observation, not proof of superiority.

Confusion matrices and PCA views are stored under `outputs/quick/figures`. PCA colours use labels only after training for visualization.

## Discussion

Differences across label budgets should be interpreted jointly with the supervised validation curves, contrastive loss, confusion matrices, and PCA geometry. A higher linear-probe score would support the claim that pretraining exposed linearly useful structure; a lower score would show that this particular objective, augmentation policy, architecture, and training budget did not beat direct supervision. It would not refute contrastive learning in general. The quick run is solely an integration check.

## Limitations

MNIST is small, grayscale, centred, and far simpler than natural imagery. NT-Xent is batch-size sensitive. The compact architecture and short schedule may undertrain SimCLR. Quick mode has one seed and cannot quantify run-to-run uncertainty; even three full-mode seeds provide limited statistical power. Labelled validation data and unequal compute/image exposure prevent interpreting this as a pure compute-matched comparison. Logistic convergence warnings, if present, should also be inspected.

## Conclusions

The saved results support only the budget-specific observations above. Claims about label efficiency require the full multi-seed experiment; theoretical expectations are not substituted for measurements.

## References

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. Proceedings of the 37th ICML, PMLR 119, 1597–1607. https://proceedings.mlr.press/v119/chen20j.html

LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). *The MNIST database of handwritten digits*. http://yann.lecun.com/exdb/mnist/
