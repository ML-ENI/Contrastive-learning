# Presentation outline (5–10 minutes)

## 1. Question and fair comparison — 45 s

- Can unlabelled-image pretraining reduce dependence on labels?
- MNIST; 1%, 10%, 100% labelled budgets; fixed IDs per seed.
- Caveat: SimCLR sees extra unlabelled images and uses extra compute.

## 2. Three approaches — 75 s

- Supervised CNN: encoder + class head, cross-entropy.
- Small SimCLR: two moderate views → shared encoder → MLP → NT-Xent; discard MLP, freeze encoder, fit linear probe.
- Frozen random encoder + the same probe isolates whether pretraining learned useful representations.
- Normalized-pixel logistic regression as a simple reference.

## 3. Protocol and safeguards — 60 s

- Stratified train/validation/test; no overlap; test used only after selection.
- Same CNN encoder and same labelled IDs for neural methods.
- No labels in contrastive batches; no flips that change digit identity.

## 4. Results — 2–3 min

- Show `outputs/full_reduced/figures/method_comparison.png`; stress that quick mode is a smoke test.
- Show representative confusion matrices and point out class-specific errors.
- Show supervised curves and contrastive loss; distinguish optimization from generalization.

## 5. Representation analysis — 60 s

- Show positive/negative cosine similarity and nearest-neighbor retrieval before/after pretraining.
- Use PCA as a complementary 2D view.
- Colours are labels used only for post-hoc visualization.
- The SimCLR-versus-random probe difference is the direct quantitative pretraining control.

## 6. Conclusions and limitations — 60 s

- State only measured budget-specific outcomes from the generated report.
- Mention MNIST simplicity, batch-size/schedule sensitivity, unequal compute, and actual seed count.
- Next step: full three-seed Colab run, then report mean ± standard deviation.
