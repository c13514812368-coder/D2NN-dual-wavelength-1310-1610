# 1560 nm three-class feature upgrade

The original train/val split and all prior experiments are preserved.
The dataset is synthetic Imagenet-1k-SD-1.4. There is no independent test split.

## Method

RGB images at 64x64 produce 568 handcrafted features: multiscale colors, contrasts, orientation histograms and low-frequency DCT.
A training-only RBF Nystroem map (384 dimensions) feeds two LDA axes; 14 PCA axes retain unsupervised variation.
The linear/kernel choice uses a 20% inner holdout from the training set, not the fixed validation set.
Training-only percentile scaling maps the 16 components to input phases in [0.2,1.8] pi.
No pretrained CNN or classifier probabilities are used as the optical input.
The original calibrated optical map and all 11292 monitor samples are preserved.
Adam uses cosine decay and gradient clipping. The final category still comes from three optical ports.

The reported accuracies use the differentiable optical simulation. The newly trained geometry has not been revalidated with full FDTD.

## Fixed-split results

| Model | Validation accuracy |
|---|---:|
| Original 1560 nm D2NN | 79.05% |
| SVM on rich 568-D features (diagnostic) | 94.13% |
| SVM on compressed 16-D input (diagnostic) | 92.54% |
| New 1560 nm D2NN, best epoch 9 | 91.75% |

Best checkpoint noiseless train accuracy: 91.65%.
Final-epoch validation accuracy: 91.03%.
Improvement over the old best: 12.70 percentage points.

## Per-class recall at the best checkpoint

| Class | Correct / total | Recall |
|---|---:|---:|
| natural_landform | 369 / 420 | 87.86% |
| water_coast | 385 / 420 | 91.67% |
| built_environment | 402 / 420 | 95.71% |

## Interpretation

This improvement includes a richer digital encoder and supervised compression; it is not evidence of a wavelength-count gain.
Any future dual-wavelength comparison must use matched input encoders and fixed splits.
The fixed validation set selects the best checkpoint, so an independent untouched test set is still needed for a final test claim.

## Reproduce

Run feature_upgrade_20261011/prepare_features.py, then train_rich16_1560_20261011/run.py.
The serialized encoder is encoder.joblib and the input cache is encoded16.npz.
report_results.py reloads the saved best checkpoint and verifies its full-validation result.
