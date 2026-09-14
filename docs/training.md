# Training CSE-SMNet

The official training pipeline preserves the validated **normal-only, video-level** Avenue development protocol used by the supplied experiment script.

## Entry point

```bash
python train.py --data-root data/train --output-dir outputs/avenue
```

Use `--no-resume` to ignore an existing resumable checkpoint. If CUDA is unavailable, the entry point automatically falls back to CPU.

## Expected Avenue normal-training layout

```text
data/train/
├── O1/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── O2/
├── O3/
└── ...
```

Each training sample uses three consecutive frames to predict the next frame. The split is performed at the **source-video level**, preventing clips from the same video from appearing in both training and validation subsets.

## Default validated protocol

- Seed: `42`
- Sequence length: `3`
- Image size: `256 x 256`
- Batch size: `4`
- Epochs: `100`
- Validation ratio: `0.10` at video level
- Optimizer: Adam
- Learning rate: `1e-4`
- Weight decay: `1e-5`
- Scheduler: StepLR, step size `20`, gamma `0.5`
- Gradient clipping: `5.0`

Prediction objective:

```text
L_pred = 0.00 L_MSE + 0.50 L_SSIM + 0.30 L_temporal + 0.20 L_gradient
```

MTAR structured-memory regularization:

```text
lambda_separate = 1e-3
lambda_diverse  = 2e-3
lambda_usage    = 2e-4
```

The retrieval-temperature schedule uses the supplied values `0.07 -> 0.10` with cosine interpolation.

## Model selection

The best checkpoint is selected using the **complete normal-validation training objective**. No anomaly label is used for training, validation, checkpoint selection, or hyperparameter selection in this pipeline.

The labeled test set should be used only for the final anomaly-detection evaluation (for example, frame-level AUC and EER).

## Outputs

The output directory contains:

```text
outputs/avenue/
├── best_model.pth
├── last_model.pth
├── checkpoint.pth
├── normal_train_val_split.json
├── training_history.csv
├── memory_diagnostics_latest.json
├── memory_diagnostics_best.json
├── memory_diagnostics_final_best.json
├── memory_artifacts_per_epoch/
├── memory_visualizations_best/
└── memory_visualizations_final/
```

`checkpoint.pth` stores model, optimizer, scheduler, RNG state, split metadata, MTAR configuration, and current metrics for exact-resume behavior.
