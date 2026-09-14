# CSE-SMNet

Official PyTorch implementation of **CSE-SMNet** for future-frame prediction based video anomaly detection.

> Repository preparation is in progress. The current revision contains the verified CSE-SMNet architecture, Avenue normal-training dataset, prediction losses, modular normal-only training pipeline, and the validated single-video frame-level evaluation pipeline with MTAR diagnostics.

## Repository organization

```text
CSE-SMNet/
├── configs/
│   ├── avenue_train.py
│   └── evaluation.py
├── cse_smnet/
│   ├── data/
│   │   ├── avenue.py
│   │   └── video_sequence.py
│   ├── engine/
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── losses/
│   │   └── prediction.py
│   ├── models/
│   │   ├── cse_smnet.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── spatial/
│   │   │   └── cse_block.py
│   │   └── temporal/
│   │       ├── mtar.py
│   │       └── timestamp.py
│   └── utils/
│       ├── checkpoint.py
│       ├── data_split.py
│       ├── memory.py
│       ├── metrics.py
│       ├── reproducibility.py
│       └── visualization.py
├── docs/
│   ├── source_file_mapping.md
│   ├── training.md
│   └── evaluation.md
├── train.py
└── test.py
```

## Training

Prepare the normal Avenue training videos under `data/train/`, then run:

```bash
python train.py --data-root data/train --output-dir outputs/avenue
```

The supplied training protocol uses a normal-only **video-level** train/validation split. Clips from the same source video are never divided across both subsets.

See [`docs/training.md`](docs/training.md) for the validated optimization settings, structured-memory regularization, resume behavior, and generated artifacts.

## Checkpoint compatibility

Public class/module terminology follows the revised CSE-SMNet manuscript. Legacy registered submodule keys required by the already-trained model are retained internally, so existing compatible `state_dict` checkpoints can be loaded without conversion.

## Evaluation

Evaluate a single ordered-frame test video with the trained best model:

```bash
python test.py \
  --video-dir data/ShanghaiTech/testing/frames/01_0016 \
  --weights outputs/shanghaitech/best_model.pth \
  --training-checkpoint outputs/shanghaitech/checkpoint.pth \
  --output-dir outputs/shanghaitech/01_0016 \
  --anomaly-range 13:236
```

The validated anomaly score is the per-video min-max normalization of `-PSNR` between the predicted and true future frames. Ground-truth ranges are optional; omitting them runs score export without label-based metrics. See [`docs/evaluation.md`](docs/evaluation.md) for frame alignment, AUC/EER computation, MTAR temperature restoration, and visualization semantics.

The training pipeline intentionally does not use anomaly labels for model selection.
