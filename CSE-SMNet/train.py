"""Official CSE-SMNet training entry point.

Example
-------
python train.py --data-root data/train --output-dir outputs/avenue
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from configs import AvenueTrainConfig
from cse_smnet import CSESMNet
from cse_smnet.data import AvenueDataset
from cse_smnet.engine import CSESMNetTrainer
from cse_smnet.losses import CombinedPredictionLoss
from cse_smnet.utils.data_split import make_video_level_split, save_split_manifest
from cse_smnet.utils.memory import find_mtar_module
from cse_smnet.utils.reproducibility import seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CSE-SMNet on Avenue normal videos.")
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/avenue"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-resume", action="store_true", help="Start a fresh run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AvenueTrainConfig(
        train_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        resume=not args.no_resume,
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    model = CSESMNet(seq_len=cfg.seq_len).to(device)
    memory_module = find_mtar_module(model)

    criterion = CombinedPredictionLoss(
        lambda_mse=cfg.lambda_mse,
        lambda_ssim=cfg.lambda_ssim,
        lambda_temp=cfg.lambda_temp,
        lambda_grad=cfg.lambda_grad,
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.lr_step_size,
        gamma=cfg.lr_gamma,
    )

    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
    ])

    if not cfg.train_root.is_dir():
        raise FileNotFoundError(
            f"Normal training directory does not exist: {cfg.train_root}"
        )

    dataset = AvenueDataset(
        root_dir=str(cfg.train_root),
        seq_len=cfg.seq_len,
        transform=transform,
    )
    train_dataset, val_dataset, train_videos, val_videos = make_video_level_split(
        dataset,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    save_split_manifest(
        cfg.output_dir / "normal_train_val_split.json",
        protocol_version=cfg.protocol_version,
        seed=cfg.seed,
        val_ratio=cfg.val_ratio,
        train_video_ids=train_videos,
        val_video_ids=val_videos,
        train_clips=len(train_dataset),
        val_clips=len(val_dataset),
        memory_regularization=cfg.memory_regularization_dict(),
    )

    train_generator = torch.Generator().manual_seed(cfg.seed)
    val_generator = torch.Generator().manual_seed(cfg.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    print("=" * 80)
    print("CSE-SMNet | NORMAL-ONLY AVENUE TRAINING")
    print("=" * 80)
    print(f"Device              : {device}")
    print(f"Training videos     : {len(train_videos)}")
    print(f"Validation videos   : {len(val_videos)}")
    print(f"Training clips      : {len(train_dataset)}")
    print(f"Validation clips    : {len(val_dataset)}")
    print(f"Uses anomaly labels : False")
    print(f"Output directory    : {cfg.output_dir}")
    print("=" * 80)

    trainer = CSESMNetTrainer(
        model=model,
        memory_module=memory_module,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=cfg,
        train_video_ids=train_videos,
        val_video_ids=val_videos,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
