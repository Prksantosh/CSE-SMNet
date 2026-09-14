"""Official single-video evaluation entry point for CSE-SMNet."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from configs.evaluation import EvaluationConfig
from cse_smnet.data.video_sequence import VideoSequenceDataset
from cse_smnet.engine.evaluator import CSESMNetEvaluator
from cse_smnet.models import CSESMNet


def parse_anomaly_range(value: str) -> Tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            "Anomaly ranges must use START:END, e.g. 13:236."
        ) from exc
    if end < start:
        raise argparse.ArgumentTypeError("Anomaly range END must be >= START.")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CSE-SMNet on one test video.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing ordered test frames.")
    parser.add_argument("--weights", type=Path, required=True, help="best_model.pth or compatible checkpoint.")
    parser.add_argument("--training-checkpoint", type=Path, default=None, help="Optional checkpoint.pth used to restore the best-epoch MTAR temperature.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grayscale", action="store_true", help="Read grayscale frames and replicate them to three channels.")
    parser.add_argument(
        "--anomaly-range",
        type=parse_anomaly_range,
        action="append",
        default=[],
        metavar="START:END",
        help="Inclusive anomalous frame range. Repeat for multiple ranges. Values are one-based unless --zero-based-labels is used.",
    )
    parser.add_argument("--zero-based-labels", action="store_true")
    parser.add_argument("--no-frame-visualizations", action="store_true")
    parser.add_argument("--visualization-stride", type=int, default=1)
    parser.add_argument("--no-memory-visualizations", action="store_true")
    parser.add_argument("--fallback-memory-temperature", type=float, default=0.20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.visualization_stride < 1:
        raise ValueError("--visualization-stride must be >= 1.")

    config = EvaluationConfig(
        test_video_dir=args.video_dir,
        best_model_path=args.weights,
        training_checkpoint_path=args.training_checkpoint,
        results_dir=args.output_dir,
        seq_len=args.seq_len,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grayscale_dataset=args.grayscale,
        anomaly_ranges=args.anomaly_range,
        ground_truth_one_based=not args.zero_based_labels,
        save_frame_visualizations=not args.no_frame_visualizations,
        visualization_stride=args.visualization_stride,
        save_memory_visualizations=not args.no_memory_visualizations,
        fallback_inference_memory_temperature=args.fallback_memory_temperature,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoSequenceDataset(
        config.test_video_dir,
        seq_len=config.seq_len,
        image_size=config.image_size,
        grayscale_dataset=config.grayscale_dataset,
    )
    print(
        f"Test video={config.test_video_dir.name} | frames={len(dataset.frames)} | "
        f"samples={len(dataset)} | device={device}"
    )

    model = CSESMNet(seq_len=config.seq_len).to(device)
    evaluator = CSESMNetEvaluator(
        model=model,
        dataset=dataset,
        device=device,
        config=config,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
