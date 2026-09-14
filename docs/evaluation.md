# CSE-SMNet evaluation

The official evaluator reproduces the supplied future-frame anomaly-detection protocol while replacing experiment-specific paths with command-line arguments.

## Scoring protocol

For every valid target frame, CSE-SMNet receives the preceding `seq_len` frames and predicts the immediately following frame. The evaluator computes target-vs-prediction PSNR and defines the raw anomaly score as `-PSNR`. Scores are min-max normalized within the evaluated video. When anomaly frame ranges are supplied, frame-level ROC-AUC and EER are computed on the target-frame indices only.

The first `seq_len` frames do not have model-produced anomaly scores because they are used to form the initial prediction context.

## Example

```bash
python test.py \
  --video-dir data/ShanghaiTech/testing/frames/01_0016 \
  --weights outputs/shanghaitech/best_model.pth \
  --training-checkpoint outputs/shanghaitech/checkpoint.pth \
  --output-dir outputs/shanghaitech/01_0016 \
  --anomaly-range 13:236
```

`--anomaly-range` is inclusive and one-based by default. Repeat it when a video contains multiple anomalous intervals. Omit all anomaly ranges to run prediction and export scores without using labels.

## MTAR temperature

If the resumable training checkpoint is supplied, the evaluator reconstructs the MTAR retrieval temperature corresponding to the saved best epoch from the training metadata. Otherwise it uses the configured fallback temperature (default `0.20`).

## Visualization semantics

The model metric and anomaly score always compare the true future target `I_(t+1)` with the prediction `I_hat_(t+1)`. To reproduce the supplied diagnostic visualization, the displayed difference map, heatmap, and overlay instead use `|I_t - I_hat_(t+1)|`. The repository names this a **last-input/prediction difference map** rather than a target prediction-error map to avoid conflating the two operations.

## Outputs

The evaluator writes raw and normalized scores, PSNR values, evaluated target indices, optional labels, `per_frame_results.csv`, `test_metrics_and_memory.json`, an anomaly-score plot, MTAR memory-usage diagnostics, optional MTAR memory visualizations, and optional per-frame visualizations.
