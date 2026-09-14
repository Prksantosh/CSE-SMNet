# Source-to-repository mapping

| Supplied research file | Official repository location | Revised public name / role |
|---|---|---|
| `autoencoder_skip.py` | `cse_smnet/models/cse_smnet.py` | `CSESMNet` |
| `aerhcnet.py` | `cse_smnet/models/spatial/cse_block.py` | `CSEBlock` |
| `encoder.py` | `cse_smnet/models/encoder.py` | `EncoderStage` |
| `decoder.py` | `cse_smnet/models/decoder.py` | `DecoderStage` |
| `mem_GALSTM_single_cell_usage_regularized.py` | `cse_smnet/models/temporal/mtar.py` | `MTAR`, `MTARCell` |
| `timestamp(2).py` | `cse_smnet/models/temporal/timestamp.py` | `TimestampTransform` |
| `losses.py` | `cse_smnet/losses/prediction.py` | prediction-loss components |
| `avenue_dataset.py` | `cse_smnet/data/avenue.py` | `AvenueDataset` |
| `config.py` + training-script constants | `configs/avenue_train.py` | `AvenueTrainConfig` |
| `train_mtar_save_memory_bank_each_epoch.py` | `train.py` | official training entry point |
| same training script: epoch/validation logic | `cse_smnet/engine/trainer.py` | `CSESMNetTrainer` |
| same training script: video split | `cse_smnet/utils/data_split.py` | video-level split utilities |
| same training script: MTAR diagnostics/snapshots | `cse_smnet/utils/memory.py` | MTAR memory utilities |
| same training script: RNG handling | `cse_smnet/utils/reproducibility.py` | reproducibility utilities |
| same training script: PSNR | `cse_smnet/utils/metrics.py` | development metrics |
| `test_mtar_lastinput_prediction_error_maps(1).py` | `test.py`, `configs/evaluation.py`, `cse_smnet/data/video_sequence.py`, `cse_smnet/engine/evaluator.py`, `cse_smnet/utils/{checkpoint,metrics,memory,visualization}.py` | Modularized validated future-frame evaluation; legacy RHCNet naming removed; target-vs-prediction PSNR scoring preserved; last-input/prediction diagnostic map renamed for clarity. |

Legacy registered submodule names `cgcde` and `e3d` remain internal where needed so existing trained state dictionaries load without key conversion. Public access uses the revised CSE-SMNet terminology.
