# Meta Transfer Learning Project

## Overview

This project reuses Meta's pretrained discrete-gesture EMG decoder on University of Utah prosthetics data. A lightweight TCN adapter translates Utah recordings into the input expected by Meta's model while the pretrained Meta backbone remains frozen.

The repository supports two experiments:

- `Zero_Shot_Meta.ipynb`: no-training baseline using a fixed Utah-to-Meta input conversion.
- `Meta_TL.ipynb`: trains only the Utah adapter while preserving the pretrained Meta weights.

## Data pipeline

```text
FeedbackDecode/Trellis recording session
  |-- TrainingData_*.kdf
  |-- <session-id>.ns5
  |-- <session-id>.ns2
  `-- RecStart_<session-id>.mat or SSStruct_<session-id>.mat
       |
       `-- run_alignment_example.m
            `-- alignAndExportKDFNS5.m
                 |-- readKDF.m
                 |-- CalculateNIPOffset.m
                 |-- fastNSxRead.m
                 |-- convertKDFToH5.m
                 `-- convertNSxOutputToH5.m
                      |
                      `-- aligned_h5/
                           |-- TrainingData_*_kdf.h5
                           `-- TrainingData_*_ns5_aligned.h5
                                |
                                `-- Data_Alignment.ipynb
                                     `-- Aligned_Train_Data_Preprocessed.pt
                                          `-- Labeled_Dataset.ipynb
                                               `-- Gesture_Trial_Dataset_Labeled.pt
                                                    |-- Zero_Shot_Meta.ipynb
                                                    `-- Meta_TL.ipynb
```

## Starting from raw KDF and NS5 recordings

Keep these recording files together:

- `TrainingData_*.kdf`
- Matching `.ns5` and `.ns2` files with the same session stem
- `RecStart_<session-id>.mat` or `SSStruct_<session-id>.mat`

Keep these MATLAB files on the MATLAB path:

- `run_alignment_example.m`
- `alignAndExportKDFNS5.m`
- `convertKDFToH5.m`
- `convertNSxOutputToH5.m`
- `readKDF.m`
- `CalculateNIPOffset.m`
- `fastNSxRead.m`

Edit the session directory and filenames in `run_alignment_example.m`, then run:

```matlab
run_alignment_example
```

For another KDF interval from the same recording, change only `kdfFile` and rerun. The KDF timestamps automatically select the corresponding NS5 interval.

### Timestamp alignment

KDF `NIPTime` and NS5 samples use the 30 kHz NIP clock. The synchronization offset is calculated from the matching NS2 and `RecStart`/`SSStruct` file:

```text
absolute NS5 sample = KDF NIPTime + NIP offset
local aligned NS5 row = KDF NIPTime - first KDF NIPTime + 1
```

The exporter writes only the NS5 interval spanning the first through last KDF timestamp.

## HDF5 contracts

### KDF HDF5

`convertKDFToH5.m` writes Python-facing records-by-variables datasets:

- `/trainNIPtime`
- `/trainKin`
- `/trainFeat`
- `/trainTargets`
- `/trainKalman`

`Data_Alignment.ipynb` primarily uses `/trainNIPtime` and `/trainKin`.

### NS5 HDF5

`convertNSxOutputToH5.m` writes:

- `/data`: raw `int16` neural data
- Alignment and source metadata as HDF5 attributes

The MATLAB display reports channels by samples. Python/HDF5 consumers should verify orientation and transpose to time by channels when necessary.

### Aligned PyTorch output

`Data_Alignment.ipynb` preprocesses and aligns the HDF5 inputs, producing:

- `ns5_sample`: one-indexed row within the aligned NS5 segment
- `trainNIPtime`: one NIP timestamp per NS5 sample
- `ns5_vector`: time by 32 channels, `float32`
- `trainKin`: one-hot gesture labels with all-zero rest rows

KDF labels are forward-filled across intervening 30 kHz NS5 samples.

## Label creation

`Labeled_Dataset.ipynb` reads `Aligned_Train_Data_Preprocessed.pt` and writes `Gesture_Trial_Dataset_Labeled.pt`.

Current behavior:

- Keeps gesture classes 0-4 and builds nine label columns; the final four remain zero.
- Computes a 25 ms moving-average envelope from mean absolute EMG across channels.
- Estimates the resting baseline with the median and MAD-derived robust standard deviation.
- Searches from 100 ms before the original label through at most 700 ms after its end.
- Requires at least 12 ms above `median + 1.5 * robust sigma`.
- Selects the candidate burst with the greatest integrated activity above threshold.
- Expands burst boundaries while above `median + 0.75 * robust sigma`.
- Pads detected bursts by 40 ms before and 100 ms after.
- Extends the stored original prompt label 100 ms forward.
- Disables artifact masking and extreme-spike suppression. `valid_mask` stays one; compatibility mask arrays stay zero.
- Plots all 20 trials per gesture for review.
- Stores `train`, `val`, `test`, and `all_trials` splits.

The currently documented manual split is:

- G0: validation trial 7; test trial 8
- G1-G4: validation trial 4; test trial 5
- All remaining usable trials: training

## Starting from `Gesture_Trial_Dataset_Labeled.pt`

1. Install the Meta package and environment.
2. Place `model_checkpoint.ckpt` in `emg_models/discrete_gestures/`.
3. Update `REPO_ROOT`, `BASE_DIR`, `DATA_PATH`, and `CKPT_PATH` in the notebooks.
4. Run `Zero_Shot_Meta.ipynb` for the no-training baseline.
5. Run `Meta_TL.ipynb` to train and evaluate the adapter.

## Training requirements

`Meta_TL.ipynb` requires:

- `Gesture_Trial_Dataset_Labeled.pt`
- Importable `generic_neuromotor_interface.networks.DiscreteGesturesArchitecture`
- `emg_models/discrete_gestures/model_checkpoint.ckpt`

Utah recordings are sampled at 30 kHz, preprocessed, and downsampled to 2 kHz for the Meta model. The EMG preprocessing uses a 59-61 Hz notch, 1 kHz low-pass, 40 Hz high-pass, and per-channel standard-deviation normalization. After Meta's first convolutional layer, 2 kHz input becomes 198 time bins per second.

Predictions can be evaluated per complete trial or across active time bins.
