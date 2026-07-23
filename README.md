# Meta Transfer Learning Project

## Overview

This project reuses Meta’s pretrained discrete-gesture EMG decoder on University of Utah prosthetics data. A lightweight TCN adapter is trained to translate the Utah recordings into the input expected by Meta’s model. The Meta backbone remains frozen with weights from open source .ckpt file.

The repository supports two experiments:

Zero_Shot_Meta.ipynb — no-training baseline using a fixed Utah-to-Meta input conversion.
Meta_TL.ipynb — main experiment; trains only the Utah adapter while preserving the pretrained Meta weights.

## Workflow

```text
FeedbackDecode/Trellis recording session
  ├─ TrainingData_*.kdf
  ├─ recording.ns5
  ├─ matching recording.ns2
  └─ RecStart_<session-id>.mat or SSStruct_<session-id>.mat
       ├─ readKDF.m
       │    └─ trainNIPtime and trainKin
       │         └─ MISSING kdf export step
       │              └─ kdf(andother).h5 (write new ns5->h5 code)
       └─ load_ns5_segments.m
            ├─ +unrl_utils/fastNSxRead.m
            ├─ +project_utils/CalculateNIPOffset_bhm.m
            ├─ matching .ns2 synchronization file
            └─ returned scaled training segment
                 └─ MISSING ns5 export step (write new ns5->h5 code)
                      └─ ns5_training.h5

kdf(andother).h5 + ns5_training.h5
  └─ Data_Alignment.ipynb
       └─ Aligned_Train_Data.pt
            └─ Aligned_Train_Data_Preprocessed.pt
                 └─ Labeled_Dataset.ipynb
                      └─ Gesture_Trial_Dataset_Labeled.pt
                           ├─ Zero_Shot_Meta.ipynb
                           └─ Meta_TL.ipynb

```

## Starting from Gesture_Trial_Dataset_Labeled.pt

1. Install the Meta package and its environment.
2. Place model_checkpoint.ckpt in emg_models/discrete_gestures/.
3. Update REPO_ROOT, BASE_DIR, DATA_PATH, and CKPT_PATH in the notebooks.
4. Run Zero_Shot_Meta.ipynb to establish the no-training baseline.
5. Run Meta_TL.ipynb to train and evaluate the adapter.

## Starting from raw KDF and NS5 recordings

The raw-to-training path is not yet fully reproducible. Before treating it as a complete pipeline, recover or implement:

1. The exact +unrl_utils/fastNSxRead.m used by load_ns5_segments.m, or standardize the project on the available reader.
2. A MATLAB export code for /trainNIPtime and /trainKin to the KDF H5 file.
3. A MATLAB driver that calls load_ns5_segments.m and writes /data to ns5_training.h5

## Raw recording organization

Keep all files from a recording session together. load_ns5_segments.m derives experiment_id from the session folder name and searches that folder for RecStart_<experiment_id>.mat or SSStruct_<experiment_id>.mat. It derives the synchronization filename by replacing .ns5 with .ns2, so the .ns2 and .ns5 files must share the same stem.

## File and data contracts
### KDF H5 input

Data_Alignment.ipynb reads only:

/trainNIPtime — one-dimensional after flattening
/trainKin — time × kinematic columns, with at least 7 columns

kdf(andother).h5 also contains NIPtime, features, kinematics, targets, trainFeat, and trainTargets, but those datasets are not used by the alignment notebook.

### NS5 H5 input

Data_Alignment.ipynb expects:

/data — either 32 × samples or samples × 32

ns5_training.h5 is float32 with shape 32 × 5,851,861. It contains only the training interval represented by trainNIPtime, not full recording session.

### Aligned PyTorch output

Data_Alignment.ipynb writes:

ns5_sample — one-indexed rows within the training segment
trainNIPtime — one NIP timestamp per NS5 sample
ns5_vector — time × 32, float32
trainKin — time × 7, one-hot labels with all-zero rest rows

The notebook forward-fills each KDF label across the intervening 30 kHz NS5 samples. It uses trainNIPtime[0] as the first timestamp of the NS5 segment; this is valid only when ns5_training.h5 was extracted using the same first and last KDF timestamps.

### Labeled trial dataset

Labeled_Dataset.ipynb reads Aligned_Train_Data_Preprocessed.pt and writes Gesture_Trial_Dataset_Labeled.pt. It:

Keeps gestures 0–4.
Builds nine label columns; the last four remain zero.
Detects trials from active label blocks.
Uses the EMG envelope to refine labels and creates artifact masks.
Applies manually selected validation and test trials.
Stores train, val, test, and all_trials.

The currently documented split is:

G0 — validation trial 7; test trial 8
G1–G4 — validation trial 4; test trial 5
(chosen manually after inspecting cleanest trials)
All remaining usable trials — training

## Training requirements

Meta_TL.ipynb requires:
Gesture_Trial_Dataset_Labeled.pt

An importable: generic_neuromotor_interface.networks.DiscreteGesturesArchitecture
emg_models/discrete_gestures/model_checkpoint.ckpt

Utah recordings are downsampled from 30 kHz to 2 kHz and use the same preprocessing as Meta (40 Hz high-pass filter and normalizing st dev to 1)
After the first conv layer of Meta model, the 2kHz is downsampled to 198 time bins per second.
Predictions can be evaluated as complete-trial or across all active time bins (usually 50-80 per trial).
## Current local directory layout

```text
utah-neuro/
├─ generic_neuromotor_interface/
│  ├─ config/
│  ├─ emg_data/
│  ├─ emg_models/discrete_gestures/
│  │  ├─ model_checkpoint.ckpt
│  │  └─ model_config.yaml
│  ├─ generic_neuromotor_interface/
│  ├─ notebooks/
│  ├─ results/
│  ├─ Gesture_Trial_Dataset_Labeled.pt
│  ├─ Meta_TL.ipynb
│  ├─ Zero_Shot_Frozen_Meta.ipynb
│  ├─ environment.yml
│  └─ setup.py
├─ MATLAB_Jupyter/
│  ├─ +unrl_utils/
│  ├─ Data_Alignment.ipynb
│  ├─ Labeled_Dataset.ipynb
│  ├─ convertNSxOutputToH5.m     (perhaps missing ns5->h5 export code)
│  ├─ load_ns5_segments.m
│  ├─ loadRawData.m
│  ├─ readEventParamsFile.m
│  ├─ readKDF.m
│  ├─ kdf(andother).h5
│  ├─ ns5.h5
│  └─ RecStart_MATLAB.mat
├─ README-project-layout.md
├─ .gitignore
└─ .gitattributes

```

