# Meta_Transfer_Learning
# Utah Neuromotor Interface: MATLAB and Jupyter Workflows

This repository contains the analysis and modeling workflow used to adapt a frozen Meta discrete-gesture model to Utah sEMG recordings.

## Contents

- `Data_Alignment.ipynb` — Aligns raw KDF and NS5/HDF5 recordings into an analysis-ready dataset.
- `Labeled_Dataset.ipynb` — Creates gesture labels and fixed train/validation/test splits.
- `Zero_Shot_Meta.ipynb` — Frozen-model baseline with a fixed 32-to-16 channel bridge; no training.
- `Meta_TL.ipynb` — Primary task-only transfer-learning experiment. The Meta backbone remains frozen; only the Utah adapter is trained.
- `DA_Meta_TL.ipynb` — Domain-adaptation version of the transfer-learning experiment, with optional CORAL alignment.
- `Optimized_Meta_Lockd_DA.ipynb` — Earlier optimized domain-adaptation experiment for comparison and development.

## Recommended workflow

1. Run `Data_Alignment.ipynb` only when rebuilding data from raw recordings.
2. Run `Labeled_Dataset.ipynb` to reproduce labels and fixed splits.
3. Run `Zero_Shot_Meta.ipynb` to establish a no-training baseline.
4. Run `Meta_TL.ipynb` to reproduce the main experiment.
5. Run `DA_Meta_TL.ipynb` after reproducing the task-only result.

## Setup

Install the project environment and dependencies, including PyTorch, NumPy, SciPy, h5py, Matplotlib, and Pandas. A CUDA-enabled PyTorch installation is recommended for training.

Update the paths at the top of each notebook for your local environment. The model notebooks require:

- the `generic_neuromotor_interface` package;
- the pretrained Meta checkpoint;
- `Gesture_Trial_Dataset_Labeled.pt`;
- raw KDF/NS5/HDF5 data only when regenerating aligned data.

## Important conventions

- Utah recordings are processed from 30 kHz to 2 kHz using anti-aliased resampling and a 40 Hz high-pass filter.
- Each example is a one-second window centered on the final active-valid label interval.
- The Meta output mapping is `[5, 6, 7, 8, 4]`.
- Target alignment uses the model’s exact left context and stride.
- Do not silently exclude validation or test trials.
- Report task BCE and complete-trial mean-logit accuracy. Label active-bin accuracy separately.

## Reproducibility

For every run, record the notebook and commit, checkpoint path, dataset version, random seed, hardware/software versions, split counts, crop audit, diagnostic output, and best-validation checkpoint.

## Data handling

Do not commit large recordings, checkpoints, `.pt`, `.h5`, or `.mat` files unless large-file storage is configured. Keep restricted or participant-sensitive data in approved storage.
