# Meta_Transfer_Learning

This repository has the workflow used to adapt a frozen Meta discrete-gesture model to Utah sEMG recordings.

## Contents

- `Data_Alignment.ipynb` — Aligns raw KDF and NS5/HDF5 recordings into an analysis-ready dataset.
- `Labeled_Dataset.ipynb` — Creates gesture labels and fixed train/validation/test splits.
- `Zero_Shot_Meta.ipynb` — Frozen-model baseline with a fixed 32-to-16 channel bridge; no training.
- `Meta_TL.ipynb` — Primary task-only transfer-learning experiment. The Meta backbone remains frozen; only the Utah adapter is trained.
- `load_ns5_segments.m` — aligns kdf and ns5 on trainNIPtime timestamps
- `readKDF.m` — extracts kdf data
- `convertNSXtoh5.m` — converts ns5 data from FastNSXread to h5 for python

## Workflow

1. Run `Data_Alignment.ipynb` only when rebuilding data from raw recordings.
2. Run `Labeled_Dataset.ipynb` to reproduce labels and fixed splits.
3. Run `Zero_Shot_Meta.ipynb` to establish a no-training baseline.
4. Run `Meta_TL.ipynb` to reproduce the main experiment.

## Setup

Install the project environment and dependencies, including PyTorch, NumPy, SciPy, h5py, Matplotlib, and Pandas.
Update the paths at the top of each notebook for your local environment. The model notebooks require:

- the `generic_neuromotor_interface` package (from Meta repo)
- the pretrained Meta checkpoint (.ckpt file)
- `Gesture_Trial_Dataset_Labeled.pt` (made by Labeled_Dataset.ipynb)
- raw KDF/NS5/HDF5 data only when regenerating aligned data (recorded on feedback decode/trellis software)

- Utah recordings are downsampled from 30 kHz to 2 kHz and use the same preprocessing as Meta (40 Hz high-pass filter and normalizing st dev to 1)
- After the first conv layer of Meta model, the 2kHz is downsampled to 198 time bins
- Most active labels (the sequence of time bins during which a gesture was prompted) last for 50-80 time bins
- The training cell (in Meta_TL.ipynb) reports all-or-nothing accuracy for each trial by making one prediction for the highest accumulation of probability across all time bins. A following cell reports accuracy across predictions at each of the trial's active time bins.
