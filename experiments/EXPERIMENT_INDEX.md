# Utah EMG Experiment Index

The MATLAB/Jupyter workspace is organized by experimental method. Large `.pt`
datasets remain in `generic_neuromotor_interface` so existing dataset paths stay
stable.

## 00_data_preparation

- Alignment, labeling, and dataset-building notebooks
- MATLAB alignment/export helpers
- `session_20260723/`: original NSx/KDF session files
- `aligned_h5/`: aligned intermediate datasets

Notebooks:

- `01_align_utah_recordings.ipynb`
- `02_label_original_utah_dataset.ipynb`
- `03_label_updated_utah_dataset.ipynb`

## 01_meta_tl_baseline

- Current Meta-TL pipeline and baseline controls
- Meta-TL versus standalone decoder comparison
- Baseline checkpoints, histories, confusion matrices, and figures

Notebooks:

- `02_current_meta_tl.ipynb` (primary baseline)
- `03_compare_meta_tl_vs_standalone.ipynb`
- `04_standalone_decoder_append_cells.ipynb`
- `05_task_only_meta_adapter.ipynb` (Meta frozen; Utah adapter trained)
- `06_zero_shot_frozen_meta.ipynb` (strict no-training control)

## 02_rotation_invariance

- Rotation-invariant Meta adapter training
- Frozen cross-day evaluation
- Training and cross-day results

Notebooks:

- `01_train_rotation_invariant_meta_tl.ipynb`
- `02_test_rotation_invariant_cross_day.ipynb`

## 03_online_domain_adaptation

- Blind burst detector and full label-blind pipeline
- Frozen replay evaluation
- CORAL, GAN, and DANN experiments
- Checkpoints, training histories, and online/cross-day results

Method notebooks:

- `meta_coral_experiment/train_coral_domain_adaptation.ipynb`
- `meta_coral_experiment/00_meta_locked_da_audited_baseline.ipynb`
  (audited task-only/CORAL-toggle baseline)
- `meta_coral_experiment/test_frozen_coral_online.ipynb`
- `adversarial_experiments/gan/train_gan_domain_adaptation.ipynb`
- `adversarial_experiments/gan/test_frozen_gan_cross_day.ipynb`
- `adversarial_experiments/gan/test_frozen_gan_online.ipynb`
- `adversarial_experiments/dann/train_dann_domain_adaptation.ipynb`
- `adversarial_experiments/dann/test_frozen_dann_online.ipynb`

## 04_representation_learning

- Utah-only, prototype, and prototype-plus-relational ablations
- Unpaired Meta-teacher/Utah-student distillation
- Meta-Utah joint embedding
- Learning-speed and cross-day epoch-trajectory experiments

Notebooks:

- `01_train_joint_embedding.ipynb`
- `02_compare_learning_speed.ipynb`
- `03_optimize_native_frozen_teacher.ipynb`

## Dataset locations kept stable

- `generic_neuromotor_interface/New_Gesture_Trial_Dataset_Labeled (2).pt`:
  main same-day Utah dataset
- `generic_neuromotor_interface/Gesture_Trial_Dataset_Labeled.pt`:
  separate-day evaluation dataset
- `generic_neuromotor_interface/emg_data/`: Meta source dataset
