# Utah EMG experiment archive

This directory is the version-controlled snapshot of the recent REU methods and
their compact results. Start with `EXPERIMENT_INDEX.md` for the experiment map.

## Contents

- `01_meta_tl_baseline`: original Meta transfer-learning baseline and comparisons
- `02_rotation_invariance`: learnable-adapter rotational-invariance experiments
- `03_online_domain_adaptation`: frozen online evaluation, CORAL, GAN, and DANN
- `04_representation_learning`: frozen-teacher distillation, joint embedding, and
  Utah-only learning-speed comparisons

The alignment and labeling workflows are kept in the sibling
[`data_preparation`](../data_preparation) directory.

Notebooks, scripts, metric tables, configuration files, and PNG result figures are
included. Raw EMG recordings, generated datasets, model checkpoints, notebook
autosaves, and duplicate vector/PDF figures are intentionally excluded because
they are large or reproducible. Existing notebooks may contain local absolute
paths and should be updated for another machine before running.
