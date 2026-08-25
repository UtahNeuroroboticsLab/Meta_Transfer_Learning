# Meta Transfer Learning Project

## Overview

This project studies gesture decoding from University of Utah prosthetics EMG
recordings. It includes the data-creation workflow, adaptations of Meta's
released neuromotor model, Utah-only baselines, and evaluations of robustness
across sessions and online-style conditions.

## Repository structure

```text
data_preparation/
  current_pipeline/       Current alignment and labeling workflow
  original_pipeline/      Original repository data-creation workflow
experiments/
  01_meta_tl_baseline/    Transfer-learning and Utah-only reference models
  02_rotation_invariance/ Invariant-adapter experiments
  03_online_domain_adaptation/
                          Online evaluation and domain-adaptation methods
  04_representation_learning/
                          Distillation and joint-embedding experiments
```

The [experiment index](experiments/EXPERIMENT_INDEX.md) provides the notebook
map. Each method folder keeps its compact metrics and figures next to the code
that produced them.

## Data and model requirements

The workflows use aligned Utah EMG recordings and, where applicable, Meta's
released discrete-gesture model. Raw recordings, generated PyTorch datasets,
and trained weights are intentionally excluded from Git because of their size
and sensitivity.

Before running a notebook, update its local data and checkpoint paths for your
machine. Begin in `data_preparation/current_pipeline` when starting from raw
recordings, or in the relevant `experiments` folder when starting from an
already labeled dataset.
