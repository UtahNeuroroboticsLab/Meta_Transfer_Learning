# Representation learning experiments

## Current code

- `01_train_joint_embedding.ipynb` — jointly trains Meta and Utah branches in a shared embedding space
- `02_compare_learning_speed.ipynb` — compares validation learning speed across completed methods
- `03_optimize_native_frozen_teacher.ipynb` — final frozen-Meta distillation experiment with native 512-D embeddings, active-bin pooling, official preprocessing, and validation-only loss-weight search
- `frozen_meta_teacher_native.py` — loads Meta's released frozen checkpoint and caches its class targets

## Retained results

- `results/joint_embedding` — joint-embedding metrics and figures
- `results/learning_speed` — learning-speed comparison tables and figure
- `results/trained_teacher_baseline` — matched Utah-only, prototype-only, and prototype-plus-relational baseline summaries
- `results/native_frozen_teacher` — final official frozen-teacher grid and untouched evaluations

Large model checkpoints were removed because every retained experiment can regenerate them. Superseded PCA-teacher runs, duplicate unpaired runs, and per-epoch cross-day checkpoints were also removed.

## Main conclusion

The final frozen official Meta teacher matched the Utah-only model at 50% same-day and 25% cross-day accuracy. The earlier trained-teacher prototype-plus-relational run reached 31.25% cross-day accuracy, while the joint-embedding model remained near chance cross-day. None reliably outperformed the standalone Utah approach.
