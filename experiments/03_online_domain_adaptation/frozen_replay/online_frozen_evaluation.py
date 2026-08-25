"""Frozen, trial-by-trial evaluation for 02_current_meta_tl.ipynb.

Run this file from the same live Jupyter kernel after Cell 5 has restored the
best validation-selected adapter:

    %run C:/Users/Micah/Documents/Codex/2026-07-28/i/outputs/online_frozen_evaluation.py

No parameter, buffer, optimizer, or normalization statistic is updated.
"""

from pathlib import Path
import copy
import csv
import hashlib
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


# Edit this list if you want to add/remove historical datasets.
ONLINE_DATASETS = [
    Path(r"C:\Users\Micah\utah-neuro\generic_neuromotor_interface\New_Gesture_Trial_Dataset_Labeled.pt"),
    Path(r"C:\Users\Micah\utah-neuro\generic_neuromotor_interface\Gesture_Trial_Dataset_Labeled.pt"),
]

ONLINE_OUTPUT_DIR = Path(__file__).resolve().parent / "online_eval_results"
ONLINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_CHECKPOINT_PATH = ONLINE_OUTPUT_DIR / "best_frozen_meta_adapter.pt"


REQUIRED_GLOBALS = [
    "model", "best_adapter_state", "best_epoch", "best_val_accuracy",
    "best_val_task_loss", "saved", "GestureOneSecond2kHzDataset",
    "extract_one_second_active_centered_trial", "map_meta_outputs_to_utah",
    "align_target_and_mask_to_logits", "trial_predictions", "ao_unpack_batch",
    "AO_CLASS_NAMES", "AO_KEPT_CLASSES", "UTAH_TO_META_OUTPUTS", "DEVICE",
]
missing = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "Run this in the same live kernel after 02_current_meta_tl.ipynb Cell 5. "
        f"Missing names: {missing}"
    )

if best_adapter_state is None:
    raise RuntimeError("The best validation-selected adapter is unavailable.")

# Restore the selected state explicitly, then freeze the entire composite model.
model.adapter.load_state_dict(copy.deepcopy(best_adapter_state))
model.to(DEVICE)
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)

if any(parameter.requires_grad for parameter in model.parameters()):
    raise RuntimeError("At least one model parameter is still trainable.")


def state_digest(module):
    """SHA-256 over names, dtypes, shapes, parameters, and persistent buffers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


frozen_digest_before = state_digest(model)
torch.save(
    {
        "format_version": 1,
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_val_accuracy),
        "best_validation_task_bce": float(best_val_task_loss),
        "utah_to_meta_outputs": list(UTAH_TO_META_OUTPUTS),
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "model_sha256": frozen_digest_before,
    },
    FROZEN_CHECKPOINT_PATH,
)
print("Saved immutable evaluation checkpoint:", FROZEN_CHECKPOINT_PATH)


def iter_saved_trials(dataset_object):
    """Yield (original_split, trial) without re-splitting historical data."""
    if isinstance(dataset_object, dict):
        known = [key for key in ("train", "val", "test") if key in dataset_object]
        if known:
            for split_name in known:
                for trial in dataset_object[split_name]:
                    yield split_name, trial
            return
        if "trials" in dataset_object:
            for trial in dataset_object["trials"]:
                yield "trials", trial
            return
    if isinstance(dataset_object, (list, tuple)):
        for trial in dataset_object:
            yield "all", trial
        return
    raise TypeError(
        "Expected a list of trials or a dict containing train/val/test or trials; "
        f"got {type(dataset_object).__name__}."
    )


def trial_fingerprint(trial):
    """Hash the exact one-second raw EMG crop used by this notebook."""
    extracted = extract_one_second_active_centered_trial(trial)
    if extracted is None:
        return None
    emg_tc = extracted[0].detach().cpu().contiguous().numpy()
    return hashlib.sha256(emg_tc.tobytes()).hexdigest()


print("Building leakage guard from every trial in the model-development dataset...")
development_hashes = set()
for development_split in ("train", "val", "test"):
    for development_trial in saved[development_split]:
        fingerprint = trial_fingerprint(development_trial)
        if fingerprint is not None:
            development_hashes.add(fingerprint)
print("Development trial fingerprints:", len(development_hashes))


all_rows = []
dataset_summaries = []

for dataset_path in ONLINE_DATASETS:
    if not dataset_path.exists():
        print("SKIP missing dataset:", dataset_path)
        continue

    print("\nLoading historical dataset:", dataset_path)
    historical = torch.load(dataset_path, map_location="cpu", weights_only=False)
    ordered_trials = list(iter_saved_trials(historical))
    del historical

    correct = 0
    evaluated = 0
    skipped = 0
    overlap_count = 0
    confusion = torch.zeros(AO_KEPT_CLASSES, AO_KEPT_CLASSES, dtype=torch.long)
    dataset_rows = []

    # A one-item loader preserves dataset order and mimics one arriving trial.
    for arrival_index, (original_split, trial) in enumerate(ordered_trials, start=1):
        fingerprint = trial_fingerprint(trial)
        if fingerprint is None:
            skipped += 1
            continue

        overlaps_development = fingerprint in development_hashes
        overlap_count += int(overlaps_development)

        stream_dataset = GestureOneSecond2kHzDataset(
            [trial],
            f"{dataset_path.stem}:{original_split}",
        )
        stream_loader = DataLoader(
            stream_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        batch = next(iter(stream_loader))
        emg, target, valid_mask, gestures, trial_nums = ao_unpack_batch(batch)

        if tuple(emg.shape[1:]) != (32, 2000):
            raise RuntimeError(
                f"Unexpected input shape after preprocessing: {tuple(emg.shape)}"
            )

        start = time.perf_counter()
        with torch.inference_mode():
            raw_meta_output = model(emg)
            logits = map_meta_outputs_to_utah(raw_meta_output)
            target, valid_mask = align_target_and_mask_to_logits(
                target, valid_mask, logits.shape[-1]
            )
            result = trial_predictions(
                logits, target, valid_mask, gestures
            )[0]
        latency_ms = 1000.0 * (time.perf_counter() - start)

        true_class = int(result["true"])
        predicted_class = int(result["prediction"])
        is_correct = predicted_class == true_class
        evaluated += 1
        correct += int(is_correct)
        confusion[true_class, predicted_class] += 1

        row = {
            "dataset": dataset_path.name,
            "arrival_index": arrival_index,
            "original_split": original_split,
            "trial_num": int(trial_nums[0].item()),
            "true_class": true_class,
            "true_name": AO_CLASS_NAMES[true_class],
            "predicted_class": predicted_class,
            "predicted_name": AO_CLASS_NAMES[predicted_class],
            "correct": int(is_correct),
            "cumulative_accuracy": correct / evaluated,
            "active_bins": int(result["active_bins"]),
            "latency_ms": latency_ms,
            "overlaps_development_data": int(overlaps_development),
        }
        for class_index, score in enumerate(result["normalized_scores"].tolist()):
            row[f"score_{class_index}_{AO_CLASS_NAMES[class_index].replace(' ', '_')}"] = score
        dataset_rows.append(row)
        all_rows.append(row)

        print(
            f"{dataset_path.stem} | arrival {arrival_index:03d} | "
            f"true={row['true_name']:<11} pred={row['predicted_name']:<11} | "
            f"running accuracy={correct}/{evaluated} ({correct/evaluated:.1%})"
            + (" | OVERLAP" if overlaps_development else "")
        )

    accuracy = correct / evaluated if evaluated else float("nan")
    nonoverlap_rows = [row for row in dataset_rows if not row["overlaps_development_data"]]
    nonoverlap_accuracy = (
        sum(row["correct"] for row in nonoverlap_rows) / len(nonoverlap_rows)
        if nonoverlap_rows else float("nan")
    )
    summary = {
        "dataset": dataset_path.name,
        "trials_present": len(ordered_trials),
        "trials_evaluated": evaluated,
        "trials_skipped": skipped,
        "development_overlaps": overlap_count,
        "accuracy_all": accuracy,
        "nonoverlap_trials": len(nonoverlap_rows),
        "accuracy_nonoverlap": nonoverlap_accuracy,
        "median_inference_latency_ms": (
            float(np.median([row["latency_ms"] for row in dataset_rows]))
            if dataset_rows else float("nan")
        ),
        "confusion_true_rows_predicted_columns": confusion.tolist(),
    }
    dataset_summaries.append(summary)
    print("Summary:", json.dumps(summary, indent=2))


if not all_rows:
    raise RuntimeError("No historical trials were evaluated.")

csv_path = ONLINE_OUTPUT_DIR / "online_trial_predictions.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)

summary_path = ONLINE_OUTPUT_DIR / "online_summary.json"
with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(dataset_summaries, handle, indent=2)

frozen_digest_after = state_digest(model)
if frozen_digest_after != frozen_digest_before:
    raise RuntimeError("Model state changed during online evaluation.")

print("\nFrozen-state verification passed:", frozen_digest_after)
print("Trial-level results:", csv_path)
print("Dataset summaries:", summary_path)
print(
    "Use accuracy_nonoverlap as the cleanest external estimate whenever "
    "development_overlaps is nonzero."
)
