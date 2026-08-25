"""Label-blind EMG burst detector and benchmark.

The detector sees only continuous EMG. Final auto-relabeled activity labels
are used only after all predicted events have been finalized.

Run:
    python blind_emg_burst_detector.py

Optionally pass another labeled trial dataset with contiguous ``all_trials``:
    python blind_emg_burst_detector.py path/to/labeled_dataset.pt
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_INPUT = Path(
    r"C:\Users\Micah\utah-neuro\generic_neuromotor_interface\New_Gesture_Trial_Dataset_Labeled.pt"
)
INPUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
OUTPUT_DIR = Path(__file__).resolve().parent / "blind_burst_results_tuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 30_000
KEPT_CLASSES = 5

# Detector settings, adapted from 02_label_original_utah_dataset.ipynb.
ENVELOPE_WINDOW_MS = 25
CALIBRATION_SECONDS = 10.0
HIGH_MAD_MULT = 1.5
LOW_MAD_MULT = 0.75
MIN_BURST_MS = 12
PRE_PAD_MS = 40
POST_PAD_MS = 120
MERGE_GAP_MS = 300


def force_emg_time_channels(value: object) -> np.ndarray:
    x = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D EMG, got {x.shape}")
    if x.shape[1] == 32:
        return x
    if x.shape[0] == 32:
        return x.T
    raise ValueError(f"Could not identify 32-channel EMG orientation: {x.shape}")


def force_labels_time_features(value: object, expected_time: int) -> np.ndarray:
    x = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D labels, got {x.shape}")
    if x.shape[0] == expected_time:
        return x
    if x.shape[1] == expected_time:
        return x.T
    raise ValueError(f"Labels {x.shape} do not match EMG length {expected_time}")


def intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def causal_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Past-only rolling mean with a progressively filled initial window."""
    x = np.asarray(x, dtype=np.float64)
    cumulative = np.cumsum(x, dtype=np.float64)
    result = cumulative.copy()
    if window < len(x):
        result[window:] -= cumulative[:-window]
    denominators = np.minimum(np.arange(1, len(x) + 1), window)
    result /= denominators
    return result.astype(np.float32)


def robust_med_sigma(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float32)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma = 1.4826 * mad
    if sigma < 1e-8:
        sigma = float(np.std(x) + 1e-8)
    return median, sigma


def expand_intervals(
    source: list[tuple[int, int]], before: int, after: int, length: int
) -> list[tuple[int, int]]:
    return [(max(0, start - before), min(length, end + after)) for start, end in source]


def merge_intervals(
    source: list[tuple[int, int]], max_gap: int = 0
) -> list[tuple[int, int]]:
    if not source:
        return []
    ordered = sorted(source)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + max_gap:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def mask_from_intervals(source: list[tuple[int, int]], length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in source:
        mask[start:end] = True
    return mask


def detect_bursts_from_emg(emg_tc: np.ndarray) -> dict[str, object]:
    """Complete label-blind detection stage."""
    sample_count = len(emg_tc)
    raw_envelope = np.mean(np.abs(emg_tc), axis=1, dtype=np.float64).astype(np.float32)
    envelope_window = max(1, round(ENVELOPE_WINDOW_MS * FS / 1000))
    env = causal_moving_average(raw_envelope, envelope_window)

    calibration_samples = min(sample_count, round(CALIBRATION_SECONDS * FS))
    baseline_median, baseline_sigma = robust_med_sigma(env[:calibration_samples])
    high_threshold = baseline_median + HIGH_MAD_MULT * baseline_sigma
    low_threshold = baseline_median + LOW_MAD_MULT * baseline_sigma

    # Artifact masking is intentionally disabled. Large EMG features are
    # candidates for true activity and must remain visible to the detector.
    detector_env = env

    minimum_samples = max(1, round(MIN_BURST_MS * FS / 1000))
    high_runs = [
        (start, end)
        for start, end in intervals(detector_env > high_threshold)
        if end - start >= minimum_samples
    ]

    # Grow every confidently detected run through the lower hysteresis level.
    grown = []
    for start, end in high_runs:
        while start > 0 and detector_env[start - 1] > low_threshold:
            start -= 1
        while end < sample_count and detector_env[end] > low_threshold:
            end += 1
        grown.append((start, end))

    predicted_events = merge_intervals(
        expand_intervals(
            grown,
            round(PRE_PAD_MS * FS / 1000),
            round(POST_PAD_MS * FS / 1000),
            sample_count,
        ),
        max_gap=round(MERGE_GAP_MS * FS / 1000),
    )
    predicted_mask = mask_from_intervals(predicted_events, sample_count)

    return {
        "envelope": env,
        "predicted_events": predicted_events,
        "predicted_mask": predicted_mask,
        "baseline_median": baseline_median,
        "baseline_sigma": baseline_sigma,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "calibration_samples": calibration_samples,
    }


def match_events(
    predicted: list[tuple[int, int]], truth: list[tuple[int, int]]
) -> tuple[list[dict[str, float]], list[int], list[int]]:
    """Greedy one-to-one matching by greatest positive temporal overlap."""
    candidates = []
    for predicted_index, (pred_start, pred_end) in enumerate(predicted):
        for truth_index, (true_start, true_end) in enumerate(truth):
            overlap = max(0, min(pred_end, true_end) - max(pred_start, true_start))
            if overlap:
                union = max(pred_end, true_end) - min(pred_start, true_start)
                candidates.append((overlap / union, overlap, predicted_index, truth_index))
    candidates.sort(reverse=True)
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    matches = []
    for iou, overlap, predicted_index, truth_index in candidates:
        if predicted_index in used_predicted or truth_index in used_truth:
            continue
        used_predicted.add(predicted_index)
        used_truth.add(truth_index)
        pred_start, pred_end = predicted[predicted_index]
        true_start, true_end = truth[truth_index]
        matches.append(
            {
                "predicted_index": predicted_index,
                "truth_index": truth_index,
                "overlap_samples": overlap,
                "iou": iou,
                "onset_error_ms": 1000 * (pred_start - true_start) / FS,
                "offset_error_ms": 1000 * (pred_end - true_end) / FS,
            }
        )
    unmatched_predicted = sorted(set(range(len(predicted))) - used_predicted)
    unmatched_truth = sorted(set(range(len(truth))) - used_truth)
    return matches, unmatched_predicted, unmatched_truth


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


if not INPUT_PATH.exists():
    raise FileNotFoundError(INPUT_PATH)

print("Loading labeled continuous-trial dataset:", INPUT_PATH)
recording = torch.load(INPUT_PATH, map_location="cpu", weights_only=False)
if "all_trials" not in recording:
    raise KeyError(
        "Expected the final Labeled_Dataset-style file containing all_trials; "
        f"keys={list(recording)}"
    )

ordered_trials = sorted(recording["all_trials"], key=lambda trial: int(trial["start_idx"]))
if not ordered_trials:
    raise RuntimeError("all_trials is empty")
for left, right in zip(ordered_trials, ordered_trials[1:]):
    if int(left["end_idx"]) != int(right["start_idx"]):
        raise RuntimeError(
            "Saved trials are not contiguous, so their continuous timing cannot "
            "be reconstructed without the missing source samples."
        )

# Reconstruct the saved continuous EMG stream. No trainKin field is accessed
# before the detector has finalized all predictions.
emg_tensor = torch.cat([trial["ns5_vector"] for trial in ordered_trials], dim=0)
emg = force_emg_time_channels(emg_tensor)
print(f"Continuous EMG: {emg.shape}, duration={len(emg) / FS:.2f} s")

# Crucial ordering: predictions are finalized before labels are read.
detector = detect_bursts_from_emg(emg)
predicted_events = detector["predicted_events"]
predicted_mask = detector["predicted_mask"]
print("Blind detection complete; predicted events:", len(predicted_events))

# Final auto-relabeled labels are accessed only below this line. Original prompt
# labels, relabel seeds, artifact masks, and valid masks are never used.
final_label_tensor = torch.cat([trial["trainKin"] for trial in ordered_trials], dim=0)
labels = force_labels_time_features(final_label_tensor, len(emg))
truth_mask = np.any(labels[:, :KEPT_CLASSES] > 0.5, axis=1)
truth_events = intervals(truth_mask)

true_positive_bins = int(np.sum(predicted_mask & truth_mask))
false_positive_bins = int(np.sum(predicted_mask & ~truth_mask))
false_negative_bins = int(np.sum(~predicted_mask & truth_mask))
true_negative_bins = int(np.sum(~predicted_mask & ~truth_mask))
bin_precision = safe_ratio(true_positive_bins, true_positive_bins + false_positive_bins)
bin_recall = safe_ratio(true_positive_bins, true_positive_bins + false_negative_bins)
bin_f1 = safe_ratio(2 * bin_precision * bin_recall, bin_precision + bin_recall)

matches, unmatched_predicted, unmatched_truth = match_events(predicted_events, truth_events)
event_precision = safe_ratio(len(matches), len(predicted_events))
event_recall = safe_ratio(len(matches), len(truth_events))
event_f1 = safe_ratio(2 * event_precision * event_recall, event_precision + event_recall)

duration_minutes = len(emg) / FS / 60

summary = {
    "input_path": str(INPUT_PATH.resolve()),
    "duration_seconds": len(emg) / FS,
    "calibration_seconds": detector["calibration_samples"] / FS,
    "baseline_median": detector["baseline_median"],
    "baseline_robust_sigma": detector["baseline_sigma"],
    "high_threshold": detector["high_threshold"],
    "low_threshold": detector["low_threshold"],
    "ground_truth_source": "final auto-relabeled trainKin from Labeled_Dataset pipeline",
    "original_prompt_labels_used": False,
    "artifact_masking_used": False,
    "predicted_events": len(predicted_events),
    "truth_events": len(truth_events),
    "matched_events": len(matches),
    "missed_events": len(unmatched_truth),
    "false_events": len(unmatched_predicted),
    "false_events_per_minute": len(unmatched_predicted) / duration_minutes,
    "event_precision": event_precision,
    "event_recall": event_recall,
    "event_f1": event_f1,
    "median_onset_error_ms": float(np.median([m["onset_error_ms"] for m in matches])) if matches else None,
    "median_offset_error_ms": float(np.median([m["offset_error_ms"] for m in matches])) if matches else None,
    "bin_true_positive": true_positive_bins,
    "bin_false_positive": false_positive_bins,
    "bin_false_negative": false_negative_bins,
    "bin_true_negative": true_negative_bins,
    "bin_precision": bin_precision,
    "bin_recall": bin_recall,
    "bin_f1": bin_f1,
    "settings": {
        "sampling_rate_hz": FS,
        "envelope_window_ms": ENVELOPE_WINDOW_MS,
        "high_mad_multiplier": HIGH_MAD_MULT,
        "low_mad_multiplier": LOW_MAD_MULT,
        "minimum_burst_ms": MIN_BURST_MS,
        "pre_pad_ms": PRE_PAD_MS,
        "post_pad_ms": POST_PAD_MS,
        "merge_gap_ms": MERGE_GAP_MS,
    },
}

summary_path = OUTPUT_DIR / "blind_burst_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

event_csv_path = OUTPUT_DIR / "blind_detected_events.csv"
match_by_predicted = {int(match["predicted_index"]): match for match in matches}
with event_csv_path.open("w", newline="", encoding="utf-8") as handle:
    fieldnames = [
        "predicted_event", "start_seconds", "end_seconds", "duration_ms",
        "matched_truth_event", "iou", "onset_error_ms", "offset_error_ms",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for index, (start, end) in enumerate(predicted_events):
        match = match_by_predicted.get(index)
        writer.writerow(
            {
                "predicted_event": index,
                "start_seconds": start / FS,
                "end_seconds": end / FS,
                "duration_ms": 1000 * (end - start) / FS,
                "matched_truth_event": int(match["truth_index"]) if match else "",
                "iou": match["iou"] if match else "",
                "onset_error_ms": match["onset_error_ms"] if match else "",
                "offset_error_ms": match["offset_error_ms"] if match else "",
            }
        )

# Downsample only for plotting; metrics use every original sample.
plot_stride = max(1, len(emg) // 100_000)
plot_indices = np.arange(0, len(emg), plot_stride)
plot_time = plot_indices / FS
figure, axis = plt.subplots(figsize=(15, 5))
axis.plot(plot_time, detector["envelope"][plot_indices], color="black", lw=0.55, label="Causal EMG envelope")
axis.axhline(detector["high_threshold"], color="tab:red", ls="--", lw=1, label="Start threshold")
axis.axhline(detector["low_threshold"], color="tab:orange", ls=":", lw=1, label="Continuation threshold")
axis.fill_between(plot_time, 0, 1, where=truth_mask[plot_indices], transform=axis.get_xaxis_transform(), color="tab:blue", alpha=0.12, label="Final auto-relabeled activity")
axis.fill_between(plot_time, 0, 1, where=predicted_mask[plot_indices], transform=axis.get_xaxis_transform(), color="tab:green", alpha=0.16, label="Blind detection")
axis.set(xlabel="Time (s)", ylabel="Mean absolute EMG envelope", title="Label-blind EMG burst detection across the continuous recording")
axis.legend(frameon=False, ncol=5, loc="upper right")
axis.spines[["top", "right"]].set_visible(False)
figure.tight_layout()
figure_path = OUTPUT_DIR / "blind_burst_overview.png"
figure.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(figure)


def save_gesture_alignment_plots() -> list[str]:
    """Notebook-style blocks aligning EMG, final labels, and predictions."""
    gesture_events: dict[int, list[tuple[int, int, int]]] = {
        gesture: [] for gesture in range(KEPT_CLASSES)
    }
    for truth_index, (start, end) in enumerate(truth_events):
        class_counts = np.sum(labels[start:end, :KEPT_CLASSES] > 0.5, axis=0)
        gesture = int(np.argmax(class_counts))
        gesture_events[gesture].append((truth_index, start, end))

    saved_paths = []
    before_samples = round(0.200 * FS)
    after_samples = round(1.000 * FS)
    trials_per_figure = 10

    for gesture, events_for_gesture in gesture_events.items():
        for block_start in range(0, len(events_for_gesture), trials_per_figure):
            block = events_for_gesture[block_start:block_start + trials_per_figure]
            if not block:
                continue

            envelope_parts = []
            truth_parts = []
            prediction_parts = []
            boundaries = [0]
            trial_centers = []
            trial_names = []

            for truth_index, event_start, event_end in block:
                start = max(0, event_start - before_samples)
                end = min(len(emg), event_end + after_samples)
                envelope_parts.append(detector["envelope"][start:end])
                truth_parts.append(truth_mask[start:end])
                prediction_parts.append(predicted_mask[start:end])
                segment_length = end - start
                trial_centers.append(boundaries[-1] + segment_length / 2)
                trial_names.append(f"G{gesture} event {truth_index + 1}")
                boundaries.append(boundaries[-1] + segment_length)

            block_envelope = np.concatenate(envelope_parts)
            block_truth = np.concatenate(truth_parts)
            block_prediction = np.concatenate(prediction_parts)
            max_points = 60_000
            stride = max(1, len(block_envelope) // max_points)
            indices = np.arange(0, len(block_envelope), stride)
            time_seconds = indices / FS

            figure, axis = plt.subplots(figsize=(16, 5.5))
            axis.plot(
                time_seconds,
                block_envelope[indices],
                color="black",
                lw=0.65,
                label="Causal mean-absolute raw EMG envelope",
                zorder=3,
            )
            axis.axhline(
                detector["high_threshold"], color="tab:red", ls="--", lw=1,
                label="Blind start threshold", zorder=2,
            )
            axis.axhline(
                detector["low_threshold"], color="tab:orange", ls=":", lw=1,
                label="Blind continuation threshold", zorder=2,
            )
            axis.fill_between(
                time_seconds, 0, 1, where=block_truth[indices],
                transform=axis.get_xaxis_transform(), color="tab:blue", alpha=0.16,
                label="Final auto-relabeled activity", zorder=0,
            )
            axis.fill_between(
                time_seconds, 0, 1, where=block_prediction[indices],
                transform=axis.get_xaxis_transform(), color="tab:green", alpha=0.20,
                label="Blind predicted activity", zorder=1,
            )
            for boundary in boundaries[1:-1]:
                axis.axvline(boundary / FS, color="0.65", lw=0.7, ls="--")
            axis.set_xticks(
                np.asarray(trial_centers) / FS,
                trial_names,
                rotation=25,
                ha="right",
            )
            axis.set(
                ylabel="Mean absolute EMG envelope",
                title=(
                    f"Gesture G{gesture}: blind predictions vs final auto-relabeled activity "
                    f"(events {block_start + 1}-{block_start + len(block)})"
                ),
            )
            axis.legend(frameon=False, ncol=4, loc="upper right")
            axis.spines[["top", "right"]].set_visible(False)
            figure.tight_layout()
            stem = f"gesture_g{gesture}_events_{block_start + 1:02d}_{block_start + len(block):02d}_alignment"
            for extension, kwargs in (("png", {"dpi": 200}), ("svg", {})):
                path = OUTPUT_DIR / f"{stem}.{extension}"
                figure.savefig(path, bbox_inches="tight", **kwargs)
                saved_paths.append(str(path))
            plt.close(figure)
    return saved_paths


alignment_figure_paths = save_gesture_alignment_plots()
summary["alignment_figures"] = alignment_figure_paths
# Rewrite after adding the detailed figure paths.
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("\nSaved:")
print(" ", summary_path)
print(" ", event_csv_path)
print(" ", figure_path)
print(f"  {len(alignment_figure_paths)} detailed gesture-alignment figures")
print("\nReference: final auto-relabeled activity; artifact masking disabled.")
