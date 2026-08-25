"""Full label-blind continuous replay: burst detection -> frozen gesture model.

Inference uses only EMG and detector-produced timing. Final auto-relabeled
``trainKin`` is accessed only after every detection and class prediction has
been finalized, solely for scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from torch import nn

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm


REPO_ROOT = Path(r"C:\Users\Micah\utah-neuro\generic_neuromotor_interface")
DEFAULT_DATASET_PATH = REPO_ROOT / "New_Gesture_Trial_Dataset_Labeled.pt"
DEFAULT_FROZEN_CHECKPOINT_PATH = Path(
    r"C:\Users\Micah\utah-neuro\MATLAB_Jupyter\03_online_domain_adaptation"
    r"\frozen_replay\online_eval_results\best_frozen_meta_adapter.pt"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "full_label_blind_results"

# Optional command-line arguments make the same verified evaluator reusable
# from Jupyter via runpy or %run:
#   1. labeled trial dataset
#   2. frozen checkpoint
#   3. output directory
DATASET_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
FROZEN_CHECKPOINT_PATH = (
    Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FROZEN_CHECKPOINT_PATH
)
OUTPUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture


FS = 30_000
META_FS = 2_000
RAW_SAMPLES = 30_000
META_SAMPLES = 2_000
CHANNELS = 32
META_CHANNELS = 16
NUM_META_CLASSES = 9
CLASS_NAMES = ["thumb left", "thumb right", "thumb up", "thumb down", "thumb press"]
SHORT_NAMES = ["left", "right", "up", "down", "press"]
UTAH_TO_META_OUTPUTS = [5, 6, 7, 8, 4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Frozen blind detector settings selected in the preceding benchmark.
ENVELOPE_WINDOW_MS = 25
CALIBRATION_SECONDS = 10.0
HIGH_MAD_MULT = 1.5
LOW_MAD_MULT = 0.75
MIN_BURST_MS = 12
PRE_PAD_MS = 40
POST_PAD_MS = 120
MERGE_GAP_MS = 300

# Adapter architecture from 02_current_meta_tl.ipynb.
HIDDEN_CHANNELS = 48
GROUPS = 8
KERNEL_SIZE = 5


def intervals(mask):
    mask = np.asarray(mask, dtype=bool)
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def causal_moving_average(x, window):
    x = np.asarray(x, dtype=np.float64)
    cumulative = np.cumsum(x, dtype=np.float64)
    result = cumulative.copy()
    if window < len(x):
        result[window:] -= cumulative[:-window]
    result /= np.minimum(np.arange(1, len(x) + 1), window)
    return result.astype(np.float32)


def robust_med_sigma(x):
    x = np.asarray(x, dtype=np.float32)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma = 1.4826 * mad
    if sigma < 1e-8:
        sigma = float(np.std(x) + 1e-8)
    return median, sigma


def merge_intervals(source, max_gap=0):
    if not source:
        return []
    merged = [tuple(sorted(source)[0])]
    for start, end in sorted(source)[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + max_gap:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def mask_from_intervals(source, length):
    mask = np.zeros(length, dtype=bool)
    for start, end in source:
        mask[start:end] = True
    return mask


def blind_detect(emg_tc):
    envelope = causal_moving_average(
        np.mean(np.abs(emg_tc), axis=1, dtype=np.float64),
        round(ENVELOPE_WINDOW_MS * FS / 1000),
    )
    calibration_samples = min(len(envelope), round(CALIBRATION_SECONDS * FS))
    median, sigma = robust_med_sigma(envelope[:calibration_samples])
    high_threshold = median + HIGH_MAD_MULT * sigma
    low_threshold = median + LOW_MAD_MULT * sigma
    minimum_samples = round(MIN_BURST_MS * FS / 1000)
    high_runs = [
        (start, end)
        for start, end in intervals(envelope > high_threshold)
        if end - start >= minimum_samples
    ]
    grown = []
    for start, end in high_runs:
        while start > 0 and envelope[start - 1] > low_threshold:
            start -= 1
        while end < len(envelope) and envelope[end] > low_threshold:
            end += 1
        grown.append((start, end))
    padded = [
        (
            max(0, start - round(PRE_PAD_MS * FS / 1000)),
            min(len(envelope), end + round(POST_PAD_MS * FS / 1000)),
        )
        for start, end in grown
    ]
    events = merge_intervals(padded, round(MERGE_GAP_MS * FS / 1000))
    return {
        "events": events,
        "mask": mask_from_intervals(events, len(envelope)),
        "envelope": envelope,
        "baseline_median": median,
        "baseline_sigma": sigma,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
    }


def make_group_norm(channels, requested_groups=8):
    groups = min(requested_groups, channels)
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def make_wn_conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=1, groups=1, bias=True):
    return weight_norm(
        nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )
    )


class DepthwiseSeparableTemporalConv(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = dilation * (KERNEL_SIZE - 1) // 2
        self.depthwise = make_wn_conv1d(
            channels, channels, KERNEL_SIZE, padding=padding,
            dilation=dilation, groups=channels, bias=False,
        )
        self.pointwise = make_wn_conv1d(channels, channels, 1, bias=False)
        self.norm = make_group_norm(channels, GROUPS)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.norm(self.pointwise(self.depthwise(x))))


class RegularizedTCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.temporal_1 = DepthwiseSeparableTemporalConv(channels, dilation)
        self.temporal_2 = DepthwiseSeparableTemporalConv(channels, dilation)
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        return x + self.residual_scale * self.temporal_2(self.temporal_1(x))


class AdapterToMetaInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Sequential(
            make_wn_conv1d(CHANNELS, HIDDEN_CHANNELS, 1, bias=False),
            make_group_norm(HIDDEN_CHANNELS, GROUPS),
            nn.SiLU(),
        )
        self.temporal_stack = nn.Sequential(
            *[RegularizedTCNBlock(HIDDEN_CHANNELS, dilation) for dilation in (1, 2, 4, 8)]
        )
        self.output_projection = make_wn_conv1d(HIDDEN_CHANNELS, META_CHANNELS, 1, bias=True)
        self.output_gain = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        return self.output_gain * self.output_projection(self.temporal_stack(self.input_projection(x)))


class FrozenMetaWithAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = AdapterToMetaInput()
        self.meta_model = DiscreteGesturesArchitecture(output_channels=NUM_META_CLASSES)

    def forward(self, x):
        return self.meta_model(self.adapter(x))


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


HIGHPASS_SOS = butter(4, 40.0, btype="highpass", fs=META_FS, output="sos")


def one_second_window(emg_tc, center):
    desired_start = int(center) - RAW_SAMPLES // 2
    desired_end = desired_start + RAW_SAMPLES
    source_start = max(0, desired_start)
    source_end = min(len(emg_tc), desired_end)
    window = np.zeros((RAW_SAMPLES, CHANNELS), dtype=np.float32)
    destination_start = source_start - desired_start
    window[destination_start:destination_start + source_end - source_start] = emg_tc[source_start:source_end]
    return window, desired_start, desired_end


def preprocess_window(window_tc):
    x = window_tc.T.astype(np.float32, copy=False)
    x = resample_poly(x, up=1, down=15, axis=1).astype(np.float32, copy=False)
    x = sosfiltfilt(HIGHPASS_SOS, x, axis=1).astype(np.float32, copy=False)
    if x.shape != (CHANNELS, META_SAMPLES):
        raise RuntimeError(f"Unexpected preprocessed shape {x.shape}")
    return x


def match_events(predicted, truth):
    candidates = []
    for pi, (ps, pe) in enumerate(predicted):
        for ti, item in enumerate(truth):
            ts, te = item["start"], item["end"]
            overlap = max(0, min(pe, te) - max(ps, ts))
            if overlap:
                union = max(pe, te) - min(ps, ts)
                candidates.append((overlap / union, overlap, pi, ti))
    candidates.sort(reverse=True)
    used_p, used_t, matches = set(), set(), []
    for iou, overlap, pi, ti in candidates:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matches.append({"predicted_index": pi, "truth_index": ti, "iou": iou, "overlap": overlap})
    return matches, sorted(set(range(len(predicted))) - used_p), sorted(set(range(len(truth))) - used_t)


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


if not DATASET_PATH.exists() or not FROZEN_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Missing dataset or checkpoint: {DATASET_PATH}, {FROZEN_CHECKPOINT_PATH}")

print("Loading prior-session continuous trials:", DATASET_PATH)
saved = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
trials = sorted(saved["all_trials"], key=lambda trial: int(trial["start_idx"]))
for left, right in zip(trials, trials[1:]):
    if int(left["end_idx"]) != int(right["start_idx"]):
        raise RuntimeError("all_trials cannot be reconstructed as a continuous stream")
emg_tensor = torch.cat([trial["ns5_vector"] for trial in trials], dim=0)
emg = emg_tensor.numpy().astype(np.float32, copy=False)

print("Running label-blind burst detection...")
detection = blind_detect(emg)
predicted_events = detection["events"]
print("Blind events:", len(predicted_events))

print("Loading and freezing selected model...")
checkpoint = torch.load(FROZEN_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
model = FrozenMetaWithAdapter()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(DEVICE).eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
digest_before = state_digest(model)
if digest_before != checkpoint["model_sha256"]:
    raise RuntimeError("Loaded model digest does not match saved checkpoint")

# Predictions are generated before any final label tensor is read.
prediction_rows = []
meta_left_context = int(model.meta_model.left_context)
meta_stride = int(model.meta_model.stride)
output_centers_2k = meta_left_context + np.arange(198) * meta_stride

for event_index, (event_start, event_end) in enumerate(predicted_events):
    event_center = (event_start + event_end - 1) // 2
    window, window_start, window_end = one_second_window(emg, event_center)
    model_input = torch.from_numpy(preprocess_window(window)).unsqueeze(0).to(DEVICE)
    inference_start = time.perf_counter()
    with torch.inference_mode():
        raw_output = model(model_input)
        mapped_logits = raw_output[:, UTAH_TO_META_OUTPUTS, :]
    latency_ms = 1000 * (time.perf_counter() - inference_start)
    if mapped_logits.shape != (1, 5, 198):
        raise RuntimeError(f"Unexpected mapped output shape {tuple(mapped_logits.shape)}")

    output_centers_global = window_start + output_centers_2k * 15
    predicted_active_bins = (output_centers_global >= event_start) & (output_centers_global < event_end)
    if not predicted_active_bins.any():
        nearest = int(np.argmin(np.abs(output_centers_global - event_center)))
        predicted_active_bins[nearest] = True
    scores = mapped_logits[0, :, torch.from_numpy(predicted_active_bins).to(DEVICE)].mean(dim=1)
    probabilities = torch.softmax(scores, dim=0).detach().cpu().numpy()
    predicted_class = int(np.argmax(probabilities))
    decision_sample = max(event_end, window_end)
    prediction_rows.append(
        {
            "event_index": event_index,
            "event_start": event_start,
            "event_end": event_end,
            "event_duration_ms": 1000 * (event_end - event_start) / FS,
            "window_start": window_start,
            "window_end": window_end,
            "event_truncated_by_window": int(event_start < window_start or event_end > window_end),
            "predicted_active_bins": int(predicted_active_bins.sum()),
            "predicted_class": predicted_class,
            "predicted_name": CLASS_NAMES[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "inference_latency_ms": latency_ms,
            "decision_sample": decision_sample,
            **{f"score_{index}_{SHORT_NAMES[index]}": float(value) for index, value in enumerate(probabilities)},
        }
    )

digest_after = state_digest(model)
if digest_after != digest_before:
    raise RuntimeError("Frozen model changed during replay")
print("All predictions finalized; frozen-state verification passed.")

# Labels enter only here, after detection and prediction are immutable.
labels = torch.cat([trial["trainKin"] for trial in trials], dim=0).numpy()
truth_mask = np.any(labels[:, :5] > 0.5, axis=1)
truth_intervals = intervals(truth_mask)
split_lookup = {}
for split_name in ("train", "val", "test"):
    for trial in saved[split_name]:
        split_lookup[(int(trial["gesture"]), int(trial["trial_num"]))] = split_name

truth_events = []
stream_origin = int(trials[0]["start_idx"])
for truth_index, (start, end) in enumerate(truth_intervals):
    counts = np.sum(labels[start:end, :5] > 0.5, axis=0)
    true_class = int(np.argmax(counts))
    global_midpoint = stream_origin + (start + end) // 2
    owner = next(
        trial for trial in trials
        if int(trial["start_idx"]) <= global_midpoint < int(trial["end_idx"])
    )
    key = (int(owner["gesture"]), int(owner["trial_num"]))
    truth_events.append(
        {
            "truth_index": truth_index,
            "start": start,
            "end": end,
            "true_class": true_class,
            "true_name": CLASS_NAMES[true_class],
            "gesture": key[0],
            "trial_num": key[1],
            "split": split_lookup.get(key, "unknown"),
        }
    )

matches, unmatched_predictions, unmatched_truth = match_events(predicted_events, truth_events)
match_by_prediction = {item["predicted_index"]: item for item in matches}
confusion = np.zeros((5, 5), dtype=np.int64)
correct_classifications = 0
for row in prediction_rows:
    match = match_by_prediction.get(row["event_index"])
    if match is None:
        row.update({
            "matched": 0, "truth_index": "", "split": "false_detection",
            "trial_num": "", "true_class": "", "true_name": "", "class_correct": 0,
            "event_iou": 0.0, "onset_error_ms": "", "offset_error_ms": "",
            "decision_latency_from_true_end_ms": "",
        })
        continue
    truth = truth_events[match["truth_index"]]
    true_class = truth["true_class"]
    correct = row["predicted_class"] == true_class
    correct_classifications += int(correct)
    confusion[true_class, row["predicted_class"]] += 1
    row.update({
        "matched": 1, "truth_index": truth["truth_index"], "split": truth["split"],
        "trial_num": truth["trial_num"], "true_class": true_class,
        "true_name": truth["true_name"], "class_correct": int(correct),
        "event_iou": match["iou"],
        "onset_error_ms": 1000 * (row["event_start"] - truth["start"]) / FS,
        "offset_error_ms": 1000 * (row["event_end"] - truth["end"]) / FS,
        "decision_latency_from_true_end_ms": 1000 * (row["decision_sample"] - truth["end"]) / FS,
    })

split_metrics = {}
for split_name in ("train", "val", "test"):
    split_truth = [event for event in truth_events if event["split"] == split_name]
    split_matched_rows = [row for row in prediction_rows if row["matched"] and row["split"] == split_name]
    split_correct = sum(row["class_correct"] for row in split_matched_rows)
    split_metrics[split_name] = {
        "truth_events": len(split_truth),
        "detected_events": len(split_matched_rows),
        "detection_recall": safe_ratio(len(split_matched_rows), len(split_truth)),
        "conditional_gesture_accuracy": safe_ratio(split_correct, len(split_matched_rows)),
        "end_to_end_correct_recall": safe_ratio(split_correct, len(split_truth)),
    }

summary = {
    "protocol": "EMG-only detection, window centering, active-bin selection, and frozen classification; labels used afterward for scoring only",
    "dataset": str(DATASET_PATH),
    "checkpoint": str(FROZEN_CHECKPOINT_PATH),
    "checkpoint_epoch": int(checkpoint["best_epoch"]),
    "model_sha256": digest_after,
    "duration_seconds": len(emg) / FS,
    "predicted_events": len(predicted_events),
    "truth_events": len(truth_events),
    "matched_events": len(matches),
    "missed_events": len(unmatched_truth),
    "false_detections": len(unmatched_predictions),
    "detection_precision": safe_ratio(len(matches), len(predicted_events)),
    "detection_recall": safe_ratio(len(matches), len(truth_events)),
    "conditional_gesture_accuracy": safe_ratio(correct_classifications, len(matches)),
    "end_to_end_correct_recall": safe_ratio(correct_classifications, len(truth_events)),
    "end_to_end_correct_precision": safe_ratio(correct_classifications, len(predicted_events)),
    "median_inference_latency_ms": float(np.median([row["inference_latency_ms"] for row in prediction_rows])),
    "median_decision_latency_from_true_end_ms": float(np.median([
        row["decision_latency_from_true_end_ms"] for row in prediction_rows if row["matched"]
    ])),
    "events_truncated_by_one_second_window": sum(row["event_truncated_by_window"] for row in prediction_rows),
    "confusion_true_rows_predicted_columns": confusion.tolist(),
    "split_metrics": split_metrics,
    "detector": {
        "high_threshold": detection["high_threshold"],
        "low_threshold": detection["low_threshold"],
        "merge_gap_ms": MERGE_GAP_MS,
        "artifact_masking": False,
    },
    "important_limitations": [
        "Detector parameters were selected after inspecting this prior session, so this is a pipeline benchmark rather than a final untouched detector test.",
        "One-second buffered zero-phase high-pass preprocessing is label-blind but not sample-by-sample causal.",
        "The gesture model has no rest class, so every false detection receives a gesture label.",
    ],
}

csv_path = OUTPUT_DIR / "full_online_event_predictions.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
    writer.writeheader()
    writer.writerows(prediction_rows)
summary_path = OUTPUT_DIR / "full_online_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

figure, axis = plt.subplots(figsize=(8, 7))
image = axis.imshow(confusion, cmap="Blues")
figure.colorbar(image, ax=axis, label="Matched detected events")
axis.set(
    title=f"Label-blind matched-event gesture confusion\nAccuracy = {summary['conditional_gesture_accuracy']:.1%}",
    xlabel="Predicted gesture", ylabel="True gesture",
    xticks=np.arange(5), yticks=np.arange(5), xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
)
threshold = confusion.max() / 2 if confusion.size else 0
for true_index in range(5):
    for predicted_index in range(5):
        axis.text(predicted_index, true_index, str(confusion[true_index, predicted_index]),
                  ha="center", va="center", color="white" if confusion[true_index, predicted_index] > threshold else "black")
figure.tight_layout()
confusion_path = OUTPUT_DIR / "full_online_gesture_confusion.png"
figure.savefig(confusion_path, dpi=250, bbox_inches="tight")
figure.savefig(OUTPUT_DIR / "full_online_gesture_confusion.svg", bbox_inches="tight")
plt.close(figure)

plot_stride = max(1, len(emg) // 120_000)
indices = np.arange(0, len(emg), plot_stride)
figure, axis = plt.subplots(figsize=(18, 6))
axis.plot(indices / FS, detection["envelope"][indices], color="black", lw=0.45, label="Causal EMG envelope")
axis.axhline(detection["high_threshold"], color="tab:red", ls="--", lw=0.9, label="Detector start threshold")
colors = plt.get_cmap("tab10").colors
for truth in truth_events:
    axis.axvspan(truth["start"] / FS, truth["end"] / FS, color=colors[truth["true_class"]], alpha=0.10)
for row in prediction_rows:
    color = colors[row["predicted_class"]] if row["matched"] else "red"
    axis.axvspan(row["event_start"] / FS, row["event_end"] / FS, ymin=0.91, ymax=0.99, color=color, alpha=0.75)
axis.set(
    xlabel="Continuous replay time (s)", ylabel="Mean absolute EMG envelope",
    title="Full label-blind replay: truth activity shading and predicted event/class strips",
)
axis.legend(frameon=False, loc="upper right")
axis.spines[["top", "right"]].set_visible(False)
figure.tight_layout()
timeline_path = OUTPUT_DIR / "full_online_timeline.png"
figure.savefig(timeline_path, dpi=220, bbox_inches="tight")
figure.savefig(OUTPUT_DIR / "full_online_timeline.svg", bbox_inches="tight")
plt.close(figure)

print(json.dumps(summary, indent=2))
print("Saved:", OUTPUT_DIR)
