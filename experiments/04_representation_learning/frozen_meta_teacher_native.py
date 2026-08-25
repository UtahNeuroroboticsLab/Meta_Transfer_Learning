"""Native 512-D targets from Meta's released discrete-gesture model.

No Meta parameter is optimized.  Each event is pooled only across Meta's
official 80-120 ms post-prompt target interval, with the released model's
gesture confidence providing the within-interval weights.  The resulting
five class prototypes and geometry are cached for reuse.
"""

from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


CACHE_VERSION = 2
SOURCE_DIM = 512
META_SAMPLE_RATE = 2_000
META_WINDOW_SAMPLES = 2_000
PULSE_START_SECONDS = 0.08
PULSE_END_SECONDS = 0.12


def _load_official_network(repo_root, device):
    repo_root = Path(repo_root)
    checkpoint_path = repo_root / "emg_models" / "discrete_gestures" / "model_checkpoint.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Official Meta checkpoint not found: {checkpoint_path}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from generic_neuromotor_interface.networks import DiscreteGesturesArchitecture

    network = DiscreteGesturesArchitecture(output_channels=9)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict")
    if state is None:
        raise KeyError("Official Meta checkpoint is missing state_dict")
    network_state = {
        key.replace("network.", "", 1): value
        for key, value in state.items()
        if key.startswith("network.")
    }
    network.load_state_dict(network_state or state, strict=True)
    network.to(device).eval()
    for parameter in network.parameters():
        parameter.requires_grad = False
    return network, checkpoint_path


@torch.inference_mode()
def _hidden_and_logits(network, x):
    x = network.compression(x)
    x = network.conv_layer(x)
    x = network.relu(x)
    x = network.dropout(x)
    x = network.post_conv_layer_norm(x.transpose(1, 2))
    x, _ = network.lstm(x)
    hidden = network.post_lstm_layer_norm(x)
    logits = network.projection(hidden).transpose(1, 2).contiguous()
    return hidden, logits


@torch.inference_mode()
def _confidence_weighted_active_pool(network, x, mapped_meta_output_ids):
    hidden, logits = _hidden_and_logits(network, x)
    output_samples = torch.arange(
        network.left_context,
        META_WINDOW_SAMPLES,
        network.stride,
        device=x.device,
    )
    relative_seconds = output_samples.float() / META_SAMPLE_RATE - 0.5
    active = (
        (relative_seconds >= PULSE_START_SECONDS)
        & (relative_seconds <= PULSE_END_SECONDS)
    )
    if not bool(active.any()) or hidden.shape[1] != len(output_samples):
        raise RuntimeError("Could not align Meta's official active output bins")

    batch_indices = torch.arange(len(x), device=x.device)
    class_logits = logits[batch_indices, mapped_meta_output_ids]
    active_scores = class_logits[:, active]
    weights = torch.softmax(active_scores, dim=1)
    pooled = (hidden[:, active] * weights.unsqueeze(-1)).sum(dim=1)
    return F.normalize(pooled, dim=1), int(active.sum())


def load_or_build_native_teacher_targets(
    *, repo_root, experiment_root, meta_dataset, meta_batch_size,
    class_names, meta_output_ids, seed, device, num_workers=0,
):
    repo_root = Path(repo_root)
    experiment_root = Path(experiment_root)
    checkpoint_path = repo_root / "emg_models" / "discrete_gestures" / "model_checkpoint.ckpt"
    cache_dir = experiment_root / "results" / "_generated_cache"
    cache_path = cache_dir / "native_512d_confidence_active_targets.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Official Meta checkpoint not found: {checkpoint_path}")

    stat = checkpoint_path.stat()
    signature = {
        "cache_version": CACHE_VERSION,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "meta_event_count": int(len(meta_dataset)),
        "class_names": list(class_names),
        "meta_output_ids": [int(x) for x in meta_output_ids],
        "seed": int(seed),
        "preprocessing": "official raw downloaded Meta EMG; no per-window scaling",
        "pooling": "official 80-120 ms active interval, softmax confidence weighted",
    }
    start = time.perf_counter()
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("signature") == signature:
            info = {
                "cache_hit": True,
                "cache_path": str(cache_path),
                "checkpoint_path": str(checkpoint_path),
                "seconds": time.perf_counter() - start,
                "active_output_bins": int(cached["active_output_bins"]),
            }
            print("Loaded cached native official Meta teacher targets:", cache_path)
            return (
                cached["teacher_prototypes"].to(device),
                cached["teacher_geometry"].to(device),
                cached["prototype_counts"].to(device),
                info,
            )

    print("Building native 512-D targets from Meta's released frozen checkpoint")
    network, checkpoint_path = _load_official_network(repo_root, device)
    loader = DataLoader(
        meta_dataset,
        batch_size=int(meta_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(int(seed)),
    )
    sums = torch.zeros(5, SOURCE_DIM, device=device)
    counts = torch.zeros(5, device=device)
    active_output_bins = None
    for x, y in loader:
        x, y = x.to(device), y.long().to(device)
        output_ids = torch.as_tensor(meta_output_ids, device=device, dtype=torch.long)[y]
        embeddings, active_output_bins = _confidence_weighted_active_pool(
            network, x, output_ids
        )
        for class_index in range(5):
            keep = y == class_index
            if bool(keep.any()):
                sums[class_index] += embeddings[keep].sum(dim=0)
                counts[class_index] += keep.sum()
    if not bool((counts > 0).all()):
        raise RuntimeError(f"Missing mapped Meta classes: {counts.tolist()}")
    prototypes = F.normalize(sums / counts[:, None], dim=1)
    geometry = torch.cdist(prototypes, prototypes)

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "teacher_prototypes": prototypes.cpu(),
            "teacher_geometry": geometry.cpu(),
            "prototype_counts": counts.cpu(),
            "active_output_bins": int(active_output_bins),
            "teacher_frozen": True,
            "embedding_dim": SOURCE_DIM,
        },
        cache_path,
    )
    del network
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    info = {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "checkpoint_path": str(checkpoint_path),
        "seconds": time.perf_counter() - start,
        "active_output_bins": int(active_output_bins),
    }
    print("Saved reusable native official Meta teacher targets:", cache_path)
    print("Prototype counts:", counts.long().cpu().tolist())
    print("Confidence-pooled active output bins:", active_output_bins)
    return prototypes, geometry, counts, info
