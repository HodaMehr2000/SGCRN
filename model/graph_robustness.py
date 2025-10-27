"""Utility for comparing learned traffic graphs during peak and off-peak periods.

This script follows a lightweight procedure requested by reviewers to provide
empirical evidence for the stability of learned traffic graphs:

1) Select two time windows (e.g., "7-9" and "1-3") within the training split.
2) Learn an adjacency matrix for each window with DAGMA (or use model-exported ones).
3) Threshold the learned weights and report simple statistics:
   - Graph density, average out-degree / in-degree, symmetry
   - Edge overlap (Jaccard) between peak vs off-peak, and vs a global graph
4) Optionally, save a heatmap of the difference A_peak - A_offpeak for visual inspection.

Example:
    python analysis/graph_robustness.py \
        --dataset PEMSD8 --peak-hours "7-9,16-19" --off-peak-hours "1-3" \
        --edge-threshold 0.01 --lambda1 0.02 --steps-per-day 288 \
        --output-json outputs/pemsd8_graph_stability.json --heatmap-out outputs/diff.png

Notes:
- By default, overlap is computed on binarized graphs (thresholded weights).
- If --weighted-jaccard is provided, a weighted Jaccard is reported instead.
- If you already exported adjacency matrices from your own model, you can skip the
  built-in DAGMA learning and point the script to your files in a future extension;
  for now it learns with DAGMA internally (as reviewers asked).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from dagma.linear import DagmaLinear

from lib.dataloader import normalize_dataset, split_data_by_ratio
from lib.load_dataset import load_st_dataset


HourRange = Tuple[float, float]


def parse_hour_ranges(ranges: str) -> List[HourRange]:
    """Parse a comma separated list of hour ranges (e.g., "7-9,16-19")."""
    if not ranges:
        raise ValueError("At least one hour range is required.")
    parsed: List[HourRange] = []
    for part in ranges.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid hour range '{part}'. Expected format 'start-end'.")
        start_str, end_str = part.split("-", maxsplit=1)
        start_hour, end_hour = float(start_str), float(end_str)
        parsed.append((start_hour, end_hour))
    if not parsed:
        raise ValueError("No valid hour ranges found.")
    return parsed


def _hour_to_step(hour: float, steps_per_day: int) -> int:
    return int(round(hour * steps_per_day / 24.0)) % steps_per_day


def build_time_mask(length: int, steps_per_day: int, ranges: Sequence[HourRange]) -> np.ndarray:
    """Return a boolean mask selecting samples whose timestamp falls in `ranges`.

    Time is grouped modulo `steps_per_day`, so if training starts mid-day the mapping
    still works by index modulo (e.g., 288 for 5-minute PeMS data).
    """
    if length <= 0:
        raise ValueError("Segment length must be positive.")
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be positive.")
    if not ranges:
        raise ValueError("At least one hour range must be provided.")

    mask = np.zeros(length, dtype=bool)
    time_indices = np.arange(length) % steps_per_day

    for start_hour, end_hour in ranges:
        start_step = _hour_to_step(start_hour, steps_per_day)
        end_step = _hour_to_step(end_hour, steps_per_day)

        if start_step == end_step:
            continue  # Ignore zero-width ranges.

        if start_step < end_step:
            mask |= (time_indices >= start_step) & (time_indices < end_step)
        else:  # Wraps around midnight.
            mask |= (time_indices >= start_step) | (time_indices < end_step)

    return mask


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise ValueError(f"Expected 2D data after squeezing, got shape {squeezed.shape}.")
    return squeezed


def learn_graph_weights(segment: np.ndarray, lambda1: float, symmetrize: bool = False) -> np.ndarray:
    """Fit DAGMA on `segment` and return the absolute weight matrix (diagonal zeroed)."""
    if segment.shape[0] < 2:
        raise ValueError("DAGMA requires at least two samples to fit a graph.")

    data_2d = _ensure_2d(segment)
    model = DagmaLinear(loss_type="l2")
    weights = model.fit(data_2d, lambda1=lambda1)
    weights = np.asarray(weights, dtype=float)
    np.fill_diagonal(weights, 0.0)
    if symmetrize:
        weights = 0.5 * (weights + weights.T)
        np.fill_diagonal(weights, 0.0)
    return np.abs(weights)


def _off_diagonal_mask(size: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


@dataclass
class GraphMetrics:
    weights: np.ndarray
    adjacency: np.ndarray
    edge_count: int
    density: float
    avg_out_degree: float
    avg_in_degree: float
    symmetry: float  # 1 - mean(|A - A^T|)


def compute_graph_metrics(weights: np.ndarray, threshold: float) -> GraphMetrics:
    """Compute classic graph metrics on the thresholded (binary) adjacency.

    Note: `threshold` is applied to absolute weights.
    """
    adjacency = (weights >= threshold).astype(int)
    mask = _off_diagonal_mask(adjacency.shape[0])
    edge_count = int(adjacency[mask].sum())
    possible_edges = mask.sum()
    density = float(edge_count / possible_edges) if possible_edges else 0.0
    avg_out_degree = float(adjacency.sum(axis=1).mean())
    avg_in_degree = float(adjacency.sum(axis=0).mean())
    symmetry = 1.0 - float(np.mean(np.abs(adjacency - adjacency.T)))
    return GraphMetrics(weights, adjacency, edge_count, density, avg_out_degree, avg_in_degree, symmetry)


def jaccard_binary(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    mask = _off_diagonal_mask(a.shape[0])
    a_edges = a[mask]
    b_edges = b[mask]
    intersection = np.logical_and(a_edges, b_edges).sum()
    union = np.logical_or(a_edges, b_edges).sum()
    overlap = float(intersection / union) if union else 0.0
    return {
        "jaccard": overlap,
        "common_edges": int(intersection),
        "total_union": int(union),
    }


def jaccard_weighted(A: np.ndarray, B: np.ndarray) -> float:
    """Weighted Jaccard: sum(min(A,B))/sum(max(A,B)) on off-diagonal entries."""
    mask = _off_diagonal_mask(A.shape[0])
    a = A[mask]
    b = B[mask]
    denom = np.maximum(a, b).sum()
    return float(np.minimum(a, b).sum() / denom) if denom else 0.0


def save_difference_heatmap(peak: np.ndarray, off_peak: np.ndarray, output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required to save a heatmap.") from exc

    diff = peak - off_peak
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(diff, cmap="coolwarm", interpolation="nearest")
    ax.set_title("A_peak - A_off-peak")
    ax.set_xlabel("Destination node")
    ax.set_ylabel("Source node")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _downsample_every_k(arr: np.ndarray, k: int) -> np.ndarray:
    return arr[::k] if k and k > 1 else arr


def run_analysis(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    # Optional seeding for reproducibility signal
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)

    data = load_st_dataset(args.dataset)
    if not np.isfinite(data).all():
        raise ValueError("Input data contains NaN/Inf; please clean or normalize properly.")

    data_normalized, _ = normalize_dataset(data, args.normalizer, column_wise=args.column_wise)
    train_data, _, _ = split_data_by_ratio(data_normalized, args.val_ratio, args.test_ratio)

    # Optional temporal downsampling to speed up DAGMA on large segments
    train_data = _downsample_every_k(train_data, args.every_k)

    total_steps = train_data.shape[0]
    peak_ranges = parse_hour_ranges(args.peak_hours)
    off_peak_ranges = parse_hour_ranges(args.off_peak_hours)

    peak_mask = build_time_mask(total_steps, args.steps_per_day, peak_ranges)
    off_peak_mask = build_time_mask(total_steps, args.steps_per_day, off_peak_ranges)

    if peak_mask.sum() < args.prediction_length:
        raise ValueError(
            f"Peak window contains only {int(peak_mask.sum())} samples. "
            f"Increase its duration or decrease --prediction-length."
        )
    if off_peak_mask.sum() < args.prediction_length:
        raise ValueError(
            f"Off-peak window contains only {int(off_peak_mask.sum())} samples. "
            f"Increase its duration or decrease --prediction-length."
        )

    peak_segment = train_data[peak_mask]
    off_peak_segment = train_data[off_peak_mask]

    # Learn weights via DAGMA (reviewer-requested).
    peak_weights = learn_graph_weights(peak_segment, args.lambda1, symmetrize=args.symmetrize)
    off_peak_weights = learn_graph_weights(off_peak_segment, args.lambda1, symmetrize=args.symmetrize)

    # Also learn a global graph on the full training split to compare stability vs. global.
    global_weights = learn_graph_weights(train_data, args.lambda1, symmetrize=args.symmetrize)

    peak_metrics = compute_graph_metrics(peak_weights, args.edge_threshold)
    off_peak_metrics = compute_graph_metrics(off_peak_weights, args.edge_threshold)
    global_metrics = compute_graph_metrics(global_weights, args.edge_threshold)

    # Overlaps (binary or weighted-Jaccard)
    if args.weighted_jaccard:
        overlap_peak_off = {"weighted_jaccard": jaccard_weighted(peak_metrics.weights, off_peak_metrics.weights)}
        overlap_peak_global = {"weighted_jaccard": jaccard_weighted(peak_metrics.weights, global_metrics.weights)}
        overlap_off_global = {"weighted_jaccard": jaccard_weighted(off_peak_metrics.weights, global_metrics.weights)}
    else:
        overlap_peak_off = jaccard_binary(peak_metrics.adjacency, off_peak_metrics.adjacency)
        overlap_peak_global = jaccard_binary(peak_metrics.adjacency, global_metrics.adjacency)
        overlap_off_global = jaccard_binary(off_peak_metrics.adjacency, global_metrics.adjacency)

    if args.heatmap_out:
        save_difference_heatmap(peak_metrics.weights, off_peak_metrics.weights, args.heatmap_out)

    # Compose a caption-friendly snippet
    if args.weighted_jaccard:
        ov = overlap_peak_off["weighted_jaccard"]
        ov_label = "Weighted-Jaccard"
    else:
        ov = overlap_peak_off["jaccard"]
        ov_label = "Jaccard"

    paper_snippet = (
        f"Peak vs. Off-peak (thr={args.edge_threshold:.3f}): "
        f"density={peak_metrics.density:.3f} vs. {off_peak_metrics.density:.3f}, "
        f"avg-out={peak_metrics.avg_out_degree:.2f} vs. {off_peak_metrics.avg_out_degree:.2f}, "
        f"avg-in={peak_metrics.avg_in_degree:.2f} vs. {off_peak_metrics.avg_in_degree:.2f}, "
        f"symmetry={peak_metrics.symmetry:.3f} vs. {off_peak_metrics.symmetry:.3f}, "
        f"{ov_label}={ov:.3f}."
    )

    results = {
        "dataset": args.dataset,
        "threshold": args.edge_threshold,
        "weighted_jaccard": bool(args.weighted_jaccard),
        "symmetrize": bool(args.symmetrize),
        "every_k": args.every_k,
        "peak": {
            "edge_count": peak_metrics.edge_count,
            "density": peak_metrics.density,
            "avg_out_degree": peak_metrics.avg_out_degree,
            "avg_in_degree": peak_metrics.avg_in_degree,
            "symmetry": peak_metrics.symmetry,
        },
        "off_peak": {
            "edge_count": off_peak_metrics.edge_count,
            "density": off_peak_metrics.density,
            "avg_out_degree": off_peak_metrics.avg_out_degree,
            "avg_in_degree": off_peak_metrics.avg_in_degree,
            "symmetry": off_peak_metrics.symmetry,
        },
        "global": {
            "edge_count": global_metrics.edge_count,
            "density": global_metrics.density,
            "avg_out_degree": global_metrics.avg_out_degree,
            "avg_in_degree": global_metrics.avg_in_degree,
            "symmetry": global_metrics.symmetry,
        },
        "overlap": {
            "peak_offpeak": overlap_peak_off,
            "peak_global": overlap_peak_global,
            "offpeak_global": overlap_off_global,
        },
        "paper_snippet": paper_snippet,
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)

    _print_summary(results)
    return results


def _print_summary(results: Dict[str, Dict[str, float]]) -> None:
    peak = results["peak"]
    off_peak = results["off_peak"]
    global_g = results["global"]

    print("Graph metrics (threshold >= %.3f)" % results["threshold"])
    print("".ljust(52, "-"))
    print(f"{'Metric':<20}{'Peak':>12}{'Off-Peak':>12}{'Global':>12}")
    print(f"{'Edge count':<20}{peak['edge_count']:>12}{off_peak['edge_count']:>12}{global_g['edge_count']:>12}")
    print(f"{'Graph density':<20}{peak['density']:>12.4f}{off_peak['density']:>12.4f}{global_g['density']:>12.4f}")
    print(f"{'Avg out-degree':<20}{peak['avg_out_degree']:>12.2f}{off_peak['avg_out_degree']:>12.2f}{global_g['avg_out_degree']:>12.2f}")
    print(f"{'Avg in-degree':<20}{peak['avg_in_degree']:>12.2f}{off_peak['avg_in_degree']:>12.2f}{global_g['avg_in_degree']:>12.2f}")
    print(f"{'Symmetry':<20}{peak['symmetry']:>12.3f}{off_peak['symmetry']:>12.3f}{global_g['symmetry']:>12.3f}")
    print()

    # Overlaps
    print("Overlaps:")
    for k, v in results["overlap"].items():
        if "weighted_jaccard" in v:
            print(f"  {k.replace('_',' ')}: Weighted-Jaccard = {v['weighted_jaccard']:.4f}")
        else:
            print(f"  {k.replace('_',' ')}: Jaccard = {v['jaccard']:.4f} ({v['common_edges']} common / {v['total_union']} union)")

    print("\nPaper snippet:\n", results["paper_snippet"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare learned graph structure during peak and off-peak periods."
    )
    parser.add_argument("--dataset", default="PEMSD4", help="Dataset name (e.g., PEMSD4, PEMSD8).")
    parser.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", dest="test_ratio", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument(
        "--prediction-length",
        dest="prediction_length",
        type=int,
        default=12,
        help="Minimum samples required inside a time window.",
    )
    parser.add_argument("--lambda1", type=float, default=0.02, help="Sparsity penalty for DAGMA.")
    parser.add_argument("--normalizer", default="max01", help="Normalization method for the dataset.")
    parser.add_argument("--column-wise", action="store_true", help="Apply column-wise normalization.")
    parser.add_argument(
        "--steps-per-day",
        dest="steps_per_day",
        type=int,
        default=288,
        help="Number of temporal steps per day (PeMS datasets use 288 for 5-minute intervals).",
    )
    parser.add_argument(
        "--peak-hours",
        dest="peak_hours",
        default="7-9,16-19",
        help="Comma separated hour ranges designating peak traffic periods (e.g., '7-9,16-19').",
    )
    parser.add_argument(
        "--off-peak-hours",
        dest="off_peak_hours",
        default="1-3",
        help="Comma separated hour ranges designating off-peak periods (e.g., '1-3').",
    )
    parser.add_argument(
        "--edge-threshold",
        dest="edge_threshold",
        type=float,
        default=0.01,
        help="Threshold applied to absolute DAGMA weights when forming the adjacency.",
    )
    parser.add_argument(
        "--weighted-jaccard",
        dest="weighted_jaccard",
        action="store_true",
        help="If set, report weighted Jaccard instead of binary Jaccard.",
    )
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        help="Symmetrize learned weights by (W+W^T)/2 before thresholding.",
    )
    parser.add_argument(
        "--every-k",
        dest="every_k",
        type=int,
        default=1,
        help="Use every k-th sample in segments to speed up fitting (k>=1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed (if any randomness is present in your pipeline).",
    )
    parser.add_argument(
        "--heatmap-out",
        dest="heatmap_out",
        help="Optional path to save a heatmap of A_peak - A_off-peak (requires matplotlib).",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        help="Optional path to save the computed metrics as JSON.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    try:
        run_analysis(parser.parse_args())
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Error: {exc}", file=sys.stderr)
        raise
