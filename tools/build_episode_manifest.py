#!/usr/bin/env python3
"""
Utility to pre-compute high-quality support/query pools for few-shot episodes.

The script scans a dataset split, evaluates each sample per foreground class,
and emits a JSON manifest describing the samples that satisfy configurable
foreground coverage thresholds. Training can then consume the manifest to draw
support/query images that are less likely to yield empty prototype grids.

python -m tools.build_episode_manifest --dataset-root ../_datasets/Exp_Disaster_Few-Shot --split trainset --output ./data/trainset/manifest.json

python -m tools.build_episode_manifest --dataset-root ../_datasets/Exp_Disaster_Few-Shot --split valset --output ./data/valset/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.exp_disaster_dataset import _read_label_mask

DEFAULT_INPUT_SIZE = 512
DEFAULT_BACKBONE_STRIDE = 14
DEFAULT_PROTO_GRID = 16
DEFAULT_THRESHOLD = 0.95


@dataclass
class SampleQuality:
    """Per-sample quality metrics for a given foreground class."""

    name: str
    fg_ratio: float
    max_patch_score: float
    fg_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a JSON manifest of high-quality few-shot episodes.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path pointing to Exp_Disaster_Few-Shot (contains trainset/valset).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="trainset",
        help="Dataset split to scan (trainset or valset). Default: trainset.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="Class IDs to consider as foreground. "
        "Default: trainset→1..8, valset→20.",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=5,
        help="Number of support images per episode. Used for validation only.",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=1,
        help="Number of query images per episode. Used for validation only.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Model input resolution (assumes square). Default: 512.",
    )
    parser.add_argument(
        "--backbone-stride",
        type=int,
        default=DEFAULT_BACKBONE_STRIDE,
        help="Downsampling factor of the vision backbone. Default: 14.",
    )
    parser.add_argument(
        "--proto-grid-size",
        type=int,
        default=DEFAULT_PROTO_GRID,
        help="Prototype grid resolution. Default: 16.",
    )
    parser.add_argument(
        "--fg-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Foreground activation threshold used by the model. Default: 0.95.",
    )
    parser.add_argument(
        "--min-support-ratio",
        type=float,
        default=0.01,
        help="Minimum foreground pixel ratio required for support samples.",
    )
    parser.add_argument(
        "--min-query-ratio",
        type=float,
        default=0.005,
        help="Minimum foreground pixel ratio required for query samples.",
    )
    parser.add_argument(
        "--min-support-patch",
        type=float,
        default=0.95,
        help="Minimum pooled foreground coverage for supports.",
    )
    parser.add_argument(
        "--min-query-patch",
        type=float,
        default=0.75,
        help="Minimum pooled foreground coverage for queries.",
    )
    parser.add_argument(
        "--support-pool",
        type=int,
        default=16,
        help="Maximum number of support candidates stored per class.",
    )
    parser.add_argument(
        "--query-pool",
        type=int,
        default=32,
        help="Maximum number of query candidates stored per class.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination path for the JSON manifest.",
    )
    return parser.parse_args()


def resolve_default_classes(split: str) -> List[int]:
    split_lower = split.lower()
    if "val" in split_lower:
        return [20]
    if "train" in split_lower:
        return list(range(1, 9))
    # Fallback: include standard land-cover ids plus disaster id
    return list(range(1, 9)) + [20]


def infer_kernel_size(input_size: int, backbone_stride: int, proto_grid_size: int) -> int:
    """
    Reproduce the effective prototype window used by AM²P.

    DINO-style backbones typically yield feature maps of size
    max(input_size // stride, DEFAULT_FEATURE_SIZE). A prototype grid splits
    the feature map evenly, so the effective window is feature_hw // grid_hw.
    """
    feature_hw = max(input_size // backbone_stride, 32)
    kernel = max(1, feature_hw // proto_grid_size)
    return kernel


def compute_quality_scores(
    labels_dir: str,
    target_classes: Iterable[int],
    kernel_size: int,
) -> Dict[int, List[SampleQuality]]:
    """Compute per-class quality scores for all samples within the split."""
    per_class: Dict[int, List[SampleQuality]] = {cls_id: [] for cls_id in target_classes}
    label_files = sorted(
        fname for fname in os.listdir(labels_dir) if fname.lower().endswith((".tif", ".tiff"))
    )
    if not label_files:
        raise FileNotFoundError(f"No GeoTIFF labels found under {labels_dir}.")

    for label_name in label_files:
        stem = os.path.splitext(label_name)[0]
        label_path = os.path.join(labels_dir, label_name)
        mask = _read_label_mask(label_path)
        total_pixels = mask.size
        tensor_mask = None  # lazily materialise to torch only if needed

        for cls_id in target_classes:
            fg_mask = (mask == cls_id).astype(np.float32)
            fg_pixels = int(fg_mask.sum())
            if fg_pixels == 0:
                continue

            fg_ratio = fg_pixels / float(total_pixels)
            if tensor_mask is None:
                tensor_mask = torch.from_numpy(mask.astype(np.int16))

            fg_tensor = (tensor_mask == cls_id).float().unsqueeze(0).unsqueeze(0)
            pooled = F.avg_pool2d(fg_tensor, kernel_size, stride=1, padding=0)
            max_patch_score = float(pooled.max().item()) if pooled.numel() > 0 else 0.0

            per_class[cls_id].append(
                SampleQuality(
                    name=stem,
                    fg_ratio=float(fg_ratio),
                    max_patch_score=max_patch_score,
                    fg_pixels=fg_pixels,
                )
            )

    return per_class


def filter_candidates(
    stats: List[SampleQuality],
    min_ratio: float,
    min_patch: float,
    limit: int,
) -> List[SampleQuality]:
    """Filter and rank sample statistics using the provided thresholds."""
    filtered = [
        sample
        for sample in stats
        if sample.fg_ratio >= min_ratio and sample.max_patch_score >= min_patch
    ]
    if not filtered:
        return []

    filtered.sort(key=lambda sample: (sample.max_patch_score, sample.fg_ratio), reverse=True)
    if limit > 0:
        return filtered[:limit]
    return filtered


def build_manifest(args: argparse.Namespace) -> Dict[str, object]:
    split_dir = os.path.join(args.dataset_root, args.split)
    labels_dir = os.path.join(split_dir, "labels")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Expected labels directory at {labels_dir}.")

    target_classes = args.classes or resolve_default_classes(args.split)
    target_classes = sorted(set(int(cls_id) for cls_id in target_classes))

    kernel_size = infer_kernel_size(args.input_size, args.backbone_stride, args.proto_grid_size)
    per_class_scores = compute_quality_scores(labels_dir, target_classes, kernel_size)

    manifest: Dict[str, object] = {
        "version": 1,
        "meta": {
            "split": args.split,
            "dataset_root": os.path.abspath(args.dataset_root),
            "n_shots": args.n_shots,
            "n_queries": args.n_queries,
            "kernel_size": kernel_size,
            "fg_threshold": args.fg_threshold,
            "min_support_ratio": args.min_support_ratio,
            "min_support_patch": args.min_support_patch,
            "min_query_ratio": args.min_query_ratio,
            "min_query_patch": args.min_query_patch,
            "classes": target_classes,
        },
        "classes": {},
    }

    for cls_id in target_classes:
        stats = per_class_scores.get(cls_id, [])
        support_pool = filter_candidates(
            stats,
            min_ratio=args.min_support_ratio,
            min_patch=args.min_support_patch,
            limit=args.support_pool,
        )

        query_pool = filter_candidates(
            stats,
            min_ratio=args.min_query_ratio,
            min_patch=args.min_query_patch,
            limit=args.query_pool,
        )

        manifest["classes"][str(cls_id)] = {
            "support_pool": [asdict(sample) for sample in support_pool],
            "query_pool": [asdict(sample) for sample in query_pool],
            "total_candidates": len(stats),
        }

    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Wrote manifest for split '{args.split}' to {args.output}")


if __name__ == "__main__":
    main()
