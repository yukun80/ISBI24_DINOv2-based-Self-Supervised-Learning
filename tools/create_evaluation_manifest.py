#!/usr/bin/env python3
"""
Utility to generate a JSON manifest for region-specific evaluation.

This script scans the valset, groups images by geographical region,
and creates fixed support/query splits for each region.

python -m tools.create_evaluation_manifest --dataset-root ../_datasets/Exp_Disaster_Few-Shot --split valset --output data/valset/manifest.json
"""
import argparse
import json
import os
import random
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a region-specific evaluation manifest.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the Exp_Disaster_Few-Shot dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valset",
        help="Dataset split to process (default: valset).",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=5,
        help="Number of support images per region (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the JSON manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    image_dir = os.path.join(args.dataset_root, args.split, "images")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found at {image_dir}")

    regions = defaultdict(list)
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".tif", ".tiff")):
            continue

        basename = os.path.splitext(filename)[0]
        try:
            region_name = basename.split("_")[0]
            regions[region_name].append(basename)
        except IndexError:
            print(f"Warning: Could not parse region from filename: {filename}")

    manifest = {
        "version": 1,
        "meta": {
            "split": args.split,
            "n_shots": args.n_shots,
            "seed": args.seed,
            "description": "Region-specific support/query splits for validation.",
        },
        "regions": {},
    }

    for region, basenames in regions.items():
        if len(basenames) < args.n_shots + 1:
            print(
                f"Warning: Not enough images in region '{region}' to create a split. Need at least {args.n_shots + 1}, found {len(basenames)}. Skipping."
            )
            continue

        random.shuffle(basenames)
        support_set = basenames[: args.n_shots]
        query_set = basenames[args.n_shots :]

        manifest["regions"][region] = {
            "support": support_set,
            "query": query_set,
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully created evaluation manifest at {args.output}")


if __name__ == "__main__":
    main()
