"""
Offline inference and visualization for Exp_Disaster-Few-Shot.

This script runs inference on a specified dataset split using a deterministic
evaluation manifest. It requires a manifest file defining region-specific
support/query episodes, ensuring reproducible and standardized evaluation.

Example:
python3 predict.py \
    --weights path/to/your/snapshot.pth \
    --evaluation-manifest data/valset/manifest.json \
    --output-dir runs/predict/my_prediction \
    --config-json path/to/your/run/config.json
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloaders.exp_disaster_dataset import ExpDisasterFewShotDataset
from models.grid_proto_fewshot import FewShotSeg
from util.inference_utils import (
    blend_overlay,
    colorize_prediction,
    compute_binary_metrics,
    tensor_to_uint8,
)
from util.metric import Metric
from util.utils import deep_update_dict, set_seed

# The valset/testset is always for the single "disaster" class (ID 20)
TARGET_CLASS_IDS: Tuple[int, ...] = (20,)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the prediction script."""
    parser = argparse.ArgumentParser(description="Inference & visualization on Exp_Disaster-Few-Shot.")
    parser.add_argument("--weights", required=True, help="Path to the trained snapshot (.pth).")
    parser.add_argument(
        "--evaluation-manifest",
        required=True,
        help="Path to the JSON manifest defining evaluation episodes.",
    )
    parser.add_argument("--config-json", help="Optional run config JSON to override model/task defaults.")
    parser.add_argument("--output-dir", help="Directory to write predictions. Defaults to runs/predict/<snapshot>.")
    parser.add_argument("--dataset-root", help="Override dataset root directory.")
    parser.add_argument("--split", default="valset", help="Dataset split to evaluate (default: valset).")
    parser.add_argument("--n-shots", type=int, help="Override number of support shots from the manifest.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count (default: 0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode sampling.")
    parser.add_argument("--no-save-npy", action="store_true", help="Disable saving binary masks as .npy.")
    parser.add_argument("--no-save-mask", action="store_true", help="Disable saving color masks as PNG.")
    parser.add_argument("--no-save-overlay", action="store_true", help="Disable saving image overlays.")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="Alpha for overlay blending (default: 0.5).")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force evaluation device. Defaults to auto.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Constructs the configuration dictionary from args and config files."""
    config = {}  # Start with a clean config

    # Load config from the experiment's JSON file for model architecture consistency
    config_json_path = args.config_json
    if not config_json_path:
        snapshot_path = Path(args.weights).expanduser().resolve()
        candidate = snapshot_path.parent.parent / "config.json"
        if candidate.is_file():
            config_json_path = str(candidate)
    
    if config_json_path and os.path.isfile(config_json_path):
        with open(config_json_path, "r", encoding="utf-8") as fp:
            run_config = json.load(fp)
        deep_update_dict(config, run_config)
    else:
        # Fallback to a default config if no JSON is found
        from config_ssl_upload import ex
        config = deepcopy(dict(ex.configurations[0]()))
        print("Warning: Could not find config.json. Using default configuration.")

    # --- Override config with command-line arguments ---
    config["reload_model_path"] = args.weights
    config["model"]["reload_model_path"] = args.weights

    if args.dataset_root:
        config["path"][config["dataset"]]["data_dir"] = os.path.expanduser(args.dataset_root)
    if args.n_shots is not None:
        config["task"]["n_shots"] = args.n_shots
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.seed is not None:
        config["seed"] = args.seed

    # Nested validation config for output control
    val_cfg = config.setdefault("validation", {})
    val_cfg["evaluation_manifest"] = args.evaluation_manifest
    val_cfg["overlay_alpha"] = args.overlay_alpha
    val_cfg["save_numpy_preds"] = not args.no_save_npy
    val_cfg["save_color_mask"] = not args.no_save_mask
    val_cfg["save_overlay"] = not args.no_save_overlay
    if args.output_dir:
        val_cfg["output_root"] = os.path.expanduser(args.output_dir)

    return config


def prepare_output_dirs(
    config: Dict[str, Any],
    weights_path: str
) -> Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]:
    """Creates all necessary output directories for prediction artifacts."""
    val_cfg = config.get("validation", {})
    override_root = val_cfg.get("output_root")
    if override_root:
        output_root = Path(override_root)
    else:
        log_dir = config.get("path", {}).get("log_dir", "runs/predict")
        snapshot_name = Path(weights_path).stem
        output_root = Path(log_dir).expanduser() / snapshot_name

    npy_dir = output_root / "npy" if val_cfg.get("save_numpy_preds") else None
    mask_dir = output_root / "masks" if val_cfg.get("save_color_mask") else None
    overlay_dir = output_root / "overlays" if val_cfg.get("save_overlay") else None

    output_root.mkdir(parents=True, exist_ok=True)
    for directory in (npy_dir, mask_dir, overlay_dir):
        if directory:
            directory.mkdir(exist_ok=True)

    return output_root, npy_dir, mask_dir, overlay_dir


def main() -> None:
    """Main function to run the prediction pipeline."""
    args = parse_args()
    config = build_config(args)
    val_cfg = config.get("validation", {})

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensor_dtype = torch.float32
    set_seed(config.get("seed", 42))

    # --- Model Initialization ---
    model = FewShotSeg(
        image_size=config["input_size"][0],
        pretrained_path=config["reload_model_path"],
        cfg=config["model"],
    )
    model = model.to(device=device, dtype=tensor_dtype).eval()

    # --- Dataset Initialization ---
    dataset = ExpDisasterFewShotDataset(
        root_dir=os.path.expanduser(str(config["path"][config["dataset"]]["data_dir"])),
        split=args.split,
        target_classes=TARGET_CLASS_IDS,
        n_shots=config["task"]["n_shots"],
        evaluation_manifest=val_cfg["evaluation_manifest"],
        episode_seed=config.get("seed", 42),
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check manifest path and content.")

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # One episode at a time
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: batch[0], # Custom collate to handle single-item batches
    )

    # --- Execution ---
    output_root, npy_dir, mask_dir, overlay_dir = prepare_output_dirs(config, args.weights)
    metric = Metric(max_label=1)
    per_episode_records: List[Dict[str, Any]] = []

    print(f"Starting inference... Artifacts will be saved to: {output_root}")
    with torch.no_grad():
        for episode in tqdm(dataloader, desc="Predicting", total=len(dataset)):
            support_images = [[shot.unsqueeze(0).to(device, tensor_dtype) for shot in way] for way in episode["support_images"]]
            support_fg_mask = [[mask["fg_mask"].float().unsqueeze(0).to(device, tensor_dtype) for mask in way] for way in episode["support_mask"]]
            support_bg_mask = [[mask["bg_mask"].float().unsqueeze(0).to(device, tensor_dtype) for mask in way] for way in episode["support_mask"]]
            query_images = [img.unsqueeze(0).to(device, tensor_dtype) for img in episode["query_images"]]
            query_labels = torch.cat([label.long().to(device) for label in episode["query_labels"]], dim=0)

            logits, *_ = model(support_images, support_fg_mask, support_bg_mask, query_images, isval=True)
            pred_masks = logits.argmax(dim=1).cpu().numpy()
            gt_masks = query_labels.cpu().numpy()

            # --- Record metrics and save artifacts ---
            for i, query_name in enumerate(episode["query_names"]):
                pred_mask_np, gt_mask_np = pred_masks[i], gt_masks[i]
                metric.record(pred_mask_np, gt_mask_np, labels=[1])
                sample_metrics = compute_binary_metrics(pred_mask_np, gt_mask_np)
                artifact_paths: Dict[str, str] = {}

                if npy_dir:
                    np.save(npy_dir / f"{query_name}_pred.npy", pred_mask_np.astype(np.uint8))
                    artifact_paths["npy"] = str(npy_dir / f"{query_name}_pred.npy")

                color_mask = colorize_prediction(pred_mask_np) if (mask_dir or overlay_dir) else None
                if mask_dir and color_mask is not None:
                    Image.fromarray(color_mask).save(mask_dir / f"{query_name}_pred.png")
                    artifact_paths["mask_png"] = str(mask_dir / f"{query_name}_pred.png")
                
                if overlay_dir and color_mask is not None:
                    overlay = blend_overlay(tensor_to_uint8(query_images[i]), color_mask, val_cfg["overlay_alpha"])
                    Image.fromarray(overlay).save(overlay_dir / f"{query_name}_overlay.png")
                    artifact_paths["overlay_png"] = str(overlay_dir / f"{query_name}_overlay.png")

                per_episode_records.append({
                    "query_name": query_name,
                    "region": episode.get("region", "unknown"),
                    "support_names": episode["support_names"][0],
                    "metrics": sample_metrics,
                    "artifacts": artifact_paths,
                })

    # --- Final Report ---
    aggregate_metrics = metric.get_aggregate_stats(labels=[1])
    report_payload = {
        "weights_path": args.weights,
        "evaluation_manifest": args.evaluation_manifest,
        "split": args.split,
        "aggregate_metrics": aggregate_metrics,
        "samples": per_episode_records,
    }
    metrics_path = output_root / "metrics_report.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(report_payload, fp, indent=2)

    print("\n--- Inference Complete ---")
    print(f"  IoU  : {aggregate_metrics['iou']['global_mean']:.4f}")
    print(f"  Dice : {aggregate_metrics['dice']['global_mean']:.4f}")
    print(f"  F1   : {aggregate_metrics['f1']['global_mean']:.4f}")
    print(f"  OA   : {aggregate_metrics['overall_accuracy']['global_mean']:.4f}")
    print(f"\nArtifacts written to: {output_root}")
    print(f"Metrics report saved to: {metrics_path}")


if __name__ == "__main__":
    main()