"""
Offline inference and visualization for Exp_Disaster-Few-Shot.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config_ssl_upload import ex
from dataloaders.exp_disaster_dataset import ExpDisasterFewShotDataset
from models.grid_proto_fewshot import FewShotSeg
from util.inference_utils import blend_overlay, colorize_prediction, compute_binary_metrics, tensor_to_uint8
from util.metric import Metric
from util.utils import deep_update_dict, set_seed

"""
python3 predict.py --weights runs/disaster_fewshot_run_EXP_DISASTER_FEWSHOT_5shot/3/snapshots/15000.pth \
  --config-json runs/disaster_fewshot_run_EXP_DISASTER_FEWSHOT_5shot/3/config.json \
  --output-dir runs/disaster_fewshot_run_EXP_DISASTER_FEWSHOT_5shot/3/disaster_preds/predict \
  --split valset --support-manifest data/valset/manifest.json \
    --num-workers 8 --overlay-alpha 0.5

"""

TARGET_CLASS_IDS: Tuple[int, ...] = (20,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference & visualization on Exp_Disaster-Few-Shot.")
    parser.add_argument("--weights", required=True, help="Path to the trained snapshot (.pth).")
    parser.add_argument("--config-json", help="Optional Sacred config JSON to override defaults.")
    parser.add_argument("--output-dir", help="Directory to write predictions. Defaults to runs/predict/<snapshot>.")
    parser.add_argument("--dataset-root", help="Override dataset root directory.")
    parser.add_argument("--split", default="valset", help="Dataset split to evaluate (default: valset).")
    parser.add_argument("--support-manifest", help="Path to deterministic support manifest.")
    parser.add_argument("--n-shots", type=int, help="Override number of support shots.")
    parser.add_argument("--n-queries", type=int, help="Override number of queries per episode.")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, help="Random seed for episode sampling.")
    parser.add_argument("--no-save-npy", action="store_true", help="Disable saving binary masks as .npy.")
    parser.add_argument("--no-save-mask", action="store_true", help="Disable saving color masks as PNG.")
    parser.add_argument("--no-save-overlay", action="store_true", help="Disable saving image overlays.")
    parser.add_argument("--overlay-alpha", type=float, help="Alpha for overlay blending (default: 0.5).")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force evaluation device. Defaults to auto.")
    args, unknown = parser.parse_known_args()
    if unknown:
        hint = ""
        json_like = [arg for arg in unknown if arg.endswith(".json")]
        if json_like:
            hint = " Did you forget to prefix the JSON path with `--support-manifest`?"
        parser.error(f"unrecognized arguments: {' '.join(unknown)}.{hint}")
    return args


def load_base_config() -> Dict[str, Any]:
    default_summary = ex.configurations[0]()
    return deepcopy(dict(default_summary))


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_base_config()
    config_json_path = args.config_json
    if not config_json_path:
        snapshot_path = Path(args.weights).expanduser().resolve()
        candidates = [
            snapshot_path.parent.parent / "config.json",
            snapshot_path.parent / "config.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                config_json_path = str(candidate)
                break
    if config_json_path:
        config_json_path = os.path.expanduser(config_json_path)
        with open(config_json_path, "r", encoding="utf-8") as fp:
            run_config = json.load(fp)
        deep_update_dict(config, run_config)
        config.setdefault("validation", {})
        config["validation"].setdefault("config_json", config_json_path)

    config["reload_model_path"] = args.weights
    config["model"]["reload_model_path"] = args.weights
    config.setdefault("validation", {})
    config["validation"].setdefault("save_numpy_preds", True)
    config["validation"].setdefault("save_color_mask", True)
    config["validation"].setdefault("save_overlay", True)
    config["validation"].setdefault("overlay_alpha", 0.5)

    if args.dataset_root:
        dataset_key = config["dataset"]
        config["path"][dataset_key]["data_dir"] = os.path.expanduser(args.dataset_root)
    if args.support_manifest:
        manifest = os.path.expanduser(args.support_manifest)
        config["support_txt_file"] = manifest
        config["validation"]["support_manifest"] = manifest
    config["episode_manifest"] = None
    config["validation"]["episode_manifest"] = None
    if args.n_shots is not None:
        config["n_shots"] = args.n_shots
        config["task"]["n_shots"] = args.n_shots
    if args.n_queries is not None:
        config["n_queries"] = args.n_queries
        config["task"]["n_queries"] = args.n_queries
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
        config["validation"]["num_workers"] = args.num_workers
    if args.seed is not None:
        config["seed"] = args.seed

    if args.overlay_alpha is not None:
        config["validation"]["overlay_alpha"] = float(args.overlay_alpha)

    config["validation"]["save_numpy_preds"] = not args.no_save_npy
    config["validation"]["save_color_mask"] = not args.no_save_mask
    config["validation"]["save_overlay"] = not args.no_save_overlay
    if args.output_dir:
        config["validation"]["output_root"] = os.path.expanduser(args.output_dir)

    config["validation"]["val_snapshot_path"] = args.weights
    return config


def _default_support_subset(dataset: ExpDisasterFewShotDataset, n_shots: int) -> List[str]:
    selected: List[str] = []
    for record in dataset.records.values():
        if record.name not in selected and set(record.classes) & set(dataset.target_classes):
            selected.append(record.name)
        if len(selected) >= n_shots:
            break
    return selected[:n_shots]


def prepare_output_root(
    config: Dict[str, Any], weights_path: str
) -> Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]:
    validation_cfg = config.get("validation", {})
    override_root = validation_cfg.get("output_root")
    if override_root:
        output_root = Path(override_root).expanduser()
    else:
        log_dir = config["path"]["log_dir"]
        snapshot_name = Path(weights_path).stem
        output_root = Path(log_dir).expanduser() / "predict" / snapshot_name

    save_numpy = bool(validation_cfg.get("save_numpy_preds", True))
    save_color_mask = bool(validation_cfg.get("save_color_mask", True))
    save_overlay = bool(validation_cfg.get("save_overlay", True))

    npy_dir = output_root / "npy" if save_numpy else None
    mask_dir = output_root / "masks" if save_color_mask else None
    overlay_dir = output_root / "overlays" if save_overlay else None

    output_root.mkdir(parents=True, exist_ok=True)
    for directory in (npy_dir, mask_dir, overlay_dir):
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    return output_root, npy_dir, mask_dir, overlay_dir


def move_to_precision(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def main() -> None:
    args = parse_args()
    config = build_config(args)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensor_dtype = torch.float32

    set_seed(config.get("seed", 0))

    model = FewShotSeg(
        image_size=config["input_size"][0],
        pretrained_path=config["model"].get("reload_model_path") or config["reload_model_path"],
        cfg=config["model"],
    )
    model = model.to(device=device, dtype=tensor_dtype).eval()

    dataset_root = os.path.expanduser(str(config["path"][config["dataset"]]["data_dir"]))
    support_manifest = config["validation"].get("support_manifest") or config.get("support_txt_file")
    if support_manifest:
        support_manifest = os.path.expanduser(str(support_manifest))
    dataset = ExpDisasterFewShotDataset(
        root_dir=dataset_root,
        split=args.split,
        target_classes=TARGET_CLASS_IDS,
        n_shots=config["task"]["n_shots"],
        n_queries=config["task"]["n_queries"],
        transforms=None,
        max_iters_per_epoch=config["validation"].get("max_iters_per_epoch", 1),
        support_manifest=support_manifest,
        episode_seed=config.get("seed", 0),
    )

    if not dataset.fixed_support:
        dataset.fixed_support = _default_support_subset(dataset, config["task"]["n_shots"])
    if len(dataset) == 0:
        raise RuntimeError("No query samples found for inference.")

    num_workers = config["validation"].get("num_workers", config.get("num_workers", 0))
    pin_memory = bool(config["validation"].get("pin_memory", device.type == "cuda"))

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: batch[0],
    )

    output_root, npy_dir, mask_dir, overlay_dir = prepare_output_root(config, args.weights)
    overlay_alpha = float(config["validation"].get("overlay_alpha", 0.5))

    metric = Metric(max_label=1)
    metric.reset()
    per_episode_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for episode_idx, episode in enumerate(tqdm(dataloader, desc="Predict", total=len(dataset))):
            support_images = [
                [move_to_precision(shot.unsqueeze(0), device, tensor_dtype) for shot in way]
                for way in episode["support_images"]
            ]
            support_fg_mask = [
                [move_to_precision(mask["fg_mask"].float().unsqueeze(0), device, tensor_dtype) for mask in way]
                for way in episode["support_mask"]
            ]
            support_bg_mask = [
                [move_to_precision(mask["bg_mask"].float().unsqueeze(0), device, tensor_dtype) for mask in way]
                for way in episode["support_mask"]
            ]
            query_images = [move_to_precision(img.unsqueeze(0), device, tensor_dtype) for img in episode["query_images"]] 
            query_labels = torch.cat(
                [label.long().to(device=device, non_blocking=True) for label in episode["query_labels"]],
                dim=0,
            )

            logits, *_ = model(
                support_images,
                support_fg_mask,
                support_bg_mask,
                query_images,
                isval=True,
                val_wsize=None,
            )
            pred_masks = logits.argmax(dim=1).cpu().numpy()
            gt_masks = query_labels.cpu().numpy()

            episode_class = int(episode["class_ids"][0][0]) if episode.get("class_ids") else TARGET_CLASS_IDS[0]
            support_names_raw = episode.get("support_names", [[]])
            support_names = [str(name) for name in support_names_raw[0]] if support_names_raw else []

            for query_tensor, query_name, pred_mask_np, gt_mask_np in zip(
                query_images, episode["query_names"], pred_masks, gt_masks
            ):
                metric.record(pred_mask_np, gt_mask_np, labels=[1])
                sample_metrics = compute_binary_metrics(pred_mask_np, gt_mask_np)
                artifact_paths: Dict[str, str] = {}

                if npy_dir and config["validation"].get("save_numpy_preds", True):
                    npy_path = npy_dir / f"{query_name}_pred.npy"
                    np.save(npy_path, pred_mask_np.astype(np.uint8))
                    artifact_paths["npy"] = str(npy_path)

                save_color_flag = config["validation"].get("save_color_mask", True)
                save_overlay_flag = config["validation"].get("save_overlay", True)
                color_mask = colorize_prediction(pred_mask_np) if (save_color_flag or save_overlay_flag) else None

                if mask_dir and color_mask is not None and save_color_flag:
                    mask_path = mask_dir / f"{query_name}_pred.png"
                    Image.fromarray(color_mask).save(mask_path)
                    artifact_paths["mask_png"] = str(mask_path)

                if overlay_dir and color_mask is not None and save_overlay_flag:
                    query_uint8 = tensor_to_uint8(query_tensor)
                    overlay = blend_overlay(query_uint8, color_mask, overlay_alpha)
                    overlay_path = overlay_dir / f"{query_name}_overlay.png"
                    Image.fromarray(overlay).save(overlay_path)
                    artifact_paths["overlay_png"] = str(overlay_path)

                per_episode_records.append(
                    {
                        "index": episode_idx,
                        "class_id": episode_class,
                        "query_name": query_name,
                        "support_names": support_names,
                        "metrics": sample_metrics,
                        "artifacts": artifact_paths,
                    }
                )

    dice_mean, dice_std, dice_global_mean, dice_global_std = metric.get_mDice(labels=[1])
    iou_mean, iou_std, iou_global_mean, iou_global_std = metric.get_mIoU(labels=[1])
    prf1_stats = metric.get_precision_recall_f1(labels=[1])
    oa_mean, oa_std, oa_global_mean, oa_global_std = metric.get_overall_accuracy(labels=[1])

    precision_stats = prf1_stats["precision"]
    recall_stats = prf1_stats["recall"]
    f1_stats = prf1_stats["f1"]

    precision_class_mean = np.asarray(precision_stats[0])
    precision_class_std = np.asarray(precision_stats[1]) if precision_stats[1] is not None else None
    precision_global_mean = float(precision_stats[2])
    precision_global_std = float(precision_stats[3]) if precision_stats[3] is not None else 0.0

    recall_class_mean = np.asarray(recall_stats[0])
    recall_class_std = np.asarray(recall_stats[1]) if recall_stats[1] is not None else None
    recall_global_mean = float(recall_stats[2])
    recall_global_std = float(recall_stats[3]) if recall_stats[3] is not None else 0.0

    f1_class_mean = np.asarray(f1_stats[0])
    f1_class_std = np.asarray(f1_stats[1]) if f1_stats[1] is not None else None
    f1_global_mean = float(f1_stats[2])
    f1_global_std = float(f1_stats[3]) if f1_stats[3] is not None else 0.0

    oa_class_mean = np.asarray(oa_mean)
    oa_class_std = np.asarray(oa_std) if oa_std is not None else None
    oa_global_mean = float(oa_global_mean)
    oa_global_std = float(oa_global_std)

    aggregate_metrics = {
        "dice": {
            "class_mean": dice_mean.tolist(),
            "class_std": dice_std.tolist(),
            "global_mean": float(dice_global_mean),
            "global_std": float(dice_global_std),
        },
        "iou": {
            "class_mean": iou_mean.tolist(),
            "class_std": iou_std.tolist(),
            "global_mean": float(iou_global_mean),
            "global_std": float(iou_global_std),
        },
        "precision": {
            "class_mean": precision_class_mean.tolist(),
            "class_std": (precision_class_std.tolist() if precision_class_std is not None else None),
            "global_mean": precision_global_mean,
            "global_std": precision_global_std,
        },
        "recall": {
            "class_mean": recall_class_mean.tolist(),
            "class_std": (recall_class_std.tolist() if recall_class_std is not None else None),
            "global_mean": recall_global_mean,
            "global_std": recall_global_std,
        },
        "f1": {
            "class_mean": f1_class_mean.tolist(),
            "class_std": (f1_class_std.tolist() if f1_class_std is not None else None),
            "global_mean": f1_global_mean,
            "global_std": f1_global_std,
        },
        "overall_accuracy": {
            "class_mean": oa_class_mean.tolist(),
            "class_std": (oa_class_std.tolist() if oa_class_std is not None else None),
            "global_mean": oa_global_mean,
            "global_std": oa_global_std,
        },
    }

    report_payload = {
        "weights_path": args.weights,
        "config_json": args.config_json,
        "support_manifest": support_manifest,
        "fixed_support": dataset.fixed_support,
        "split": args.split,
        "aggregate": aggregate_metrics,
        "samples": per_episode_records,
    }

    metrics_path = output_root / config["validation"].get("metrics_filename", "metrics_report.json")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(report_payload, fp, indent=2)

    print(
        f"Inference complete. "
        f"IoU={aggregate_metrics['iou']['global_mean']:.4f}, "
        f"Precision={precision_global_mean:.4f}, "
        f"Recall={recall_global_mean:.4f}, "
        f"F1={f1_global_mean:.4f}, "
        f"OA={oa_global_mean:.4f}"
    )
    print(f"Artifacts written to: {output_root}")
    print(f"Metrics report: {metrics_path}")


if __name__ == "__main__":
    main()
