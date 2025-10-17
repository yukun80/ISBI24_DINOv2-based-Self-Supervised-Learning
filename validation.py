"""
Validation script tailored for the Exp_Disaster_Few-Shot dataset.
"""
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config_ssl_upload import ex
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

"""
python3 validation.py with \
    validation.val_snapshot_path=path/to/your/snapshot.pth \
    validation.evaluation_manifest=data/valset/manifest.json

# The script will automatically try to find config.json from the snapshot path.
# You can also specify it manually:
python3 validation.py with \
    validation.val_snapshot_path=runs/disaster_fewshot_run_EXP_DISASTER_FEWSHOT_5shot/3/snapshots/20000.pth \
    validation.config_json=runs/disaster_fewshot_run_EXP_DISASTER_FEWSHOT_5shot/3/config.json \
    validation.evaluation_manifest=data/valset/manifest.json
"""


def _prepare_observer_artifacts(_run) -> None:
    if not _run.observers:
        return
    os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
    for source_file, _ in _run.experiment_info['sources']:
        abs_path = f'{_run.observers[0].dir}/source/{source_file}'
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')



def _ensure_image_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def _ensure_mask_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    return tensor


@ex.automain
def main(_run, _config, _log):
    precision = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(False)

    _prepare_observer_artifacts(_run)
    set_seed(_config['seed'])

    validation_cfg: Dict[str, Any] = dict(_config.get('validation', {}))
    snapshot_path: Optional[str] = (
        validation_cfg.get('val_snapshot_path')
        or _config.get('reload_model_path')
        or _config.get('model', {}).get('reload_model_path')
        or None
    )
    if snapshot_path:
        snapshot_path = os.path.expanduser(str(snapshot_path))
        if not os.path.isfile(snapshot_path):
            _log.warning("Snapshot path '%s' does not exist; falling back to random weights.", snapshot_path)
            snapshot_path = None

    config_json_path = validation_cfg.get('config_json')
    if not config_json_path and snapshot_path:
        candidate = Path(snapshot_path).expanduser().resolve().parent.parent / "config.json"
        if candidate.is_file():
            config_json_path = str(candidate)
    if config_json_path:
        config_json_path = os.path.expanduser(str(config_json_path))
        if os.path.isfile(config_json_path):
            _log.info("Loading configuration overrides from %s", config_json_path)
            with open(config_json_path, "r", encoding="utf-8") as fp:
                config_override = json.load(fp)
            deep_update_dict(_config, config_override)
            _config.setdefault('validation', {})
            _config['validation'].setdefault('config_json', config_json_path)
            validation_cfg = dict(_config.get('validation', {}))
        else:
            _log.warning("Requested config_json '%s' not found.", config_json_path)

    if snapshot_path:
        _config['reload_model_path'] = snapshot_path
        _config.setdefault('model', {})
        _config['model']['reload_model_path'] = snapshot_path
        validation_cfg.setdefault('val_snapshot_path', snapshot_path)

    _log.info('###### Create model ######')
    model = FewShotSeg(
        image_size=_config['input_size'][0],
        pretrained_path=snapshot_path,
        cfg=_config['model'],
    )
    model = model.to(device, precision).eval()

    dataset_root = os.path.expanduser(str(_config['path'][_config['dataset']]['data_dir']))

    # Evaluation is now driven exclusively by a manifest.
    evaluation_manifest = validation_cfg.get('evaluation_manifest')
    if not evaluation_manifest:
        raise ValueError(
            "Missing 'validation.evaluation_manifest' path in configuration. "
            "Deterministic evaluation requires a pre-generated manifest. "
            "Please create one using 'tools/create_evaluation_manifest.py'."
        )

    evaluation_manifest = os.path.expanduser(str(evaluation_manifest))
    if not os.path.isfile(evaluation_manifest):
        raise FileNotFoundError(f"Evaluation manifest '{evaluation_manifest}' not found.")

    _log.info(f"Using evaluation manifest: {evaluation_manifest}")
    dataset = ExpDisasterFewShotDataset(
        root_dir=dataset_root,
        split='valset',
        target_classes=[20],  # valset is always class 20
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries'],
        evaluation_manifest=evaluation_manifest,
        episode_seed=_config['seed'],
    )

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset created from manifest '{evaluation_manifest}' is empty.")

    num_workers = int(validation_cfg.get('num_workers', 0))
    pin_memory = bool(validation_cfg.get('pin_memory', device.type == 'cuda'))

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: batch[0],
    )

    metric = Metric(max_label=1)
    metric.reset()

    output_dir_name = validation_cfg.get('output_dir_name', 'disaster_preds')
    output_root_override = validation_cfg.get('output_root')
    if output_root_override:
        output_root = Path(output_root_override).expanduser()
    elif _run.observers:
        output_root = Path(_run.observers[0].dir) / output_dir_name
    else:
        output_root = Path(_config['path']['log_dir']).expanduser() / "manual_validation" / output_dir_name

    save_numpy = bool(validation_cfg.get('save_numpy_preds', True))
    save_color_mask = bool(validation_cfg.get('save_color_mask', True))
    save_overlay = bool(validation_cfg.get('save_overlay', True))
    overlay_alpha = float(validation_cfg.get('overlay_alpha', 0.5))
    metrics_filename = validation_cfg.get('metrics_filename', 'metrics_report.json')

    npy_dir: Optional[Path] = None
    color_dir: Optional[Path] = None
    overlay_dir: Optional[Path] = None
    output_root.mkdir(parents=True, exist_ok=True)
    if save_numpy or save_color_mask or save_overlay:
        if save_numpy:
            npy_dir = output_root / "npy"
            npy_dir.mkdir(parents=True, exist_ok=True)
        if save_color_mask:
            color_dir = output_root / "masks"
            color_dir.mkdir(parents=True, exist_ok=True)
        if save_overlay:
            overlay_dir = output_root / "overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)

    _log.info('###### Starting validation ######')
    per_episode_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for episode_idx, episode in enumerate(tqdm(dataloader, total=len(dataset))):
            support_images = [
                [_ensure_image_batch_dim(shot).to(device, precision, non_blocking=True) for shot in way]
                for way in episode['support_images']
            ]
            support_fg_mask = [
                [_ensure_mask_batch_dim(mask['fg_mask']).float().to(device, precision, non_blocking=True) for mask in way]
                for way in episode['support_mask']
            ]
            support_bg_mask = [
                [_ensure_mask_batch_dim(mask['bg_mask']).float().to(device, precision, non_blocking=True) for mask in way]
                for way in episode['support_mask']
            ]
            query_images = [
                _ensure_image_batch_dim(img).to(device, precision, non_blocking=True)
                for img in episode['query_images']
            ]
            query_labels = torch.cat(
                [
                    _ensure_mask_batch_dim(label).long().to(device, non_blocking=True)
                    for label in episode['query_labels']
                ],
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

            episode_class = int(episode['class_ids'][0][0]) if episode.get('class_ids') else 20
            raw_support_names = episode.get('support_names', [[]])
            support_names = [str(name) for name in raw_support_names[0]] if raw_support_names else []

            for query_tensor, query_name, pred_mask_np, gt_mask_np in zip(
                query_images, episode['query_names'], pred_masks, gt_masks
            ):
                metric.record(pred_mask_np, gt_mask_np, labels=[1])
                sample_metrics = compute_binary_metrics(pred_mask_np, gt_mask_np)
                artifact_paths: Dict[str, str] = {}

                if save_numpy and npy_dir:
                    npy_path = npy_dir / f"{query_name}_pred.npy"
                    np.save(npy_path, pred_mask_np.astype(np.uint8))
                    artifact_paths['npy'] = str(npy_path)

                color_mask = colorize_prediction(pred_mask_np) if (save_color_mask or save_overlay) else None

                if save_color_mask and color_dir and color_mask is not None:
                    color_path = color_dir / f"{query_name}_pred.png"
                    Image.fromarray(color_mask).save(color_path)
                    artifact_paths['mask_png'] = str(color_path)

                if save_overlay and overlay_dir and color_mask is not None:
                    query_uint8 = tensor_to_uint8(query_tensor)
                    overlay = blend_overlay(query_uint8, color_mask, overlay_alpha)
                    overlay_path = overlay_dir / f"{query_name}_overlay.png"
                    Image.fromarray(overlay).save(overlay_path)
                    artifact_paths['overlay_png'] = str(overlay_path)

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

    precision_stats = prf1_stats['precision']
    recall_stats = prf1_stats['recall']
    f1_stats = prf1_stats['f1']

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

    _run.log_scalar('dice_fg_mean', dice_global_mean)
    _run.log_scalar('dice_fg_std', dice_global_std)
    _run.log_scalar('iou_fg_mean', iou_global_mean)
    _run.log_scalar('iou_fg_std', iou_global_std)
    _run.log_scalar('precision_fg_mean', precision_global_mean)
    _run.log_scalar('precision_fg_std', precision_global_std)
    _run.log_scalar('recall_fg_mean', recall_global_mean)
    _run.log_scalar('recall_fg_std', recall_global_std)
    _run.log_scalar('f1_fg_mean', f1_global_mean)
    _run.log_scalar('f1_fg_std', f1_global_std)
    _run.log_scalar('overall_accuracy_mean', oa_global_mean)
    _run.log_scalar('overall_accuracy_std', oa_global_std)

    _log.info(
        f"Validation Dice (fg) mean={dice_global_mean:.4f}±{dice_global_std:.4f}; "
        f"class-wise={dice_mean}"
    )
    _log.info(
        f"Validation IoU (fg) mean={iou_global_mean:.4f}±{iou_global_std:.4f}; "
        f"class-wise={iou_mean}"
    )

    _log.info(
        f"Precision={precision_global_mean:.4f}±{precision_global_std:.4f}, "
        f"Recall={recall_global_mean:.4f}±{recall_global_std:.4f}, "
        f"F1={f1_global_mean:.4f}±{f1_global_std:.4f}, "
        f"OA={oa_global_mean:.4f}±{oa_global_std:.4f}"
    )

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
        "snapshot_path": snapshot_path,
        "evaluation_manifest": evaluation_manifest,
        "n_shots": _config['task']['n_shots'],
        "n_queries": _config['task']['n_queries'],
        "aggregate": aggregate_metrics,
        "samples": per_episode_records,
    }

    report_path = output_root / metrics_filename
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report_payload, fp, indent=2)
    _log.info("Saved validation metrics to %s", report_path)
