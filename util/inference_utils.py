"""
Shared utilities for validation and offline inference.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

ForegroundColor = Tuple[int, int, int]

DEFAULT_FG_COLOR: ForegroundColor = (255, 0, 0)
DEFAULT_BG_COLOR: ForegroundColor = (0, 0, 0)


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """
    Convert an image tensor (C, H, W) with values in [0, 1] or [0, 255] to uint8 HWC.
    """
    array = image.detach().cpu().numpy()
    if array.ndim == 4:
        # Validation loaders sometimes keep a leading batch dimension for single images.
        if array.shape[0] != 1:
            raise ValueError(f"Expected a single image tensor, got batch dimension {array.shape[0]}")
        array = array[0]
    if array.ndim == 2:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dimensions, got shape {array.shape}")
    array = np.transpose(array, (1, 2, 0))
    if array.max() > 1.0:
        array = np.clip(array, 0.0, 255.0)
    else:
        array = np.clip(array, 0.0, 1.0) * 255.0
    return array.astype(np.uint8)


def colorize_prediction(
    mask: np.ndarray,
    fg_color: ForegroundColor = DEFAULT_FG_COLOR,
    bg_color: ForegroundColor = DEFAULT_BG_COLOR,
) -> np.ndarray:
    """
    Render a binary mask as a 3-channel uint8 image.
    """
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[...] = np.array(bg_color, dtype=np.uint8)
    color[mask == 1] = np.array(fg_color, dtype=np.uint8)
    return color


def blend_overlay(
    image_uint8: np.ndarray,
    mask_color: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay the colorized mask onto the RGB image.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = image_uint8.astype(np.float32).copy()
    mask_region = mask_color[..., 0] > 0
    if not np.any(mask_region):
        return image_uint8
    blended[mask_region] = (
        (1.0 - alpha) * blended[mask_region]
        + alpha * mask_color[mask_region].astype(np.float32)
    )
    return blended.clip(0, 255).astype(np.uint8)


def compute_binary_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute confusion-matrix metrics for binary segmentation masks.
    """
    pred_fg = pred.astype(np.uint8) == 1
    target_fg = target.astype(np.uint8) == 1
    tp = np.logical_and(pred_fg, target_fg).sum()
    fp = np.logical_and(pred_fg, ~target_fg).sum()
    fn = np.logical_and(~pred_fg, target_fg).sum()
    tn = np.logical_and(~pred_fg, ~target_fg).sum()

    denom_iou = tp + fp + fn
    denom_prec = tp + fp
    denom_rec = tp + fn
    denom_total = tp + fp + fn + tn

    iou = tp / denom_iou if denom_iou > 0 else 0.0
    precision = tp / denom_prec if denom_prec > 0 else 0.0
    recall = tp / denom_rec if denom_rec > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    oa = (tp + tn) / denom_total if denom_total > 0 else 0.0
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "overall_accuracy": float(oa),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total": int(denom_total),
    }
