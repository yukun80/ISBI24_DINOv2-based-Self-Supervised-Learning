"""
Unified training script for the few-shot disaster segmentation model.

This script handles the complete meta-training and validation workflow. It uses Sacred
for experiment configuration and tracking. All parameters can be overridden from the
command line.

Example of a full training command:
------------------------------------
python3 training.py with \
    modelname=dinov2_l14 \
    clsname=grid_proto \
    lora=8 \
    task.n_shots=5 \
    n_steps=60000 \
    lr=5e-5 \
    grad_accumulation_steps=2 \
    precision.use_amp=True \
    validation.run_every_n_steps=1000 \
    validation.n_val_episodes=100 \
    dataloading.region_based_sampling=True
"""

from contextlib import nullcontext
import os
import torch
import torch.nn as nn
import torch.optim
from torch import amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from models.grid_proto_fewshot import FewShotSeg
from torch.utils.tensorboard import SummaryWriter
from dataloaders.exp_disaster_dataset import ExpDisasterFewShotDataset
import dataloaders.augutils as myaug

from util.utils import set_seed, compose_wt_simple
from util.metric import Metric

from config_ssl_upload import ex
from tqdm.auto import tqdm

from typing import Any, Dict, List

#
# --- Helper Functions: Readability and Modularity --- #
#


def get_train_transforms(_config: Dict[str, Any]) -> callable:
    """Returns the data augmentation pipeline for training."""
    return myaug.transform_with_label({"aug": myaug.get_aug(_config["which_aug"], _config["input_size"][0])})


def get_dataset(
    _config: Dict[str, Any], transforms: callable, split: str, manifest: str = None
) -> ExpDisasterFewShotDataset:
    """Initializes and returns a dataset for a given split."""
    dataset_root = _config["path"][_config["dataset"]]["data_dir"]
    return ExpDisasterFewShotDataset(
        root_dir=dataset_root,
        split=split,
        target_classes=range(1, 9) if split == "trainset" else [20],
        n_shots=_config["task"]["n_shots"],
        n_queries=_config["task"]["n_queries"],
        transforms=transforms if split == "trainset" else None,
        max_iters_per_epoch=_config["max_iters_per_load"],
        episode_seed=_config["seed"],
        evaluation_manifest=manifest,
    )


def process_episode_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Moves all tensor-like data in a single episode batch to the specified device.
    This function is designed to be DRY and centralizes data preparation for a single episode.
    """
    processed_batch = {}
    for key, value in batch.items():
        if key in ["support_images", "query_images"]:
            # Structure: list of lists of tensors OR list of tensors
            # Add a batch dimension of 1 to each image tensor.
            if isinstance(value[0], list):
                processed_batch[key] = [
                    [shot.unsqueeze(0).to(device, non_blocking=True) for shot in way] for way in value
                ]
            else:
                processed_batch[key] = [v.unsqueeze(0).to(device, non_blocking=True) for v in value]
        elif key == "support_mask":
            # Structure: list of lists of dicts of tensors
            # Add a batch dimension of 1 to each mask tensor.
            processed_batch[key] = [
                [{k: v.float().unsqueeze(0).to(device, non_blocking=True) for k, v in shot.items()} for shot in way]
                for way in value
            ]
        elif key == "query_labels":
            labels = torch.cat([v.long().to(device, non_blocking=True) for v in value], dim=0)
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            processed_batch[key] = labels
        else:
            processed_batch[key] = value  # Keep metadata as is
    return processed_batch


def run_validation(
    model: nn.Module, dataloader: DataLoader, device: torch.device, criterion: nn.Module, _run: Any
) -> float:
    """
    Runs evaluation on the validation set on a single episode at a time.
    """
    model.eval()
    metric = Metric(max_label=1)
    val_losses = []

    with torch.no_grad():
        for episode_batch in tqdm(dataloader, desc="Validating"):
            episode_batch = process_episode_batch(episode_batch, device)
            support_fg_mask = [[shot["fg_mask"] for shot in way] for way in episode_batch["support_mask"]]
            support_bg_mask = [[shot["bg_mask"] for shot in way] for way in episode_batch["support_mask"]]

            query_pred, _, _ = model(
                episode_batch["support_images"],
                support_fg_mask,
                support_bg_mask,
                episode_batch["query_images"],
                isval=True,
                val_wsize=None,
            )

            loss = criterion(query_pred, episode_batch["query_labels"])
            val_losses.append(loss.item())

            pred_masks = query_pred.argmax(dim=1).cpu().numpy()
            gt_masks = episode_batch["query_labels"].cpu().numpy()
            metric.record(pred_masks, gt_masks, labels=[1])

    _, _, miou, _ = metric.get_mIoU(labels=[1])
    avg_loss = np.mean(val_losses)

    _run.log_scalar("val_loss", avg_loss, dataloader.dataset.max_iters)
    _run.log_scalar("val_miou", miou, dataloader.dataset.max_iters)

    model.train()
    return miou, avg_loss


@ex.automain
def main(_run, _config, _log):
    #
    # 1. Setup and Initialization
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(_config["seed"])

    # AMP setup
    precision_cfg = _config.get("precision", {})
    amp_enabled = bool(precision_cfg.get("use_amp", False)) and device.type == "cuda"
    autocast_dtype = torch.float16 if str(precision_cfg.get("dtype", "fp16")).lower() == "fp16" else torch.bfloat16
    scaler = amp.GradScaler(device=device.type, enabled=amp_enabled)

    # Sacred experiment setup
    if _run.observers:
        snapshot_dir = os.path.join(_run.observers[0].dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(_run.observers[0].dir, "logs"))

    # Batch size handling: The data structure is not designed for multi-batching at the
    # DataLoader level. We enforce a loader batch_size of 1 and use gradient accumulation
    # to achieve the desired effective batch size.
    loader_batch_size = 1
    effective_batch_size = _config["batch_size"] * _config["grad_accumulation_steps"]
    if _config["batch_size"] > 1:
        _log.warning(
            f"DataLoader 'batch_size' is forced to 1. Effective batch size of {effective_batch_size} will be achieved via gradient accumulation."
        )
        grad_accumulation_steps = effective_batch_size
    else:
        grad_accumulation_steps = _config["grad_accumulation_steps"]

    #
    # 2. Model, Optimizer, and Loss
    #
    _log.info("###### Initializing Model, Optimizer, and Loss ######")
    model = FewShotSeg(
        image_size=_config["input_size"][0], pretrained_path=_config["reload_model_path"] or None, cfg=_config["model"]
    ).to(device)
    model.train()

    if _config["optim_type"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **_config["optim"])
    elif _config["optim_type"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=_config["lr"], eps=1e-5)
    else:
        raise NotImplementedError(f"Optimizer '{_config['optim_type']}' not implemented.")

    scheduler = MultiStepLR(optimizer, milestones=_config["lr_milestones"], gamma=_config["lr_step_gamma"])
    criterion = nn.CrossEntropyLoss(
        ignore_index=_config["ignore_label"], weight=compose_wt_simple(_config["use_wce"], _config["dataset"])
    )

    #
    # 3. Data Loading
    #
    _log.info("###### Loading Data ######")
    collate_fn = lambda batch: batch[0]  # Unpacks the single-item list from the DataLoader
    train_transforms = get_train_transforms(_config)
    train_dataset = get_dataset(_config, train_transforms, "trainset")
    trainloader = DataLoader(
        train_dataset,
        batch_size=loader_batch_size,
        shuffle=True,
        num_workers=_config["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_cfg = _config["validation"]
    val_dataset = get_dataset(_config, None, "valset", val_cfg["val_manifest"])
    val_dataset.max_iters = val_cfg["n_val_episodes"]
    valloader = DataLoader(
        val_dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=_config["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    #
    # 4. Training Loop
    #
    _log.info("###### Starting Training ######")
    i_iter = 0
    log_loss = {"loss": 0, "align_loss": 0}
    best_val_miou = -1.0
    n_sub_epoches = max(1, _config["n_steps"] // _config["max_iters_per_load"], _config["epochs"])

    for sub_epoch in range(n_sub_epoches):
        _log.info(f"###### Starting epoch {sub_epoch + 1} / {n_sub_epoches} ######")

        for episode_batch in tqdm(trainloader, desc=f"Epoch {sub_epoch + 1}"):
            i_iter += 1
            episode_batch = process_episode_batch(episode_batch, device)
            support_fg_mask = [[shot["fg_mask"] for shot in way] for way in episode_batch["support_mask"]]
            support_bg_mask = [[shot["bg_mask"] for shot in way] for way in episode_batch["support_mask"]]

            with amp.autocast(device.type, dtype=autocast_dtype, enabled=amp_enabled):
                query_pred, align_loss, _ = model(
                    episode_batch["support_images"],
                    support_fg_mask,
                    support_bg_mask,
                    episode_batch["query_images"],
                    isval=False,
                    val_wsize=None,
                )
                query_loss = criterion(query_pred.float(), episode_batch["query_labels"])
                total_loss = query_loss + align_loss

            scaler.scale(total_loss / grad_accumulation_steps).backward()

            if i_iter % grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            scheduler.step()

            # Logging
            log_loss["loss"] += query_loss.item()
            log_loss["align_loss"] += align_loss.item()
            if i_iter % _config["print_interval"] == 0:
                avg_loss = log_loss["loss"] / _config["print_interval"]
                avg_align_loss = log_loss["align_loss"] / _config["print_interval"]
                _log.info(f"Step {i_iter}: Train Loss: {avg_loss:.4f}, Align Loss: {avg_align_loss:.4f}")
                writer.add_scalar("train/loss", avg_loss, i_iter)
                writer.add_scalar("train/align_loss", avg_align_loss, i_iter)
                log_loss = {"loss": 0, "align_loss": 0}

            # Validation and Snapshotting
            if i_iter % val_cfg["run_every_n_steps"] == 0:
                val_miou, val_loss = run_validation(model, valloader, device, criterion, _run)
                _log.info(f"Validation @ Step {i_iter}: mIoU: {val_miou:.4f}, Loss: {val_loss:.4f}")
                writer.add_scalar("val/mIoU", val_miou, i_iter)
                writer.add_scalar("val/loss", val_loss, i_iter)

                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    best_model_path = os.path.join(snapshot_dir, "best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                    _log.info(f"New best model saved with mIoU: {best_val_miou:.4f}")

            if i_iter % _config["save_snapshot_every"] == 0:
                snapshot_path = os.path.join(snapshot_dir, f"snapshot_{i_iter}.pth")
                torch.save(model.state_dict(), snapshot_path)
                _log.info(f"###### Saved periodic snapshot to {snapshot_path} ######")

            if i_iter >= _config["n_steps"]:
                break
        if i_iter >= _config["n_steps"]:
            break

    _log.info("###### Training finished ######")
