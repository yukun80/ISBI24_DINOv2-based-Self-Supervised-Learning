"""
Training the model
Extended from original implementation of ALPNet.
"""

from scipy.ndimage import distance_transform_edt as eucl_distance
from contextlib import nullcontext
import os
import shutil
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

from config_ssl_upload import ex
from tqdm.auto import tqdm

# import Tensor
from torch import Tensor
from typing import List, Tuple, Union, cast, Iterable, Set, Any, Callable, TypeVar

"""
python3 training.py with \
    modelname=`dinov2_b14 \
    dataset=EXP_DISASTER_FEWSHOT \
    num_workers=8 \
    use_wce=True \
    clsname=grid_proto \
    input_size="(512, 512)" \
    proto_grid_size=16 \
    task.n_shots=5 \
    task.n_queries=1 \
    batch_size=8 \
    grad_accumulation_steps=1 \
    n_steps=80000 \
    epochs=1 \
    max_iters_per_load=2000 \
    lr=5e-5 \
    optim.lr=5e-5 \
    lr_milestones="[20000, 32000]" \
    lr_step_gamma=0.5 \
    save_snapshot_every=5000 \
    lora=16 \
    ttt=True \
    precision.use_amp=False \
    precision.dtype=fp32 \
    precision.grad_scaler=True \
    seed=3407 \
    episode_manifest=./data/trainset/manifest.json
    
python3 training.py with \
    modelname=dinov2_s14 \
    dataset=EXP_DISASTER_FEWSHOT \
    num_workers=8 \
    use_wce=True \
    clsname=grid_proto \
    input_size="(512, 512)" \
    proto_grid_size=16 \
    task.n_shots=5 \
    task.n_queries=1 \
    batch_size=1 \
    grad_accumulation_steps=8 \
    n_steps=80000 \
    epochs=1 \
    max_iters_per_load=2000 \
    lr=5e-4 \
    optim.lr=5e-4 \
    lr_milestones="[20000, 32000]" \
    lr_step_gamma=0.5 \
    save_snapshot_every=5000 \
    lora=4 \
    ttt=True \
    precision.use_amp=True \
    precision.dtype=fp16 \
    precision.grad_scaler=True \
    seed=3407 \
    episode_manifest=./data/trainset/manifest.json
"""


def get_dice_loss(prediction: torch.Tensor, target: torch.Tensor, smooth=1.0):
    """
    prediction: (B, 1, H, W)
    target: (B, H, W)
    """
    if prediction.shape[1] > 1:
        # use only the foreground prediction
        prediction = prediction[:, 1, :, :]
    prediction = torch.sigmoid(prediction)
    intersection = (prediction * target).sum(dim=(-2, -1))
    union = prediction.sum(dim=(-2, -1)) + target.sum(dim=(1, 2)) + smooth

    dice = (2.0 * intersection + smooth) / union
    dice_loss = 1.0 - dice.mean()

    return dice_loss


def get_train_transforms(_config):
    tr_transforms = myaug.transform_with_label({"aug": myaug.get_aug(_config["which_aug"], _config["input_size"][0])})
    return tr_transforms


def get_dataset(_config):
    transforms = get_train_transforms(_config)
    dataset_root = _config["path"][_config["dataset"]]["data_dir"]
    return ExpDisasterFewShotDataset(
        root_dir=dataset_root,
        split="trainset",
        target_classes=range(1, 9),
        n_shots=_config["task"]["n_shots"],
        n_queries=_config["task"]["n_queries"],
        transforms=transforms,
        max_iters_per_epoch=_config["max_iters_per_load"],
        episode_seed=_config["seed"],
        episode_manifest=_config.get("episode_manifest"),
    )


@ex.automain
def main(_run, _config, _log):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    precision_cfg = _config.get("precision", {})
    requested_amp = bool(precision_cfg.get("use_amp", False))
    amp_available = device.type == "cuda"
    amp_enabled = requested_amp and amp_available
    if requested_amp and not amp_available:
        _log.info("AMP requested but CUDA is unavailable; falling back to float32.")
    autocast_dtype_str = str(precision_cfg.get("dtype", "fp16")).lower()
    autocast_dtype = torch.float16 if autocast_dtype_str == "fp16" else torch.bfloat16
    scaler_enabled = bool(precision_cfg.get("grad_scaler", True)) and amp_enabled
    scaler = amp.GradScaler(device=device_type, enabled=scaler_enabled) if amp_enabled else None
    tensor_dtype = torch.float32
    if _run.observers:
        os.makedirs(f"{_run.observers[0].dir}/snapshots", exist_ok=True)
        for source_file, _ in _run.experiment_info["sources"]:
            os.makedirs(os.path.dirname(f"{_run.observers[0].dir}/source/{source_file}"), exist_ok=True)
            _run.observers[0].save_file(source_file, f"source/{source_file}")
        shutil.rmtree(f"{_run.observers[0].basedir}/_sources")

    set_seed(_config["seed"])

    writer = SummaryWriter(f"{_run.observers[0].dir}/logs")
    _log.info("###### Create model ######")
    if _config["reload_model_path"] != "":
        _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    else:
        _config["reload_model_path"] = None
    model = FewShotSeg(
        image_size=_config["input_size"][0], pretrained_path=_config["reload_model_path"], cfg=_config["model"]
    )

    model = model.to(device)
    model.train()
    if amp_enabled:
        _log.info(f"###### AMP enabled with dtype {autocast_dtype_str} ######")

    _log.info("###### Load data ######")
    data_name = _config["dataset"]
    tr_parent = get_dataset(_config)

    # dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config["batch_size"],
        shuffle=True,
        num_workers=_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    _log.info("###### Set optimizer ######")
    if _config["optim_type"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **_config["optim"])
    elif _config["optim_type"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=_config["lr"], eps=1e-5)
    else:
        raise NotImplementedError

    scheduler = MultiStepLR(optimizer, milestones=_config["lr_milestones"], gamma=_config["lr_step_gamma"])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config["ignore_label"], weight=my_weight)

    i_iter = 0  # total number of iteration
    # number of times for reloading
    n_sub_epoches = max(1, _config["n_steps"] // _config["max_iters_per_load"], _config["epochs"])
    log_loss = {"loss": 0, "align_loss": 0}

    _log.info("###### Training ######")
    epoch_losses = []
    for sub_epoch in range(n_sub_epoches):
        _log.info(f"###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######")
        pbar = tqdm(trainloader)
        optimizer.zero_grad()
        for idx, sample_batched in enumerate(tqdm(trainloader)):
            losses = []
            i_iter += 1
            support_images = [
                [shot.to(device=device, dtype=tensor_dtype) for shot in way] for way in sample_batched["support_images"]
            ]
            support_fg_mask = [
                [shot[f"fg_mask"].float().to(device=device, dtype=tensor_dtype) for shot in way]
                for way in sample_batched["support_mask"]
            ]
            support_bg_mask = [
                [shot[f"bg_mask"].float().to(device=device, dtype=tensor_dtype) for shot in way]
                for way in sample_batched["support_mask"]
            ]

            query_images = [
                query_image.to(device=device, dtype=tensor_dtype) for query_image in sample_batched["query_images"]
            ]
            query_labels = torch.cat(
                [query_label.long().to(device) for query_label in sample_batched["query_labels"]], dim=0
            )
            if query_labels.ndim == 4 and query_labels.shape[1] == 1:
                query_labels = query_labels.squeeze(1)

            loss = 0.0
            try:
                with amp.autocast(device_type, dtype=autocast_dtype) if amp_enabled else nullcontext():
                    query_pred, align_loss, _ = model(
                        support_images,
                        support_fg_mask,
                        support_bg_mask,
                        query_images,
                        isval=False,
                        val_wsize=None,
                    )
                    query_loss = criterion(query_pred.float(), query_labels.long())
                    loss = query_loss + align_loss
            except Exception as e:
                print(f"faulty batch detected, skip: {e}")
                # offload cuda memory
                del support_images, support_fg_mask, support_bg_mask, query_images, query_labels
                continue
            pbar.set_postfix({"loss": loss.item()})
            loss_to_backward = loss / _config["grad_accumulation_steps"]
            if amp_enabled:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()
            if (idx + 1) % _config["grad_accumulation_steps"] == 0:
                step_executed = True
                if amp_enabled:
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_executed = scaler.get_scale() >= prev_scale
                else:
                    optimizer.step()
                if step_executed:
                    scheduler.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            query_loss_value = query_loss.detach().cpu().item()
            align_loss_value = align_loss.detach().cpu().item() if torch.is_tensor(align_loss) else float(align_loss)

            _run.log_scalar("loss", query_loss_value)
            _run.log_scalar("align_loss", align_loss_value)

            log_loss["loss"] += query_loss_value
            log_loss["align_loss"] += align_loss_value

            # print loss and take snapshots
            if (i_iter + 1) % _config["print_interval"] == 0:
                writer.add_scalar("loss", loss, i_iter)
                writer.add_scalar("query_loss", query_loss_value, i_iter)
                writer.add_scalar("align_loss", align_loss_value, i_iter)

                loss = log_loss["loss"] / _config["print_interval"]
                align_loss = log_loss["align_loss"] / _config["print_interval"]

                log_loss["loss"] = 0
                log_loss["align_loss"] = 0

                print(f"step {i_iter+1}: loss: {loss}, align_loss: {align_loss},")

            if (i_iter + 1) % _config["save_snapshot_every"] == 0:
                _log.info("###### Taking snapshot ######")
                torch.save(model.state_dict(), os.path.join(f"{_run.observers[0].dir}/snapshots", f"{i_iter + 1}.pth"))

            if (i_iter - 1) >= _config["n_steps"]:
                break  # finish up
        epoch_losses.append(np.mean(losses))
        print(f"Epoch {sub_epoch} loss: {np.mean(losses)}")
