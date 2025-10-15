"""
Validation script tailored for the Exp_Disaster_Few-Shot dataset.
"""
import os
import shutil
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config_ssl_upload import ex
from dataloaders.exp_disaster_dataset import ExpDisasterFewShotDataset
from models.grid_proto_fewshot import FewShotSeg
from util.metric import Metric
from util.utils import set_seed


def _prepare_observer_artifacts(_run) -> None:
    if not _run.observers:
        return
    os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
    for source_file, _ in _run.experiment_info['sources']:
        abs_path = f'{_run.observers[0].dir}/source/{source_file}'
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


def _default_support_subset(dataset: ExpDisasterFewShotDataset, n_shots: int) -> List[str]:
    selected: List[str] = []
    for record in dataset.records.values():
        if record.name not in selected and set(record.classes) & set(dataset.target_classes):
            selected.append(record.name)
        if len(selected) >= n_shots:
            break
    return selected[:n_shots]


@ex.automain
def main(_run, _config, _log):
    precision = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(False)

    _prepare_observer_artifacts(_run)
    set_seed(_config['seed'])

    _log.info('###### Create model ######')
    pretrained_path = _config['reload_model_path'] or None
    model = FewShotSeg(
        image_size=_config['input_size'][0],
        pretrained_path=pretrained_path,
        cfg=_config['model'],
    )
    model = model.to(device, precision).eval()

    dataset_root = _config['path'][_config['dataset']]['data_dir']
    support_manifest = _config.get('support_txt_file')
    dataset = ExpDisasterFewShotDataset(
        root_dir=dataset_root,
        split='valset',
        target_classes=[20],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries'],
        transforms=None,
        max_iters_per_epoch=1,
        support_manifest=support_manifest,
        episode_seed=_config['seed'],
    )

    if not dataset.fixed_support:
        dataset.fixed_support = _default_support_subset(dataset, _config['task']['n_shots'])
        _log.info(f"Using default support set: {dataset.fixed_support}")
    else:
        _log.info(f"Loaded support set: {dataset.fixed_support}")

    if len(dataset) == 0:
        raise RuntimeError("No validation queries available after excluding fixed support samples.")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: batch[0],
    )

    metric = Metric(max_label=1, n_scans=len(dataset))
    metric.reset()

    output_dir = os.path.join(_run.observers[0].dir, "disaster_preds") if _run.observers else None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    _log.info('###### Starting validation ######')
    with torch.no_grad():
        for episode in tqdm(dataloader, total=len(dataset)):
            support_images = [[shot.to(device, precision) for shot in way]
                              for way in episode['support_images']]
            support_fg_mask = [
                [mask['fg_mask'].float().to(device, precision) for mask in way]
                for way in episode['support_mask']
            ]
            support_bg_mask = [
                [mask['bg_mask'].float().to(device, precision) for mask in way]
                for way in episode['support_mask']
            ]
            query_images = [img.to(device, precision) for img in episode['query_images']]
            query_labels = torch.cat([label.long().to(device) for label in episode['query_labels']], dim=0)

            logits, *_ = model(
                support_images,
                support_fg_mask,
                support_bg_mask,
                query_images,
                isval=True,
                val_wsize=None,
            )
            pred_mask = logits.argmax(dim=1).cpu().numpy()[0]
            gt_mask = query_labels.cpu().numpy()[0]

            metric.record(pred_mask, gt_mask, labels=[1])

            if output_dir:
                for name, pred in zip(episode['query_names'], [pred_mask]):
                    np.save(os.path.join(output_dir, f"{name}_pred.npy"), pred.astype(np.uint8))

    dice_mean, dice_std, dice_global_mean, dice_global_std = metric.get_mDice(labels=[1])
    iou_mean, iou_std, iou_global_mean, iou_global_std = metric.get_mIoU(labels=[1])

    _run.log_scalar('dice_fg_mean', dice_global_mean)
    _run.log_scalar('dice_fg_std', dice_global_std)
    _run.log_scalar('iou_fg_mean', iou_global_mean)
    _run.log_scalar('iou_fg_std', iou_global_std)

    _log.info(
        f"Validation Dice (fg) mean={dice_global_mean:.4f}±{dice_global_std:.4f}; "
        f"class-wise={dice_mean}"
    )
    _log.info(
        f"Validation IoU (fg) mean={iou_global_mean:.4f}±{iou_global_std:.4f}; "
        f"class-wise={iou_mean}"
    )
