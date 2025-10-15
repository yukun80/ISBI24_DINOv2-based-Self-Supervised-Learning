"""
Few-shot dataset wrappers for the Exp_Disaster_Few-Shot landcover corpus.
The loaders generate PANet-compatible episodes with support/query tensors.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    """Lightweight container holding paired image/label metadata."""

    name: str
    image_path: str
    label_path: str
    classes: Tuple[int, ...]


def _read_rgb_tensor(path: str) -> np.ndarray:
    """Read a 3-channel GeoTIFF and return an H x W x 3 float image in [0, 1]."""
    with rasterio.open(path) as src:
        array = src.read(out_dtype=np.float32)
    # rasterio yields C x H x W; transpose to H x W x C
    array = np.transpose(array, (1, 2, 0))
    if array.max() > 1.0:
        array /= 255.0
    return array


def _read_label_mask(path: str) -> np.ndarray:
    """Read a single-channel GeoTIFF mask as H x W int array."""
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.int32)
    return mask


def _load_support_list(manifest_path: str) -> List[str]:
    """
    Load a deterministic support manifest.
    Supports either newline-delimited text or JSON list of basenames.
    """
    if not manifest_path:
        return []

    if manifest_path.endswith(".json"):
        with open(manifest_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict) and "support" in data:
            data = data["support"]
        if not isinstance(data, list):
            raise ValueError(f"Unexpected payload in {manifest_path}: {type(data)}")
        return [str(item).strip() for item in data if str(item).strip()]

    support: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as fp:
        for line in fp:
            name = line.strip()
            if name:
                support.append(name)
    return support


class ExpDisasterFewShotDataset(Dataset):
    """
    Episode generator for the disaster few-shot benchmark.

    Parameters
    ----------
    root_dir:
        Repository-relative path pointing at ``Exp_Disaster_Few-Shot``.
    split:
        Subdirectory to use (``trainset`` or ``valset``).
    target_classes:
        Iterable of label IDs considered foreground candidates.
    n_shots:
        Number of support images per episode.
    n_queries:
        Number of query images per episode (default 1).
    transforms:
        Callable returned by ``dataloaders.augutils.transform_with_label``.
    max_iters_per_epoch:
        Dataset length when sampled by a DataLoader (default 1000).
    fixed_support:
        Optional list of basenames to use as support samples every episode.
    support_manifest:
        Optional path to a manifest describing support basenames.
    episode_seed:
        RNG seed for reproducible sampling.
    episode_manifest:
        Optional JSON manifest specifying per-class support/query pools.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        target_classes: Iterable[int],
        n_shots: int,
        n_queries: int = 1,
        transforms=None,
        max_iters_per_epoch: int = 1000,
        fixed_support: Optional[Sequence[str]] = None,
        support_manifest: Optional[str] = None,
        episode_seed: int = 0,
        remap_to_one: bool = True,
        episode_manifest: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.target_classes = tuple(sorted(set(int(x) for x in target_classes if x != 0)))
        if not self.target_classes:
            raise ValueError("target_classes must contain at least one non-zero label ID.")

        self.n_shots = int(n_shots)
        self.n_queries = int(n_queries)
        self.transforms = transforms
        self.max_iters = int(max_iters_per_epoch)
        self.remap_to_one = remap_to_one

        support_from_file = _load_support_list(support_manifest) if support_manifest else []
        if fixed_support and support_from_file:
            raise ValueError("Provide either fixed_support or support_manifest, not both.")
        if episode_manifest and (fixed_support or support_from_file):
            raise ValueError("episode_manifest cannot be combined with fixed_support or support_manifest.")
        self.fixed_support = list(fixed_support or support_from_file)

        self.rng = random.Random(episode_seed)
        self.episode_manifest_path = episode_manifest
        self.episode_manifest: Optional[Dict[int, Dict[str, List[str]]]] = None
        self.manifest_class_ids: Tuple[int, ...] = ()

        base_dir = os.path.join(self.root_dir, split)
        self.image_dir = os.path.join(base_dir, "images")
        self.label_dir = os.path.join(base_dir, "labels")
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            raise FileNotFoundError(
                f"Expected 'images' and 'labels' folders under {base_dir}, "
                f"found {os.listdir(base_dir) if os.path.isdir(base_dir) else 'missing'}"
            )

        self.records: Dict[str, SampleRecord] = {}
        self.class_to_records: Dict[int, List[SampleRecord]] = {
            cls_id: [] for cls_id in self.target_classes
        }
        self._build_index()

        if self.fixed_support:
            missing = [name for name in self.fixed_support if name not in self.records]
            if missing:
                raise ValueError(f"Support manifest references unknown samples: {missing}")
            # Filter supports to only those containing desired classes.
            filtered_support = []
            for name in self.fixed_support:
                record = self.records[name]
                if set(record.classes) & set(self.target_classes):
                    filtered_support.append(name)
            self.fixed_support = filtered_support

        if not any(self.class_to_records.values()):
            raise RuntimeError(
                f"No samples found for the requested classes {self.target_classes} "
                f"in split '{split}'."
            )

        if self.episode_manifest_path:
            self._load_episode_manifest(self.episode_manifest_path)

    def _build_index(self) -> None:
        for label_name in sorted(os.listdir(self.label_dir)):
            if not label_name.lower().endswith(".tif"):
                continue
            stem = os.path.splitext(label_name)[0]
            image_path = os.path.join(self.image_dir, f"{stem}.tif")
            label_path = os.path.join(self.label_dir, label_name)
            if not os.path.isfile(image_path):
                continue

            mask = _read_label_mask(label_path)
            classes = tuple(sorted({int(x) for x in np.unique(mask)} - {0}))
            record = SampleRecord(
                name=stem, image_path=image_path, label_path=label_path, classes=classes
            )
            self.records[stem] = record
            for cls_id in self.target_classes:
                if cls_id in classes:
                    self.class_to_records.setdefault(cls_id, []).append(record)

    def _sample_records(
        self, cls_id: int, k: int, exclude: Optional[Sequence[str]] = None
    ) -> List[SampleRecord]:
        pool = self.class_to_records.get(cls_id, [])
        if not pool:
            raise RuntimeError(f"No samples contain class {cls_id} in split '{self.split}'.")

        exclude_names = set(exclude or [])
        available = [rec for rec in pool if rec.name not in exclude_names]
        if len(available) >= k:
            return self.rng.sample(available, k=k)

        # Fall back to sampling with replacement when not enough unique samples exist.
        sampled = []
        for _ in range(k):
            sampled.append(self.rng.choice(pool))
        return sampled

    def _load_episode_manifest(self, manifest_path: str) -> None:
        with open(manifest_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        classes_section = payload.get("classes")
        if not isinstance(classes_section, dict):
            raise ValueError(f"Malformed episode manifest: expected 'classes' mapping in {manifest_path}.")

        parsed: Dict[int, Dict[str, List[str]]] = {}
        missing_by_class: Dict[int, int] = {}

        def extract_names(entries: Iterable) -> List[str]:
            names: List[str] = []
            for entry in entries or []:
                if isinstance(entry, str):
                    candidate = entry.strip()
                elif isinstance(entry, dict):
                    candidate = str(entry.get("name", "")).strip()
                else:
                    candidate = ""
                if candidate:
                    names.append(candidate)
            deduped: List[str] = []
            seen = set()
            for name in names:
                if name not in seen:
                    seen.add(name)
                    deduped.append(name)
            return deduped

        for cls_key, cls_payload in classes_section.items():
            try:
                cls_id = int(cls_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid class key '{cls_key}' in {manifest_path}") from exc

            if cls_id not in self.target_classes or not isinstance(cls_payload, dict):
                continue

            support_names = extract_names(cls_payload.get("support_pool", []))
            query_names = extract_names(cls_payload.get("query_pool", []))
            support_filtered: List[str] = []
            query_filtered: List[str] = []

            for name in support_names:
                record = self.records.get(name)
                if record and cls_id in record.classes:
                    support_filtered.append(name)
                else:
                    missing_by_class[cls_id] = missing_by_class.get(cls_id, 0) + 1

            for name in query_names:
                record = self.records.get(name)
                if record and cls_id in record.classes:
                    query_filtered.append(name)
                else:
                    missing_by_class[cls_id] = missing_by_class.get(cls_id, 0) + 1

            if support_filtered or query_filtered:
                parsed[cls_id] = {"support": support_filtered, "query": query_filtered}

        if not parsed:
            raise RuntimeError(f"No valid entries found in episode manifest {manifest_path}.")

        if missing_by_class:
            print(
                "Episode manifest references samples that are missing or incompatible: "
                + ", ".join(
                    f"class {cls_id}: {count} entries"
                    for cls_id, count in sorted(missing_by_class.items())
                )
            )

        self.episode_manifest = parsed
        self.manifest_class_ids = tuple(sorted(parsed.keys()))

    def _sample_manifest_names(
        self,
        cls_id: int,
        pool_key: str,
        k: int,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if not self.episode_manifest or cls_id not in self.episode_manifest:
            return []
        pool = self.episode_manifest[cls_id].get(pool_key, [])
        if not pool:
            return []
        exclude_names = set(exclude or [])
        candidates = [name for name in pool if name not in exclude_names]
        if len(candidates) <= k:
            return list(candidates)
        return self.rng.sample(candidates, k=k)

    def _sample_support_from_manifest(self, cls_id: int) -> List[SampleRecord]:
        names = self._sample_manifest_names(cls_id, "support", self.n_shots)
        records = [self.records[name] for name in names if name in self.records]
        if len(records) < self.n_shots:
            exclude = [record.name for record in records]
            fallback = self._sample_records(cls_id, self.n_shots - len(records), exclude=exclude)
            records.extend(fallback)
        return records

    def _sample_query_from_manifest(
        self, cls_id: int, exclude: Optional[Sequence[str]] = None
    ) -> List[SampleRecord]:
        names = self._sample_manifest_names(cls_id, "query", self.n_queries, exclude=exclude)
        records = [self.records[name] for name in names if name in self.records]
        if len(records) < self.n_queries:
            exclude_names = set(exclude or [])
            exclude_names.update(record.name for record in records)
            fallback = self._sample_records(
                cls_id, self.n_queries - len(records), exclude=list(exclude_names)
            )
            records.extend(fallback)
        return records

    def _prepare_tensor(
        self, record: SampleRecord, cls_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = _read_rgb_tensor(record.image_path)
        mask = _read_label_mask(record.label_path)
        fg_mask = (mask == cls_id).astype(np.float32)
        if fg_mask.sum() < 1:
            # Avoid empty support/query by broadening condition slightly.
            raise ValueError(f"Sample '{record.name}' contains no pixels for class {cls_id}.")

        comp = np.concatenate([image, fg_mask[..., None]], axis=-1)
        if self.transforms is not None:
            img_np, mask_np = self.transforms(
                comp,
                c_img=image.shape[-1],
                c_label=1,
                nclass=2,
                use_onehot=False,
            )
        else:
            img_np = image
            mask_np = fg_mask[..., None]

        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask_np.squeeze(-1)).float()

        if self.remap_to_one:
            mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor

    def _encode_support(
        self, support_records: Sequence[SampleRecord], cls_id: int
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        support_images: List[torch.Tensor] = []
        support_masks: List[Dict[str, torch.Tensor]] = []
        for record in support_records:
            image, mask = self._prepare_tensor(record, cls_id)
            fg_mask = mask.clone()
            bg_mask = torch.ones_like(mask) - fg_mask
            support_images.append(image)
            support_masks.append({"fg_mask": fg_mask, "bg_mask": bg_mask})
        return support_images, support_masks

    def _encode_queries(
        self, query_records: Sequence[SampleRecord], cls_id: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        query_images: List[torch.Tensor] = []
        query_labels: List[torch.Tensor] = []
        query_names: List[str] = []
        for record in query_records:
            image, mask = self._prepare_tensor(record, cls_id)
            query_images.append(image)
            query_labels.append(mask.unsqueeze(0).long())
            query_names.append(record.name)
        return query_images, query_labels, query_names

    def __len__(self) -> int:
        if self.fixed_support and self.split.lower().startswith("val"):
            return len(
                [
                    rec
                    for rec in self.records.values()
                    if rec.name not in set(self.fixed_support)
                    and set(rec.classes) & set(self.target_classes)
                ]
            )
        return self.max_iters

    def __getitem__(self, index: int):  # pylint: disable=unused-argument
        if self.episode_manifest:
            cls_id = self.rng.choice(self.manifest_class_ids)
        elif len(self.target_classes) == 1:
            cls_id = self.target_classes[0]
        else:
            cls_id = self.rng.choice(self.target_classes)

        if self.fixed_support:
            support_records = [self.records[name] for name in self.fixed_support]
        elif self.episode_manifest:
            support_records = self._sample_support_from_manifest(cls_id)
        else:
            support_records = self._sample_records(cls_id, self.n_shots)

        exclude_names = [record.name for record in support_records]
        if self.fixed_support:
            # During validation iterate deterministically over remaining samples.
            candidate_records = [
                rec
                for rec in self.class_to_records.get(cls_id, [])
                if rec.name not in exclude_names
            ]
            if not candidate_records:
                raise RuntimeError(
                    f"No query candidates left for class {cls_id} after excluding support."
                )
            query_records = [candidate_records[index % len(candidate_records)]]
        elif self.episode_manifest:
            query_records = self._sample_query_from_manifest(cls_id, exclude_names)
        else:
            query_records = self._sample_records(cls_id, self.n_queries, exclude=exclude_names)

        support_images, support_masks = self._encode_support(support_records, cls_id)
        query_images, query_labels, query_names = self._encode_queries(query_records, cls_id)

        return {
            "class_ids": [[cls_id for _ in support_images]],
            "support_images": [support_images],
            "support_mask": [support_masks],
            "query_images": query_images,
            "query_labels": query_labels,
            "query_names": query_names,
        }
