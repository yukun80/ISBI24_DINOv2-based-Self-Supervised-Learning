"""
Few-shot dataset wrappers for the Exp_Disaster_Few-Shot landcover corpus.

This module provides a flexible dataset class that supports two modes:
1.  **Training Mode (default):** Dynamically samples episodes from the entire
    dataset, using a cached quality check to ensure training stability.
2.  **Evaluation Mode:** When an `evaluation_manifest` is provided, it loads a
    pre-defined set of episodes, enabling deterministic and reproducible
    evaluation, typically on a validation or test set.
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
    """
    Lightweight container holding essential metadata for a single sample.
    Includes pre-computed pixel counts for efficient quality checks.
    """

    name: str
    image_path: str
    label_path: str
    classes: Tuple[int, ...]
    fg_pixel_counts: Dict[int, int]


@dataclass(frozen=True)
class EvaluationEpisode:
    """Defines a single, deterministic evaluation episode."""

    region: str
    class_id: int
    support_names: Tuple[str, ...]
    query_name: str


def _read_rgb_tensor(path: str) -> np.ndarray:
    """Read a 3-channel GeoTIFF and return an H x W x 3 float image in [0, 1]."""
    with rasterio.open(path) as src:
        array = src.read(out_dtype=np.float32)
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
    Load a deterministic support manifest (newline-delimited text).
    This is retained for validation purposes where a fixed support set is desired.
    """
    if not manifest_path:
        return []
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

    Supports both dynamic sampling for training and manifest-based deterministic
    evaluation.

    Parameters
    ----------
    root_dir, split, target_classes, n_shots, n_queries, transforms:
        Core dataset and episode configuration.
    max_iters_per_epoch:
        Dataset length for dynamic training mode.
    evaluation_manifest:
        Path to a JSON manifest defining specific episodes for evaluation.
        If provided, the dataset enters a deterministic "evaluation mode".
    fixed_support, support_manifest:
        Legacy options for providing a fixed support set. Incompatible with
        `evaluation_manifest`.
    min_support_pixels, min_query_pixels:
        Quality thresholds for dynamic sampling during training.
    episode_seed:
        RNG seed for reproducible sampling.
    remap_to_one:
        If True, remaps the multi-class mask to a binary {0, 1} mask.
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
        evaluation_manifest: Optional[str] = None,
        fixed_support: Optional[Sequence[str]] = None,
        support_manifest: Optional[str] = None,
        min_support_pixels: int = 10,
        min_query_pixels: int = 1,
        episode_seed: int = 0,
        remap_to_one: bool = True,
        region_based_sampling: bool = False,
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
        self.min_support_pixels = int(min_support_pixels)
        self.min_query_pixels = int(min_query_pixels)
        self.region_based_sampling = region_based_sampling and self.split == 'trainset'

        if evaluation_manifest and (fixed_support or support_manifest):
            raise ValueError(
                "'evaluation_manifest' cannot be used with 'fixed_support' or 'support_manifest'."
            )

        support_from_file = _load_support_list(support_manifest) if support_manifest else []
        self.fixed_support = list(fixed_support or support_from_file)

        self.rng = random.Random(episode_seed)
        self.evaluation_episodes: Optional[List[EvaluationEpisode]] = None

        base_dir = os.path.join(self.root_dir, split)
        self.image_dir = os.path.join(base_dir, "images")
        self.label_dir = os.path.join(base_dir, "labels")
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Expected 'images' and 'labels' folders under {base_dir}.")

        self.records: Dict[str, SampleRecord] = {}
        self.class_to_records: Dict[int, List[SampleRecord]] = {
            cls_id: [] for cls_id in self.target_classes
        }
        self.region_and_class_to_records: Dict[str, Dict[int, List[SampleRecord]]] = {}
        self.valid_episode_sources: List[Tuple[str, int]] = []
        self._build_index()

        if evaluation_manifest:
            self._load_evaluation_manifest(evaluation_manifest)
        elif self.fixed_support:
            self._validate_fixed_support()

        if not self.evaluation_episodes and not any(self.class_to_records.values()):
            raise RuntimeError(
                f"No samples found for the requested classes {self.target_classes} in split '{self.split}'."
            )

    def _build_index(self) -> None:
        """
        Builds an index of all samples in the dataset.

        This method populates:
        - `self.records`: A dictionary mapping sample names to SampleRecord objects.
        - `self.class_to_records`: A dictionary mapping class IDs to lists of records.
        - `self.region_and_class_to_records`: A nested dictionary for region-based sampling,
          mapping region names to class IDs to lists of records.
        - `self.valid_episode_sources`: A list of (region, class_id) tuples that have
          enough samples to form a region-based training episode.
        """
        self.region_and_class_to_records = {}

        for label_name in sorted(os.listdir(self.label_dir)):
            if not label_name.lower().endswith((".tif", ".tiff")):
                continue
            stem = os.path.splitext(label_name)[0]
            image_path = os.path.join(self.image_dir, f"{stem}.tif")
            label_path = os.path.join(self.label_dir, label_name)
            if not os.path.isfile(image_path):
                continue

            mask = _read_label_mask(label_path)
            pixel_counts: Dict[int, int] = {c: int(np.sum(mask == c)) for c in self.target_classes if np.sum(mask == c) > 0}
            if not pixel_counts:
                continue

            record = SampleRecord(
                name=stem, image_path=image_path, label_path=label_path,
                classes=tuple(pixel_counts.keys()), fg_pixel_counts=pixel_counts,
            )
            self.records[stem] = record
            
            region = stem.rsplit('_', 1)[0] if '_' in stem else stem

            for cls_id in record.classes:
                self.class_to_records.setdefault(cls_id, []).append(record)
                self.region_and_class_to_records.setdefault(region, {}).setdefault(cls_id, []).append(record)

        if self.region_based_sampling:
            self.valid_episode_sources = []
            for region, class_map in self.region_and_class_to_records.items():
                for cls_id, records in class_map.items():
                    if len(records) >= self.n_shots + self.n_queries:
                        self.valid_episode_sources.append((region, cls_id))
            
            if not self.valid_episode_sources:
                print(
                    "Warning: Could not find any region with enough samples for region-based sampling. "
                    "Falling back to global sampling for the entire run."
                )
                self.region_based_sampling = False

    def _load_evaluation_manifest(self, manifest_path: str) -> None:
        """Parses a region-based evaluation manifest into a flat list of episodes."""
        with open(manifest_path, "r", encoding="utf-8") as fp:
            manifest = json.load(fp)

        self.evaluation_episodes = []
        # Assuming valset has a single target class
        class_id = self.target_classes[0]

        for region, splits in manifest.get("regions", {}).items():
            support_names = tuple(splits.get("support", []))
            query_names = splits.get("query", [])

            if not support_names or not query_names:
                continue

            # Verify all names exist in the index
            all_names = list(support_names) + query_names
            missing = [name for name in all_names if name not in self.records]
            if missing:
                print(f"Warning: Region '{region}' references missing samples, skipping: {missing}")
                continue

            for query_name in query_names:
                episode = EvaluationEpisode(
                    region=region, class_id=class_id, 
                    support_names=support_names, query_name=query_name
                )
                self.evaluation_episodes.append(episode)
        
        if not self.evaluation_episodes:
            raise ValueError(f"Evaluation manifest '{manifest_path}' yielded no valid episodes.")

    def _validate_fixed_support(self) -> None:
        missing = [name for name in self.fixed_support if name not in self.records]
        if missing:
            raise ValueError(f"Support manifest references unknown samples: {missing}")
        self.fixed_support = [name for name in self.fixed_support if set(self.records[name].classes) & set(self.target_classes)]

    def _sample_records_with_check(
        self,
        cls_id: int,
        k: int,
        min_pixels: int,
        exclude: Optional[Sequence[str]] = None,
        record_pool: Optional[List[SampleRecord]] = None,
    ) -> List[SampleRecord]:
        """
        Samples records for a given class, with quality checks.

        Args:
            cls_id: The class ID to sample for.
            k: The number of records to sample.
            min_pixels: The minimum number of foreground pixels required.
            exclude: A list of record names to exclude from sampling.
            record_pool: If provided, sample from this pool instead of the global
                         class-to-records mapping.

        Returns:
            A list of sampled SampleRecord objects.
        """
        pool = record_pool if record_pool is not None else self.class_to_records.get(cls_id, [])
        if not pool:
            # This case should ideally not be hit if called from _get_region_based_training_item,
            # as the source is pre-validated. It's a safeguard.
            raise RuntimeError(f"No samples contain class {cls_id} in the provided record pool or split '{self.split}'.")

        exclude_names = set(exclude or [])
        candidates = [
            rec for rec in pool 
            if rec.name not in exclude_names and rec.fg_pixel_counts.get(cls_id, 0) >= min_pixels
        ]

        if not candidates:
            # Fallback: if no candidates meet the pixel threshold, sample from what's available,
            # ignoring the pixel count. This is the original behavior.
            available = [rec for rec in pool if rec.name not in exclude_names] or pool
            return self.rng.choices(available, k=k)

        # If there are enough candidates, use sample for no replacement. Otherwise, use choices.
        return self.rng.sample(candidates, k=k) if len(candidates) >= k else self.rng.choices(candidates, k=k)

    def _prepare_tensor(self, record: SampleRecord, cls_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = _read_rgb_tensor(record.image_path)
        mask = _read_label_mask(record.label_path)
        fg_mask = (mask == cls_id).astype(np.float32)
        comp = np.concatenate([image, fg_mask[..., None]], axis=-1)
        
        if self.transforms:
            img_np, mask_np = self.transforms(comp, c_img=image.shape[-1], c_label=1, nclass=2, use_onehot=False)
        else:
            img_np, mask_np = image, fg_mask[..., None]

        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask_np.squeeze(-1)).float()
        return (img_tensor, (mask_tensor > 0.5).float()) if self.remap_to_one else (img_tensor, mask_tensor)

    def _encode_support(self, support_records: Sequence[SampleRecord], cls_id: int) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        support_images, support_masks = [], []
        for record in support_records:
            image, mask = self._prepare_tensor(record, cls_id)
            support_images.append(image)
            support_masks.append({"fg_mask": mask.clone(), "bg_mask": 1 - mask})
        return support_images, support_masks

    def _encode_queries(self, query_records: Sequence[SampleRecord], cls_id: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        query_images, query_labels, query_names = [], [], []
        for record in query_records:
            image, mask = self._prepare_tensor(record, cls_id)
            query_images.append(image)
            query_labels.append(mask.unsqueeze(0).long())
            query_names.append(record.name)
        return query_images, query_labels, query_names

    def __len__(self) -> int:
        if self.evaluation_episodes is not None:
            return len(self.evaluation_episodes)
        return self.max_iters

    def __getitem__(self, index: int):
        """
        Returns a single training/evaluation episode.

        The behavior of this method depends on the dataset's mode:
        - Evaluation Mode: Returns a deterministic episode from a manifest.
        - Training Mode: Returns a dynamically sampled episode. If `region_based_sampling`
          is enabled, it attempts to sample from within a single region.
        """
        if self.evaluation_episodes is not None:
            return self._get_evaluation_item(index)
        
        if self.region_based_sampling:
            return self._get_region_based_training_item(index)
        
        return self._get_global_training_item(index)

    def _get_evaluation_item(self, index: int):
        """Gets a pre-defined episode for deterministic evaluation."""
        episode_def = self.evaluation_episodes[index]
        cls_id = episode_def.class_id

        support_records = [self.records[name] for name in episode_def.support_names]
        query_records = [self.records[episode_def.query_name]]

        support_images, support_masks = self._encode_support(support_records, cls_id)
        query_images, query_labels, query_names = self._encode_queries(query_records, cls_id)

        return {
            "class_ids": [[cls_id]],
            "support_images": [support_images],
            "support_mask": [support_masks],
            "support_names": [[record.name for record in support_records]],
            "query_images": query_images,
            "query_labels": query_labels,
            "query_names": query_names,
            "region": episode_def.region,
        }

    def _get_region_based_training_item(self, index: int):
        """
        Generates a dynamic training episode by sampling from within a single region.

        If a valid episode cannot be formed (e.g., due to quality checks failing),
        this method gracefully falls back to the global sampling strategy for the
        current item.
        """
        # 1. Select a pre-validated (region, class_id) source.
        region, cls_id = self.rng.choice(self.valid_episode_sources)
        
        # 2. Get the pool of records for this source.
        regional_pool = self.region_and_class_to_records[region][cls_id]
        
        # 3. Sample support set from the regional pool with quality checks.
        support_records = self._sample_records_with_check(
            cls_id, self.n_shots, self.min_support_pixels, record_pool=regional_pool
        )
        
        # If not enough valid support records were found, fallback to global sampling.
        if len(support_records) < self.n_shots:
            return self._get_global_training_item(index)

        exclude_names = [record.name for record in support_records]
        
        # 4. Sample query set from the same regional pool.
        query_records = self._sample_records_with_check(
            cls_id, self.n_queries, self.min_query_pixels, exclude=exclude_names, record_pool=regional_pool
        )

        # If not enough valid query records were found, also fallback.
        if len(query_records) < self.n_queries:
            return self._get_global_training_item(index)

        # 5. Encode and return the episode.
        support_images, support_masks = self._encode_support(support_records, cls_id)
        query_images, query_labels, query_names = self._encode_queries(query_records, cls_id)

        return {
            "class_ids": [[cls_id for _ in support_records]],
            "support_images": [support_images],
            "support_mask": [support_masks],
            "support_names": [[record.name for record in support_records]],
            "query_images": query_images,
            "query_labels": query_labels,
            "query_names": query_names,
            "region": region,  # Add region for potential debugging
        }

    def _get_global_training_item(self, index: int):
        """Generates a dynamic episode by sampling from the entire training set (globally)."""
        cls_id = self.rng.choice(self.target_classes) if len(self.target_classes) > 1 else self.target_classes[0]

        if self.fixed_support:
            support_records = [self.records[name] for name in self.fixed_support]
        else:
            support_records = self._sample_records_with_check(cls_id, self.n_shots, self.min_support_pixels)

        exclude_names = [record.name for record in support_records]
        
        # This logic is specific to validation with a fixed support set, which is not the
        # primary training scenario. It's preserved for compatibility.
        if self.fixed_support and self.split.lower().startswith("val"):
            candidate_records = [rec for rec in self.class_to_records.get(cls_id, []) if rec.name not in exclude_names]
            if not candidate_records:
                raise RuntimeError(f"No query candidates left for class {cls_id} after excluding fixed support.")
            query_records = [candidate_records[index % len(candidate_records)]]
        else:
            query_records = self._sample_records_with_check(cls_id, self.n_queries, self.min_query_pixels, exclude=exclude_names)

        support_images, support_masks = self._encode_support(support_records, cls_id)
        query_images, query_labels, query_names = self._encode_queries(query_records, cls_id)

        return {
            "class_ids": [[cls_id for _ in support_records]],
            "support_images": [support_images],
            "support_mask": [support_masks],
            "support_names": [[record.name for record in support_records]],
            "query_images": query_images,
            "query_labels": query_labels,
            "query_names": query_names,
        }
