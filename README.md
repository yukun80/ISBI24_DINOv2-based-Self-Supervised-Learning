# DINOv2 Few-Shot Disaster Segmentation

This repository now focuses on adapting the ALPNet + DINOv2 few-shot pipeline to the **Exp_Disaster_Few-Shot** remote-sensing benchmark. The goal is to meta-train on diverse land-cover categories and evaluate 5-shot transfer on disaster-response segmentation tasks.

## Dataset Layout

Place the dataset next to the repo so it can be reached via the relative path `../_datasets/Exp_Disaster_Few-Shot/`:

```
../_datasets/Exp_Disaster_Few-Shot/
├── trainset/
│   ├── images/*.tif
│   └── labels/*.tif   (IDs {0..8}, 0 = background)
└── valset/
    ├── images/*.tif
    ├── labels/*.tif   (0 = background, 20 = disaster foreground)
    └── optional *_png previews
```

The code reads GeoTIFF tiles with `rasterio`, normalises RGB channels to `[0,1]`, and remaps the active foreground to `{0,1}` inside each episode.

## Quick Start

```
# 1. Train meta-learner on the land-cover split
./main.sh training

# 2. Evaluate 5-shot transfer on the disaster validation split
./main.sh validation
```

Both commands invoke Sacred runs via `training.py` / `validation.py`. Outputs, logs, and predictions are stored under `./runs/mySSL_*` (see `config_ssl_upload.py` for the observer layout). Validation additionally saves NumPy masks to `<run>/disaster_preds/`.

Key runtime options are exposed through Sacred:

- `task.n_shots` / `task.n_queries`: support and query counts per episode (defaults: 5-shot, single query).
- `which_aug`: augmentation recipe (`disaster_aug` blends flips, rotations, and gamma jitter).
- `support_txt_file`: optional manifest (text or JSON) listing deterministic support tiles for validation.
- `episode_manifest`: optional JSON produced by `tools/build_episode_manifest.py` that constrains training episodes to high-quality supports/queries.

### Episode manifest workflow

When thin or noisy masks cause `Failed to find prototypes` warnings during training, pre-compute curated support/query pools:

```
python3 tools/build_episode_manifest.py \
    --dataset-root ../_datasets/Exp_Disaster_Few-Shot \
    --split trainset \
    --output data/train_high_quality.json
```

Pass the resulting file via Sacred (e.g. `./main.sh training episode_manifest=data/train_high_quality.json`) to force the loader to sample from tiles whose pooled foreground coverage exceeds the configured thresholds.

Inspect the resolved configuration with

```
python3 training.py with print_config=True
```

## Citation

If this adaptation helps your research, please cite the original DINOv2 few-shot paper:

```bibtex
@misc{ayzenberg2024dinov2,
      title={DINOv2 based Self Supervised Learning For Few Shot Medical Image Segmentation},
      author={Lev Ayzenberg and Raja Giryes and Hayit Greenspan},
      year={2024},
      eprint={2403.03273},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
