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

The project is run via `training.py` and `validation.py` scripts using [Sacred](https://sacred.readthedocs.io/).

**1. Train the model:**
```bash
python3 training.py
```
You can override configuration parameters from the command line. For example:
```bash
python3 training.py with task.n_shots=3 optim.lr=1e-4
```

**2. Evaluate the model:**
```bash
python3 validation.py with validation.val_snapshot_path=<path/to/your/snapshot.pth>
```

Key runtime options are exposed through Sacred:

- `task.n_shots` / `task.n_queries`: support and query counts per episode (defaults: 5-shot, single query).
- `which_aug`: augmentation recipe (`disaster_aug` blends flips, rotations, and gamma jitter).
- `support_txt_file`: optional manifest (text or JSON) listing deterministic support tiles for validation.

For deterministic validation episodes, you can generate a manifest for the `valset`:

```
python3 tools/build_episode_manifest.py \
    --dataset-root ../_datasets/Exp_Disaster_Few-Shot \
    --split valset \
    --output data/valset_manifest.json
```
Pass the resulting file through `validation.episode_manifest=<path>`.

## Validation & Visualization

- Run end-to-end validation (Dice/IoU/Precision/Recall/F1/OA) and export masks/overlays:
  ```
  python3 validation.py with validation.val_snapshot_path=<path/to/snapshot.pth>
  ```
  The script auto-loads `config.json` from the snapshot run folder (override via `validation.config_json=<path/to/config.json>`) so backbone/LoRA settings match training. Artifacts land in the Sacred run directory under `disaster_preds/`, along with a `metrics_report.json` summary.

- Produce offline visualisations from a checkpoint without spawning a Sacred run:
  ```
  python3 predict.py \
      --weights runs/disaster_fewshot_run_EXP_DISASTER_FEWSHOT_5shot/3/snapshots/25000.pth \
      --output-dir ./runs/predict/step_25000
  ```
  Additional knobs include `--support-manifest`, `--no-save-overlay`, `--overlay-alpha`, and `--dataset-root`.

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
