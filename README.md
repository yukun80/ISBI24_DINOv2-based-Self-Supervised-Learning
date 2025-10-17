# DINOv2 Few-Shot Disaster Segmentation

This repository adapts a DINOv2-based few-shot learning pipeline for a remote-sensing disaster segmentation task. The model meta-trains on diverse land-cover categories from the **Exp_Disaster_Few-Shot** dataset and evaluates 5-shot transfer performance on disaster scenarios.

The codebase uses PyTorch for deep learning and Sacred for experiment management.

## Dataset Layout

Place the dataset next to the repository so it can be reached via the relative path `../_datasets/Exp_Disaster_Few-Shot/`:

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

The dataloader reads GeoTIFF tiles using `rasterio`, normalizes RGB channels to `[0,1]`, and remaps the active foreground to a binary `{0,1}` mask within each episode.

## Workflow

The project workflow is split into three main stages: training, evaluation manifest generation, and validation/prediction.

**1. Train the Model**

The training script uses a robust **dynamic episode sampling** strategy. At each step, it creates a new few-shot task by randomly sampling images from the entire training set, ensuring the model learns a generalizable segmentation capability.

```bash
# Start training with default parameters
python3 training.py
```

You can override configuration parameters from the command line. For example:
```bash
python3 training.py with task.n_shots=3 optim.lr=1e-4
```

**2. Create Evaluation Manifest**

For **reproducible and standardized evaluation**, you must first generate a manifest file. This script groups the `valset` by geographical region and creates deterministic support/query splits for each region.

```bash
python3 -m tools.create_evaluation_manifest \
    --dataset-root ../_datasets/Exp_Disaster_Few-Shot \
    --split valset \
    --output data/valset/manifest.json
```

**3. Evaluate the Model**

Run end-to-end validation using the manifest created in the previous step. The script calculates key metrics (Dice/IoU/etc.), saves prediction masks, and generates a final `metrics_report.json`.

```bash
python3 validation.py with \
    validation.val_snapshot_path=path/to/your/snapshot.pth \
    validation.evaluation_manifest=data/valset/manifest.json
```

The script automatically loads the corresponding `config.json` from the snapshot's run folder to ensure model settings match. Artifacts are saved to the Sacred run directory under `disaster_preds/`.


## Offline Prediction

To produce visualizations from a checkpoint without spawning a new Sacred run, use the `predict.py` script. It also requires the evaluation manifest.

```bash
python3 predict.py \
    --weights path/to/your/snapshot.pth \
    --evaluation-manifest data/valset/manifest.json \
    --output-dir runs/predict/my_prediction
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
```