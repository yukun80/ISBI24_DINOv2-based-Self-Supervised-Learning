# GEMINI.md

## Project Overview

This project implements a few-shot **remote-sensing disaster segmentation** model based on DINOv2 and the **AM²P** prototype pooling module. The pipeline meta-trains on land-cover categories from the Exp_Disaster_Few-Shot dataset and evaluates 5-shot transfer on disaster targets using a **deterministic, region-based evaluation protocol**.

The codebase is written in Python, uses PyTorch for deep learning, and Sacred for experiment management.

## Building and Running

### 1. Dataset Layout

-   Place the dataset at `../_datasets/Exp_Disaster_Few-Shot/` relative to the repo.
-   `trainset/` contains GeoTIFF images/labels with IDs {0..8} (0 = background).
-   `valset/` contains GeoTIFF images/labels with IDs {0,20} (20 = disaster foreground).
-   Raster files are read via `rasterio`; no further preprocessing is required.

### 2. Workflow

The recommended workflow involves three stages:

**1. Create Evaluation Manifest (for validation/prediction)**

For reproducible evaluation, first generate a manifest file from the `valset`. This creates deterministic, region-specific support/query episodes.

```bash
python3 -m tools.create_evaluation_manifest --dataset-root ../_datasets/Exp_Disaster_Few-Shot --split valset --output data/valset/manifest.json
```

**2. Training**

The training script uses robust dynamic episode sampling. It does not require a manifest.

```bash
python3 training.py with task.n_shots=5 optim.lr=1e-4
```

**3. Validation**

Validation is driven by the manifest created in step 1.

```bash
python3 validation.py with \
    validation.val_snapshot_path=\'path/to/snapshot.pth\' \
    validation.evaluation_manifest=\'data/valset/manifest.json\'
```

Detailed configuration lives in `config_ssl_upload.py` and can be overridden through Sacred command-line arguments.

## Development Conventions

### Code Style

The code follows the PEP 8 style guide.

### Testing

The project does not have a dedicated automated testing framework. The `validation.py` script, driven by a deterministic manifest, serves as the primary tool for evaluating model performance and correctness.

### Contribution Guidelines

There are no explicit contribution guidelines.

## Key Files

-   `README.md`: High-level overview for end-users.
-   `training.py`: Main script for training the model using **dynamic episode sampling**.
-   `validation.py`: Main script for validating the model using a **deterministic evaluation manifest**.
-   `predict.py`: Standalone script for inference and visualization, also driven by an **evaluation manifest**.
-   `config_ssl_upload.py`: Central configuration file for the Sacred experiment.
-   `tools/create_evaluation_manifest.py`: Script to generate a deterministic, region-based evaluation manifest for `valset`.
-   `models/grid_proto_fewshot.py`: Defines the main model architecture (`FewShotSeg`).
-   `models/am2p.py`: Implements the AM²P prototype module.
-   `dataloaders/exp_disaster_dataset.py`: Implements a flexible episodic loader with two modes: **dynamic sampling for training** and **manifest-based loading for evaluation**.
-   `util/`: Contains utility functions and metrics.