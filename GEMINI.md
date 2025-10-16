# GEMINI.md

## Project Overview

This project implements a few-shot **remote-sensing disaster segmentation** model based on DINOv2 and the **AM²P** prototype pooling module. The pipeline meta-trains on land-cover categories from the Exp_Disaster_Few-Shot dataset and evaluates 5-shot transfer on disaster targets. DINOv2 serves as the sole backbone, and a prototype-based head performs segmentation. The codebase is written in Python, uses PyTorch for deep learning, and Sacred for experiment management.

## Building and Running

### 1. Dataset Layout

-   Place the dataset at `../_datasets/Exp_Disaster_Few-Shot/` relative to the repo.
-   `trainset/` contains GeoTIFF images/labels with IDs {0..8} (0 = background).
-   `valset/` contains GeoTIFF images/labels with IDs {0,20} (20 = disaster foreground).
-   Raster files are read via `rasterio`; no further preprocessing is required.

### 2. Training and Validation

The project is run directly using the `training.py` and `validation.py` scripts with Sacred.

**Training Example:**
```bash
python3 training.py with task.n_shots=5 optim.lr=1e-4
```

**Validation Example:**
```bash
python3 validation.py with validation.val_snapshot_path='path/to/snapshot.pth'
```

Detailed configuration lives in `config_ssl_upload.py` and can be overridden through Sacred command-line arguments.

## Development Conventions

### Code Style

The code follows the PEP 8 style guide.

### Testing

The project does not have a dedicated testing framework. The validation script (`validation.py`) is used to evaluate the model's performance.

### Contribution Guidelines

There are no explicit contribution guidelines.

## Key Files

-   `README.md`: Provides a high-level overview of the project.
-   `training.py`: The main script for training the model.
-   `validation.py`: The main script for validating the model.
-   `config_ssl_upload.py`: Contains the configuration for the project.
-   `models/grid_proto_fewshot.py`: Defines the main model architecture (`FewShotSeg`).
-   `models/am2p.py`: Implements the AM²P prototype module used by the few-shot segmentation model.
-   `dataloaders/`: Contains the GeoTIFF episodic loader (`exp_disaster_dataset.py`) and augmentation helpers.
-   `util/`: Contains utility functions and metrics.
