# Repository Guidelines

## Project Structure & Module Organization
`main.sh` orchestrates Sacred runs for `training.py` and `validation.py`, with hyperparameters defined in `config_ssl_upload.py`. All remote-sensing data access lives in `dataloaders/exp_disaster_dataset.py`, which samples GeoTIFF episodes. Model logic sits under `models/`, with `models/grid_proto_fewshot.py` hosting the ALPNet + DINOv2 implementation. Keep the dataset mounted at `../_datasets/Exp_Disaster_Few-Shot` so the relative paths in the config remain valid.

## Build, Test, and Development Commands
- `./main.sh training` meta-trains on the land-cover split using the default 5-shot configuration.
- `./main.sh validation` evaluates 5-shot transfer on the disaster split and saves predictions under the Sacred run folder (`disaster_preds/`).
- `python3 training.py with print_config=True` prints the resolved Sacred configuration; append overrides such as `task.n_shots=3` or `which_aug=disaster_aug`.

## Disaster Dataset Adaptation
`config_ssl_upload.py` exposes Sacred knobs for proto grid size, LoRA rank, augmentation policy, and support/query counts. The loader `dataloaders/exp_disaster_dataset.py` reads 512×512 RGB GeoTIFFs with `rasterio`, remaps foreground IDs to `{0,1}`, and yields PANet-style support/query tensors. Meta-training pulls from land-cover IDs `{1…8}`, while validation fixes the disaster label at `20`. Set `support_txt_file=<manifest>` to use reproducible support tiles; otherwise the loader selects the first viable five.

## Coding Style & Naming Conventions
Stick to Python 3.10+, 4-space indent, and PEP 8 spacing. Use `snake_case` for variables/functions and `UPPER_SNAKE_CASE` for constants. Centralize new hyperparameters in `config_ssl_upload.py` rather than sprinkling literals across scripts. Add comments sparingly—focus on clarifying geospatial quirks or non-obvious tensor reshaping.

## Testing Guidelines
We rely on episodic evaluations instead of unit tests. When introducing changes, run `./main.sh validation` and report Dice/IoU along with the location of saved masks (`disaster_preds/`). For data-loader or augmentation tweaks, perform a short meta-train smoke test (`n_steps=50`) to ensure loss curves remain stable.

## Commit & Pull Request Guidelines
Use imperative, present-tense commit subjects (e.g., `Refine disaster episodes`). Summarize PRs with: (1) Sacred overrides used, (2) any new config keys/defaults, and (3) validation metrics or qualitative overlays. Flag data-index or manifest changes so reviewers can reproduce the 5-shot setup quickly.
