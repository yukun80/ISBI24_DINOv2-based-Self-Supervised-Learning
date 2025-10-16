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

## Validation & Visualization Roadmap
- Extend `validation.py` to load the trained snapshot via Sacred config (`reload_model_path`/`validation.val_snapshot_path`), honor support manifests, and export IoU/Precision/Recall/F1/OA together with Dice. Store per-episode predictions as PNG/NPY under `disaster_preds/`.
- Enhance `util/metric.py` with true-negative tracking and shared helpers (`get_prf1`, `get_overall_accuracy`) so both validation and inference scripts can report consistent binary metrics while guarding against zero-division.
- Add `predict.py` to run offline inference on `valset` using a specified weight file, dumping raw masks (`*_pred.npy`), color masks (bg=black, slide=red), and optional overlays, plus a JSON manifest summarizing sample metadata.
- Centralize new knobs in `config_ssl_upload.py["validation"]` (snapshot path, output root, overlay toggle, worker count) to avoid scattering literals across scripts; expose them via Sacred overrides.
- Document the workflow: `python3 validation.py with validation.val_snapshot_path=<pth>` for metrics and `python3 predict.py --weights <pth>` for visual review; collect resulting stats from `runs/.../disaster_preds/metrics_report.json`.

## Commit & Pull Request Guidelines
Use imperative, present-tense commit subjects (e.g., `Refine disaster episodes`). Summarize PRs with: (1) Sacred overrides used, (2) any new config keys/defaults, and (3) validation metrics or qualitative overlays. Flag data-index or manifest changes so reviewers can reproduce the 5-shot setup quickly.

## AM²P Notes
- The legacy ALP module (`models/alpmodule.py`) has been removed; `models/am2p.py::AM2P` is the sole prototype builder.
- Connected components, anchor sampling, multi-scale stats, and cosine fusion now live inside `AM2P`; keep related overrides in `config_ssl_upload.py["model"]["am2p"]`.
- Utilities such as `tools/build_episode_manifest.py` and downstream trainers assume the AM²P interface that emits two-channel logits `(bg, fg)` with cached prototype metadata.
- When extending prototype logic, update `models/am2p.py` and run `python -m py_compile models/am2p.py models/grid_proto_fewshot.py` before committing.
