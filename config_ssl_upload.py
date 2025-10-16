"""
Experiment configuration tailored for the Exp_Disaster_Few-Shot pipeline.
"""
import glob
import itertools
import os

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('disaster_fewshot')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    """Default configuration for training/validation on Exp_Disaster_Few-Shot."""
    seed = 1234
    gpu_id = 0
    num_workers = 6

    dataset = 'EXP_DISASTER_FEWSHOT'
    use_coco_init = False

    # Training schedule
    n_steps = 100
    batch_size = 1
    lr_milestones = [50, 80]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 20
    save_snapshot_every = 25000
    max_iters_per_load = 1000
    epochs = 1
    which_aug = 'disaster_aug'
    input_size = (512, 512)
    grad_accumulation_steps = 1
    precision = {
        'use_amp': False,
        'dtype': 'fp16',
        'grad_scaler': True,
    }

    # Few-shot episode structure
    n_shots = 5
    n_queries = 1
    task = {
        'n_ways': 1,
        'n_shots': n_shots,
        'n_queries': n_queries,
        'npart': 1,
    }

    # Optimisation
    optim_type = 'sgd'
    lr = 1e-3
    momentum = 0.9
    weight_decay = 0.0005
    optim = {
        'lr': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
    }

    # Model / backbone
    modelname = 'dinov2_l14'
    clsname = 'grid_proto'
    proto_grid_size = 16
    lora = 4
    reload_model_path = ''
    use_wce = True
    ttt = True

    am2p_defaults = {
        'radii': [4, 8, 16],
        'alpha': 1.2,
        'nmax_comp': 8,
        'mmax_total': 64,
        'theta_min': 8,
        'tau_area': 9,
        'beta': 0.3,
        'temp': 0.07,
        'epsilon': 1e-6,
    }

    model = {
        'align': True,
        'dinov2_loss': False,
        'use_coco_init': use_coco_init,
        'which_model': modelname,
        'cls_name': clsname,
        'proto_grid_size': proto_grid_size,
        'feature_hw': [input_size[0] // proto_grid_size, input_size[1] // proto_grid_size],
        'reload_model_path': reload_model_path,
        'lora': lora,
        'debug': False,
        'use_pos_enc': False,
        'am2p': am2p_defaults,
    }

    support_txt_file = None
    episode_manifest = None

    exp_prefix = ''
    exp_str = '_'.join(
        [exp_prefix or 'run',
         dataset,
         f'{task["n_shots"]}shot'])

    path = {
        'log_dir': './runs',
        'EXP_DISASTER_FEWSHOT': {'data_dir': "../_datasets/Exp_Disaster_Few-Shot"},
    }

    validation = {
        'val_snapshot_path': reload_model_path,
        'support_manifest': support_txt_file,
        'episode_manifest': episode_manifest,
        'config_json': '',
        'num_workers': 0,
        'max_iters_per_epoch': 1,
        'pin_memory': True,
        'output_dir_name': 'disaster_preds',
        'output_root': '',
        'save_numpy_preds': True,
        'save_color_mask': True,
        'save_overlay': True,
        'overlay_alpha': 0.5,
        'metrics_filename': 'metrics_report.json',
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    """Attach a FileStorageObserver for each run."""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
