#!/bin/bash
set -e
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

MODE=$1
if [ "$MODE" != "training" ] && [ "$MODE" != "validation" ]; then
    echo "mode must be either training or validation"
    exit 1
fi

DATASET='EXP_DISASTER_FEWSHOT'
INPUT_SIZE=512
PROTO_GRID=16
N_SHOTS=5
N_QUERIES=1
WHICH_AUG='disaster_aug'
NWORKER=6
MODEL_NAME='dinov2_l14'
LORA=4
TTT="True"
NSTEP=100
MAX_ITER=1000
SNAPSHOT_INTERVAL=25000
SEED='1234'
DECAY=0.95

RUN_TAG="${MODE}_${MODEL_NAME}_disaster_grid_${PROTO_GRID}_res_${INPUT_SIZE}"

echo "Launching ${RUN_TAG} on GPU ${GPUID1}"

python3 "$MODE.py" with \
    "modelname=$MODEL_NAME" \
    'optim_type=sgd' \
    dataset=$DATASET \
    num_workers=$NWORKER \
    'use_wce=True' \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    lora=$LORA \
    ttt=$TTT \
    "input_size=($INPUT_SIZE, $INPUT_SIZE)" \
    which_aug=$WHICH_AUG \
    task.n_shots=$N_SHOTS \
    task.n_queries=$N_QUERIES \
    seed=$SEED
