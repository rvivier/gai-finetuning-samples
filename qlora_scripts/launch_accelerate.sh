#!/bin/bash

set -e

pip install -r requirements.txt

mkdir -p /tmp/huggingface-cache/
export HUGGINGFACE_HUB_CACHE="/tmp/.cache"

declare -a OPTS=(

    --model_id tiiuae/falcon-7b
    --model_path /opt/ml/input/data/pre-trained/
    --dataset_path /opt/ml/input/data/training
    #for test, increase to 10 later
    --epochs 1
    --per_device_train_batch_size 4
    --lr 2e-4
)

accelerate env

echo accelerate launch --multi_gpu --mixed_precision=fp16 --num_machines=1 --num_processes="$SM_NUM_GPUS" run_clm_multi_gpu.py "${OPTS[@]}" "$@"
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes="$SM_NUM_GPUS" run_clm_multi_gpu.py "${OPTS[@]}" "$@"