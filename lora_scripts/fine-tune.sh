#!/bin/bash

set -e

#torch 2.0.1 needed to be installed to remove an error, then deepspeed and apex needed to be updated as they had references to torch._six
#git clone https://github.com/NVIDIA/apex
#cd apex
#specific commit used at the time of the writing for that code.
#git checkout 7b2e71b0d4013f8e2f9f1c8dd21980ff1d76f1b6
#pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
#cd ..

pip install -r requirements.txt

mkdir -p /tmp/huggingface-cache/
export HF_DATASETS_CACHE="/tmp/huggingface-cache"
export TRANSFORMERS_OFFLINE=1

declare -a OPTS=(
    --model_id falcon-7b
    --model_path /opt/ml/input/data/pre-trained/
    --dataset_path /opt/ml/input/data/training
    #for test, increase to 10 later
    --epochs 1
    --per_device_train_batch_size 4
    --lr 2e-4
)

# Force using single GPU
#echo python run_clm.py "${OPTS[@]}" "$@"
#python run_clm.py "${OPTS[@]}" "$@"

#Multi gpu
echo torchrun --nnodes 1 --nproc_per_node "$SM_NUM_GPUS" run_clm.py "${OPTS[@]}" "$@"
torchrun --nnodes 1 --nproc_per_node "$SM_NUM_GPUS" run_clm.py "${OPTS[@]}" "$@"