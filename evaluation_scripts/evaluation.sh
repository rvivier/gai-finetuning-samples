#!/bin/bash

pip install -r requirements.txt

#accelerate launch $2/main.py \
python $2/main.py \
    --model $1 \
    --model_args pretrained=$3,parallelize=True,trust_remote_code=True \
    --tasks $4 \
    --batch_size 8