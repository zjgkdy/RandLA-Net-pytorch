#!/bin/bash

python train.py \
    --task movable \
    --dataset_cfg config/semantic-kitti.yaml \
    --dataset data/semantic_kitti \
    --logs_dir ./runs \
    --name test \
    --batch_size 6 \
    --num_workers 8