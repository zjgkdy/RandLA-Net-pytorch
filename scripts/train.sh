#!/bin/bash
Task=movable
DatasetCfg=config/semantic-kitti-movable.yaml
Dataset=data/SemanticKITTI/sequences_0.06
LogsDir=./runs   
Name=test
BatchSize=6
NumWorkers=8

export DEBUG=1
export CUDA_VISIBLE_DEVICES=2

python train.py \
    --task $Task \
    --dataset_cfg $DatasetCfg \
    --dataset $Dataset \
    --logs_dir $LogsDir \
    --name $Name \
    --batch_size $BatchSize \
    --num_workers $NumWorkers