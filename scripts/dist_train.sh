#!/bin/bash
Task=semantic
DatasetCfg=config/semantic-kitti.yaml
Dataset=data/SemanticKITTI/sequences
LogsDir=./runs   
Name=test
BatchSize=4
NumWorkers=4

export DEBUG=0
export CUDA_VISIBLE_DEVICES=1,2,3

python3 -m torch.distributed.launch --nproc_per_node=3 \
    ./train.py \
    --task $Task \
    --dataset_cfg $DatasetCfg \
    --dataset $Dataset \
    --logs_dir $LogsDir \
    --name $Name \
    --batch_size $BatchSize \
    --num_workers $NumWorkers
