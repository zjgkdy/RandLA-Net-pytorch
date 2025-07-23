#!/bin/bash
export DEBUG=0
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=2,3

CheckpointPath=pretrain_model/checkpoint.tar
LogsDir=./log/0722   
MaxEpoch=100
TrainBatchSize=4 
ValBatchSize=20
NumWorkers=10

python3 -m torch.distributed.launch --nproc_per_node=2 \
    ./tools/train_SemanticKITTI.py \
    --log_dir $LogsDir \
    --max_epoch $MaxEpoch \
    --batch_size $TrainBatchSize \
    --val_batch_size $ValBatchSize \
    --num_workers $NumWorkers
   # --checkpoint_path $CheckpointPath \
