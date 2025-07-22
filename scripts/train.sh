#!/bin/bash
CheckpointPath=pretrain_model/checkpoint.tar
LogsDir=./log/0722   
MaxEpoch=100
TrainBatchSize=4 
ValBatchSize=20
NumWorkers=10

export DEBUG=1
export CUDA_VISIBLE_DEVICES=3

python train_SemanticKITTI.py \
    --checkpoint_path $CheckpointPath \
    --log_dir $LogsDir \
    --max_epoch $MaxEpoch \
    --batch_size $TrainBatchSize \
    --val_batch_size $ValBatchSize \
    --num_workers $NumWorkers