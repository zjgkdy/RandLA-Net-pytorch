#!/bin/bash
CheckpointPath=pretrain_model/checkpoint.tar
LogsDir=./log/0722   
MaxEpoch=100
TrainBatchSize=4 
ValBatchSize=20
NumWorkers=10

export DEBUG=0
export CUDA_VISIBLE_DEVICES=2,3

python3 -m torch.distributed.launch --nproc_per_node=2 \
    ./train_SemanticKITTI.py \
    # --checkpoint_path $CheckpointPath \
    --log_dir $LogsDir \
    --max_epoch $MaxEpoch \
    --batch_size $TrainBatchSize \
    --val_batch_size $ValBatchSize \
    --num_workers $NumWorkers